"""
Training script for Koopman Autoencoder models.

This script provides a complete training pipeline for learning Koopman operator
representations of dynamical systems using PyTorch.

Usage:
    python train.py --config generic_sparse --env duffing --num_steps 20000

Or use it programmatically:
    from train import train
    cfg = get_config("generic_sparse")
    cfg.ENV.ENV_NAME = "duffing"
    train(cfg, log_dir="./runs/experiment_001")
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn

from config import Config, get_config
from data import make_env, VectorWrapper, generate_trajectory
from model import make_model
from evaluation import (
    EvaluationSettings,
    evaluate_model,
    rollout_every_step_reencode,
)


class MetricsLogger:
    """Simple file-based metrics logger.
    
    Logs metrics to JSON files for later analysis or plotting.
    Can easily be replaced with wandb later.
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics_file = log_dir / 'metrics_history.jsonl'
        self.metrics_history: List[Dict] = []
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar metric."""
        entry = {
            'step': step,
            'name': name,
            'value': value,
        }
        # Append to JSONL file (one JSON object per line)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        self.metrics_history.append(entry)
    
    def log_dict(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            name = f"{prefix}/{key}" if prefix else key
            self.log_scalar(name, value, step)
    
    def close(self):
        """Save final summary."""
        summary_file = self.log_dir / 'metrics_summary.json'
        
        # Compute summary statistics
        summary = {}
        metrics_by_name = {}
        for entry in self.metrics_history:
            name = entry['name']
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(entry['value'])
        
        for name, values in metrics_by_name.items():
            summary[name] = {
                'final': values[-1] if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
                'mean': sum(values) / len(values) if values else None,
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    nx: torch.Tensor,
) -> Dict[str, float]:
    """Perform one training step.
    
    Args:
        model: Koopman machine model
        optimizer: PyTorch optimizer
        x: Current states [batch_size, observation_size]
        nx: Next states [batch_size, observation_size]
        
    Returns:
        Dictionary of metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Compute loss
    loss, metrics = model.loss(x, nx)
    
    # Backward pass
    loss.backward()
    optimizer.step()

    return metrics


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Create optimizer with a specific learning rate for the Koopman matrix.
    
    This constructs parameter groups so that parameters named with 'kmat' use
    cfg.TRAIN.K_MATRIX_LR while all other parameters use cfg.TRAIN.LR.
    """
    kmat_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'kmat' in name:
            kmat_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': cfg.TRAIN.LR,
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        })
    if kmat_params:
        param_groups.append({
            'params': kmat_params,
            'lr': cfg.TRAIN.K_MATRIX_LR,
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        })

    return torch.optim.AdamW(param_groups)


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    env_step_fn,
    num_steps: int = 50,
) -> Dict[str, Any]:
    """Quick evaluation helper used during training and unit tests."""

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        true_traj = generate_trajectory(env_step_fn, x.cpu(), length=num_steps)
        pred_traj = rollout_every_step_reencode(model, x.to(device), num_steps)

        pred_traj_cpu = pred_traj.cpu()
        step_error = torch.norm(pred_traj_cpu - true_traj, dim=-1).mean(dim=1)

        return {
            "true_trajectory": true_traj,
            "pred_trajectory": pred_traj_cpu,
            "pred_error": step_error,
            "mean_error": step_error.mean().item(),
            "final_error": step_error[-1].item(),
        }



def train(
    cfg: Config,
    log_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
) -> nn.Module:
    """Main training function.
    
    Args:
        cfg: Configuration object
        log_dir: Directory for tensorboard logs and checkpoints
        checkpoint_path: Path to checkpoint to resume from
        device: Device to train on ('cpu', 'cuda', 'mps')
        
    Returns:
        Trained model
    """
    # Setup logging directory and save config
    if log_dir is None:
        log_dir = './runs/kae'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(str(run_dir / 'config.json'))
    
    logger = MetricsLogger(run_dir)
    
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
    
    env = make_env(cfg)
    env = VectorWrapper(env, cfg.TRAIN.BATCH_SIZE)
    
    model = make_model(cfg, env.observation_size)
    model = model.to(device)
    
    optimizer = build_optimizer(model, cfg)
    
    start_step = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from checkpoint at step {start_step}")
    
    # Pre-generate random number generators for data
    # Each batch gets a non-overlapping seed range to avoid collisions
    # Batch i uses seeds: cfg.SEED + i * BATCH_SIZE to cfg.SEED + (i+1) * BATCH_SIZE - 1
    num_batches = cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE
    rngs = [torch.Generator().manual_seed(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(num_batches)]
    
    print(f"Training {cfg.MODEL.MODEL_NAME} on {cfg.ENV.ENV_NAME}")
    print(f"Device: {device}")
    print(f"Observation size: {env.observation_size}")
    print(f"Target size: {cfg.MODEL.TARGET_SIZE}")
    print(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Total steps: {cfg.TRAIN.NUM_STEPS}")
    print(f"Log directory: {run_dir}")
    print("-" * 80)
    
    best_loss = float('inf')
    
    for step in range(start_step, cfg.TRAIN.NUM_STEPS):
        # Generate batch
        rng = rngs[step % num_batches]
        x = env.reset(rng)
        nx = env.step(x)
        
        x = x.to(device)
        nx = nx.to(device)
        
        metrics = train_step(model, optimizer, x, nx)
        
        logger.log_dict(metrics, step, prefix='train')
        
        if step % 100 == 0:
            print(f"Step {step}/{cfg.TRAIN.NUM_STEPS} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Res: {metrics['residual_loss']:.4f} | "
                  f"Recon: {metrics['reconst_loss']:.4f} | "
                  f"Sparsity: {metrics['sparsity_ratio']:.3f}")
        
        # Periodic evaluation
        if step % 500 == 0 or step == cfg.TRAIN.NUM_STEPS - 1:
            eval_results = evaluate(model, x[:4], lambda s: env.step(s), num_steps=200)
            logger.log_scalar('eval/mean_error', eval_results['mean_error'], step)
            logger.log_scalar('eval/final_error', eval_results['final_error'], step)
            
            print(f"  Eval | Mean error: {eval_results['mean_error']:.4f} | "
                  f"Final error: {eval_results['final_error']:.4f}")
        
        # Save checkpoint
        if step % 1000 == 0 or step == cfg.TRAIN.NUM_STEPS - 1:
            checkpoint_dict = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg.to_dict(),
                'metrics': metrics,
            }
            
            # Save latest checkpoint
            torch.save(checkpoint_dict, run_dir / 'last.pt')
            
            # Save best checkpoint
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save(checkpoint_dict, run_dir / 'checkpoint.pt')
                print(f"  Saved best checkpoint (loss: {best_loss:.4f})")
    
    # Save final metrics and close logger
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.close()

    print("-" * 80)
    print("Running standardized evaluation suite...")

    eval_dir = run_dir / "evaluation"
    eval_settings = EvaluationSettings()
    eval_results = evaluate_model(
        model=model,
        cfg=cfg,
        device=device,
        settings=eval_settings,
        output_dir=eval_dir,
    )

    with open(run_dir / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    primary_system = cfg.ENV.ENV_NAME
    primary_metrics = eval_results.get(primary_system)
    if primary_metrics is not None:
        print(f"Primary system ({primary_system}) MSE summary:")
        for horizon in eval_settings.horizons:
            if primary_system == "parabolic" and horizon > 100:
                continue
            horizon_key = str(horizon)
            no_re = primary_metrics["modes"]["no_reencode"]["horizons"].get(horizon_key)
            every = primary_metrics["modes"]["every_step"]["horizons"].get(horizon_key)
            best = primary_metrics["best_periodic"].get(horizon_key)
            if no_re is None or every is None:
                continue
            best_str = "best-PR=N/A" if best is None else f"best-PR={best['mean']:.4e} ({best['mode']})"
            print(
                f"  Horizon {horizon}: "
                f"no-reencode={no_re['mean']:.4e}, "
                f"every-step={every['mean']:.4e}, "
                f"{best_str}"
            )

    print(f"Evaluation artifacts saved to {eval_dir}")
    print("-" * 80)
    print(f"Training complete! Checkpoints saved to {run_dir}")
    
    return model


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train Koopman Autoencoder')
    
    # Configuration
    parser.add_argument('--config', type=str, default='generic',
                        choices=['default', 'generic', 'generic_sparse', 
                                'generic_prediction', 'lista', 'lista_nonlinear'],
                        help='Training configuration preset')
    parser.add_argument('--env', type=str, default='duffing',
                        choices=['duffing', 'pendulum', 'lotka_volterra', 
                                'lorenz63', 'parabolic'],
                        help='Dynamical system environment')
    
    # Training
    parser.add_argument('--num_steps', type=int, default=20000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Model
    parser.add_argument('--target_size', type=int, default=None,
                        help='Latent dimension (overrides config default)')
    parser.add_argument('--sparsity_coeff', type=float, default=None,
                        help='Sparsity loss weight (overrides config default)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs/kae',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Create config
    cfg = get_config(args.config)
    cfg.ENV.ENV_NAME = args.env
    cfg.TRAIN.NUM_STEPS = args.num_steps
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed
    
    # Override config with command-line args
    if args.lr is not None:
        cfg.TRAIN.LR = args.lr
    if args.target_size is not None:
        cfg.MODEL.TARGET_SIZE = args.target_size
    if args.sparsity_coeff is not None:
        cfg.MODEL.SPARSITY_COEFF = args.sparsity_coeff
    
    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'
    
    # Train
    train(cfg, log_dir=args.log_dir, checkpoint_path=args.checkpoint, device=args.device)


if __name__ == '__main__':
    main()

