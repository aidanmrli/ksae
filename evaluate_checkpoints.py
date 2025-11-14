"""
Standalone script to evaluate trained checkpoints.

This script loads checkpoint.pt and last.pt from a training run directory
and runs the standardized evaluation suite on them.

Usage:
    python evaluate_checkpoints.py --run_dir runs/kae/20251114-111432 --system duffing
    python evaluate_checkpoints.py --run_dir runs/kae/20251114-111432 --system pendulum --device cpu
    python evaluate_checkpoints.py --run_dir runs/kae/20251114-111432 --system lorenz63 --checkpoints checkpoint.pt
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from config import Config
from data import make_env
from model import make_model
from evaluation import EvaluationSettings, evaluate_model


def get_dt_from_config(cfg: Config) -> float:
    """Extract dt from environment config based on ENV_NAME."""
    env_name = cfg.ENV.ENV_NAME.lower()
    if env_name == 'duffing':
        return cfg.ENV.DUFFING.DT
    elif env_name == 'pendulum':
        return cfg.ENV.PENDULUM.DT
    elif env_name == 'lotka_volterra':
        return cfg.ENV.LOTKA_VOLTERRA.DT
    elif env_name == 'lorenz63':
        return cfg.ENV.LORENZ63.DT
    elif env_name == 'parabolic':
        return cfg.ENV.PARABOLIC.DT
    elif env_name == 'lyapunov':
        return cfg.ENV.LYAPUNOV.DT
    else:
        return 0.01  # default fallback


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    cfg: Config,
    device: str,
    system: str,
    output_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint and evaluate it.
    
    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint_name: Name identifier for this checkpoint (e.g., "best", "last")
        cfg: Configuration object
        device: Device to run evaluation on
        system: System/environment name to evaluate on
        output_dir: Optional directory to save evaluation results
        
    Returns:
        Evaluation results dictionary or None if checkpoint not found
    """
    if not checkpoint_path.exists():
        print(f"  Skipping {checkpoint_name}: checkpoint not found at {checkpoint_path}")
        return None
    
    print(f"\nEvaluating {checkpoint_name} checkpoint...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_step = checkpoint.get('step', 'unknown')
    print(f"  Loaded checkpoint (step={ckpt_step}). Building eval env/model...", flush=True)
    
    # Load config from checkpoint if available, otherwise use provided cfg
    # Keep original config for model creation (model architecture depends on training observation size)
    if 'config' in checkpoint:
        model_cfg = Config.from_dict(checkpoint['config'])
    else:
        model_cfg = cfg
    
    # Create model using original config (from training)
    model_env = make_env(model_cfg)
    eval_model = make_model(model_cfg, model_env.observation_size)
    
    # Create evaluation config with specified system for environment creation
    eval_cfg = Config.from_dict(model_cfg.to_dict())
    eval_cfg.ENV.ENV_NAME = system
    
    # Get dt from evaluation config (uses the specified system)
    dt = get_dt_from_config(eval_cfg)
    
    # Verify observation size compatibility
    eval_env = make_env(eval_cfg)
    if eval_env.observation_size != eval_model.observation_size:
        print(
            f"  WARNING: System '{system}' has observation size {eval_env.observation_size} "
            f"but model expects {eval_model.observation_size}. "
            f"Evaluation may fail or be skipped."
        )
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    eval_model = eval_model.to(device)
    eval_model.eval()
    eval_model.dt = dt
    
    # Create evaluation settings
    eval_settings = EvaluationSettings()
    eval_settings.systems = [system]
    
    # Evaluate
    if output_dir is None:
        output_dir = checkpoint_path.parent / f"evaluation_{checkpoint_name}"
    else:
        output_dir = output_dir / f"evaluation_{checkpoint_name}"
    
    print(f"  Calling evaluate_model() for system={system} ...", flush=True)
    eval_results = evaluate_model(
        model=eval_model,
        cfg=eval_cfg,
        device=device,
        settings=eval_settings,
        output_dir=output_dir,
    )
    print(f"  evaluate_model() finished for {checkpoint_name}.", flush=True)
    
    # Save results
    results_file = checkpoint_path.parent / f"evaluation_results_{checkpoint_name}.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Print summary
    print(f"  {checkpoint_name.upper()} - Evaluation summary:")
    system_metrics = eval_results.get(system)
    if system_metrics is not None:
        print(f"    System: {system}")
        for horizon in eval_settings.horizons:
            if system == "parabolic" and horizon > 100:
                continue
            horizon_key = str(horizon)
            modes = system_metrics.get("modes", {})
            no_re = modes.get("no_reencode", {}).get("horizons", {}).get(horizon_key)
            every = modes.get("every_step", {}).get("horizons", {}).get(horizon_key)
            best = system_metrics.get("best_periodic", {}).get(horizon_key)
            if no_re is None or every is None:
                continue
            best_str = "best-PR=N/A" if best is None else f"best-PR={best['mean']:.4e} ({best['mode']})"
            print(
                f"      Horizon {horizon}: "
                f"no-reencode={no_re['mean']:.4e}, "
                f"every-step={every['mean']:.4e}, "
                f"{best_str}"
            )
    
    print(f"  Evaluation artifacts saved to {output_dir}")
    return eval_results


def main():
    """Command-line interface for checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Koopman Autoencoder checkpoints',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to training run directory containing checkpoints'
    )
    
    # Required arguments
    parser.add_argument(
        '--system',
        type=str,
        required=True,
        choices=['duffing', 'pendulum', 'lotka_volterra', 'lorenz63', 'parabolic', 'lyapunov'],
        help='System/environment to evaluate on'
    )
    
    # Optional arguments
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        default=['checkpoint.pt', 'last.pt'],
        help='Checkpoint files to evaluate (relative to run_dir)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run evaluation on'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: run_dir)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    
    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load config from checkpoint or config.json
    cfg = None
    config_json_path = run_dir / 'config.json'
    if config_json_path.exists():
        print(f"Loading config from {config_json_path}")
        cfg = Config.from_json(str(config_json_path))
    else:
        # Try to load from first checkpoint
        first_checkpoint = run_dir / args.checkpoints[0]
        if first_checkpoint.exists():
            print(f"Loading config from checkpoint {first_checkpoint}")
            checkpoint = torch.load(first_checkpoint, map_location=args.device)
            if 'config' in checkpoint:
                cfg = Config.from_dict(checkpoint['config'])
            else:
                raise ValueError(
                    f"No config found in checkpoint and no config.json in {run_dir}. "
                    "Cannot determine model configuration."
                )
        else:
            raise ValueError(
                f"No config.json found and no checkpoint available. "
                f"Cannot determine model configuration."
            )
    
    print(f"Configuration loaded: {cfg.ENV.ENV_NAME} system, {cfg.MODEL.MODEL_NAME} model")
    print("-" * 80)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    
    # Evaluate each checkpoint
    all_results = {}
    for checkpoint_name in args.checkpoints:
        checkpoint_path = run_dir / checkpoint_name
        # Extract name without extension for cleaner output
        name_key = checkpoint_name.replace('.pt', '')
        
        results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_name=name_key,
            cfg=cfg,
            device=args.device,
            system=args.system,
            output_dir=output_dir,
        )
        
        if results is not None:
            all_results[name_key] = results
    
    # Save combined summary
    if all_results:
        summary = {
            "run_dir": str(run_dir),
            "evaluated_checkpoints": list(all_results.keys()),
            "system": args.system,
        }
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCombined summary saved to {summary_file}")
    
    print("-" * 80)
    print(f"Evaluation complete! Results saved to {output_dir}")
    
    return all_results


if __name__ == '__main__':
    main()

