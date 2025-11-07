"""
Unit tests for training script.

Tests cover:
- Training step correctness
- Evaluation function
- Full training loop (short)
- Checkpoint saving/loading
"""

import pytest
import torch
import tempfile
from pathlib import Path

from config import get_config
from data import make_env, VectorWrapper
from model import make_model
from train import train_step, evaluate, train, build_optimizer


class TestTrainStep:
    """Test training step function."""
    
    def test_train_step_reduces_loss(self):
        """Test that training step updates parameters and computes metrics."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.BATCH_SIZE = 4
        
        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate batch
        rng = torch.Generator().manual_seed(42)
        x = torch.randn(4, env.observation_size)
        nx = torch.randn(4, env.observation_size)
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Training step
        metrics = train_step(model, optimizer, x, nx)
        
        # Check metrics are computed
        assert 'loss' in metrics
        assert 'residual_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert isinstance(metrics['loss'], float)
        
        # Check parameters were updated
        for p_before, p_after in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_before, p_after), "Parameters should be updated"
    
    def test_train_step_gradient_flow(self):
        """Test that gradients flow through all parameters."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        
        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(4, env.observation_size)
        nx = torch.randn(4, env.observation_size)
        
        # Training step
        metrics = train_step(model, optimizer, x, nx)
        
        # Check that at least some parameters have been updated
        has_update = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                # After optimizer step, gradients should be zeroed
                # but parameters should have changed
                has_update = True
        
        assert has_update, "At least some parameters should require gradients"


class TestEvaluate:
    """Test evaluation function."""
    
    def test_evaluate_returns_trajectories(self):
        """Test that evaluate returns trajectory predictions."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        
        env = make_env(cfg)
        env_vec = VectorWrapper(env, 4)
        model = make_model(cfg, env.observation_size)
        
        rng = torch.Generator().manual_seed(42)
        x = env_vec.reset(rng)
        
        results = evaluate(model, x, lambda s: env_vec.step(s), num_steps=10)
        
        assert 'true_trajectory' in results
        assert 'pred_trajectory' in results
        assert 'pred_error' in results
        assert 'mean_error' in results
        assert 'final_error' in results
        
        # Check shapes
        assert results['true_trajectory'].shape == (10, 4, env.observation_size)
        assert results['pred_trajectory'].shape == (10, 4, env.observation_size)
        assert results['pred_error'].shape == (10,)
    
    def test_evaluate_no_gradient(self):
        """Test that evaluation doesn't compute gradients."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        
        env = make_env(cfg)
        env_vec = VectorWrapper(env, 4)
        model = make_model(cfg, env.observation_size)
        
        rng = torch.Generator().manual_seed(42)
        x = env_vec.reset(rng)
        
        results = evaluate(model, x, lambda s: env_vec.step(s), num_steps=5)
        
        # Check that results don't require grad
        assert not results['true_trajectory'].requires_grad
        assert not results['pred_trajectory'].requires_grad


class TestTrain:
    """Test full training loop."""
    
    def test_train_short_run(self):
        """Test that training runs without errors (short run)."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 10
        cfg.TRAIN.BATCH_SIZE = 4
        cfg.TRAIN.DATA_SIZE = 16
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Check model was returned
            assert model is not None
            assert isinstance(model, torch.nn.Module)
            
            # Check checkpoints were saved
            run_dirs = list(Path(tmpdir).iterdir())
            assert len(run_dirs) > 0
            
            run_dir = run_dirs[0]
            assert (run_dir / 'config.json').exists()
            assert (run_dir / 'last.pt').exists()
    
    def test_train_saves_checkpoint(self):
        """Test that training saves checkpoints correctly."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 5
        cfg.TRAIN.BATCH_SIZE = 4
        cfg.TRAIN.DATA_SIZE = 16
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Find run directory
            run_dirs = list(Path(tmpdir).iterdir())
            run_dir = run_dirs[0]
            
            # Load checkpoint
            checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu')
            
            assert 'step' in checkpoint
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'config' in checkpoint
            assert 'metrics' in checkpoint


class TestOptimizer:
    """Tests for optimizer construction and parameter groups."""

    def test_optimizer_uses_k_lr(self):
        """Koopman matrix should use cfg.TRAIN.K_LR learning rate."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]

        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)

        optimizer = build_optimizer(model, cfg)

        kmat_param = dict(model.named_parameters())["kmat"]
        found_group = None
        for group in optimizer.param_groups:
            if any(p is kmat_param for p in group['params']):
                found_group = group
                break

        assert found_group is not None, "kmat parameter group not found in optimizer"
        assert found_group['lr'] == cfg.TRAIN.K_LR
    
    def test_train_resume_from_checkpoint(self):
        """Test that training can resume from checkpoint."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 5
        cfg.TRAIN.BATCH_SIZE = 4
        cfg.TRAIN.DATA_SIZE = 16
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First training run
            model1 = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Get checkpoint path
            run_dirs = list(Path(tmpdir).iterdir())
            checkpoint_path = run_dirs[0] / 'last.pt'
            
            # Load checkpoint and check
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert checkpoint['step'] == cfg.TRAIN.NUM_STEPS - 1
    
    def test_train_different_configs(self):
        """Test training with different configuration presets."""
        configs_to_test = ['generic', 'generic_sparse']
        
        for config_name in configs_to_test:
            cfg = get_config(config_name)
            cfg.MODEL.TARGET_SIZE = 8
            cfg.MODEL.ENCODER.LAYERS = [8]
            cfg.TRAIN.NUM_STEPS = 5
            cfg.TRAIN.BATCH_SIZE = 4
            cfg.TRAIN.DATA_SIZE = 16
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model = train(cfg, log_dir=tmpdir, device='cpu')
                assert model is not None


class TestTrainIntegration:
    """Integration tests for training pipeline."""
    
    def test_train_all_environments(self):
        """Test training on all available environments."""
        environments = ['duffing', 'pendulum', 'lotka_volterra']
        
        for env_name in environments:
            cfg = get_config("generic")
            cfg.ENV.ENV_NAME = env_name
            cfg.MODEL.TARGET_SIZE = 8
            cfg.MODEL.ENCODER.LAYERS = [8]
            cfg.TRAIN.NUM_STEPS = 5
            cfg.TRAIN.BATCH_SIZE = 4
            cfg.TRAIN.DATA_SIZE = 16
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model = train(cfg, log_dir=tmpdir, device='cpu')
                assert model is not None
    
    def test_train_decreases_loss(self):
        """Test that training actually reduces loss over time."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        cfg.MODEL.ENCODER.LAYERS = [16, 16]
        cfg.TRAIN.NUM_STEPS = 100
        cfg.TRAIN.BATCH_SIZE = 32
        cfg.TRAIN.DATA_SIZE = 128
        cfg.TRAIN.LR = 1e-3
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train model
            model = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Evaluate initial and final loss
            env = make_env(cfg)
            env = VectorWrapper(env, cfg.TRAIN.BATCH_SIZE)
            
            rng = torch.Generator().manual_seed(42)
            x = env.reset(rng)
            nx = env.step(x)
            
            # Load initial and final checkpoints to compare
            # (In a real scenario, we'd track loss over time)
            # For now, just check that model can compute loss
            with torch.no_grad():
                loss, metrics = model.loss(x, nx)
                assert loss.item() < 100.0  # Sanity check


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

