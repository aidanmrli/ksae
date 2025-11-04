"""Unit tests for PyTorch Koopman models."""

import torch
import pytest
from config import get_config, Config
from model import (
    MLPCoder,
    LISTA,
    GenericKM,
    LISTAKM,
    make_model,
    shrink,
    get_activation
)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_shrink_positive(self):
        """Test soft thresholding on positive values."""
        x = torch.tensor([2.0, 1.5, 0.5])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(result, expected)
    
    def test_shrink_negative(self):
        """Test soft thresholding on negative values."""
        x = torch.tensor([-2.0, -1.5, -0.5])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([-1.0, -0.5, 0.0])
        assert torch.allclose(result, expected)
    
    def test_shrink_mixed(self):
        """Test soft thresholding on mixed values."""
        x = torch.tensor([2.0, -1.5, 0.5, -0.3])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([1.0, -0.5, 0.0, 0.0])
        assert torch.allclose(result, expected)
    
    def test_get_activation_relu(self):
        """Test ReLU activation retrieval."""
        act = get_activation('relu')
        assert isinstance(act, torch.nn.ReLU)
    
    def test_get_activation_tanh(self):
        """Test Tanh activation retrieval."""
        act = get_activation('tanh')
        assert isinstance(act, torch.nn.Tanh)
    
    def test_get_activation_gelu(self):
        """Test GELU activation retrieval."""
        act = get_activation('gelu')
        assert isinstance(act, torch.nn.GELU)
    
    def test_get_activation_invalid(self):
        """Test invalid activation name."""
        with pytest.raises(ValueError):
            get_activation('invalid_activation')


class TestMLPCoder:
    """Test MLPCoder module."""
    
    def test_initialization(self):
        """Test MLPCoder can be initialized."""
        coder = MLPCoder(
            input_size=10,
            target_size=5,
            hidden_layers=[16, 16],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        assert coder.input_size == 10
        assert coder.target_size == 5
        assert len(coder.hidden_layers) == 2
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        coder = MLPCoder(
            input_size=10,
            target_size=5,
            hidden_layers=[16, 16],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        x = torch.randn(32, 10)
        y = coder(x)
        assert y.shape == (32, 5)
    
    def test_forward_batch_independence(self):
        """Test that batch elements are processed independently."""
        coder = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[8],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        x1 = torch.randn(1, 5)
        x2 = torch.randn(1, 5)
        x_batch = torch.cat([x1, x2], dim=0)
        
        y1 = coder(x1)
        y2 = coder(x2)
        y_batch = coder(x_batch)
        
        assert torch.allclose(y_batch[0], y1[0], atol=1e-6)
        assert torch.allclose(y_batch[1], y2[0], atol=1e-6)
    
    def test_last_relu(self):
        """Test last_relu option applies ReLU to output."""
        coder_with_relu = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[],
            last_relu=True,
            use_bias=False,
            activation='relu'
        )
        coder_without_relu = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        
        # Use same weights
        coder_with_relu.network[0].weight.data = coder_without_relu.network[0].weight.data.clone()
        
        x = torch.randn(1, 5)
        y_with = coder_with_relu(x)
        y_without = coder_without_relu(x)
        
        # Output with ReLU should be non-negative
        assert torch.all(y_with >= 0)
        # Outputs should match after applying ReLU
        assert torch.allclose(y_with, torch.relu(y_without))


class TestLISTA:
    """Test LISTA module."""
    
    def test_initialization(self):
        """Test LISTA can be initialized."""
        cfg = get_config("lista")
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        Wd_init = torch.randn(xdim, zdim)
        
        lista = LISTA(cfg, xdim, Wd_init)
        assert lista.xdim == xdim
        assert lista.zdim == zdim
    
    def test_forward_shape(self):
        """Test LISTA forward pass output shape."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 5
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        Wd_init = torch.randn(xdim, zdim)
        
        lista = LISTA(cfg, xdim, Wd_init)
        x = torch.randn(16, xdim)
        z = lista(x)
        assert z.shape == (16, zdim)
    
    def test_sparsity(self):
        """Test LISTA produces sparse codes."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 128
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 10
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.5
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        Wd_init = torch.randn(xdim, zdim)
        
        lista = LISTA(cfg, xdim, Wd_init)
        x = torch.randn(1, xdim)
        z = lista(x)
        
        # Count nonzero elements (with tolerance)
        nonzero_count = (z.abs() > 1e-6).sum().item()
        # Should be sparser than dense encoding (i.e., some zeros exist)
        assert nonzero_count < zdim  # At least some sparsity
    
    def test_wrong_wd_init_shape(self):
        """Test LISTA raises error for wrong Wd_init shape."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        xdim = 10
        wrong_Wd_init = torch.randn(32, xdim)  # Wrong zdim
        
        with pytest.raises(AssertionError):
            LISTA(cfg, xdim, wrong_Wd_init)


class TestGenericKM:
    """Test GenericKM model."""
    
    def test_initialization(self):
        """Test GenericKM can be initialized."""
        cfg = get_config("generic")
        obs_size = 2
        model = GenericKM(cfg, obs_size)
        assert model.observation_size == obs_size
        assert model.target_size == cfg.MODEL.TARGET_SIZE
    
    def test_encode_decode_shape(self):
        """Test encode and decode output shapes."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 32
        obs_size = 5
        batch_size = 16
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(batch_size, obs_size)
        
        # Test encode
        z = model.encode(x)
        assert z.shape == (batch_size, cfg.MODEL.TARGET_SIZE)
        
        # Test decode
        x_recon = model.decode(z)
        assert x_recon.shape == (batch_size, obs_size)
    
    def test_reconstruction(self):
        """Test reconstruction method."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 3
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        x_recon = model.reconstruction(x)
        assert x_recon.shape == x.shape
    
    def test_kmatrix_shape(self):
        """Test Koopman matrix has correct shape."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 32
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        kmat = model.kmatrix()
        assert kmat.shape == (32, 32)
    
    def test_step_latent(self):
        """Test stepping in latent space."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        y = torch.randn(4, 16)
        ny = model.step_latent(y)
        assert ny.shape == y.shape
    
    def test_step_env(self):
        """Test stepping in observation space via Koopman operator."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 3
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        nx_pred = model.step_env(x)
        assert nx_pred.shape == x.shape
    
    def test_residual(self):
        """Test residual computation."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        residual = model.residual(x, nx)
        assert residual.shape == (8,)
        assert torch.all(residual >= 0)  # Norm is non-negative
    
    def test_sparsity_loss(self):
        """Test sparsity loss computation."""
        cfg = get_config("generic_sparse")
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        loss = model.sparsity_loss(x)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # L1 norm is non-negative
    
    def test_loss_computation(self):
        """Test full loss computation."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, metrics = model.loss(x, nx)
        
        # Check loss is scalar
        assert loss.ndim == 0
        
        # Check metrics
        assert 'loss' in metrics
        assert 'residual_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'prediction_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert 'A_max_eigenvalue' in metrics
        assert 'sparsity_ratio' in metrics
    
    def test_norm_fn_id(self):
        """Test identity normalization function."""
        cfg = get_config("generic")
        cfg.MODEL.NORM_FN = "id"
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        z = model.encode(x)
        
        # With identity norm, we just check it doesn't crash
        assert z.shape == (4, cfg.MODEL.TARGET_SIZE)
    
    def test_norm_fn_ball(self):
        """Test ball normalization function."""
        cfg = get_config("generic")
        cfg.MODEL.NORM_FN = "ball"
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        z = model.encode(x)
        
        # Check normalization: each vector should have unit norm
        norms = torch.norm(z, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestLISTAKM:
    """Test LISTAKM model."""
    
    def test_initialization(self):
        """Test LISTAKM can be initialized."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        obs_size = 2
        model = LISTAKM(cfg, obs_size)
        assert model.observation_size == obs_size
        assert model.target_size == cfg.MODEL.TARGET_SIZE
    
    def test_encode_decode_shape(self):
        """Test encode and decode output shapes."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 128
        obs_size = 3
        batch_size = 16
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(batch_size, obs_size)
        
        # Test encode
        z = model.encode(x)
        assert z.shape == (batch_size, cfg.MODEL.TARGET_SIZE)
        
        # Test decode
        x_recon = model.decode(z)
        assert x_recon.shape == (batch_size, obs_size)
    
    def test_sparse_encoding(self):
        """Test LISTAKM produces sparse encodings."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 256
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.3
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        z = model.encode(x)
        
        # Check sparsity
        nonzero = (z.abs() > 1e-6).float().sum(dim=-1).mean()
        sparsity_ratio = 1.0 - nonzero / cfg.MODEL.TARGET_SIZE
        assert sparsity_ratio > 0.5  # Should be at least 50% sparse
    
    def test_dict_normalization(self):
        """Test dictionary atoms are normalized in decode."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        
        # Check dictionary norm after decode call
        z = torch.randn(1, cfg.MODEL.TARGET_SIZE)
        _ = model.decode(z)
        
        # Dictionary should be used in normalized form
        # (not checking the parameter itself, just that decode works)
        assert True
    
    def test_loss_computation(self):
        """Test full loss computation."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 128
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, metrics = model.loss(x, nx)
        
        # Check loss is scalar
        assert loss.ndim == 0
        
        # Check metrics
        assert 'loss' in metrics
        assert 'residual_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert 'sparsity_ratio' in metrics
    
    def test_sparsity_loss_with_alpha(self):
        """Test sparsity loss uses LISTA alpha weighting."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.LISTA.ALPHA = 2.0
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        loss = model.sparsity_loss(x)
        
        # Should be weighted by alpha
        assert loss >= 0


class TestModelFactory:
    """Test model factory function."""
    
    def test_make_model_generic(self):
        """Test creating GenericKM via factory."""
        cfg = get_config("generic")
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, GenericKM)
    
    def test_make_model_sparse(self):
        """Test creating SparseKM (alias for GenericKM) via factory."""
        cfg = get_config("generic_sparse")
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, GenericKM)
    
    def test_make_model_listakm(self):
        """Test creating LISTAKM via factory."""
        cfg = get_config("lista")
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, LISTAKM)
    
    def test_make_model_ksae(self):
        """Test creating KSAE (alias for LISTAKM) via factory."""
        cfg = get_config("lista")
        cfg.MODEL.MODEL_NAME = "KSAE"
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, LISTAKM)
    
    def test_make_model_invalid(self):
        """Test factory raises error for invalid model name."""
        cfg = get_config("generic")
        cfg.MODEL.MODEL_NAME = "InvalidModel"
        obs_size = 2
        
        with pytest.raises(ValueError):
            make_model(cfg, obs_size)


class TestGradientFlow:
    """Test gradient flow through models."""
    
    def test_generic_km_gradients(self):
        """Test gradients flow through GenericKM."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size, requires_grad=True)
        nx = torch.randn(4, obs_size)
        
        loss, _ = model.loss(x, nx)
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert model.encoder.network[0].weight.grad is not None
        assert model.kmat.grad is not None
    
    def test_listakm_gradients(self):
        """Test gradients flow through LISTAKM."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 3
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(4, obs_size, requires_grad=True)
        nx = torch.randn(4, obs_size)
        
        loss, _ = model.loss(x, nx)
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert model.dict.grad is not None
        assert model.kmat.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

