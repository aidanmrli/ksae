"""
PyTorch implementation of Koopman Autoencoder models.

This module provides:
- MLPCoder: Multi-layer perceptron for encoding/decoding
- LISTA: Learned Iterative Soft-Thresholding Algorithm for sparse coding
- KoopmanMachine: Abstract base class for Koopman operator learning
- GenericKM: Standard Koopman autoencoder with MLP encoder
- LISTAKM: Koopman machine with LISTA sparse encoder
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
from config import Config


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def shrink(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft thresholding operator (shrinkage). Used in LISTA.
    
    Args:
        x: Input tensor
        threshold: Threshold value for soft thresholding
        
    Returns:
        Shrunk tensor
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        name: Activation name ('relu', 'tanh', 'gelu')
        
    Returns:
        Activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(activations.keys())}")
    return activations[name]


# ---------------------------------------------------------------------------
# Network Components
# ---------------------------------------------------------------------------


class MLPCoder(nn.Module):
    """Multi-layer perceptron for encoding or decoding.
    
    Args:
        input_size: Input dimension
        target_size: Output dimension
        hidden_layers: List of hidden layer sizes
        last_relu: Whether to apply ReLU to the output
        use_bias: Whether to use bias in linear layers
        activation: Activation function name
    """
    
    def __init__(
        self,
        input_size: int,
        target_size: int,
        hidden_layers: List[int],
        last_relu: bool = False,
        use_bias: bool = False,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.hidden_layers = hidden_layers
        self.last_relu = last_relu
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size, bias=use_bias))
            layers.append(get_activation(activation))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, target_size, bias=use_bias))
        if last_relu:
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [..., input_size]
            
        Returns:
            Output tensor of shape [..., target_size]
        """
        return self.network(x)


class LISTA(nn.Module):
    """Learned Iterative Soft-Thresholding Algorithm (LISTA) encoder.
    
    This module implements a LISTA-style encoder: an unrolled, fixed-depth
    approximation to sparse coding built from alternating affine transforms
    and an elementwise soft-thresholding nonlinearity.

    Canonical LISTA (Gregor & LeCun, 2010) uses a linear pre-activation
    z-affine map W_e x and shared "mutual-inhibition" matrix S, with the
    nonlinearity given by the soft-thresholding (shrinkage) operator
    T_λ(v)_i = sign(v_i) * max(|v_i| - λ, 0). The overall encoder is
    therefore nonlinear due to T_λ.
    
    Shapes (standard convention):
        x ∈ ℝ^{xdim},  z ∈ ℝ^{zdim}
        Dictionary W_d ∈ ℝ^{xdim × zdim}  (columns are atoms)
        Linear encoder W_e = (1/L) W_dᵀ ∈ ℝ^{zdim × xdim}
        Inhibition S = I - (1/L) W_dᵀ W_d ∈ ℝ^{zdim × zdim}

    Iterations:
        c = W_e x
        z^(0) = T_{α/L}(c)
        for k = 0..K-1:
            z^(k+1) = T_{α/L}(S z^(k) + c)
        return z^(K)

    Notes:
        • If `use_linear_encode=True`, the module uses the canonical linear
          pre-activation W_e x. If `False`, an MLP can be used to produce c;
          this yields a LISTA-style unrolled network rather than canonical LISTA.
        • L is a Lipschitz constant estimate (e.g., ≥ spectral norm of W_dᵀ W_d).
        • α controls sparsity; K is the number of unrolled iterations.

    Args:
        cfg: Configuration object.
        xdim: Input dimension.
        Wd_init: Initial dictionary matrix with shape [xdim, zdim].
    """
    
    def __init__(self, cfg: Config, xdim: int, Wd_init: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.xdim = xdim
        self.zdim = cfg.MODEL.TARGET_SIZE
        self.num_loops = cfg.MODEL.ENCODER.LISTA.NUM_LOOPS
        self.alpha = cfg.MODEL.ENCODER.LISTA.ALPHA
        self.L = cfg.MODEL.ENCODER.LISTA.L
        self.use_linear_encode = cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER
        
        assert Wd_init.shape == (xdim, self.zdim), \
            f"Wd_init shape {Wd_init.shape} doesn't match expected ({xdim}, {self.zdim})"
        
        if self.use_linear_encode:
            self.We = nn.Linear(xdim, self.zdim, bias=False)
            # Initialize as (1/L) * Wd^T
            with torch.no_grad():
                self.We.weight.copy_((1.0 / self.L) * Wd_init.T)  # [zdim, xdim]
        else:
            self.We = MLPCoder(
                input_size=xdim,
                target_size=self.zdim,
                hidden_layers=cfg.MODEL.ENCODER.LAYERS,
                use_bias=cfg.MODEL.ENCODER.USE_BIAS,
                last_relu=cfg.MODEL.ENCODER.LAST_RELU,
                activation=cfg.MODEL.ENCODER.ACTIVATION,
            )
        
        S_init = torch.eye(self.zdim) - (1.0 / self.L) * (Wd_init.T @ Wd_init)
        self.S = nn.Parameter(S_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: iterative soft-thresholding.
        
        Args:
            x: Input tensor of shape [..., xdim]
            
        Returns:
            Sparse codes of shape [..., zdim]
        """
        # Initial encoding
        nonsparse_code = self.We(x)
        
        # Initialize with soft-thresholding of initial encoding
        z = shrink(nonsparse_code, self.alpha / self.L)
        
        # Iterative refinement
        for _ in range(self.num_loops):
            z = shrink(z @ self.S + nonsparse_code, self.alpha / self.L)
        
        return z


# ---------------------------------------------------------------------------
# Koopman Machine Base Class
# ---------------------------------------------------------------------------

class KoopmanMachine(ABC, nn.Module):
    """Abstract base class for Koopman operator learning.
    
    The Koopman operator is a linear operator that provides a mathematical 
    framework for representing the dynamics of a nonlinear dynamical system (NLDS) 
    in terms of an infinite-dimensional linear operator. 
    Formally, the Koopman operator advances a measurement function forward in time 
    through the underlying system dynamics.
    
    This class provides the interface for learning Koopman representations.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__()
        self.cfg = cfg
        self.observation_size = observation_size
        self.target_size = cfg.MODEL.TARGET_SIZE
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Latent codes of shape [..., target_size]
        """
        pass
    
    @abstractmethod
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to observation space.
        
        Args:
            y: Latent representations of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        pass
    
    @abstractmethod
    def kmatrix(self) -> torch.Tensor:
        """Extract the learned Koopman matrix from parameters.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        pass
    
    def residual(self, x: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss between consecutive states in latent space.
        Determines how linearly aligned x & nx are in the latent space.
        
        Args:
            x: Current states of shape [..., observation_size]
            nx: Next states of shape [..., observation_size]
            
        Returns:
            Residual norms of shape [...]
        """
        y = self.encode(x)
        ny = self.encode(nx)
        kmat = self.kmatrix()
        return torch.norm(y @ kmat - ny, dim=-1)
    
    def reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction via encode-decode.
        
        Args:
            x: shape [..., observation_size]
            
        Returns:
            Reconstructions of shape [..., observation_size]
        """
        return self.decode(self.encode(x))
    
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss on latent codes.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return torch.norm(z, p=1, dim=-1).mean()
    
    def step_latent(self, y: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space using Koopman matrix.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Next latent codes of shape [..., target_size]
        """
        kmat = self.kmatrix()
        return y @ kmat
    
    def step_env(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next observation using Koopman dynamics.
        
        Args:
            x: Current observations of shape [..., observation_size]
            
        Returns:
            Predicted next observations of shape [..., observation_size]
        """
        y = self.encode(x)
        ny = self.step_latent(y)
        nx = self.decode(ny)
        return nx
    
    def loss(
        self,
        x: torch.Tensor,
        nx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and metrics.
        
        Args:
            x: Current states of shape [batch_size, observation_size]
            nx: Next states of shape [batch_size, observation_size]
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Linear prediction loss
        kmat = self.kmatrix()
        prediction = self.decode(self.encode(x) @ kmat)
        prediction_loss = torch.norm(prediction - nx, dim=-1).mean()
        
        # Linear dynamics alignment loss
        residual_loss = self.residual(x, nx).mean()
        
        # Reconstruction loss
        reconst_loss = torch.norm(x - self.reconstruction(x), dim=-1).mean()
        reconst_loss += torch.norm(nx - self.reconstruction(nx), dim=-1).mean()
        
        # Sparsity loss
        sparsity_loss = self.sparsity_loss(x)
        sparsity_loss += self.sparsity_loss(nx)
        sparsity_loss *= 0.5
        
        # Koopman matrix eigenvalues
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(kmat)
            max_eigenvalue = torch.max(eigvals.real)
        
        # Nonzero codes
        with torch.no_grad():
            z = self.encode(x)
            num_nonzero_codes = (z != 0).float().sum(dim=-1).mean()
            sparsity_ratio = 1.0 - num_nonzero_codes / self.target_size
        
        # Total weighted loss
        total_loss = (
            self.cfg.MODEL.RES_COEFF * residual_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.cfg.MODEL.SPARSITY_COEFF * sparsity_loss
        )
        
        metrics = {
            'loss': total_loss.item(),
            'residual_loss': residual_loss.item(),
            'reconst_loss': reconst_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'A_max_eigenvalue': max_eigenvalue.item(),
            'sparsity_ratio': sparsity_ratio.item(),
        }
        
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------------------------


class GenericKM(KoopmanMachine):
    """Generic Koopman Machine with MLP encoder and decoder.
    
    This is the standard Koopman autoencoder with configurable MLP architectures.
    Optionally supports normalization of latent codes.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        
        # Encoder
        self.encoder = MLPCoder(
            input_size=observation_size,
            target_size=cfg.MODEL.TARGET_SIZE,
            hidden_layers=cfg.MODEL.ENCODER.LAYERS,
            use_bias=cfg.MODEL.ENCODER.USE_BIAS,
            last_relu=cfg.MODEL.ENCODER.LAST_RELU,
            activation=cfg.MODEL.ENCODER.ACTIVATION,
        )
        
        # Decoder
        self.decoder = MLPCoder(
            input_size=cfg.MODEL.TARGET_SIZE,
            target_size=observation_size,
            hidden_layers=cfg.MODEL.DECODER.LAYERS,
            use_bias=cfg.MODEL.DECODER.USE_BIAS,
            last_relu=False,
            activation=cfg.MODEL.DECODER.ACTIVATION,
        )
        
        # Koopman matrix (learnable)
        self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))

        self.norm_fn_name = cfg.MODEL.NORM_FN
    
    def _norm_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to latent codes.
        
        Args:
            x: Latent codes
            
        Returns:
            Normalized latent codes
        """
        if self.norm_fn_name == 'id':
            return x
        elif self.norm_fn_name == 'ball':
            return x / torch.norm(x, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown norm function '{self.norm_fn_name}'")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Latent codes of shape [..., target_size]
        """
        y = self.encoder(x)
        return self._norm_fn(y)
    
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to observation space.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        return self.decoder(y)
    
    def kmatrix(self) -> torch.Tensor:
        """Get the Koopman matrix.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        return self.kmat
    
    def step_latent(self, y: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space with normalization.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Next latent codes of shape [..., target_size]
        """
        ny = y @ self.kmatrix()
        return self._norm_fn(ny)

# TODO: test this class with experiments. Sweep over the sparsity coefficient values.
# TODO: test this on the Lyapunov environment
class LISTAKM(KoopmanMachine):
    """Koopman Machine with LISTA sparse encoder.
    
    Uses the Learned Iterative Soft-Thresholding Algorithm (LISTA) for sparse
    encoding. The decoder uses a normalized dictionary.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        
        # Initialize dictionary (decoder weights)
        Wd_init = torch.randn(cfg.MODEL.TARGET_SIZE, observation_size) * 0.01
        self.register_buffer('dict_init', Wd_init.clone())
        self.dict = nn.Parameter(Wd_init)
        
        # LISTA encoder
        self.lista = LISTA(cfg, observation_size, Wd_init)
        
        # Koopman matrix (learnable)
        self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations using LISTA.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Sparse latent codes of shape [..., target_size]
        """
        return self.lista(x)
    
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode using normalized dictionary.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        # Normalize dictionary atoms
        wd = self.dict / torch.norm(self.dict, dim=1, keepdim=True).clamp(min=1e-4)
        return y @ wd
    
    def kmatrix(self) -> torch.Tensor:
        """Get the Koopman matrix.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        return self.kmat
    
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss weighted by LISTA alpha.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return self.cfg.MODEL.ENCODER.LISTA.ALPHA * torch.norm(z, p=1, dim=-1).mean()


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


_MODEL_REGISTRY = {
    "GenericKM": GenericKM,
    "SparseKM": GenericKM,  # Same as GenericKM, configured via sparsity coeff
    "LISTAKM": LISTAKM,
}


def make_model(cfg: Config, observation_size: int) -> KoopmanMachine:
    """Factory function to create model from configuration.
    
    Args:
        cfg: Configuration object with MODEL.MODEL_NAME specifying the model type
        observation_size: Dimension of the observation space
        
    Returns:
        KoopmanMachine instance
        
    Raises:
        ValueError: If MODEL_NAME is not in registry
    """
    model_name = cfg.MODEL.MODEL_NAME
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](cfg, observation_size)