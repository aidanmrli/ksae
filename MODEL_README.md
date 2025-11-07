# PyTorch Koopman Model Implementation

## Overview

This document describes the PyTorch implementation of Koopman Autoencoder models, converted from the original JAX/Flax implementation in `notebooks/koopman_copy.py`.

## Files Created

### Core Implementation
- **`model.py`**: Complete PyTorch implementation of Koopman models
  - `MLPCoder`: Multi-layer perceptron for encoding/decoding
  - `LISTA`: Learned Iterative Soft-Thresholding Algorithm for sparse coding
  - `KoopmanMachine`: Abstract base class
  - `GenericKM`: Standard Koopman autoencoder with MLP encoder
  - `LISTAKM`: Koopman machine with LISTA sparse encoder
  - `make_model()`: Factory function to create models from config

### Testing
- **`tests/test_model.py`**: 39 unit tests covering all model components
- **`tests/test_integration.py`**: 7 integration tests verifying model-data compatibility

### Examples
- **`example_usage.py`**: Demonstrates usage with 4 complete examples

## Architecture

### Model Components

```
KoopmanMachine (Abstract Base)
├── GenericKM (Standard Koopman AE)
│   ├── MLPCoder encoder
│   ├── MLPCoder decoder
│   └── Learnable Koopman matrix
└── LISTAKM (Sparse Koopman AE)
    ├── LISTA encoder (sparse)
    ├── Dictionary decoder (normalized)
    └── Learnable Koopman matrix
```

### Key Features

1. **Modular Design**: Components can be easily swapped and customized
2. **Config-Driven**: Uses the dataclass config system from `config.py`
3. **Environment-Compatible**: Works seamlessly with dynamical systems from `data.py`
4. **Well-Tested**: 46 comprehensive tests ensure correctness
5. **Documented**: Extensive docstrings following ML research standards

## Usage

### Quick Start

```python
from config import get_config
from data import make_env, VectorWrapper
from model import make_model
import torch

# 1. Create config
cfg = get_config("generic")  # or "lista" for sparse
cfg.ENV.ENV_NAME = "duffing"
cfg.MODEL.TARGET_SIZE = 64

# 2. Create environment and model
env = make_env(cfg)
model = make_model(cfg, env.observation_size)

# 3. Generate training data
vec_env = VectorWrapper(env, batch_size=32)
rng = torch.Generator()
rng.manual_seed(0)
x = vec_env.reset(rng)
nx = vec_env.step(x)

# 4. Compute loss
loss, metrics = model.loss(x, nx)
print(f"Loss: {metrics['loss']:.4f}")
```

### Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(num_steps):
    # Sample data
    x = vec_env.reset(rng)
    nx = vec_env.step(x)
    
    # Training step
    optimizer.zero_grad()
    loss, metrics = model.loss(x, nx)
    loss.backward()
    optimizer.step()
```

### Prediction

```python
# Multi-step prediction
x0 = env.reset(rng)
predictions = [x0]
x = x0

for _ in range(num_steps):
    x = model.step_env(x)
    predictions.append(x)

trajectory = torch.stack(predictions)
```

## Model Types

### GenericKM (Standard Koopman AE)

**Configuration**: `get_config("generic")`

- MLP encoder and decoder
- Configurable architecture via `MODEL.ENCODER.LAYERS` and `MODEL.DECODER.LAYERS`
- Optional normalization: `MODEL.NORM_FN` ("id" or "ball")
- Optional sparsity: `MODEL.SPARSITY_COEFF` > 0

**Use cases**: General-purpose Koopman learning, medium-dimensional latent spaces (16-128)

### LISTAKM (Sparse Koopman AE)

**Configuration**: `get_config("lista")`

- LISTA sparse encoder with iterative soft-thresholding
- Normalized dictionary decoder
- High-dimensional sparse latent space (512-2048)
- Configurable via:
  - `MODEL.ENCODER.LISTA.NUM_LOOPS`: Number of LISTA iterations
  - `MODEL.ENCODER.LISTA.ALPHA`: Sparsity threshold
  - `MODEL.ENCODER.LISTA.L`: Lipschitz constant estimate
  - `MODEL.ENCODER.LISTA.LINEAR_ENCODER`: Use linear (True) or MLP (False) encoder

**Use cases**: Sparse feature discovery, interpretable representations, overcomplete dictionaries

## Loss Components

The model computes a weighted combination of losses:

```python
total_loss = (
    RES_COEFF * residual_loss +        # Alignment in latent space
    RECONST_COEFF * reconstruction_loss + # Reconstruction quality
    PRED_COEFF * prediction_loss +      # Prediction accuracy
    SPARSITY_COEFF * sparsity_loss      # L1 regularization
)
```

Configure weights in `config.py` under `MODEL.*_COEFF`.

## Metrics

Each `loss()` call returns metrics:

- `loss`: Total weighted loss
- `residual_loss`: Koopman alignment error
- `reconst_loss`: Reconstruction error
- `prediction_loss`: One-step prediction error
- `sparsity_loss`: L1 norm of latent codes
- `A_max_eigenvalue`: Maximum eigenvalue of Koopman matrix
- `sparsity_ratio`: Fraction of zero-valued codes

## Testing

Run tests to verify correctness:

```bash
# All model tests
pytest tests/test_model.py -v

# Integration tests
pytest tests/test_integration.py -v

# All tests
pytest tests/test_model.py tests/test_integration.py -v
```

**Test coverage**:
- Utility functions (7 tests)
- MLPCoder (4 tests)
- LISTA (4 tests)
- GenericKM (11 tests)
- LISTAKM (6 tests)
- Model factory (5 tests)
- Gradient flow (2 tests)
- Integration (7 tests)

## Examples

Run complete examples:

```bash
python example_usage.py
```

This demonstrates:
1. GenericKM training on Duffing oscillator
2. LISTAKM training on Pendulum (sparse)
3. Koopman eigenvalue analysis
4. Testing on all dynamical systems

## Comparison: JAX vs PyTorch

### JAX/Flax → PyTorch Conversions

| JAX/Flax | PyTorch |
|----------|---------|
| `nn.Module` | `nn.Module` |
| `setup()` | `__init__()` |
| `__call__()` | `forward()` |
| `self.param()` | `nn.Parameter()` |
| `nn.Dense()` | `nn.Linear()` |
| `jnp.ndarray` | `torch.Tensor` |
| `jax.random.PRNGKey` | `torch.Generator` |
| Functional style | Object-oriented |

### Key Differences

1. **Parameter initialization**: PyTorch uses `nn.Parameter` and `.data` assignments
2. **Random number generation**: PyTorch uses `torch.Generator` for reproducibility
3. **vmap**: Replaced with explicit loops or `torch.vmap` where needed
4. **jit compilation**: Not needed in PyTorch; autograd handles gradients
5. **Eigenvalue computation**: Uses `torch.linalg.eigvals` (no backend specification needed)

## Integration with Existing Codebase

The PyTorch models integrate seamlessly with:

- **`config.py`**: Uses `Config` dataclasses for all configuration
- **`data.py`**: Compatible with all `Env` classes and `VectorWrapper`
- Supports all dynamical systems: Duffing, Pendulum, Lotka-Volterra, Lorenz63, Parabolic

## Best Practices

1. **Seeding**: Always seed RNG for reproducibility
   ```python
   rng = torch.Generator()
   rng.manual_seed(42)
   ```

2. **Batch processing**: Use `VectorWrapper` for efficient batch sampling
   ```python
   vec_env = VectorWrapper(env, batch_size=32)
   ```

3. **Configuration**: Start with predefined configs and modify as needed
   ```python
   cfg = get_config("generic")
   cfg.MODEL.TARGET_SIZE = 128  # Customize
   ```

4. **Loss monitoring**: Track all metrics during training
   ```python
   loss, metrics = model.loss(x, nx)
   print(metrics)  # Monitor all components
   ```

5. **Device management**: Move model and data to GPU if available
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   x = x.to(device)
   ```

## Future Extensions

Potential improvements:
- Add learning rate schedulers
- Implement advanced Koopman matrix parameterizations
- Add support for control inputs (actuated systems)
- Implement multi-step prediction losses
- Add visualization utilities for latent spaces

## References

- **LISTA**: Gregor & LeCun (2010) - "Learning Fast Approximations of Sparse Coding"
- **Koopman Theory**: Koopman (1931) - "Hamiltonian Systems and Transformation in Hilbert Space"
- **Deep Learning for Koopman**: Champion et al. (2019) - "Data-driven discovery of coordinates and governing equations"

## Questions?

For issues or questions:
1. Check test cases in `tests/test_model.py` for usage examples
2. Run `example_usage.py` to see working demonstrations
3. Refer to docstrings in `model.py` for detailed API documentation

