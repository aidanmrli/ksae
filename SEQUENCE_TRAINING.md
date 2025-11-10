# Sequence-Based Training with ODE Integration

This document describes the new continuous-time Koopman training scheme implemented in this codebase.

## Overview

The training scheme now supports **sequence-based learning** with **ODE integration** for continuous-time Koopman dynamics. Instead of training on single-step transitions `(x_t, x_{t+1})`, the model now trains on sequences of length T=10.

## Mathematical Framework

### Continuous-Time Koopman Dynamics

The model learns a linear Koopman operator K that governs the evolution of latent states:

```
dz/dt = K @ z
```

This continuous-time formulation is integrated using adaptive Runge-Kutta methods (via `torchdiffeq`) to predict future states.

### Training Procedure

For each training batch:

1. **Generate sequence window**: `(x_t, x_{t+1}, ..., x_{t+T})` where T=10 by default

2. **Encode all states**: 
   - `z_i = φ(x_i)` for i = 0...T

3. **Integrate Koopman dynamics**:
   - Starting from `z_0`, integrate `dz/dt = Kz` over time points to get predicted latents `ẑ_1, ..., ẑ_T`
   - Uses adaptive RK method (dopri5) via `torchdiffeq.odeint`

4. **Decode both encoded and advanced latents**:
   - Reconstructions: `x̃_i = ψ(z_i)`
   - Predictions: `x̂_i = ψ(ẑ_i)`

5. **Compute losses**:
   - **Alignment**: `Σ||ẑ_i - z_i||²` for i = 1...T (latent space consistency)
   - **Reconstruction**: `Σ||x_i - x̃_i||²` for i = 0...T (encode-decode accuracy)
   - **Prediction**: `Σ||x_i - x̂_i||²` for i = 1...T (forward prediction)
   - **Sparsity**: `||z||₁` (L1 regularization with coefficient 10⁻³)
   - **Decoder regularization**: column-norm regularization on decoder weights

6. **Backprop and update** with AdamW:
   - Encoder/decoder: LR = 10⁻⁴
   - Koopman matrix K: LR = 10⁻⁵

## Key Changes

### 1. Data Generation (`data.py`)

**New Functions**:
- `generate_sequence_window()`: Creates sequences of T+1 consecutive states
- `VectorWrapper.generate_sequence_batch()`: Generates batched sequences

**Usage**:
```python
env = VectorWrapper(env, batch_size=256)
x_seq = env.generate_sequence_batch(rng, window_length=10)
# Returns: [batch_size, 11, observation_size] (includes initial state)
```

### 2. Model Updates (`model.py`)

**New Methods in `KoopmanMachine`**:
- `koopman_ode_func(t, z)`: ODE function for dz/dt = Kz
- `integrate_latent_ode(z0, t_span)`: Integrates latent dynamics using torchdiffeq or fallback Euler
- `rollout_sequence_ode(x0, num_steps, dt)`: Rolls out predictions using ODE integration
- `decoder_column_norm_regularization()`: Penalizes deviation from unit-norm decoder columns
- `loss_sequence(x_seq, dt)`: New sequence-based loss function

**Key Features**:
- Automatic fallback to Euler integration if `torchdiffeq` is not installed
- Works with both `GenericKM` and `LISTAKM` models
- Maintains backward compatibility with single-step training

### 3. Configuration (`config.py`)

**New Parameters in `TrainConfig`**:
```python
USE_SEQUENCE_LOSS: bool = True  # Enable sequence-based training
SEQUENCE_LENGTH: int = 10       # Number of forward steps (T)
```

**New Parameters in `ModelConfig`**:
```python
SPARSITY_COEFF: float = 1e-3      # Changed from 1.0 to match pseudologic
DECODER_REG_COEFF: float = 0.01   # Decoder column-norm regularization
```

### 4. Training Loop (`train.py`)

**Updates to `train_step()`**:
- Now accepts sequences and dispatches to appropriate loss function
- Passes `cfg` and `dt` for ODE integration

**Updates to `train()`**:
- Extracts `dt` from environment config
- Generates sequence batches when `USE_SEQUENCE_LOSS=True`
- Prints alignment/prediction losses for sequence mode
- Maintains backward compatibility with single-step mode

## Usage

### Quick Start

```python
from config import get_config
from train import train

# Get configuration
cfg = get_config("generic_sparse")
cfg.ENV.ENV_NAME = "duffing"

# Enable sequence training (enabled by default)
cfg.TRAIN.USE_SEQUENCE_LOSS = True
cfg.TRAIN.SEQUENCE_LENGTH = 10

# Set loss coefficients
cfg.MODEL.RES_COEFF = 1.0      # Alignment
cfg.MODEL.RECONST_COEFF = 1.0  # Reconstruction
cfg.MODEL.PRED_COEFF = 1.0     # Prediction
cfg.MODEL.SPARSITY_COEFF = 1e-3

# Train
model = train(cfg)
```

### Command Line

```bash
python example_sequence_train.py
```

### Disable Sequence Training (Use Old Behavior)

```python
cfg.TRAIN.USE_SEQUENCE_LOSS = False  # Use single-step training
```

## Dependencies

**Required**:
- `torch >= 2.2`
- Standard dependencies (numpy, scipy, matplotlib)

**Optional**:
- `torchdiffeq >= 0.2.3` (for adaptive RK integration)
  - If not installed, falls back to fixed-step RK4 integration
  - Install with: `pip install torchdiffeq`

## Backward Compatibility

All existing code continues to work:
- Set `cfg.TRAIN.USE_SEQUENCE_LOSS = False` to use single-step training
- Old configs and checkpoints remain compatible
- The `loss()` method still works for single-step training

## Implementation Details

### ODE Integration

The ODE integrator solves:
```
dz/dt = K @ z
z(0) = z_0
```

over time points `t = [0, dt, 2*dt, ..., T*dt]` where `dt` is the environment's time step (e.g., 0.01 for Duffing).

**Integration Methods**:
1. **dopri5** (default with torchdiffeq): Adaptive Dormand-Prince 5th order Runge-Kutta
2. **rk4** (fallback): Fixed-step 4th order Runge-Kutta when torchdiffeq unavailable

### Loss Computation

The sequence loss sums errors across the entire sequence:

```python
# Alignment (latent consistency)
alignment_loss = Σ ||ẑ_i - z_i||² for i=1..T

# Reconstruction (encode-decode)
reconst_loss = Σ ||x_i - ψ(z_i)||² for i=0..T

# Prediction (forward dynamics)
prediction_loss = Σ ||x_i - ψ(ẑ_i)||² for i=1..T
```

All losses are averaged over the batch dimension.

### Decoder Regularization

Encourages unit-norm columns in decoder weight matrices:

```python
decoder_reg = mean((||decoder_column|| - 1)²)
```

This helps maintain well-conditioned latent representations.

## Advantages

1. **Continuous-time formulation**: More faithful to underlying dynamics
2. **Multi-step training**: Learns to predict further into the future
3. **Consistent integration**: Same ODE solver for training and evaluation
4. **Better generalization**: Sequence training improves long-horizon prediction
5. **Adaptive integration**: dopri5 adjusts step size for accuracy vs. speed

## Limitations

- Slower training per step (10x longer sequences + ODE integration)
- Higher memory usage (stores full sequences)
- Requires careful tuning of loss coefficients

## Future Work

- Implement parallel sequence generation for efficiency
- Add support for variable-length sequences
- Experiment with different ODE solvers
- Add curriculum learning (start with short sequences, increase over time)

## Examples

See `example_sequence_train.py` for a complete working example.

## References

This implementation follows the training scheme described in the project pseudologic, with continuous-time Koopman dynamics and multi-step sequence training.

