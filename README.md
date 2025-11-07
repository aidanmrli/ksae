# Koopman Sparse Autoencoder (KSAE)

PyTorch-based research codebase for learning Koopman operator representations of nonlinear dynamical systems using autoencoders with sparsity constraints.

## Overview

This repository implements several variants of Koopman autoencoders:

- **GenericKM**: Standard Koopman autoencoder with MLP encoder
- **SparseKM**: Koopman autoencoder with L1 sparsity regularization
- **LISTAKM**: Learned Iterative Soft-Thresholding Algorithm (LISTA) based sparse encoder

The code is designed for **rapid prototyping** and **reproducible research** with a focus on:
- Clean, modular architecture
- Type-safe configuration system
- Comprehensive unit tests
- Notebook-friendly design
- Local-first development (no cloud dependencies)

## Quick Start

### Installation

```bash
pip install -e .
```

### Train a Model

```bash
# Train with defaults
python train.py --config generic_sparse --env duffing --num_steps 4000

# Sweep over sparsity coefficient
python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.001
python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.01
python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.1

# Custom learning rate and latent dimension
python train.py --config lista --env lotka_volterra --lr 1e-4 --target_size 1024

python train.py \
  --config generic_sparse \
  --env lyapunov \
  --num_steps 5000 \
  --batch_size 256 \
  --target_size 64 \
  --sparsity_coeff 0.001 \
  --pairwise \
  --seed 0 \
  --device cuda

python train.py \
  --config lista \
  --env lyapunov \
  --num_steps 4000 \
  --batch_size 256 \
  --target_size 128 \
  --sparsity_coeff 0.01 \
  --pairwise \
  --seed 0 \
  --device cuda
```

### Programmatic Training

```python
from config import get_config
from train import train

# Start with a preset configuration
cfg = get_config("generic_sparse")
cfg.ENV.ENV_NAME = "duffing"

# Customize any parameters
cfg.MODEL.SPARSITY_COEFF = 0.005
cfg.TRAIN.NUM_STEPS = 15000
cfg.TRAIN.LR = 5e-4

# Train
model = train(cfg, log_dir="./runs/my_experiment")
```

See `example_train.py` for more examples.

## Design Philosophy

### Configuration System

This codebase uses a **preset-based configuration** approach that differs from typical ML scripts:

#### Original JAX Notebook Approach
The original code used inline configuration with manual parameter setting:

```python
# koopman_copy.py (original JAX version)
if TRAIN_CFG == 'generic_sparse':
    cfg.TRAIN.LR = 1e-3
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.ENCODER.LAST_RELU = True
    cfg.MODEL.ENCODER.USE_BIAS = True
    cfg.MODEL.RECONST_COEFF = 0.5
    cfg.MODEL.SPARSITY_COEFF = .01
    # ... 15-20 more parameters
```

#### New PyTorch Approach
We've captured these parameter combinations as **validated presets** in `config.py`:

```python
# config.py
def get_train_generic_sparse_config() -> Config:
    """Training configuration for GenericKM with L1 regularization."""
    cfg = Config()
    cfg.TRAIN.LR = 1e-3
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    # ... all parameters that work well together
    return cfg
```

### Why This Design?

**1. Eliminates Configuration Errors**
- Parameter combinations are tested and known to work
- No accidentally mixing incompatible settings
- Type-safe configuration via Python dataclasses

**2. Two Interfaces for Two Use Cases**

**CLI: Quick Experiments**
```bash
# Common parameters exposed for sweeps
python train.py --config generic_sparse --env duffing \
    --sparsity_coeff 0.01 --lr 1e-3 --num_steps 10000
```
- Uses sensible preset defaults
- Only exposes frequently-tuned hyperparameters
- Great for parameter sweeps and quick iterations

**Python API: Full Control**
```python
# Advanced users can modify anything
cfg = get_config("generic_sparse")
cfg.MODEL.ENCODER.LAYERS = [128, 128, 128]  # Custom architecture
cfg.MODEL.RECONST_COEFF = 0.3  # Fine-tune loss weights
cfg.MODEL.NORM_FN = "ball"  # Change normalization
train(cfg)
```
- Full access to all ~40 configuration parameters
- No command-line verbosity
- Perfect for notebooks and research experiments

**3. Better Than Alternatives**

❌ **Full CLI Exposure** (every parameter as a flag)
```bash
# Gets unmanageable quickly
python train.py --model GenericKM --target_size 64 --encoder_layers 64 64 \
    --encoder_activation relu --encoder_last_relu --encoder_use_bias \
    --decoder_layers --decoder_use_bias --norm_fn id \
    --res_coeff 1.0 --reconst_coeff 0.5 --pred_coeff 0.0 --sparsity_coeff 0.01 \
    --env duffing --dt 0.01 --batch_size 256 --num_steps 10000 --lr 1e-3
    # ... this is getting ridiculous
```

❌ **Config Files Only** (YAML/JSON)
```yaml
# config.yaml - must edit file for every experiment
model:
  model_name: GenericKM
  target_size: 64
  encoder:
    layers: [64, 64]
    # ... 30 more lines
```
- Not notebook-friendly
- Hard to version control experiments
- Requires file I/O for simple changes

✅ **Our Hybrid Approach**
- Presets for validated configurations
- Simple CLI for common hyperparameters
- Full Python API for power users
- Best for ML research workflows

### Exposed CLI Parameters

We expose parameters that you typically **sweep over** in experiments:

| Parameter | Why Exposed | Example Use |
|-----------|-------------|-------------|
| `--sparsity_coeff` | Primary hyperparameter for sparse models | `0.001, 0.01, 0.1` |
| `--lr` | Learning rate tuning | `1e-4, 5e-4, 1e-3` |
| `--target_size` | Latent dimension experiments | `32, 64, 128` |
| `--num_steps` | Training duration | `5000, 10000, 20000` |
| `--batch_size` | Memory/speed tradeoffs | `128, 256, 512` |

Architecture parameters (encoder layers, activation functions, etc.) are **not exposed** because:
- They're part of the model definition (preset captures the right combination)
- Changing them often requires changing other parameters too
- They're easily modified via Python API for architecture search

## Repository Structure

```
ksae/
├── config.py              # Configuration system with presets
├── data.py                # Dynamical systems environments
├── model.py               # Koopman autoencoder models
├── train.py               # Training script (CLI + API)
├── example_train.py       # Usage examples
├── plot_metrics.py        # Visualization utilities
├── tests/                 # Unit tests
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_train.py
├── notebooks/             # Research notebooks
├── README.md              # This file
└── TRAINING.md            # Detailed training guide
```

## Available Configurations

### `generic` - Standard Koopman Autoencoder
```bash
python train.py --config generic --env duffing
```
- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: [64, 64] MLP
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.02)
- **Use case**: Baseline, dense representations

### `generic_sparse` - Sparse Koopman with L1
```bash
python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.01
```
- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: [64, 64] MLP with ReLU + bias
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.5), Sparsity (0.01)
- **Use case**: Learning interpretable sparse representations

### `generic_prediction` - Prediction-focused
```bash
python train.py --config generic_prediction --env duffing
```
- **Loss weights**: Prediction (1.0), others disabled
- **Use case**: Pure forecasting without reconstruction

### `lista` - LISTA Sparse Encoder
```bash
python train.py --config lista --env lotka_volterra --target_size 2048
```
- **Model**: LISTAKM
- **Target size**: 2048 (overcomplete)
- **Encoder**: LISTA with 10 iterations
- **Decoder**: Normalized dictionary
- **Loss weights**: Residual (1.0), Reconstruction (1.0), Sparsity (1.0)
- **Use case**: High-dimensional sparse coding

### `lista_nonlinear` - LISTA with MLP
```bash
python train.py --config lista_nonlinear --env lorenz63
```
- **Model**: LISTAKM with nonlinear pre-activation
- **Encoder**: [64, 64, 64] MLP → LISTA
- **Use case**: More expressive sparse representations

## Environments

| Environment | Dimension | Description |
|------------|-----------|-------------|
| `duffing` | 2D | Duffing oscillator with two stable centers |
| `pendulum` | 2D | Simple pendulum |
| `lotka_volterra` | 2D | Predator-prey dynamics |
| `lorenz63` | 3D | Chaotic Lorenz attractor |
| `parabolic` | 2D | Parabolic attractor (analytical Koopman) |

## Training Output

Each training run creates a timestamped directory:

```
runs/kae/20251106-223912/
├── config.json              # Full configuration (reproducibility)
├── checkpoint.pt            # Best model (lowest loss)
├── last.pt                  # Latest checkpoint
├── metrics_history.jsonl    # Time series of all metrics
├── metrics_summary.json     # Summary statistics
└── final_metrics.json       # Final step metrics
```

### Visualize Training

```bash
# Plot all metrics
python plot_metrics.py runs/kae/20251106-223912

# Plot specific metrics
python plot_metrics.py runs/kae/20251106-223912 --metrics loss sparsity_ratio

# Print summary
python plot_metrics.py runs/kae/20251106-223912 --summary

# Save plot
python plot_metrics.py runs/kae/20251106-223912 --save training_curves.png
```

## Hyperparameter Sweeps

### Example: Sparsity Coefficient Sweep

```bash
# Bash script for sweep
for sparsity in 0.0001 0.001 0.01 0.1; do
    python train.py --config generic_sparse --env duffing \
        --sparsity_coeff $sparsity --num_steps 10000 \
        --log_dir ./runs/sparsity_sweep
done

# Analyze results
for dir in ./runs/sparsity_sweep/*/; do
    echo "Results for $dir:"
    python plot_metrics.py "$dir" --summary | grep sparsity_ratio
done
```

### Python API for Sweeps

```python
from config import get_config
from train import train

sparsity_values = [0.0001, 0.001, 0.01, 0.1]

for sparsity in sparsity_values:
    cfg = get_config("generic_sparse")
    cfg.ENV.ENV_NAME = "duffing"
    cfg.MODEL.SPARSITY_COEFF = sparsity
    cfg.TRAIN.NUM_STEPS = 10000
    
    model = train(cfg, log_dir=f"./runs/sparsity_{sparsity}")
```

## Loading Trained Models

```python
import torch
from config import Config
from data import make_env
from model import make_model

# Load configuration
cfg = Config.from_json('runs/kae/20251106-223912/config.json')

# Create model
env = make_env(cfg)
model = make_model(cfg, env.observation_size)

# Load weights
checkpoint = torch.load('runs/kae/20251106-223912/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use model
with torch.no_grad():
    x = torch.randn(1, env.observation_size)
    z = model.encode(x)  # Latent representation
    x_recon = model.decode(z)  # Reconstruction
    x_next = model.step_env(x)  # Prediction
```

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_train.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## Advanced Usage

### Custom Configuration

```python
from config import Config, ModelConfig, TrainConfig

# Build from scratch
cfg = Config()
cfg.SEED = 42
cfg.ENV.ENV_NAME = "pendulum"
cfg.MODEL = ModelConfig(
    MODEL_NAME="GenericKM",
    TARGET_SIZE=128,
    SPARSITY_COEFF=0.005,
)
cfg.TRAIN = TrainConfig(
    NUM_STEPS=20000,
    BATCH_SIZE=512,
    LR=5e-4,
)

train(cfg)
```

### Checkpoint Resumption

```bash
# Resume training from checkpoint
python train.py --checkpoint runs/kae/20251106-223912/last.pt
```

### Device Selection

```bash
# Auto-detect best device
python train.py --config generic --env duffing --device cuda

# Force CPU
python train.py --config generic --env duffing --device cpu

# Use Apple Silicon GPU
python train.py --config generic --env duffing --device mps
```

## Future: Weights & Biases Integration

The current logging system (`MetricsLogger`) can be easily swapped with W&B:

```python
# In train.py, replace MetricsLogger with:
import wandb

wandb.init(project="koopman-ae", config=cfg.to_dict())
wandb.log(metrics, step=step)
```

This keeps the codebase flexible for cloud-based experiment tracking when needed.

## References

- **Koopman Theory**: Brunton, S. L., Budišić, M., Kaiser, E., & Kutz, J. N. (2022). Modern Koopman theory for dynamical systems. SIAM Review.
- **LISTA**: Gregor, K., & LeCun, Y. (2010). Learning fast approximations of sparse coding. ICML.
- **Koopman Autoencoders**: Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). Deep learning for universal linear embeddings of nonlinear dynamics. Nature Communications.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ksae2024,
  title = {Koopman Sparse Autoencoder},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ksae}
}
```

## License

See `LICENSE` file for details.

