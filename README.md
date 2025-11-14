# Sparse Koopman Autoencoder (SKAE)

PyTorch-based research codebase for learning Koopman operator representations of nonlinear dynamical systems using autoencoders with sparsity constraints.

## Overview

This repository implements several variants of Koopman autoencoders:

- **GenericKM**: Standard Koopman autoencoder with MLP encoder
- **SparseKM**: Koopman autoencoder with L1 sparsity regularization
- **LISTAKM**: Learned Iterative Soft-Thresholding Algorithm (LISTA) based sparse encoder

## Quick Start

### Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management.

**Prerequisites**:
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install the project and dependencies**:
```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd skae

# Install from lock file (reproducible, recommended)
uv sync

# Alternative: Install without lock file
uv pip install -e .
```

### Train a Model

```bash
# Train with defaults on the Duffing Oscillator
uv run python train.py --config generic_sparse --env duffing --pairwise --num_steps 20000

# Sweep over sparsity coefficient
python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.001
python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.01
python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.1

# Custom learning rate and latent dimension
python train.py \
  --config generic_sparse \
  --env lyapunov \
  --num_steps 20000 \
  --batch_size 256 \
  --target_size 64 \
  --reconst_coeff 0.02 \
  --pred_coeff 1.0 \
  --sparsity_coeff 0.001 \
  --pairwise \
  --seed 0 \
  --device cuda

python train.py \
  --config lista \
  --env lyapunov \
  --num_steps 5000 \
  --batch_size 256 \
  --target_size 64 \
  --reconst_coeff 0.02 \
  --pred_coeff 1.0 \
  --pairwise \
  --seed 0 \
  --device cuda
```

## Repository Structure

```
skae/
├── config.py              # Configuration system with presets
├── data.py                # Dynamical systems environments
├── model.py               # Koopman autoencoder models
├── train.py               # Training script (CLI + API)
├── evaluation.py          # Model evaluation
├── example_train.py       # Usage examples
├── plot_metrics.py        # Visualization utilities
├── tests/                 # Unit tests
├── notebooks/             # Research notebooks
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

### `generic_sparse` - Sparse Koopman with L1 regularization
```bash
python train.py --config generic_sparse --env duffing --sparsity_coeff 0.01
```
- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: [64, 64] MLP with ReLU + bias
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.5), Sparsity (0.01)

### `generic_prediction` - Prediction-focused
```bash
python train.py --config generic_prediction --env duffing
```
- **Loss weights**: Prediction (1.0), others disabled

### `lista` - LISTA Sparse Encoder
```bash
python train.py --config lista --env lotka_volterra --target_size 2048
```
- **Model**: LISTAKM
- **Target size**: 2048 (overcomplete)
- **Encoder**: LISTA with 10 iterations
- **Decoder**: Normalized dictionary
- **Loss weights**: Residual (1.0), Reconstruction (1.0), Sparsity (1.0)

### `lista_nonlinear` - LISTA with MLP
```bash
python train.py --config lista_nonlinear --env lorenz63
```
- **Model**: LISTAKM with nonlinear pre-activation
- **Encoder**: [64, 64, 64] MLP → LISTA

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

## Model Evaluation

The evaluation module (`evaluation.py`) provides comprehensive evaluation of trained Koopman models using multiple rollout strategies and horizon-wise metrics.

### Automatic Evaluation

Evaluation runs automatically at the end of training and saves results to `runs/kae/<timestamp>/evaluation/`:

```
runs/kae/20251106-223912/evaluation/
├── metrics.json                    # Full evaluation metrics
├── duffing/
│   ├── phase_portrait_1000_no_reencode.png
│   ├── phase_portrait_1000_every_step.png
│   ├── phase_portrait_1000_periodic_*.png
│   └── mse_vs_horizon.png
└── lyapunov/
    ├── phase_portrait_comparison.png  # True vs learned system
    └── ...
```

### Rollout Strategies

The evaluation protocol tests three rollout modes:

1. **No reencoding**: Evolves entirely in latent space
2. **Every-step reencoding**: Reencodes at each step
3. **Periodic reencoding**: Reencodes every k steps

### Evaluation Metrics

For each system and rollout mode, the evaluation computes:

- **Horizon-wise MSE**: Mean squared error at specific horizons (100, 1000 steps)
- **MSE curves**: Cumulative MSE vs prediction horizon
- **Phase portraits**: Visual comparison of predicted vs ground truth trajectories
- **Best periodic period**: Automatically selects optimal reencoding period per horizon

### Evaluation Output

The `metrics.json` file contains structured metrics:

```json
{
  "duffing": {
    "modes": {
      "no_reencode": {
        "horizons": {
          "100": {"mean": 0.0012, "std": 0.0003, "num_valid": 100},
          "1000": {"mean": 0.0456, "std": 0.0123, "num_valid": 100}
        },
        "mse_curve": [0.001, 0.002, ...]
      },
      ...
    },
    "best_periodic": {
      "100": {"mode": "periodic_25", "mean": 0.0008},
      "1000": {"mode": "periodic_50", "mean": 0.0234}
    }
  }
}
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

## License

See `LICENSE` file for details.

