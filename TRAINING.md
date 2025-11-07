# Training Guide

This guide explains how to train Koopman Autoencoder models using the PyTorch-based training system.

## Quick Start

### 1. Command Line Training

Train a model from the command line:

```bash
# Basic training
python train.py --config generic --env duffing --num_steps 10000

# Sparse model with custom settings
python train.py --config generic_sparse --env pendulum --num_steps 20000 --lr 5e-4

# LISTA sparse model
python train.py --config lista --env lotka_volterra --num_steps 15000 --target_size 1024
```

#### Available Options

- `--config`: Model configuration (`generic`, `generic_sparse`, `generic_prediction`, `lista`, `lista_nonlinear`)
- `--env`: Environment (`duffing`, `pendulum`, `lotka_volterra`, `lorenz63`, `parabolic`)
- `--num_steps`: Number of training steps (default: 20000)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate (overrides config default)
- `--target_size`: Latent dimension (overrides config default)
- `--sparsity_coeff`: Sparsity loss weight (overrides config default, useful for sweeps)
- `--seed`: Random seed (default: 0)
- `--log_dir`: Directory for logs and checkpoints (default: `./runs/kae`)
- `--device`: Device to train on (`cpu`, `cuda`, `mps`)

### 2. Programmatic Training

Use the training API in your own scripts:

```python
from config import get_config
from train import train

# Create config
cfg = get_config("generic_sparse")
cfg.ENV.ENV_NAME = "duffing"
cfg.TRAIN.NUM_STEPS = 10000
cfg.MODEL.TARGET_SIZE = 64
cfg.MODEL.SPARSITY_COEFF = 0.01

# Train model
model = train(cfg, log_dir="./runs/my_experiment", device='cpu')
```

See `example_train.py` for more examples.

### 3. Run Example Experiments

```bash
python example_train.py
```

This will run a quick training experiment to verify everything works.

## Output Files

Each training run creates a timestamped directory with the following files:

```
runs/kae/20251106-223912/
├── config.json              # Configuration used for this run
├── checkpoint.pt            # Best model checkpoint (lowest loss)
├── last.pt                  # Latest model checkpoint
├── final_metrics.json       # Final step metrics
├── metrics_history.jsonl    # Complete time series of all metrics
└── metrics_summary.json     # Summary statistics (min, max, mean, final)
```

### Loading Checkpoints

```python
import torch
from config import Config
from model import make_model

# Load config
cfg = Config.from_json('runs/kae/20251106-223912/config.json')

# Create model
env = make_env(cfg)
model = make_model(cfg, env.observation_size)

# Load weights
checkpoint = torch.load('runs/kae/20251106-223912/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Resuming Training

```bash
python train.py --checkpoint runs/kae/20251106-223912/last.pt
```

## Visualizing Results

### Plot Training Metrics

```bash
# Plot all metrics
python plot_metrics.py runs/kae/20251106-223912

# Plot specific metrics
python plot_metrics.py runs/kae/20251106-223912 --metrics loss residual_loss

# Save plot to file
python plot_metrics.py runs/kae/20251106-223912 --save training_curves.png

# Print summary statistics
python plot_metrics.py runs/kae/20251106-223912 --summary
```

### Custom Analysis

Load metrics in Python for custom analysis:

```python
import json

# Load time series data
metrics = []
with open('runs/kae/20251106-223912/metrics_history.jsonl', 'r') as f:
    for line in f:
        metrics.append(json.loads(line))

# Load summary
with open('runs/kae/20251106-223912/metrics_summary.json', 'r') as f:
    summary = json.load(f)

# Analyze as you like
import pandas as pd
df = pd.DataFrame(metrics)
df_train = df[df['name'].str.startswith('train/')]
```

## Model Configurations

### Generic Koopman Autoencoder

Standard model with MLP encoder and linear decoder:

```bash
python train.py --config generic --env duffing --num_steps 10000
```

- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: 2-layer MLP [64, 64]
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.02)

### Sparse Koopman Autoencoder

Adds L1 sparsity regularization:

```bash
python train.py --config generic_sparse --env pendulum --num_steps 10000
```

- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: 2-layer MLP [64, 64] with ReLU + bias
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.5), Sparsity (0.01)

### LISTA Sparse Koopman

Uses Learned Iterative Soft-Thresholding:

```bash
python train.py --config lista --env lotka_volterra --num_steps 15000
```

- **Model**: LISTAKM
- **Target size**: 2048 (high-dimensional sparse codes)
- **Encoder**: LISTA with 10 iterations
- **Decoder**: Normalized dictionary
- **Loss weights**: Residual (1.0), Reconstruction (1.0), Sparsity (1.0)

## Environment Details

| Environment | Dimension | Dynamics |
|------------|-----------|----------|
| `duffing` | 2D | Duffing oscillator |
| `pendulum` | 2D | Pendulum |
| `lotka_volterra` | 2D | Predator-prey |
| `lorenz63` | 3D | Lorenz attractor |
| `parabolic` | 2D | Parabolic attractor |

## Tips & Best Practices

### Hyperparameter Tuning

1. **Start with generic config**: Establish baseline performance
2. **Adjust target_size**: Start small (16-64), increase if needed
3. **Try sparse regularization**: Sweep sparsity_coeff (0.0001 - 0.1) to find sweet spot
4. **Balance loss weights**: Adjust reconst_coeff vs res_coeff based on goals

**Example: Sparsity Coefficient Sweep**
```bash
# Try different sparsity values
for sparsity in 0.001 0.005 0.01 0.05 0.1; do
    python train.py --config generic_sparse --env duffing \
        --sparsity_coeff $sparsity --num_steps 10000 \
        --log_dir ./runs/sparsity_sweep
done

# Compare results
for dir in ./runs/sparsity_sweep/*/; do
    python plot_metrics.py "$dir" --summary | grep -A 1 "train/sparsity_ratio"
done
```

### Monitoring Training

Watch these metrics:

- **loss**: Should decrease over time
- **residual_loss**: Alignment in latent space (lower = more linear)
- **reconst_loss**: Reconstruction quality
- **sparsity_ratio**: Fraction of zero codes (higher = sparser)
- **eval/mean_error**: Multi-step prediction error

### Debugging

If training is unstable:
- Reduce learning rate (try 1e-4 instead of 1e-3)
- Reduce batch size
- Check for NaN in metrics
- Try simpler environment first (duffing, pendulum)

If reconstruction is poor:
- Increase reconst_coeff
- Add decoder layers
- Reduce sparsity_coeff

If predictions are poor:
- Increase target_size
- Adjust res_coeff
- Try longer training

## Testing

Run tests to verify the training system:

```bash
# All training tests
pytest tests/test_train.py -v

# Specific test
pytest tests/test_train.py::TestTrain::test_train_short_run -v
```

## Future: Weights & Biases Integration

The current logging system can be easily replaced with Weights & Biases:

```python
# In train.py, replace MetricsLogger with:
import wandb

wandb.init(project="koopman-ae", config=cfg.to_dict())

# Replace logger.log_dict() with:
wandb.log(metrics, step=step)
```

This keeps the codebase flexible for when you want cloud-based experiment tracking.

