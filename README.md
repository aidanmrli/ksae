# Koopman Sparse Autoencoder (KSAE)

This repo is a compact research sandbox for exploring fast sparse latent representations via LISTA, Koopman autoencoders (KAE), and their combination into a Koopman Sparse Autoencoder (KSAE) with periodic reencoding for long-horizon stability.

## Quickstart

```bash
pip install -e .[develop]

# Train LISTA on synthetic sparse coding data
python main.py train-lista --dict-dim 400 --input-dim 100 --epochs 50

# Train Koopman autoencoder on Duffing with continuous-time Koopman + Tustin, and conservative LRs
python main.py train-kae \
  --system duffing \
  --num-samples 50 \
  --sequence-length 500 \
  --dt 0.01 \
  --latent-dim 128 \
  --encoder-hidden 256 256 256 \
  --decoder-hidden \
  --koopman-mode continuous \
  --control-discretization tustin \
  --lambda-align 1.0 \
  --lambda-recon 1.0 \
  --lambda-pred 1.0 \
  --lambda-sparse 1e-3 \
  --lr-main 1e-4 \
  --lr-koopman 1e-5 \
  --weight-decay 1e-4 \
  --inference-reencode-period 20 \
  --context-length 10 

# Train Koopman sparse autoencoder with LISTA encoder
python main.py train-ksae \
  --system duffing \
  --latent-dim 256 \
  --lista-T 3 \
  --lambda-sparse 1e-3 \
  --koopman-mode continuous \
  --control-discretization tustin \
  --lr-main 1e-4 \
  --lr-koopman 1e-5 \
  --lr-lista 1e-4 \
  --inference-reencode-period 20

# Evaluate KSAE rollouts with a fixed reencoding period (saves plots if provided)
python main.py eval-ksae --checkpoint runs/ksae/<run>/checkpoint.pt --rollout 1000 --inference-reencode-period 25 --plot-dir plots/ksae/<run> --max-plots 100

# Evaluate KoopmanAE with paper-style metrics (automatic PR search)
# Defaults for dynamical systems: dt=0.01, sequence-length=1001, num-samples=50, rollout=1000
# Prints MSE@100 and MSE@1000 with/without PR, plus best k ∈ {10,25,50,100}
python main.py eval-kae \
  --checkpoint runs/kae/20251023-191828/checkpoint.pt \
  --system duffing \
  --latent-dim 128 \
  --encoder-hidden 256 256 256 \
  --decoder-hidden \
  --seed 123
```

By default each training command writes an artefact directory under `runs/<model>/<timestamp>/` containing checkpoints, configs, and metric history; evaluation commands print metrics to stdout and optionally emit rollout plots with `--plot-dir`.

Optional knobs of interest:
- `--normalize-decoder-columns/--no-normalize-decoder-columns` (default on): column-normalize decoder to avoid degenerate tiny latents.
- `--koopman-mode {continuous,discrete}` with `--control-discretization {tustin,zoh}` for dynamics parameterization.
- Learning rates: `--lr-main` (encoder/decoder), `--lr-koopman` (dynamics), `--lr-lista` (KSAE encoder).
- `--lambda-sparse` adds small L1 (default `1e-3`) on Koopman embeddings (KSAE and KAE).
- Re-encoding: `--inference-reencode-period` (eval) and `--train-reencode-period` (training) support 0 (off), 20, or 50 as beneficial settings.
 - Training windows: `--context-length T` samples windowed minibatches of length T prediction steps (uses T+1 states). Set `--sequence-length S` to the per-trajectory simulated length (e.g., S=500 as in §4.1).

## Repository Layout

- `main.py` – CLI entrypoint (train/eval/plot)
- `models.py` – LISTA, Koopman AE, and KSAE modules
- `losses.py` – reusable loss terms for Koopman training
- `data.py` – synthetic sparse coding data + classic nonlinear dynamical systems
- `train.py` – training loops with AdamW, gradient clipping, phase scheduling
- `eval.py` – evaluation metrics, rollouts, plotting, sanity checks
- `utils.py` – logging, seeding, JSON helpers
- `pyproject.toml` – minimal dependencies (`torch`, `numpy`, `scipy`, `matplotlib`, `tqdm`)

## Default Hyperparameters

| Component | Key defaults |
|-----------|--------------|
| LISTA | `lr=1e-3`, `T=3`, supervised MSE to cached codes |
| KAE | `encoder_hidden=(256,256)`, `lr_main=1e-3`, `lr_koopman=1e-4`, teacher-forced prediction & alignment losses |
| KSAE | LISTA encoder (`T=3`), sparse penalty `λ_sparse=1e-3`, optional LISTA freeze for first 20 epochs |
| Rollouts | Periodic reencode every 0 (off), 20, or 50 steps depending on CLI flags |

Adjust hyperparameters directly on the CLI; each run stores a JSON snapshot of the parsed arguments for reproducibility.

## Evaluation & Plots

- `python main.py eval-lista --checkpoint ...` prints code MSE, reconstruction MSE, PSNR, and sparsity vs. target codes.
- `python main.py eval-kae --checkpoint ... --rollout 1000 --inference-reencode-period 25 --plot-dir plots/kae` measures reconstruction/prediction/rollout MSE and saves phase plots.
- `python main.py eval-ksae ...` reports identical metrics plus latent sparsity.

## Sanity Checks & Light Tests

Run built-in smoke tests (no external framework required):

```bash
python -m compileall main.py train.py eval.py models.py data.py losses.py utils.py
python eval.py  # checks LISTA shrinkage and Koopman identity behaviour
```

These cover shrinkage correctness, Koopman identity rollouts, and ensure the modules import cleanly.

## Notes

- Dynamics datasets include Pendulum, Duffing, Lotka–Volterra, Lorenz63, and a parabolic attractor simulated with RK4 integration.
- Periodic reencoding can be toggled separately for training (`--train-reencode-period`) and evaluation (`--inference-reencode-period` on eval commands).
- Outputs include `metrics.json` (history of scalar metrics) alongside the best checkpoint to streamline offline analysis.
