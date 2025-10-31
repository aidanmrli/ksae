# KSAE

This is a repo in progress.

## Quickstart

```bash
pip install -e .[develop]

# not tested: Train LISTA on synthetic sparse coding data
python main.py train-lista --dict-dim 400 --input-dim 100 --epochs 50

# Train Koopman autoencoder with exact CT latent integration (matrix exponential)
# this does K = exp(A * dt)
# Use SciPy offline ODE cache for data generation; open-loop multi-step loss
python main.py train-kae \
  --system duffing \
  --num-samples 20480 \
  --batch-size 256 \
  --epochs 200 \
  --context-length 1 \
  --sequence-length 2 \
  --dt 0.01 \
  --noise-std 0.0 \
  --latent-dim 64 \
  --encoder-hidden 64 64 \
  --decoder-hidden \
  --lr-main 1e-3 \
  --lr-koopman 1e-3 \
  --lambda-recon 0.5 \
  --lambda-align 1.0 \
  --lambda-pred 0.0 \
  --lambda-sparse 0.01 \
  --seed 123

# Evaluate KoopmanAE with paper-style metrics (automatic PR search)
# Prints MSE@100 and MSE@1000 with/without PR, plus best k ∈ {10,25,50,100}
python main.py eval-kae \
  --checkpoint runs/kae/YOUR_RUN_TIMESTAMP/checkpoint.pt \
  --system duffing \
  --latent-dim 64 \
  --encoder-hidden 64 64 \
  --decoder-hidden \
  --dt 0.01 \
  --koopman-mode continuous \
  --latent-mode ct_matrix_exp \
  --gamma-method auto \
  --use-offline-cache \
  --test-rollout-steps 200 \
  --inference-reencode-period 25 \
  --plot-dir plots \
  --phase-portrait-samples 50 \
  --seed 42

# Discrete version: directly learn discrete-time
# matrix K with update rule z_{t+1} = z_t @ K
python main.py train-kae \
  --system duffing \
  --num-samples 20480 \
  --batch-size 256 \
  --epochs 200 \
  --context-length 1 \
  --sequence-length 2 \
  --dt 0.01 \
  --noise-std 0.0 \
  --latent-dim 64 \
  --encoder-hidden 64 64 \
  --decoder-hidden \
  --koopman-mode discrete \
  --lr-main 1e-3 \
  --lr-koopman 1e-3 \
  --lambda-recon 0.5 \
  --lambda-align 1.0 \
  --lambda-pred 0.0 \
  --lambda-sparse 0.01 \
  --seed 123

python main.py eval-kae \
  --checkpoint runs/kae/YOUR_RUN_TIMESTAMP/checkpoint.pt \
  --system duffing \
  --latent-dim 64 \
  --encoder-hidden 64 64 \
  --decoder-hidden \
  --dt 0.01 \
  --koopman-mode discrete \
  --use-offline-cache \
  --test-rollout-steps 200 \
  --inference-reencode-period 25 \
  --plot-dir plots \
  --phase-portrait-samples 50 \
  --seed 42
```

By default each training command writes an artefact directory under `runs/<model>/<timestamp>/` containing checkpoints, configs, and metric history; evaluation commands print metrics to stdout and optionally emit rollout plots with `--plot-dir`.

Optional knobs of interest:
- `--normalize-decoder-columns/--no-normalize-decoder-columns` (default on): column-normalize decoder to avoid degenerate tiny latents.
- `--koopman-mode {continuous,discrete}` controls whether dynamics are parameterized in CT (A,B) or DT (K,L).
- Latent dynamics (CT only): `--latent-mode {ct_matrix_exp,disc_tustin}`.
  - `ct_matrix_exp` uses exact Φ=exp(AΔt) and Γ integral; pick `--gamma-method {auto,inverse,augmented}`.
  - `disc_tustin` uses bilinear (Tustin) for A; input mapping uses `--control-discretization {tustin,zoh}`.
- Learning rates: `--lr-main` (encoder/decoder), `--lr-koopman` (dynamics), `--lr-lista` (KSAE encoder).
- `--lambda-sparse` adds small L1 (default `1e-3`) on Koopman embeddings (KSAE and KAE).
- Re-encoding: `--inference-reencode-period` (eval) and `--train-reencode-period` (training) support 0 (off), 20, or 50 as beneficial settings.
 - Training windows: `--context-length T` samples windowed minibatches of length T prediction steps (uses T+1 states). Set `--sequence-length S` to the per-trajectory simulated length (e.g., S=500 as in §4.1).
 - Offline data cache (SciPy, ground-truth only): `--use-offline-cache --cache-dir data --ode-rtol 1e-5 --ode-atol 1e-7`.

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
- `python main.py eval-kae --checkpoint ... --test-rollout-steps 1000 --inference-reencode-period 25 --plot-dir plots/kae` measures reconstruction/prediction/rollout MSE and saves phase plots.
- `python main.py eval-ksae ...` reports identical metrics plus latent sparsity.
