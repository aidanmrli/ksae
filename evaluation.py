"""Comprehensive evaluation utilities for Koopman Autoencoder models.

This module implements the evaluation protocol described in the research
specification. It supports multiple rollout strategies, computes horizon-wise
mean-squared error metrics, and produces qualitative plots such as phase
portraits and MSE-vs-horizon curves.

Key features
------------
- Rollout generators for:
  * No reencoding (latent-only evolution)
  * Every-step reencoding (state-space evolution)
  * Periodic reencoding with configurable period k
- Evaluation over multiple dynamical systems, horizons, and reencoding periods
- Aggregation of metrics across unseen initial conditions (mean ± std)
- Automatic selection of the best periodic reencoding period per horizon
- Qualitative plots with ground truth trajectories in transparent gray

The public entry point is :func:`evaluate_model`, which returns a nested metrics
dictionary and optionally saves metrics/plots to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import Config
from data import VectorWrapper, generate_trajectory, make_env
from model import KoopmanMachine


# ---------------------------------------------------------------------------
# Rollout generators
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_no_reencode(model: KoopmanMachine, x0: torch.Tensor, horizon: int) -> torch.Tensor:
    """Roll out the Koopman dynamics without reencoding.

    Args:
        model: Trained Koopman machine.
        x0: Initial states with shape ``[batch, state_dim]``.
        horizon: Number of prediction steps.

    Returns:
        Predicted trajectory with shape ``[horizon, batch, state_dim]``.
    """
    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)

    latent = model.encode(x0)
    predictions: List[torch.Tensor] = []

    for _ in range(horizon):
        latent = model.step_latent(latent)
        x_pred = model.decode(latent)
        predictions.append(x_pred)

        if not torch.isfinite(x_pred).all():
            # Mark remaining steps as NaN to signal explosion
            nan_frame = torch.full_like(x_pred, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

    return torch.stack(predictions, dim=0)


@torch.no_grad()
def rollout_every_step_reencode(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Roll out the Koopman dynamics with reencoding at every step."""

    model.eval()
    device = next(model.parameters()).device
    state = x0.to(device)
    predictions: List[torch.Tensor] = []

    for _ in range(horizon):
        state = model.step_env(state)
        predictions.append(state)

        if not torch.isfinite(state).all():
            nan_frame = torch.full_like(state, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

    return torch.stack(predictions, dim=0)


@torch.no_grad()
def rollout_periodic_reencode(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
    period: int,
) -> torch.Tensor:
    """Roll out the Koopman dynamics with periodic reencoding every *period* steps."""

    if period <= 0:
        raise ValueError("period must be a positive integer")

    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)

    latent = model.encode(x0)
    predictions: List[torch.Tensor] = []

    for step in range(horizon):
        latent = model.step_latent(latent)
        x_pred = model.decode(latent)
        predictions.append(x_pred)

        if not torch.isfinite(x_pred).all():
            nan_frame = torch.full_like(x_pred, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

        if (step + 1) % period == 0:
            latent = model.encode(x_pred)

    return torch.stack(predictions, dim=0)


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _compute_horizon_mse(
    squared_errors: torch.Tensor,
    horizon: int,
) -> Tuple[float, float, List[float], int]:
    """Compute mean ± std MSE for a specific horizon.

    Args:
        squared_errors: Tensor ``[time, batch]`` with per-step squared L2 norms.
        horizon: Horizon length (<= time dimension of squared_errors).

    Returns:
        Tuple ``(mean, std, per_ic, num_valid)`` where *per_ic* is a list of the
        per-initial-condition MSE values used for aggregation.
    """

    horizon = min(horizon, squared_errors.size(0))
    horizon_errors = squared_errors[:horizon]

    # Average over time, ignoring NaNs (exploding rollouts)
    per_ic = torch.nanmean(horizon_errors, dim=0)
    valid_mask = torch.isfinite(per_ic)

    if valid_mask.sum() == 0:
        return float("nan"), float("nan"), [], 0

    valid_errors = per_ic[valid_mask]
    mean = valid_errors.mean().item()
    std = valid_errors.std(unbiased=False).item() if valid_errors.numel() > 1 else 0.0
    return mean, std, valid_errors.tolist(), int(valid_mask.sum().item())


def _cumulative_mse_curve(squared_errors: torch.Tensor) -> List[float]:
    """Compute cumulative MSE curve averaged across initial conditions."""

    time_steps = squared_errors.size(0)
    steps = torch.arange(1, time_steps + 1, dtype=torch.float32, device=squared_errors.device)
    cumulative = torch.cumsum(squared_errors, dim=0)
    with torch.no_grad():
        curve = torch.nanmean(cumulative / steps.view(-1, 1), dim=1)
    return curve.cpu().tolist()


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _ensure_matplotlib():
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  # pylint: disable=unused-import


def _save_phase_portrait_overlay(
    true_sequences: torch.Tensor,
    predicted_sequences: Dict[str, torch.Tensor],
    path: Path,
    max_samples: int = 20,
) -> None:
    """Save a phase portrait overlay plot.

    Args:
        true_sequences: Tensor with shape ``[batch, time + 1, state_dim]``.
        predicted_sequences: Mapping from mode name to tensor with shape
            ``[batch, time, state_dim]``.
        path: Output path for the PNG file.
        max_samples: Maximum number of trajectories to render.
    """

    if true_sequences.size(-1) < 2:
        return  # Phase portrait not meaningful

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    path.parent.mkdir(parents=True, exist_ok=True)

    batch = true_sequences.size(0)
    indices = torch.arange(batch)

    # Filter trajectories with finite predictions for all modes
    finite_mask = torch.ones(batch, dtype=torch.bool)
    true_xy = true_sequences[:, :, :2]

    for preds in predicted_sequences.values():
        flat = preds.reshape(preds.size(0), -1)
        finite_mask &= torch.isfinite(flat).all(dim=1)

    indices = indices[finite_mask]
    if indices.numel() == 0:
        return

    indices = indices[:max_samples]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plot ground truth trajectories in light gray
    for idx in indices.tolist():
        gt = true_xy[idx].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color=(0.5, 0.5, 0.5), alpha=0.25, linewidth=1.5)

    rng = np.random.default_rng(42)
    colors = {
        mode: mcolors.hsv_to_rgb([float(rng.random()), 0.65 + 0.3 * float(rng.random()), 0.9])
        for mode in predicted_sequences.keys()
    }

    for mode, preds in predicted_sequences.items():
        color = colors[mode]
        for idx in indices.tolist():
            pred_xy = torch.cat([
                true_xy[idx, :1],
                preds[idx, :, :2],
            ], dim=0).cpu().numpy()
            ax.plot(
                pred_xy[:, 0],
                pred_xy[:, 1],
                color=color,
                alpha=0.9,
                linewidth=1.2,
                label=mode if idx == indices[0].item() else None,
            )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Phase portrait (1000-step)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_phase_portrait_single_mode(
    true_sequences: torch.Tensor,
    predicted: torch.Tensor,
    path: Path,
    max_samples: int = 20,
    title: Optional[str] = None,
) -> None:
    """Save a phase portrait for a single rollout mode, coloring each trajectory.

    Args:
        true_sequences: Tensor with shape ``[batch, time + 1, state_dim]``.
        predicted: Tensor with shape ``[batch, time, state_dim]`` for the mode.
        path: Output PNG path.
        max_samples: Maximum number of trajectories to render.
        title: Optional plot title.
    """

    if true_sequences.size(-1) < 2:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    path.parent.mkdir(parents=True, exist_ok=True)

    batch = true_sequences.size(0)
    true_xy = true_sequences[:, :, :2]

    # Keep only trajectories with finite predictions
    flat = predicted.reshape(predicted.size(0), -1)
    finite_mask = torch.isfinite(flat).all(dim=1)
    indices = torch.arange(batch)[finite_mask]
    if indices.numel() == 0:
        return
    indices = indices[:max_samples]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Colormap per-trajectory
    cmap = cm.get_cmap("tab20", indices.numel())

    for j, idx in enumerate(indices.tolist()):
        # ground truth in light gray
        gt = true_xy[idx].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color=(0.6, 0.6, 0.6), alpha=0.35, linewidth=1.2)

        # predicted
        pred_xy = torch.cat([true_xy[idx, :1], predicted[idx, :, :2]], dim=0).cpu().numpy()
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color=cmap(j), linewidth=1.5)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Phase portrait (single mode)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _save_mse_curve_plot(curves: Dict[str, List[float]], path: Path, highlight_horizons: Sequence[int]) -> None:
    """Save MSE vs horizon curves for each rollout mode."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for mode, curve in curves.items():
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, linewidth=2, label=mode)

    for horizon in highlight_horizons:
        ax.axvline(horizon, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Mean MSE")
    ax.set_title("MSE vs horizon")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


@dataclass
class EvaluationSettings:
    """Container for evaluation hyper-parameters."""

    systems: Sequence[str] = (
        # "pendulum",
        "duffing",
        # "lotka_volterra",
        # "lorenz63",
        # "parabolic",
        "lyapunov",
    )
    horizons: Sequence[int] = (100, 1000)
    periodic_reencode_periods: Sequence[int] = (10, 25, 50, 100)
    batch_size: int = 100
    phase_portrait_samples: int = 20
    seed_offset: int = 12345


def evaluate_model(
    model: KoopmanMachine,
    cfg: Config,
    device: torch.device | str = "cuda",
    settings: Optional[EvaluationSettings] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Evaluate a trained Koopman model using the standardized protocol.

    Args:
        model: Trained Koopman machine.
        cfg: Configuration used during training (provides baseline hyper-params).
        device: Device for model inference.
        settings: Optional evaluation settings. Defaults to the research spec.
        output_dir: Optional path to save metrics and plots. When provided, the
            function writes ``metrics.json``, phase portraits, and MSE curves.

    Returns:
        Nested dictionary with metrics for each system and rollout mode.
    """

    if settings is None:
        settings = EvaluationSettings()

    model = model.to(device)
    model.eval()

    max_horizon = max(settings.horizons)
    results: Dict[str, Dict] = {}

    for system in settings.systems:
        eval_cfg = Config.from_dict(cfg.to_dict())
        eval_cfg.ENV.ENV_NAME = system

        env = make_env(eval_cfg)
        if env.observation_size != model.observation_size:
            # Skip incompatible systems to avoid runtime errors
            continue

        vec_env = VectorWrapper(env, settings.batch_size)
        rng = torch.Generator().manual_seed(cfg.SEED + settings.seed_offset)
        init_states = vec_env.reset(rng)  # CPU tensor

        # Generate ground truth trajectories (time-major)
        true_future = generate_trajectory(vec_env.step, init_states, length=max_horizon)
        true_full = torch.cat([init_states.unsqueeze(0), true_future], dim=0)

        # Prepare initial states on device for model rollout
        init_states_device = init_states.to(device)

        predictions: Dict[str, torch.Tensor] = {}
        predictions["no_reencode"] = rollout_no_reencode(model, init_states_device, max_horizon)
        predictions["every_step"] = rollout_every_step_reencode(model, init_states_device, max_horizon)

        for period in settings.periodic_reencode_periods:
            mode_name = f"periodic_{period}"
            predictions[mode_name] = rollout_periodic_reencode(
                model,
                init_states_device,
                max_horizon,
                period=period,
            )

        mode_metrics: Dict[str, Dict] = {}
        periodic_summary: Dict[str, Dict[str, float]] = {str(h): {} for h in settings.horizons}

        # Convert ground truth to match predictions for metric computation
        true_future_cpu = true_future.float()

        for mode_name, pred in predictions.items():
            pred_cpu = pred.detach().cpu().float()
            squared_diff = torch.sum((pred_cpu - true_future_cpu) ** 2, dim=-1)
            squared_diff = torch.where(torch.isfinite(squared_diff), squared_diff, torch.nan)

            horizons_metrics = {}
            for horizon in settings.horizons:
                if system == "parabolic" and horizon > 100:
                    # Skip 1000-step metric for parabolic attractor
                    continue

                mean, std, per_ic, num_valid = _compute_horizon_mse(squared_diff, horizon)
                horizons_metrics[str(horizon)] = {
                    "mean": mean,
                    "std": std,
                    "num_valid": num_valid,
                    "values": per_ic,
                }

                if mode_name.startswith("periodic_") and num_valid > 0:
                    periodic_summary[str(horizon)][mode_name] = mean

            mode_metrics[mode_name] = {
                "horizons": horizons_metrics,
                "mse_curve": _cumulative_mse_curve(squared_diff),
            }

        # Determine best periodic reencoding period per horizon
        best_periodic: Dict[str, Dict[str, float]] = {}
        for horizon in settings.horizons:
            horizon_key = str(horizon)
            if system == "parabolic" and horizon > 100:
                continue

            candidates = periodic_summary[horizon_key]
            if not candidates:
                continue

            best_mode = min(candidates.items(), key=lambda item: item[1])
            best_periodic[horizon_key] = {
                "mode": best_mode[0],
                "mean": best_mode[1],
            }

        # Save qualitative plots when requested
        files: Dict[str, str] = {}
        if output_dir is not None:
            system_dir = output_dir / system
            system_dir.mkdir(parents=True, exist_ok=True)

            # Phase portraits: separate images for each mode at horizon 1000
            if max(settings.horizons) >= 1000 and true_full.size(0) >= 1001:
                horizon_key = "1000"
                true_sequences = true_full[:1001].permute(1, 0, 2)  # [batch, time+1, dim]

                # 1. No reencoding
                predicted_no_re = predictions["no_reencode"][:1000].permute(1, 0, 2).cpu()
                portrait_path_no = system_dir / f"phase_portrait_{horizon_key}_no_reencode.png"
                _save_phase_portrait_single_mode(
                    true_sequences=true_sequences.cpu(),
                    predicted=predicted_no_re,
                    path=portrait_path_no,
                    max_samples=settings.phase_portrait_samples,
                    title=f"Phase portrait ({horizon_key}-step) - no_reencode",
                )
                files["phase_portrait_1000_no_reencode"] = str(portrait_path_no)

                # 2. Every step reencoding
                predicted_every = predictions["every_step"][:1000].permute(1, 0, 2).cpu()
                portrait_path_every = system_dir / f"phase_portrait_{horizon_key}_every_step.png"
                _save_phase_portrait_single_mode(
                    true_sequences=true_sequences.cpu(),
                    predicted=predicted_every,
                    path=portrait_path_every,
                    max_samples=settings.phase_portrait_samples,
                    title=f"Phase portrait ({horizon_key}-step) - every_step",
                )
                files["phase_portrait_1000_every_step"] = str(portrait_path_every)

                # 3. Best periodic reencoding
                best_mode_name = best_periodic.get(horizon_key, {}).get("mode")
                if best_mode_name:
                    predicted_best = predictions[best_mode_name][:1000].permute(1, 0, 2).cpu()
                    portrait_path_periodic = system_dir / f"phase_portrait_{horizon_key}_{best_mode_name}.png"
                    _save_phase_portrait_single_mode(
                        true_sequences=true_sequences.cpu(),
                        predicted=predicted_best,
                        path=portrait_path_periodic,
                        max_samples=settings.phase_portrait_samples,
                        title=f"Phase portrait ({horizon_key}-step) - {best_mode_name}",
                    )
                    files["phase_portrait_1000_periodic"] = str(portrait_path_periodic)

            curves = {
                mode: data["mse_curve"]
                for mode, data in mode_metrics.items()
            }
            curve_path = system_dir / "mse_vs_horizon.png"
            _save_mse_curve_plot(curves, curve_path, settings.horizons)
            files["mse_curve"] = str(curve_path)

        results[system] = {
            "modes": mode_metrics,
            "best_periodic": best_periodic,
            "files": files,
        }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(results, f, indent=2)
        results["metrics_file"] = str(metrics_path)

    return results


__all__ = [
    "EvaluationSettings",
    "evaluate_model",
    "rollout_every_step_reencode",
    "rollout_no_reencode",
    "rollout_periodic_reencode",
]


