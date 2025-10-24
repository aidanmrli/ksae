"""Plotting utilities for rollouts and phase portraits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402


def save_rollout_plot(true_sequence: torch.Tensor, predicted_sequence: torch.Tensor, horizon: int, path: Path) -> None:
    """Create a simple plot comparing rollout trajectories."""
    fig, axes = plt.subplots(true_sequence.shape[-1], 1, figsize=(6, 2 * true_sequence.shape[-1]))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    time_axis = range(horizon)
    for dim, ax in enumerate(axes):
        ax.plot(time_axis, true_sequence[1:, dim], label="true")
        ax.plot(time_axis, predicted_sequence[:, dim], label="pred")
        ax.set_ylabel(f"dim {dim}")
    axes[-1].set_xlabel("time step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_phase_portrait(true_sequence: torch.Tensor, predicted_sequence: torch.Tensor, path: Path) -> None:
    """Plot phase portrait (x1 vs x2) comparing true and predicted trajectories.

    Expects:
    - true_sequence: Tensor with shape (horizon + 1, state_dim)
    - predicted_sequence: Tensor with shape (horizon, state_dim)
    The first dimension is time. The predicted sequence aligns with true_sequence[1:].
    Only the first two state dimensions are plotted.
    """
    if true_sequence.dim() != 2 or predicted_sequence.dim() != 2:
        raise ValueError("Sequences must be 2D tensors: (time, state_dim)")
    if true_sequence.size(1) < 2 or predicted_sequence.size(1) < 2:
        # Not enough dimensions for a phase portrait; no-op save to avoid breaking pipelines
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.text(0.5, 0.5, "Phase portrait requires ≥2 dims", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    # Build continuous trajectories in (x1, x2) space
    # - True trajectory includes the initial state x0
    # - Predicted trajectory is prepended with x0 so it starts from the same point
    true_xy = true_sequence[:, :2]
    pred_xy = torch.cat([true_sequence[0:1, :2], predicted_sequence[:, :2]], dim=0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(true_xy[:, 0].numpy(), true_xy[:, 1].numpy(), label="true", color="C0", linewidth=2)
    ax.plot(pred_xy[:, 0].numpy(), pred_xy[:, 1].numpy(), label="pred", color="C1", linewidth=2, linestyle="--")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Phase portrait")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_phase_portraits_overlay(
    true_sequences: list[torch.Tensor],
    predicted_sequences: list[torch.Tensor],
    path: Path,
) -> None:
    """Plot many phase portraits on one figure.

    - true_sequences: list of T+1 by D tensors (includes initial state)
    - predicted_sequences: list of T by D tensors (unrolled predictions)
    Only the first two state dims are used.
    Ground truth trajectories are drawn first in semi-transparent gray.
    Predicted trajectories are drawn on top in a solid color.
    """
    if len(true_sequences) == 0 or len(predicted_sequences) == 0:
        return
    n = min(len(true_sequences), len(predicted_sequences))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    rng = np.random.default_rng()

    # Draw all ground-truth trajectories in the background
    for i in range(n):
        ts = true_sequences[i]
        if ts.dim() != 2 or ts.size(1) < 2:
            continue
        true_xy = ts[:, :2].numpy()
        ax.plot(true_xy[:, 0], true_xy[:, 1], color=(0.2, 0.2, 0.2), alpha=0.15, linewidth=1.0)

    # Draw predicted trajectories on top
    for i in range(n):
        ts = true_sequences[i]
        ps = predicted_sequences[i]
        if ts.dim() != 2 or ps.dim() != 2 or ts.size(1) < 2 or ps.size(1) < 2:
            continue
        pred_xy = torch.cat([ts[0:1, :2], ps[:, :2]], dim=0).numpy()
        # Random vivid color per trajectory via HSV
        h = float(rng.random())
        s = 0.65 + 0.35 * float(rng.random())
        v = 0.85 + 0.15 * float(rng.random())
        rgb = mcolors.hsv_to_rgb([h, s, v])
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color=rgb, alpha=0.9, linewidth=1.2)

    ax.axhline(0.0, color="gray", lw=0.5)
    ax.axvline(0.0, color="gray", lw=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Phase Portraits: predicted (color) vs truth (gray)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


