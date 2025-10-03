from __future__ import annotations

import os
from pathlib import Path

import torch

from eval import save_phase_portrait, save_rollout_plot


def test_save_phase_portrait_and_rollout_plot(tmp_path: Path) -> None:
    # Create a simple synthetic 2D trajectory: a circle in phase space
    horizon = 50
    t = torch.linspace(0, 2 * torch.pi, steps=horizon + 1)
    x1 = torch.cos(t)
    x2 = torch.sin(t)
    true_sequence = torch.stack([x1, x2], dim=1)  # (H+1, 2)

    # Predicted is the same curve but slightly phase-shifted (toy mismatch)
    tp = torch.linspace(0, 2 * torch.pi, steps=horizon)
    x1p = torch.cos(tp + 0.05)
    x2p = torch.sin(tp + 0.05)
    predicted_sequence = torch.stack([x1p, x2p], dim=1)  # (H, 2)

    # Save phase portrait
    phase_path = tmp_path / "phase_test.png"
    save_phase_portrait(true_sequence, predicted_sequence, phase_path)
    assert phase_path.exists() and phase_path.stat().st_size > 0

    # Save rollout plot
    rollout_path = tmp_path / "rollout_test.png"
    save_rollout_plot(true_sequence, predicted_sequence, horizon, rollout_path)
    assert rollout_path.exists() and rollout_path.stat().st_size > 0


