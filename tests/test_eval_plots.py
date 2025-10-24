from __future__ import annotations

from pathlib import Path

import torch

from plotting import save_phase_portrait, save_rollout_plot, save_phase_portraits_overlay
from models import KoopmanAE


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


def test_tustin_discretization_step_runs() -> None:
    # Simple smoke test: forward and rollout work under continuous-time + Tustin
    model = KoopmanAE(input_dim=2, latent_dim=2, encoder_hidden=(), decoder_hidden=(), koopman_continuous=True, dt=0.05)
    x = torch.randn(3, 6, 2)
    out = model(x)
    assert out["predictions"].shape == (3, 5, 2)
    roll = model.rollout(x[:, 0], horizon=5, reencode_period=2)
    assert roll.shape == (3, 5, 2)


def test_phase_portraits_overlay_saves(tmp_path: Path) -> None:
    # Create a few short synthetic trajectories
    n = 5
    horizon = 30
    true_list = []
    pred_list = []
    for i in range(n):
        t = torch.linspace(0, 2 * torch.pi, steps=horizon + 1)
        # Slightly different radius per traj
        r = 1.0 + 0.05 * i
        x1 = r * torch.cos(t)
        x2 = r * torch.sin(t)
        true_sequence = torch.stack([x1, x2], dim=1)
        # Predicted: small phase shift
        tp = torch.linspace(0, 2 * torch.pi, steps=horizon)
        x1p = r * torch.cos(tp + 0.03 * i)
        x2p = r * torch.sin(tp + 0.03 * i)
        pred_sequence = torch.stack([x1p, x2p], dim=1)
        true_list.append(true_sequence)
        pred_list.append(pred_sequence)

    out_path = tmp_path / "overlay.png"
    save_phase_portraits_overlay(true_list, pred_list, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0

