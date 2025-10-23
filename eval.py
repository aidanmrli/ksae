"""Evaluation utilities, rollouts, and lightweight sanity checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import KSAE, LISTA, KoopmanAE


@torch.no_grad()
def evaluate_lista(model: LISTA, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    code_mse = 0.0
    recon_mse = 0.0
    sparsity = 0.0
    count = 0
    for batch in loader:
        x = batch["x"].to(device)
        z_star = batch["z_star"].to(device)
        dictionary = batch["dictionary"].to(device)
        pred = model(x)
        batch_size = x.size(0)
        code_mse += torch.mean((pred - z_star) ** 2).item() * batch_size
        # dictionary has shape (batch, input_dim, dict_dim), use first example
        dict_matrix = dictionary[0] if dictionary.dim() == 3 else dictionary
        recon = pred @ dict_matrix.T
        recon_mse += torch.mean((recon - x) ** 2).item() * batch_size
        sparsity += (pred.abs() < 1e-3).float().mean().item() * batch_size
        count += batch_size
    if count == 0:
        return {"code_mse": 0.0, "reconstruction_mse": 0.0, "sparsity": 0.0}
    code_mse /= count
    recon_mse /= count
    sparsity /= count
    return {
        "code_mse": code_mse,
        "reconstruction_mse": recon_mse,
        "psnr": compute_psnr(recon_mse),
        "sparsity": sparsity,
    }


@torch.no_grad()
def evaluate_koopman(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    rollout_horizon: int,
    reencode_period: Optional[int],
    plot_dir: Optional[Path] = None,
    max_plots: int = 0,
) -> Dict[str, float]:
    model.eval()
    recon_mse = 0.0
    pred_mse = 0.0
    align_mse = 0.0
    rollout_mse = 0.0
    sparsity = 0.0
    count = 0
    rollout_count = 0
    plots_saved = 0

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        outputs = model(batch["x"], batch.get("u"))
        batch_size = batch["x"].size(0)
        count += batch_size

        reconstructions = outputs["reconstructions"]
        recon_mse += torch.mean((reconstructions - batch["x"]) ** 2).item() * batch_size

        if outputs["predictions"].numel() > 0:
            target = batch["x"][:, 1: outputs["predictions"].shape[1] + 1]
            pred_mse += torch.mean((outputs["predictions"] - target) ** 2).item() * batch_size

        if outputs["predicted_latents"].numel() > 0:
            encoded_next = outputs["encoded"][:, 1: outputs["predicted_latents"].shape[1] + 1]
            align_mse += torch.mean((outputs["predicted_latents"] - encoded_next) ** 2).item() * batch_size

        if isinstance(model, (KSAE,)):
            sparsity += (outputs["encoded"].abs() < 1e-3).float().mean().item() * batch_size

        if rollout_horizon > 0:
            horizon = min(rollout_horizon, batch["x"].shape[1] - 1)
            if horizon > 0:
                controls = batch.get("u")
                if controls is not None:
                    controls = controls[:, :horizon]
                rollout = model.rollout(batch["x"][:, 0], horizon, reencode_period=reencode_period, controls=controls)
                target = batch["x"][:, 1 : horizon + 1]
                rollout_mse += torch.mean((rollout - target) ** 2).item() * batch_size
                rollout_count += batch_size

                if plot_dir is not None and plots_saved < max_plots:
                    save_rollout_plot(
                        batch["x"][0, : horizon + 1].cpu(),
                        rollout[0].cpu(),
                        horizon,
                        plot_dir / f"rollout_{batch_idx:03d}.png",
                    )
                    plots_saved += 1

                    # Save phase portrait for 2D+ systems (use first two dims)
                    if batch["x"].shape[-1] >= 2 and plots_saved <= max_plots:
                        save_phase_portrait(
                            batch["x"][0, : horizon + 1].cpu(),
                            rollout[0].cpu(),
                            plot_dir / f"phase_{batch_idx:03d}.png",
                        )
                        # keep plots_saved aligned with the cap for total saved artifacts
                        plots_saved += 0  # no increment to let rollout+phase count as one sample

    if count == 0:
        return {
            "reconstruction_mse": 0.0,
            "prediction_mse": 0.0,
            "alignment_mse": 0.0,
            "rollout_mse": 0.0,
            "psnr": 0.0,
            "sparsity": 0.0,
        }

    recon_mse /= count
    pred_mse = pred_mse / count if pred_mse else 0.0
    align_mse = align_mse / count if align_mse else 0.0
    rollout_mse = rollout_mse / rollout_count if rollout_count else 0.0
    result = {
        "reconstruction_mse": recon_mse,
        "prediction_mse": pred_mse,
        "alignment_mse": align_mse,
        "rollout_mse": rollout_mse,
        "psnr": compute_psnr(pred_mse if pred_mse > 0 else recon_mse),
    }
    if isinstance(model, KSAE):
        result["sparsity"] = sparsity / count if count else 0.0
    return result


def compute_psnr(mse: float, peak: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


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
    fig.savefig(path)
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
    fig.savefig(path)
    plt.close(fig)


def lista_shrinkage_sanity() -> None:
    model = LISTA(dict_dim=4, input_dim=4, iterations=1)
    with torch.no_grad():
        model.theta_raw.fill_(math.log(math.e - 1))  # theta ~1.0
    x = torch.tensor([[2.0, -0.5, 0.2, -3.0]])
    z = model(x)
    assert torch.allclose(z[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.all(z.abs() <= torch.tensor([2.0, 0.5, 0.2, 3.0])).item()


def koopman_identity_sanity() -> None:
    model = KoopmanAE(input_dim=2, latent_dim=2, encoder_hidden=(), decoder_hidden=())
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.state_encoder[0].weight.copy_(torch.eye(2))
        model.state_decoder[0].weight.copy_(torch.eye(2))
        model.K.copy_(torch.eye(2))
    x = torch.randn(2, 5, 2)
    outputs = model(x)
    assert torch.allclose(outputs["reconstructions"], x, atol=1e-5)
    assert outputs["predictions"].shape[1] == 4
    assert torch.allclose(outputs["predictions"], x[:, 1:], atol=1e-5)


if __name__ == "__main__":
    lista_shrinkage_sanity()
    koopman_identity_sanity()
    print("Sanity checks passed.")
