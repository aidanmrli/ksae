"""Loss functions for LISTA, Koopman AE, and KSAE training objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn import functional as F


def lista_supervised_loss(pred_codes: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
    """Supervised LISTA objective: mean squared error to target sparse codes."""
    return F.mse_loss(pred_codes, target_codes)


def lista_reconstruction_loss(
    signals: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
) -> torch.Tensor:
    """Optional reconstruction loss used when learning a dictionary jointly."""
    # dictionary has shape (batch, input_dim, dict_dim), use first example
    dict_matrix = dictionary[0] if dictionary.dim() == 3 else dictionary
    recon = codes @ dict_matrix.T
    return F.mse_loss(recon, signals)


@dataclass
class KoopmanLossWeights:
    reconstruction: float = 1.0
    alignment: float = 1.0
    prediction: float = 1.0
    sparsity: float = 0.0
    frobenius: float = 0.0


_SHAPES_PRINTED = False


def mean_l2_norm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the mean of L2 norms between predictions and targets."""
    return torch.linalg.norm(pred - target, ord=2, dim=-1).mean()


def compute_koopman_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    weights: KoopmanLossWeights,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Aggregate the individual loss terms for Koopman-style models."""
    components: Dict[str, torch.Tensor] = {}
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    reconstructions = outputs["reconstructions"]
    encoded_all = outputs["encoded"]
    predicted_latents = outputs["predicted_latents"]
    predictions = outputs["predictions"]
    x = batch["x"]

    # 1. Reconstruction loss
    if weights.reconstruction > 0:
        # Compare reconstructed states with original states
        recon_loss = mean_l2_norm_loss(reconstructions, x)
        components["reconstruction"] = recon_loss
        total_loss = total_loss + weights.reconstruction * recon_loss

    # 2. Alignment loss (dynamics in latent space)
    if weights.alignment > 0 and predicted_latents.numel() > 0:
        # Compare predicted latents with encoded future states
        true_next_latents = encoded_all[:, 1:]
        align_loss = mean_l2_norm_loss(predicted_latents, true_next_latents)
        components["alignment"] = align_loss
        total_loss = total_loss + weights.alignment * align_loss

    # 3. Prediction loss (dynamics in data space)
    if weights.prediction > 0 and predictions.numel() > 0:
        # Compare predicted future states with true future states
        true_next_states = x[:, 1:]
        pred_loss = mean_l2_norm_loss(predictions, true_next_states)
        components["prediction"] = pred_loss
        total_loss = total_loss + weights.prediction * pred_loss

    # 4. Sparsity loss (L1 on latent states)
    if weights.sparsity > 0:
        # L1 norm of each latent vector, averaged over batch and time
        sparsity_loss = torch.linalg.norm(encoded_all, ord=1, dim=-1).mean()
        components["sparsity"] = sparsity_loss
        total_loss = total_loss + weights.sparsity * sparsity_loss

    # 5. Frobenius norm loss on Koopman matrix (regularizer)
    if weights.frobenius > 0:
        koopman_matrix = None
        if hasattr(model, "A"):
            koopman_matrix = model.A
        elif hasattr(model, "K"):
            koopman_matrix = model.K
            
        if koopman_matrix is not None:
            frob_loss = torch.norm(koopman_matrix, p="fro")
            components["frobenius"] = frob_loss
            total_loss = total_loss + weights.frobenius * frob_loss

    return total_loss, components
