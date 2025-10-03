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


def compute_koopman_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    weights: KoopmanLossWeights,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Aggregate the individual loss terms for Koopman-style models."""
    components: Dict[str, torch.Tensor] = {}
    encoded = outputs.get("encoded")
    predicted_latents = outputs.get("predicted_latents")
    reconstructions = outputs.get("reconstructions")
    predictions = outputs.get("predictions")

    if weights.reconstruction > 0 and reconstructions is not None:
        components["reconstruction"] = F.mse_loss(reconstructions, batch["x"])

    if (
        weights.alignment > 0
        and predicted_latents is not None
        and encoded is not None
        and predicted_latents.numel() > 0
    ):
        encoded_next = encoded[:, 1:]
        # ensure encoded_next matches predicted_latents shape
        encoded_next = encoded_next[..., : predicted_latents.shape[-1]]
        components["alignment"] = F.mse_loss(predicted_latents, encoded_next)

    if (
        weights.prediction > 0
        and predictions is not None
        and predictions.numel() > 0
    ):
        target = batch["x"][:, 1:]
        components["prediction"] = F.mse_loss(predictions, target)

    if weights.sparsity > 0 and encoded is not None:
        components["sparsity"] = encoded.abs().mean()

    if weights.frobenius > 0 and hasattr(model, "K"):
        components["frobenius"] = torch.norm(model.K, p="fro")

    total = torch.zeros((), device=batch["x"].device)
    for name, value in components.items():
        weight = getattr(weights, name)
        total = total + weight * value
    return total, components
