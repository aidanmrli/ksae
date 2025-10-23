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


def compute_koopman_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    weights: KoopmanLossWeights,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Aggregate the individual loss terms for Koopman-style models.
    
    Implements the three core losses from the paper (Eq. 7):
    1. Alignment: L_Align = Σ ||ẑ_{t+i} - φ(x_{t+i})||²
    2. Reconstruction: L_Reconst = Σ ||x_{t+i} - ψ(z_{t+i})||²
    3. Prediction: L_Pred = Σ ||x_{t+i} - ψ(ẑ_{t+i})||²
    
    where ẑ denotes latents obtained by advancing with Koopman dynamics.
    """
    global _SHAPES_PRINTED
    
    components: Dict[str, torch.Tensor] = {}
    encoded = outputs.get("encoded")
    predicted_latents = outputs.get("predicted_latents")
    reconstructions = outputs.get("reconstructions")
    predictions = outputs.get("predictions")

    # ===== Alignment Loss: L_Align =====
    # Compare predicted latents (from Koopman dynamics) with re-encoded future observations
    # Prevents K, L from drifting away from what φ produces
    if (
        weights.alignment > 0
        and predicted_latents is not None
        and encoded is not None
        and predicted_latents.numel() > 0
    ):
        encoded_next = encoded[:, 1:]
        # print(f"encoded_next:      {encoded_next[:2, :1, :2]}")
        # print(f"predicted_latents: {predicted_latents[:2, :1, :2]}")
        
        if not _SHAPES_PRINTED:
            print("\n=== SHAPE SANITY CHECK ===")
            print(f"predicted_latents: {predicted_latents.shape}")
            print(f"encoded_next:      {encoded_next.shape}")
            _SHAPES_PRINTED = True
        
        # L2 norm at each timestep, sum over time, mean over batch
        diff_sq = (predicted_latents - encoded_next).pow(2).sum(dim=2)  # (batch, time)
        components["alignment"] = diff_sq.sqrt().sum(dim=1).mean()

    # ===== Reconstruction Loss: L_Reconst =====
    # Decode encoded latents and compare with original observations
    # Ensures ψ ∘ φ forms a proper autoencoder
    if weights.reconstruction > 0 and reconstructions is not None:
        if not _SHAPES_PRINTED:
            print(f"reconstructions:   {reconstructions.shape}")
            print(f"batch['x']:        {batch['x'].shape}")
            _SHAPES_PRINTED = True
        
        # L2 norm at each timestep, sum over time, mean over batch
        diff_sq = (reconstructions - batch["x"]).pow(2).sum(dim=2)  # (batch, time)
        components["reconstruction"] = diff_sq.sqrt().sum(dim=1).mean()

    # ===== Prediction Loss: L_Pred =====
    # Decode predicted latents and compare with future observations
    # Makes decoded rollouts match future states when advancing in latent space
    if weights.prediction > 0 and predictions is not None and predictions.numel() > 0:
        target = batch["x"][:, 1: predictions.shape[1] + 1]
        
        if not _SHAPES_PRINTED:
            print(f"predictions:       {predictions.shape}")
            print(f"target:            {target.shape}")
            print("========================\n")
            _SHAPES_PRINTED = True
        
        # L2 norm at each timestep, sum over time, mean over batch
        diff_sq = (predictions - target).pow(2).sum(dim=2)  # (batch, time)
        components["prediction"] = diff_sq.sqrt().sum(dim=1).mean()

    # ===== Auxiliary Regularization Terms =====
    if weights.sparsity > 0 and encoded is not None:
        components["sparsity"] = encoded.abs().mean()

    if weights.frobenius > 0:
        if hasattr(model, "A"):
            components["frobenius"] = torch.norm(model.A, p="fro")
        elif hasattr(model, "K"):
            components["frobenius"] = torch.norm(model.K, p="fro")

    # ===== Aggregate Weighted Loss =====
    total = (
        weights.alignment * components.get("alignment", 0.0)
        + weights.reconstruction * components.get("reconstruction", 0.0)
        + weights.prediction * components.get("prediction", 0.0)
        + weights.sparsity * components.get("sparsity", 0.0)
        + weights.frobenius * components.get("frobenius", 0.0)
    )
    return total, components
