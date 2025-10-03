"""Model definitions for LISTA, Koopman Autoencoder (KAE), and Koopman Sparse AE (KSAE)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def soft_threshold(values: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Soft-thresholding used by LISTA."""
    return torch.sign(values) * F.relu(torch.abs(values) - theta)


class LISTA(nn.Module):
    """Learned ISTA encoder with shared shrinkage thresholds."""

    def __init__(
        self,
        dict_dim: int,
        input_dim: int,
        iterations: int = 3,
        shrinkage: str = "soft",
        theta_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if shrinkage != "soft":  # keep implementation focused on the common case
            raise ValueError(f"Unsupported shrinkage '{shrinkage}'")
        self.dict_dim = dict_dim
        self.input_dim = input_dim
        self.iterations = iterations
        self.theta_epsilon = theta_epsilon

        self.We = nn.Parameter(torch.randn(dict_dim, input_dim) * 0.1)
        self.S = nn.Parameter(torch.eye(dict_dim))
        self.theta_raw = nn.Parameter(torch.zeros(dict_dim))

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.theta_raw) + self.theta_epsilon

    def shrink(self, values: torch.Tensor) -> torch.Tensor:
        return soft_threshold(values, self.theta)

    def forward(self, inputs: torch.Tensor, return_sequence: bool = False) -> torch.Tensor | List[torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        b = F.linear(inputs, self.We)  # encoder stage
        z = self.shrink(b)
        iterates = [z]
        for _ in range(1, self.iterations):
            c = b + F.linear(iterates[-1], self.S)
            z = self.shrink(c)
            iterates.append(z)
        if return_sequence:
            return iterates
        return iterates[-1]

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, return_sequence=False)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] | None = None,
    activation: str = "relu",
    final_activation: Optional[str] = None,
) -> nn.Sequential:
    """Construct a feedforward network with optional hidden layers."""
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden in hidden_dims or []:
        layers.append(nn.Linear(last_dim, hidden))
        if activation:
            act_cls = _ACTIVATIONS.get(activation)
            if act_cls is None:
                raise ValueError(f"Unknown activation '{activation}'")
            layers.append(act_cls())
        last_dim = hidden
    layers.append(nn.Linear(last_dim, output_dim))
    if final_activation:
        act_cls = _ACTIVATIONS.get(final_activation)
        if act_cls is None:
            raise ValueError(f"Unknown final activation '{final_activation}'")
        layers.append(act_cls())
    return nn.Sequential(*layers)


class KoopmanAE(nn.Module):
    """Koopman autoencoder with linear latent dynamics."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int] | None = (256, 256),
        decoder_hidden: Sequence[int] | None = None,
        control_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.control_dim = control_dim

        self.encoder = build_mlp(input_dim, latent_dim, hidden_dims=encoder_hidden)
        if decoder_hidden is None:
            decoder_hidden = tuple(reversed(tuple(encoder_hidden))) if encoder_hidden else (256,)
        self.decoder = build_mlp(latent_dim, input_dim, hidden_dims=decoder_hidden, activation="relu")

        self.K = nn.Parameter(torch.eye(latent_dim))
        self.L = nn.Parameter(torch.zeros(latent_dim, control_dim)) if control_dim > 0 else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch, seq_len, feat = x.shape
            encoded = self.encoder(x.view(-1, feat))
            return encoded.view(batch, seq_len, self.latent_dim)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            batch, seq_len, feat = z.shape
            decoded = self.decoder(z.view(-1, feat))
            return decoded.view(batch, seq_len, self.input_dim)
        return self.decoder(z)

    def koopman_step(self, z: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        next_latent = F.linear(z, self.K)
        if self.control_dim > 0 and u is not None and self.L is not None:
            next_latent = next_latent + F.linear(u, self.L)
        return next_latent

    def forward(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("Expected x with shape (batch, seq_len, input_dim)")
        batch, seq_len, _ = x.shape
        encoded = self.encode(x)
        reconstructions = self.decode(encoded)

        if seq_len < 2:
            empty_latents = torch.empty(batch, 0, self.latent_dim, device=x.device)
            empty_preds = torch.empty(batch, 0, self.input_dim, device=x.device)
            return {
                "encoded": encoded,
                "reconstructions": reconstructions,
                "predicted_latents": empty_latents,
                "predictions": empty_preds,
            }

        predicted_latents = []
        predictions = []
        for t in range(seq_len - 1):
            control_t = u[:, t] if u is not None else None
            z_hat = self.koopman_step(encoded[:, t], control_t)
            predicted_latents.append(z_hat)
            predictions.append(self.decode(z_hat))
        predicted_latents_tensor = torch.stack(predicted_latents, dim=1)
        predictions_tensor = torch.stack(predictions, dim=1)

        return {
            "encoded": encoded,
            "reconstructions": reconstructions,
            "predicted_latents": predicted_latents_tensor,
            "predictions": predictions_tensor,
        }

    def rollout(
        self,
        x0: torch.Tensor,
        horizon: int,
        reencode_period: Optional[int] = None,
        controls: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Roll out predictions given the first observation x0."""
        if x0.dim() != 2:
            raise ValueError("x0 must have shape (batch, input_dim)")
        z = self.encoder(x0)
        outputs = []
        for step in range(horizon):
            control_t = None
            if controls is not None and controls.size(1) > step:
                control_t = controls[:, step]
            z = self.koopman_step(z, control_t)
            x_hat = self.decoder(z)
            outputs.append(x_hat)
            if reencode_period and reencode_period > 0 and (step + 1) % reencode_period == 0:
                z = self.encoder(x_hat)
        return torch.stack(outputs, dim=1)


class KSAE(nn.Module):
    """Koopman Sparse Autoencoder using LISTA as the encoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        lista_iterations: int = 3,
        decoder_hidden: Sequence[int] | None = (256,),
        control_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.control_dim = control_dim

        self.encoder = LISTA(dict_dim=latent_dim, input_dim=input_dim, iterations=lista_iterations)
        self.decoder = build_mlp(latent_dim, input_dim, hidden_dims=decoder_hidden or (), activation="relu")
        self.K = nn.Parameter(torch.eye(latent_dim))
        self.L = nn.Parameter(torch.zeros(latent_dim, control_dim)) if control_dim > 0 else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch, seq_len, feat = x.shape
            flattened = x.view(-1, feat)
            encoded = self.encoder.encode(flattened)
            return encoded.view(batch, seq_len, self.latent_dim)
        return self.encoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            batch, seq_len, feat = z.shape
            decoded = self.decoder(z.view(-1, feat))
            return decoded.view(batch, seq_len, self.input_dim)
        return self.decoder(z)

    def koopman_step(self, z: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        next_latent = F.linear(z, self.K)
        if self.control_dim > 0 and u is not None and self.L is not None:
            next_latent = next_latent + F.linear(u, self.L)
        return next_latent

    def forward(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("Expected x with shape (batch, seq_len, input_dim)")
        batch, seq_len, _ = x.shape
        encoded = self.encode(x)
        reconstructions = self.decode(encoded)

        if seq_len < 2:
            empty_latents = torch.empty(batch, 0, self.latent_dim, device=x.device)
            empty_preds = torch.empty(batch, 0, self.input_dim, device=x.device)
            return {
                "encoded": encoded,
                "reconstructions": reconstructions,
                "predicted_latents": empty_latents,
                "predictions": empty_preds,
            }

        predicted_latents = []
        predictions = []
        for t in range(seq_len - 1):
            control_t = u[:, t] if u is not None else None
            z_hat = self.koopman_step(encoded[:, t], control_t)
            predicted_latents.append(z_hat)
            predictions.append(self.decode(z_hat))
        predicted_latents_tensor = torch.stack(predicted_latents, dim=1)
        predictions_tensor = torch.stack(predictions, dim=1)
        return {
            "encoded": encoded,
            "reconstructions": reconstructions,
            "predicted_latents": predicted_latents_tensor,
            "predictions": predictions_tensor,
        }

    def rollout(
        self,
        x0: torch.Tensor,
        horizon: int,
        reencode_period: Optional[int] = None,
        controls: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x0.dim() != 2:
            raise ValueError("x0 must have shape (batch, input_dim)")
        z = self.encoder.encode(x0)
        outputs = []
        for step in range(horizon):
            control_t = None
            if controls is not None and controls.size(1) > step:
                control_t = controls[:, step]
            z = self.koopman_step(z, control_t)
            x_hat = self.decoder(z)
            outputs.append(x_hat)
            if reencode_period and reencode_period > 0 and (step + 1) % reencode_period == 0:
                z = self.encoder.encode(x_hat)
        return torch.stack(outputs, dim=1)
