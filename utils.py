"""Utility helpers for experiment management and reproducibility."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    """Return a torch.device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if it does not already exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Persist a JSON file with indentation for readability."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


@dataclass
class AverageMeter:
    """Track the running average of scalar values."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple logging setup for command line usage."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a mapping of tensors to the desired device."""
    return {key: value.to(device) for key, value in batch.items()}


def flatten_dict(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested metric dictionaries using slash-delimited keys."""
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=f"{full_key}/"))
        else:
            flat[full_key] = value
    return flat


# ---------------------------------------------------------------------------
# ODE discretization utilities for linear continuous-time systems
# ---------------------------------------------------------------------------


def make_time_grid(num_steps: int, dt: float, start_time: float = 0.0) -> torch.Tensor:
    """Create a 1D time grid tensor of length num_steps+1 starting at start_time.

    The grid corresponds to [t0, t0+dt, ..., t0+num_steps*dt].
    """
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    # Use float32 for broad compatibility; consumers can .to(dtype) as needed
    times = torch.arange(0, num_steps + 1, dtype=torch.float32) * float(dt) + float(start_time)
    return times


def _compute_gamma_via_inverse(
    A: torch.Tensor,
    Phi: torch.Tensor,
    B: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """Compute Γ = A^{-1}(Φ - I)B when A is well-conditioned.

    Returns None if a numerical issue is detected (ill-conditioned A).
    """
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    # Heuristic conditioning check: use smallest singular value threshold
    try:
        svals = torch.linalg.svdvals(A)
        min_sv = torch.min(svals)
        max_sv = torch.max(svals)
        # If A is near-singular or extremely ill-conditioned, fall back
        if not torch.isfinite(min_sv) or min_sv < eps or (torch.isfinite(max_sv) and (max_sv / min_sv) > 1e6):
            return None
    except RuntimeError:
        # SVD may fail on some devices/dtypes; fall back to augmented expm
        return None
    rhs = (Phi - I) @ B
    try:
        Gamma = torch.linalg.solve(A, rhs)
        return Gamma
    except RuntimeError:
        return None


def _compute_gamma_via_augmented_expm(A: torch.Tensor, B: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute Γ using augmented matrix exponential.

    expm( [A B; 0 0] dt ) = [ Φ  Γ; 0  I ]  => top-right block is Γ.
    """
    n = A.shape[0]
    m = B.shape[1]
    Z = torch.zeros((n + m, n + m), dtype=A.dtype, device=A.device)
    Z[:n, :n] = A
    Z[:n, n:] = B
    M = torch.linalg.matrix_exp(Z * float(dt))
    Gamma = M[:n, n:]
    return Gamma


def discretize_linear_ct_matrix_exp(
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    dt: float | torch.Tensor,
    *,
    gamma_method: str = "auto",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Discretize linear CT system z' = A z + B u with zero-order hold input.

    Returns (Φ, Γ) where Φ = exp(A dt), Γ = ∫_0^{dt} exp(A s) B ds.
    If B is None or has zero columns, Γ is None.

    gamma_method: 'auto' (default), 'inverse', or 'augmented'.
    """
    if isinstance(dt, torch.Tensor):
        dt_val = float(dt.detach().cpu().item())
    else:
        dt_val = float(dt)
    Phi = torch.linalg.matrix_exp(A * dt_val)
    if B is None or B.numel() == 0:
        return Phi, None
    method = gamma_method
    if method not in ("auto", "inverse", "augmented"):
        raise ValueError("gamma_method must be one of {'auto','inverse','augmented'}")
    if method == "inverse":
        Gamma = _compute_gamma_via_inverse(A, Phi, B)
        if Gamma is None:
            # Fallback silently if inverse path not viable
            Gamma = _compute_gamma_via_augmented_expm(A, B, dt_val)
        return Phi, Gamma
    if method == "augmented":
        return Phi, _compute_gamma_via_augmented_expm(A, B, dt_val)
    # auto: try inverse-based formula, then fall back
    Gamma = _compute_gamma_via_inverse(A, Phi, B)
    if Gamma is None:
        Gamma = _compute_gamma_via_augmented_expm(A, B, dt_val)
    return Phi, Gamma


# Optional regularizers (kept lightweight for experimentation)
def frobenius_penalty(A: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Frobenius norm penalty on a matrix parameter."""
    return weight * torch.sum(A * A)


def spectral_radius_upper_bound(A: torch.Tensor) -> torch.Tensor:
    """Return a differentiable upper bound on the spectral radius via ||A||_2.

    This computes the operator 2-norm (largest singular value).
    """
    return torch.linalg.norm(A, ord=2)
