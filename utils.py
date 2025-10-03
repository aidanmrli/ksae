"""Utility helpers for experiment management and reproducibility."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
