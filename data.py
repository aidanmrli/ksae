"""Datasets for sparse coding and nonlinear dynamical systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Sparse coding dataset for LISTA
# ---------------------------------------------------------------------------


def make_dictionary(input_dim: int, dict_dim: int, seed: int) -> np.ndarray:
    """Create a random dictionary with unit-norm columns."""
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(input_dim, dict_dim)).astype(np.float32)
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True) + 1e-8
    return dictionary


class ToySparseCodingDataset(Dataset[Dict[str, torch.Tensor]]):
    """Synthetic sparse coding dataset with Laplace-distributed codes."""

    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        dict_dim: int,
        sparsity: float = 0.1,
        noise_std: float = 0.01,
        seed: int = 0,
        dictionary: Optional[np.ndarray] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.dictionary = dictionary if dictionary is not None else make_dictionary(input_dim, dict_dim, seed)
        codes = self._sample_codes(num_samples, dict_dim, sparsity)
        signals = codes @ self.dictionary.T + self.rng.normal(scale=noise_std, size=(num_samples, input_dim)).astype(
            np.float32
        )
        self.codes = torch.from_numpy(codes)
        self.signals = torch.from_numpy(signals.astype(np.float32))
        self.dictionary_tensor = torch.from_numpy(self.dictionary.astype(np.float32))

    def _sample_codes(self, num_samples: int, dict_dim: int, sparsity: float) -> np.ndarray:
        mask = self.rng.uniform(size=(num_samples, dict_dim)) < sparsity
        laplace = self.rng.laplace(scale=1.0, size=(num_samples, dict_dim)).astype(np.float32)
        return laplace * mask.astype(np.float32)

    def __len__(self) -> int:
        return self.codes.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.signals[index],
            "z_star": self.codes[index],
            "dictionary": self.dictionary_tensor,
        }


# ---------------------------------------------------------------------------
# Dynamical systems dataset for Koopman-style models
# ---------------------------------------------------------------------------


@dataclass
class DynamicalSystemSpec:
    """Specification of a deterministic dynamical system x' = f(t, x, u)."""

    name: str
    state_dim: int
    dynamics: Callable[[float, np.ndarray, Optional[np.ndarray]], np.ndarray]
    init_sampler: Callable[[np.random.Generator], np.ndarray]
    control_dim: int = 0
    control_sampler: Optional[Callable[[np.random.Generator, int], np.ndarray]] = None


def rk4_step(
    f: Callable[[float, np.ndarray, Optional[np.ndarray]], np.ndarray],
    t: float,
    x: np.ndarray,
    dt: float,
    control: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fourth-order Runge-Kutta integrator for a single step."""
    k1 = f(t, x, control)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, control)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, control)
    k4 = f(t + dt, x + dt * k3, control)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_system(
    spec: DynamicalSystemSpec,
    num_steps: int,
    dt: float,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Simulate a trajectory and optional control sequence.
    
    Args:
        spec: Specification of the dynamical system.
        num_steps: Number of steps to simulate.
        dt: Time step.
        noise_std: Standard deviation of the noise.
        rng: Random number generator.
    """
    x = spec.init_sampler(rng).astype(np.float32)
    controls = None
    if spec.control_dim > 0 and spec.control_sampler is not None:
        controls = spec.control_sampler(rng, num_steps).astype(np.float32)

    states = np.zeros((num_steps, spec.state_dim), dtype=np.float32)
    if controls is not None:
        ctrl_seq = controls
    else:
        ctrl_seq = np.zeros((num_steps, spec.control_dim), dtype=np.float32)

    current = x
    for idx in range(num_steps):
        noise = rng.normal(scale=noise_std, size=spec.state_dim).astype(np.float32)
        states[idx] = current + noise
        u_t = ctrl_seq[idx] if spec.control_dim > 0 else None
        current = rk4_step(spec.dynamics, idx * dt, current, dt, u_t)
    return states, controls


class DynamicalSystemDataset(Dataset[Dict[str, torch.Tensor]]):
    """Batch of simulated trajectories for Koopman models.
    
    Args:
        spec: Specification of the dynamical system.
        num_samples: Number of trajectories to simulate.
        seq_len: Length of each trajectory.
        dt: Time step.
        noise_std: Standard deviation of the noise.
        seed: Random seed.
    """

    def __init__(
        self,
        spec: DynamicalSystemSpec,
        num_samples: int,
        seq_len: int,
        dt: float = 0.01,
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        rng = np.random.default_rng(seed)
        trajectories = []
        controls = [] if spec.control_dim > 0 else None
        for _ in range(num_samples):
            states, ctrl = simulate_system(spec, seq_len, dt, noise_std, rng)
            trajectories.append(states)
            if spec.control_dim > 0:
                assert controls is not None
                controls.append(ctrl if ctrl is not None else np.zeros((seq_len, spec.control_dim), dtype=np.float32))
        self.trajectories = torch.from_numpy(np.stack(trajectories)).float()
        self.controls = torch.from_numpy(np.stack(controls)).float() if controls is not None else None

    def __len__(self) -> int:
        return self.trajectories.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        batch = {"x": self.trajectories[index]}
        if self.controls is not None:
            batch["u"] = self.controls[index]
        return batch


class CachedDynamicalSystemDataset(DynamicalSystemDataset):
    """Dataset backed by cached trajectories stored on disk.

    This subclasses DynamicalSystemDataset for compatibility with existing wrappers.
    """

    def __init__(
        self,
        spec: DynamicalSystemSpec,
        trajectories: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> None:
        # Bypass parent generation; just set attributes
        self.spec = spec
        self.trajectories = torch.from_numpy(trajectories.astype(np.float32))
        self.controls = torch.from_numpy(controls.astype(np.float32)) if controls is not None else None


# ---------------------------------------------------------------------------
# Windowed view over trajectory datasets for training
# ---------------------------------------------------------------------------


class WindowedSequenceDataset(Dataset[Dict[str, torch.Tensor]]):
    """Sample fixed-length windows from a trajectory dataset.

    This wraps a dataset that exposes full trajectories via attributes
    `trajectories` (Tensor[num_samples, seq_len, state_dim]) and optional
    `controls` (Tensor[num_samples, seq_len, control_dim]).

    Each item returns a window of length `horizon` in terms of prediction steps,
    which corresponds to `horizon + 1` states and `horizon` controls:

        x: shape (horizon + 1, state_dim)
        u: shape (horizon, control_dim) if controls exist, else omitted

    If `subset_length` is provided, window start indices are drawn uniformly
    from [0, min(subset_length, full_seq_len) - (horizon + 1)].
    """

    def __init__(
        self,
        base: Dataset[Dict[str, torch.Tensor]] | DynamicalSystemDataset,
        *,
        horizon: int,
        subset_length: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        # Support torch.utils.data.Subset wrapping DynamicalSystemDataset
        if hasattr(base, "dataset") and hasattr(base, "indices"):
            # type: ignore[attr-defined]
            self._inner: DynamicalSystemDataset = base.dataset  # type: ignore[assignment]
            self._indices = list(base.indices)  # type: ignore[attr-defined]
        else:
            if not isinstance(base, DynamicalSystemDataset):
                raise TypeError("WindowedSequenceDataset expects a DynamicalSystemDataset or a Subset thereof")
            self._inner = base
            self._indices = list(range(len(base)))
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1 to define prediction steps")
        # Determine full sequence length from the inner dataset
        full_seq_len = int(self._inner.trajectories.shape[1])
        max_start_from_full = max(0, full_seq_len - (self.horizon + 1))
        if subset_length is None:
            self.max_start = max_start_from_full
        else:
            # Restrict start positions to the first subset_length steps
            restricted = max(0, min(int(subset_length), full_seq_len) - (self.horizon + 1))
            self.max_start = min(max_start_from_full, restricted)
        if self.max_start < 0:
            raise ValueError("Invalid configuration: horizon exceeds available sequence length")
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        traj_idx = self._indices[index]
        start = int(self._rng.integers(0, self.max_start + 1)) if self.max_start > 0 else 0
        end_state = start + self.horizon + 1  # exclusive
        x_full = self._inner.trajectories[traj_idx]
        x_window = x_full[start:end_state]
        sample: Dict[str, torch.Tensor] = {"x": x_window}
        if self._inner.controls is not None:
            u_full = self._inner.controls[traj_idx]
            u_window = u_full[start : start + self.horizon]
            sample["u"] = u_window
        return sample

# ---------------------------------------------------------------------------
# Predefined systems
# ---------------------------------------------------------------------------


def pendulum_spec() -> DynamicalSystemSpec:
    """
    Pendulum model represents a freely swinging pole. 
    The initial conditions indicate the states from which the pole is released, 
    deviating slightly from the inverted position by ±10 degrees. 
    
    The state consists of the angle (x1) and the angular velocity (x2) and we report errors in radians
    """
    g_over_l = 9.81 / 1.0

    def dynamics(_t: float, state: np.ndarray, _u: Optional[np.ndarray]) -> np.ndarray:
        x1, x2 = state
        return np.array([x2, -g_over_l * np.sin(x1)], dtype=np.float32)

    def init_sampler(rng: np.random.Generator) -> np.ndarray:
        x1 = rng.uniform(-np.pi, np.pi)
        x2 = rng.uniform(-2.0, 2.0)
        return np.array([x1, x2], dtype=np.float32)

    return DynamicalSystemSpec(name="pendulum", state_dim=2, dynamics=dynamics, init_sampler=init_sampler)


def duffing_spec() -> DynamicalSystemSpec:
    """Duffing Oscillator represents a model for the motion of a damped and force-driven particle.
    
    It follows a nonlinear second order differential equation:
    2-dot(x) = x - x^3
    
    This particular instance admits two center points at (x, dot(x)) = (±1, 0), 
    and an unstable fixed point at the origin, (x, dot(x)) = (0, 0).
    
    Sample initial conditions x1 uniformly from [-2, 2] and x2 from a uniform distribution on [-1, 1].
    """
    def dynamics(_t: float, state: np.ndarray, _u: Optional[np.ndarray]) -> np.ndarray:
        x, v = state
        # Undamped, unforced Duffing: x'' = x - x^3
        return np.array([v, x - x**3], dtype=np.float32)

    def init_sampler(rng: np.random.Generator) -> np.ndarray:
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-1.0, 1.0)
        return np.array([x1, x2], dtype=np.float32)

    return DynamicalSystemSpec(name="duffing", state_dim=2, dynamics=dynamics, init_sampler=init_sampler)


def lotka_volterra_spec() -> DynamicalSystemSpec:
    """Lotka-Volterra model for the population evolution of biological systems with predator-prey interactions.
    
    The dynamics are governed by the following equations:

    dot(x1) = alpha * x1 - beta * x1 * x2
    dot(x2) = delta * x1 * x2 - gamma * x2
    
    Note that x1 and x2 represent the prey and predator populations, respectively.
    
    The system is known for its abrupt switch in population growth and admits two fixed points, one at the origin
    and a centre point at (x1, x2) = (gamma / delta, alpha / beta).
    
    We set alpha = beta = gamma = delta = 0.2 and 
    sample initial conditions x1 and x2 from a uniform distribution on [0.02, 3.0].
    """
    alpha, beta, gamma, delta = 0.2, 0.2, 0.2, 0.2

    def dynamics(_t: float, state: np.ndarray, _u: Optional[np.ndarray]) -> np.ndarray:
        prey, predator = state
        return np.array([
            alpha * prey - beta * prey * predator,
            delta * prey * predator - gamma * predator,
        ], dtype=np.float32)

    def init_sampler(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=0.02, high=3.0, size=2).astype(np.float32)

    return DynamicalSystemSpec(name="lotka_volterra", state_dim=2, dynamics=dynamics, init_sampler=init_sampler)


def lorenz63_spec() -> DynamicalSystemSpec:
    """
    Lorenz 63 model is a simple three-dimensional system that exhibits chaotic behavior.
    It features equilibrium points, some stable and some unstable, 
    and is renowned for the "butterfly effect" arising from its sensitivity to initial conditions.
    
    The dynamics are governed by the following equations:

    dot(x1) = sigma * (x2 - x1)
    dot(x2) = x1 * (rho - x3) - x2
    dot(x3) = x1 * x2 - beta * x3

    We use the standard values used for the chaotic Lorenz attractor from Lorenz (1963):
    sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0
    Initial conditions are generated by perturbing the point (0, 1, 1.05) with Gaussian-distributed noise having a standard deviation of 1.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def dynamics(_t: float, state: np.ndarray, _u: Optional[np.ndarray]) -> np.ndarray:
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ], dtype=np.float32)

    def init_sampler(rng: np.random.Generator) -> np.ndarray:
        base_point = np.array([0.0, 1.0, 1.05], dtype=np.float32)
        noise = rng.normal(loc=0.0, scale=1.0, size=3).astype(np.float32)
        return base_point + noise

    return DynamicalSystemSpec(name="lorenz63", state_dim=3, dynamics=dynamics, init_sampler=init_sampler)


def parabolic_spec() -> DynamicalSystemSpec:
    f"""Parabolic Attractor dynamical system with single fixed point at the origin.
    The dynamics are governed by the following equations:
    
    dot(x1) = mu * x1
    dot(x2) = lambda * (x2 - x1^2)
    The system admits a solution that is asymptotically attracted to x2 = x1^2 for lambda < mu < 0.
    
    The Koopman embedding, z, that adheres to globally linear dynamics, can be coded 
    by augmenting the state with the additional nonlinear measurement of z3 = x1^2.
    Then the derivative of z is given by:
    
    dot(z) = [mu * z1, lambda * z2 - lambda * z3, 2 * mu * z3]
    
    Set lambda = -1.0 and mu = -0.1 and sample initial conditions x1, x2 from a uniform distribution on [-1.0, 1.0].
    """
    mu, lam = -0.1, -1.0

    def dynamics(_t: float, state: np.ndarray, _u: Optional[np.ndarray]) -> np.ndarray:
        x1, x2 = state
        dx1 = mu * x1
        dx2 = lam * (x2 - x1**2)
        return np.array([dx1, dx2], dtype=np.float32)

    def init_sampler(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=-1.0, high=1.0, size=2).astype(np.float32)

    return DynamicalSystemSpec(name="parabolic", state_dim=2, dynamics=dynamics, init_sampler=init_sampler)


SYSTEM_REGISTRY: Dict[str, DynamicalSystemSpec] = {
    "pendulum": pendulum_spec(),
    "duffing": duffing_spec(),
    "lotka_volterra": lotka_volterra_spec(),
    "lorenz63": lorenz63_spec(),
    "parabolic": parabolic_spec(),
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def create_dataloader(
    dataset: Dataset[Dict[str, torch.Tensor]],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader[Dict[str, torch.Tensor]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def create_lista_datasets(
    *,
    num_samples: int,
    input_dim: int,
    dict_dim: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    batch_size: int = 64,
    seed: int = 0,
    sparsity: float = 0.1,
    noise_std: float = 0.01,
) -> Tuple[DataLoader[Dict[str, torch.Tensor]], DataLoader[Dict[str, torch.Tensor]], ToySparseCodingDataset]:
    dataset = ToySparseCodingDataset(
        num_samples=num_samples,
        input_dim=input_dim,
        dict_dim=dict_dim,
        sparsity=sparsity,
        noise_std=noise_std,
        seed=seed,
    )
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        create_dataloader(train_set, batch_size=batch_size, shuffle=True),
        create_dataloader(val_set, batch_size=batch_size, shuffle=False),
        test_set,  # tests are evaluated directly to access ground truth codes
    )


def create_dynamics_dataloaders(
    system: str,
    *,
    num_samples: int,
    seq_len: int,
    batch_size: int,
    dt: float = 0.01,
    noise_std: float = 0.0,
    seed: int = 0,
    splits: Sequence[float] = (0.8, 0.1, 0.1),
    train_context_length: int | None = None,
    use_offline_cache: bool = False,
    cache_dir: str | Path = "data",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-7,
) -> Tuple[
    DataLoader[Dict[str, torch.Tensor]],
    DataLoader[Dict[str, torch.Tensor]],
    DataLoader[Dict[str, torch.Tensor]],
]:
    spec = SYSTEM_REGISTRY[system]
    if use_offline_cache and spec.control_dim == 0:
        trajectories, controls = load_or_generate_cached(
            spec,
            num_samples=num_samples,
            seq_len=seq_len,
            dt=dt,
            noise_std=noise_std,
            seed=seed,
            cache_dir=cache_dir,
            rtol=ode_rtol,
            atol=ode_atol,
        )
        dataset = CachedDynamicalSystemDataset(spec, trajectories=trajectories, controls=controls)
    else:
        dataset = DynamicalSystemDataset(spec, num_samples=num_samples, seq_len=seq_len, dt=dt, noise_std=noise_std, seed=seed)
    train_frac, val_frac, test_frac = splits
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Splits must sum to 1.0"
    train_size = int(len(dataset) * train_frac)
    val_size = int(len(dataset) * val_frac)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    # For training, optionally wrap with windowed sampling using the specified context length
    if train_context_length is not None:
        train_dataset = WindowedSequenceDataset(
            train_set,
            horizon=int(train_context_length),
            subset_length=int(seq_len),
            seed=seed,
        )
    else:
        train_dataset = train_set

    return (
        create_dataloader(train_dataset, batch_size=batch_size, shuffle=True),
        create_dataloader(val_set, batch_size=batch_size, shuffle=False),
        create_dataloader(test_set, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# SciPy-based offline ODE integration and caching
# ---------------------------------------------------------------------------


def integrate_spec_scipy(
    spec: DynamicalSystemSpec,
    x0: np.ndarray,
    num_steps: int,
    dt: float,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> np.ndarray:
    """Integrate a system with SciPy solve_ivp over a fixed grid.

    Returns states with shape (num_steps, state_dim) or (num_steps+1?),
    We generate exactly `num_steps` states to align with dataset convention of seq_len.
    """
    t0 = 0.0
    t1 = float((num_steps - 1) * dt)
    t_eval = np.linspace(t0, t1, num_steps, dtype=np.float64)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return spec.dynamics(float(t), y.astype(np.float32), None).astype(np.float64)

    sol = solve_ivp(rhs, (t0, t1), x0.astype(np.float64), method="RK45", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    states = sol.y.T.astype(np.float32)
    return states


def generate_scipy_trajectories(
    spec: DynamicalSystemSpec,
    num_samples: int,
    seq_len: int,
    dt: float,
    noise_std: float,
    seed: int,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    rng = np.random.default_rng(seed)
    trajs = np.zeros((num_samples, seq_len, spec.state_dim), dtype=np.float32)
    for i in range(num_samples):
        x0 = spec.init_sampler(rng).astype(np.float32)
        states = integrate_spec_scipy(spec, x0, seq_len, dt, rtol=rtol, atol=atol)
        if noise_std > 0:
            states = states + rng.normal(scale=noise_std, size=states.shape).astype(np.float32)
        trajs[i] = states
    controls = None
    return trajs, controls


def _cache_filename(
    spec: DynamicalSystemSpec,
    *,
    num_samples: int,
    seq_len: int,
    dt: float,
    noise_std: float,
    seed: int,
) -> str:
    return f"{spec.name}_N{num_samples}_T{seq_len}_dt{dt:.6f}_noise{noise_std:.6f}_seed{seed}.npz"


def load_or_generate_cached(
    spec: DynamicalSystemSpec,
    *,
    num_samples: int,
    seq_len: int,
    dt: float,
    noise_std: float,
    seed: int,
    cache_dir: str | Path = "data",
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fname = _cache_filename(spec, num_samples=num_samples, seq_len=seq_len, dt=dt, noise_std=noise_std, seed=seed)
    fpath = cache_path / fname
    if fpath.exists():
        data = np.load(fpath)
        trajs = data["trajectories"].astype(np.float32)
        ctrls = data["controls"].astype(np.float32) if "controls" in data and data["controls"].size > 0 else None
        return trajs, ctrls
    trajs, ctrls = generate_scipy_trajectories(
        spec,
        num_samples=num_samples,
        seq_len=seq_len,
        dt=dt,
        noise_std=noise_std,
        seed=seed,
        rtol=rtol,
        atol=atol,
    )
    np.savez_compressed(fpath, trajectories=trajs, controls=(ctrls if ctrls is not None else np.array([])))
    return trajs, ctrls
