import numpy as np

from data import pendulum_spec, rk4_step


def test_pendulum_dynamics_matches_docstring():
    spec = pendulum_spec()

    # Dynamics: dot(x1) = x2, dot(x2) = -(g/l) * sin(x1)
    g_over_l = 9.81 / 1.0
    x = np.array([0.5, -0.2], dtype=np.float32)
    dx = spec.dynamics(0.0, x, None)

    expected = np.array([x[1], -g_over_l * np.sin(x[0])], dtype=np.float32)
    np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)


def test_pendulum_init_sampler_ranges():
    spec = pendulum_spec()
    rng = np.random.default_rng(123)
    samples = np.stack([spec.init_sampler(rng) for _ in range(2000)])
    x1 = samples[:, 0]
    x2 = samples[:, 1]
    assert (x1 >= -np.pi - 1e-6).all() and (x1 <= np.pi + 1e-6).all()
    assert (x2 >= -2.0 - 1e-6).all() and (x2 <= 2.0 + 1e-6).all()


def test_pendulum_fixed_points():
    spec = pendulum_spec()
    g_over_l = 9.81 / 1.0

    # Fixed points occur at x2 = 0 and sin(x1) = 0 -> x1 = k*pi
    for k in range(-2, 3):
        x1 = k * np.pi
        state = np.array([x1, 0.0], dtype=np.float32)
        f = spec.dynamics(0.0, state, None)
        # Numerical equality within tolerance
        np.testing.assert_allclose(f, np.zeros_like(state), rtol=1e-5, atol=1e-5)


def test_pendulum_energy_near_conserved_short_horizon():
    spec = pendulum_spec()
    rng = np.random.default_rng(0)
    dt = 1e-3
    steps = 20000

    g_over_l = 9.81 / 1.0
    state = spec.init_sampler(rng).astype(np.float32)

    def potential(theta: float) -> float:
        # Choose zero potential at theta = 0: V(theta) = g/l * (1 - cos(theta))
        return g_over_l * (1.0 - np.cos(theta))

    def energy(st: np.ndarray) -> float:
        theta, omega = st
        return 0.5 * omega**2 + potential(theta)

    E0 = energy(state)
    Emin, Emax = E0, E0

    for k in range(steps):
        state = rk4_step(spec.dynamics, k * dt, state, dt)
        E = energy(state)
        Emin = min(Emin, E)
        Emax = max(Emax, E)

    # RK4 on conservative system should be nearly energy-preserving over short horizons
    assert (Emax - Emin) < 5e-3


