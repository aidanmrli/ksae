import numpy as np

from data import duffing_spec, rk4_step


def test_duffing_dynamics_matches_docstring():
    spec = duffing_spec()

    # Dynamics should be: dx = v, dv = x - x^3
    x = np.array([0.5, -0.2], dtype=np.float32)
    dx = spec.dynamics(0.0, x, None)

    expected = np.array([x[1], x[0] - x[0] ** 3], dtype=np.float32)
    np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)


def test_duffing_init_sampler_ranges():
    spec = duffing_spec()
    rng = np.random.default_rng(123)
    samples = np.stack([spec.init_sampler(rng) for _ in range(1000)])
    x1 = samples[:, 0]
    x2 = samples[:, 1]
    assert (x1 >= -2.0 - 1e-6).all() and (x1 <= 2.0 + 1e-6).all()
    assert (x2 >= -1.0 - 1e-6).all() and (x2 <= 1.0 + 1e-6).all()


def test_duffing_energy_near_conserved_short_horizon():
    spec = duffing_spec()
    rng = np.random.default_rng(0)
    dt = 1e-3
    steps = 20000

    state = spec.init_sampler(rng).astype(np.float32)

    def potential(x):
        # V(x) = -1/2 x^2 + 1/4 x^4 so that dV/dx = -x + x^3
        return -0.5 * x**2 + 0.25 * x**4

    def energy(st):
        return 0.5 * st[1] ** 2 + potential(st[0])

    E0 = energy(state)
    Emin, Emax = E0, E0

    for k in range(steps):
        state = rk4_step(spec.dynamics, k * dt, state, dt)
        E = energy(state)
        Emin = min(Emin, E)
        Emax = max(Emax, E)

    # RK4 on conservative system should be nearly energy-preserving over short horizons
    assert (Emax - Emin) < 5e-3
