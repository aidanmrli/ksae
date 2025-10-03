import numpy as np

from data import lotka_volterra_spec, rk4_step


def test_lotka_volterra_dynamics_matches_docstring():
    spec = lotka_volterra_spec()

    # Dynamics: dx1 = a*x1 - b*x1*x2, dx2 = d*x1*x2 - g*x2 with a=b=g=d=0.2
    alpha = beta = gamma = delta = 0.2
    x = np.array([0.5, 0.8], dtype=np.float32)
    dx = spec.dynamics(0.0, x, None)

    expected = np.array([
        alpha * x[0] - beta * x[0] * x[1],
        delta * x[0] * x[1] - gamma * x[1],
    ], dtype=np.float32)
    np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)


def test_lotka_volterra_init_sampler_ranges():
    spec = lotka_volterra_spec()
    rng = np.random.default_rng(123)
    samples = np.stack([spec.init_sampler(rng) for _ in range(1000)])
    x1 = samples[:, 0]
    x2 = samples[:, 1]
    assert (x1 >= 0.02 - 1e-6).all() and (x1 <= 3.0 + 1e-6).all()
    assert (x2 >= 0.02 - 1e-6).all() and (x2 <= 3.0 + 1e-6).all()


def test_lotka_volterra_fixed_points():
    spec = lotka_volterra_spec()
    alpha = beta = gamma = delta = 0.2

    # Fixed points: (0,0) and (gamma/delta, alpha/beta)
    origin = np.array([0.0, 0.0], dtype=np.float32)
    center = np.array([gamma / delta, alpha / beta], dtype=np.float32)

    f_origin = spec.dynamics(0.0, origin, None)
    f_center = spec.dynamics(0.0, center, None)

    np.testing.assert_allclose(f_origin, np.zeros_like(origin), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(f_center, np.zeros_like(center), rtol=1e-7, atol=1e-7)


def test_lotka_volterra_positive_states_remain_positive_short_horizon():
    spec = lotka_volterra_spec()
    rng = np.random.default_rng(0)
    dt = 1e-3
    steps = 5000
    state = spec.init_sampler(rng).astype(np.float32)

    for k in range(steps):
        state = rk4_step(spec.dynamics, k * dt, state, dt)
        assert state[0] > 0.0 and state[1] > 0.0


