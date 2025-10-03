import numpy as np

from data import parabolic_spec, rk4_step


def test_parabolic_dynamics_matches_docstring():
    spec = parabolic_spec()

    # Dynamics should be: dx1 = mu * x1, dx2 = lambda * (x2 - x1^2)
    mu, lam = -0.1, -1.0
    x = np.array([0.5, 0.2], dtype=np.float32)
    dx = spec.dynamics(0.0, x, None)

    expected = np.array([mu * x[0], lam * (x[1] - x[0] ** 2)], dtype=np.float32)
    np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)


def test_parabolic_converges_to_parabola():
    spec = parabolic_spec()
    rng = np.random.default_rng(0)

    # Simulate for a moderate horizon
    dt = 0.01
    steps = 5000
    state = spec.init_sampler(rng).astype(np.float32)

    for k in range(steps):
        state = rk4_step(spec.dynamics, k * dt, state, dt)

    x1, x2 = state
    # Asymptotically attracted to x2 = x1^2; allow small tolerance
    assert abs(x2 - x1 ** 2) < 5e-2


def test_koopman_embedding_linear_evolution():
    spec = parabolic_spec()

    mu, lam = -0.1, -1.0

    def z_dot(_t: float, z: np.ndarray) -> np.ndarray:
        # dot(z) = [mu * z1, lam * z2 - lam * z3, 2 * mu * z3]
        return np.array([mu * z[0], lam * z[1] - lam * z[2], 2.0 * mu * z[2]], dtype=np.float32)

    rng = np.random.default_rng(1)
    x = spec.init_sampler(rng).astype(np.float32)
    z = np.array([x[0], x[1], x[0] ** 2], dtype=np.float32)

    dt = 0.01
    steps = 1000

    x_curr = x.copy()
    z_curr = z.copy()
    for k in range(steps):
        x_curr = rk4_step(spec.dynamics, k * dt, x_curr, dt)
        # Evolve z linearly according to z_dot
        k1 = z_dot(k * dt, z_curr)
        k2 = z_dot(k * dt + 0.5 * dt, z_curr + 0.5 * dt * k1)
        k3 = z_dot(k * dt + 0.5 * dt, z_curr + 0.5 * dt * k2)
        k4 = z_dot(k * dt + dt, z_curr + dt * k3)
        z_curr = z_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Check consistency between nonlinear state and embedding components
    np.testing.assert_allclose(z_curr[0], x_curr[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(z_curr[1], x_curr[1], rtol=1e-3, atol=1e-3)
    # z3 should approximate x1^2
    np.testing.assert_allclose(z_curr[2], x_curr[0] ** 2, rtol=1e-2, atol=1e-2)


