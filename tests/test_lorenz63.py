import numpy as np

from data import lorenz63_spec, rk4_step


def test_lorenz63_dynamics_matches_docstring():
    spec = lorenz63_spec()

    # Dynamics should be: dot(x1) = sigma*(x2-x1), dot(x2) = x1*(rho-x3)-x2, dot(x3) = x1*x2 - beta*x3
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dx = spec.dynamics(0.0, x, None)

    expected = np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ], dtype=np.float32)
    np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)


def test_lorenz63_initial_conditions_distribution():
    spec = lorenz63_spec()
    rng = np.random.default_rng(42)

    # Sample many initial conditions and check they're centered around (0, 1, 1.05)
    num_samples = 10000
    samples = np.array([spec.init_sampler(rng) for _ in range(num_samples)])

    expected_mean = np.array([0.0, 1.0, 1.05])
    sample_mean = np.mean(samples, axis=0)

    # Should be close to the base point (within statistical fluctuation)
    np.testing.assert_allclose(sample_mean, expected_mean, atol=0.05)

    # Check that standard deviation is approximately 1
    sample_std = np.std(samples, axis=0)
    np.testing.assert_allclose(sample_std, 1.0, atol=0.05)


def test_lorenz63_chaotic_trajectory():
    spec = lorenz63_spec()
    rng = np.random.default_rng(0)

    # Simulate a trajectory and check it doesn't settle to equilibrium
    dt = 0.01
    steps = 10000
    state = spec.init_sampler(rng).astype(np.float32)

    trajectory = np.zeros((steps, 3), dtype=np.float32)
    for k in range(steps):
        trajectory[k] = state
        state = rk4_step(spec.dynamics, k * dt, state, dt)

    # For chaotic systems, the trajectory should explore a region of state space
    # Check that the state varies significantly over time
    x_range = np.ptp(trajectory[:, 0])  # peak-to-peak range
    y_range = np.ptp(trajectory[:, 1])
    z_range = np.ptp(trajectory[:, 2])

    # All dimensions should have significant variation (more than 10 units range)
    assert x_range > 10.0, f"x range {x_range} too small for chaotic behavior"
    assert y_range > 10.0, f"y range {y_range} too small for chaotic behavior"
    assert z_range > 20.0, f"z range {z_range} too small for chaotic behavior"


def test_lorenz63_sensitivity_to_initial_conditions():
    """Test the butterfly effect - small differences in initial conditions lead to large differences."""
    spec = lorenz63_spec()
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)

    dt = 0.01
    steps = 5000

    # Two trajectories with slightly different initial conditions
    state1 = spec.init_sampler(rng1).astype(np.float32)
    state2 = spec.init_sampler(rng2).astype(np.float32)

    # Ensure they're different
    assert not np.allclose(state1, state2, atol=1e-3)

    # Evolve both trajectories
    for k in range(steps):
        state1 = rk4_step(spec.dynamics, k * dt, state1, dt)
        state2 = rk4_step(spec.dynamics, k * dt, state2, dt)

    # By the end, they should be very different (chaotic divergence)
    distance = np.linalg.norm(state1 - state2)
    assert distance > 1.0, f"Trajectories didn't diverge enough: distance = {distance}"
