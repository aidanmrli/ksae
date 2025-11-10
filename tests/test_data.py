"""Unit tests for data.py environments.

Tests the PyTorch-based dynamical system implementations for correctness,
including mathematical dynamics, integration, trajectory generation, and batching.
"""

import unittest
import torch
from config import Config
from data import (
    Env, Wrapper, VectorWrapper,
    Pendulum, Duffing, LotkaVolterra, Lorenz63, Parabolic,
    integrate_euler, integrate_rk4, generate_trajectory, make_env
)


class TestBaseClasses(unittest.TestCase):
    """Test base environment classes and wrappers."""

    def setUp(self):
        """Set up test configuration."""
        self.cfg = Config()
        self.cfg.SEED = 42
        self.cfg.ENV.ENV_NAME = "duffing"
        self.cfg.ENV.DUFFING.DT = 0.01

    def test_env_abstract(self):
        """Test that Env is abstract and cannot be instantiated."""
        with self.assertRaises(TypeError):
            Env(self.cfg)

    def test_wrapper_unwrapped(self):
        """Test that wrapper properly unwraps to base environment."""
        env = Duffing(self.cfg)
        wrapped = Wrapper(env)
        self.assertIs(wrapped.unwrapped, env)
        self.assertEqual(wrapped.observation_size, 2)
        self.assertEqual(wrapped.action_size, 0)

    def test_vector_wrapper_batch_size(self):
        """Test VectorWrapper properly handles batching."""
        env = Duffing(self.cfg)
        batch_size = 16
        vec_env = VectorWrapper(env, batch_size)

        rng = torch.Generator()
        rng.manual_seed(42)
        batch_states = vec_env.reset(rng)

        self.assertEqual(batch_states.shape, (batch_size, 2))
        self.assertEqual(batch_states.dtype, torch.float32)

    def test_vector_wrapper_step(self):
        """Test VectorWrapper step function."""
        env = Duffing(self.cfg)
        batch_size = 8
        vec_env = VectorWrapper(env, batch_size)

        rng = torch.Generator()
        rng.manual_seed(42)
        batch_states = vec_env.reset(rng)
        next_states = vec_env.step(batch_states)

        self.assertEqual(next_states.shape, (batch_size, 2))
        self.assertEqual(next_states.dtype, torch.float32)
        # States should change after stepping
        self.assertFalse(torch.allclose(batch_states, next_states))

    def test_vector_wrapper_diverse_initial_states(self):
        """Test VectorWrapper produces different initial states for each environment."""
        env = Duffing(self.cfg)
        batch_size = 16
        vec_env = VectorWrapper(env, batch_size)

        rng = torch.Generator()
        rng.manual_seed(42)
        batch_states = vec_env.reset(rng)

        # Each environment should have a different initial state
        # Check that not all states are identical
        unique_states = torch.unique(batch_states, dim=0)
        self.assertGreater(len(unique_states), 1, 
                          "All initial states are identical - RNG splitting failed")
        
        # Most states should be unique (with high probability for random initialization)
        self.assertGreater(len(unique_states), batch_size * 0.8,
                          f"Expected diverse states, got {len(unique_states)}/{batch_size} unique")


class TestIntegration(unittest.TestCase):
    """Test numerical integration functions."""

    def test_integrate_euler_simple(self):
        """Test Euler integration with simple linear dynamics."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        dt = 0.1

        def linear_dynamics(state, action=None):
            # dx/dt = -state (exponential decay)
            return -state

        x_next = integrate_euler(x, None, dt, linear_dynamics)
        expected = x + dt * linear_dynamics(x)

        torch.testing.assert_close(x_next, expected)
        self.assertEqual(x_next.shape, x.shape)
        self.assertEqual(x_next.dtype, torch.float32)

    def test_integrate_euler_preserves_shape(self):
        """Test that Euler integration preserves state shape."""
        for dim in [2, 3, 5]:
            x = torch.randn(dim, dtype=torch.float32)
            dt = 0.01

            def identity_dynamics(state, action=None):
                return torch.zeros_like(state)

            x_next = integrate_euler(x, None, dt, identity_dynamics)
            self.assertEqual(x_next.shape, x.shape)

    def test_integrate_rk4_simple(self):
        """Test RK4 integration with simple linear dynamics."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        dt = 0.1

        def linear_dynamics(state, action=None):
            # dx/dt = -state (exponential decay)
            return -state

        x_next = integrate_rk4(x, None, dt, linear_dynamics)

        # Compute expected RK4 result manually
        k1 = linear_dynamics(x)
        k2 = linear_dynamics(x + 0.5 * dt * k1)
        k3 = linear_dynamics(x + 0.5 * dt * k2)
        k4 = linear_dynamics(x + dt * k3)
        expected = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        torch.testing.assert_close(x_next, expected)
        self.assertEqual(x_next.shape, x.shape)
        self.assertEqual(x_next.dtype, torch.float32)

    def test_integrate_rk4_preserves_shape(self):
        """Test that RK4 integration preserves state shape."""
        for dim in [2, 3, 5]:
            x = torch.randn(dim, dtype=torch.float32)
            dt = 0.01

            def identity_dynamics(state, action=None):
                return torch.zeros_like(state)

            x_next = integrate_rk4(x, None, dt, identity_dynamics)
            self.assertEqual(x_next.shape, x.shape)

    def test_rk4_vs_euler_accuracy(self):
        """Test that RK4 is more accurate than Euler for analytical solution."""
        # Test with exponential decay: dx/dt = -k*x, solution: x(t) = x0 * exp(-k*t)
        x0 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        k = 0.5
        dt = 0.1
        num_steps = 10

        def exponential_decay(state, action=None):
            return -k * state

        # Compute analytical solution
        t_final = dt * num_steps
        analytical = x0 * torch.exp(torch.tensor(-k * t_final, dtype=torch.float32))

        # Compute RK4 solution
        x_rk4 = x0.clone()
        for _ in range(num_steps):
            x_rk4 = integrate_rk4(x_rk4, None, dt, exponential_decay)

        # Compute Euler solution
        x_euler = x0.clone()
        for _ in range(num_steps):
            x_euler = integrate_euler(x_euler, None, dt, exponential_decay)

        # RK4 error should be significantly smaller than Euler error
        rk4_error = torch.norm(x_rk4 - analytical)
        euler_error = torch.norm(x_euler - analytical)

        self.assertLess(rk4_error.item(), euler_error.item())
        self.assertLess(rk4_error.item(), 1e-4)  # RK4 should be very accurate

    def test_rk4_with_nonlinear_dynamics(self):
        """Test RK4 with nonlinear dynamics."""
        x = torch.tensor([1.0, 0.5], dtype=torch.float32)
        dt = 0.01

        def nonlinear_dynamics(state, action=None):
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = x1 - x1**3  # Duffing-like dynamics
            return torch.stack([dx1, dx2])

        x_next = integrate_rk4(x, None, dt, nonlinear_dynamics)

        self.assertEqual(x_next.shape, x.shape)
        self.assertEqual(x_next.dtype, torch.float32)
        # State should change after integration
        self.assertFalse(torch.allclose(x, x_next))

    def test_rk4_energy_conservation(self):
        """Test that RK4 conserves energy better than Euler for harmonic oscillator."""
        # Simple harmonic oscillator: dx1/dt = x2, dx2/dt = -omega^2 * x1
        # Energy: E = 0.5 * (x2^2 + omega^2 * x1^2)
        omega = 1.0
        x0 = torch.tensor([1.0, 0.0], dtype=torch.float32)
        dt = 0.1
        num_steps = 100

        def harmonic_oscillator(state, action=None):
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = -omega**2 * x1
            return torch.stack([dx1, dx2])

        def energy(state):
            x1, x2 = state[0], state[1]
            return 0.5 * (x2**2 + omega**2 * x1**2)

        E0 = energy(x0)

        # Test RK4
        x_rk4 = x0.clone()
        for _ in range(num_steps):
            x_rk4 = integrate_rk4(x_rk4, None, dt, harmonic_oscillator)
        E_rk4 = energy(x_rk4)

        # Test Euler
        x_euler = x0.clone()
        for _ in range(num_steps):
            x_euler = integrate_euler(x_euler, None, dt, harmonic_oscillator)
        E_euler = energy(x_euler)

        # Both should conserve energy, but RK4 should be better
        rk4_error = abs(E_rk4 - E0) / E0
        euler_error = abs(E_euler - E0) / E0

        self.assertLess(rk4_error.item(), euler_error.item())
        self.assertLess(rk4_error.item(), 0.01)  # RK4 should maintain energy to within 1%

    def test_rk4_with_control_input(self):
        """Test RK4 integration with control input."""
        x = torch.tensor([0.0, 0.0], dtype=torch.float32)
        u = torch.tensor([1.0, 0.5], dtype=torch.float32)
        dt = 0.1

        def controlled_dynamics(state, action=None):
            if action is None:
                return state
            return state + action  # Simple additive control

        x_next = integrate_rk4(x, u, dt, controlled_dynamics)

        self.assertEqual(x_next.shape, x.shape)
        # With additive control, state should change
        self.assertFalse(torch.allclose(x, x_next))

    def test_rk4_zero_timestep(self):
        """Test that RK4 with zero timestep returns original state."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        dt = 0.0

        def any_dynamics(state, action=None):
            return torch.ones_like(state) * 100.0  # Large derivative

        x_next = integrate_rk4(x, None, dt, any_dynamics)

        torch.testing.assert_close(x_next, x)

    def test_rk4_matches_analytical_sine_wave(self):
        """Test RK4 accuracy against analytical solution for oscillator."""
        # Test system: dx1/dt = x2, dx2/dt = -x1
        # Analytical solution: x1(t) = A*cos(t), x2(t) = -A*sin(t)
        A = 2.0
        x0 = torch.tensor([A, 0.0], dtype=torch.float32)
        dt = 0.01
        t_final = 2.0 * torch.pi  # One full period
        num_steps = int(t_final / dt)

        def oscillator_dynamics(state, action=None):
            x1, x2 = state[0], state[1]
            return torch.stack([x2, -x1])

        # Numerical integration with RK4
        x_rk4 = x0.clone()
        for _ in range(num_steps):
            x_rk4 = integrate_rk4(x_rk4, None, dt, oscillator_dynamics)

        # Analytical solution at t_final
        analytical = torch.tensor([A * torch.cos(torch.tensor(t_final)),
                                   -A * torch.sin(torch.tensor(t_final))],
                                  dtype=torch.float32)

        # After one full period, should return to initial state (within numerical error)
        torch.testing.assert_close(x_rk4, x0, rtol=1e-2, atol=1e-2)


class TestTrajectoryGeneration(unittest.TestCase):
    """Test trajectory generation utilities."""

    def test_generate_trajectory_length(self):
        """Test that generate_trajectory produces correct length."""
        def step_fn(state):
            return state + 0.01 * torch.randn_like(state)

        init_state = torch.tensor([1.0, 2.0], dtype=torch.float32)
        length = 50

        traj = generate_trajectory(step_fn, init_state, length=length)

        self.assertEqual(traj.shape, (length, 2))
        self.assertEqual(traj.dtype, torch.float32)

    def test_generate_trajectory_with_actions(self):
        """Test trajectory generation with action sequence."""
        def step_fn(state, action):
            return state + action

        init_state = torch.tensor([0.0, 0.0], dtype=torch.float32)
        actions = torch.ones(10, 2, dtype=torch.float32) * 0.1

        traj = generate_trajectory(step_fn, init_state, actions=actions)

        self.assertEqual(traj.shape, (10, 2))
        # Each step adds 0.1, so final state should be approximately [1.0, 1.0]
        expected_final = torch.tensor([1.0, 1.0], dtype=torch.float32)
        torch.testing.assert_close(traj[-1], expected_final, rtol=1e-5, atol=1e-5)

    def test_generate_trajectory_batched(self):
        """Test trajectory generation with batched states."""
        def step_fn(state):
            # Simple linear decay
            return 0.99 * state

        batch_size = 4
        state_dim = 2
        init_state = torch.ones(batch_size, state_dim, dtype=torch.float32)
        length = 20

        traj = generate_trajectory(step_fn, init_state, length=length)

        self.assertEqual(traj.shape, (length, batch_size, state_dim))


class TestPendulum(unittest.TestCase):
    """Test Pendulum environment."""

    def setUp(self):
        """Set up pendulum configuration."""
        self.cfg = Config()
        self.cfg.ENV.PENDULUM.DT = 0.01

    def test_pendulum_initialization(self):
        """Test pendulum creates with correct parameters."""
        env = Pendulum(self.cfg)
        self.assertEqual(env.observation_size, 2)
        self.assertEqual(env.action_size, 0)
        self.assertAlmostEqual(env.g_over_l, 9.81, places=5)
        self.assertEqual(env.dt, 0.01)

    def test_pendulum_reset(self):
        """Test pendulum reset produces valid initial states."""
        env = Pendulum(self.cfg)
        rng = torch.Generator()
        rng.manual_seed(42)

        state = env.reset(rng)

        self.assertEqual(state.shape, (2,))
        self.assertEqual(state.dtype, torch.float32)
        # Angle should be in [-π, π]
        self.assertGreaterEqual(state[0].item(), -torch.pi)
        self.assertLessEqual(state[0].item(), torch.pi)
        # Angular velocity should be in [-2, 2]
        self.assertGreaterEqual(state[1].item(), -2.0)
        self.assertLessEqual(state[1].item(), 2.0)

    def test_pendulum_dynamics(self):
        """Test pendulum dynamics are physically correct."""
        env = Pendulum(self.cfg)
        # Test at stable equilibrium (hanging down)
        state = torch.tensor([0.0, 0.0], dtype=torch.float32)
        next_state = env.step(state)

        # At equilibrium with no velocity, should remain (approximately) at rest
        torch.testing.assert_close(next_state, state, rtol=1e-2, atol=1e-2)

        # Test with small angle and some velocity - should change
        state = torch.tensor([0.1, 0.5], dtype=torch.float32)
        next_state = env.step(state)

        # Angle should change due to velocity
        self.assertNotEqual(next_state[0].item(), state[0].item())
        # Velocity should change due to gravity torque
        self.assertNotEqual(next_state[1].item(), state[1].item())

    def test_pendulum_energy_conservation(self):
        """Test that pendulum approximately conserves energy."""
        env = Pendulum(self.cfg)
        state = torch.tensor([torch.pi / 4, 0.0], dtype=torch.float32)

        def total_energy(s):
            theta, omega = s[0], s[1]
            # E = (1/2) * L^2 * omega^2 + g * L * (1 - cos(theta))
            L = 1.0
            g = 9.81
            kinetic = 0.5 * L**2 * omega**2
            potential = g * L * (1 - torch.cos(theta))
            return kinetic + potential

        E0 = total_energy(state)

        # Simulate for several steps
        for _ in range(100):
            state = env.step(state)

        E1 = total_energy(state)

        # Energy should be approximately conserved (within numerical error)
        # With Euler integration, there will be some drift
        rel_error = abs(E1 - E0) / E0
        self.assertLess(rel_error, 0.1)  # 10% tolerance for Euler method


class TestDuffing(unittest.TestCase):
    """Test Duffing oscillator environment."""

    def setUp(self):
        """Set up Duffing configuration."""
        self.cfg = Config()
        self.cfg.ENV.DUFFING.DT = 0.01

    def test_duffing_initialization(self):
        """Test Duffing creates with correct parameters."""
        env = Duffing(self.cfg)
        self.assertEqual(env.observation_size, 2)
        self.assertEqual(env.action_size, 0)
        self.assertEqual(env.dt, 0.01)

    def test_duffing_reset(self):
        """Test Duffing reset produces valid initial states."""
        env = Duffing(self.cfg)
        rng = torch.Generator()
        rng.manual_seed(42)

        state = env.reset(rng)

        self.assertEqual(state.shape, (2,))
        self.assertEqual(state.dtype, torch.float32)
        # Position should be in [-2, 2]
        self.assertGreaterEqual(state[0].item(), -2.0)
        self.assertLessEqual(state[0].item(), 2.0)
        # Velocity should be in [-1, 1]
        self.assertGreaterEqual(state[1].item(), -1.0)
        self.assertLessEqual(state[1].item(), 1.0)

    def test_duffing_fixed_points(self):
        """Test Duffing fixed points are stable."""
        env = Duffing(self.cfg)

        # Test fixed point at (1, 0)
        state = torch.tensor([1.0, 0.0], dtype=torch.float32)
        next_state = env.step(state)

        # Should remain approximately at fixed point
        torch.testing.assert_close(next_state, state, rtol=1e-2, atol=1e-2)

        # Test fixed point at (-1, 0)
        state = torch.tensor([-1.0, 0.0], dtype=torch.float32)
        next_state = env.step(state)

        torch.testing.assert_close(next_state, state, rtol=1e-2, atol=1e-2)

    def test_duffing_dynamics_correctness(self):
        """Test that Duffing dynamics follow x'' = x - x^3."""
        env = Duffing(self.cfg)
        x = torch.tensor([0.5, 0.0], dtype=torch.float32)

        next_state = env.step(x)

        # Expected next state using RK4 (matches env.step integrator)
        def dynamics_fn(state, action=None):
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = x1 - x1**3
            return torch.stack([dx1, dx2])

        expected = integrate_rk4(x, None, env.dt, dynamics_fn)

        torch.testing.assert_close(next_state, expected, rtol=1e-7, atol=1e-7)


class TestLotkaVolterra(unittest.TestCase):
    """Test Lotka-Volterra environment."""

    def setUp(self):
        """Set up Lotka-Volterra configuration."""
        self.cfg = Config()
        self.cfg.ENV.LOTKA_VOLTERRA.DT = 0.01

    def test_lotka_volterra_initialization(self):
        """Test Lotka-Volterra creates with correct parameters."""
        env = LotkaVolterra(self.cfg)
        self.assertEqual(env.observation_size, 2)
        self.assertEqual(env.action_size, 0)
        self.assertEqual(env.alpha, 0.2)
        self.assertEqual(env.beta, 0.2)
        self.assertEqual(env.gamma, 0.2)
        self.assertEqual(env.delta, 0.2)

    def test_lotka_volterra_reset(self):
        """Test Lotka-Volterra reset produces valid initial states."""
        env = LotkaVolterra(self.cfg)
        rng = torch.Generator()
        rng.manual_seed(42)

        state = env.reset(rng)

        self.assertEqual(state.shape, (2,))
        self.assertEqual(state.dtype, torch.float32)
        # Both populations should be in [0.02, 3.0]
        self.assertGreaterEqual(state[0].item(), 0.02)
        self.assertLessEqual(state[0].item(), 3.0)
        self.assertGreaterEqual(state[1].item(), 0.02)
        self.assertLessEqual(state[1].item(), 3.0)

    def test_lotka_volterra_fixed_point(self):
        """Test Lotka-Volterra fixed point at (1, 1)."""
        env = LotkaVolterra(self.cfg)

        # Fixed point: (gamma/delta, alpha/beta) = (1, 1)
        state = torch.tensor([1.0, 1.0], dtype=torch.float32)
        next_state = env.step(state)

        # Should remain approximately at fixed point
        torch.testing.assert_close(next_state, state, rtol=1e-2, atol=1e-2)

    def test_lotka_volterra_populations_positive(self):
        """Test that populations remain positive during simulation."""
        env = LotkaVolterra(self.cfg)
        rng = torch.Generator()
        rng.manual_seed(42)

        state = env.reset(rng)

        for _ in range(100):
            state = env.step(state)
            # Populations should remain positive
            self.assertGreater(state[0].item(), 0.0)
            self.assertGreater(state[1].item(), 0.0)


class TestLorenz63(unittest.TestCase):
    """Test Lorenz63 environment."""

    def setUp(self):
        """Set up Lorenz63 configuration."""
        self.cfg = Config()
        self.cfg.ENV.LORENZ63.DT = 0.01

    def test_lorenz63_initialization(self):
        """Test Lorenz63 creates with correct parameters."""
        env = Lorenz63(self.cfg)
        self.assertEqual(env.observation_size, 3)
        self.assertEqual(env.action_size, 0)
        self.assertEqual(env.sigma, 10.0)
        self.assertEqual(env.rho, 28.0)
        self.assertAlmostEqual(env.beta, 8.0 / 3.0, places=5)

    def test_lorenz63_reset(self):
        """Test Lorenz63 reset produces valid initial states."""
        env = Lorenz63(self.cfg)
        rng = torch.Generator()
        rng.manual_seed(42)

        state = env.reset(rng)

        self.assertEqual(state.shape, (3,))
        self.assertEqual(state.dtype, torch.float32)
        # Should be perturbation around (0, 1, 1.05)
        # With std=1, expect most samples within ±3 of base point
        self.assertLess(abs(state[0].item()), 5.0)
        self.assertLess(abs(state[1].item() - 1.0), 5.0)
        self.assertLess(abs(state[2].item() - 1.05), 5.0)

    def test_lorenz63_dynamics_correctness(self):
        """Test that Lorenz63 dynamics are computed correctly."""
        env = Lorenz63(self.cfg)
        state = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        next_state = env.step(state)

        # Expected next state using RK4 (matches env.step integrator)
        def dynamics_fn(s, action=None):
            x, y, z = s[0], s[1], s[2]
            dx = env.sigma * (y - x)
            dy = x * (env.rho - z) - y
            dz = x * y - env.beta * z
            return torch.stack([dx, dy, dz])

        expected = integrate_rk4(state, None, env.dt, dynamics_fn)

        torch.testing.assert_close(next_state, expected, rtol=1e-7, atol=1e-7)

    def test_lorenz63_chaotic_behavior(self):
        """Test that nearby trajectories diverge (butterfly effect)."""
        env = Lorenz63(self.cfg)

        state1 = torch.tensor([0.0, 1.0, 1.05], dtype=torch.float32)
        state2 = state1 + torch.tensor([1e-6, 0.0, 0.0], dtype=torch.float32)

        # Evolve both trajectories
        for _ in range(1000):
            state1 = env.step(state1)
            state2 = env.step(state2)

        # Distance should grow significantly (chaotic divergence)
        distance = torch.norm(state1 - state2)
        # Expect at least 10× growth over initial perturbation (1e-6)
        self.assertGreater(distance.item(), 1e-5)


class TestParabolic(unittest.TestCase):
    """Test Parabolic environment."""

    def setUp(self):
        """Set up Parabolic configuration."""
        self.cfg = Config()
        self.cfg.ENV.PARABOLIC.LAMBDA = -1.0
        self.cfg.ENV.PARABOLIC.MU = -0.1
        self.cfg.ENV.PARABOLIC.DT = 0.1

    def test_parabolic_initialization(self):
        """Test Parabolic creates with correct parameters."""
        env = Parabolic(self.cfg)
        self.assertEqual(env.observation_size, 2)
        self.assertEqual(env.action_size, 0)
        self.assertEqual(env.const_lambda, -1.0)
        self.assertEqual(env.const_mu, -0.1)
        self.assertEqual(env.dt, 0.1)

    def test_parabolic_reset(self):
        """Test Parabolic reset produces valid initial states."""
        env = Parabolic(self.cfg)
        rng = torch.Generator()
        rng.manual_seed(42)

        state = env.reset(rng)

        self.assertEqual(state.shape, (2,))
        self.assertEqual(state.dtype, torch.float32)
        # Both components should be in [-1, 1]
        self.assertGreaterEqual(state[0].item(), -1.0)
        self.assertLessEqual(state[0].item(), 1.0)
        self.assertGreaterEqual(state[1].item(), -1.0)
        self.assertLessEqual(state[1].item(), 1.0)

    def test_parabolic_dynamics_correctness(self):
        """Test that Parabolic dynamics are computed correctly."""
        env = Parabolic(self.cfg)
        state = torch.tensor([0.5, 1.0], dtype=torch.float32)

        next_state = env.step(state)

        # Expected next state using RK4 (matches env.step integrator)
        def dynamics_fn(s, action=None):
            x1, x2 = s[0], s[1]
            dx1 = env.const_mu * x1
            dx2 = env.const_lambda * (x2 - x1**2)
            return torch.stack([dx1, dx2])

        expected = integrate_rk4(state, None, env.dt, dynamics_fn)

        torch.testing.assert_close(next_state, expected, rtol=1e-7, atol=1e-7)

    def test_parabolic_fixed_point(self):
        """Test Parabolic fixed point at origin."""
        env = Parabolic(self.cfg)
        state = torch.tensor([0.0, 0.0], dtype=torch.float32)

        next_state = env.step(state)

        # Should remain at origin
        torch.testing.assert_close(next_state, state, rtol=1e-5, atol=1e-5)

    def test_parabolic_manifold_attraction(self):
        """Test that trajectories are attracted to x2 = x1^2 manifold."""
        env = Parabolic(self.cfg)
        state = torch.tensor([0.5, 1.0], dtype=torch.float32)

        # Target manifold value
        target_x2 = state[0]**2  # 0.25

        # Simulate for many steps
        for _ in range(100):
            state = env.step(state)

        # x2 should be closer to x1^2 than initial condition
        final_distance = abs(state[1] - state[0]**2)
        initial_distance = abs(1.0 - 0.25)

        self.assertLess(final_distance.item(), initial_distance)


class TestFactory(unittest.TestCase):
    """Test environment factory function."""

    def setUp(self):
        """Set up test configuration."""
        self.cfg = Config()
        self.cfg.ENV.PENDULUM.DT = 0.01
        self.cfg.ENV.DUFFING.DT = 0.01
        self.cfg.ENV.LOTKA_VOLTERRA.DT = 0.01
        self.cfg.ENV.LORENZ63.DT = 0.01
        self.cfg.ENV.PARABOLIC.LAMBDA = -1.0
        self.cfg.ENV.PARABOLIC.MU = -0.1
        self.cfg.ENV.PARABOLIC.DT = 0.1

    def test_make_env_all_systems(self):
        """Test that make_env creates all registered systems."""
        systems = ["pendulum", "duffing", "lotka_volterra", "lorenz63", "parabolic"]
        expected_classes = [Pendulum, Duffing, LotkaVolterra, Lorenz63, Parabolic]

        for system, expected_cls in zip(systems, expected_classes):
            self.cfg.ENV.ENV_NAME = system
            env = make_env(self.cfg)
            self.assertIsInstance(env, expected_cls)

    def test_make_env_invalid_name(self):
        """Test that make_env raises error for invalid name."""
        self.cfg.ENV.ENV_NAME = "invalid_system"

        with self.assertRaises(ValueError) as context:
            make_env(self.cfg)

        self.assertIn("Unknown environment", str(context.exception))


if __name__ == "__main__":
    unittest.main()

