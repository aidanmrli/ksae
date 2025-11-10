"""Tests for the Env interface.

Tests Env, Wrapper, VectorWrapper, and trajectory generation functions.
"""

import pytest
import torch
import numpy as np

from config import Config
from data import (
    Env,
    Wrapper,
    VectorWrapper,
    make_env,
    generate_trajectory,
)


def test_env_creation():
    """Test basic Env creation and properties."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    
    assert env.observation_size == 2
    assert env.action_size == 0
    assert env.unwrapped is env


def test_env_reset():
    """Test environment reset generates valid initial states."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    
    # Reset with generator seed
    rng1 = torch.Generator()
    rng1.manual_seed(42)
    state1 = env.reset(rng1)
    assert isinstance(state1, torch.Tensor)
    assert state1.shape == (2,)
    assert torch.all(torch.isfinite(state1))
    
    # Reset with same seed should give same state
    rng2 = torch.Generator()
    rng2.manual_seed(42)
    state2 = env.reset(rng2)
    assert torch.allclose(state1, state2)
    
    # Reset with different seed should give different state
    rng3 = torch.Generator()
    rng3.manual_seed(123)
    state3 = env.reset(rng3)
    assert not torch.allclose(state1, state3)


def test_env_step():
    """Test environment step advances dynamics correctly."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    rng = torch.Generator()
    rng.manual_seed(42)
    state = env.reset(rng)
    
    # Take a step
    next_state = env.step(state)
    assert isinstance(next_state, torch.Tensor)
    assert next_state.shape == state.shape
    assert torch.all(torch.isfinite(next_state))
    
    # State should change (not stuck)
    assert not torch.allclose(state, next_state)


def test_env_step_deterministic():
    """Test that stepping is deterministic."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    rng = torch.Generator()
    rng.manual_seed(42)
    state = env.reset(rng)
    
    next1 = env.step(state)
    next2 = env.step(state)
    
    assert torch.allclose(next1, next2)


def test_wrapper_delegation():
    """Test that Wrapper correctly delegates to wrapped env."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "pendulum"
    cfg.ENV.PENDULUM.DT = 0.01
    base_env = make_env(cfg)
    wrapper = Wrapper(base_env)
    
    # Check delegation
    assert wrapper.observation_size == base_env.observation_size
    assert wrapper.action_size == base_env.action_size
    assert wrapper.unwrapped is base_env
    
    # Check reset delegation
    rng = torch.Generator()
    rng.manual_seed(42)
    state1 = base_env.reset(rng)
    rng2 = torch.Generator()
    rng2.manual_seed(42)
    state2 = wrapper.reset(rng2)
    assert torch.allclose(state1, state2)
    
    # Check step delegation
    next1 = base_env.step(state1)
    next2 = wrapper.step(state2)
    assert torch.allclose(next1, next2)


def test_vector_wrapper_reset():
    """Test VectorWrapper generates batches of initial states."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    batch_size = 32
    vec_env = VectorWrapper(env, batch_size)
    
    rng = torch.Generator()
    rng.manual_seed(42)
    states = vec_env.reset(rng)
    
    # Check shape
    assert isinstance(states, torch.Tensor)
    assert states.shape == (batch_size, 2)
    assert torch.all(torch.isfinite(states))
    
    # Check diversity (not all the same)
    assert not torch.allclose(states[0], states[1])


def test_vector_wrapper_step():
    """Test VectorWrapper applies step to batches."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    batch_size = 16
    vec_env = VectorWrapper(env, batch_size)
    
    rng = torch.Generator()
    rng.manual_seed(42)
    states = vec_env.reset(rng)
    next_states = vec_env.step(states)
    
    # Check output shape
    assert next_states.shape == states.shape
    assert torch.all(torch.isfinite(next_states))
    
    # Check that states evolved
    assert not torch.allclose(states, next_states)
    
    # Verify vectorization matches individual steps
    individual_next = env.step(states[0])
    assert torch.allclose(next_states[0], individual_next, atol=1e-5)


def test_generate_trajectory_single():
    """Test trajectory generation for single initial state."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    rng = torch.Generator()
    rng.manual_seed(42)
    init_state = env.reset(rng)
    length = 50
    
    traj = generate_trajectory(env.step, init_state, length=length)
    
    # Check shape: (length, state_dim)
    assert traj.shape == (length, 2)
    assert torch.all(torch.isfinite(traj))
    
    # Verify trajectory evolves
    assert not torch.allclose(traj[0], traj[-1])


def test_generate_trajectory_batch():
    """Test trajectory generation for batch of initial states."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    batch_size = 8
    vec_env = VectorWrapper(env, batch_size)
    
    rng = torch.Generator()
    rng.manual_seed(42)
    init_states = vec_env.reset(rng)
    length = 50
    
    traj = generate_trajectory(vec_env.step, init_states, length=length)
    
    # Check shape: (length, batch, state_dim)
    assert traj.shape == (length, batch_size, 2)
    assert torch.all(torch.isfinite(traj))


def test_generate_trajectory_reproducibility():
    """Test that trajectory generation is reproducible."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "pendulum"
    cfg.ENV.PENDULUM.DT = 0.01
    env = make_env(cfg)
    rng = torch.Generator()
    rng.manual_seed(123)
    init_state = env.reset(rng)
    
    traj1 = generate_trajectory(env.step, init_state, length=30)
    traj2 = generate_trajectory(env.step, init_state, length=30)
    
    assert torch.allclose(traj1, traj2)


def test_make_env_registry():
    """Test make_env function with registered systems."""
    # Test all registered systems
    systems = ["duffing", "pendulum", "lotka_volterra", "lorenz63", "parabolic"]
    
    for name in systems:
        cfg = Config()
        cfg.ENV.ENV_NAME = name
        if name == "duffing":
            cfg.ENV.DUFFING.DT = 0.01
        elif name == "pendulum":
            cfg.ENV.PENDULUM.DT = 0.01
        elif name == "lotka_volterra":
            cfg.ENV.LOTKA_VOLTERRA.DT = 0.01
        elif name == "lorenz63":
            cfg.ENV.LORENZ63.DT = 0.01
        elif name == "parabolic":
            cfg.ENV.PARABOLIC.DT = 0.01
        
        env = make_env(cfg)
        assert isinstance(env, Env)
        assert env.observation_size > 0
        
        # Verify reset and step work
        rng = torch.Generator()
        rng.manual_seed(42)
        state = env.reset(rng)
        next_state = env.step(state)
        assert next_state.shape == state.shape


def test_make_env_invalid_system():
    """Test make_env raises error for unknown system."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "nonexistent_system"
    with pytest.raises(ValueError, match="Unknown environment"):
        make_env(cfg)


def test_vector_wrapper_properties():
    """Test VectorWrapper preserves environment properties."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "lotka_volterra"
    cfg.ENV.LOTKA_VOLTERRA.DT = 0.01
    env = make_env(cfg)
    vec_env = VectorWrapper(env, 10)
    
    assert vec_env.observation_size == env.observation_size
    assert vec_env.action_size == env.action_size
    assert vec_env.unwrapped is env


def test_trajectory_length_consistency():
    """Test that generated trajectories have requested length."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.01
    env = make_env(cfg)
    rng = torch.Generator()
    rng.manual_seed(42)
    init_state = env.reset(rng)
    
    for length in [1, 10, 100, 500]:
        traj = generate_trajectory(env.step, init_state, length=length)
        assert traj.shape[0] == length


def test_env_batch_consistency():
    """Test that batch and individual stepping give consistent results."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "pendulum"
    cfg.ENV.PENDULUM.DT = 0.01
    env = make_env(cfg)
    vec_env = VectorWrapper(env, 1)
    
    # Single state
    rng1 = torch.Generator()
    rng1.manual_seed(42)
    state_single = env.reset(rng1)
    next_single = env.step(state_single)
    
    # Batch of 1
    rng2 = torch.Generator()
    rng2.manual_seed(42)
    state_batch = vec_env.reset(rng2)
    next_batch = vec_env.step(state_batch)
    
    # Results should match
    assert torch.allclose(next_single, next_batch[0], atol=1e-5)

