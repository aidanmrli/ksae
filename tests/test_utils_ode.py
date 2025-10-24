import numpy as np
import torch

from utils import make_time_grid, discretize_linear_ct_matrix_exp


def test_make_time_grid_values():
    grid = make_time_grid(num_steps=5, dt=0.1, start_time=0.5)
    expected = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=torch.float32)
    assert grid.dtype == torch.float32
    assert torch.allclose(grid, expected, atol=1e-7)


def test_discretize_linear_ct_no_input_matches_matrix_exp():
    A = torch.diag(torch.tensor([-0.5, -1.0], dtype=torch.float64))
    dt = 0.1
    Phi_expected = torch.linalg.matrix_exp(A * dt)
    Phi, Gamma = discretize_linear_ct_matrix_exp(A.to(torch.float64), None, dt)
    assert Gamma is None
    assert torch.allclose(Phi, Phi_expected, rtol=1e-10, atol=1e-12)


def test_discretize_linear_ct_with_input_zero_A_reduces_to_dtB():
    A = torch.zeros((3, 3), dtype=torch.float64)
    B = torch.randn(3, 2, dtype=torch.float64)
    dt = 0.2
    Phi, Gamma = discretize_linear_ct_matrix_exp(A, B, dt)
    assert Phi.shape == (3, 3)
    assert Gamma is not None and Gamma.shape == (3, 2)
    # With A=0, exp(A t) = I, so Γ = ∫_0^dt B ds = dt B
    assert torch.allclose(Gamma, dt * B, rtol=1e-10, atol=1e-12)


