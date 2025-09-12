import torch
import pytest

from ksae import soft_threshold


def test_soft_threshold_vector_correctness():
    v = torch.tensor([0.5, -1.2, 0.0, 3.0, -0.05], dtype=torch.float32)
    th = torch.tensor([0.2, 0.8, 0.0, 0.1, 0.1], dtype=torch.float32)
    expected = torch.sign(v) * torch.relu(torch.abs(v) - th)
    out = soft_threshold(v, th)
    assert torch.allclose(out, expected, atol=0.0, rtol=0.0)


def test_soft_threshold_broadcast_batchwise():
    values = torch.tensor(
        [[1.0, -0.5, 0.2, -2.0],
         [0.0, 0.5, -1.5, 3.0]],
        dtype=torch.float32,
    )
    thresholds = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    expected = torch.sign(values) * torch.relu(torch.abs(values) - thresholds)
    out = soft_threshold(values, thresholds)
    assert torch.allclose(out, expected)


def test_soft_threshold_zero_threshold_is_identity():
    v = torch.linspace(-2, 2, steps=9)
    th = torch.zeros(9)
    out = soft_threshold(v, th)
    assert torch.allclose(out, v)


def test_soft_threshold_large_threshold_yields_zero():
    v = torch.tensor([1.0, -2.0, 0.3], dtype=torch.float32)
    th = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    out = soft_threshold(v, th)
    assert torch.count_nonzero(out) == 0


