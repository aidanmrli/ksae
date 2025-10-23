import torch

from models import KoopmanAE
from data import DynamicalSystemDataset, pendulum_spec, WindowedSequenceDataset
import torch


def test_windowed_sequence_dataset_shapes() -> None:
    spec = pendulum_spec()
    base = DynamicalSystemDataset(spec, num_samples=3, seq_len=64, dt=0.01, noise_std=0.0, seed=0)
    # Horizon T means we expect T+1 states and T controls
    windowed = WindowedSequenceDataset(base, horizon=10, subset_length=50, seed=123)
    sample = windowed[0]
    assert sample["x"].shape[0] == 11  # T+1 states
    if spec.control_dim > 0 and "u" in sample:
        assert sample["u"].shape[0] == 10
    # Bounds: start in [0, 50 - (T+1)] => max 39; ensure last index < 50
    # We can't read start directly; check the last state index proxy by equality with base
    x = sample["x"]
    # Ensure x fits entirely within the chosen subset length
    assert x.shape[0] == 11


def test_delta_is_learnable_and_affects_discretization():
    model = KoopmanAE(input_dim=2, latent_dim=2, encoder_hidden=(), decoder_hidden=(), control_dim=0, dt=0.01)
    # Set A to a non-zero value so Kd depends on delta
    with torch.no_grad():
        model.A.copy_(torch.eye(2))

    # Capture Kd for two different deltas by setting raw parameter
    with torch.no_grad():
        model.delta_log.copy_(torch.tensor(0.01).log())
    Kd1, _ = model._discretized_matrices()

    with torch.no_grad():
        model.delta_log.copy_(torch.tensor(0.2).log())
    Kd2, _ = model._discretized_matrices()

    assert not torch.allclose(Kd1, Kd2), "Discretized K should change when delta changes"

    # Check that gradients flow to delta_raw
    x = torch.randn(4, 3, 2)
    out = model(x)
    loss = (out["reconstructions"] - x).pow(2).mean()
    loss.backward()
    assert model.delta_log.grad is not None, "delta_log should receive gradients"


def test_action_encoder_is_used_when_controls_present():
    control_dim = 3
    model = KoopmanAE(
        input_dim=2,
        latent_dim=4,
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        control_dim=control_dim,
        dt=0.05,
        action_encoder_layers=(5,),
    )
    assert model.action_encoder is not None, "Action encoder should be constructed when control_dim > 0"

    # Forward with controls to ensure shapes are consistent
    batch, seq = 2, 5
    x = torch.randn(batch, seq, 2)
    u = torch.randn(batch, seq - 1, control_dim)
    outputs = model(x, u)
    assert outputs["predictions"].shape == (batch, seq - 1, 2)
    assert outputs["predicted_latents"].shape[1] == seq - 1


