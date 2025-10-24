import numpy as np
from pathlib import Path

from data import pendulum_spec, load_or_generate_cached


def test_offline_cache_roundtrip(tmp_path: Path):
    spec = pendulum_spec()
    num_samples, seq_len, dt, noise_std, seed = 4, 50, 0.01, 0.0, 123

    trajs1, ctrls1 = load_or_generate_cached(
        spec,
        num_samples=num_samples,
        seq_len=seq_len,
        dt=dt,
        noise_std=noise_std,
        seed=seed,
        cache_dir=tmp_path,
    )
    assert trajs1.shape == (num_samples, seq_len, spec.state_dim)
    assert ctrls1 is None

    # Second load reads from disk, must match
    trajs2, ctrls2 = load_or_generate_cached(
        spec,
        num_samples=num_samples,
        seq_len=seq_len,
        dt=dt,
        noise_std=noise_std,
        seed=seed,
        cache_dir=tmp_path,
    )
    assert np.allclose(trajs1, trajs2)
    assert ctrls2 is None


