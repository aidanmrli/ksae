"""Unit tests for the evaluation utilities."""

import numpy as np
import torch

from config import get_config
from data import make_env
from evaluation import (
    EvaluationSettings,
    evaluate_model,
    rollout_every_step_reencode,
    rollout_no_reencode,
    rollout_periodic_reencode,
    _estimate_learned_attractors,
    _save_lyapunov_phase_portrait_comparison,
)
from model import make_model


def _build_model_and_states(system: str, batch_size: int = 4) -> tuple:
    cfg = get_config("generic")
    cfg.ENV.ENV_NAME = system
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.eval()

    rng = torch.Generator().manual_seed(0)
    states = torch.randn(batch_size, env.observation_size, generator=rng)
    return model, states, cfg


def test_rollout_modes_shape():
    model, states, _ = _build_model_and_states("pendulum")

    horizon = 8
    no_re = rollout_no_reencode(model, states, horizon)
    every = rollout_every_step_reencode(model, states, horizon)
    periodic = rollout_periodic_reencode(model, states, horizon, period=2)

    assert no_re.shape == (horizon, states.shape[0], states.shape[1])
    assert every.shape == (horizon, states.shape[0], states.shape[1])
    assert periodic.shape == (horizon, states.shape[0], states.shape[1])


def test_evaluate_model_generates_outputs(tmp_path):
    model, _, cfg = _build_model_and_states("duffing")

    settings = EvaluationSettings(
        systems=("duffing",),
        horizons=(10,),
        periodic_reencode_periods=(2,),
        batch_size=4,
        phase_portrait_samples=2,
    )

    results = evaluate_model(
        model=model,
        cfg=cfg,
        device="cpu",
        settings=settings,
        output_dir=tmp_path,
    )

    assert "duffing" in results
    duffing_metrics = results["duffing"]
    assert "modes" in duffing_metrics
    assert "no_reencode" in duffing_metrics["modes"]
    assert "best_periodic" in duffing_metrics
    assert "10" in duffing_metrics["best_periodic"]

    metrics_json = tmp_path / "metrics.json"
    assert metrics_json.exists(), "Evaluation should write metrics.json"

    curve_png = tmp_path / "duffing" / "mse_vs_horizon.png"
    assert curve_png.exists(), "Evaluation should write MSE curve plot"


def test_estimate_learned_attractors_contracts_to_origin():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("bias", torch.nn.Parameter(torch.zeros(1)))

        def step_env(self, state: torch.Tensor) -> torch.Tensor:
            return state * 0.5

    model = DummyModel()
    attractors = _estimate_learned_attractors(
        model=model,
        grid_lim=1.0,
        num_samples=16,
        num_steps=6,
        tolerance=1e-3,
        device=torch.device("cpu"),
    )

    assert attractors.shape[0] == 1
    assert np.allclose(attractors[0], 0.0, atol=1e-2)


def test_lyapunov_phase_portrait_outputs(tmp_path):
    model, _, cfg = _build_model_and_states("lyapunov")
    env = make_env(cfg)

    comp_path = tmp_path / "comparison.png"
    files = _save_lyapunov_phase_portrait_comparison(
        model=model,
        env=env,
        path=comp_path,
        num_trajectories=1,
        grid_lim=0.75,
        grid_n=5,
    )

    assert comp_path.exists()
    true_hist = tmp_path / "phase_portrait_vector_hist_true.png"
    learned_hist = tmp_path / "phase_portrait_vector_hist_learned.png"
    assert true_hist.exists()
    assert learned_hist.exists()
    assert "phase_portrait_comparison" in files

