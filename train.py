"""Training loops for LISTA, Koopman AE, and KSAE."""

from __future__ import annotations

import logging
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data import create_dynamics_dataloaders, create_lista_datasets
from losses import KoopmanLossWeights, compute_koopman_losses, lista_reconstruction_loss, lista_supervised_loss
from models import KSAE, LISTA, KoopmanAE
from utils import AverageMeter, configure_logging, ensure_dir, resolve_device, save_json, set_seed
from plotting import save_phase_portraits_overlay


def _current_run_dir(base: str, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(Path(base) / tag / timestamp)


def _resolve_dynamics_spec(loader: DataLoader):
    """Return the underlying DynamicalSystemSpec from a potentially wrapped dataset.

    Supports the following dataset wrappers:
    - torch.utils.data.Subset (via `.dataset`)
    - WindowedSequenceDataset (via `._inner`)
    - Direct DynamicalSystemDataset exposing `.spec`
    """
    dataset = loader.dataset
    # Unwrap WindowedSequenceDataset if present
    if hasattr(dataset, "_inner"):
        dataset = dataset._inner  # type: ignore[attr-defined]
    # Unwrap Subset if present
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset  # type: ignore[attr-defined]
    spec = getattr(dataset, "spec", None)
    if spec is None:
        raise AttributeError("Could not resolve dynamics spec from dataset")
    return spec


def train_lista(args: Namespace) -> Path:
    """Train LISTA to approximate sparse codes generated synthetically."""
    configure_logging()
    set_seed(args.seed)
    device = resolve_device(getattr(args, "device", None))

    train_loader, val_loader, test_set = create_lista_datasets(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        dict_dim=args.dict_dim,
        batch_size=args.batch_size,
        seed=args.seed,
        sparsity=args.sparsity,
        noise_std=args.noise_std,
    )

    model = LISTA(dict_dim=args.dict_dim, input_dim=args.input_dim, iterations=args.T)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = _current_run_dir(args.output_dir, "lista")
    config = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in vars(args).items() if k != "func"}
    save_json(config, run_dir / "config.json")
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_lista_epoch(model, train_loader, optimizer, device, args)
        val_metrics = evaluate_lista(model, val_loader, device)
        val_loss = val_metrics["code_mse"]
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        logging.info(
            "[LISTA][epoch %d] train %.4f | val code MSE %.4f | recon %.4f | sparsity %.4f",
            epoch,
            train_loss,
            val_metrics["code_mse"],
            val_metrics["reconstruction_mse"],
            val_metrics["sparsity"],
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, run_dir / "checkpoint.pt")

    save_json({"history": history}, run_dir / "metrics.json")
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, run_dir / "last.pt")
    return run_dir


def _train_lista_epoch(
    model: LISTA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: Namespace,
) -> float:
    model.train()
    meter = AverageMeter()
    for batch in loader:
        x = batch["x"].to(device)
        z_star = batch["z_star"].to(device)
        dictionary = batch["dictionary"].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = lista_supervised_loss(pred, z_star)
        if getattr(args, "lambda_recon", 0.0) > 0:
            loss = loss + args.lambda_recon * lista_reconstruction_loss(x, pred, dictionary)
        loss.backward()
        if getattr(args, "grad_clip", None):
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        meter.update(loss.item(), x.size(0))
    return meter.average


def train_kae(args: Namespace) -> Path:
    """Train a Koopman autoencoder on a chosen dynamical system."""
    return _train_koopman_common(args, model_type="kae")


def train_ksae(args: Namespace) -> Path:
    """Train Koopman Sparse Autoencoder with LISTA encoder."""
    return _train_koopman_common(args, model_type="ksae")


def _train_koopman_common(args: Namespace, model_type: str) -> Path:
    configure_logging()
    set_seed(args.seed)
    device = resolve_device(getattr(args, "device", None))

    train_loader, val_loader, test_loader = create_dynamics_dataloaders(
        system=args.system,
        num_samples=args.num_samples,
        seq_len=args.sequence_length,
        batch_size=args.batch_size,
        dt=args.dt,
        noise_std=args.noise_std,
        seed=args.seed,
        train_context_length=getattr(args, "context_length", None),
        use_offline_cache=getattr(args, "use_offline_cache", False),
        cache_dir=getattr(args, "cache_dir", "data"),
        ode_rtol=getattr(args, "ode_rtol", 1e-5),
        ode_atol=getattr(args, "ode_atol", 1e-7),
    )

    spec = _resolve_dynamics_spec(train_loader)

    # If training is windowed (context-length), emit a clear warning so users
    # understand that optimization happens on short windows rather than the
    # full simulated trajectory length.
    try:
        dataset_for_warning = train_loader.dataset
        # Unwrap potential Subset to access .dataset/.trajectories where possible
        inner_for_warning = getattr(dataset_for_warning, "_inner", getattr(dataset_for_warning, "dataset", dataset_for_warning))
        full_len = getattr(getattr(inner_for_warning, "trajectories", None), "shape", [None, None])[1]
        horizon_attr = getattr(dataset_for_warning, "horizon", None)
        if horizon_attr is not None and isinstance(full_len, int) and (horizon_attr + 1) < full_len:
            logging.warning(
                "Training with short windows: context-length=%d (uses %d states) while full trajectory has %d states. "
                "Remove --context-length to train on full sequences.",
                int(horizon_attr), int(horizon_attr + 1), int(full_len),
            )
    except Exception:
        # Best-effort warning; do not fail if structure differs
        pass

    if model_type == "kae":
        model = KoopmanAE(
            input_dim=spec.state_dim,
            latent_dim=args.latent_dim,
            encoder_hidden=args.encoder_hidden,
            decoder_hidden=args.decoder_hidden,
            control_dim=spec.control_dim,
            koopman_continuous=(args.koopman_mode == "continuous"),
            dt=args.dt,
            control_discretization=args.control_discretization,
            latent_mode=args.latent_mode,
            gamma_method=args.gamma_method,
            action_encoder_layers=(getattr(args, "action_encoder_layers", None) or getattr(args, "action_encoder_hidden", None)),
        )
        sparsity_weight = args.lambda_sparse
    elif model_type == "ksae":
        model = KSAE(
            input_dim=spec.state_dim,
            latent_dim=args.latent_dim,
            lista_iterations=args.lista_T,
            decoder_hidden=args.decoder_hidden,
            control_dim=spec.control_dim,
            koopman_continuous=(args.koopman_mode == "continuous"),
            dt=args.dt,
            control_discretization=args.control_discretization,
            latent_mode=args.latent_mode,
            gamma_method=args.gamma_method,
            action_encoder_layers=(getattr(args, "action_encoder_layers", None) or getattr(args, "action_encoder_hidden", None)),
        )
        sparsity_weight = args.lambda_sparse
    else:
        raise ValueError(f"Unknown model type '{model_type}'")

    model.to(device)

    param_groups = []
    # Dynamics parameter group (continuous: A/B; discrete: K/L) + learned delta
    koopman_params = []
    if hasattr(model, "A"):
        koopman_params.append(model.A)
        if getattr(model, "B", None) is not None:
            koopman_params.append(model.B)
    else:
        koopman_params.append(model.K)
        if getattr(model, "L", None) is not None:
            koopman_params.append(model.L)
    # Include learned time-step log-parameter δ_log
    if hasattr(model, "delta_log"):
        koopman_params.append(model.delta_log)
    param_groups.append({"params": [p for p in koopman_params if p is not None], "lr": args.lr_koopman})

    decoder_params = list(model.state_decoder.parameters())
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": args.lr_main})

    if isinstance(model, KoopmanAE):
        encoder_params = list(model.state_encoder.parameters())
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_main})
    else:
        encoder_params = list(model.state_encoder.parameters())
        for param in encoder_params:
            param.requires_grad = args.freeze_lista_epochs == 0
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_lista})

    # Action encoder parameters (if present)
    action_enc = getattr(model, "action_encoder", None)
    if action_enc is not None:
        action_params = list(action_enc.parameters())
        if action_params:
            param_groups.append({"params": action_params, "lr": args.lr_main})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, lr=args.lr_main)

    weights = KoopmanLossWeights(
        reconstruction=args.lambda_recon,
        alignment=args.lambda_align,
        prediction=args.lambda_pred,
        sparsity=sparsity_weight,
        frobenius=args.lambda_frob,
    )

    run_dir = _current_run_dir(args.output_dir, model_type)
    config = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in vars(args).items() if k != "func"}
    save_json(config, run_dir / "config.json")
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        if model_type == "ksae" and args.freeze_lista_epochs and epoch > args.freeze_lista_epochs:
            for param in model.state_encoder.parameters():
                param.requires_grad = True

        train_loss, train_components = _train_koopman_epoch(model, train_loader, optimizer, device, weights, args)
        val_metrics = evaluate_koopman(model, val_loader, device, args.val_rollout_steps, args.inference_reencode_period)
        val_loss = val_metrics["prediction_mse"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_alignment": float(train_components.get("alignment", 0.0)),
            "train_reconstruction": float(train_components.get("reconstruction", 0.0)),
            "train_prediction": float(train_components.get("prediction", 0.0)),
            **val_metrics,
        })

        logging.info(
            "[%s][epoch %d] train total %.5f | train align %.5f | train recon %.5f | train pred %.5f | val pred %.5f",
            model_type.upper(),
            epoch,
            train_loss,
            train_components.get("alignment", 0.0),
            train_components.get("reconstruction", 0.0),
            train_components.get("prediction", 0.0),
            val_metrics["prediction_mse"],
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, run_dir / "checkpoint.pt")

    save_json({"history": history}, run_dir / "metrics.json")
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, run_dir / "last.pt")

    # After training finishes, generate training data phase portraits overlay (for Koopman models)
    try:
        model.eval()
        plot_dir = ensure_dir(Path("plots"))
        agg_true_list = []
        agg_pred_list = []
        overlay_cap = 100  # limit number of trajectories to overlay
        with torch.no_grad():
            # Prefer plotting from the full underlying trajectories (not the windowed
            # training samples) so the overlay reflects long-horizon behaviour.
            base_ds = train_loader.dataset
            # Unwrap WindowedSequenceDataset -> inner DynamicalSystemDataset
            if hasattr(base_ds, "_inner"):
                base_ds = base_ds._inner  # type: ignore[attr-defined]
            if hasattr(base_ds, "dataset"):
                base_ds = base_ds.dataset  # type: ignore[attr-defined]
            full_trajs = getattr(base_ds, "trajectories", None)
            full_ctrls = getattr(base_ds, "controls", None)

            if isinstance(full_trajs, torch.Tensor) and full_trajs.dim() == 3:
                num_traj = int(full_trajs.shape[0])
                full_len = int(full_trajs.shape[1])
                # Rollout horizon limited by requested steps and available sequence
                h = max(1, min(int(args.val_rollout_steps), full_len - 1))
                for idx in range(min(overlay_cap, num_traj)):
                    x_full = full_trajs[idx : idx + 1].to(device)
                    ctrls = None
                    if isinstance(full_ctrls, torch.Tensor) and full_ctrls.dim() == 3:
                        ctrls = full_ctrls[idx : idx + 1, :h].to(device)
                    rollout = model.rollout(
                        x_full[:, 0],
                        h,
                        reencode_period=getattr(args, "inference_reencode_period", 0),
                        controls=ctrls,
                    )
                    agg_true_list.append(x_full[0, : h + 1].detach().cpu())
                    agg_pred_list.append(rollout[0].detach().cpu())
            else:
                # Fallback: use whatever the training loader provided (may be short windows)
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    x = batch["x"]
                    horizon = min(int(args.val_rollout_steps), x.shape[1] - 1)
                    if horizon <= 0:
                        continue
                    controls = batch.get("u")
                    if controls is not None:
                        controls = controls[:, :horizon]
                    rollout = model.rollout(
                        x[:, 0],
                        horizon,
                        reencode_period=getattr(args, "inference_reencode_period", 0),
                        controls=controls,
                    )
                    remaining = overlay_cap - len(agg_true_list)
                    if remaining <= 0:
                        break
                    take = min(remaining, x.size(0))
                    agg_true_list.extend(x[:take, : horizon + 1].detach().cpu())
                    agg_pred_list.extend(rollout[:take].detach().cpu())
        if len(agg_true_list) > 0:
            save_phase_portraits_overlay(agg_true_list, agg_pred_list, plot_dir / "train_phase_portraits.png")
    except Exception:
        # Do not fail training completion due to plotting issues
        pass
    return run_dir


def _train_koopman_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: KoopmanLossWeights,
    args: Namespace,
) -> tuple[float, Dict[str, float]]:
    model.train()
    meter = AverageMeter()
    align_meter = AverageMeter()
    recon_meter = AverageMeter()
    pred_meter = AverageMeter()
    for batch in loader:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch["x"], batch.get("u"))
        total_loss, components = compute_koopman_losses(outputs, batch, model, weights)
        batch_size = batch["x"].size(0)
        # Track weighted alignment and reconstruction terms
        if "alignment" in components:
            align_meter.update((weights.alignment * components["alignment"]).item(), batch_size)
        if "reconstruction" in components:
            recon_meter.update((weights.reconstruction * components["reconstruction"]).item(), batch_size)
        # Research-paper training: open-loop multi-step prediction loss from a single initial encoding.
        # In the state-prediction setup (no controls), encode x0 once, integrate forward in latent
        # space without re-encoding, decode at each step, and sum/average the prediction error
        # over the rollout window. This replaces the teacher-forced one-step prediction loss.
        if weights.prediction > 0:
            seq_horizon = batch["x"].shape[1] - 1
            if seq_horizon > 0:
                # Ignore controls for the state-prediction setup; fall back to provided controls otherwise
                controls = None
                if getattr(model, "control_dim", 0) > 0:
                    controls = batch.get("u")
                    if controls is not None:
                        controls = controls[:, :seq_horizon]
                rollout = model.rollout(
                    batch["x"][:, 0],
                    horizon=seq_horizon,
                    reencode_period=None,  # pure open-loop; no periodic re-encoding during training
                    controls=controls,
                )
                target = batch["x"][:, 1 : seq_horizon + 1]
                rollout_mse = torch.mean((rollout - target) ** 2)
                # Replace teacher-forced prediction term (if it was included) with open-loop loss
                if "prediction" in components:
                    total_loss = total_loss - weights.prediction * components["prediction"]
                total_loss = total_loss + weights.prediction * rollout_mse
                pred_meter.update((weights.prediction * rollout_mse).item(), batch_size)
            else:
                if "prediction" in components:
                    pred_meter.update((weights.prediction * components["prediction"]).item(), batch_size)
        total_loss.backward()
        if getattr(args, "grad_clip", None):
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        _normalize_decoder_columns(model, eps=getattr(args, "column_norm_eps", 1e-8))
        meter.update(total_loss.item(), batch["x"].size(0))
    return meter.average, {
        "alignment": align_meter.average,
        "reconstruction": recon_meter.average,
        "prediction": pred_meter.average,
    }


# Delayed import to avoid circular dependency when the module is loaded.
from eval import evaluate_koopman, evaluate_lista  # noqa: E402


def _normalize_decoder_columns(model: torch.nn.Module, eps: float = 1e-8) -> None:
    """Normalize the columns of the first Linear layer in the decoder.

    This discourages the encoder from shrinking latent codes to minimize alignment loss.
    
    This should be done after each optimizer step.
    """
    decoder = getattr(model, "state_decoder", None)
    if decoder is None:
        return
    # Support Sequential decoders
    first_linear = None
    if isinstance(decoder, torch.nn.Sequential):
        for module in decoder:
            if isinstance(module, torch.nn.Linear):
                first_linear = module
                break
    elif isinstance(decoder, torch.nn.Linear):
        first_linear = decoder
    if first_linear is None:
        return
    with torch.no_grad():
        weight = first_linear.weight  # (out_features, in_features)
        norms = torch.linalg.norm(weight, dim=0, keepdim=True)
        norms = torch.clamp(norms, min=eps)
        weight.div_(norms)
