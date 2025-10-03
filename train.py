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


def _current_run_dir(base: str, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_dir(Path(base) / tag / timestamp)


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
    config = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in vars(args).items()}
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
    )

    if model_type == "kae":
        model = KoopmanAE(
            input_dim=train_loader.dataset.dataset.spec.state_dim,
            latent_dim=args.latent_dim,
            encoder_hidden=args.encoder_hidden,
            decoder_hidden=args.decoder_hidden,
            control_dim=train_loader.dataset.dataset.spec.control_dim,
        )
        sparsity_weight = 0.0
    elif model_type == "ksae":
        model = KSAE(
            input_dim=train_loader.dataset.dataset.spec.state_dim,
            latent_dim=args.latent_dim,
            lista_iterations=args.lista_T,
            decoder_hidden=args.decoder_hidden,
            control_dim=train_loader.dataset.dataset.spec.control_dim,
        )
        sparsity_weight = args.lambda_sparse
    else:
        raise ValueError(f"Unknown model type '{model_type}'")

    model.to(device)

    param_groups = []
    koopman_params = [model.K]
    if getattr(model, "L", None) is not None:
        koopman_params.append(model.L)
    param_groups.append({"params": [p for p in koopman_params if p is not None], "lr": args.lr_koopman})

    decoder_params = list(model.decoder.parameters())
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": args.lr_main})

    if isinstance(model, KoopmanAE):
        encoder_params = list(model.encoder.parameters())
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_main})
    else:
        encoder_params = list(model.encoder.parameters())
        for param in encoder_params:
            param.requires_grad = args.freeze_lista_epochs == 0
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_lista})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    weights = KoopmanLossWeights(
        reconstruction=args.lambda_recon,
        alignment=args.lambda_align,
        prediction=args.lambda_pred,
        sparsity=sparsity_weight,
        frobenius=args.lambda_frob,
    )

    run_dir = _current_run_dir(args.output_dir, model_type)
    config = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in vars(args).items()}
    save_json(config, run_dir / "config.json")
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        if model_type == "ksae" and args.freeze_lista_epochs and epoch > args.freeze_lista_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = True

        train_loss = _train_koopman_epoch(model, train_loader, optimizer, device, weights, args)
        val_metrics = evaluate_koopman(model, val_loader, device, args.eval_rollout, args.reencode_period)
        val_loss = val_metrics["prediction_mse"]
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        logging.info(
            "[%s][epoch %d] train %.4f | val pred %.4f | recon %.4f | align %.4f",
            model_type.upper(),
            epoch,
            train_loss,
            val_metrics["prediction_mse"],
            val_metrics["reconstruction_mse"],
            val_metrics["alignment_mse"],
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, run_dir / "checkpoint.pt")

    save_json({"history": history}, run_dir / "metrics.json")
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, run_dir / "last.pt")
    return run_dir


def _train_koopman_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: KoopmanLossWeights,
    args: Namespace,
) -> float:
    model.train()
    meter = AverageMeter()
    for batch in loader:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch["x"], batch.get("u"))
        total_loss, _ = compute_koopman_losses(outputs, batch, model, weights)
        total_loss.backward()
        if getattr(args, "grad_clip", None):
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        meter.update(total_loss.item(), batch["x"].size(0))
    return meter.average


# Delayed import to avoid circular dependency when the module is loaded.
from eval import evaluate_koopman, evaluate_lista  # noqa: E402
