"""Command-line interface for training and evaluating LISTA, KAE, and KSAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data import create_dynamics_dataloaders, create_lista_datasets
from eval import evaluate_koopman, evaluate_lista
from models import KSAE, LISTA, KoopmanAE
from train import train_kae, train_lista, train_ksae
from utils import resolve_device, set_seed


def _parse_hidden(values: Optional[list[int]]) -> Optional[tuple[int, ...]]:
    if values is None:
        return None
    if len(values) == 0:
        return ()
    return tuple(int(v) for v in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koopman Sparse Autoencoder experiments")
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # Train LISTA
    # ------------------------------------------------------------------
    train_lista_parser = subparsers.add_parser("train-lista", help="Train a LISTA encoder")
    train_lista_parser.add_argument("--input-dim", type=int, default=100)
    train_lista_parser.add_argument("--dict-dim", type=int, default=400)
    train_lista_parser.add_argument("--T", type=int, default=3, help="Number of LISTA iterations")
    train_lista_parser.add_argument("--num-samples", type=int, default=5000)
    train_lista_parser.add_argument("--batch-size", type=int, default=128)
    train_lista_parser.add_argument("--epochs", type=int, default=50)
    train_lista_parser.add_argument("--sparsity", type=float, default=0.1)
    train_lista_parser.add_argument("--noise-std", type=float, default=0.01)
    train_lista_parser.add_argument("--lr", type=float, default=1e-3)
    train_lista_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_lista_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_lista_parser.add_argument("--lambda-recon", type=float, default=0.0)
    train_lista_parser.add_argument("--seed", type=int, default=42)
    train_lista_parser.add_argument("--device", type=str, default=None)
    train_lista_parser.add_argument("--output-dir", type=str, default="runs")
    train_lista_parser.set_defaults(func=lambda args: print(train_lista(args)))

    # ------------------------------------------------------------------
    # Train Koopman AE
    # ------------------------------------------------------------------
    train_kae_parser = subparsers.add_parser("train-kae", help="Train a Koopman autoencoder")
    _add_koopman_train_arguments(train_kae_parser)
    train_kae_parser.set_defaults(func=lambda args: print(train_kae(args)))

    # ------------------------------------------------------------------
    # Train KSAE
    # ------------------------------------------------------------------
    train_ksae_parser = subparsers.add_parser("train-ksae", help="Train a Koopman sparse autoencoder")
    _add_koopman_train_arguments(train_ksae_parser, include_lista=True)
    train_ksae_parser.set_defaults(func=lambda args: print(train_ksae(args)))

    # ------------------------------------------------------------------
    # Eval commands
    # ------------------------------------------------------------------
    eval_lista_parser = subparsers.add_parser("eval-lista", help="Evaluate a LISTA checkpoint")
    eval_lista_parser.add_argument("--checkpoint", type=str, required=True)
    eval_lista_parser.add_argument("--input-dim", type=int, default=100)
    eval_lista_parser.add_argument("--dict-dim", type=int, default=400)
    eval_lista_parser.add_argument("--T", type=int, default=3)
    eval_lista_parser.add_argument("--num-samples", type=int, default=5000)
    eval_lista_parser.add_argument("--batch-size", type=int, default=256)
    eval_lista_parser.add_argument("--sparsity", type=float, default=0.1)
    eval_lista_parser.add_argument("--noise-std", type=float, default=0.01)
    eval_lista_parser.add_argument("--seed", type=int, default=42)
    eval_lista_parser.add_argument("--device", type=str, default=None)
    eval_lista_parser.set_defaults(func=run_eval_lista)

    eval_kae_parser = subparsers.add_parser("eval-kae", help="Evaluate a Koopman AE checkpoint")
    _add_koopman_eval_arguments(eval_kae_parser)
    eval_kae_parser.set_defaults(func=lambda args: run_eval_koopman(args, model_type="kae"))

    eval_ksae_parser = subparsers.add_parser("eval-ksae", help="Evaluate a KSAE checkpoint")
    _add_koopman_eval_arguments(eval_ksae_parser, include_lista=True)
    eval_ksae_parser.set_defaults(func=lambda args: run_eval_koopman(args, model_type="ksae"))

    return parser


def _add_koopman_train_arguments(parser: argparse.ArgumentParser, include_lista: bool = False) -> None:
    parser.add_argument("--system", type=str, default="pendulum")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Minibatch context/window length T in prediction steps (uses T+1 states).",
    )
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--encoder-hidden", type=int, nargs="*", default=[256, 256, 256, 256])
    parser.add_argument("--decoder-hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--action-encoder-layers", type=int, nargs="*", default=[], help="Hidden sizes for action encoder MLP")
    parser.add_argument("--lr-main", type=float, default=1e-4, help="LR for encoder/decoder; AdamW")
    parser.add_argument("--lr-koopman", type=float, default=1e-5, help="LR for Koopman dynamics (A/B or K/L)")
    parser.add_argument("--lr-lista", type=float, default=1e-4, help="LR for LISTA encoder when using KSAE")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-pred", type=float, default=1.0)
    parser.add_argument("--lambda-frob", type=float, default=1e-4)
    parser.add_argument("--lambda-sparse", type=float, default=1e-3, help="L1 penalty on latent embeddings")
    parser.add_argument("--eval-rollout", type=int, default=200)
    parser.add_argument("--reencode-period", type=int, default=20, help="Period for reencoding during evaluation")
    parser.add_argument("--train-reencode-period", type=int, default=0, help="Use rollout-based training loss with this period if > 0")
    parser.add_argument("--koopman-mode", type=str, choices=["continuous", "discrete"], default="continuous",
                        help="Parameterization of Koopman dynamics")
    parser.add_argument(
        "--control-discretization",
        type=str,
        choices=["tustin", "zoh"],
        default="tustin",
        help="Discretization for controls when using continuous-time dynamics",
    )
    # Decoder column normalization toggle
    parser.add_argument("--normalize-decoder-columns", dest="normalize_decoder_columns", action="store_true")
    parser.add_argument("--no-normalize-decoder-columns", dest="normalize_decoder_columns", action="store_false")
    parser.set_defaults(normalize_decoder_columns=True)
    parser.add_argument("--column-norm-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    if include_lista:
        parser.add_argument("--lista-T", type=int, default=3)
        parser.add_argument("--freeze-lista-epochs", type=int, default=20)
    else:
        parser.add_argument("--lista-T", type=int, default=0)
        parser.add_argument("--freeze-lista-epochs", type=int, default=0)


def _add_koopman_eval_arguments(parser: argparse.ArgumentParser, include_lista: bool = False) -> None:
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--system", type=str, default="pendulum")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--encoder-hidden", type=int, nargs="*", default=[256, 256, 256, 256])
    parser.add_argument("--decoder-hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--action-encoder-layers", type=int, nargs="*", default=[], help="Hidden sizes for action encoder MLP")
    parser.add_argument("--koopman-mode", type=str, choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--control-discretization", type=str, choices=["tustin", "zoh"], default="tustin")
    parser.add_argument("--rollout", type=int, default=500)
    parser.add_argument("--reencode-period", type=int, default=0)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--max-plots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    if include_lista:
        parser.add_argument("--lista-T", type=int, default=3)


def run_eval_lista(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)

    model = LISTA(dict_dim=args.dict_dim, input_dim=args.input_dim, iterations=args.T)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    _, _, test_set = create_lista_datasets(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        dict_dim=args.dict_dim,
        batch_size=args.batch_size,
        seed=args.seed,
        sparsity=args.sparsity,
        noise_std=args.noise_std,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_lista(model, test_loader, device)
    print(json.dumps(metrics, indent=2))


def run_eval_koopman(args: argparse.Namespace, model_type: str) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)

    _, _, test_loader = create_dynamics_dataloaders(
        system=args.system,
        num_samples=args.num_samples,
        seq_len=args.sequence_length,
        batch_size=args.batch_size,
        dt=args.dt,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    encoder_hidden = _parse_hidden(args.encoder_hidden)
    decoder_hidden = _parse_hidden(args.decoder_hidden)
    action_encoder_layers = _parse_hidden(getattr(args, "action_encoder_layers", None))

    if model_type == "kae":
        model = KoopmanAE(
            input_dim=test_loader.dataset.dataset.spec.state_dim,
            latent_dim=args.latent_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            control_dim=test_loader.dataset.dataset.spec.control_dim,
            koopman_continuous=(args.koopman_mode == "continuous"),
            dt=args.dt,
            control_discretization=args.control_discretization,
            action_encoder_layers=action_encoder_layers,
        )
    else:
        model = KSAE(
            input_dim=test_loader.dataset.dataset.spec.state_dim,
            latent_dim=args.latent_dim,
            lista_iterations=getattr(args, "lista_T", 3),
            decoder_hidden=decoder_hidden,
            control_dim=test_loader.dataset.dataset.spec.control_dim,
            koopman_continuous=(args.koopman_mode == "continuous"),
            dt=args.dt,
            control_discretization=args.control_discretization,
            action_encoder_layers=action_encoder_layers,
        )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    plot_dir = Path(args.plot_dir) if args.plot_dir else None
    metrics = evaluate_koopman(
        model,
        test_loader,
        device,
        rollout_horizon=args.rollout,
        reencode_period=args.reencode_period,
        plot_dir=plot_dir,
        max_plots=args.max_plots,
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
