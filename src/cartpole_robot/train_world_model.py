from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from cartpole_robot.world_model import (
    MLPDynamicsModel,
    choose_device,
    fit_standardizer,
    load_transition_data,
    save_ensemble_checkpoint,
    standardize,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch MLP ensemble world model."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/swingup_random.npz"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("world_models/cartpole_swingup_ensemble.pt"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/world_model"))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    generator: torch.Generator,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(inputs[indices], dtype=torch.float32),
        torch.as_tensor(targets[indices], dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def evaluate_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        return float(loss_fn(predictions, targets).item())


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    transitions = load_transition_data(args.dataset)
    raw_inputs = transitions.inputs
    raw_targets = transitions.targets
    input_mean, input_std = fit_standardizer(raw_inputs)
    target_mean, target_std = fit_standardizer(raw_targets)
    inputs = standardize(raw_inputs, input_mean, input_std)
    targets = standardize(raw_targets, target_mean, target_std)

    sample_count = len(inputs)
    if sample_count < 10:
        raise SystemExit("Need at least 10 transitions to train a world model.")

    indices = rng.permutation(sample_count)
    val_count = max(1, int(sample_count * args.val_split))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    hidden_sizes = tuple([args.hidden_size] * args.hidden_layers)
    input_dim = inputs.shape[1]
    output_dim = targets.shape[1]
    obs_dim = transitions.obs.shape[1]
    action_dim = transitions.actions.shape[1]
    loss_fn = nn.MSELoss()

    val_inputs = torch.as_tensor(inputs[val_indices], dtype=torch.float32, device=device)
    val_targets = torch.as_tensor(targets[val_indices], dtype=torch.float32, device=device)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    models: list[MLPDynamicsModel] = []

    try:
        for model_index in range(args.ensemble_size):
            model = MLPDynamicsModel(input_dim, output_dim, hidden_sizes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            bootstrap_indices = rng.choice(
                train_indices,
                size=len(train_indices),
                replace=True,
            )
            generator = torch.Generator()
            generator.manual_seed(args.seed + model_index)
            loader = make_loader(
                inputs,
                targets,
                bootstrap_indices,
                args.batch_size,
                generator,
            )

            for epoch in range(1, args.epochs + 1):
                model.train()
                train_loss_total = 0.0
                batch_count = 0

                for batch_inputs, batch_targets in loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(batch_inputs)
                    loss = loss_fn(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()

                    train_loss_total += float(loss.item())
                    batch_count += 1

                train_loss = train_loss_total / max(batch_count, 1)
                val_loss = evaluate_loss(model, val_inputs, val_targets, loss_fn)
                global_step = (model_index * args.epochs) + epoch
                writer.add_scalar(f"model_{model_index}/train_loss", train_loss, epoch)
                writer.add_scalar(f"model_{model_index}/val_loss", val_loss, epoch)
                writer.add_scalar("ensemble/train_loss", train_loss, global_step)
                writer.add_scalar("ensemble/val_loss", val_loss, global_step)

                if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                    print(
                        f"model {model_index + 1}/{args.ensemble_size} "
                        f"epoch {epoch:>4}/{args.epochs}: "
                        f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
                    )

            models.append(model)

        save_ensemble_checkpoint(
            args.output,
            models=models,
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            metadata={
                "dataset": str(args.dataset),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "ensemble_size": args.ensemble_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
        )
    finally:
        writer.close()

    print(f"Saved world model ensemble to {args.output}")
    print(f"Device: {device}")
    print(f"Train transitions: {len(train_indices):,}")
    print(f"Validation transitions: {len(val_indices):,}")
