"""Reproducible PyTorch regression demo for the ML2021 HW1 COVID data."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Path to covid.train.csv")
    parser.add_argument("--test", type=Path, required=True, help="Path to covid.test.csv")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    mse = float(np.mean(error**2))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else float("nan"),
    }


def make_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(32, 1),
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if train.columns[0] != "id" or test.columns[0] != "id":
        raise ValueError("Expected 'id' as the first column in both CSV files")
    if not train.columns[-1].startswith("tested_positive"):
        raise ValueError("Expected the last training column to be the regression target")
    feature_names = train.columns[1:-1].tolist()
    if feature_names != test.columns[1:].tolist():
        raise ValueError("Training and test feature columns do not match")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing values found; inspect the data before training")

    x_all = train.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    y_all = train.iloc[:, -1].to_numpy(dtype=np.float32)
    x_test = test.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.isfinite(x_all).all() or not np.isfinite(y_all).all() or not np.isfinite(x_test).all():
        raise ValueError("Non-finite input found")

    indices = np.random.default_rng(args.seed).permutation(len(train))
    n_valid = round(len(train) * 0.2)
    valid_idx, train_idx = indices[:n_valid], indices[n_valid:]
    x_train_raw, x_valid_raw = x_all[train_idx], x_all[valid_idx]
    y_train, y_valid = y_all[train_idx], y_all[valid_idx]

    # Fit preprocessing on the training split only: validation stays unseen.
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std[std == 0] = 1
    x_train = (x_train_raw - mean) / std
    x_valid = (x_valid_raw - mean) / std
    x_test = (x_test - mean) / std

    mean_prediction = np.full_like(y_valid, y_train.mean())
    baseline = metrics(y_valid, mean_prediction)
    # A stronger non-neural baseline: ridge regression on the same split/scaling.
    ridge_lambda = 10.0
    ridge_weights = np.linalg.solve(
        x_train.T @ x_train + ridge_lambda * np.eye(x_train.shape[1], dtype=np.float32),
        x_train.T @ (y_train - y_train.mean()),
    )
    ridge_predictions = x_valid @ ridge_weights + y_train.mean()
    ridge_metrics = metrics(y_valid, ridge_predictions)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    x_valid_tensor = torch.from_numpy(x_valid)
    model = make_model(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(features).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

        model.eval()
        with torch.no_grad():
            valid_predictions = model(x_valid_tensor).squeeze(1).numpy()
        train_loss = running_loss / len(train_idx)
        valid_loss = metrics(y_valid, valid_predictions)["mse"]
        history.append({"epoch": epoch, "train_mse": train_loss, "valid_mse": valid_loss})
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        elif epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid_predictions = model(x_valid_tensor).squeeze(1).numpy()
        test_predictions = model(torch.from_numpy(x_test)).squeeze(1).numpy()
    model_metrics = metrics(y_valid, valid_predictions)

    summary = {
        "task": "continuous-value regression, not individual diagnosis or classification",
        "seed": args.seed,
        "rows_train": len(train_idx),
        "rows_valid": len(valid_idx),
        "rows_unlabeled_test": len(test),
        "features": len(feature_names),
        "best_epoch": best_epoch,
        "baseline_mean": baseline,
        "baseline_ridge": ridge_metrics,
        "pytorch_mlp": model_metrics,
        "caveat": "Metrics are from one random validation split; the unlabeled test set has no score.",
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(args.out / "training_history.csv", index=False)
    pd.DataFrame({"id": test["id"], "tested_positive": test_predictions}).to_csv(
        args.out / "test_predictions.csv", index=False
    )
    torch.save(model.state_dict(), args.out / "model.pt")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], label="Train")
    ax.plot([row["epoch"] for row in history], [row["valid_mse"] for row in history], label="Validation")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch: {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Mean squared error", title="PyTorch MLP training curve")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "training_curve.png", dpi=180)
    plt.close(fig)

    fig, (ax_all, ax_detail) = plt.subplots(1, 2, figsize=(9, 4))
    names = ["Mean", "Ridge", "PyTorch MLP"]
    values = [baseline["rmse"], ridge_metrics["rmse"], model_metrics["rmse"]]
    ax_all.bar(names, values, color=["#b9c4d1", "#147d70", "#3767b3"])
    ax_all.set(ylabel="Validation RMSE", title="All models")
    ax_detail.bar(names[1:], values[1:], color=["#147d70", "#3767b3"])
    ax_detail.set(ylim=(0.9, 1.02), title="Strong baselines (zoomed)")
    for ax in (ax_all, ax_detail):
        for bar in ax.patches:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom")
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Validation RMSE: lower is better")
    fig.tight_layout()
    fig.savefig(args.out / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
