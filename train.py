"""Train, evaluate, and save one neural-network experiment."""

import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import get_data
from model import NeuralNetwork


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(name, parameters, learning_rate: float):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if name == "momentum":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


def evaluate(model, X_test, y_test) -> float:
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(X_test), dim=1)
        return (predictions == y_test).float().mean().item()


def train_model(
    optimizer_name: str = "adam",
    hidden_size: int = 16,
    num_hidden_layers: int = 2,
    initialization: str = "he",
    epochs: int = 200,
    learning_rate: float = 0.01,
    seed: int = 42,
    run_name: str = "baseline_adam",
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be a non-empty filename stem")

    set_seed(seed)
    figures_dir = Path("results/figures")
    checkpoints_dir = Path("results/checkpoints")
    tables_dir = Path("results/tables")
    for directory in (figures_dir, checkpoints_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = get_data()
    model = NeuralNetwork(
        input_size=2,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        output_size=2,
        initialization=initialization,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        optimizer_name,
        model.parameters(),
        learning_rate,
    )

    history = []
    started_at = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({"epoch": epoch + 1, "training_loss": loss.item()})
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"{run_name} | Epoch {epoch + 1}/{epochs} "
                f"| Loss: {loss.item():.5f}"
            )

    training_seconds = time.perf_counter() - started_at
    test_accuracy = evaluate(model, X_test, y_test)
    final_training_loss = history[-1]["training_loss"]

    print(f"\nRun: {run_name}")
    print("Optimizer:", optimizer_name)
    print("Final training loss:", round(final_training_loss, 5))
    print("Test accuracy:", round(test_accuracy * 100, 2), "%")
    print("Training time (seconds):", round(training_seconds, 3))

    checkpoint_path = checkpoints_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": 2,
            "output_size": 2,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "initialization": initialization,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "momentum": 0.9 if optimizer_name == "momentum" else None,
            "epochs": epochs,
            "seed": seed,
            "test_accuracy": test_accuracy,
            "final_training_loss": final_training_loss,
        },
        checkpoint_path,
    )

    history_path = tables_dir / f"{run_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        [row["epoch"] for row in history],
        [row["training_loss"] for row in history],
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Cross-Entropy Loss")
    plt.title(f"Training Loss - {run_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{run_name}_loss.png", dpi=200)
    plt.close()

    return {
        "run_name": run_name,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "momentum": 0.9 if optimizer_name == "momentum" else None,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "initialization": initialization,
        "epochs": epochs,
        "seed": seed,
        "final_training_loss": final_training_loss,
        "test_accuracy": test_accuracy,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint_path),
    }


if __name__ == "__main__":
    train_model(
        optimizer_name="adam",
        hidden_size=16,
        num_hidden_layers=2,
        initialization="he",
        epochs=200,
        learning_rate=0.01,
        seed=42,
        run_name="baseline_adam",
    )
