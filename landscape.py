"""Visualize a two-dimensional training-loss slice around a checkpoint."""

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


def make_filter_normalized_direction(model):
    direction = []
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            random_direction = torch.randn_like(parameter)
            direction_norm = torch.norm(random_direction)
            parameter_norm = torch.norm(parameter)
            if direction_norm > 0:
                random_direction = (
                    random_direction / direction_norm
                ) * parameter_norm
            direction.append(random_direction)
        else:
            direction.append(torch.zeros_like(parameter))
    return direction


def add_directions(model, base_parameters, direction1, direction2, alpha, beta):
    with torch.no_grad():
        for parameter, base, d1, d2 in zip(
            model.parameters(),
            base_parameters,
            direction1,
            direction2,
        ):
            parameter.copy_(base + alpha * d1 + beta * d2)


def main(
    checkpoint_path: str = "results/checkpoints/baseline_adam.pt",
    resolution: int = 21,
):
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_file}. Run python train.py first."
        )
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=True,
    )
    model = NeuralNetwork(
        input_size=checkpoint.get("input_size", 2),
        hidden_size=checkpoint["hidden_size"],
        num_hidden_layers=checkpoint["num_hidden_layers"],
        output_size=checkpoint.get("output_size", 2),
        initialization=checkpoint["initialization"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_train, _, y_train, _ = get_data()
    criterion = nn.CrossEntropyLoss()
    base_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    torch.manual_seed(123)
    direction1 = make_filter_normalized_direction(model)
    direction2 = make_filter_normalized_direction(model)
    coordinates = np.linspace(-1.0, 1.0, resolution)
    loss_grid = np.zeros((resolution, resolution), dtype=np.float64)

    try:
        for i, alpha in enumerate(coordinates):
            for j, beta in enumerate(coordinates):
                add_directions(
                    model,
                    base_parameters,
                    direction1,
                    direction2,
                    alpha,
                    beta,
                )
                with torch.no_grad():
                    loss_grid[j, i] = criterion(
                        model(X_train), y_train
                    ).item()
    finally:
        with torch.no_grad():
            for parameter, base in zip(model.parameters(), base_parameters):
                parameter.copy_(base)

    A, B = np.meshgrid(coordinates, coordinates)
    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(A, B, loss_grid, cmap="viridis", edgecolor="none")
    axis.set_xlabel("Direction 1 coefficient")
    axis.set_ylabel("Direction 2 coefficient")
    axis.set_zlabel("Training loss")
    axis.set_title("2D Slice of the Neural Network Loss Landscape")
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_landscape_3d.png", dpi=200)
    plt.close(figure)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(A, B, loss_grid, levels=30, cmap="viridis")
    plt.colorbar(contour, label="Training loss")
    plt.scatter(
        [0], [0], color="red", marker="x", s=100, label="Trained parameters"
    )
    plt.xlabel("Direction 1 coefficient")
    plt.ylabel("Direction 2 coefficient")
    plt.title("Loss Landscape Contour")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_landscape_contour.png", dpi=200)
    plt.close()

    grid_frame = pd.DataFrame(
        loss_grid,
        index=coordinates,
        columns=coordinates,
    )
    grid_frame.index.name = "direction_2_coefficient"
    grid_frame.columns.name = "direction_1_coefficient"
    grid_frame.to_csv(tables_dir / "loss_landscape_grid.csv")
    print("Saved loss landscape figures and results/tables/loss_landscape_grid.csv")


if __name__ == "__main__":
    main()
