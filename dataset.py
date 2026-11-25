"""Reproducible Two Moons data generation and visualization."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(random_state: int = 42):
    """Return standardized train/test tensors, fitting on training data only."""
    features, labels = make_moons(
        n_samples=2000,
        noise=0.20,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def plot_dataset(output_path: str | Path = "results/figures/two_moons_dataset.png"):
    """Save a plot of the reproducible, unscaled Two Moons dataset."""
    features, labels = make_moons(
        n_samples=2000,
        noise=0.20,
        random_state=42,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        features[:, 0],
        features[:, 1],
        c=labels,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.25,
        s=18,
    )
    plt.title("Two Moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Number of input features:", X_train.shape[1])
    print("Number of classes:", len(torch.unique(y_train)))
    plot_dataset()
