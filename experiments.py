"""Run controlled architecture, optimizer, and initialization experiments."""

from pathlib import Path

import pandas as pd

from train import train_model

EPOCHS = 200
LEARNING_RATE = 0.01
SEEDS = (42, 43, 44)

CONFIGURATIONS = [
    {
        "optimizer_name": "sgd",
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "initialization": "he",
        "run_name": "optimizer_sgd",
    },
    {
        "optimizer_name": "momentum",
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "initialization": "he",
        "run_name": "optimizer_momentum",
    },
    {
        "optimizer_name": "adam",
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "initialization": "he",
        "run_name": "optimizer_adam",
    },
    {
        "optimizer_name": "adam",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "initialization": "he",
        "run_name": "depth_2",
    },
    {
        "optimizer_name": "adam",
        "hidden_size": 32,
        "num_hidden_layers": 5,
        "initialization": "he",
        "run_name": "depth_5",
    },
    {
        "optimizer_name": "adam",
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "initialization": "he",
        "run_name": "width_16",
    },
    {
        "optimizer_name": "adam",
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "initialization": "he",
        "run_name": "width_64",
    },
    {
        "optimizer_name": "adam",
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "initialization": "xavier",
        "run_name": "init_xavier",
    },
    {
        "optimizer_name": "adam",
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "initialization": "he",
        "run_name": "init_he",
    },
]


def main():
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for config in CONFIGURATIONS:
        for seed in SEEDS:
            run_name = (
                config["run_name"]
                if seed == SEEDS[0]
                else f"{config['run_name']}_seed_{seed}"
            )
            print(f"\n{'=' * 60}\nStarting: {run_name}\n{'=' * 60}")
            result = train_model(
                optimizer_name=config["optimizer_name"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_hidden_layers"],
                initialization=config["initialization"],
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                seed=seed,
                run_name=run_name,
            )
            result["configuration"] = config["run_name"]
            results.append(result)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(
        tables_dir / "experiment_comparison.csv",
        index=False,
    )
    summary = (
        results_frame.groupby("configuration", as_index=False)
        .agg(
            seeds=("seed", "count"),
            final_training_loss_mean=("final_training_loss", "mean"),
            final_training_loss_std=("final_training_loss", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            training_seconds_mean=("training_seconds", "mean"),
        )
    )
    summary.to_csv(tables_dir / "experiment_summary.csv", index=False)

    print("\nExperiment summary:")
    print(summary.to_string(index=False))
    print("\nSaved results/tables/experiment_comparison.csv")
    print("Saved results/tables/experiment_summary.csv")


if __name__ == "__main__":
    main()
