# Experiment and Environment Notes

Fill this in from the actual machine and generated results. Do not use
placeholder values as measured results.

## Environment

- GitHub repository URL: [add URL after creating the remote repository]
- Operating system/version: [record]
- Python version: [record]
- PyTorch version: [record]
- NumPy / Matplotlib / pandas / scikit-learn versions: [record]
- CPU / GPU and relevant hardware notes: [record]
- Execution date: [record]
- Dependency lock file: [create with `python -m pip freeze > requirements-lock.txt`]

## Run log

| Run/configuration | Seed | Optimizer and settings | Depth x width | Initialization | Epochs | Training loss | Test accuracy | Time (s) | Notes |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| [record from `results/tables/experiment_comparison.csv`] | | | | | | | | | |

Use `results/tables/experiment_summary.csv` for across-seed mean and standard
deviation. Include failed or repeated runs in notes rather than silently
discarding them.

## Observations

- Optimizer comparison: [record observations supported by the CSV/plots]
- Depth comparison: [record observations]
- Width comparison: [record observations]
- Initialization comparison: [record observations]
- Landscape slice: [describe the plotted region and its limits]

## Reproducibility notes

[Record exact commands, any changes to configuration, warnings/errors, and
whether all outputs were regenerated from source.]
