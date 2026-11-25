# Loss Landscape Investigation

Reproducible CPU-friendly experiments on how neural-network architecture,
optimization, and initialization affect training on the Two Moons
classification problem. The loss-landscape plots show two-dimensional slices
through parameter space; they do not prove that a point is a local or global
minimum or a saddle point.

## Requirements

- Windows 10/11, VS Code, and Git
- Python 3.10 or 3.11 (recommended by the execution guide)
- Packages listed in [`requirements.txt`](requirements.txt)
- No GPU required

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter. If PowerShell blocks activation,
use Command Prompt and run `.venv\Scripts\activate.bat`.

## Run the project

Run commands from this project directory:

```powershell
python dataset.py
python train.py
python landscape.py
python experiments.py
```

The scripts create the `results/figures`, `results/tables`, and
`results/checkpoints` directories as needed. The experiment matrix runs nine
controlled configurations (optimizer, depth, width, and initialization), each
with seeds 42, 43, and 44. It may take several minutes on CPU.

## Outputs

- Dataset: `results/figures/two_moons_dataset.png`
- Baseline: checkpoint, epoch history CSV, and loss plot named
  `baseline_adam` under their respective results folders
- Landscape: 3D surface, contour plot, and sampled loss grid CSV
- Experiments: per-run loss plots and histories, raw results in
  `experiment_comparison.csv`, and mean/std summary in
  `experiment_summary.csv`

Checkpoints are intentionally ignored by Git because they are generated
binary files. Review generated plots and tables before submission. Never
report illustrative or unrun values as experimental results.

## Reproducibility and interpretation

The dataset split and preprocessing are fixed with random state 42. Each
training run sets Python, NumPy, and PyTorch seeds and records its seed,
optimizer, architecture, initialization, training duration, and test accuracy.
Standardization is fitted on training data only. Test data is held out from
gradient updates.

`landscape.py` perturbs the trained weights along two independently generated,
filter-normalized directions, with bias directions held at zero. Its plots
describe only the sampled slice and use training loss; compare plots only when
the directions and coordinate ranges are comparable. Further Hessian or
multi-direction analysis is required for stronger claims about curvature or
critical points.

To record the environment used for a submission:

```powershell
python -m pip freeze > requirements-lock.txt
python --version
git --version
```

## Repository layout

```text
dataset.py
model.py
train.py
landscape.py
experiments.py
requirements.txt
README.md
TECHNICAL_REPORT.md
EXPERIMENT_NOTES.md
VIVA_PREPARATION.md
results/
  figures/
  tables/
  checkpoints/
```

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the report structure,
[`EXPERIMENT_NOTES.md`](EXPERIMENT_NOTES.md) for a results-recording template,
and [`VIVA_PREPARATION.md`](VIVA_PREPARATION.md) for the guide's viva topics.
The GitHub repository URL and machine-specific observations must be filled in
by the project owner after creating/running the project.
