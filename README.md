# Smart speaker

AI 231 project using local voice command models. The [voice service](src/voice/README.md)
accepts a numeric microphone index, waits for speech with VAD, and prints the
detected intent and required parameters.

## One repository setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```powershell
uv sync --locked
uv run --locked python scripts/check_setup.py
uv run --locked python -m src.voice --list-mics
uv run --locked python -m src.voice --mic 2
```

Generated models are ignored by Git. On a fresh clone, restore an existing
bundle into `model/artifacts/tinyvcm/` or follow **Reproduce the models** below
before running the setup check or voice service.

Replace `2` with your microphone's index. Stop with **Ctrl+C**.

The root `pyproject.toml` manages the app and local `model/` package as one
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/).
It uses one root `.venv`, one `uv.lock`, and Python **3.12.13** from
`.python-version`. `--locked` rejects dependency changes that have not been
resolved into the lockfile. There is no separate model-environment setup step.
On Debian/Ubuntu, microphone capture also needs the system packages
`libportaudio2` and `libsndfile1`.

## Models and Git

The root `.gitignore` controls exclusions for the whole repository. Model
rules use `/model/` paths; shared cache, environment, and temporary-file rules
apply throughout the project.

The generated runtime bundle under `model/artifacts/tinyvcm/` is ignored by
Git: six FP32/INT8 ONNX files plus `bundle.json`, about **9.2 MB** total. Keep
the bundle locally or regenerate it using the recipe below.
`scripts/check_setup.py` checks each model's byte size and SHA-256 against
`bundle.json`, then loads both VCM variants.

| Included in the repository | Generated locally and ignored |
|---|---|
| Application and model source, tests, configuration | `.venv`, caches, local dependencies |
| `pyproject.toml`, `uv.lock`, `.python-version` | Raw datasets and processed features |
| Training and export recipe | Runtime ONNX files, bundle metadata, and training checkpoints |
| Pinned source ledgers and experiment notebook | Generated reports, figures, manifests, review queue, archives, and logs |

The model package's `requirements-runtime.lock` remains a snapshot for the
standalone Pi distribution. The repository setup uses the root `uv.lock`.

## Reproduce the models

The recorded acquisition, preparation, training, model selection, export, and
calibration steps have a root entry point. Inspect the steps without changing
data or models:

```powershell
uv run --locked python scripts/reproduce.py --dry-run
```

To execute the recipe in a fresh checkout:

```powershell
uv run --locked --extra train python scripts/reproduce.py
uv run --locked python scripts/check_setup.py
```

Training requires an NVIDIA GPU with a compatible CUDA driver. The `train`
extra pins PyTorch 2.13.0 with CUDA 13.0 on Windows/Linux, matching the recorded
training version, and adds training dependencies to the same root `.venv`.
It downloads the public datasets and runs the full training schedule; it is
not part of ordinary setup. Existing training runs cause an early error to
avoid overwriting them.

Acquisition uses the revisions, URLs, and checksums in
`model/data/manifests/*.source.json`, including the pinned SLURP upstream and
audio-mirror revisions. Updating a dataset is an explicit change to those
ledgers. Seeds and training options are recorded in the recipe. Reusing a saved
bundle preserves the exact weights; retraining can produce different
weights and metrics across GPU, driver, and platform combinations.

The [model documentation](model/README.md) and
[executed experiment report](model/TinyVCM_report.ipynb) describe the datasets,
model accuracy, and existing evaluation results. Running setup does not rerun
that experiment.

## Tests

```powershell
uv run --locked python -m unittest discover -s tests -v
```

For the model's PyTorch tests as well:

```powershell
uv run --locked --extra train python -m pytest
```

Notebook dependencies are available in the same workspace with
`uv sync --locked --extra train --extra report`.
