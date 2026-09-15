# TinyVCM: independently trained intent and parameter speech models

## Repository setup

For this repository, use the [root setup](../README.md): `uv sync --locked`
installs the voice app dependencies and this model package into one root
`.venv`. The root `uv.lock` locks runtime, training, and notebook dependencies;
training and report extras are optional. The runtime ONNX bundle and its
metadata are ignored by Git. A fresh clone needs a restored bundle or a model
rebuild before inference can start.

All exclusions are maintained in the repository's root `.gitignore`, including
generated reports, figures, dataset manifests, and the review queue. Source
code, configuration, dependency locks, pinned `*.source.json` ledgers, and the
experiment notebook remain available to Git.

Run `uv run --locked python scripts/reproduce.py --dry-run` from the repository
root to inspect the complete model rebuild recipe. Acquisition uses the
versioned source ledgers and checks their recorded hashes. The standalone Pi
distribution instructions below describe the exported package, which can also
be used outside this repository.

The executed **[TinyVCM_report.ipynb](TinyVCM_report.ipynb)** is the experiment report. It contains dataset provenance, leakage exclusions, training curves, held-out human voice tests, error analysis, model sizes, and measured CPU replay results. This is a measured research baseline; read its accuracy and coverage limits before using the outputs.

The VCM takes audio and chooses an intent. A separate character-CTC ASR can transcribe the retained parameter audio. The ASR never selects the action. Everything runs locally; inference needs neither PyTorch nor a network connection. The CLI emits JSON for your application to consume.

The completed FP32 tests measured **74.1% raw intent accuracy**, **17.1% supported-command recall after confidence rejection**, and **48.4% SLURP ASR word error rate**. The full pipeline produced the exact command and parameters for **150 of 1,466 resolvable parser references (10.2%)**. Those references come from source text and the fixed example catalog; they are not manually adjudicated slot labels. This baseline is not ready for reliable everyday command control.

## Raspberry Pi 4 quick start

Use a **64-bit OS** and Python 3.11 or newer. Actual Pi measurements remain pending until `scripts/benchmark.py` is run on the board. Commands below run from this `model/` directory.

For transfer, `artifacts/tinyvcm_pi4.tar.gz` contains the runtime, models, executed notebook, and 60 public voice-test clips. Extract it with `tar -xzf tinyvcm_pi4.tar.gz`, then enter the extracted `model/` directory. The package supports inference and the supplied benchmark; re-executing the notebook requires the full experiment workspace and report dependencies.

```bash
sudo apt-get install python3-venv libportaudio2 libsndfile1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-runtime.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m tinyvcm devices
.venv/bin/python -m tinyvcm listen --mic "USB" --asr on --catalog configs/catalog.example.json
```

Choose `--mic 2` by the current input index, or a unique substring such as `--mic "USB"`. Add `--host-api ALSA` if several backends expose the same device. `devices` lists the actual host devices; indices can change after reconnecting hardware. `--channel 0` chooses the input channel. The capture path tries 16 kHz and otherwise resamples the device's native rate; an explicitly requested unsupported `--capture-rate` fails visibly.

Settings can also be saved in `configs/pi4.toml`. Command-line values take precedence over config values, which take precedence over defaults.

```bash
python -m tinyvcm listen --config configs/pi4.toml --mic "USB" --asr off
python -m tinyvcm predict --wav path/to/command.wav --asr on --catalog configs/catalog.example.json
python -m tinyvcm predict --wav path/to/command.wav --asr off --save-parameter parameter.wav
python scripts/benchmark.py --variant fp32 --threads 2 --output reports/benchmark_pi4_fp32.json
python scripts/benchmark.py --variant int8 --threads 2 --output reports/benchmark_pi4_int8.json
```

An early `stage: intent` event is provisional. Final results use `accepted`, `rejected`, `needs_parameters`, or `needs_clarification`. Commands that need names remain `needs_parameters` with ASR disabled. Capture uses 200 ms pre-roll, 400 ms trailing silence, and an 8-second limit. Timing is a conservative baseline: there is **no trained readiness or parameter-boundary head**. A stable intent trigger plus a 500 ms look-back defines the predicted slice. The report measures failures of this policy.

## Vocabulary and configurable names

Trained labels: `POWER_ON`, `POWER_OFF`, `BRIGHTNESS_UP`, `BRIGHTNESS_DOWN`, `COLOR_SET`, `MEDIA_PLAY`, `MEDIA_RESUME`, `TIME_QUERY`, `DATE_QUERY`, and `UNKNOWN`. See `reports/coverage.json` for exact per-class counts. Numeric brightness/volume, pause/stop, and timers lacked sufficient public examples for this run and are unsupported VCM commands.

Edit the catalog's `entities` list to add names and aliases. Each entry has `id`, `name`, `kind` (`light` or `switch`), and optional `aliases`. These are example bindings, not connected devices. Avoid aliases that match several entries (for example a generic `lights` entry alongside `bedroom lights`); ambiguous matches require clarification. The ASR alphabet is English letters, spaces, and apostrophes. A new catalog name does not retrain either model or guarantee that the small ASR will recognize it. Colors are a bounded vocabulary; raw media text is passed as a selection argument for a downstream application.

`TIME_QUERY` and `DATE_QUERY` currently emit local time/date requests with no slots. Place-specific time queries are a known semantic limitation of this broad public label set. Relative brightness defaults to ten percentage points when no amount is present. No physical actions, Home Assistant calls, or media playback occur in this package.

## Reproduce acquisition and training

Use the root rebuild command above on a CUDA workstation. For individual research steps, install the root extras with `uv sync --locked --extra train --extra report`, activate that root `.venv`, and run the commands below from `model/`. `reports/environment.json` records the exact versions and GPU used in the original experiment. Training starts from random initialization; ASR assistant adaptation starts only from the independently trained acoustic ASR checkpoint.

```bash
python scripts/acquire.py metadata
python scripts/acquire.py mirror
python scripts/acquire.py libri
python scripts/prepare.py vcm
python scripts/prepare.py asr
python -m tinyvcm train-vcm --run vcm_seed17 --epochs 50 --batch-size 48
python -m tinyvcm train-vcm --run vcm_context_seed17 --epochs 60 --batch-size 48 --width 192 --pooling mean_max --kernels 7 11 15 19 23
python -m tinyvcm train-asr --stage acoustic --run asr_acoustic_seed17 --epochs 40 --batch-size 48
python -m tinyvcm train-asr --stage commands --run asr_commands_seed17 --epochs 30 --batch-size 48 --lr 0.0005 --initialize runs/asr_acoustic_seed17/best.pt
python -m tinyvcm.export vcm
python -m tinyvcm.evaluate vcm
python scripts/evaluate_open_set.py
python -m tinyvcm.export asr
python -m tinyvcm.evaluate asr
python scripts/acquire_accent.py
python scripts/evaluate_accent.py
python -m pytest -q > logs/contract_tests.log
python scripts/record_environment.py
python scripts/benchmark.py --variant fp32 --output reports/benchmark_windows_fp32.json
python scripts/benchmark.py --variant int8 --output reports/benchmark_windows_int8.json
python scripts/build_report.py
```

Existing run directories are protected against accidental overwrite; use `--resume` to continue or change `--run`. `scripts/run_vcm_pipeline.py` and `scripts/run_asr_pipeline.py` automate the same dependencies. Dataset archives, extracted audio, feature caches, checkpoints, and everything under `artifacts/` stay local and are excluded from Git. Source ledgers pin acquisition to immutable revisions and archive checksums. Large generated manifests are rebuilt by `prepare.py`.

Before building the report, create a second environment `.runtime-venv` with only `requirements-runtime.lock` and run `scripts/runtime_smoke.py` in it. The script verifies that PyTorch is absent and records `reports/runtime_smoke.json`. `scripts/finish_experiment.py` automates export through report/package finalization after ASR completes in this Windows workspace. For another host, keep benchmark filenames labelled by the real host; the notebook's current execution is the recorded Windows experiment. `reports/pi_dependency_resolution.txt` is a binary-package resolution check, not a Pi installation log.

Training, model selection, threshold calibration, and final testing are distinct partitions. The public SLURP splits share speakers; this is explicitly **not** a speaker-disjoint benchmark. We exclude microphone pairs that cross source partitions, preserve connected families when splitting development/calibration, and disclose the exclusions. The first VCM selection used development macro F1; final test results did not select a checkpoint. Silence, white-noise replay, and source distant microphones are separate tests.

## Contents

| Path | Purpose |
|---|---|
| `TinyVCM_report.ipynb` | Executed, comprehensive experiment report |
| `tinyvcm/` | Shared frontend, independent models, training, export, evaluation, routing, CLI |
| `data/manifests/` | Source hashes and reproducible example manifests |
| `data/annotations/review_queue.jsonl` | Public clips awaiting verified boundary annotations |
| `runs/` | Checkpoints, optimizer state, training histories and configurations |
| `artifacts/tinyvcm/` | FP32/INT8 VCM, streaming VCM, ASR, and bundle contract |
| `reports/` | Machine-readable metrics, individual voice-test results and figures |
| `tests/` | Causality, microphone selection, argument binding, and runtime contracts |
| `scripts/benchmark.py` | Host-labelled CPU/RAM benchmark that also runs on Pi 4 |

## Attribution and data terms

- **SLURP:** Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser, *SLURP: A Spoken Language Understanding Resource Package*, EMNLP 2020. [Upstream](https://github.com/pswietojanski/slurp), [paper](https://aclanthology.org/2020.emnlp-main.588/). Text CC BY 4.0; real audio CC BY-NC 4.0. Audio transport uses a pinned [qmeeus/slurp mirror](https://huggingface.co/datasets/qmeeus/slurp); upstream IDs and text remain authoritative.
- **LibriSpeech:** Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur, *LibriSpeech: an ASR corpus based on public domain audio books*, ICASSP 2015. [OpenSLR 12](https://www.openslr.org/12/), CC BY 4.0. This phase uses train-clean-100, dev-clean, and test-clean, with documented duration/text filtering.
- **Speech Accent Archive:** Steven H. Weinberger and Matthew C. Kelley, George Mason University. [Source](https://accent.gmu.edu/language/tagalog/), CC BY-NC-SA 4.0. Ten unchanged public recordings form a small evaluation-only diagnostic. Elicitation prompts are not verified verbatim transcripts, and shared read text is not a Philippine-English command benchmark. Retain attribution and applicable license terms with recordings and derived material.

FP32 is the default deployment variant. INT8 is measured as an experimental alternative; reduced file size does not establish a speed gain on Pi hardware. The package has no wake word, no trained silence detector, no verified unknown-name generalization, and no boundary timestamps. These limitations and the measured command rejection tradeoff are part of the report.
