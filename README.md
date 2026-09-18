# Smart speaker: train, validate and publish

The current package uses the **v12 shared encoder with finite parameter heads** (seed 17, epoch 2) and the independent local **CTC ASR** (epoch 19). It includes the microphone input, energy VAD, audio frontend, VCM, finite heads, ASR, routing and JSON output. Everything runs locally.

This checkpoint remains experimental. Its accuracy gates are unmet. The notebook and `validation.json` separate software checks from recognition quality; physical Pi4 and live-microphone performance still need measurement.

## Workstation workflow

```bash
python train.py
python publish.py
```

`train.py` sets up one `.venv`, checks the selected components, validates the current held-out dataset and updates **Architecture.ipynb** and `validation.json`. It preserves the current weights by default. To explicitly continue training the enabled models before validation:

```bash
python train.py --fit
```

Training uses the current local data and checkpoints. It requires PyTorch and, by default, CUDA; set `training.device = "cpu"` if needed. The VCM family encoder remains frozen while finite heads train. ASR trains separately. Train/development partitions select weights and provisional thresholds; calibration/test partitions are never used for fitting. Debug runs with example limits must use `update_artifacts = false`.

For quick checks or an already installed environment:

```bash
python train.py --tests-only
python train.py --no-setup
```

## Select what to test

Edit **pipeline.toml**. For example, test only VAD and VCM:

```toml
[tests]
vad = true
vcm = true
asr = false
pipeline = "vad and vcm and asr"
```

Use `true`, `false`, or quoted expressions with switch names, `and`, `or`, `not` and parentheses. Names refer to switches in the same section. Unknown names, cycles and arbitrary Python expressions are rejected. Set `pipeline = true` to exercise all stages regardless of the separate component-test switches.

- `[workflow]` selects setup, training, checks, dataset validation and report updates.
- `[tests]` selects checks and dataset metrics. Disabling a test does not remove a shipped component.
- `[runtime]` selects the stages used for microphone/audio-file inference. ASR without VCM is transcription-only diagnostics.
- `[validation]` selects the held-out split, example limit and inference threads. `max_examples = 0` uses the full split. `require_quality_gate = true` fails the command when the declared quality gates are unmet.
- `[training]` selects model continuation and its settings. `--fit` enables training for that run.
- `[publish]` controls the output and bundled ARM64 wheels. Defaults support CPython 3.11–3.13 on 64-bit Pi OS.
- `[audio]` controls the energy VAD, preroll and endpoint silence within the checkpoint's eight-second limit.

The same editable config travels with the published pipeline. `--config PATH` selects a different config with either workstation script or the Pi application.

## Publish and run on Raspberry Pi 4

`python publish.py` checks the executable using packaged human-speech samples, then writes **dist/smart-speaker-rpi4.zip**. On the workstation, the first publication downloads hash-pinned ARM64 Python wheels; later builds reuse the verified cache. The ZIP contains both model graphs, all runtime code in one executable `pipeline.pyz`, config, requirements, small attributed test recordings, build evidence and the wheels. It excludes training datasets, PyTorch and research checkpoints.

Copy the ZIP to a Pi4 running **64-bit Raspberry Pi OS** with **Python 3.11, 3.12 or 3.13**, then:

```bash
unzip smart-speaker-rpi4.zip
cd smart-speaker
python3 pipeline.pyz setup --system-packages
python3 pipeline.pyz --test
python3 pipeline.pyz --list-mics
python3 pipeline.pyz --mic 2
```

Replace `2` with the numeric microphone index. Stop with **Ctrl+C**. The launcher uses its local `.venv` automatically.

`--system-packages` installs `python3-venv`, `libportaudio2` and `libsndfile1` through apt and needs internet/sudo. If these OS prerequisites are already present, run `python3 pipeline.pyz setup`; Python dependencies install **offline from the bundled wheels**. Recognition itself needs no network. For a different supported desktop platform, `setup --online` fetches that platform's pinned wheels.

```bash
python3 pipeline.pyz --wav recording.wav
python3 pipeline.pyz --mic 2 --channel 0 --capture-rate 48000
```

WAV mode runs the configured VAD by default. Set `runtime.vad = false` for already segmented recordings. Microphone mode requires VAD. The same endpointing code is used by microphone and replay, with channel selection, resampling, repeated utterances, timeout and overflow handling. `--test` never opens a microphone; it writes `launch-validation.json`. Its small packaged sample set is a launch smoke check, not the full held-out benchmark.

**ASR is included**, but ordinary v12 command inference retains its disabled custom-media gate. `runtime.asr = true` makes ASR available; it does not bypass family or parameter acceptance. Only separately accepted CUSTOM media can reach ASR for title/artist extraction. Uncertain finite values request clarification, and ASR never changes the VCM-selected family. Parameter boundaries remain unverified, so eligible ASR uses explicitly labelled whole-utterance audio. The service emits JSON; it does not control real devices or call a cloud/LLM service. The catalog contains example entities.

## Current files

```text
train.py                setup, optional training, selected checks, validation/report
publish.py              complete Pi pipeline publisher
pipeline.toml           workflow, test, runtime and training configuration
Architecture.ipynb      single architecture, implementation and report notebook
requirements.lock       pinned CPU dependencies and wheel hashes
artifacts/              current VCM/ASR weights, graphs, metadata and provenance
data/                   current manifests, references, source ledgers and audio
validation.json         latest workstation validation result
dist/                   generated transfer ZIP, checksum and build report
.cache/                 disposable features, wheels, logs and training candidates
.venv/                  single workstation environment
```

The two root Python scripts load implementation modules directly from the notebook. Old notebooks, test trees, obsolete model versions and superseded documentation are removed. The dataset retains only audio referenced by the current VCM/ASR manifests; source/license ledgers and hashes remain available. Audio and generated/cache files are ignored by Git, so keep the current `data/audio` directory with the workstation copy.
