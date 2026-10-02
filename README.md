# Streaming VCM: diagnostic semifinal

## Smart speaker publication

Publish the selected model without training:

```sh
python publish.py --benchmarks skip
# Equivalent:
python start.py --skip-train --benchmarks skip --target all
```

The release folder `dist/speaker/<release>/` contains three ZIPs and `release.json` with sizes and SHA-256 hashes:

| ZIP | Contents | Default start |
| --- | --- | --- |
| `smart-speaker-inference.zip` | Models, inference, compiled SpeakerUI, config and licenses; no benchmark corpus | UI + inference server |
| `smart-speaker-benchmark.zip` | Models, CLI inference, 6,768 pinned recordings, benchmark/guided test runners and provenance; no UI/Node | Recorded benchmark |
| `smart-speaker-full.zip` | Both | UI + inference server |

Use `--target inference`, `benchmark`, or `full` to build one profile. `--target pi` and `standalone` are legacy aliases for full and benchmark. `--output DIR` selects a new release directory. Existing ZIPs are immutable. The legacy `publish.py --bundle ...` model-only interface remains available.

### Start on Raspberry Pi

Extract one ZIP into a new directory, then run `python3 start.py` (Windows: `python start.py`). First setup installs locked dependencies locally and requires internet. Pi requires a 64-bit OS. The inference and full ZIPs serve `http://127.0.0.1:3000`; no training framework, Git checkout or Pi-side UI compilation is required. With missing audio libraries, run `python3 start.py --system-packages` on Debian/Pi OS.

Open **Debug → Inference**. Microphones populate automatically and can be refreshed. Select the actual input, click Listen, say “Hey Jarvis”, pause until command-ready, then speak. Device discovery errors now show the underlying audio error. `python3 start.py --list-mics` lists the same host devices. USB/OS driver detection and live capture still need verification on your Pi.

The bottom-left text shows intent, typed slots, confidence and action outcome. Detailed JSON and the original model qualification decision remain in Debug/history. WAV upload accepts mono 16 kHz recordings and never executes actions; it stops live capture first.

### Configuration and engine selection

`config.toml` is created **beside `start.py`** on first startup from `config.example.toml`. Copy `.env.example` to **`.env` in that same root directory** for secrets. Move a legacy `SpeakerUI/.env` to the root. Restart after editing. Precedence: CLI > process environment > root `.env` > `config.toml` > defaults.

```toml
[inference]
engine = "ncnn" # Alternative: "onnxruntime"
mode = "demo" # Alternative: "qualified"
intent_threshold = 0.95
slot_threshold = 0.90
threads = 1

[actions]
enabled = true
```

Both engines ship in **all three ZIPs**, with ncnn the default and no silent fallback. CLI override: `python3 start.py --engine onnxruntime --wav command.wav`; environment override: `INFERENCE_ENGINE=onnxruntime`. ncnn runs an FP32 conversion of the same selected checkpoint; ONNX Runtime runs the selected INT8 export, so predictions can differ. Wake/VAD use ONNX Runtime with either command engine.

Inference/full default to live microphone demo execution at **95% intent and 90% for every slot**, with exact schema and intent/target consistency checks. Slot confidence is conditional on the finite catalog, not measured calibration. This application policy permits execution while preserving the frozen model's unqualified status. `mode="qualified"` follows the frozen policy. Benchmarks report frozen acceptance separately and never execute devices.

Configure Home Assistant/Spotify enablement and ordinary values under `[providers]`; put tokens/secrets in `.env`. Map finite model IDs explicitly under `[bindings.lights]` and `[bindings.songs]`. Actions enabled does not imply providers are configured. Missing mappings/providers and unsupported commands have visible outcomes. Supported execution includes mapped lights/Spotify and local timer/alarm/time/reminder UI operations. CALL, MESSAGE, WEATHER and TEMPERATURE have no voice action implementation in this release. Local timer/alarm behavior requires the UI to remain running.

### Raspberry Pi evidence

In benchmark/full:

```sh
python3 start.py --benchmark-only --engine ncnn
.venv/bin/python benchmark.py --engine onnxruntime --output results/ort-01
.venv/bin/python guided_test.py --guided --mic INDEX --output results/live-01
```

Recorded runs write `metrics.json`, `REPORT.md`, `phrases.csv`, `slots.csv`, `trials.jsonl` and `timing.jsonl`, including per-track and per-intent results, CPU affinity, latency, memory and available Pi telemetry. Guided microphone testing is separate. Reports retain missing phrase denominators and failed qualification; replay does not prove microphone or physical Pi performance.

`--wav command.wav` and `--list-mics` work in every profile. Benchmark options intentionally fail in inference-only, which excludes evaluation data. For a model-only update use `start.py --update NEW.zip`; compatibility/integrity checks and previous-model recovery remain. Runtime/UI releases require fresh extraction, preserving your private `.env` and `config.toml` separately.

See [publication specification](docs/SPEC-speaker-publication-v2.md), [implementation and verification](docs/PLAN-speaker-publication-v2.md), and [benchmark protocol](docs/EXPERIMENT-pcv-full-benchmark-v1.md). No training, physical device actions, or Pi deployment was performed for this publication change.

## Earlier release workflow

The current PCV schema-v3 release workflow is documented in [PCV release training v1](docs/EXPERIMENT-pcv-release-train-v1.md). In the configured research checkout, run `.venv/Scripts/python.exe release.py --run-id release-v1` to train, test, validate and create the Pi transfer package, or add `--skip-train` to reuse the bound selection. The workflow reports model qualification separately from successful packaging. The older semifinal description below is preserved as historical evidence.

The handoff was executed from tracked source with newly trained weights. The result is **not qualified**: automatic command acceptance is disabled because calibration recall failed. WAV and microphone interfaces show tentative candidates and final rejection reasons; they never execute devices. See [MODEL_CARD.md](MODEL_CARD.md) for all counts and limitations.

The shared `vcm` package owns the causal frontend, 113,064-parameter CNN with six explicit caches, finite whole-command phoneme CTC scorer, typed catalogs and acceptance policy. The Pi distribution uses NumPy and one ONNX Runtime CPU session. It contains no PyTorch, sentence transcriber, LLM, cloud inference, historical model, training corpus or visualizer.

## Inspect the supplied diagnostic artifact

Use Python 3.12. On 64-bit Raspberry Pi OS, install the OS PortAudio and libsndfile libraries if needed for microphone/audio support. Installing dependencies needs network access; recognition does not.

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
python scripts/install_bundle.py --archive releases/semifinal-v1/vcm-diagnostic.zip --output models/selected
python -m vcm.tui --bundle models/selected --wav /path/to/mono.wav
# Live input is explicit; list/select the device using sounddevice on your own host.
python -m vcm.tui --bundle models/selected --mic 0
```

Use the existing root `.venv` in this research checkout. The WAV inspector supports mono 8–96 kHz files, resampling the complete file before streaming. Native microphone input requires 16 kHz, uses bounded capture queues, and rejects gaps/overflow/nonfinite samples until reset. Microphone inspection is push-to-talk with a 30-second limit. No physical Pi or microphone run was performed in this handoff.

The ZIP is also a standalone distribution: extract it into a new directory and install its `requirements.txt` with `--require-hashes --only-binary=:all:`. It has its own runtime source, `models/selected`, manifest, model card and licenses. The graph is available through the versioned ZIP even in a fresh clone; no historical recovery archives or chat attachments are prerequisites.

## Reproduce the bounded experiment

Dataset terms are described in [DATASET.md](DATASET.md). The `bounded` source profile is academic/research-only because it includes FSC. Start from a clean tracked checkout; unrelated untracked work is preserved.

```bash
python scripts/bootstrap_semifinal.py --profile bounded
python scripts/bootstrap_semifinal.py --profile bounded --apply --allow-restricted
python scripts/prepare_semifinal.py --output .cache/semifinal-v1/prepared
python scripts/prepare_negatives.py --base .cache/semifinal-v1/prepared --output .cache/semifinal-v1/mixed-v2
# Training-only dependencies: use the recorded root environment, including PyTorch and ONNX.
python train.py --config configs/train.json --output .cache/primary-reproduction
python train.py --config configs/train-longer.json --output .cache/longer-reproduction
python scripts/calibrate_semifinal.py --bundle .cache/longer-reproduction/bundle --manifest .cache/semifinal-v1/mixed-v2/calibration.jsonl --output .cache/longer-reproduction/calibration
python validate.py --bundle .cache/longer-reproduction/bundle --manifest .cache/semifinal-v1/mixed-v2/test.jsonl --output reports/new-test-run
python benchmark.py --bundle models/selected --manifest .cache/semifinal-v1/mixed-v2/dev.jsonl --output reports/new-host-benchmark
python publish.py --bundle models/selected --diagnostic --output dist/new-diagnostic.zip
```

Each preparation, training, evaluation and publication output is immutable: use a new output path for a new run. Existing mismatched files are never silently replaced. Original manifests contain resolved paths from this host; regeneration on another host produces new manifest/receipt hashes. Source archive hashes, split rules and configurations are pinned. GPU fitting is explicit in the frozen configs. No further training is needed to inspect the shipped artifact.

Qualified publication is a separate gate:

```bash
python publish.py --bundle models/selected --report reports/semifinal-v1/release.json --output dist/qualified.zip
```

That command **fails for this candidate**. `--diagnostic` is the explicit route for the failed candidate; it does not enable acceptance or satisfy qualification.

## Evidence and development

- [All replay commands](reports/semifinal-v1/all_commands.csv), [correct accepted commands](reports/semifinal-v1/accepted_commands.csv), [wrong acceptances](reports/semifinal-v1/wrong_acceptances.csv), [valid failures](reports/semifinal-v1/failed_commands.csv), [negatives](reports/semifinal-v1/negative_results.csv).
- [Model card](MODEL_CARD.md), [data/provenance](DATASET.md), [research findings and archive recovery](RESEARCH_SUMMARY.md), [all-ten-use-case map](configs/intents.json).
- [Architecture.ipynb](Architecture.ipynb) imports ordinary code and displays the same measured reports; it is not an implementation loader.
- Run source tests with `python -m pytest -q final_suite/tests tests`. Seven historical payload checks remain explicitly skipped. `python final_suite/run_all.py --qualify` reaches a fail-closed release decision without old recovery ZIPs.
- The optional `visualizer/` stays a development tool. `visualizer/export_assets.py --help` exports the current selected checkpoint into its existing recipe. `visualizer/build_kernel.py` requires clang with wasm32 support. Browser/WASM interaction is not part of the fresh acceptance evidence.
- [Cleanup manifest](reports/cleanup_manifest.csv) identifies every retired tracked file and its byte-preserving archive. All 433 pre-existing untracked files and the historical installed bundle remain untouched.

The originally supplied [CODEX_HANDOFF.md](CODEX_HANDOFF.md) remains the execution specification, not a claim that every research or physical-hardware gate passed. Missing independent human/Pi/noise/CV evidence and unimplemented command categories are recorded explicitly.
