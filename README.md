# Streaming VCM: diagnostic semifinal

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
