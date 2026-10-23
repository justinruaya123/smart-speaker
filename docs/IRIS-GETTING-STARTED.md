# Iris getting started

This publication uses the **original 258,912-parameter, width 160, epoch 44 FP32** VCM at `submission/models/optionb-tiny`. **The synthetic wake smoke gate failed: native 1/32 positives, trained companion 21/32 with 2/68 false wakes. No replacement was promoted; this is not a working-acoustic-release claim.** It does not select the older 969k Iris bundle or launch training. Read the [new model card](MODEL-CARD-iris-epoch44.md) for fixed scope, exposed human errors and failed qualification.

## Install and check

Use a clean project checkout, Git, Python 3.13 and uv. The UI toolchain is Node24.13+/pnpm12.5.1 with the committed lock. Do not put keys in command history or Git. First-time setup/downloads may require the network; the command recognizer itself uses local CPU models.

The inference launcher owns its setup path. For an already installed pinned runtime, use that environment's Python explicitly (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on POSIX):

```sh
python inference.py --check
python inference.py --check --no-setup
python -m scripts.optionb_epoch44 verify --load
```

The first command may prepare the environment. `--no-setup` forbids setup rather than silently changing packages. The check must identify epoch 44/258,912, the frozen graph/policy and CPU provider. It opens no microphone, trains nothing, executes no action and does not claim acoustic qualification. Artifact verification checks fifteen original payloads; ONNX loading produces one[1,32] feature fixture.

## UI and silent recorded audio

```sh
python inference.py --mode ui --no-setup
python inference.py --wav /absolute/path/to/mono16k.wav --no-setup
```

Use the actual UI microphone settings to select an input; supported device/audio controls are discoverable with `python inference.py --help`. The UI uses the combined gateway so `/ws/runtime` stays on the same origin. CLI/WAV/checks are always action-free and silent; an inherited playback flag cannot make benchmark/replay audio actuate or speak.

The sequence is **Hey Iris → one supported command → guarded action → visible receipt**. The VCM selects one complete class, then the existing experimental owner/overlap gate may veto the application. An accepted raw command and a successful action are not interchangeable. Tentative guesses never execute. A rejected capture/ownership/slot decision must remain rejected.

Start with actions disabled. Copy [the root example](../config.example.toml) only to your ignored private `config.toml` and inspect the current launcher/config help before explicitly enabling microphone actions. Provider keys, tokens and precise home data stay server-side and outside published assets. The user-authorized HA allowlist is literally:

```dotenv
HA_ENTITY_IDS=light.hue_color_lamp_1,light.dining_light,camera.dining_area_main_camera_tc70_mainstream
```

All recognized light commands target the Hue lamp; CALL selects the dining camera **video only**, not telephony or arbitrary contacts. TEMPERATURE is **always a synthetic target**, with a separate read-only observed sensor if configured. No climate service write is permitted. OpenWeather is the weather source; caches keep real observation/retrieval times. Missing services may use a visible “Synthetic — no device changed” fallback, never a fabricated current observation or a replacement for an unknown real-provider outcome. No automatic retry.

Local timers, alarms and reminders are real persisted server jobs. Alarms use Asia/Manila. Static presentation examples create none of those jobs. Provider receipt/acknowledgment does not itself prove observed physical state.

## Prepare original evaluation inputs without fitting

Published metadata is not audio. Use the exact original source checkout/environment to reconstruct permitted inputs; do not substitute invented WAVs or use another model's manifests. Existing [pinned original-source instructions](REPRODUCIBILITY-optionb-epoch44.md) specify the environment locks, source identities and platform limits.

From the publication checkout, choose a new source destination:

```sh
git clone --single-branch --branch snapshot/optionb-tiny-epoch44 \
  https://github.com/justinruaya123/smart-speaker.git .cache/optionb-epoch44-source
git -C .cache/optionb-epoch44-source checkout --detach \
  3c05b3beb46f06c3108089e0d8c79474670c6cb7
```

From **that source checkout**, after its pinned environment setup:

```sh
uv run --no-sync python -B -m scripts.submission.optionb_reproduce prepare \
  --splits test holdout --output .cache/iris-publication-evaluation/run-01
```

The unchanged preparer verifies upstream revision/Parquet/WAV/decoded-PCM hashes. It does no fitting. Use a new output path and retain its receipt. Restricted source audio remains local under its terms. The full TEST/HOLDOUT preparation has 4,614 rows (4,418+196), not 4,614 human trials; the original-human subset contains1,146 recordings. Ambiguous references stay explicitly unscored.

From the publication checkout, the integrated wrapper can prepare and export in one action-free command, using the existing pinned original interpreter:

```sh
python -B reproduce.py prepare-evaluation \
  --source-checkout PINNED_SOURCE --source-python PINNED_PYTHON \
  --output .cache/iris-publication-evaluation/run-01
```

If you already ran the original preparer above, export without downloading or fitting again:

```sh
python -B reproduce.py export-evaluation --source-checkout PINNED_SOURCE \
  --output .cache/iris-publication-evaluation/run-01
```

`PINNED_SOURCE` and `PINNED_PYTHON` are actual absolute source/interpreter paths. `--output` is relative to that original source checkout. The default export folder is `PINNED_SOURCE/.cache/iris-publication-evaluation/run-01/portable/`, containing `holdout-196.jsonl`, `human-test-holdout.jsonl` and `manifest-receipt.json`. Export rechecks original reference/WAV/decoded-PCM identities; existing output is not overwritten. Use `human-test-holdout.jsonl` for the 1,146-row human cohort and **the pinned source checkout as `--data-root`**. Ambiguous rows remain unscored, and repeats do not inflate acoustic denominators.

## Benchmark the published artifact

Once you have a real validated manifest and its source root, from the publication checkout:

```sh
python benchmark.py --model-choice epoch44 \
  --manifest PATH --data-root ROOT --output NEWDIR --repeats 3
```

`PATH`, `ROOT` and `NEWDIR` are placeholders to replace with actual paths. The output directory must not already exist. Three timing repeats do not triple acoustic denominators. The driver uses the original graph/policy; it must emit metrics/trials with source/model/policy hashes, hardware, timing scope, raw/accepted complete commands, wrong/negative accepts and ambiguous exclusions. It never captures, plays audio, calls providers, chooses thresholds or fits a model.

The original preserved PC repeat is historical 5.36/8.69 ms p50/p95, not the current runtime latency. The separately measured2026-10-24 reference-driver run is 9.34/15.34 ms **under unquiesced publication load**, not a new-adapter, idle-host or end-to-end result. Current adapter measurements require their own receipt. Pi remains unmeasured.

## Reproduce a fit: separate scientific task

Do not use benchmarking as a claim that the selected weights were independently refitted. [The pinned reproduction runbook](REPRODUCIBILITY-optionb-epoch44.md) records two repeatable CUDA fits that selected epoch 28, not original CPU epoch 44. Their exact comparison to the original failed; the absent success lock is preserved. Artifact verification, source reconstruction, GPU repeatability and original CPU exact retraining are four distinct statements.

The previous `start.py`, older root model card and historical continuation workflow describe earlier product defaults. They remain evidence, not the quickstart for the selected 258k release. Do not copy their969k/2M numbers or promote their models into this launch.

## Presentation and developer checks

The [static deck](site/index.html) is six native HTML chapters backed by [generated JSON](site/assets/iris.json). Serve it from a static host; for local review:

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory docs/site
```

Open localhost port8765. Viewing needs only a browser, not Python, a GPU, backend or microphone. Python above is just the optional authoring server. Arrow/Page keys and 1–6 navigate, F toggles fullscreen; use-case tabs use arrows/Home/End. Current chapter persists locally. Reduced motion shows actual rendered posters; animation is explicit opt-in. The real Manim Slides exports load only on “Inspect stages.”

```sh
python -B scripts/iris_publication_data.py --check
python -B -m unittest scripts.test.test_iris_publication_data -v
python -B presentation/validate.py
```

The isolated [presentation authoring instructions](../presentation/README.md) explain real rendering/export, existing-Playwright browser QA, viewer-license notices and the actual compatibility fallback. No Manim dependency is installed into the scientific/runtime environment. CI validates prebaked assets and deploys only `docs/site` after a validated push to main; PRs cannot deploy. Release and rollback are recorded separately by the release owner.
