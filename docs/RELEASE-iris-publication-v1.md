# Iris publication v1 — core execution log

## Scope and frozen baseline

- Authorized implementation: plan `2026-10-24_055828-iris-main-publication.md`, tasks 3–9 and 15–16. Core branch `codex/iris-publication`, base main `c0628940400642f204ff98e26908549b1c07e41c`.
- Frozen original source remains read-only at revision `3c05b3beb46f06c3108089e0d8c79474670c6cb7`. Its existing `.venv/Scripts/python.exe` (CPython 3.13.14, NumPy 2.3.5, ORT 1.23.2) is invoked explicitly; no package synchronization or scientific lock changes.
- Publication default is the original 258,912-parameter, width160, FP32 epoch44 graph. The five independent SHA-256 pins live in `scripts/runtime/iris_epoch44.py`; the original recipe, graph, policy, normalizer, class order, checkpoint and fifteen snapshot payloads remain unchanged. Wake/VAD/owner are separately identified companions, not part of the VCM count.
- Qualification remains false. Original CPU exact retraining is not verified; prior GPU exact comparison failure and absent CPU-success lock are preserved. Loading a graph or running a software fixture does not prove speech accuracy, physical actions, microphone reliability or Pi performance.

## Executed checks and traceable changes

All pytest commands use the explicit original Python, `-B -m pytest -p no:cacheprovider`, and `--basetemp C:/Users/Justin/AppData/Local/hermes/cache/scratch/iris-…-tests`. Numeric-library defaults are one thread before import; no provider or microphone is opened by these checks.

| Change | Observed RED / characterization | Observed GREEN |
|---|---|---|
| `dea8ec6`: deliberate feedback ignore exceptions only | `git check-ignore --no-index -v` identifies seven named pair exceptions; raw WAV, `.env`, `.cache` and `.runtime` remain ignored | First commit contains only `.gitignore` |
| `f06029f`: frozen adapter and identity | Two identity characterization tests passed; 52 adapter cases failed because the new module was absent | `test_iris_epoch44_publication.py test_iris_epoch44_runtime.py`: **54 passed**; actual ONNX `features,lengths -> logits` run and direct decision parity |
| Existing-worker integration | Five new worker cases failed: old loader sought nonexistent `bundle.json`, base `model_output` was lost, `--no-audio` and factory were absent | `test_iris_epoch44_worker.py test_iris_worker.py test_iris_epoch44_runtime.py`: **66 passed** |

`python -B -m scripts.optionb_epoch44 verify --load` returned `frozen_epoch44_verified`, 15 payloads, parameters 258912, selected epoch44, output `[1,32]`, CPU provider, zero updates and no capture/actions.

The frozen frontend was not edited. Chunk-vs-batch frontend characterization found one float32 ULP difference (1.1920929e-7); the shape/parity check uses an explicit 2e-7 absolute tolerance, while same-feature graph/logit checks are exact. The existing owner frontend imported SciPy eagerly even for native16kHz; the selected environment has no SciPy. Its unchanged resampling algorithm now imports SciPy only when resampling is requested. No resampling or owner-feature math changed and no environment was upgraded.

## Remaining acceptance owned outside this core work

Parent independent review, full original-audio evaluation, UI lifecycle integration, physical playback/capture acceptance, synthetic-wake receipt, Pi/live-human/provider qualification and any publication are separate. No independent review, push, merge or Pages deployment is claimed here.
