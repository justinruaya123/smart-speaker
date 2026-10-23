# Original OptionB epoch44 on main

## Explicitly separate models and checks

Iris remains the default model, product and `reproduce.py` workflow. The original **258,912-parameter /width160 /epoch44 /FP32 ONNX** OptionB model is an opt-in, unqualified, action-disabled comparison artifact on main. It is not substituted into Iris's 328-output recognizer or speaker/action pipeline.

- Original five-file bundle: `submission/models/optionb-tiny/`.
- Original recovery checkpoint: `submission/checkpoints/optionb-tiny-selected.pt`.
- Exact original snapshot and metadata: `submission/optionb-tiny/snapshot.json` and `inputs/`.
- Separate tested CUDA reference: `submission/optionb-tiny/gpu-reference.json`.
- Original source revision: `3c05b3beb46f06c3108089e0d8c79474670c6cb7` on the preserved `snapshot/optionb-tiny-epoch44` branch.

The original snapshot's ten source bindings apply to that pinned original checkout, **not current Iris main**. Main's newer schema, worker, dependencies and model must not be overwritten to make an old source hash pass. No old research tree, runtime, frontend or trainer is merged into Iris.

## Verify the epoch44 bytes from main

From a clone of main, Python 3.11+ with only its standard library can verify all fifteen original model/checkpoint/metadata payloads:

```sh
python -m scripts.optionb_epoch44 verify
```

With NumPy and ONNX Runtime installed in the selected Python environment, load and execute the exact original graph as a bounded software fixture:

```sh
python -m scripts.optionb_epoch44 verify --load
```

Use the environment's Python explicitly if needed (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on POSIX). The original pinned NumPy2.3.5/ONNX Runtime1.23.2 stack was exercised. The command does not install packages, decode a recording, open a microphone, train, use providers or enable actions. Its hash-verified one-batch feature fixture is not acoustic-quality evidence. `verify` never reports freshly reproduced weights or source bindings for current Iris.

Both commands return zero only for their declared artifact/load checks. Scoped Git attributes preserve original bytes under Windows CRLF settings. Unknown, modified, linked or missing payloads fail closed; changing a mutable manifest cannot approve a replacement graph.

## Execute the original GPU recipe without changing Iris

The scientifically frozen code is deliberately executed in its **own pinned Git checkout and root environment**. This avoids importing Iris's newer shared schema/worker or synchronizing its active environment. From main, choose a new ignored destination:

```sh
git clone --single-branch --branch snapshot/optionb-tiny-epoch44 https://github.com/justinruaya123/smart-speaker.git .cache/optionb-epoch44-source
git -C .cache/optionb-epoch44-source checkout --detach 3c05b3beb46f06c3108089e0d8c79474670c6cb7
```

From that new snapshot checkout, the tested **Windows x86_64/CPython3.13.14** setup and source-only GPU fit are:

```sh
uv sync --locked --python 3.13.14 --group training --group benchmark
uv pip install --python .venv/Scripts/python.exe --require-hashes --no-deps --requirement submission/optionb-tiny/testing.lock
uv pip install --python .venv/Scripts/python.exe --require-hashes --no-deps --requirement submission/optionb-tiny/accelerator-windows-cu128.lock
uv run --no-sync python -m scripts.submission.optionb_reproduce verify
uv run --no-sync python -m scripts.submission.optionb_reproduce prepare --output .cache/optionb-source-repro-v1/run-01
uv run --no-sync python -m scripts.submission.optionb_fit --output .cache/optionb-source-repro-v1/run-01 --device cuda
```

Keep all downloads, regenerated WAV/PCM, features and fitting outputs in this new run. Do not copy another run's audio/features/checkpoints to represent a fresh build. Explicit CUDA forbids CPU fallback; the original lower-level fitter separately retains its tested `auto` initial-device fallback. This CUDA wheel lock is Windows/CPython3.13-specific; it is not a qualified Linux/MPS/XPU installation recipe.

**Important:** the final fit command still exits1 after completing training/export when it compares the GPU model to the original CPU epoch44 model. The two tested GPU fits select epoch28, not44; their fresh DEV-selected policy is .55/.30, not .50/.10. The original strict CPU comparison and absent `selection-before-test.json` are preserved. Do not chain through that failure as an original-epoch44 success or forge/rename a lock. Network download failures are also failures, not CUDA crashes or successful builds; preserve their logs and retry only hash-verified transfers. The published original run records the WinError10054 auxiliary-download recovery explicitly.

## Verify GPU repeatability independently

After actually completing a new full GPU fit, return to main and verify its retained outputs against the independently pinned GPU reference:

```sh
python -m scripts.optionb_epoch44 verify-gpu --source-checkout .cache/optionb-epoch44-source --fit .cache/optionb-source-repro-v1/run-01/fit
```

This read-only checker requires the exact clean source revision, all original asset/source hashes, the complete failure-evidence bindings, all selected/foundation/model/normalizer/class/policy bytes, fresh DEV/calibration predictions and policy grid, all48 epochs' losses/DEV results excluding timing, final draw counts,4,608 updates and recorded CUDA execution. Incomplete training, changed tensors/graphs/records, CPU execution, missing bindings, a changed reference or an invented original success lock reject.

CUDA evidence must also agree across `training-device.json`, `training-result.json` and `training-comparison.json`: model/input/gradient/optimizer-moment and DEV scoring placement, a successful synchronized compute probe, and the corresponding comparison checks are mandatory. Rebinding a mutable failure receipt cannot approve contradictory CPU placement. CPU optimizer scalar counters and the saved CPU checkpoint/export handoff remain valid and are not mislabelled CPU training.

Exit0 means **GPU repeatability verified**, explicitly **not CPU epoch44 retraining**. This checker never performs fitting itself. The original comparison bundle retains .50/.10 for comparison; its freshly selected .55/.30 is recorded separately and not installed. Future GPU/platform stacks are not promised bit-identical results.

## Published evidence and limits

Two independent original-source/environment/data acquisitions completed48-epoch CUDA runs. All37 tensors, both checkpoint bytes, the five-file bundle, every epoch's losses/DEV results excluding timing,466 DEV/calibration prediction records and policy grid matched exactly. Their original-source selected Python suite passed463 cases. Those are prior full-run observations, not newly invented training by the main-branch checker.

HF dataset `justinruaya123/ai231-smart-speaker`, branch `optionb-tiny-reproduction-v1`, immutable revision `009b87cb6cb2297073797b8c9571925f70fe92dd`, folder `reproduction/optionb-tiny-v1/`, preserves the original frozen model and separate GPU comparison model. All49 published files were anonymously downloaded and hashed. Detailed immutable Git evidence is under `experiments/optionb-clean-pipeline-v2/` at the original source revision above; HF carries matching numerical evidence. Current main keeps the compact reference and additive integration receipt.

Original data/model/source terms remain applicable; no recording rights are granted by this mirror. Main's existing [dataset provenance](DATASETS.md) and the original snapshot's source notices distinguish public metadata from restricted recordings. No held-out PCM/native/PC acceptance, fresh human acoustic qualification, Pi, live microphone/provider operation, installed-default replacement or action enablement is claimed by either passing verification command.

## Main integration software scope

The Windows integration check against base `2c669589229619d0e282ebda621dc9ae3002951e` reproduced the same37 existing Iris failures on both unchanged main and the additive candidate. These include Windows path/EOL assumptions and local QA prerequisites; they are not new epoch44 regressions, nor a claim that Iris's full reproduction passed. The original failed GPU-provenance review, superseded verifier source and baseline-aware evidence are retained under `experiments/optionb-epoch44-main-v1/`. The opt-in verifier's focused tests and the original-source GPU results are separate from this Iris baseline.
