# OptionB tiny snapshot and GitHub reproduction

## Frozen model

The selected finite OptionB candidate is the original **width160 /258,912-parameter epoch44** atomic-command CNN, with its exact 16kHz/40-bin frontend, human-TRAIN normalizer, 32-class order and separate **.50 posterior /.10 margin** demo policy. No failed fine-tune, larger encoder, ASR, transcript parser or independent intent/slot acceptance gate replaces it.

The snapshot is versioned on **`snapshot/optionb-tiny-epoch44`**, separate from the historical `codex/final-submission` branch's push-triggered 2M release workflow. The snapshot does not change installed runtime defaults, qualification or action permissions.

- Bundle: `submission/models/optionb-tiny/` (all five original files).
- Selected recovery checkpoint: `submission/checkpoints/optionb-tiny-selected.pt`.
- Model/data/source identities: `submission/optionb-tiny/snapshot.json`.
- Metadata-only compressed membership, references, original tiny feature hashes and golden predictions: `submission/optionb-tiny/inputs/`.
- Original training: CPU, four Torch threads, seed53, six human-only epochs then42 mixed epochs,96 batches of32 per epoch. Preserve this recipe when testing training reproducibility; using CUDA is a different numerical run. The measured original fit took2,987.91 seconds.

The input snapshot preserves exact ordered membership and labels. It removes obsolete local audio/feature paths and the older2M feature-cache hashes; those are not tiny-model fitting inputs. The tiny fit freshly extracted40-bin features with BLAS limited to one thread. Its24,566 original TRAIN/DEV tensor hashes and original1,889-human normalization membership are retained separately. Source records identify ten pinned Parquet shards and184 auxiliary synthetic DEV recordings. No raw recordings, feature tensors, private configuration or environment payloads are committed. Source-specific data/model terms remain applicable; see [dataset terms](DATASETS.md) and the original reports.

## Fresh-checkout baseline

Clone the dedicated branch into a **new directory**. Do not copy an old `.venv`, `.cache`, recordings or feature tensors into the initial test. Use the repository's unchanged lockfile and a new root environment:

```sh
uv sync --locked --python 3.13.14 --group training --group benchmark
uv run --no-sync python -m scripts.runtime.optionb_worker --bundle submission/models/optionb-tiny --check
```

The second command verifies the exact tracked candidate and loads CPU ONNX without operating a microphone or providers. The older root `reproduce.py`, `benchmark.py` and submission package builders still target the historical2M/935k workflows; they are not evidence of tiny-model reproduction.

### Required setup for source fitting/export and tests

Model loading needs only the first locked setup above. **Full CPU fitting/export additionally requires the separately hashed reproduction helpers:** the preserved canonical exporter lives in the original model-test module and imports pytest. Installing only the scientific training group is not sufficient for export. Keep the original scientific `pyproject.toml`/`uv.lock` unchanged; install the declared helper lock into the **same new root environment** before any fit:

```sh
uv sync --locked --python 3.13.14 --group training --group benchmark
uv pip install --python .venv/Scripts/python.exe --require-hashes --no-deps --requirement submission/optionb-tiny/testing.lock
```

The explicit interpreter path shown is for the currently tested Windows checkout. Use `uv run --no-sync` for subsequent stage commands so a scientific-only sync does not remove these separately declared helpers. Their lock includes the complete dependency closure with hashes and pins packaging to the already-installed scientific version. The fit entry point rejects missing export helpers **before** feature extraction or optimizer work; it does not discover this import after48epochs. [Independent correction closure](../experiments/optionb-source-repro-v1/fit-review-v2.json) and75parent-owned/canonical software tests pass. The full corpus fit remains a separate acceptance run, not a result inferred from those tests.

For a new immutable run directory, execute the portable source stages in order:

```sh
uv run --no-sync python -m scripts.submission.optionb_reproduce verify
uv run --no-sync python -m scripts.submission.optionb_reproduce prepare --output .cache/optionb-source-repro-v1/run-01
uv run --no-sync python -m scripts.submission.optionb_fit --output .cache/optionb-source-repro-v1/run-01
```

The fit uses only original TRAIN/DEV for extraction/fitting and human DEV/calibration for policy reconstruction; reserved test/holdout metadata checks do not decode their PCM. It verifies all24,566 freshly extracted TRAIN/DEV tensor hashes and the exact original1,889-human normalizer before entering the unchanged four-thread CPU fit. It then compares selected epoch44, configuration, draw counters, every tensor, original graph bytes and the138-point policy grid. A mismatch leaves failure/comparison artifacts and cannot create a successful selection lock or replace the published model. The original measured fit took49.8minutes; actual rerun duration can vary. Choose a new run ID after any failed attempt, preserving earlier outputs.

**Full tiny-model clean-clone reproduction is not yet verified at this snapshot.** The old tiny research CLIs refer to ignored study caches and immutable historical source locks. Running them from a clean checkout is an acceptance test, not permission to silently copy those caches, weaken hashes or overwrite their receipts. The requested GitHub experiment must record the original failing command/stage before any portable follow-up fix. Later reconstruction must download exact public source bytes, verify original WAV/PCM identities, regenerate the tiny TRAIN/DEV features and normalizer, and compare actual fresh inference/metrics with the golden expectations. A training reproduction must perform all4,608 updates and compare selected epoch/tensors separately from source/model inference replay. Elapsed time and latency need not be bitwise equal.

## Actual GitHub baseline — 2026-10-23

The dedicated branch was pushed and freshly cloned at **`0591fbe2c3ed7349d5f4c12aa31ed0c0845322e9`**. A new root environment installed the unchanged scientific lock with CPython3.13.14, PyTorch2.10.0+cpu and NumPy2.3.5. The tracked five-file candidate loaded successfully with the exact graph hash and disabled qualification/actions. No old environment, recordings or derived features were copied into this acceptance run.

Eight original invocations were actually attempted: tiny preparation/fitting, demo selection, release replay, PC timing, report verification, metadata verification and pytest. [The preserved baseline receipt](../experiments/optionb-source-repro-v1/baseline.json) records their actual failures: historical ignored-cache bindings, immutable old study output/selection locks and pytest absent from the scientific environment. These are not successful full reproduction runs. Test tooling is now separately pinned in `submission/optionb-tiny/testing.lock`; the original scientific manifest and lock stay untouched.

The fresh checkout downloaded and independently rehashed **ten pinned Parquet shards /2,122,800,782 bytes**; see [source-download evidence](../experiments/optionb-source-repro-v1/source-downloads-v1.json). This accessed held-out payload bytes inside the public sources, but did not decode/infer their PCM. Auxiliary DEV WAV retrieval and the full source-derived fitting/evaluation pipeline remain separate stages.

[UI software evidence](../experiments/optionb-source-repro-v1/ui-software-v1.json) records fresh pinned Node24.13.0/pnpm12.5.1 dependencies, **117 passing tests**, clean Svelte/server checks and the ordinary `check`, `test` and `build` wrappers, all exit0 with a clean source tree. All three compiled adapters match the earlier frozen hashes. The initial child subtask used a no-write bake adaptation; the parent then exercised the literal build wrapper and verified its two generated tracked files remained byte-identical.

The first independent preparer review [failed on three reproduced issues](../experiments/optionb-source-repro-v1/preparer-review-v1.json):26 frozen training locators omit URL fields, derived writer paths could traverse symlinks/junctions, and `.cache` alone was incorrectly accepted as a run. The separate correction passed156 focused/original regression tests and [independent closure](../experiments/optionb-source-repro-v1/preparer-review-v2.json). Its actual fresh-checkout execution reconstructed **23,315 TRAIN /1,251 DEV /210 calibration rows**, verifying original WAV/decoded-PCM identities and ordered labels without copying old audio or features; [preparation evidence](../experiments/optionb-source-repro-v1/preparation-v1.json) records the acceptance.

The initial CPU fit/export stage passed74 bounded tests, including a real two-step optimizer fixture; [independent review](../experiments/optionb-source-repro-v1/fit-review-v1.json) then reproduced a late missing-pytest export dependency. The separately declared hashed-helper setup and early canonical-exporter import correction passed75 tests and [independent closure](../experiments/optionb-source-repro-v1/fit-review-v2.json). The original exporter and scientific lock were not changed.

Reviewed fitting revision **`d00a3c40a26dc4fa1076d38923ae832dfa4058ae`** was pushed and pulled into the new checkout, where **299 selected OptionB Python tests passed**. Actual source-only extraction matched **all24,566 TRAIN/DEV tensor hashes** and the exact original **1,889-human normalizer** before optimization; see [feature/normalization preflight evidence](../experiments/optionb-source-repro-v1/feature-preflight-v1.json). The full48-epoch CPU fit is running from random initialization. Its selected checkpoint, export,138-point policy reconstruction, complete inference/wording metrics and separate PC timing repeat are **not yet accepted**. Test/holdout PCM remains excluded before a successful exact selection lock. No software fixture, partial epoch or matching feature gate is presented as full training reproduction or fresh acoustic qualification.

## What stays unqualified

[The current release report](RELEASE-optionb-tiny-v1.md) retains exposed-cohort metrics and known errors. Reproducing them does not establish fresh-blind accuracy, independent listener truth, all93 human wording cells, live capture or Pi readiness. There are no new independent-speaker recordings. Pi tests are excluded; actions remain disabled, and the snapshot is not a promoted/installed release.
