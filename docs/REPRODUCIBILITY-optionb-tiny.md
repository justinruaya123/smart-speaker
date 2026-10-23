# OptionB tiny snapshot and GitHub reproduction

## Independent GPU rebuild and HF publication — 2026-10-24

**PASS: the same GPU model was rebuilt independently on the pinned CUDA stack. This is not a pass against the original CPU epoch44 reference.** A second new GitHub checkout at `ad4d6262e7f91b2cb4cb4f75a1b1b082d6b088f3` started without an environment, audio, feature cache or checkpoints. Its own CPython3.13.14/scientific/helper/CUDA environment was installed from the existing locks. No application source or new reproducibility framework was added for this run.

[The operational receipt](../experiments/optionb-clean-pipeline-v2/summary.json) records eight required TRAIN/DEV Parquet downloads plus184 auxiliary files,23,315TRAIN/1,251DEV/210calibration reconstructed rows with exact WAV/PCM,24,566 exact feature hashes and the1,889-human normalizer. Two preparer attempts hit remote WinError10054; only61 missing auxiliary files were then downloaded through an authenticated client and all pinned hashes checked before resuming the unchanged preparer. This disclosed recovery is not an untouched one-command success.

The second full required-CUDA run completed48epochs/4,608updates and selected epoch28, just like the first. [The comparison](../experiments/optionb-clean-pipeline-v2/gpu-repeat-comparison.json) verifies **all37 tensors, selected/foundation checkpoint bytes, all five exported bundle files, all48 epochs' losses/DEV results excluding timing, and final draw counts exactly**. [Fresh prediction/policy verification](../experiments/optionb-clean-pipeline-v2/runtime-and-policy-comparison.json) additionally verifies byte-identical466 DEV/calibration prediction records and the complete policy-selection grid. The passing claim is limited to the tested Windows/Torch2.10.0+cu128/CUDA12.8/RTX3080 recipe, not other GPUs/platforms.

Both stock fit commands still exit1 at `exact_comparison_and_lock` because neither reproduces the original CPU epoch44 model. The frozen preview adapter intentionally rejects the new comparison graph, while the existing general tiny Runtime loads and executes it. The comparison bundle retains the original `.50/.10` policy; fresh selection chooses `.55/.30` but is not applied or installed. Original exact-fit replay/native/PC acceptance remains blocked, no lock is forged, and the original best model remains unchanged.

**Published to the existing HF dataset** `justinruaya123/ai231-smart-speaker`, new branch `optionb-tiny-reproduction-v1`, immutable revision **`009b87cb6cb2297073797b8c9571925f70fe92dd`**, folder `reproduction/optionb-tiny-v1/`:

- `frozen/`:26 exact original Git model/source/snapshot payloads, including the original epoch44 checkpoint and bundle.
- `gpu-rebuild/`: the independently identical, non-promoted epoch28 comparison checkpoint and five-file bundle.
- `rebuild-evidence/`, `manifest.json` and `gpu-rebuild-manifest.json`: source, numerical and scope evidence.

[Publication readback](../experiments/optionb-clean-pipeline-v2/hf-publication.json) verifies all49 files by anonymous re-download and SHA256, unchanged original dataset branches, and unchanged frozen model bytes. [The pulled-artifact runtime check](../experiments/optionb-clean-pipeline-v2/HF-pulled-runtime-proof.json) loads the downloaded original through the unmodified frozen worker and executes the downloaded GPU comparison graph through the general Runtime; this is software loading, not acoustic qualification. The second checkout also passes463 selected CUDA Python cases without failures/skips and verifies all15+10 original scientific identities. No held-out PCM inference, new native/PC timing, Pi, live microphone/provider execution, installed-default change or qualification is claimed.

## Frozen model

The selected finite OptionB candidate is the original **width160 /258,912-parameter epoch44** atomic-command CNN, with its exact 16kHz/40-bin frontend, human-TRAIN normalizer, 32-class order and separate **.50 posterior /.10 margin** demo policy. No failed fine-tune, larger encoder, ASR, transcript parser or independent intent/slot acceptance gate replaces it.

The snapshot is versioned on **`snapshot/optionb-tiny-epoch44`**, separate from the historical `codex/final-submission` branch's push-triggered 2M release workflow. The snapshot does not change installed runtime defaults, qualification or action permissions.

- Bundle: `submission/models/optionb-tiny/` (all five original files).
- Selected recovery checkpoint: `submission/checkpoints/optionb-tiny-selected.pt`.
- Model/data/source identities: `submission/optionb-tiny/snapshot.json`.
- Metadata-only compressed membership, references, original tiny feature hashes and golden predictions: `submission/optionb-tiny/inputs/`.
- Original training: CPU, four Torch threads, seed53, six human-only epochs then42 mixed epochs,96 batches of32 per epoch. Preserve this recipe when testing training reproducibility; using CUDA is a different numerical run. The measured original fit took2,987.91 seconds.

The input snapshot preserves exact ordered membership and labels. It removes obsolete local audio/feature paths and the older2M feature-cache hashes; those are not tiny-model fitting inputs. The tiny fit freshly extracted40-bin features with BLAS limited to one thread. Its24,566 original TRAIN/DEV tensor hashes and original1,889-human normalization membership are retained separately. Source records identify ten pinned Parquet shards and184 auxiliary synthetic DEV recordings. No raw recordings, feature tensors, private configuration or environment payloads are committed. Source-specific data/model terms remain applicable; see [dataset terms](DATASETS.md) and the original reports.

## GPU-preferred follow-up

The user subsequently requested **GPU-preferred training with CPU fallback**, superseding the CPU-only rerun preference below. The original frozen model and original scientific lock remain unchanged. The actual RTX3080 machine initially had a CPU-only Torch wheel; its partial CPU fit was stopped after39 epochs and retained as [interrupted evidence](../experiments/optionb-source-repro-v1/cpu-fit-interrupted-v1.json), not a full reproduction success.

For the tested **Windows x86_64 /CPython3.13 /CUDA12.8** environment, install the unchanged scientific lock and required hashed helpers, then apply the separately hash-pinned accelerator wheel to that same root environment:

```sh
uv pip install --python .venv/Scripts/python.exe --require-hashes --no-deps --requirement submission/optionb-tiny/accelerator-windows-cu128.lock
```

This command has run successfully. [The original model's real CUDA compute probe](../experiments/optionb-source-repro-v1/cuda-compute-probe-v1.json) verifies GPU model/input/length/target/loss/gradient/optimizer-moment placement, a finite backward pass and changed weights. It does not establish completed corpus training. The GPU-aware fitting entry point is implemented and independently reviewed under the [new accelerator protocol](../experiments/optionb-source-repro-v1/accelerator-protocol-v1.json): `auto` must choose a genuinely working accelerator, explicit CUDA must fail if unusable, and CPU is a recorded initial-device fallback rather than a silent retry after an arbitrary fitting error. No untested MPS/XPU platform qualification is implied. Cross-device results may differ even with identical seeds; retain the exact frozen comparisons and all mismatches instead of claiming GPU/CPU tensor identity or replacing the published model.

The device-aware fitter now passes **88 real-CUDA software tests** and **83 CPU tests with5 CUDA-only skips**, including actual GPU DEV scoring, CPU checkpoint handoff and canonical export. A five-step CPU-port fixture matches the original CPU tensors, draws, losses and schedule exactly. [The bounded software evidence](../experiments/optionb-source-repro-v1/accelerator-software-v1.json) retains the measured CPU/GPU fixture difference rather than issuing a false exact-fit lock. [Independent GPU review](../experiments/optionb-source-repro-v1/accelerator-review-v1.json) passed, including88 independently executed software cases and real CUDA fixtures; the full corpus run remains a separate acceptance stage. The integrated precommit suite passes463 cases with CUDA and458 with5 CUDA-only skips in the CPU environment.

The implemented accelerator backend is **CUDA**, with tested CPU fallback; other native GPU backends are not implemented or qualified by this port. The fitter accepts `--device auto` by default, explicit `--device cuda` to forbid fallback on the verified RTX3080, and `--fit-name fit-gpu-01` to retain the interrupted `fit/` while reusing the same fresh checkout's verified source manifests. All raw features are still extracted and hash-checked again. The real full-corpus outcome is recorded below, separately from software fixtures.

### Actual full GPU run — 2026-10-24

Reviewed revision **`d675f7c2d06541701bbdfd1f632079f9c30ee2cc`** was pushed to the snapshot branch, pulled into the existing genuinely new checkout, and passed88 fresh-checkout CUDA tests. Its same root environment passed `uv pip check`; all15 snapshot assets,10 scientific bindings and4 reviewed fitting files remained unchanged after the run. The actual command was:

```sh
uv run --no-sync python -u -m scripts.submission.optionb_fit --output .cache/optionb-source-repro-v1/run-01 --device cuda --fit-name fit-gpu-01
```

[The full GPU receipt](../experiments/optionb-source-repro-v1/gpu-full-fit-v1.json) binds exact retained/compressed result bytes and the complete48-epoch curve. It verifies **24,566 freshly regenerated exact feature hashes**, the **1,889-human TRAIN normalizer**, **6 human +42 mixed epochs**, **4,608 optimizer steps** and **147,456 draws**, with all final per-kind draw counts matching the original recipe. Model, batch/lengths/targets, gradients, AdamW moments and DEV scoring were on `cuda:0`. Only AdamW's non-capturable scalar step counters and the saved/exported checkpoint reside on CPU; this is not CPU fitting. GPU optimization plus DEV scoring took135.19seconds; the complete preflight/fit/export/comparison stage took367.12seconds.

**Full GPU training and canonical CPU export worked; exact frozen-result reproduction failed.** The selected GPU epoch was28 rather than44. Maximum checkpoint tensor difference was0.7949769311; the GPU graph differs from the frozen graph by up to6.649948121 logits on the466-row DEV/calibration parity cohort, so this is not classified as harmless rounding. The GPU model's own CPU ONNX export passed the original3e-5 parity contract (maximum5.125999451e-6). Fresh DEV/calibration selection chose `.55 posterior /.30 margin`, not frozen `.50/.10`.

The CLI therefore exited1 at **`exact_comparison_and_lock`**, preserving the candidate and all diagnostics without creating `selection-before-test.json`, changing tolerances, replacing the frozen best model, or decoding/inferencing held-out PCM. The dependent exact-fit replay/native/PC pipeline remains blocked, not passed. GPU execution capability, exact source/features, cross-device checkpoint identity and acoustic qualification are separate outcomes. `auto` still permits tested initial CPU fallback on machines without working CUDA; this actual RTX3080 run used explicit required CUDA and never fell back.

[Post-run fresh-checkout software verification](../experiments/optionb-source-repro-v1/fresh-published-software-v2.json) passed **336 published Python cases with CUDA**, **119 UI cases**, the literal pinned Node24.13.0/pnpm12.5.1 `check`/`test`/`build` wrappers, and loading of the unchanged frozen epoch44 graph/policy. The checkout remained clean. The separate precommit463-case integration includes127 replay correction cases that were initially unpublished pending independent closure; it is not misreported as that earlier published checkout's test count.

### Replay software review closure

The first full replay review [failed on R1-R3](../experiments/optionb-source-repro-v1/replay-review-v1.json); its exact source/test bytes remain [archived](../experiments/optionb-source-repro-v1/replay-before-review-fix-v1/manifest.json). The bounded correction rejects conflicting frozen-reader hashes before helper loading, requires complete stored result/evidence/provenance contracts, and rechecks reused audio directory/file containment before PCM access. [Independent closure](../experiments/optionb-source-repro-v1/replay-review-v2.json) reproduced42 genuine original missing-rejection failures and both positive controls, then passed127 corrected cases with no remaining findings. This is software verification, not actual corpus inference.

The published replay CLI offers `selection`, `prepare`, `protocols`, `replay` and `verify`. All inference-producing stages require an exact successful **`<run>/fit/selection-before-test.json`** first. The separate accelerated `fit-gpu-01` run above failed exact comparison and has no such success lock; do not rename/copy its outputs, substitute the original checkpoint, or forge a lock to make this chain pass. For a genuinely new source run the fitter's default output component is `fit`; alternate fit-name downstream selection is not implemented by this CLI. Exact-fit replay/native/PC acceptance remains blocked by the recorded numerical result, not by an unresolved R1-R3 review finding.

### Final published software check — 2026-10-24

The corrected replay was published as **`c09e745ae94277edb40bcefc6d113b9393f2c074`**, then pulled into the fresh checkout. The complete17-file selected Python suite passed **463 cases with CUDA, no failures or skips**, in124.34seconds. All49 installed packages are compatible, the15 snapshot assets and10 source bindings verify, and the working tree is clean. The119-case UI/literal check/test/build result remains applicable because no UI source changed. All GPU, replay and native-harness software reviews are closed.

[The final machine-readable status](../experiments/optionb-source-repro-v1/final-status-v1.json) explicitly records the remaining negative acceptance result: full48-epoch GPU execution works, but original CPU checkpoint/graph/policy equality failed. The actual replay `selection` CLI exits1 for the missing default-fit success lock; neither the interrupted `fit` nor the completed nonexact `fit-gpu-01` has a success lock. No original-audio held-out replay, native replay or PC timing repeat is claimed. Preserve the exact frozen epoch44 model and the nonexact GPU diagnostics as separate results; do not weaken the gates to declare same-result reproduction.

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

The fit uses only original TRAIN/DEV for extraction/fitting and human DEV/calibration for policy reconstruction; reserved test/holdout metadata checks do not decode their PCM. It verifies all24,566 freshly extracted TRAIN/DEV tensor hashes and the exact original1,889-human normalizer before the device-explicit port of the original four-thread recipe. Default `auto` performs a real CUDA compute probe and may choose CPU only as an initial recorded fallback; explicit `cuda` forbids that fallback. It then compares selected epoch44, configuration, draw counters, every tensor, original graph bytes and the138-point policy grid. A mismatch leaves failure/comparison artifacts and cannot create a successful selection lock or replace the published model. The original measured fit took49.8minutes; actual rerun duration can vary. Choose a new run ID after any failed attempt, preserving earlier outputs.

**Full tiny-model clean-clone reproduction is not yet verified at this snapshot.** The old tiny research CLIs refer to ignored study caches and immutable historical source locks. Running them from a clean checkout is an acceptance test, not permission to silently copy those caches, weaken hashes or overwrite their receipts. The requested GitHub experiment must record the original failing command/stage before any portable follow-up fix. Later reconstruction must download exact public source bytes, verify original WAV/PCM identities, regenerate the tiny TRAIN/DEV features and normalizer, and compare actual fresh inference/metrics with the golden expectations. A training reproduction must perform all4,608 updates and compare selected epoch/tensors separately from source/model inference replay. Elapsed time and latency need not be bitwise equal.

## Actual GitHub baseline — 2026-10-23

The dedicated branch was pushed and freshly cloned at **`0591fbe2c3ed7349d5f4c12aa31ed0c0845322e9`**. A new root environment installed the unchanged scientific lock with CPython3.13.14, PyTorch2.10.0+cpu and NumPy2.3.5. The tracked five-file candidate loaded successfully with the exact graph hash and disabled qualification/actions. No old environment, recordings or derived features were copied into this acceptance run.

Eight original invocations were actually attempted: tiny preparation/fitting, demo selection, release replay, PC timing, report verification, metadata verification and pytest. [The preserved baseline receipt](../experiments/optionb-source-repro-v1/baseline.json) records their actual failures: historical ignored-cache bindings, immutable old study output/selection locks and pytest absent from the scientific environment. These are not successful full reproduction runs. Test tooling is now separately pinned in `submission/optionb-tiny/testing.lock`; the original scientific manifest and lock stay untouched.

The fresh checkout downloaded and independently rehashed **ten pinned Parquet shards /2,122,800,782 bytes**; see [source-download evidence](../experiments/optionb-source-repro-v1/source-downloads-v1.json). This accessed held-out payload bytes inside the public sources, but did not decode/infer their PCM. Auxiliary DEV WAV retrieval and the full source-derived fitting/evaluation pipeline remain separate stages.

[UI software evidence](../experiments/optionb-source-repro-v1/ui-software-v1.json) records fresh pinned Node24.13.0/pnpm12.5.1 dependencies, **117 passing tests**, clean Svelte/server checks and the ordinary `check`, `test` and `build` wrappers, all exit0 with a clean source tree. All three compiled adapters match the earlier frozen hashes. The initial child subtask used a no-write bake adaptation; the parent then exercised the literal build wrapper and verified its two generated tracked files remained byte-identical.

The first independent preparer review [failed on three reproduced issues](../experiments/optionb-source-repro-v1/preparer-review-v1.json):26 frozen training locators omit URL fields, derived writer paths could traverse symlinks/junctions, and `.cache` alone was incorrectly accepted as a run. The separate correction passed156 focused/original regression tests and [independent closure](../experiments/optionb-source-repro-v1/preparer-review-v2.json). Its actual fresh-checkout execution reconstructed **23,315 TRAIN /1,251 DEV /210 calibration rows**, verifying original WAV/decoded-PCM identities and ordered labels without copying old audio or features; [preparation evidence](../experiments/optionb-source-repro-v1/preparation-v1.json) records the acceptance.

The initial CPU fit/export stage passed74 bounded tests, including a real two-step optimizer fixture; [independent review](../experiments/optionb-source-repro-v1/fit-review-v1.json) then reproduced a late missing-pytest export dependency. The separately declared hashed-helper setup and early canonical-exporter import correction passed75 tests and [independent closure](../experiments/optionb-source-repro-v1/fit-review-v2.json). The original exporter and scientific lock were not changed.

Reviewed fitting revision **`d00a3c40a26dc4fa1076d38923ae832dfa4058ae`** was pushed and pulled into the new checkout, where **299 selected OptionB Python tests passed**. Actual source-only extraction matched **all24,566 TRAIN/DEV tensor hashes** and the exact original **1,889-human normalizer** before optimization; see [feature/normalization preflight evidence](../experiments/optionb-source-repro-v1/feature-preflight-v1.json). The full48-epoch CPU fit was launched from random initialization, then stopped after39 epochs under the later GPU directive above; its incomplete evidence is retained. Its selected checkpoint, export,138-point policy reconstruction, complete inference/wording metrics and separate PC timing repeat are **not yet accepted**. Test/holdout PCM remains excluded before a successful exact selection lock. No software fixture, partial epoch or matching feature gate is presented as full training reproduction or fresh acoustic qualification.

## What stays unqualified

[The current release report](RELEASE-optionb-tiny-v1.md) retains exposed-cohort metrics and known errors. Reproducing them does not establish fresh-blind accuracy, independent listener truth, all93 human wording cells, live capture or Pi readiness. There are no new independent-speaker recordings. Pi tests are excluded; actions remain disabled, and the snapshot is not a promoted/installed release.
