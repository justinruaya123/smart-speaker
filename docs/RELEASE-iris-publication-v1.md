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

The frozen frontend was not edited. Chunk-vs-batch frontend characterization found one float32 ULP difference (1.1920929e-7); the shape/parity check uses an explicit 2e-7 absolute tolerance, while same-feature graph/logit checks are exact. The existing owner frontend imported SciPy eagerly even for native16kHz; the selected environment has no SciPy. Its unchanged resampling algorithm now imports SciPy only when resampling is requested. No resampling or owner-feature math changed and the original scientific environment was not upgraded.

## Completed core behavior and commit map

| Commit | Effect / revert scope |
|---|---|
| `dea8ec6` | Exact seven-pair `.gitignore` exceptions only |
| `f06029f` | Independent five-file pins, original atomic decision and actual CPU graph adapter |
| `db07323` | Existing worker selects epoch44; preserves model output before owner veto; separate owner/wake companions |
| `8a7cd12` | Published-artifact benchmark, recipe identity and explicit legacy selection |
| `0906c2e` | Default read-only reproduction verification, explicit source fit and evaluation-only preparation/export |
| `3632008` | Source inference launcher; compiled combined gateway only; default actions off; replay/CLI silent |
| `bc8b63b` | Seven real pre-baked MP3 + decoded PCM WAV pairs and hash-bound generation manifest |
| `bb7b405` | Closed host feedback, final-utterance/phase validation, deduplication, capture suspend/resume and cleanup |
| `60cc532` | Independent wake override retains original VAD; legacy packages do not require new feedback assets; capture failures remain visible |
| `3d0802b` | Explicit wrong-positive accepts, rejected positives and exact-command denominators |
| `de8fb51` | Public inference rejects a legacy bundle before setup; bootstrap/fit fail-on-call test guards |

Revert dependent changes in reverse order with `git revert`, not reset/force. These commits do not fit or replace the VCM, change frozen thresholds, promote a wake model, enable actions globally, or modify scientific locks.

### Public commands and cross-agent contract

```sh
python -B inference.py --check --no-setup
python -B inference.py --wav PATH.wav --no-setup
python -B inference.py --wav WAKE_AND_COMMAND.wav --wake-wav --no-setup
python -B inference.py --list-devices --no-setup
python -B inference.py --mode ui --speaker OUTPUT_INDEX --no-browser
python -B benchmark.py --model-choice epoch44 --manifest ORIGINAL.jsonl --data-root ORIGINAL_ROOT --output NEW_RESULT_DIRECTORY --repeats 3
python -B reproduce.py --load
python -B reproduce.py prepare-evaluation --source-checkout PINNED_SOURCE --source-python PINNED_PYTHON --output .cache/iris-publication-evaluation/NEW_RUN
# Original training is opt-in, remains an exact-comparison study, and propagates its exit status:
python -B reproduce.py fit --execute-fit --source-checkout PINNED_SOURCE --source-python PINNED_PYTHON --output .cache/EXPLICIT_FIT/NEW_RUN --device cpu
# Historical continuation workflow and benchmark remain explicitly available:
python -B reproduce.py --model-choice legacy --help
python -B benchmark.py --model-choice legacy --help
```

`--no-setup` uses the current interpreter and already compiled/pinned UI tools. Normal UI setup reuses the unchanged runtime lock and pinned UI bootstrap; it never calls the old package-install/`train.py --skip-train` path. `IRIS_PUBLICATION=true` is set only for this launcher's epoch44 UI. Actions are off unless explicitly configured privately; `--no-audio`, CLI, check and WAV override inherited enable flags. UI uses owner-protected inference. The source launcher validates the recipe before setup and checks the live model SHA/258912/epoch44/instance token before announcing UI readiness.

Worker feedback contract: `op:'feedback'` with `cue_id:'intent'|'success'|'failure'` and nonempty known final `utterance_id`; optional `session_id` must match. Other fields, paths/text/URLs, unknown/WAV utterances, invalid phase transitions and conflicting completions reject. Replies follow the blocking output stream's drain/close, not a browser request or a scheduled-but-unplayed sound. Session+utterance+phase and bounded request-ID deduplication prevent replay. A new active command capture is not interrupted by an old completion cue. Feedback errors do not change a VCM result or retry any provider. Wake acknowledgments rotate Yes/Hmm/Yeah/Here, suspend input through playback plus an 80ms tail, retain the wake/owner reference and sample sequence, and fail closed if capture cannot resume. Ready advertises `capabilities.feedback:true` for epoch44, actual feedback/actions flags, atomic .50/.10 policy and companion hashes. Physical audibility is not asserted by these tests.

`IRIS_WAKE_ROOT` / `--wake-root` is an explicit optional **wake-only** override. Default remains the existing native companion; VAD and experimental owner remain separate original assets. No failed candidate is installed by these commits.

## Final executed core verification

- Final focused command: explicit original Python, `-B -m pytest -p no:cacheprovider -q`, all eight new test modules plus existing `test_iris_worker.py`, scratch basetemp: **102 passed in 11.89s**.
- The same focused selection in the parent-provisioned unchanged-main locked product runtime (NumPy2.5.3 / ORT1.30.0), with test tooling added read-only from the original environment: **102 passed in 13.04s**, XML `.cache/iris-core-product.xml`. No environment install/update was performed for this check.
- Actual `inference.py --check --no-setup` in both environments loaded the real selected graph plus owner/wake/VAD, with no capture/actions. Actual locked-runtime `--list-devices --no-setup` enumerated35 inputs/38 outputs without opening streams.
- Seven actual Edge-TTS7.3.1/`en-US-AriaNeural` acknowledgments/tones were built once (four speech cues, three designed tones); every MP3 was decoded with real ffmpeg and compared exactly to its committed PCM WAV. Tests verify inventory, distinct PCM, hashes, finite/unclipped samples, duration and exact text. No runtime TTS/MP3 decoder/network dependency was added.
- Evaluation export read back the parent's real pinned preparation:4,418 TEST +196 HOLDOUT recordings, original ordered references, WAV hashes and original header-bound decoded-PCM hashes. New scratch portable manifests contain196 holdout rows (2 ambiguous) and1,146 human test/holdout rows (187 ambiguous), without fitting or copying recordings. The unchanged preparer command is tested to request only `--splits test holdout` and propagate failures.
- Broader product-runtime regression, excluding the227 already characterized original-verifier cases: **275 passed /38 failed /313 collected**. Programmatic XML-vs-parent-baseline comparison found all37 known failure IDs plus exactly one prerequisite failure: `scripts/test/test_iris_configuration.py::test_profile_inventory_no_ui_or_benchmark_leaks` requires compiled SpeakerUI, absent from this isolated Python-only worktree. No UI files were touched to hide this prerequisite. Full post-integration acceptance must build the UI and rerun it; this is not an all-green full-suite claim.
- Benchmark regression after adding explicit counters:13 passed, one known Windows hardware-label test deselected. That excluded ID was independently reproduced against unchanged main before any change to its hardware helper.
- `git diff --exit-code c0628940400642f204ff98e26908549b1c07e41c --` all original fifteen payloads, VCM frontend, scientific manifests/locks, historical card/dataset docs and UI lock returned0. No original evidence was rebound.

### Real published-artifact replay (not idle-host or Pi timing)

Executed `benchmark.py --model-choice epoch44` using the parent's hash-bound1,146-row original-human manifest and source root, repeats3, output `.cache/iris-publication-v1/core-human-sourceenv-01/`. Receipt UTC: `2026-10-23T22:40:08.559211+00:00` (October24 local UTC+08). The host was not declared quiescent; do not compare this with a different concurrent-load run as a speedup.

- 1,146 recordings /3,438 calls;187 ambiguous excluded;959 scored (689 positives,270 negatives);zero unstable outputs across repeats.
- Raw correct complete commands545/689; accepted correct465/689; negative accepts25/270. Repeats do not inflate acoustic denominators. Qualification remains false.
- PCM/frontend/ONNX/typed-decision p50 **4.46715ms**, p95 **8.13932ms**; whole-process lifetime peak RSS **76.921875MiB**. Timing excludes decoding, file I/O/hashing, utterance duration, wake/VAD waits, owner, UI/gateway, feedback and providers. This is not end-to-end user latency.
- Original source environment remained NumPy2.3.5 / ORT1.23.2 / SoundFile0.13.1. Pi and live microphone were not measured.
- Manifest SHA `3decdc2e414cc5280c47f7ff6892773f33ffeb5c0f6448c90c1462a6def0dba2`; metrics SHA `6dd68dc5e156d50ce27255ae9eaba1c0451976423ea72db01a0f778c583324c3`; trials SHA `4da2e33aa7cd90d9c39cb51112035bf04f8d80aa535ab10e9efffb680a2595c8`.
- As-run runtime SHA `cae5f5e138b5aa0d8b125a6aacf634c158e131d4ba02577480ec72924dcfb5dc`; as-run shared benchmark SHA `f3e456329d3af658347a096c16678d9a6e3a9078dc9aa9017841ed2b84c8ebb5`. The later explicit-counter extension is a separate commit; the earlier receipt is not rewritten to pretend it ran later source.

### Integration source receipt (after `de8fb51`)

This new binding is for publication integration, not a replacement for the original snapshot's source bindings.

| Path | SHA-256 |
|---|---|
| `inference.py` | `baae86b166e5d5d7459b7b6d5bb54a5a25f0e008d92671706ae359bcb341cf31` |
| `benchmark.py` | `fcde1517981346a276852b10d4a3263578fd1e6322a81a8f4c8a729b942974a9` |
| `reproduce.py` | `53c2d3f8dfe57472a7f921cd065f4de7b2775a1c4320d95854dfea4ec2378bf7` |
| `scripts/runtime/iris_epoch44.py` | `cae5f5e138b5aa0d8b125a6aacf634c158e131d4ba02577480ec72924dcfb5dc` |
| `scripts/runtime/speaker_worker.py` | `1d89b88f7056d19d2b065aebfc40fffe543c12ca6954086c1ed0776d51743962` |
| `scripts/runtime/mvp_wake.py` | `ffb875f16a24f29fc3360bb247cdad1ec42d89af4ec6a1140fa2fe61532f3382` |
| `scripts/runtime/audio.py` | `100cf2a534274a5976782289b482e0c540e6c24cf0a429c377d4d64e6a2e07dc` |
| `scripts/runtime/iris_feedback.py` | `dd738bac25f16c530068b95ce6b1ca0247e9c8c3f69bc39d9085c8530a7162be` |
| `scripts/runtime/iris_pi_benchmark.py` | `483b1793108895f65edc5746ccdd8c7362b02db3a8ae23eb7c4272a9ce22fd7c` |
| `scripts/iris_inference.py` | `389f4f04027d56b325c9361528c9d3229d20c3740d771d72a638a19f616bb7c4` |
| `scripts/iris_epoch44_benchmark.py` | `0d98cb682bc1250da5406f644f1dd4fa91073c93a6d352550798772d4611f90f` |
| `scripts/iris_reproduction.py` | `e2ab6aa23394f8e0953edede146fa28334588337bc0a02c55d0642bc331cdb64` |
| `scripts/build_iris_feedback.py` | `e9c4592ef8169ed0c71a5ccda98da4c4ad681ea05277ebbfd86776c4482a3cfb` |
| `submission/feedback/manifest.json` | `4649d644b7a77d5ed75c22de40a20b63059b953e081446815a8a22afe7d22b4a` |
| protected `vcm/frontend.py` | `cfeabf0f6b2b4f280dbd0a44b65aaab7b68907ff57d4ff95f65f5dfa23abe97a` |

## Issues and remaining acceptance

One early RED selection matched an extra historical-default reproduction test before its bootstrap guard was installed. That old entry point created an **ignored worktree-local** `.venv` and failed on its historical CUDA-wheel platform requirement before fitting. It did not synchronize/modify the original source environment or any lock; the original Python/NumPy/ORT identities were rechecked afterward. Tests now guard both bootstrap and fitting before exercising defaults. The unused local environment is not committed or represented as the approved scientific environment.

Parent-owned untracked `experiments/iris-publication-v1/` baseline/reference receipts were left unstaged. This core log does not claim independent review. Parent integration must build/retest the combined UI, validate action/cue ordering, compare against original gold, and perform any physical playback/capture acceptance.

**Wake remains a release blocker.** Parent's independent final finding: existing native companion1/32 positives,0/68 false wakes; separately studied same-architecture candidate21/32 positives,2/68 false wakes. Both fail the predeclared90%/zero-false synthetic gate. These commits leave the existing experimental native default unchanged and do not promote or wire the failed candidate. No working Hey Iris, live-human wake ownership, physical device result or all-release-checks-green claim is made. Any independently validated external alternative is separate parent-owned work.

Original CPU exact refitting remains NOT VERIFIED; Pi/live-human/provider qualification, independent review and publication remain separate. No push, merge, credentials change or Pages deployment occurred here.
