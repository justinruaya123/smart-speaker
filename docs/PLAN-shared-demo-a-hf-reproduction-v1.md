# Baseline A treated-data publication and fresh reproduction

**Goal:** Publish only the treated data necessary for retained baseline A, with citations and immutable source locators, then execute its original full training recipe from fresh GitHub/Hugging Face main acquisitions.

**Target:** `shared-demo-slots-v1 / heads_s53 / epoch38`, 175653 parameters, 34 complete commands and 102 wordings; committed checkpoint SHA-256 `0164568747c67657836659dc1322947d6282d829717b93efe218269b2b456069`.

**Authority:** Current user request authorizes publication and actual retraining, on main. It does not authorize runtime/default changes, new data treatment, vocabulary changes, reselection from TEST or repository cleanup.

## Tasks

- [x] Freeze a portable, ordered projection of the original TRAIN/DEV/TEST references, feature identities, original configuration and comparison expectations. Preserve sampler insertion order; never reconstruct quotas from an unordered map.
- [x] Publish a versioned `shared-demo-a-v1/` dataset subset on existing Hugging Face dataset main. Include treated labels, exact upstream repository/revision/shard/row/audio/PCM locators, authored synthesis required by this model, source notices and citations. Do not mirror restricted third-party audio/features. Do not upload other experiments, environments, checkpoints or unrelated datasets. Preserve existing files/visibility.
- [x] Add the exact frozen fitter and minimal import closure under the existing isolated snapshot source root, plus one portable download/prepare/train/compare entrypoint and regression tests. Preserve existing main implementations and all frozen model files.
- [x] Test missing/substituted files, unsafe archive members, duplicate/missing recording identities, split leakage, changed feature hashes and changed weights. Verify positive cases and real optimizer execution. Publish the reviewed source to GitHub main with a normal fast-forward push.
- [x] Clone GitHub main again into an empty directory, create a separate pinned training environment, download only the manifest-declared Hugging Face main files and immutable original-source shards into an empty cache, recompute PCM/features/normalization, and execute all 48 epochs/4608 updates of the heads arm from random initialization. No old weights or fit/feature caches may initialize the fit.
- [x] Compare every selected checkpoint tensor, selected epoch, complete loss/DEV curve excluding timing, ordered draw digest/counts and saved complete-command results with original evidence. Lock the reproduced selection before TEST features/inference. Export and check Torch/ONNX decisions, compare original and reproduced models on the identical exposed regression and reserved synthetic TEST IDs. Preserve any mismatch rather than forcing equality.
- [x] Publish compact actual reproduction evidence and commands to main, read back exact remote revisions/payloads, and report completed stages plus failed/unmeasured gates.

## Boundaries and review focus

Original sources retain per-source research/consent restrictions; citations do not grant redistribution. Human reference quality and opaque historical synthetic-voice aliases remain unresolved. Locally authored Microsoft synthesis is labelled synthetic, research-only, without an invented open recording license or verified human demographic claims. No numerals-only pools, calibration fitting, completed training caches, app/UI, microphones, physical Pi or provider actions are part of this reproduction.

A fresh download means a new task-owned cache and actual network acquisition, not copying an old corpus. The source and dataset are acquired through main, then bound to their resolved immutable commits. CPU/GPU and library-version differences must remain visible; exact same-backend equality is tested, not assumed.

## Progress

- Resolved newer GitHub main `4028871f4fe9c07a7d9d7b946de3bf3512e0fdcd`; it explicitly retains A rather than the older 214592-parameter atomic model.
- Existing Hugging Face dataset main before publication: `44a1a23081a8b042c5d43a73cf233c03c8151049`. Existing model repository is private and is not a publication target for this dataset task.
- Publication checkout is a fresh main clone under `.cache/hf-main-reproduction-v1/publish`; original dirty checkouts/staged files remain untouched.
- Published HF main `13d388c31985be7f42af56b46570281728ecc054`, then source main `86f2630`; preserved the first acquisition failure before optimization and issued narrow cache-link/output-root corrections.
- Successful fresh checkout consumed GitHub main `64c3ea19bb0a0f0bd52866ad04f6c8b41b5a92e4` and HF main13d388c in a new39-package CUDA environment/new empty cache. All48epochs/4608updates/147456draws completed; epoch38, all41selected tensors, final tensors, full curves, sampling and all642decisions match; exported ONNX bytes are identical.
- Independent accounting verified12dataset files,9original shards,28090feature arrays and the1889-human normalizer. Corrected scoped tests:198passed/4skipped. Fourteen full-suite failure nodes match untouched main; existing browser/wake gates and external-review provider failure remain visible.
- [Final report](EXPERIMENT-shared-demo-a-reproduction-v1.md) and `submission/shared-demo-a/reproduction-v1/` preserve actual acquisition/environment/training/evaluation/audit evidence. Native checkpoint dictionary ordering and normalizer CRLF differences are explicitly distinguished from identical values/tensors/graph; no gates were retuned.
- Ruling: keep failed source acquisition and inherited CI defects as evidence rather than delete caches, weaken hash boundaries, refit/reselect using TEST or expand this task into application/website repairs. Runtime installation, actions, microphone, fresh-human and physical-Pi qualification remain out of scope.
