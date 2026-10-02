# PCV release v1 implementation plan

**Goal:** Deliver a trained finite VCM and one-command local training/test/validation/publishing workflow, a Pi4 transfer ZIP with wake/VAD, frontend JSON, validation verbatims and measured-report writer, plus the requested two-slide deck.

**Architecture:** Preserve the schema-v3 command contract and immutable historical studies. Audit a fresh pinned HF revision before a new source-only training experiment. Reuse the endpoint-context ONNX candidate as a declared baseline, compare development-selected improvements, and package graph, normalization, catalog, schema and policy together. The runtime emits commands or rejection without ASR, cloud inference or device actions. Physical Pi evidence is supplied by the user's later run.

**Tech stack:** Existing root Python environment, PyTorch only for training, ONNX Runtime CPU for deployment, existing NumPy frontend and finite decoder, pretrained local wake/VAD; editable PPTX via the presentation skill's artifact-tool runtime.

**Spec:** User's 2026-10-02 instruction to finish training and deliver the pipeline/deck/ZIP, schema in `docs/EXPERIMENT-pcv-updated-voice-v1.md`, governing `REQUIREMENTS.md`.

## Constraints and execution decision

- The user explicitly authorizes implementation, publisher changes and slides now, lifting the previous deliverable deferral. No additional design approval is needed for this agreed flow.
- Preserve dirty work, old source snapshots, checkpoints and results. No branch switch, commit, push, repository cleanup or installed artifact replacement.
- Only supplied HF downstream audio; offline pretrained teacher remains allowed. External recording fallback requires no suitable supplied example for that training input.
- Do not tune on test; new revision membership must be compared by decoded PCM and speaker against all prior fitting/evaluation membership before initializing a candidate.
- Keep model qualification, pipeline/package checks and physical Pi evidence separate. Never hide failed accuracy, claim raw candidates were accepted, or claim absent phrase audio passes.
- New durable Markdown stays flat under `docs/`; audio/checkpoints/raw output stay ignored under `.cache/`, generated deliverables under `dist/`.
- Integration owner is the root task. Parallel delivery agent owns only dedicated runtime/package/benchmark source and minimal existing publisher dispatch. Slides agent owns only the dedicated slide builder and generated slide directories.

## Review focus

1. New dataset revision moves a recording between splits: prevent inherited-checkpoint evaluation leakage and retain a movement audit.
2. `--skip-train` with stale or mismatched model/data: verify selected release identities, never silently retrain or use another graph.
3. Endpoint versus causal graph interfaces: validate state shapes/output rate and reject incompatible swaps.
4. OOV, unknown light names, silence and invalid PCM: emit explicit rejection; diagnostic candidate fields cannot masquerade as accepted commands.
5. Desktop run of the Pi benchmark: mark hardware evidence correctly and write missing/untested values, never fabricated Pi statistics.

## Tasks

- [x] Refresh/audit source (`scripts/train/pcv_release_data.py`, dedicated source refresh experiment). Tests cover unchanged-file reuse and prior split contamination. Freeze source revision, file hashes, sample metadata and differences.
- [x] New training experiment (`scripts/train/pcv_release_training.py`, dedicated protocol/cache). Audited and frozen; 72 teacher/136 student epochs completed. Hard-label and distilled arms compared on identical development data, with selection before test. Old-model/new-test comparisons are not treated as clean because source split movement was detected. Model quality gates fail and remain unresolved.
- [x] Runtime and Pi delivery (`scripts/runtime/pcv_final_runtime.py`, `scripts/pcv_final_delivery.py`, minimal existing publisher hook). Validated artifact identity, pure CPU dependencies, wake/VAD loading, complete-utterance processing, typed JSON, bounded source samples, 415 definitions, handoff and Markdown/JSON benchmark. No physical Pi or live microphone evidence claimed.
- [x] One command (`release.py` thin entry and `scripts/pcv_release.py`). Normal and `--skip-train` workflows completed. The latter checks bound source/checkpoint identities and preserves locked evaluation. Quality failure yields an explicitly disabled diagnostic package; execution/integrity errors stop the workflow.
- [x] Two-slide deck (`scripts/report/pcv_release_slides.mjs`). Actual frozen model/data/training/results populated; Pi metrics pending. Native PowerPoint and two-page PDF rendered and inspected. Repeated same-basename builds use distinct scratch directories and preserve prior finals.
- [x] Final integration: normal and `--skip-train` flows, ZIP integrity, isolated extracted-package parity, schema/rejection/integrity tests and module exclusion passed. Installed artifact hashes unchanged. Independent review findings repaired.
- [ ] Model acceptance: human test 502/636 raw exact; calibrated quality/coverage gates fail. The 93-phrase baseline and all-415 acoustic coverage are not passed. New suitable evidence and a separately frozen improvement experiment are required before enabling commands.
- [ ] User-operated physical Pi/live-speaker validation, plus public submission/license/DOI confirmation.

## Current evidence log

- HF API reports revision `25111444af3adff7588d86ab27c304bb080895cf`. Both training files, test and holdout differ from prior `e3e203b360d5c3205d59f99ad65cd380f9c0f517`; six numeral payload hashes match. All 81,686 recordings are audited. Eighteen old-training PCM identities moved into new test/holdout, so every source-fitted model starts fresh.
- Baseline for comparison: context128 INT8, 935,485 parameters, 1,558,203 bytes; locked prior raw test 473/609 human and 3,286/3,368 supplied synthetic. Prior policy remains unqualified and disabled. These are historical comparison figures, not claims about the new revision.
- Existing `.cache/`/`dist/` ignore rules cover new payloads and expose source/protocol/compact evidence. No ignore-only commit is required.
- Preparation is frozen at `390c1ca93fbf2e8ffa0f685e839f32c0f84f2f3bf147d920ad827f081386646c`: 1,031 finite entries, all 415 definitions, fresh normalization on 1,868 human training recordings. Published test and holdout membership is preserved.
- The initial normal workflow exposed a one-shot preparation entry point being called on already prepared data. Its failure receipt is retained. The workflow now verifies the existing immutable preparation; fresh normal execution has advanced into human-first fitting.
- Final delivery verification passes 59 tests plus three mocked capture subtests. Independent review repairs separate replay/live evidence and enforce exact rejection reasons. The latest ZIP contains the actual graph, wake/VAD, 415 definitions, bound evidence and both slide formats. All 15 bundled predictions and typed outputs match frozen evaluation exactly in an isolated extracted runtime. Details, receipts and remaining failed model gates are in [the release report](EXPERIMENT-pcv-release-train-v1.md).
