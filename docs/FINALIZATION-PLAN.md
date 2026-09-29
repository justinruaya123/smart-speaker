# Repository-only VCM finalization implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan in this task. Preserve all pre-existing untracked files.

**Goal:** Execute CODEX_HANDOFF.md with a reproducible streaming constrained-phoneme candidate, complete evaluation and a self-contained diagnostic or qualified ONNX package.

**Architecture:** A shared `vcm` package owns the causal frontend, six-cache CNN, whole-command CTC scoring, typed catalog and acceptance policy. Training is separate from a NumPy/ONNX Runtime inference distribution. Historical notebook implementation is extracted before replacing its dynamic loader.

**Tech stack:** Existing root Python environment, PyTorch for fitting/export, NumPy/einops, ONNX Runtime CPU, pytest; no pretrained neural initialization.

**Spec:** [CODEX_HANDOFF.md](../CODEX_HANDOFF.md). User explicitly authorized execution on 2026-09-29, superseding the maintenance hold for this tracked-source migration and permitting the specified constrained-phoneme approach. No physical actions or external service integration.

## Global constraints

- Baseline cd98092436fb71e8f7ce196b4392d60f6a2a624e; same branch throughout.
- Pre-existing untracked hashes recorded in `.cache/handoff-execution/untracked-before.json`; do not modify or use these as hidden inputs.
- Pinned downloads must match source ledgers. Human/source/speaker splits and normalization provenance recorded before fitting.
- 16 kHz, 400-sample periodic Hann, 160 hop, FFT 512, 40 HTK mel bins 20–7600 Hz, floor 1e-8; causal 113,064-parameter CNN, 39 phones plus blank.
- Two seeds for one primary candidate; at most one justified ablation with two seeds. No tuning on sealed test.
- No externally pretrained weights, sentence transcriber, LLM, cloud recognition or device executor.
- Missing human/Pi/calibration/quality evidence prevents qualified publication; diagnostic publication must state that explicitly.

## Review focus

Nonzero caches and incomplete final blocks; normalization leakage; missing or changed catalog/policy identity; dropped microphone samples and stale finalization; distribution containing stale models or historical source dependencies.

## Task 1: Reproducible baseline and source extraction

- [x] Verify clean tracked baseline, preservation manifest, scoped ignores and pinned procurement.
- [x] Test extracted historical source equivalence; extract notebook modules into ordinary files without changing behavior. Keep historical tests runnable through a compatibility adapter, without a meta-path loader.
- [x] Archive original tracked source using an immutable manifest before retiring duplicates.

## Task 2: Streaming core and command contract

- [x] Add failing tests in `tests/test_streaming_vcm.py` for `StreamingFrontend.feed(pcm)`, `CausalVCM.forward(features, states)` and chunk/causality/nonzero-state parity.
- [x] Implement `vcm/frontend.py`, `model.py`, `decoder.py`, `catalog.py`, `acceptance.py`, `runtime.py`; freeze `configs/train.json` and fictional catalogs.
- [x] Verify exact types/numbers, duplicate names/capabilities, complete CTC sequence scoring, reset/fault state, hash rejection and endpoint-only acceptance.

## Task 3: Data, bounded fitting and evaluation

- [x] Add deterministic preparation/audit and source receipts using pinned original recordings; reject overlap and infeasible targets; freeze partitions and command mappings.
- [x] Implement ordinary `train.py`, `validate.py`, `benchmark.py` interfaces; record development selection and calibration lock before test; export one selected graph with actual ORT parity.
- [x] Run two seeds, preserve negative outcomes, emit all five command CSVs plus per-family counts, coverage and release verdict. No unavailable evidence inferred.

## Task 4: Portable runtime and publication

- [x] Implement `python -m vcm.tui` WAV/microphone inspection with tentative state and sticky capture faults; tests use software input only.
- [x] Replace publisher with strict file allowlist, identity validation, qualified/diagnostic distinction, one graph and no training/research dependencies.
- [x] Test clean extracted ZIP with network blocked during inference, nonzero states, documented CLI commands and stale-file poisoning.

## Task 5: Consolidation and final review

- [x] Record exact cleanup dry plan, hashes and archive restore path; remove only tracked duplicates after replacement tests pass.
- [x] Consolidate notebook, README, MODEL_CARD, DATASET and RESEARCH_SUMMARY with measured results and physical evidence gaps; preserve visualizer source.
- [x] Run source tests, portable package verification, preservation checks and an independent final code review; fix substantive findings and commit only this task's explicit paths.

## Progress and rulings

Planning: approved handoff is the binding design; no second design approval is needed. Execution remains in the requested branch rather than a new worktree. Runtime and training share frontend/catalog identities; validation and publisher consume the exported bundle's exact policy. Existing root environment is reused.

Completion: executed the four-fit budget, diagnostic replay and packaging. Independent review findings and the historical feature-provenance limitation are recorded in `reports/semifinal-v1/review.json`. Qualification remains failed; no physical Pi, live microphone or independent human campaign was run. The explicit handoff already selects same-branch commits and a normal push, so no additional integration menu is needed.
