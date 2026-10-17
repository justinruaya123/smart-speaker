# Compact training-pipeline repair implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for this already approved, bounded comparison. Preserve existing studies and execute inline.

**Goal:** Test whether recognition-based checkpoint selection improves useful human command acceptance at exactly 935,567 parameters.

**Architecture:** Reuse `PooledContextVCM(width=128, layers=6, heads=4, feedforward=256, dropout=.1)` and the existing finite decoder/runtime. Put orchestration and selection changes in one research-only adapter; keep all frozen parent sources intact.

**Tech stack:** Root `.venv`, PyTorch/CUDA for the optional continuation, CPU ONNX Runtime for deployable comparisons, existing policy fitter and Pi publisher.

**Spec:** The user-approved four-step proposal in this conversation and [REQUIREMENTS.md](../REQUIREMENTS.md), especially the 1.2M ceiling, pure VCM inference and human evidence requirements.

## Global constraints

- Exactly 935,567 VCM parameters, hard maximum 1,200,000; architecture, frontend, full 1,079-command catalog, normalization, losses and augmentation stay fixed.
- First compare the recognition-selected saved mixed epoch 28 with the existing epoch 32 control. Weight selection uses human development recognition only; calibration fits acceptance, never weights.
- Recognition key: human raw macro speaker recall, human raw exact count, fewer human raw negative candidates, synthetic raw recall, earlier epoch. Acceptance precision and thresholds do not enter this key.
- The existing human epoch 24 `last.pt` is the explicitly authorized alternative foundation; do not claim it is the best checkpoint among all human epochs.
- If the saved candidate fails promotion, run at most one 32-epoch mixed-control continuation from human epoch 24, with seed 53 and the parent's unchanged batch/sampling/optimizer/scheduler recipe. Do not repeat the human stage.
- Keep the existing per-intent threshold grid and `.90` slot/target checks. Missing human classes retain `.95`; strict qualification remains separate.
- On **each** human fitting view, candidate correct accepts must be at least the current relaxed control, wrong accepts and UNKNOWN accepts must be no greater, and combined correct accepts must strictly increase. Control development is 96 correct / 6 wrong / 1 UNKNOWN; calibration is 40 / 0 / 0. Wrong counts include UNKNOWN errors.
- Freeze the protocol and code before candidate calibration. Freeze selected graph and policies before one regression replay; the already examined published test/holdout are regression data, not fresh validation.
- A final shipping recommendation additionally requires human test correct accepts >229, wrong accepts <=21 and UNKNOWN accepts <=2; human holdout correct accepts >=24, wrong accepts <=1 and UNKNOWN accepts <=1. Failure stops promotion without another fit or threshold revision.
- FP32 only, graph <4 MiB with numerical parity <2e-5; no new quantizer search. Keep existing INT8 failure intact.
- No installation, action integrations, branch switch, commits, pushes, dataset changes or physical-device operations. New raw output goes in ignored `.cache/speaker-pipeline-repair-v1/`, compact evidence in `experiments/speaker-pipeline-repair-v1/`, report directly in `docs/`.

## Review focus

1. A high-precision checkpoint with very few accepts must not outrank better raw human recognition.
2. A recall gain must not conceal a wrong-command or UNKNOWN increase on either fitting view.
3. Matching row IDs with changed references, synthetic rows, capture faults and excluded references must not silently change comparison populations.
4. Partial runs and frozen files must remain recoverable without overwriting evidence or starting a second continuation.
5. Export, policy binding and extracted CPU execution must agree; a desktop ZIP must not be called Pi qualification.

## Task 1: Selection and promotion adapter

**Files:** create `scripts/train/speaker_pipeline_repair.py`, `scripts/test/test_speaker_pipeline_repair.py`; register the adapter as `runtime: false` in `scripts/modules.json`.

**Interfaces:** consumes parent prepared rows, saved checkpoints and FP32 control receipts; produces `recognition_key(metrics, epoch)`, `promotion_gate(candidate_views, baseline_views)`, `verify_comparable(candidate_rows, baseline_rows)` and versioned protocol/selection receipts.

- [ ] Write tests for the observed confidence-selection failure, speaker balance, unchanged recognition under confidence changes, per-view error limits, strict gain, changed/missing/duplicate references, synthetic contamination and regression limits.
- [ ] Run `.venv/bin/python -m pytest scripts/test/test_speaker_pipeline_repair.py -q`; expected RED because the adapter is absent.
- [ ] Implement only the bounded adapter, reusing isolated parent trainer/release contexts, the existing policy fitter and export/publisher helpers.
- [ ] Run the new tests plus the existing compact/final focused tests; expected GREEN. Run root pytest and record pre-existing collection failures separately.
- [ ] Obtain one fresh-context independent review of the new adapter/tests, correct Important/Critical issues in one tested pass, then freeze source/input/control/checkpoint hashes. No commits are requested; preserve the ledger and review evidence.

## Task 2: Saved checkpoint comparison and optional continuation

**Interfaces:** consumes Task 1's frozen protocol and adapter; produces saved-comparison receipt, optional one-fit selection, candidate FP32 graph, calibration policy and development promotion decision.

- [ ] Run `.venv/bin/python -m scripts.train.speaker_pipeline_repair reuse`; expected a complete epoch-28 FP32 comparison and explicit promotion result without reading test/holdout audio.
- [ ] If and only if it fails, run `... speaker_pipeline_repair train` on an approved visible NVIDIA GPU; expected one mixed continuation, with raw-recognition checkpoint selection and preserved parent data/training recipe.
- [ ] Evaluate the single continuation-selected checkpoint using `... speaker_pipeline_repair candidate`; expected explicit promotion pass/fail under the same rule. If it fails, stop without test replay or package replacement.

## Task 3: Locked regression and handoff

**Interfaces:** consumes the development-promoted candidate and frozen policies; produces a regression receipt, per-intent/role results, report, optional Pi test ZIP and verification receipt.

- [ ] For a promoted candidate only, run `... speaker_pipeline_repair regression`; expected one locked test/holdout comparison, including failed gates, at unchanged reference populations.
- [ ] If final regression gates pass, package using the existing publisher and verify graph/policy hashes and extracted demo/qualified parity. Otherwise keep the current archive as the control and report the negative result.
- [ ] Document citations, unchanged augmentations, exact parameter/byte counts, per-intent results and the limits of reusing examined recordings in `docs/EXPERIMENT-speaker-pipeline-repair-v1.md`.
- [ ] Update `EXPERIMENTS.md` and the requirements evidence gap map. Verify protected file hashes and focused tests, then publish the measured decision without promising physical Pi readiness.
