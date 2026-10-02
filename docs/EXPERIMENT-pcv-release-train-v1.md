# PCV release training v1

Status: **Pi evaluation delivery complete; model not qualified and baseline not passed**. Fresh fitting, locked evaluation, one-command training/reuse workflows, slides and extracted-package verification are complete. The installed artifact bundle remains unchanged.

The user authorized local training, the two-slide report, the publisher, and one command covering training, software tests, model validation and Pi packaging. No A100 or physical Pi/microphone action is part of this run. ME-02/03/04/06/07/08 and the ten user-facing use cases remain in scope. Publication, licensing and hardware results remain explicit evidence items.

## Run the workflow

Use the existing root environment and pinned source cache in this research checkout:

```powershell
.venv/Scripts/python.exe release.py --run-id release-v1
.venv/Scripts/python.exe release.py --run-id release-v1 --skip-train
```

The first form prepares and trains missing stages, then verifies completed stages against their original hashes on a repeat invocation. A partial fitting stage is preserved and requires a new run ID; it is never silently overwritten. `--skip-train` requires the selected checkpoint, protocol and source bindings, and still runs software tests, locked validation, deck generation, ZIP checks and extracted-package replay. Every invocation writes its own stage receipt and logs. This is a workflow for the configured research environment, not a claim that a clean public clone already contains the private local caches or bundled slide tooling.

`publish.py --pcv-runtime` remains the dedicated publishing entry. The workflow supplies the selected bundle, fresh test manifest and its bound acquisition/source-card receipts together. A manual invocation must supply those same identities. Unqualified models additionally require `--diagnostic`; that is a labelled evaluation package, not an acceptance override.

## Data and evaluation boundary

The new snapshot is [airimonda/ai231-me2-voice-commands at 25111444af3adff7588d86ab27c304bb080895cf](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/tree/25111444af3adff7588d86ab27c304bb080895cf). It contains 81,686 readable mono 16 kHz recordings, approximately 30.869 audio hours. Of these, 15,296 are command/test/holdout recordings and 66,390 are isolated numerals. Source-qualified identifiers count 2,888 human speaker IDs and 100 synthetic voice IDs; the latter are not human speakers. Source licenses differ; no single permissive license is inferred.

The refresh found 18 old-training recordings in the new published test/holdout splits and ten old-training speakers in new test. Consequently, the old fitted teacher heads, student checkpoints and normalization cannot be used as clean candidates on the new test. Fresh heads and student weights start randomly. Only immutable features from the frozen, externally pretrained WavLM encoder may be reused by exact PCM identity. External teacher pretraining is the user's explicit exception; overlap in its original pretraining is not audited.

The publisher's utterance boundaries are preserved. Audio quality flags, conflicting metadata and missing pronunciations are audited. No cropping, splicing, synthesis, independent listening or automated transcription correction is claimed. The annotation review proposed 117 positive text corrections; 110 were approved, seven potentially conflated commands retained their prior references, and 19 uncertain texts were removed from reliable-negative supervision. The fixed rules apply consistently across splits, with publisher out-of-scope/unintelligibility flags taking precedence. Individual test transcripts and predictions did not drive the corrections; limited accidental exposure to aggregate holdout phrase bucket names is disclosed in the preparation receipt.

| Split | Human supported | Human unsupported | Human ambiguous | Supplied synthetic | Isolated numerals |
|---|---:|---:|---:|---:|---:|
| Training | 1,417 | 451 | excluded | 5,982 | 48,831 |
| Development | 193 | 61 | excluded | 811 | 7,732 |
| Calibration | 122 | 79 | excluded | 1,013 | 9,380 |
| Published test | 636 | 229 | 185 | 3,368 | 0 |
| Published holdout | 83 | 11 | 2 | 100 | 0 |

All original test and holdout rows are retained; unresolved references are separated from exact-command denominators. Fresh normalization uses only 1,868 eligible human training recordings, excluding isolated numerals. Preparation reuses 76,657 verified raw feature identities and extracts 4,029 new identities. The 1,031-entry finite catalog uses eligible training text plus all 415 approved definitions; no development/test text is added to expand the catalog.

Evidence: [source acquisition](../experiments/pcv-source-refresh-v1/acquisition-complete.json), [revision movement audit](../experiments/pcv-source-refresh-v1/revision-difference.json), [preparation completion](../experiments/pcv-release-train-v1/preparation-completion.json), [data freeze](../experiments/pcv-release-train-v1/data-freeze.json), [coverage](../experiments/pcv-release-train-v1/coverage.json).

## Model and selection

Microphone PCM passes through local pretrained Hey Jarvis and Silero VAD, then a 40-bin log-mel frontend. The command encoder runs once on the complete utterance after endpoint detection. Six bidirectional context blocks use 192 channels. Learned pooling summarizes the intent and explicit-light-target heads. The finite CTC scorer ranks only registered command phrases; it never produces an unconstrained transcript. Typed schema validation and a separately calibrated rejection policy produce the frontend event. There is no cloud model, LLM, ASR fallback or device execution.

The student has 2,066,831 parameters. Random-weight export produced a 3,149,591-byte dynamic INT8 graph with FP32 parity error 0.0000011921. These checks establish shape/export behavior, not accuracy. This new experiment declares a four-MiB graph budget before fitting. Inherited export helper fields still show their historical two-MiB checks; the new resource gate is recorded separately. Prior experiment budgets and results remain unchanged.

The schedule is 40 human-only teacher-head epochs, then 32 supplied-synthetic teacher epochs. A random student receives 40 human-only epochs. Matched hard-label and distilled continuations each receive 48 epochs, with 96 updates per epoch. If the teacher misses its predeclared development-use gate, the distilled arm is skipped and the hard arm remains. Development selects checkpoints and the arm by exact raw command recall and human negative errors; an inter-arm tie retains the hard arm. This design compares distillation on the new architecture, but the cross-version improvement cannot be attributed solely to one change because source, annotation, capacity, pooling and training budget also changed.

Calibration remains separate from test. Global acceptance requires 99% precision, 90% recall, human false acceptance no greater than 1%, and all 18 human intent categories represented in calibration. The selected graph must also pass INT8 development fidelity, resource and complete-waveform endpoint checks. Published test gates the release; separately reserved holdout is descriptive replay evidence and never selects weights or thresholds. Missing phrase recordings are untested, not passes. All-reject is not qualification.

Evidence: [training protocol](../experiments/pcv-release-train-v1/runs/release-v1/protocol.json), [random export preflight](../experiments/pcv-release-train-v1/model-preflight.json), [teacher feature receipt](../experiments/pcv-release-train-v1/teacher-feature-freeze.json).

## Pi package contract

The generated ZIP contains the selected ONNX command graph, frozen normalization, catalog, custom-slot definitions, schema and policy; local wake/VAD ONNX graphs; a frontend JSONL runtime; all 415 spoken definitions and exact desired outputs; bounded original licensed smoke recordings; an editable two-slide deck and its two-page PDF companion; source/license notices; file hashes and training/reproduction receipts. It excludes PyTorch, teacher weights, training checkpoints and the full corpus.

The offline dependency closure requires **Linux ARM64, CPython 3.12 and glibc >=2.28**. The system needs Python's venv support and audio libraries documented in the ZIP. An official-wheel audit found the unchanged pinned NumPy 2.5.3 closure unavailable for Python 3.11; no unreviewed dependency version was substituted.

After extracting on the Pi:

```bash
python3.12 pi_test.py --setup --output results/replay-01
.venv/bin/python pi_test.py --list-mics
.venv/bin/python pi_test.py --guided --mic 0 --output results/live-01
```

Use the actual listed microphone index in place of `0`. Guided mode presents all 415 cases; `--case B1-001` limits a smoke session. The script writes `REPORT.md`, `metrics.json`, per-phrase evidence and trial events into a new results directory. These outputs supply the slide's pending Pi values. Guided labels describe requested speech; actual human wording requires human verification. Do not treat replay latency, silence-only wake checks or Windows runs as physical Pi/live-speaker evidence.

Live and replay results remain separate. The guided headline includes completed live captures only; an additional end-to-end attempt denominator retains failed wake/capture trials as misses. Per-phrase full typed-output counts include rejection reasons. Unannotated replay reasons and missing recordings are explicitly reported. The runtime exposes diagnostic rankings separately from the authoritative accepted/rejected JSON result.

## Completed fitting and compressed export

All 72 teacher-head epochs and 136 student epochs completed on the local GPU: 6,912 teacher and 13,056 student updates. Development selected the hard-label student's mixed epoch 20 before any calibration or test scoring. The matched distilled arm did not improve positive command recall. This is a negative result for this distillation recipe, not evidence that every form of distillation is ineffective.

| Development model | Human raw exact / positive | Human raw negative errors | Supplied-synthetic raw exact / positive |
|---|---:|---:|---:|
| Human-only student | 154/176 | 23/49 | Not evaluated in human-only selection |
| Selected hard-label continuation | 159/176 | 28/49 | 108/108 |
| Selected distilled continuation | 157/176 | 25/49 | 108/108 |
| Teacher with supplied synthetic | 169/176 | 11/49 | 108/108 |

The trained INT8 graph is **3,173,089 bytes**, with SHA-256 `588f5f44f2b8bca9f68be605bc3a0be2be36159477c6002f21708824503c8421`. FP32 numerical parity has maximum error 0.0000066758 against the unchanged 0.00002 tolerance. INT8 preserves the FP32 aggregate human/synthetic development exact counts and negative-error count. The declared four-MiB gate passes; inherited nested two-MiB helper flags remain false and are not reinterpreted as passing their older limits.

Evidence: [selection before calibration](../experiments/pcv-release-train-v1/runs/release-v1/selection-before-calibration.json), [compressed development comparison](../experiments/pcv-release-train-v1/runs/release-v1/export-development.json).

## Locked test result and remaining model gap

| Stratum | Raw exact commands / supported positives | Raw wrong commands on negatives | Threshold-filtered exact / positives | Actual accepted exact |
|---|---:|---:|---:|---:|
| Published human test | 502/636 (78.93%) | 160/229 | 252/636 (39.62%) | 0/636 |
| Published supplied-synthetic test | 3,337/3,368 (99.08%) | Untested: no negatives | 2,918/3,368 (86.64%) | 0/3,368 |
| Separate human holdout | 79/83 (95.18%) | 8/11 | 38/83 (45.78%) | 0/83 |
| Separate supplied-synthetic holdout | 99/100 (99.00%) | Untested: no negatives | 95/100 (95.00%) | 0/100 |

The 185 ambiguous human test references and two ambiguous human holdout references remain outside exact-command denominators. Holdout is descriptive and does not select the model. Counts labelled threshold-filtered describe a diagnostic application of the frozen thresholds; the actual published policy remains disabled, so none are accepted commands. At those thresholds, human test precision is 252/260 (96.92%), with one wrong command on 229 negatives. Neither precision nor recall reaches the declared release requirement.

Calibration returns 41/122 human commands correctly at 41/41 observed precision and 0/79 negative errors; supplied-synthetic calibration is 904/1,013. Human calibration has no positives for CALL, CREATE_REMINDER, LIST_REMINDERS, MESSAGE or TEMPERATURE. A larger epoch count cannot create those missing recordings. Further trials need a new frozen experiment and suitable calibration/evaluation evidence; repeatedly selecting against this observed test would not be a valid improvement.

The baseline is **not passed**. Human test contains 71/93 baseline phrases, with 188/233 raw recording predictions correct (80.69%) and 38 phrase cells correct across every recorded repeat. Supplied synthetic covers all 93 baseline phrases, with 3,337/3,368 raw recording predictions correct and 72 perfect phrase cells. Across all 415 definitions, human test covers 72, leaving 343 untested; supplied synthetic covers 93, leaving 322 untested. All 415 ideal-cue command-only checks pass, but these are deterministic software checks, not speech accuracy or correct rejection-reason evidence.

All twelve original-waveform endpoint probes match cached features, logits and commands exactly. A separate 36-repeat desktop timing sample with one CPU ONNX thread measures 72.06 ms p50 / 113.44 ms p95, including in-memory WAV decoding, frontend, model and finite scoring. It excludes disk, utterance duration, wake/VAD waiting and the 400 ms endpoint silence. These are not Raspberry Pi measurements.

Evidence: [locked results and per-intent/slot counts](../experiments/pcv-release-train-v1/runs/release-v1/results.json), [all 415 test phrase denominators](../experiments/pcv-release-train-v1/runs/release-v1/phrases-test.json), [selection before test](../experiments/pcv-release-train-v1/runs/release-v1/selection-before-test.json).

## Findings log

- Source refresh, conservative annotation preparation, fresh normalization and graph preflight completed. The one-command integration suite passes 49 checks before completion of training. The first invocation exposed a one-shot preparation/resume mismatch; the workflow now verifies an existing immutable preparation rather than trying to overwrite it. Its failed receipt is retained.
- Fresh teacher feature extraction reuses 6,015 verified encoder outputs and computes 2,168 new ones. No old finite-head targets are reused. Both training arms and locked evaluation completed; the hard-label arm wins, but acceptance remains disabled.
- Independent review found replay/live metric mixing and command-only comparison of negative expected outputs. The repaired benchmark separates modes and capture-attempt denominators; qualification and per-phrase Pi reporting compare complete typed outputs including rejection reasons. Fresh focused verification passes 57 tests plus three mocked capture subtests in the normal workflow. No physical microphone was used.
- The first full workflow completed training, tests, locked evaluation, slides, archive verification and extracted-package replay. A repeat initially failed because the slide builder reused an immutable validation receipt. The failed output remains preserved; every build now has a unique scratch directory. Two real same-basename builds into distinct output folders verified the repair without changing prior final files.
- Final `--skip-train` completed with **59 tests and three mocked capture subtests**, unchanged locked model/test identities, and both slide formats plus detailed results/selection receipts included in the archive. An isolated import check reproduces all 15 bundled recording predictions and full typed outputs with zero numeric difference, imports no PyTorch/training modules, initializes wake/VAD, and produces no wake events on 100 silence blocks. This is desktop/package evidence only.

## Delivered artifacts

The final transfer archive is [pcv-diagnostic.zip](../dist/pcv-release-v1/20261001T193735329546Z-76e34e/pcv-diagnostic.zip), **46,834,596 bytes**, SHA-256 `511d293c98004c7244bf36adea426d0de6b7b25a3de59e9fac410a8e9ffc0bfb`. It contains 75 members: five ONNX graphs including wake/VAD, eleven pinned ARM64 dependency wheels, fifteen original SLURP smoke WAVs, all 415 expected definitions, schema/catalog/policy, the runtime and test script, handoff, source/license notices, both slides formats and bound model evidence. No fitting checkpoint, trainer or teacher model is packaged.

The editable [two-slide PowerPoint](../dist/pcv-release-v1/20261001T193735329546Z-76e34e/me2-diagnostic.pptx) and [two-page PDF](../dist/pcv-release-v1/20261001T193735329546Z-76e34e/me2-diagnostic.pdf) use actual frozen results and leave physical Pi measurements pending. Both final PDF pages were rendered and visually inspected; publication/DOI/model-license fields remain unverified rather than invented. The PDF is a visual companion; editable tables, diagram and complete source notes remain in the PowerPoint.

Receipts: [successful normal workflow](../experiments/pcv-release-train-v1/workflow/20261001T190336875921Z-16ba4f/workflow.json), [successful skip-train workflow](../experiments/pcv-release-train-v1/workflow/20261001T193735329546Z-76e34e/workflow.json), [isolated packaged-runtime verification](../experiments/pcv-release-train-v1/workflow/20261001T193735329546Z-76e34e/portable-verification.json), [final integrity and preservation check](../experiments/pcv-release-train-v1/workflow/20261001T193735329546Z-76e34e/final-verification.json).

The unresolved work is model acceptance and absent acoustic coverage, followed by physical Pi/live-speaker measurement and external submission publication/licensing. Further epochs of the completed recipe did not beat its selected development checkpoint. The five absent human calibration categories require suitable new evidence; they cannot be filled by treating synthetic voices as humans or by changing the observed test into training data. This delivery is useful for the user's planned Pi diagnostics, but is not represented as a finished command-accepting model. No commit, push, installation or physical device action was performed.
