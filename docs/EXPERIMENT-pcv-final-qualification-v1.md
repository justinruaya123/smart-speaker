# PCV final qualification v1 — 2026-10-02

## Decision

The source-only experiment is complete; the model is **not release-qualified**. Retain the 935,485-parameter endpoint-context student as the best development-selected research candidate. Its 1,558,203-byte INT8 ONNX graph improves held-out human exact commands to 473/609 (77.67%) and supplied synthetic exact commands to 3,286/3,368 (97.57%). The frozen acceptance policy fails quality and human-coverage gates and remains disabled. Actual accepted commands are zero; all-reject is not a pass. No installed model, publisher, Pi, microphone or device action was changed by this task.

The user's final two-slide document and the existing publisher's single Pi benchmark script remain deferred until a working model passes qualification. No A100 was used. This report records experiment evidence, not the deferred submission slides.

## Selected architecture and schema

Wake/VAD collects a complete utterance → mono 16 kHz PCM → 40-bin log-mel frontend at a 10 ms hop → six 128-channel context blocks with four attention heads and local convolutions → finite acoustic-cue, intent and explicit-light-target heads → exact scoring of 968 compiled finite candidates → calibrated rejection and schema-v3 typed JSON. The strided encoder emits a 20 ms output grid. It has no streaming cache or acoustic-streaming claim. It never produces an unconstrained transcript, calls a cloud recognizer or executes an action.

The schema preserves the approved defaults and slots: unified VOLUME operation/percentage; CALL defaults to Home with Mom/mother separate; MESSAGE is an empty-slot request without sending; lights default to all only when no target is supplied; explicit unknown targets reject; brightness retains light and integer percentage; PLAY_MUSIC has a song and optional artist; reminders use the editable finite task list. The [schema contract](EXPERIMENT-pcv-updated-voice-v1.md#final-architecture-contract) contains the complete mapping.

The final catalog adds only the already approved `call mother` phrase to the parent's 967 entries, preserving all previous candidates and the 40-output cue inventory. All 415 approved definitions pass an ideal-cue software oracle. That check uses neither audio nor the neural model and is **not** 415 successful spoken requests. Text catalog edits require fresh competition/rejection evaluation; they do not provide acoustic evidence for a new name.

A swappable unit comprises graph, input/output rate and state interface, normalization, label inventory, catalog, custom slots, schema and policy hashes. This candidate has `state_shapes=[]`, unlike the eleven-state causal parent. The candidate recipe records that difference. A graph must not be substituted into a different model's policy or state adapter. Publisher integration for this new interface has not been implemented or verified; it remains deferred with release qualification. The teacher is an offline training dependency and does not ship.

## Data, training and selection

Every downstream training/distillation waveform comes from the supplied [HF dataset](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands), pinned to `e3e203b360d5c3205d59f99ad65cd380f9c0f517`. Original published test/holdout membership is preserved. Human recordings are trained first, followed by synthetic recordings already included in that source. There is no newly generated audio, splicing, outside recording or test-speaker enrollment in these runs. The [source audit](EXPERIMENT-pcv-updated-voice-v1.md#2026-10-02-acquisition-and-preparation) documents hashes, waveform checks, speaker/pair separation, annotation ambiguity and exclusions; it does not claim independent listening or perfect utterance segmentation.

The authorized pretrained exception is frozen WavLM base-plus, revision `4c66d4806a428f2e922ccfa1a962776e232d487b`. Its external pretraining overlap is unaudited. Finite heads trained on its frozen representations reach 155/168 human and 108/108 supplied-synthetic development positives. A matched causal-student distillation attempt does not beat its parent; its [negative result](../experiments/pcv-updated-distill-v1/student-nonpromotion.json) is retained. The later context experiment changes architecture, output rate and schedule, so its gain is not an isolated proof that distillation caused the improvement.

The selected context student starts randomly, trains 24 human-only epochs, then 40 mixed epochs using hard supervision plus frozen teacher targets. The fitting manifest contains 56,556 rows: 1,215 supported human commands, 537 unsupported human requests, 48,831 isolated human numerals and 5,973 supplied synthetic commands. Numeral batches are bounded and semantic losses masked. The run uses seed 903, AdamW, 96 updates per epoch, batch 32/microbatch 8, a cosine learning-rate schedule and local RTX 4080 SUPER. Exact settings, source hashes and row IDs are frozen in the [training protocol](../experiments/pcv-updated-context-int8-v1/protocol.json). Mixed epoch 12 is selected by development, not final-epoch or test performance.

The final comparison uses the same 168 human positives, 63 human negatives, 108 synthetic positives and 968-entry catalog for every candidate:

| Candidate | Human exact | Synthetic exact | Raw human negative errors |
| --- | ---: | ---: | ---: |
| Causal parent v3 | 130/168 | 105/108 | 42/63 |
| Context width 80, FP32 | 138/168 | 106/108 | 30/63 |
| Context width 128, FP32 compression reference | 146/168 | 107/108 | 35/63 |
| Context width 128, INT8, selected | 146/168 | 107/108 | 36/63 |

The INT8 candidate preserves both exact-positive counts and adds one raw negative error relative to its FP32 reference, passing the previously frozen compression allowance. This is separate from acceptance quality. Raw counts mean top-ranked commands before the rejection policy, not dispatched requests. The [development receipt](../experiments/pcv-final-qualification-v1/development-selection.json) precedes calibration; the [selection lock](../experiments/pcv-final-qualification-v1/selection-before-test.json) fixes graph/catalog/policy before test. Earlier parent test results were already exposed, so this is a repeated benchmark, not a new blind corpus.

## Held-out results and failed gates

| Test evidence | Causal parent v3 | Selected context INT8 |
| --- | ---: | ---: |
| Human complete commands, raw | 398/609 (65.35%) | 473/609 (77.67%) |
| Supplied synthetic complete commands, raw | 3,227/3,368 (95.81%) | 3,286/3,368 (97.57%) |
| Human unsupported requests with a raw positive candidate | 145/187 | 120/187 |
| Separate single-speaker human holdout, raw | 79/84 | 80/84 |
| Separate supplied-synthetic holdout, raw | See parent receipt | 101/102 |

All 4,371 original test rows were replayed. The 207 human rows with ambiguous references are reported separately rather than silently relabeled as OOV or removed from the source denominator. Exact-command metrics use 609 supported and 187 unsupported human references. The 186-recording holdout remains separate; its high single-speaker accuracy is not pooled with the multi-source test.

Calibration cannot meet the frozen .99 precision/.90 recall/.01 human false-accept requirements. Its selected diagnostic threshold gives human 27/114 recall with 27/27 precision and 0/85 false accepts; synthetic recall is 560/1,009. Human calibration also lacks CALL, CREATE_REMINDER, LIST_REMINDERS, MESSAGE and TEMPERATURE. There are no supplied synthetic negatives, so synthetic rejection is unmeasured.

If the disabled policy were examined diagnostically on test, it would retain 184/609 correct human commands among 187 accepted (98.40% precision, 30.21% recall, 0/187 unsupported false accepts), and 1,788/3,368 correct synthetic commands among 1,790 accepted. These are diagnostic threshold outcomes only. The actual policy is disabled and accepts zero; neither useful coverage nor all-intent qualification is established.

| Exact baseline evidence, raw model | Utterances correct | Baseline phrases represented | Phrases correct on every available recording |
| --- | ---: | ---: | ---: |
| Human test | 197/240 | 71/93 | 43/71 |
| Supplied synthetic test | 3,286/3,368 | 93/93 | 49/93 |
| Human holdout | 65/67 | 67/93 | 65/67 |
| Supplied synthetic holdout | 101/102 | 93/93 | 92/93 |

Of all 415 requested phrases, human test covers 72 and synthetic test covers 93. Every other cell retains a zero denominator and untested status in the [test phrase tables](../experiments/pcv-final-qualification-v1/phrases-test.json). This does not complete the earlier Qwen/Dia/Parler benchmark, which remains stopped under the HF-only priority.

The [result receipt](../experiments/pcv-final-qualification-v1/results.json) contains complete per-intent, per-role, per-source and policy-mode results. Human raw examples include ALARM 30/32, VOLUME 66/94, BRIGHTNESS 13/23, LIGHT_ON 13/25, LIGHT_OFF 27/42 and CREATE_REMINDER 10/18. Training on the supplied synthetic baseline is useful but does not establish reliable human named-slot recognition.

## Export, replay and reproducibility

The trained FP32 graph passes the unchanged 0.00002 numerical tolerance with maximum error 0.0000076294. Dynamic INT8 quantizes constant MatMul weights per channel; the graph is 1,558,203 bytes (about 1.486 MiB). All twelve original-WAV endpoint probes have exactly matching fresh/cached frontend values, ONNX outputs and finite commands. This validates complete-utterance replay, not real-time waveform chunking or live capture.

On the Windows Intel Core i9-14900 host, one-thread CPU ONNX plus in-memory WAV decode, frontend and direct finite scoring measures 33.83 ms median / 118.37 ms p95 across 36 warm observations from twelve recordings. Session load is 74.19 ms. Another research workload was active, so these are contextual desktop measurements rather than controlled comparative performance. They exclude disk, utterance duration, wake/VAD and the 400 ms endpoint wait. Pi latency, RTF, peak memory and live-microphone behavior remain unmeasured.

Thirty-three focused source, split, teacher, distillation, contextual export and catalog tests pass. The close check verifies 30 protocol bindings, seven candidate bundle files and all 20 installed artifact hashes. The [verification receipt](../experiments/pcv-final-qualification-v1/verification-close.json) preserves these identities. The exporter emits two deprecation warnings; no test fails.

Candidate location: `.cache/pcv-final-qualification-v1/selected-evaluation/bundle/`. Model SHA-256: `0b2701bacaeff770d0098433e018123b4f6d607e9d8a893e251dbbafca6b3aef`. Catalog SHA-256: `8d301ee936100ad5f5c7bfad365c4b1265cdd40147d09b89fb153915ac8d81c5`. Training checkpoints, features and full predictions remain in their existing ignored experiment caches; compact evidence stays under `experiments/`. Nothing was committed or pushed by this task.

From the repository root, the completed stage entry points were:

```powershell
.venv/Scripts/python.exe scripts/train/pcv_teacher_data.py acquire
.venv/Scripts/python.exe scripts/train/pcv_teacher_data.py extract
.venv/Scripts/python.exe scripts/train/pcv_teacher_train.py
.venv/Scripts/python.exe scripts/train/pcv_student_distill.py targets
.venv/Scripts/python.exe scripts/train/pcv_student_distill.py
.venv/Scripts/python.exe scripts/train/pcv_context_training.py
.venv/Scripts/python.exe scripts/train/pcv_context_int8.py
.venv/Scripts/python.exe scripts/test/pcv_final_qualification.py prepare
.venv/Scripts/python.exe scripts/test/pcv_final_qualification.py
```

These entries preserve existing outputs and deliberately refuse to overwrite completed runs. A reproduction needs an explicit new run directory/version; these commands are not a destructive rerun instruction. The source protocols retain full recipes and dependencies.

## Remaining evidence and next experiment

Further training alone cannot supply the absent human calibration intents. Human fitting has no positive empty-slot MESSAGE example. The configured Dynamite/Butter/BTS names have no mentions in the published training metadata; no outside fallback has been invoked. Human recordings in test/holdout must not be moved into fitting to conceal these gaps.

Development inspection also identifies annotation uncertainty: examples such as `make the color of the light red` are marked unsupported by the conservative reference parser despite apparently matching the schema. That is a reference-audit finding, not permission to relabel test errors or claim a corrected score. Preserve this result and version any annotation revision before another experiment, with independent adjudication where the recording or intent is uncertain.

The next bounded work should audit fitting/development labels and missing slot evidence, then define a new frozen experiment. Use the supplied recordings wherever suitable; an outside recording is permitted only for an input with no suitable supplied match. Any fallback must carry provenance and separate denominators. New human calibration/test coverage and better rejection/recall are needed before enabling a model, starting the final slides or extending the Pi publisher. The result does not justify relaxing gates, manufacturing confidence values, pooling synthetic with human accuracy, or calling the 415 software checks spoken successes.
