# Restricted shared-VCM demo: shared-demo-slots-v1

## Outcome and decision

DEV selected **heads**, epoch **38**, **175,653 parameters**. Both matched fits completed **4,608 optimizer updates each**. The selected model passes the predeclared aggregate retained-human DEV guard: **132/167**, versus unchanged parent **128/167** on identical in-scope recording IDs. This is a research candidate, not an installed replacement or qualification of every retained command.

Locked new declared-voice TEST: **106/108 (98.15%) exact demo-slot commands**, with **0 wrong accepted outcomes** and **2 rejected supported requests**. All supported demo commands: **301/408 (73.77%)**. These are connected-command TTS results, not human demo accuracy. A perfect finite cell is an observation, never a guarantee for future speakers.

The shared-head layout is the best candidate in this one matched DEV comparison. The flat control is also below the cap. No seed-replicated superiority, global architecture optimum, or maximum acoustic/phonetic separation has been established. No fitting, reselection, threshold tuning or value changes followed TEST.

**Principal failed gate: robustness to added speech.** Exact clean verbatims achieve **203/204 (99.51%)**, but the unseen context wrappers achieve only **98/204 (48.04%)**:105 rejections and1 wrong supported outcome. Thus this candidate supports a deliberately verbatim-focused demonstration, not the broader aspiration that irrelevant speech leave commands unaffected. It is not a uniformly reliable replacement. Original-UNKNOWN official regression also has7/16 false accepts;0/32 on the small new negative set must not hide that failure.

## Authorized scope and typed slots

[Frozen protocol](../experiments/shared-demo-slots-v1/protocol.json); requirements ME-01/02/03/04/06/07/08, UC-04/06/08.

- One categorical percentage slot: **0, 50, 100**; shared by absolute-volume and brightness actions. Zero is a legitimate value, distinct from missing.
- One categorical alarm-time slot: **06:00, 12:30, 21:45**, spoken **six A M / twelve thirty P M / nine forty-five P M**. Fixed timezone **Asia/Manila** is configured metadata, not a spoken slot.
- The shared-head model emits **29 action outputs +4 percentage outputs +4 alarm-time outputs**; each slot head has three allowed values plus missing/not-applicable. The flat control emits35 complete outcomes.
- The new finite ontology has **34 supported complete typed outcomes +UNKNOWN**, covered by **102 canonical wordings**. The action ontology retains nonnumeric target identities and other finite command meanings.
- Original brightness20/60 and alarm08:00/21:00 are explicitly outside this restricted demo. Their original source references remain unchanged; they are not relabelled as the nearest supported value. Original frozen studies/catalogs remain preserved.
- Unsupported numeric/time combinations are not enabled. Missing required slots reject. Slot logits cannot rescue a rejected action. No runtime ASR, transcription parser, LLM or cloud recognition.
- Values were chosen as practical distinct demo candidates, not proved maximally separated. Dictionary distance alone is not acoustic validation. Minutes/meridiem are encoded by each complete alarm category, not independently composed.

## Data, reservations and normalization

The declared synthesis completed2200/2200 tasks: TRAIN1320 / DEV440 / TEST440, with6/2/2 provider voice IDs. Each voice supplies102 canonical wordings with clean/context forms and16 negative controls. Each complete outcome has6 examples per voice, and each of the nine demo-slot targets has12 TEST recordings across the two reserved voices.

| Split | Declared provider voice IDs |
|---|---|
| TRAIN | en-US-GuyNeural, en-US-JennyNeural, en-GB-SoniaNeural, en-AU-NatashaNeural, en-SG-WayneNeural, en-PH-JamesNeural |
| DEV | en-US-AriaNeural, en-CA-LiamNeural |
| TEST | en-US-ChristopherNeural, en-IE-EmilyNeural |

TEST IDs Christopher/Emily differ from the earlier declared training/DEV/test voice lists; opaque historical publisher voices prevent proving physical voice independence. TTS speaker identities do not establish independent human speakers. Official202 and historical cohorts are already exposed regression.

Preparation froze**1760 new TRAIN/DEV feature arrays**, extracted no new TEST features, independently replayed**672 new WAVs +58 parent WAVs**, and exactly reproduced the **1889 original-human TRAIN** normalization. TEST PCM was decoded for identity reservations only before lock. The inherited serialized feature pack materializes original calibration arrays; those arrays are excluded from optimizer/selection inputs. This is disclosed rather than described as never loaded.

| Prepared view | Sampling-group recording counts |
|---|---|
| calibration | human_negative: 118, human_positive: 92 |
| new-dev | demo_slot_positive: 108, synthetic_negative: 32, synthetic_positive: 300 |
| original-dev | human_negative: 89, human_positive: 167, synthetic_negative: 360, synthetic_positive: 1076 |
| test | demo_slot_positive: 108, synthetic_negative: 32, synthetic_positive: 300 |
| train | demo_slot_positive: 324, human_negative: 685, human_positive: 1204, synthetic_negative: 3107, synthetic_positive: 20638 |

Original TRAIN/DEV, official benchmark and prior reserved PCM identities were read-only. The admission audit rejects exact ID/file/PCM overlap and declared within-study cross-split voice reuse. Original expected commands and original class indices remain beside new demo projections; no automatic ASR relabelling, unknown-word deletion, or nearest-value coercion.

## Architecture and controlled optimization

Both models use the existing40-bin causal25ms/10ms-hop log-mel frontend, stride2 temporal input, width128 causal residual CNN, four kernel7 blocks with dilations1/2/4/8, masked mean/max pooling and dropout0.1. Both start from random weights with identical verified initial shared encoders, rather than transplanting the independently trained volume classifier.

| Arm | Neural parameters | DEV selected epoch | Retained-human DEV | Demo-slot DEV | All demo-supported DEV | Eligible |
|---|---:|---:|---:|---:|---:|---|
| flat | 175139 | 40 | 130/167 | 108/108 | 400/408 | True |
| heads | 175653 | 38 | 132/167 | 108/108 | 400/408 | True |

Each arm completes6 human-foundation +42 mixed epochs ×96 steps,32 examples/step. AdamW LR0.002, mixed cosine decay to0.00005, weight decay0.01, BF16 CUDA, gradient clip5. Foundation quotas:24 human-positive/8 human-negative. Mixed quotas:8 human-positive/4 human-negative/8 generic-synthetic-positive/8 demo-slot-positive/4 synthetic-negative. Head loss: action CE +0.5 percentage CE +0.5 alarm CE, explicitly including missing/not-applicable targets. Flat control uses complete-class CE. Normalization, sampled identities, feature augmentation sequence and optimizer budgets are shared; dropout RNG is reset after differently sized head initialization.

Selection is DEV-only and supported-first: guard retained-human correct count at least the parent, then mean human supported accuracy and new demo-slot macro accuracy, followed by declared supported tie-breaks. Negative false accepts, F1 and precision do not rank or veto weights. The fixed argmax policy is not confidence calibrated. Final-epoch weights remain diagnostic when they lose the DEV selector.

| Arm | Selected human | Final human | Selected slot exact | Final slot exact | Parent-correct human IDs lost | Newly correct human IDs | Synthetic draw fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 130 | 127 | 108 | 108 | 12 | 14 | 54.688% |
| heads | 132 | 124 | 108 | 108 | 12 | 16 | 54.688% |

The aggregate retention pass does not preserve every parent-correct recording. Full lost/gained IDs and selected/final DEV predictions are retained in the independent training audit. The independent verifier reconstructs actual source quotas, sample counts, ordered sample sequence including augmentation RNG advances, full schedule/AdamW steps, and CPU replays of selected/final DEV weights.

## Locked new declared-voice TEST

| Target | Recordings | Exact | Wrong accepted | Rejected | Accuracy |
|---|---:|---:|---:|---:|---:|
| ALARM {"time": "06:00", "timezone": "Asia/Manila"} | 12 | 11 | 0 | 1 | 91.67% |
| ALARM {"time": "12:30", "timezone": "Asia/Manila"} | 12 | 12 | 0 | 0 | 100.00% |
| ALARM {"time": "21:45", "timezone": "Asia/Manila"} | 12 | 12 | 0 | 0 | 100.00% |
| BRIGHTNESS {"light_id": "light.all", "percent": 0} | 12 | 12 | 0 | 0 | 100.00% |
| BRIGHTNESS {"light_id": "light.all", "percent": 100} | 12 | 12 | 0 | 0 | 100.00% |
| BRIGHTNESS {"light_id": "light.all", "percent": 50} | 12 | 12 | 0 | 0 | 100.00% |
| VOLUME {"operation": "set", "percent": 0} | 12 | 12 | 0 | 0 | 100.00% |
| VOLUME {"operation": "set", "percent": 100} | 12 | 11 | 0 | 1 | 91.67% |
| VOLUME {"operation": "set", "percent": 50} | 12 | 12 | 0 | 0 | 100.00% |

| TEST subgroup | Supported | Exact | Wrong accepted | Rejected | Negative false accepts |
|---|---:|---:|---:|---:|---:|
| en-IE-EmilyNeural | 204 | 151 | 1 | 52 | 0/16 |
| en-US-ChristopherNeural | 204 | 150 | 0 | 54 | 0/16 |
| clean | 204 | 203 | 0 | 1 | 0/0 |
| context | 204 | 98 | 1 | 105 | 0/0 |
| negative | 0 | 0 | 0 | 0 | 0/32 |

Clean/context paired both-correct: **98/204**. This tests the explicitly authored contexts, not literal deletion of arbitrary OOV speech or word alignment. New TEST negative false accepts: **0/32**; diagnostic, not a demonstrated open-set safety guarantee.

## Exposed official202 regression — restricted-scope accounting

The immutable original benchmark payload/references were reconstructed after the new selection lock. This is exposed regression, not fresh human qualification; its denominator differs from the old full31-command task because four old outcomes are outside the restricted demo.

| Official subgroup | Rows | Supported | Exact | Wrong accepted positives | Rejected supported | Negative false accepts |
|---|---:|---:|---:|---:|---:|---:|
| formerly_supported_now_outside_demo | 24 | 0 | 0 | 0 | 0 | 1/24 |
| original_unknown | 16 | 0 | 0 | 0 | 0 | 7/16 |
| retained_human_supported | 77 | 77 | 71 | 3 | 3 | 0/0 |
| retained_supported | 162 | 162 | 155 | 3 | 4 | 0/0 |

No full-old-task improvement is claimed from a narrowed denominator. Wrong values/targets lose complete-command credit even if the intent is correct. Source truth remains available for every now-outside-demo row. The original parent remains the original-scope baseline.

## Verification, latency and delivery

- **100 focused software tests passed** (100 passed in 19.52s); not the entire repository test suite.
- Exact new and official source-cohort reconstruction, **642 prediction rows +642 native events**, independent original-PCM runtime replay and all nine CSV cells verified.
- Torch/ONNX/native logits and complete decisions agree on **642 rows**; capture-fault IDs: `[]`.
- PC native latency,1 CPU thread: **p50 3.434 ms /p95 5.595 ms**. Includes frontend/normalizer/graph/decoder; excludes disk, utterance/capture duration and endpoint wait. Not a Pi measurement.
- Environment: `{"executable": "D:\\School\\AI231\\.worktrees\\me2\\smart-speaker-fresh\\.cache\\optionb-clean-pipeline-ad4d626-v2\\.venv\\Scripts\\python.exe", "gpu": "NVIDIA GeForce RTX 3080", "numpy": "2.3.5", "onnxruntime": "1.23.2", "platform": "Windows-11-10.0.26200-SP0", "python": "3.13.14", "torch": "2.10.0+cu128"}`.
- Selected checkpoint SHA-256: `0164568747c67657836659dc1322947d6282d829717b93efe218269b2b456069`; selection-lock SHA-256: `6956c5c6192207fb527de2fc5ae26229a7a863338431cbde797cdcbac3adb7eb`.
- Research delivery contains the selected single ONNX graph, exact normalizer/class/policy/recipe, recovery checkpoint,102 supported-wording CSV, all9 slot cells, predictions, native events, both fit curves/results and independent audits. No audio/feature corpus or credentials.
- Packaging must be verified separately through `delivery/verification.json`: archive hashes/member manifest and cold extracted literal CLI/API probes for all nine slot targets, without Torch or Python network calls. This report does not substitute for that receipt.
- Installed defaults, original study locks, disabled qualification/actions, branches and Git history are untouched; no deployment, microphone capture, Pi run, physical action, commit, push or other-agent delegation.

## Evidence locations and reproducibility

Study root: `D:\School\AI231\smart-speaker\.cache\shared-demo-slots-v1`. Source revision: `3c05b3beb46f06c3108089e0d8c79474670c6cb7` plus explicitly additive uncommitted experiment sources bound by the freeze/selection receipts.

- `data/{tasks.json,synthesis-design.json,data-freeze.json,independent-preparation-audit.json}`: authored prompts, PCM reservations and feature/normalizer evidence.
- `freeze-before-fit.json`, `parent-retained-human-DEV.json`: exact sources, untouched parent and scope-specific human floor.
- `runs/{flat_s53,heads_s53}/{config.json,best.pt,last.pt,training-result.json,training-curve.jsonl}`: complete matching fits. `best-diagnostic.pt` is separately retained.
- `selection-DEV.json`, `independent-training-audit.json`, `independent-DEV-predictions.jsonl`: supported selection, schedule/draw audits and per-recording retention churn.
- `selection-before-test.json`, `locked/bundle/`: immutable chosen model, frontend/catalog/policy/export lock before TEST frontend/inference.
- `evaluation/{results.json,per-slot.csv,predictions.jsonl,native-events.jsonl,reference-rows.jsonl,verification.json,independent-audit.json}`: all denominators, confusions and original-audio replay.
- Source entry points: `scripts.train.demo_slots_tasks`, `scripts.train.demo_slots_data`, `scripts.train.demo_slots_study {freeze,smoke,train}`, `scripts.test.demo_slots_evaluate {lock,evaluate}`. Independent verifier/package entry points live under `experiments/shared-demo-slots-v1/`.

## Remaining work

Before claiming a reliable human demonstration, collect separately authorized human recordings of these exact wordings/slots, freeze references before inference, and measure them without changing the selected weights or policy. Physical microphone/wake/endpoint and Pi latency remain untested. Do not add a nearest-value decoder, silent time defaults, broad paraphrases or post-TEST tuning to hide failures.
