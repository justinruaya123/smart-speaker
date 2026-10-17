# Iris release v1: final experiment and product evidence

This new version addresses ME-01–08 and UC-01–10. The hypothesis was improved accepted exact commands across held-out voices within frozen development limits. Data, loss, schedule, catalog and selection were frozen before fitting. Existing test populations, recordings, protocols and earlier results were preserved. The isolated release branch archived historical research before named cleanup; original worktrees and shared recordings remain untouched.

## Matched fits and decision

| Width | Objective | Seed | Selected epoch | Human macro-speaker accepted recall | Human correct | Synthetic correct | Training seconds |
|---|---|---|---|---|---|---|---|
| compact | ce | 53 | 1 | 0.3118 | 74 | 1087 | 120.9 |
| compact | ce | 67 | 3 | 0.3560 | 81 | 1133 | 125.8 |
| compact | supcon | 53 | 1 | 0.3164 | 74 | 1084 | 109.1 |
| compact | supcon | 67 | 3 | 0.3561 | 83 | 1131 | 124.9 |
| full | ce | 53 | 6 | 0.2807 | 77 | 1228 | 164.2 |
| full | supcon | 53 | 14 | 0.2801 | 83 | 1271 | 214.1 |

| Model | Personal correct accepts | Personal wrong accepts | Personal negative accepts | Human correct accepts | Human UNKNOWN accepts | 196-row accepted exact |
|---|---|---|---|---|---|---|
| V3 baseline | 31/36 (86.1%) | 1 | 1/52 | 165/636 (25.9%) | 12/229 | 138/196 (70.4%) |
| Diversity candidate | 35/36 (97.2%) | 0 | 0/52 | 171/636 (26.9%) | 14/229 | 147/196 (75.0%) |
| Selected release | 31/36 (86.1%) | 1 | 1/52 | 165/636 (25.9%) | 12/229 | 138/196 (70.4%) |

Promotion decision: `{"bundle": ".cache/iris-release-v1/diversity/release-export", "bundle_sha256": "d7671bb393f7a81b4a354e6922b8a469c76a57aec8264ba3ff030fbba1adaa03", "candidate_bundle_sha256": "b80312ca48802bde1d68e0e18cedf07360d4d8d27361835aa30158df86241466", "experiment": "iris-release-v1", "guards": {"human_correct_not_below_baseline": true, "human_unknown_not_above_baseline": false, "official_exact_not_below_baseline": true, "official_population_unchanged": true, "personal_30_of_36": true, "personal_population_unchanged": true}, "promoted": false, "selected": "vcm-mvp-hey-iris-v3", "selection_before_test_sha256": "006e795f976010ee070f853b6355339884e4e733e60d44757037a1d7094eb861", "source": ".cache/iris-release-v1/inputs/fallback-bundles/compact", "strict_qualification_passed": false, "threshold_retuned_after_test": false, "width": "compact"}`.

## Selected per-intent regression

| Intent | Human raw | Human correct accepts | Human wrong accepts | Synthetic correct accepts |
|---|---|---|---|---|
| ALARM | 20/32 (62.5%) | 11/32 (34.4%) | 1 | 309/372 (83.1%) |
| BRIGHTNESS | 13/23 (56.5%) | 9/23 (39.1%) | 1 | 304/398 (76.4%) |
| CALL | 6/6 (100.0%) | 6/6 (100.0%) | 0 | 93/96 (96.9%) |
| COLOR | 47/67 (70.1%) | 17/67 (25.4%) | 1 | 261/305 (85.6%) |
| CREATE_REMINDER | 4/18 (22.2%) | 4/18 (22.2%) | 4 | 370/405 (91.4%) |
| LIGHT_OFF | 20/45 (44.4%) | 2/45 (4.4%) | 4 | 34/69 (49.3%) |
| LIGHT_ON | 10/30 (33.3%) | 2/30 (6.7%) | 4 | 62/83 (74.7%) |
| LIST_REMINDERS | 8/18 (44.4%) | 5/18 (27.8%) | 0 | 97/102 (95.1%) |
| MESSAGE | 0/6 (0.0%) | 0/6 (0.0%) | 0 | 119/123 (96.7%) |
| NEXT | 28/36 (77.8%) | 27/36 (75.0%) | 0 | 98/99 (99.0%) |
| PAUSE | 43/50 (86.0%) | 38/50 (76.0%) | 0 | 81/88 (92.0%) |
| PLAY_MUSIC | 2/8 (25.0%) | 0/8 (0.0%) | 2 | 59/69 (85.5%) |
| STOP | 65/72 (90.3%) | 5/72 (6.9%) | 0 | 7/69 (10.1%) |
| TEMPERATURE | 17/28 (60.7%) | 5/28 (17.9%) | 4 | 303/395 (76.7%) |
| TIME | 31/49 (63.3%) | 21/49 (42.9%) | 1 | 78/84 (92.9%) |
| TIMER | 13/19 (68.4%) | 5/19 (26.3%) | 0 | 347/404 (85.9%) |
| VOLUME | 53/113 (46.9%) | 8/113 (7.1%) | 2 | 91/138 (65.9%) |
| WEATHER | 7/16 (43.8%) | 0/16 (0.0%) | 1 | 49/69 (71.0%) |

All slot values/roles, sources, requested accent conditions and individual speakers/voice IDs retain denominators in recognition-results.json. Full-command correctness, per-role correctness, raw predictions and accepted decisions remain distinct. Missing acoustic coverage is unsupported; entity configuration is not new training.

## Architecture, method, datasets and limits

# Iris finite voice-command model

Selected model: **vcm-mvp-hey-iris-v3 compact**. Parameters: **969,928**. The release retains the predeclared V3 baseline because the new candidate failed promotion. This is an experimental software release; strict acoustic qualification remains failed/unresolved.

Iris maps a whole mono 16 kHz utterance directly to one of 328 finite outputs (UNKNOWN included). The reference keeps 31 baseline intent-and-slot outcomes, all 93 wordings and requested personal commands. Untrained outputs are blocked. Configuration binds recognized IDs to supplied entities; it cannot teach a new spoken name. General questions, unrestricted reminder text and arbitrary new names/values are unsupported.

## Measured command-only results

| Model | Personal correct accepts | Personal wrong accepts | Personal negative accepts | Human correct accepts | Human UNKNOWN accepts | 196-row accepted exact |
|---|---|---|---|---|---|---|
| V3 baseline | 31/36 (86.1%) | 1 | 1/52 | 165/636 (25.9%) | 12/229 | 138/196 (70.4%) |
| Diversity candidate | 35/36 (97.2%) | 0 | 0/52 | 171/636 (26.9%) | 14/229 | 147/196 (75.0%) |
| Selected release | 31/36 (86.1%) | 1 | 1/52 | 165/636 (25.9%) | 12/229 | 138/196 (70.4%) |

Fresh designed-voice holdout: **259/444 (58.3%)** correct accepts, 9 wrong accepted positives and 3/120 negative accepts. These are designed synthetic voices, not verified human demographic coverage. The unchanged personal cohort has **zero original-human positive recordings**.

The unpromoted diversity candidate reached **320/444 (72.1%)** on the same fresh voice holdout, with 3 wrong accepted positives and 1/120 negative accepts. Its improved positive acceptance does not waive the failed human UNKNOWN guard.

The unchanged 196-row scorer gives raw exact **172/196 (87.8%)**, accepted exact **138/196 (70.4%)**. Raw argmax is diagnostic and does not execute. Human regression precision is 81.7% over 202 accepts; its 185 ambiguous references are excluded from accuracy and retained in evidence. Promotion guards: personal_population_unchanged: pass; personal_30_of_36: pass; human_correct_not_below_baseline: pass; human_unknown_not_above_baseline: FAIL; official_population_unchanged: pass; official_exact_not_below_baseline: pass. Test results only veto promotion; thresholds were not retuned after test.

## Architecture and training

Microphone → Hey Iris → command capture/Silero VAD → direct VCM plus independent speaker check → typed validation → configured executor → UI receipt. Recognition has no ASR/STT, phonetic/transcription decoder, LLM, remote model service or automatic model fallback. Offline synthesis is solely a training-data process and never ships as recognition.

The VCM uses a 40-bin log-mel frontend, stride-two convolution, six contextual blocks (four-head attention, temporal depthwise convolution and feedforward layers), masked mean pooling and one atomic command classifier. Compared widths are 128 (969,928 parameters) and 192 (2,118,280). The candidate recipe chosen on development is supcon, seed 67. It continues our own pinned V2 project encoder/head; no external pretrained command encoder is used. If the candidate fails promotion, the selected runtime instead retains the unchanged same-width V2 weights with the predeclared V3 acceptance policy.

Four matched fits compare CE and CE + 0.1 cross-speaker contrastive loss (temperature 0.1) at both widths, seed 53. Both objectives repeat at the development-selected width, seed 67. The frozen bound is 16 epochs × 96 updates, batch 32/microbatch eight, AdamW 1e-4, cosine decay, weight decay .001, smoothing .02, clip norm five and declared early stopping. Batches balance human/synthetic source, command and speaker, include UNKNOWN examples, ±6 dB training-only gain and bounded time/frequency masks. Development chooses accepted human macro-speaker recall, then human/synthetic correct counts and compute cost. Development/calibration fit empirical class acceptance thresholds; these scores are not guarantees of correctness.

The independent random-init speaker encoder uses six temporal depthwise blocks, masked mean/std pooling, a 128-D embedding and overlap head. It compares one wake speaker to one following command, rejects mismatch/overlap/missing/short evidence and resets. It has **no persistent user enrollment** and does not separate overlapping speech. Command embeddings are never used as identity. Its data comprise 9,429 original-human recordings (24.264 hours) plus separately labelled constructed mixtures and synthetic calibration/test proxies. Original human wake-to-command and real captured overlap are unmeasured.

| Speaker trial population | Same-speaker rejection | Impostor acceptance | Speaker-switch acceptance | Overlap missed | Clean speech flagged overlap |
|---|---|---|---|---|---|
| Original-human proxies | 279/1005 (27.8%) | 72/1005 (7.2%) | 56/1005 (5.6%) | 280/1005 (27.9%) | 77/1005 (7.7%) |
| LibriSpeech-train-clean-100 | 60/352 (17.0%) | 16/352 (4.5%) | 15/352 (4.3%) | 39/352 (11.1%) | 32/352 (9.1%) |
| VCM-original-human | 219/653 (33.5%) | 56/653 (8.6%) | 41/653 (6.3%) | 241/653 (36.9%) | 45/653 (6.9%) |
| Synthetic wake proxies | 0/16 (0.0%) | 2/16 (12.5%) | 0/16 (0.0%) | 8/16 (50.0%) | 0/16 (0.0%) |

Failed or unmeasured speaker gates: **human_wake_qualified, physical_pi_measured, proxy_impostor_FAR_le_5_percent, proxy_overlap_miss_le_20_percent**. In particular, the 5% impostor-acceptance and 20% overlap-miss proxy limits fail. This experimental gate is not identity authentication. Thresholds remain locked despite those failures.

Actual ONNX plus the full session policy on short original-human proxies rejected 62/80 (77.5%) same-speaker trials, accepted 0/80 (0.0%) impostors and 0/80 (0.0%) switches, and accepted 1/80 (1.2%) constructed overlaps. These results differ from raw embedding/head metrics because speech-duration, continuity and overlap requirements also reject. The wake reference requires at least 0.8 seconds of detected voiced speech; short Hey Iris phrases can fail before command recognition. Two older synthetic wake references both failed this minimum. These are proxy tests, not original human wake pairs. On the new four-voice held-out set, only 1/32 full standalone synthetic wake clips armed the owner gate; 31 were too short. This is a substantial usability limitation. Standalone full clips are not live detector windows, so no end-to-end human wake acceptance is inferred.

The own Hey Iris network has 82,114 parameters. The expanded and previous wake candidates are compared on calibration only. Selected: expanded. Test: `{"detected": 29, "duplicate_detections": 0, "false_wakes": 0, "false_wakes_per_hour": 0.0, "missed": 3, "negative_exposure": "separate clips; resets and synthetic silence padding excluded from exposure; not household continuous audio", "negative_n": 660, "negative_seconds": 1160.808125, "positive_n": 32, "recall": 0.90625}`. Zero-event false-wake upper 95% bound, when applicable: 9.290627755378923 per hour. No original-human wake positives exist; synthetic recall and short negative exposure do not establish live wake reliability.

## Use, actions and reproducibility

Download the verified Pi ZIP and hashes from this repository. Extract to a new folder; run `python3 start.py --list-mics`, configure your Home Assistant entities/scripts and private token, then `python3 start.py --mode serve --mic INDEX --benchmarks skip`. First dependency setup needs internet; acoustic inference is local thereafter. WAV replay and benchmarks always remain non-executing. Tentative intent/slots never execute. Only a final accepted command with speaker evidence and valid configured bindings can execute. Receipts distinguish service acceptance from observed device state. Jobs and execution IDs persist atomically; unknown outcomes never retry automatically.

From the [GitHub repository](https://github.com/justinruaya123/smart-speaker): `python reproduce.py` installs the pinned environment, prepares immutable inputs in a new isolated workspace, actually reruns the six VCM continuation fits, calibrates/exports/evaluates, builds the UI and packages. `python train.py` remains the lower-level workflow. `python train.py --skip-train` verifies published project artifacts and reconstructs the delivery; `--publish` uploads verified results. Project initializers are an immutable shortcut. `--from-scratch` declares a separate random-init study and does not reproduce identical historical V2 weights. Exact historical executable lineage/protocols remain in the immutable research archive; the current comparison's optimizer/RNG snapshots and input hashes are preserved. Full random-init 48-epoch reconstruction is not claimed measured by a tiny resume test.

GitHub main owns the executable recipe, dependency setup, inference graphs, [benchmark plots](https://github.com/justinruaya123/smart-speaker/tree/main/docs/assets) and reproduction receipts. Clone it and run `python reproduce.py --device cuda` on a DGX after selecting one available GPU with `CUDA_VISIBLE_DEVICES`. Hugging Face holds the dataset and **private** model/checkpoint artifacts; authenticate with an authorized Hugging Face login or `HF_TOKEN`. The workflow resolves pinned revisions and hashes, excludes completed VCM fits from new training and keeps original restricted audio at upstream sources. The [fresh-clone report](https://github.com/justinruaya123/smart-speaker/blob/main/docs/EXPERIMENT-iris-reproduction-v1.md) distinguishes newly fitted candidates from a retained frozen baseline. The complete historical initializer chain has not been replayed.

Original human data, supplied synthetic data, designed voices and ambiguous/unsupported cases are reported separately. Restricted primary/supplemental recordings are referenced at original immutable upstream revisions, not newly uploaded. LibriSpeech is CC BY 4.0; Qwen synthesis model source is Apache 2.0; other component/data terms remain component-specific and research-limited. See the dataset card and NOTICES for restrictions.

Physical Pi latency/RAM/thermals, fresh original-human wake/command qualification, live microphone quality and observed real Home Assistant device outcomes remain **unmeasured operator acceptance steps**. The software-verification receipt lists the executed checks and results. Desktop measurements do not establish Pi performance.

## Resource measurements

| Recorded host cost (one CPU thread) | Value |
|---|---|
| All packaged graph/metadata variants | 11.29 MiB |
| Five model sessions: combined load | 645.7 ms |
| Python process peak RSS | 197.2 MiB |
| Final VCM call p50 / p95 | 9.8 / 11.0 ms |
| Wake, previews, final VCM, VAD and speaker checks: serial compute budget p50 / p95 | 97.3 / 128.4 ms |

Eight fixed synthetic wake/command pairs were profiled. The complete serial budget includes every half-second prefix, endpoint-silence computation and continuity checks; it is not live user latency. Audio elapsed time, endpoint waiting, file decoding, worker IPC, Node/browser and provider calls are excluded. Recorded host/settings and individual component timings remain in the resource receipt. UI/process integration costs are reported separately; physical Pi cost is unmeasured.

## Evidence and methods

- [Command outcomes, per-intent/value/role/source/speaker denominators](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/recognition-results.json)
- [Promotion decision and failed guards](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/release-choice.json)
- [Weight/policy/catalog lock before test](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/selection-before-test.json)
- [Matched training commands, seeds and optimizer snapshots](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/training-execution.json)
- [Preserved pre-fit source and bounded compatibility checks](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/reproduction-prefit-compatibility.json)
- [Actual Torch/ONNX/ncnn command decision parity](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/decision-parity.json)
- [Independent speaker metrics](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/owner-results.json)
- [Runtime owner/session proxy results](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/owner-export.json)
- [Wake recall/duplicates/negative exposure](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/wake-results.json)
- [Frozen schedules and input hashes](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/protocol.json)
- [Verified software checks](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/software-verification.json)
- [Full runtime resource accounting](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/pipeline-resources.json)
- [Compiled product and resident-worker integration](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/ui-final-integration.json)
- [Extracted-package offline replay under both engines](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/package-replay.json)
- [Independent empty-cache reconstruction](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/reproduction-fresh-clone-check.json)
- [Independent LibriSpeech download and preparation](https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/reproduction-owner-fresh-preparation.json)

The training-only loss adapts [Khosla et al., Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html); masks adapt [Park et al., SpecAugment](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html). These are adaptations, not reproductions of their paper scores. [LibriSpeech](https://www.openslr.org/12/) supplies audio and speaker labels only. The [Qwen model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) describes the offline generator; requested voice characteristics are not independently verified.

## Implementation and verification findings

An independent review found and reproduced stale unrelated package selection, catch-up preview bursts and an inherited indefinite-capture edge. Failing regressions preceded fixes: current release path/hash matching, monotonic 2 Hz preview scheduling, and bounded capture deadline/reset. Owner/VCM acceptance is an intersection; a later application policy cannot rescue an ownership rejection. Plain WAV previews explicitly declare the action-free benchmark profile.

Final review also reproduced truncated test-cache reuse, alternate-release verification against the wrong model paths, and parity source-root misattribution. Replay caches now publish hash-bound completion receipts with complete ordered sample IDs; the official wrapper also binds its unchanged scorer's variation/slot CSV. Unsupported alternate reconstruction paths and shadow source roots reject. Exact preserved-source AST checks permit completed-fit reuse across the evaluation-only patch and redundant metadata-keyword correction while rejecting changes to loss, ranking, data or acceptance policy. The training execution receipt preserves the scratch wrapper's final-log collision and recovered metadata serialization failure; all six fits completed without rerunning or altering their weights.

The final delivery audit reproduced a fresh-start configuration failure: the generated example selected Home Assistant media playback without its required entity ID. Source and package examples now leave that provider unconfigured. A regression runs the actual first-start configuration exporter and gateway listeners with untouched templates. The final packaged UI check uses those defaults without an empty-bindings override; supplied entities remain an operator configuration step. Earlier package/proof bytes are preserved separately.

The independent source reconstruction also exposed a build-order defect: the Python package-inventory test required compiled UI files before the workflow had built them. UI installation, tests, checks and compilation now precede Python package tests. After a successful cold UI build, the retained copied run `fresh-orderfix` exposed an omitted root `benchmark.py` launcher in the reconstruction inventory. A trace of all nine application target/benchmark modes and an independent import/UI/bootstrap audit found no further required source omissions. Regressions start without compiled UI output, verify the build boundary and check every application source read against the published inventory. The supported publisher also includes card-linked late evidence and reconstruction copies as explicit hash-verified plan items; missing linked evidence rejects before upload, without binding a proof into the release it measures. Failed copied runs, previously completed acoustic evaluation/parity results and independently acquired data remain preserved.

A second GPU experiment was attempted under the five-second guard and stopped before GPU allocation when GPU4 rose by 20 utilization points. The primary synthesis continued on GPU5; subsequent fits used one GPU sequentially. Other users' jobs were preserved. Software/UI/API fixtures establish behavior, not fresh human or device quality. No threshold changes follow held-out test failures.

Source and package reconstruction use pinned inputs, hashes, source/environment receipts and exact optimizer/RNG checkpoints. The separate random-init route has a genuine interrupted/resumed CPU fit check, but its full 48-epoch study was not run in this release. Historical initializer lineage is retained on the archive; published initializers are the verified shortcut for the bounded matched experiment.

See [dataset audits](DATASETS.md), [Pi handoff and operator acceptance](HANDOFF-iris-rpi-v1.md), [archive lifecycle/recovery](DOCS.md), and [current machine receipts](../experiments/iris-release-v1/). The actual final graphs, package and publication revisions are pinned by the reproduction/publication receipts. No physical Pi, live microphone or external HA device test is claimed.
