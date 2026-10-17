# Compact VCM recall policy and Raspberry Pi 4 handoff

Study `speaker-compact-recall-v1`, authorized 2026-10-16, addresses ME-01 through ME-08. The user selected the roughly 935k model class, a hard 1.2M total VCM parameter ceiling and fewer rejections of correct commands. The bounded fit, frozen human recall policy and locked regression are complete; device-package verification is recorded separately. The [training protocol](../experiments/speaker-compact-recall-v1/protocol.json) freezes the budget and methods. This is a separate study; prior models and installed artifacts remain preserved.

## Model and training changes

The compact `PooledContextVCM` has **935,567 parameters**: width 128, six context blocks, four attention heads, feedforward width 256 and dropout .1. All encoder, acoustic/semantic heads and pooling parameters count toward the 1,200,000 ceiling. This is 82 parameters above the historical 935,485-parameter `ContextualVCM`, retaining learned semantic pooling for the existing endpoint ABI. Compared with the completed 2,066,831-parameter study, only width/feedforward scale and the declared supervision exclusions change in fitting; the frontend, outputs, normalization, catalog and decoder remain the same. These simultaneous changes and one seed do not isolate a causal architecture effect.

Outputs remain 40 acoustic labels (39 phones plus blank), 19 intent labels and two target labels. The 40-bin frontend, 20 ms output hop, complete-utterance execution and 1,079-entry finite catalog are unchanged. Runtime emits a finite typed command or rejection, without a transcript parser, ASR/STT fallback, LLM or cloud inference. No synthesizer or training framework ships in the Pi archive.

Training supervision uses **23,289 rows: 1,863 original human and 21,426 earlier synthetic**. The compact control excludes 184 new-Qwen training rows and six prepared rows flagged with unresolved slot-metadata conflicts. Exclusions are preserved separately; this is a metadata filter, without listener adjudication or label correction. Existing normalization still uses the parent 1,869 human training recordings, including the six supervision-excluded rows. It remains training-only and identical to the parent. Evaluation references are never filtered using training labels or model outcomes.

Development retains 1,251 rows (256 original human / 995 synthetic). Human calibration and all published test/holdout membership remain fixed. Synthetic provenance stays explicit. Earlier synthesis contributed designed conditions, not new independent human speakers, listener-verified labels or proven accents. This compact pass generates no new audio. Source details and limitations remain in the [primary dataset receipt](../data/sources/ai231-primary.source.json), [supplemental receipt](../data/sources/fil-supplemental.source.json) and [prior synthetic preparation](../experiments/speaker-synthetic-completion-v1/results.json).

| Setting | Human foundation | Mixed control |
|---|---:|---:|
| Epochs | 24 | 32 |
| Updates per epoch | 96 | 96 |
| Human positive / UNKNOWN per batch | 24 / 8 | 12 / 4 |
| Earlier synthetic per batch | 0 | 16 |
| Batch / microbatch | 32 / 8 | 32 / 8 |
| Initial learning rate | .0008 | .00025 |

Seed is 53. Sampling selects intent, complete command, declared speaker and then recording. The selected human foundation initializes the single mixed arm. AdamW uses weight decay .001, gradient clip 5, cosine decay to .00002 and the established CTC + .3 intent cross-entropy + .15 target cross-entropy losses. Development checkpoint selection uses the original .95 intent/.90 required-slot rule every four epochs, with the frozen human feasibility/macro-speaker ordering. If no checkpoint satisfies the quality constraints, its diagnostic selection remains explicitly unqualified.

One shared A100 is used because the SSH scheduler controller is unreachable and device memory has headroom; other jobs are untouched. Reported elapsed time is shared-host timing, without an exclusive throughput claim. CUDA TF32 is enabled; CUDA CTC backward is not bitwise deterministic. One seeded fit does not establish seed robustness. No new teacher, head, search arm or result-driven training extension is authorized.

## Augmentations and grounded studies

[Park et al., SpecAugment, Interspeech 2019](https://arxiv.org/abs/1904.08779) describe augmentation in filterbank feature space. This study reuses the conservative training-only adaptation below. Their ASR results do not establish a gain for this finite-command model, and no unmasked ablation is run here.

| Augmentation | Actual policy |
|---|---|
| Frequency mask | At most one; width uniformly 0–4 of 40 bins; probability .5 |
| Time mask | At most one; width uniformly 0–min(10, floor(.05 × valid frames)); probability .5 |
| Fill | Zero in normalized feature space |
| Scope | Fresh copies of sampled training features only |

Normalization, development, calibration, test and holdout are unaugmented. Original audio/features remain unchanged. No time warping, speed/noise/reverberation perturbation, splicing or new synthesis is added. The short mask cap is a bounded design choice rather than a published optimum.

[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) distinguish confidence calibration from classification accuracy. Here intent and conditional catalog-slot scores remain scores, without a claim that .90 means 90% empirical accuracy. No temperature-scaling component is added. Human fitting constraints are checked directly; they cannot prove unseen-human precision or fill missing human classes.

[Warden, Speech Commands, 2018](https://arxiv.org/abs/1804.03209) motivates reproducible, task-specific command evaluation. This experiment evaluates complete typed commands with rejection cases rather than claiming isolated-word benchmark performance. No new Speech Commands audio is used. The [ONNX Runtime quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) motivates checking accuracy after dynamic INT8 conversion. We reuse the constant-weight MatMul exporter and require a graph below two MiB, at most one lost raw correct human/synthetic development command per stratum and at most one extra raw human negative error. Smaller bytes or desktop timings cannot establish Pi speed.

The prior [full-model study](EXPERIMENT-speaker-final-refinement-v1.md) selected control over added Qwen data on its human development rule. Its negative added-data result motivates retaining that control recipe, without a new synthesis round. The [citation ledger](../experiments/speaker-compact-recall-v1/references.json) records each source's role and limits.

The original compact INT8 attempt is **1,595,296 bytes**, with raw human development correct commands 152->153 and synthetic correct commands 902->902, but raw human UNKNOWN candidates 39->41. Two additional raw negatives fail its <=1-extra guard. Accepted human UNKNOWNs remain 1/63; this does not erase the raw-candidate failure. The [original export receipt](../experiments/speaker-compact-recall-v1/export-development.json) and [stop receipt](../experiments/speaker-compact-recall-v1/export-stop.json) remain unchanged, without calibration/test inference at that stage.

The selected fallback is the **same 935,567 weights in FP32**, 3,864,875 bytes (3.69 MiB). Its numerical export parity passes (maximum absolute error 5.73e-6, tolerance 2e-5). A [separate pre-evaluation fallback protocol](../experiments/speaker-compact-recall-v1/fp32-recovery-v2/release-protocol-before-evaluation.json) declares a four-MiB FP32 graph ceiling while retaining the user's 1.2M parameter limit. This preserves full-precision behavior without retraining, quantizer search or weakening the failed INT8 guard. FP32 has a larger footprint and may cost more CPU/RAM; target performance needs physical measurement. All human fitting and published replay below use FP32. The fallback orchestration was added after the single independent review and receives guard/integration verification rather than another independent review.

## Relaxed rejection policy

Weights are selected first; INT8 development fidelity is checked next. Only original human development/calibration predictions fit the separate demo policy. For each emitted intent, the frozen grid is .50, .60, .70, .80, .85, .90 and .95. An intent needs positive references in both views before relaxation. Each candidate must add zero wrong accepted commands and zero UNKNOWN accepts on **each** fitting view relative to .95. Selection maximizes combined correct accepts; ties keep the higher threshold. Missing classes retain .95. The complete-command schema, semantic-intent agreement and .90 required-slot/light-target checks remain enforced.

This policy changes rejection rather than adding neural parameters. It does not guarantee unchanged errors on unseen recordings. No synthetic examples, published test or holdout results select thresholds. The graph-bound policy, strict qualified policy, source bindings and before-test lock are frozen before one published regression replay. Previously examined test data remain regression evidence, without a new blind-test claim. No threshold revisions follow that replay.

Demo output and the original qualified output are logged separately, alongside the fixed .95/.90 comparison. Strict qualification still requires all human calibration intents, precision >=.99, recall >=.90 and UNKNOWN false-accept rate <=.01. Relaxation cannot turn qualification on. Missing original human evidence remains visible. Actions are disabled.

## Device contract and commands

The package targets Raspberry Pi 4 with 64-bit Linux ARM64, CPython 3.12 and glibc >=2.28, using one ONNX Runtime CPU thread. It includes the existing eleven hash-locked offline wheels and four wake/VAD graphs. The target lock pins ORT 1.30.0 / NumPy 2.5.3; the SSH export/replay environment uses ORT 1.23.2 / NumPy 2.3.5 and Python 3.13.9. ARM execution and cross-version behavior remain unmeasured. PortAudio/libsndfile and the required interpreter must already be available on the target.

Extract into a fresh directory. Run each measurement with a fresh output directory:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3.12 pi_test.py --verify-only
python3.12 pi_test.py --setup --engine onnxruntime --mode demo --threads 1 --output results/demo-01
.venv/bin/python pi_test.py --mode qualified --threads 1 --output results/qualified-01
.venv/bin/python pi_test.py --list-mics
.venv/bin/python pi_test.py --mode demo --threads 1 --guided --mic INDEX --output results/live-01
vcgencmd measure_temp
vcgencmd get_throttled
```

The setup command installs bundled wheels with `--no-index` and hash verification. Physical Pi identity, latency, peak RSS, temperature/throttling and failed capture attempts need device measurements; preserve raw before/after thermal probes. The small bundled licensed SLURP replay is a previously observed smoke subset. Missing phrase recordings remain untested. Wake/VAD does not establish owner identity or overlap recovery. No physical device, action integration or installed-artifact replacement is performed by this study.

## Results

The bounded fit, human policy fit and locked FP32 regression are complete. The INT8 fidelity failure remains preserved. Strict qualification is **disabled**; demo acceptance is reported separately. No thresholds were revised from these test results.

Selected mixed epoch: 32; selected human-foundation epoch: 8. Fitting/development time after feature loading: 1442.97 seconds on one shared NVIDIA A100-SXM4-40GB; peak allocated CUDA memory: 536,026,624 bytes. These are training-host resources, not Pi measurements.

| Locked FP32 result | Human test | Synthetic test | Human holdout | Synthetic holdout |
|---|---:|---:|---:|---:|
| Raw correct | 493/636 (77.52%) | 3293/3368 (97.77%) | 77/83 (92.77%) | 99/100 (99.00%) |
| Fixed .95/.90 correct accepts | 189/636 (29.72%) | 3011/3368 (89.40%) | 15/83 (18.07%) | 95/100 (95.00%) |
| Relaxed demo correct accepts | 229/636 (36.01%) | 3099/3368 (92.01%) | 24/83 (28.92%) | 96/100 (96.00%) |
| Fixed precision | 189/202 (93.56%) | 3011/3035 (99.21%) | 15/15 (100.00%) | 95/96 (98.96%) |
| Relaxed precision | 229/250 (91.60%) | 3099/3127 (99.10%) | 24/25 (96.00%) | 96/97 (98.97%) |
| Fixed UNKNOWN accepts | 2/229 | No negatives | 0/11 | No negatives |
| Relaxed UNKNOWN accepts | 2/229 | No negatives | 1/11 | No negatives |

On the same compact FP32 graph, relaxation recovers 40 correct human test commands. Wrong accepted commands change from 13 to 21; UNKNOWN accepts change from 2 to 2. Human precision changes 93.56% -> 91.60%. These errors remain visible even though fitting-view constraints pass.

The prior 2,066,831-parameter INT8 model had 516/636 raw correct and 274/636 fixed correct accepts, precision 274/297, UNKNOWN accepts 3/229. The compact FP32 model has 493/636 raw correct, 189/636 fixed correct accepts and 229/636 relaxed correct accepts. This is a descriptive previously examined regression with different width/supervision/selected foundation and precision; it cannot isolate one cause.

All published rows remain: test 4418 total / 4233 scored / 185 separately excluded; holdout 196 / 194 / 2. Capture faults: 0 test, 0 holdout. Ambiguity is never converted to UNKNOWN. Human holdout positives represent only two declared people.

### Human fitting evidence and thresholds

| View | Correct accepts before -> after | Wrong accepts before -> after | UNKNOWN accepts before -> after |
|---|---:|---:|---:|
| calibration (n=210) | 33 -> 40 | 0 -> 0 | 0 -> 0 |
| development (n=256) | 84 -> 96 | 6 -> 6 | 1 -> 1 |

| Intent | Frozen threshold | Human test raw / N | Fixed correct | Relaxed correct | Relaxed precision |
|---|---:|---:|---:|---:|---:|
| ALARM | 0.80 | 29/32 | 6 | 10 | 10/11 (90.91%) |
| BRIGHTNESS | 0.60 | 13/23 | 0 | 5 | 5/7 (71.43%) |
| CALL | 0.95 | 6/6 | 6 | 6 | 6/6 (100.00%) |
| COLOR | 0.95 | 54/67 | 12 | 12 | 12/12 (100.00%) |
| CREATE_REMINDER | 0.95 | 14/18 | 0 | 0 | 0/0 (—) |
| LIGHT_OFF | 0.95 | 31/45 | 6 | 6 | 6/8 (75.00%) |
| LIGHT_ON | 0.95 | 15/30 | 3 | 3 | 3/5 (60.00%) |
| LIST_REMINDERS | 0.95 | 16/18 | 4 | 4 | 4/4 (100.00%) |
| MESSAGE | 0.95 | 6/6 | 1 | 1 | 1/1 (100.00%) |
| NEXT | 0.60 | 33/36 | 23 | 28 | 28/31 (90.32%) |
| PAUSE | 0.80 | 47/50 | 30 | 40 | 40/40 (100.00%) |
| PLAY_MUSIC | 0.95 | 5/8 | 1 | 1 | 1/1 (100.00%) |
| STOP | 0.90 | 68/72 | 42 | 46 | 46/48 (95.83%) |
| TEMPERATURE | 0.95 | 27/28 | 4 | 4 | 4/4 (100.00%) |
| TIME | 0.70 | 42/49 | 19 | 29 | 29/29 (100.00%) |
| TIMER | 0.85 | 18/19 | 4 | 5 | 5/5 (100.00%) |
| VOLUME | 0.95 | 58/113 | 26 | 26 | 26/27 (96.30%) |
| WEATHER | 0.80 | 11/16 | 2 | 3 | 3/11 (27.27%) |

Missing positive intent coverage in a human fitting view, kept at .95: CALL, MESSAGE, LIST_REMINDERS, TEMPERATURE, CREATE_REMINDER.
Strict qualification remains disabled. Missing human calibration intents: CALL, CREATE_REMINDER, LIST_REMINDERS, MESSAGE, TEMPERATURE. Qualified-mode accepted commands are zero; demo recall does not waive the strict gate.

### Exact OptionB wordings

Human: 233 scored exact spoken-wording recordings across 71/93 definitions. Raw correct 203; fixed correct/wrong/rejected 79/3/151; relaxed correct/wrong/rejected 91/3/139. Zero-recording definitions remain untested.
Synthetic: 3368 scored exact spoken-wording recordings across 93/93 definitions. Raw correct 3293; fixed correct/wrong/rejected 3011/24/333; relaxed correct/wrong/rejected 3099/28/241. Zero-recording definitions remain untested.

Matching uses the existing spoken normalization for case, punctuation, digits/number words, percent/degrees and AM/PM, plus matching typed reference and original record ID. No publisher variation bucket or paraphrase is substituted. Publisher transcript/prompt metadata is not independent listening verification. The original results.json optionb_wordings fields use stricter literal text normalization (170 human /1,799synthetic takes); those counts remain preserved and are not comparable coverage estimates. The spoken-normalized supplement restores the earlier comparison denominators of233human/3,368synthetic takes without changing any model output or policy.

Every wording, including zero cells, is in [optionb-spoken-wordings.csv](../experiments/speaker-compact-recall-v1/fp32-recovery-v2/optionb-spoken-wordings.csv), [full spoken-wording cells](../experiments/speaker-compact-recall-v1/fp32-recovery-v2/optionb-spoken-normalized.json) and [analysis identity/method receipt](../experiments/speaker-compact-recall-v1/fp32-recovery-v2/optionb-analysis-receipt.json). Complete per-intent synthetic/holdout results are in the [locked results receipt](../experiments/speaker-compact-recall-v1/fp32-recovery-v2/results.json).

The Pi archive embeds this methods/acoustic-results snapshot and core evidence under `evidence/`. Repository-relative links refer to the source checkout. A separate post-build verification receipt records the actual extracted CLI, dependency closure and desktop resource measurements; it cannot establish ARM installation or physical Pi readiness.
