# OptionB direct tiny-model fine-tune: unchanged parent retained

Date: **2026-10-23, UTC+08**. Status: **completed; strengthened v2 verification and independent code-review closure pass. Recognition-quality gate failed. No promotion.** Addresses ME-01/02/03/04/06/07/08 within the user-approved 93 OptionB verbatims /31 complete typed commands. No broader assignment, independent-listening, live-microphone or physical Pi completion is claimed.

## Decision

**Keep the original tiny checkpoint and its original policy; do not install either research fine-tune bundle.** Four real direct-continuation epochs did not beat the unchanged parent on development accepted recall, and all lost some raw human recognition. The predeclared epoch-zero fallback correctly retains the original epoch44 weights. The new ONNX graph is byte-identical to the parent, so there is **no acoustic improvement** to claim.

The separately calibrated research policy is `.985` posterior /`.30` margin, rather than the parent's `.99`/`.30`. It accepts slightly more commands but makes one wrong human-test decision. This is a threshold-only tradeoff, **not a promoted replacement**. Existing installed/submission/UI models, policies, private settings and provider behavior remain untouched.

The earlier [human-foundation continuation](EXPERIMENT-speaker-optionb-tiny-ft-v1.md) is a completed failed regression and remains preserved. Across its three arms and this direct study there were **2,688 real optimizer updates /86,016 sampled draws**, not new independent recordings. No further result-driven fitting or threshold revision follows these bounded runs.

## Why this follow-up was separately declared

The first study's development-selected model had raw human correctness103/171 and synthetic correctness387/873, versus141/171 and865/873 for the original mixed epoch44 checkpoint. Starting from the much earlier human-only epoch6 foundation did not retain the later learned coverage. Its objective preferred one extra development safe hit despite substantially worse raw recognition. That failure was already visible on development before final replay.

The [direct protocol](../experiments/speaker-optionb-tiny-direct-ft-v1/protocol.json) therefore declares a different parent and a recognition-retention floor, rather than rewriting the first study. Original test/holdout recordings had already been exposed: this is transparent regression research, not fresh-blind qualification.

## Frozen data, ancestry and recipe

- **Parent:** actual `.cache/speaker-optionb-tiny-v1/run-01/best.pt`, epoch44. The [data freeze](../experiments/speaker-optionb-tiny-direct-ft-v1/data-freeze.json) binds the exact checkpoint, data and sources. Its actual sampled ancestry contains15,204 IDs from the original23,315 TRAIN rows.
- **Inherited provenance limit:**351 actually sampled supplemental gap-fill recordings have historical publisher val/test/blank assignments. New replay filtering cannot erase those exposures. The publisher declares that corpus train-only for the current benchmark; the historical fields are not proof of current reserved-recording leakage. The caveat is retained, not silently repaired.
- **New replay:** all1,889 original humans plus4,283 provided synthetic recordings. Synthetic sampling retains at most two rows per complete-command/declared-speaker group, sorted by SHA256(id), independent of errors. All31 complete commands remain represented; no new audio, relabeling, TTS, splices, pseudo-labels or unresolved references are added.
- **Separation:** current6,075 reserved metadata rows were checked against actual parent training ancestry. Current train/dev/calibration/test/holdout views and eligibility remain unchanged; present recording/speaker/hash/pair/session/parent/alias keys are disjoint. Missing synthetic clone/voice-alias information prevents a stronger voice-independence claim. The normalizer remains the exact1,889-human TRAIN-only normalizer.
- **Same architecture:**258,912 parameters; one32-way complete-command/UNKNOWN head, width160, four separable dilated temporal blocks, stride2, masked mean/max pooling. No ASR/CTC/LLM, extra recognizer, keyword rescue or new runtime head.
- **Actual training:** seed83, four epochs of96 AdamW updates, batch32, learning rate.0001 to.000025, weight decay.0001, label smoothing.01, gradient clip5. Each batch draws12 human positives /12 human negatives /8 synthetic positives. Six human-negative draws come from fixed non-ASR-derived human TRAIN hards scored by the frozen parent. Actual draws:4,608 human-positive /4,608 human-negative /3,072 synthetic-positive. Original augmentation and inference preprocessing remain unchanged.

## Development selection and outcome

The unchanged parent is eligible as epoch0. Before ranking accepted recall, raw human-dev recall must be at least the actual parent value, and raw synthetic-dev recall must be no more than.01 below its parent value. Feasible accepted recall requires>=.99 precision,<=.01 task-negative false-accept rate and at least one correct accept. Raw human F1/precision and synthetic recall then break ties; exact ties retain the earlier epoch. Calibration never selects weights.

| Continuation epoch | Human raw correct /171 | Synthetic raw correct /873 | Dev-only correct accepts /171 | Dev wrong accepts | Retention floor |
|---|---:|---:|---:|---:|---|
| 0: unchanged parent |141|865|40|0|Pass|
| 1 |137|863|33|0|Fail|
| 2 |138|865|37|0|Fail|
| 3 |138|865|39|0|Fail|
| 4 |139|865|39|0|Fail|

These acceptance counts use each epoch's development-only operating point, not the final calibration-constrained policy. The original parent wins even without the retention safeguard:40 correct accepts versus at most39. The continuation did reduce some raw negative proposals, but did not improve usable accepted recall. Its last checkpoint genuinely differs from the parent; selecting epoch0 does not mean the optimizer was skipped.

The [model/policy lock](../experiments/speaker-optionb-tiny-direct-ft-v1/selection-before-test.json) was written before final replay. The graph has1,082,940 bytes, all258,912 parameters, all31 commands and the same class order. Export passes466 recorded human-dev/calibration parity checks plus three dynamic-batch/odd-even-frame fixtures; maximum recorded logit error4.7684e-6.

## Frozen human regression results

[Actual fresh replay](../experiments/speaker-optionb-tiny-direct-ft-v1/results.json), compared with the preserved original and failed first continuation:

| Model/policy | Holdout raw correct /80 | Holdout correct accepts /80 | Holdout wrong accepts | Test raw correct /609 | Test correct accepts /609 | Test wrong accepts |
|---|---:|---:|---:|---:|---:|---:|
| Original parent, original `.99` |76|3|0|469|41|0|
| Retained parent, research `.985` |76|5|0|469|59|1|
| First-study selected fine-tune, `.98` |35|0|0|277|8|2|

The new research gate has **6.25% holdout accepted recall**, **9.69% test accepted recall** and **98.33% test acceptance precision**. It fails the declared90% recall /99% precision target. At diagnostic`.95`, the retained parent accepts26/80 holdout positives with no observed wrong accepts, but133/609 test positives with **three** wrong accepts. That diagnostic was not selected for deployment. Zero observed errors in the small holdout cohort do not establish general safety.

Human-test negatives are256; human-holdout negatives are14. A wrong supported slot is counted both as a false accepted command and a missed correct command, not ignored because the intent matched. Two ambiguous holdout references and185 ambiguous test references remain present but unscored. Synthetic results, per-command/intent tables and all93 wording rows remain separate in the machine exports; silence variants are paired stress trials, not independent humans.

## Diagnosis and architecture judgment

The [complete source/label/architecture audit](EXPERIMENT-speaker-optionb-tiny-ft-v1.md#label-and-data-recheck) rechecked all23,315 original training references against pinned source metadata and archived annotation rules: zero rule disagreements or stale class projections. This **does not prove listener truth**:288 human TRAIN references are ASR-derived, and the36-case [training-only listening queue](../experiments/speaker-optionb-tiny-ft-v1/human-train-reference-review.csv) is blank, not adjudicated.

The strongest measured gaps remain:

1. **Rejection does not generalize:** the original parent's raw wrong proposals are1/605 human TRAIN negatives versus27/85 human DEV negatives. Its synthetic TRAIN pool has zero negatives, although synthetic DEV includes122. Reserved negatives were never moved into fitting.
2. **Real-human coverage is incomplete:** only25/31 commands and31/93 exact wordings have original-human training evidence. Missing cells are brightness60%, brightness100%, reminder `task.drink_water`, reminder `task.exercise`, MESSAGE and TIMER10s.
3. **Valid output is not correct understanding:** unsupported requests can map to nearby supported commands. In the rejected first-study candidate, two stored22°C references are accepted as18°C at posteriors0.9845 and0.9945. Typed validation accepts the output values but cannot check them against the audio independently. These errors are against frozen metadata, not new listening adjudications.
4. **A high score does not resolve this gap:** removing smoothing and emphasizing hard negatives did not beat the matched control in the first study, while the direct continuation did not beat its stronger parent. The evidence does not isolate one universal causal source or establish that a larger network is required.

No class-order, training/runtime feature, normalization, masking/padding or ONNX implementation defect was demonstrated. The simple CNN remains a computationally suitable baseline, **not a qualified recognizer**. Each encoded frame has3.705s PCM support, but pooling uses the entire valid utterance; it is not an utterance cutoff. Pooling can lose sufficiently distant word order. No real-speech causal ablation shows that this limitation produced the current errors. No architecture expansion is justified solely by these failed continuations.

The next supported intervention is independent listening/adjudication of existing TRAIN disagreements and representative approved TRAIN near-negatives, plus original-human coverage for the six missing command cells. Do not move reserved examples, change labels from predictions, merely lower thresholds, add keyword rescue or claim that synthetic positives establish real-speaker safety.

## Resources and verification boundaries

The [fresh Torch-free CPU probe](../experiments/speaker-optionb-tiny-direct-ft-v1/desktop-resource-results.json) covers256 original human-dev recordings: one thread, **p50 5.69ms /p95 8.64ms**, peak working set86,077,440 bytes, session load24.16ms. Windows private commit is separately recorded and is not physical RSS. Disk loading, speaking time, wake/VAD/endpoint waits and UI/TTS/providers are excluded. These are not physical Pi or live-microphone measurements. Actual WAV CLI smoke produces PLAY_MUSIC with `actions_enabled=false`; a selected development smoke is not an accuracy estimate.

Final replay contains4,896 original/stress cases, with reference/variant identities and per-command/intent/verbatim exports. The [first direct verifier review](../experiments/speaker-optionb-tiny-direct-ft-v1/independent-review-v1.json) identified incomplete diagnostic-decision, digest and exact-cohort checks. Its original source and receipt are preserved byte-for-byte, not silently overwritten.

The additive [v2 verifier](../experiments/speaker-optionb-tiny-direct-ft-v1/verify-results-v2.py) now **passes the actual full corpus**. Its [immutable receipt](../experiments/speaker-optionb-tiny-direct-ft-v1/verification-v2.json) reconciles all4,896 exact `(split,parent_id,variant)` identities, all raw/selected/fixed`.95` decisions, capture-fault rejections, aggregate metrics, declared prediction/selection hashes and all three CSV counts. It binds actual predictions, results, selection and CSV bytes; reconstructs the exact1,889-human normalization; reruns32 actual-graph original/stress cases; verifies395 protected inputs including the original289; and confirms the selected parent weights are unchanged. No model, policy, label, membership or measured-result reselection occurred.

**137 tests pass across nine focused modules**, including60 new software-only verifier fixtures. The new tests include same-count cohort substitutions, source-hash drift, forged diagnostic accepts, capture faults, selection/prediction digest drift and exclusive receipt preservation. These are tamper/logic fixtures, not additional acoustic recordings. The [independent v2 closure review](../experiments/speaker-optionb-tiny-direct-ft-v1/independent-review-v2.json) passes with no logic/security findings and60 separately rerun software tests; all three original findings are closed. Reconstructing every stored curve-rank component and adding more retention-floor fixtures remain non-blocking future audit-hardening suggestions; no new acoustic experiment is implied.

## Reproduction and artifacts

From project root with the existing `.venv/Scripts/python.exe`; preparation/fit/export/replay are exclusive-create steps and must not overwrite this run:

1. `-m scripts.train.speaker_optionb_tiny_direct_ft prepare` — frozen subset and inherited-parent ancestry; completed.
2. `-m scripts.train.speaker_optionb_tiny_direct_ft fit` — actual four-epoch continuation with unchanged-parent fallback; completed.
3. `-u -c 'from pathlib import Path; from scripts.test.speaker_optionb_tiny_ft_eval import select_candidate; select_candidate(Path("experiments/speaker-optionb-tiny-direct-ft-v1"))'` — guarded development-only model selection, export, then dev/cal policy; completed.
4. The same helper's `evaluate_candidate(Path("experiments/speaker-optionb-tiny-direct-ft-v1"))` — locked final original/stress replay; completed.
5. `experiments/speaker-optionb-tiny-direct-ft-v1/benchmark-runtime.py` — fresh Torch-free CPU/memory probe; completed.
6. `experiments/speaker-optionb-tiny-direct-ft-v1/verify-results-v2.py` — rerun strengthened full verification without modifying receipts; completed. `--write-receipt` was used once to exclusively create the existing `verification-v2.json`.

Source: [direct trainer](../scripts/train/speaker_optionb_tiny_direct_ft.py), [three optimizer/lineage/fallback fixtures](../scripts/test/test_speaker_optionb_tiny_direct_ft.py). Runtime artifacts and raw predictions remain under `.cache/speaker-optionb-tiny-direct-ft-v1/run-01/`. No dependency installation, model installation, commit/push, microphone/device test or physical actuation occurred.
