# Compact checkpoint-selection repair

Study `speaker-pipeline-repair-v1`, authorized 2026-10-16. Status: bounded regression complete; candidate passes the frozen aggregate human gate and is selected for Pi comparison. Addresses ME-02/03/04/06 directly and preserves ME-01/05/07/08 contracts. Installed artifacts and prior archives are unchanged.

## Question and fixed scope

Does selecting recognition weights independently of deployment acceptance recover useful human complete commands without enlarging the model? The control is the completed [compact study](EXPERIMENT-speaker-compact-recall-v1.md), selected mixed epoch 32 and its human-fitted relaxed policy.

The architecture stays exactly `PooledContextVCM(width=128, layers=6, heads=4, feedforward=256, dropout=.1)`, **935,567 total VCM parameters** under the hard 1,200,000 limit. The 40-bin frontend, human-training normalization, all 1,079 finite catalog entries, acoustic labels, semantic heads and 82-parameter pooling are unchanged. Inference remains complete-utterance finite-command recognition with no transcription output, teacher or LLM.

The confirmed selection behavior is narrower than a causal accuracy claim. Human epoch 8 was selected at 125/193 raw correct and 13/13 accepted precision; saved epoch 24 has 144/193 raw correct and 69/73 accepted precision. The acceptance feasibility rule favored epoch 8. Human epoch 24 is the explicitly authorized available alternative, not a claim that its speaker-balanced recognition is best among all human epochs. In mixed development, epoch 28 has 155/193 raw correct and .726403 macro speaker raw recall, versus epoch 32's 152/193 and .676924. These facts motivate the comparison; they do not establish which deployed policy is better.

## Protocol

The [approved plan](PLAN-speaker-pipeline-repair-v1.md) specifies the complete decision order. The machine [protocol](../experiments/speaker-pipeline-repair-v1/protocol.json) binds sources, inputs, checkpoints and the control before candidate calibration.

1. Reuse saved mixed epoch 28, which wins the corrected recognition rule over the saved mixed development history. Export and compare FP32 using the same human dev/calibration policy fitter as the control. Avoid training if it passes.
2. If that candidate fails, run one mixed-control continuation from the saved human epoch 24 `last.pt`: 32 epochs, 96 updates/epoch, seed 53, batch 32/microbatch 8, 12 human positive + 4 human negative + 16 existing synthetic examples per batch. Do not repeat the human stage.
3. Select mixed weights by human raw macro speaker recall, then human raw exact count, fewer raw human negative candidates, synthetic raw recall and earlier epoch. Acceptance confidence/precision do not choose weights. Calibration never chooses an epoch.
4. Fit the unchanged per-intent threshold grid `.50,.60,.70,.80,.85,.90,.95` on original human dev/calibration. Preserve `.90` slot/target checks, schema/intent agreement and `.95` for classes lacking positive coverage in either fitting view.
5. Promotion requires at least 96 development and 40 calibration correct accepts, a strict combined increase over 136, development wrong/UNKNOWN no greater than 6/1, and calibration wrong/UNKNOWN both zero. Wrong accepts include UNKNOWN errors. No fitting view may lose correct commands.
6. Only a development-promoted graph/policy receives a locked replay of the already examined test and holdout. Final recommendation additionally requires test >229 correct, <=21 wrong, <=2 UNKNOWN; holdout >=24 correct, <=1 wrong, <=1 UNKNOWN. Failure stops promotion; it does not trigger another fit or threshold change.

Data stays 23,289 training rows (1,863 human + 21,426 existing synthetic), 1,251 development rows (256 human + 995 synthetic), original human calibration 210, published test 4,418 and holdout 196. The 184 new Qwen and six unresolved supervision-conflict rows remain excluded from training; normalization retains its original 1,869-human-training basis. Ambiguous/publisher-ineligible evaluation references stay excluded and are never converted to UNKNOWN. Population/reference identity checks prevent silently comparing different examples.

Loss stays CTC + .30 intent CE + .15 target CE; AdamW weight decay .001, gradient clip 5, mixed LR .00025 with the existing cosine schedule down to .00002. The optimizer and seeded sampling/augmentation streams match the earlier mixed stage, conditional on the different initial weights. CUDA determinism remains warn-only; numerical reproducibility is not a bitwise guarantee.

## Augmentations and grounding

There are no new augmentations. The existing training-only adaptation of [SpecAugment (Park et al., 2019)](https://arxiv.org/abs/1904.08779) masks at most four frequency bins and `min(10, floor(.05*T))` time frames, each with probability .5, once per axis. Zero is the normalized human-training mean. No waveform synthesis, noise, reverb, speed perturbation, splicing or time warping is added. The paper studies ASR; its results do not establish gains for this finite-command model, and this comparison cannot isolate masking's effect.

[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) distinguish confidence calibration from classification performance on image/document tasks. This motivates keeping recognition selection separate from acceptance fitting; the checkpoint audit supplies this project's direct evidence. No temperature-scaling component or guaranteed recall gain is inferred from that paper.

The control's failed INT8 guard remains a negative result. This comparison uses FP32 only, <4 MiB and maximum numerical export error <2e-5, preserving the older exporter's separate two-MiB resource verdict. [ONNX Runtime's quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) describes accuracy loss and validation; it does not prove Pi timing or justify relaxing a failed guard. The [citation ledger](../experiments/speaker-pipeline-repair-v1/references.json) retains inherited studies and explicit application limits. Primary pages were checked on 2026-10-16.

## Results and decision

The recognition rule selects saved mixed epoch 28. Its FP32 graph has 3,864,875 bytes (3.69 MiB), exactly the same byte count and 935,567 parameters as the epoch-32 control. The graph SHA-256 is `fc87c98c242ccb6f37ed9fcd84bad3869b8c90eaae6267fdab2a1a5af2737f6d`. Maximum numerical export error is 6.198883056640625e-6, below 2e-5. The old exporter's two-MiB resource verdict remains false; the separately frozen four-MiB guard passes.

| Original human fitting view | Positive / UNKNOWN recordings | Control correct / wrong / UNKNOWN accepts | Epoch 28 correct / wrong / UNKNOWN accepts |
| --- | --- | --- | --- |
| Development | 193 / 63 | 96 / 6 / 1 | 99 / 5 / 1 |
| Calibration | 122 / 88 | 40 / 0 / 0 | 44 / 0 / 0 |

This is +7 combined correct accepts, one fewer development wrong accept, unchanged UNKNOWN accepts and no per-view loss. [The decision](../experiments/speaker-pipeline-repair-v1/saved-epoch28/decision.json) and [graph-bound thresholds](../experiments/speaker-pipeline-repair-v1/saved-epoch28/recall-policy.json) preserve the evidence. The candidate's own unchanged fixed rule gives 80 development and 32 calibration correct accepts; its separately fitted relaxed rule yields 99 and 44. Fitting-view gains are selection evidence, not held-out performance.

**No additional training was performed.** The optional continuation from human epoch 24 is skipped because saved epoch 28 passes the predeclared fitting-view rule. This result does not test the foundation-change hypothesis. The graph and policies were frozen in [selection-before-test.json](../experiments/speaker-pipeline-repair-v1/selection-before-test.json) before the single previously examined regression replay. The locked regression and final decision follow below.

### Locked regression comparison

The same frozen test/holdout membership is used for both checkpoints. Test: 4,418 published rows, 4,233 eligible, 185 ambiguous/ineligible exclusions. Holdout: 196 published, 194 eligible, two exclusions. Capture faults remain in the denominator; they are not silently dropped.

| View | Positives / UNKNOWN | Raw correct: epoch 32 → 28 | Fixed correct: 32 → 28 | Relaxed correct: 32 → 28 | Relaxed wrong: 32 → 28 | UNKNOWN accepts: 32 → 28 | New relaxed precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test/human | 636 / 229 | 493 → 489 | 189 → 189 | 229 → 238 | 21 → 19 | 2 → 1 | 92.61% |
| test/synthetic | 3368 / 0 | 3293 → 3300 | 3011 → 2988 | 3099 → 3078 | 28 → 18 | 0 → 0 | 99.42% |
| holdout/human | 83 / 11 | 77 → 77 | 15 → 15 | 24 → 26 | 1 → 1 | 1 → 1 | 96.30% |
| holdout/synthetic | 100 / 0 | 99 → 99 | 95 → 94 | 96 → 94 | 1 → 1 | 0 → 0 | 98.95% |

**Frozen regression gate: PASS.** All predeclared human test/holdout count limits pass; this supports a bounded regression improvement, not full qualification.

The gain is small and uneven. Human accepted-command recall rises 36.01% -> 37.42% (+1.42 percentage points) and precision 91.60% -> 92.61%, but raw human recognition falls 493 -> 489. Synthetic correct accepts fall 3,099 -> 3,078, while synthetic raw correctness improves 3,293 -> 3,300. The unchanged fixed rule still accepts exactly 189 correct human commands. This is an improvement in the joint checkpoint/policy operating point, not evidence of improved raw human recognition or a benefit from larger capacity.

Per-intent regressions matter for the full assignment: BRIGHTNESS correct accepts fall 5 -> 0 of 23, VOLUME 26 -> 21 of 113, COLOR 12 -> 10 of 67 and LIGHT_OFF 6 -> 5 of 45. STOP improves 46 -> 55, TIMER 5 -> 9 and several other classes improve. The aggregate gate did not require per-intent nonregression. Do not describe the candidate as an across-intent upgrade or use these test errors to refit thresholds. The epoch-32 control archive remains available for comparison and recovery.

No new training or GPU fitting was performed. Both checkpoints originate from the same completed compact run and human-epoch-8 foundation. Only saved-checkpoint choice and the unchanged calibration procedure are tested here; the human-epoch-24 continuation hypothesis remains untested.

### Human test by intent

| Intent | Reference recordings | Raw correct: epoch 32 → 28 | Relaxed correct: 32 → 28 | New relaxed recall | New correct / accepted |
| --- | --- | --- | --- | --- | --- |
| ALARM | 32 | 29 → 29 | 10 → 10 | 31.25% | 10 / 11 |
| BRIGHTNESS | 23 | 13 → 13 | 5 → 0 | 0.00% | 0 / 1 |
| CALL | 6 | 6 → 6 | 6 → 6 | 100.00% | 6 / 6 |
| COLOR | 67 | 54 → 50 | 12 → 10 | 14.93% | 10 / 10 |
| CREATE_REMINDER | 18 | 14 → 15 | 0 → 0 | 0.00% | 0 / 0 |
| LIGHT_OFF | 45 | 31 → 31 | 6 → 5 | 11.11% | 5 / 5 |
| LIGHT_ON | 30 | 15 → 15 | 3 → 5 | 16.67% | 5 / 9 |
| LIST_REMINDERS | 18 | 16 → 16 | 4 → 4 | 22.22% | 4 / 4 |
| MESSAGE | 6 | 6 → 6 | 1 → 1 | 16.67% | 1 / 1 |
| NEXT | 36 | 33 → 33 | 28 → 28 | 77.78% | 28 / 31 |
| PAUSE | 50 | 47 → 47 | 40 → 42 | 84.00% | 42 / 42 |
| PLAY_MUSIC | 8 | 5 → 5 | 1 → 1 | 12.50% | 1 / 1 |
| STOP | 72 | 68 → 67 | 46 → 55 | 76.39% | 55 / 57 |
| TEMPERATURE | 28 | 27 → 27 | 4 → 6 | 21.43% | 6 / 6 |
| TIME | 49 | 42 → 42 | 29 → 31 | 63.27% | 31 / 31 |
| TIMER | 19 | 18 → 18 | 5 → 9 | 47.37% | 9 / 9 |
| VOLUME | 113 | 58 → 58 | 26 → 21 | 18.58% | 21 / 22 |
| WEATHER | 16 | 11 → 11 | 3 → 4 | 25.00% | 4 / 11 |

Accepted counts group by predicted intent; precision therefore includes wrong-intent predictions and UNKNOWN inputs. Complete-command correctness requires every typed slot. The UNKNOWN denominator remains separate (229 test, 11 holdout). All synthetic/holdout per-intent cells are in [results.json](../experiments/speaker-pipeline-repair-v1/results.json); the [slot-role receipt](../experiments/speaker-pipeline-repair-v1/role-results.json) counts typed slot values with matching intent and explicitly different denominators.

### OptionB spoken-wording coverage

Human: 233 recordings covering 71/93 wordings; raw correct 204; relaxed correct/wrong/rejected 97/4/132.
Synthetic: 3368 recordings covering 93/93 wordings; raw correct 3300; relaxed correct/wrong/rejected 3078/18/272.

The human OptionB subset improves from 91 to 97 correct accepts but adds one wrong accept (3 -> 4). This subset-specific error regression is separate from the overall human wrong-accept decrease.

Matching uses the same existing spoken normalization as the control (digits/number words, percent/degrees, AM/PM), exact typed-reference equality and original recording IDs. No paraphrase or publisher variation bucket is substituted. Missing human wordings remain untested, and publisher transcripts are not independently listened labels. [All 93 wording cells](../experiments/speaker-pipeline-repair-v1/optionb-spoken-wordings.csv) include zero-recording cells; the [analysis receipt](../experiments/speaker-pipeline-repair-v1/optionb-analysis-receipt.json) binds the normalization and predictions.

## Verification and limits

The 18 new selector/promotion/reference/regression checks were observed failing before their fixes, then passing. The reviewed focused suite passes 67 tests in 18.91 seconds. One independent fresh-context review found an Important nested-control-binding omission; it was reproduced RED and fixed GREEN. Capture-fault identity comparison was also reproduced and fixed; new exclusion receipts are bound and checked in integration. Root pytest still stops at two pre-existing collection errors: duplicate test_catalog modules and missing train.register_sources (44.98 seconds). [Review and resolutions](../experiments/speaker-pipeline-repair-v1/review.json) preserve the independent scope and empirical claims it declined to judge. No second review, commit, push, installation or physical-device action occurred.

The Pi archive is a comparison candidate with demo and qualified modes and actions disabled. The same runtime, wheels and wake/VAD assets are reused; no training code, PyTorch or training checkpoints ship. Archive identity, offline documentation links and extracted CPU demo/qualified parity are checked separately; these do not establish physical Pi performance. The target remains Linux ARM64 CPython 3.12 with glibc >=2.28; host inference uses Python 3.13.9 and ONNX Runtime 1.23.2, while the included target ORT wheel is 1.30.0. Cross-version ARM execution remains unmeasured.

Previously examined recordings provide a paired regression comparison, not untouched validation. One saved-checkpoint change plus renewed policy fitting cannot isolate their individual contributions; the foundation was unchanged and no continuation effect was measured. Missing original human classes, independent listening/label review, wake-owner/overlap validation and physical Pi/live microphone measurements remain unresolved. Synthetic coverage cannot close human qualification gates. Actions remain disabled.
