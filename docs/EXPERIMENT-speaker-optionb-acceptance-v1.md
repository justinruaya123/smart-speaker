# OptionB-only accepted-command investigation

Date: 2026-10-22. Study: `speaker-optionb-acceptance-v1`. Requirements: ME-01/02/03/04/08; the finite OptionB realization of UC-01 through UC-10.

## Decision

The low accepted recall/F1 is real, not an arithmetic error. Under the OptionB-only contract, the frozen shipping full model finds **79/80** human complete commands before confidence gates but correctly accepts **39/80**. Accepted precision is **100.00%**, recall **48.75%**, and complete-command micro-F1 **65.55%**. No wrong human command is accepted in this small observed population.

Restricting the finite catalog to **all 31 complete typed OptionB commands** does **not** increase correct accepts. The auxiliary intent head and its fixed acceptance checks remain the main measured bottleneck. Do not bypass those heads or call the ungated 98.75% positive recall deployable recognition quality: the restricted raw decoder still proposes commands on **11/14** task negatives.

This is a bounded diagnostic on an already examined holdout, not a fresh blind qualification, newly trained human-only model, or demonstrated training improvement. No threshold/model selection, training, normalization fit, source relabeling, installation or physical action occurred.

## Frozen scope and controls

- [Protocol](../experiments/speaker-optionb-acceptance-v1/protocol.json) and [pre-replay selection](../experiments/speaker-optionb-acceptance-v1/selection-before-replay.json).
- Primary graph: `submission/models/full/model.onnx`, SHA-256 `3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e`; 2,066,831 parameters. Existing INT8 shipping comparison, not a new export.
- Unchanged complete-utterance CPU inference, original PCM/frontend/normalization, two ONNX threads and one BLAS thread.
- Unchanged demo policy: intent probability at least **.95**, agreeing intent head, required slot probabilities at least **.90**, and the existing generic/named-light target check. Strict model qualification remains disabled; demo-policy accepts are not qualified deployment evidence.
- All **196 original recordings** replayed for each inference arm. All **94 eligible original-human parents** also received the parent's three predeclared one-second silence probes: **282 dependent artificial trials per arm**, not new recordings.
- The full-catalog control reproduces the parent original and stress decisions. Actual semantic-head winners/probabilities and target probabilities remain equal across catalogs; no semantic-head renormalization or intent rescue.
- **289 input identities** bound before replay and reverified after it, including all 78 parent protected inputs, model/policy/normalization, full-model reproduction data and all 196 original WAV identities. The parent-pinned source map remains unchanged; the new helper is an explicitly invoked evaluation-only test module, not a published runtime module.

### What "OptionB-only" means

Membership is exact canonical equality of the **complete typed command**, including required slots, not merely matching the family or finding a keyword. All 93 benchmark definitions and all 31 command combinations remain in scope. Paraphrases still count as positives when their complete reference command is in that set.

The original human population becomes:

| Reference group | Recordings | Treatment |
|---|---:|---|
| OptionB complete commands | 80 | Positive references |
| Original true UNKNOWN | 11 | Rejection references |
| Named-light extensions outside OptionB | 3 | Explicit additional task negatives |
| Ambiguous human references | 2 | Retained but unscored |
| Synthetic OptionB commands | 100 | Separate synthetic positive track |

The three out-of-scope requests concern kitchen/bedroom lights. OptionB's light targets are generic `light.all`. None of these records is moved, relabeled in its source manifest, or removed to improve a score; only the declared evaluation view changes. This explains the change from the earlier 83-positive denominator to 80. It is a task-scope change, not improved model weights.

### Catalog ablation

| Arm | Catalog / action rule | Human raw correct | Human correct accepted | Precision | Recall | F1 | False accepts |
|---|---|---:|---:|---:|---:|---:|---:|
| Full catalog, fixed policy | Original 1,079 entries | 79/80 | 39/80 | 100.00% | 48.75% | 65.55% | 0/14 |
| Full catalog + OptionB output scope | Reject-only allowlist after the unchanged policy | 79/80 | 39/80 | 100.00% | 48.75% | 65.55% | 0/14 |
| OptionB catalog, fixed policy | 669 entries: 390 inherited supported aliases + all 279 inherited negative competitors | 79/80 | 39/80 | 100.00% | 48.75% | 65.55% | 0/14 |

The one shared restricted catalog is frozen before inference and is never filtered by a recording's reference answer. It retains every existing alias of every permitted command and every null-command competitor; it adds no holdout-derived phrase or pronunciation. Four original raw predictions change across catalogs, but no accepted prediction changes. The accepted-command macro-F1 over the 17 observed positive families is **64.01%**; the missing family is reported, not given a fabricated score.

The restricted raw decoder alone has **86.81%** precision, **98.75%** recall and **92.40%** complete-command F1 across positives and negatives. Its raw negative proposals are **9/11 original UNKNOWNs** and **2/3 out-of-OptionB requests**. The fixed application policy rejects all 14. These raw values are diagnostics, not an alternative selected policy.

All three arms correctly accept **99/100 synthetic commands**: precision/recall/F1 are each **99.00%**, with one wrong positive command accepted. There are no synthetic negatives here, so synthetic false-accept rate is undefined. Do not pool this track with original human evidence.

## Why correct accepts, recall and F1 are low

"Correct accepted" is the exact complete-command true-positive count after every gate. Raw correctness is a different stage.

For the 80 human OptionB positives, the exhaustive disjoint accounting is:

| Outcome | Recordings |
|---|---:|
| Correct complete command accepted | 39 |
| Correct raw command, correct intent-head winner, but probability below .95 | 26 |
| Correct raw command, but intent head chooses another family / UNKNOWN | 13 |
| Correct raw command and sufficient intent confidence, but target gate fails | 1 |
| Incorrect raw command, rejected | 1 |

Thus **40 otherwise correct raw commands are rejected**, and all **41** positive misses remain false negatives. Accepted-command TP=39, FP=0, FN=41:

- Precision = TP / (TP + FP) = **100.00%**.
- Recall = TP / (TP + FN) = **48.75%**.
- F1 = 2TP / (2TP + FP + FN) = **65.55%**.

High precision does not compensate for missing more than half the valid requests. A wrong accepted parameter would count as both a false positive and a false negative, even with the correct intent; regression tests protect this accounting.

### First-rejection strings hide important differences

The application checks intent confidence before agreement. All 13 correct-candidate/wrong-head cases are therefore initially labeled `intent_confidence`, not `intent_disagreement`. Treating them as ordinary cutoff misses would be misleading. The actual intent head is correct on only **67/80** OptionB positives, compared with the finite decoder's **79/80** complete commands.

Independent gate flags also find one slot-confidence failure overlapping an earlier intent rejection. The disjoint table is not a claim that lowering the intent threshold would safely recover all 26 matching-head cases.

Concrete fixed-reference examples:

- `play a song`: raw `PLAY_MUSIC` is correct, but the actual intent head chooses `LIGHT_ON`; the candidate intent probability is about **0.00653**. Lowering a cutoff alone does not repair the head disagreement.
- `weather`: raw command and winning intent agree, but probability is about **0.68385**, below .95. `what's the weather` similarly has about **0.89513**.
- `change the lights to green`: raw color/target are correct and intent probability is about **0.99630**, but named-target probability is about **0.22006**, above the generic-target limit .10. The original target head still fails even when the restricted catalog's `light_id` slot score becomes 1.
- `put on the music`: the raw decoder chooses `PAUSE`; the actual intent head chooses `PLAY_MUSIC`. The application rejects rather than using the reference label to rescue the command.

These are frozen publisher/reference comparisons, not independent listening adjudications.

## Per-intent human holdout results

The full and restricted catalogs have the same original accepted results. N/A means no in-scope positive recording, not failure or success. Precision is 100% for each family with an accepted output; it is undefined for families with no outputs.

| Intent | Positives | Raw correct | Correct accepted | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| PLAY_MUSIC | 3 | 2 | 0 | 0.00% | 0.00% |
| WEATHER | 3 | 3 | 1 | 33.33% | 50.00% |
| TIME | 3 | 3 | 1 | 33.33% | 50.00% |
| LIGHT_ON | 2 | 2 | 2 | 100.00% | 100.00% |
| LIGHT_OFF | 0 | 0 | 0 | N/A | N/A |
| PAUSE | 3 | 3 | 3 | 100.00% | 100.00% |
| STOP | 3 | 3 | 1 | 33.33% | 50.00% |
| NEXT | 3 | 3 | 3 | 100.00% | 100.00% |
| VOLUME | 4 | 4 | 2 | 50.00% | 66.67% |
| CALL | 3 | 3 | 2 | 66.67% | 80.00% |
| MESSAGE | 3 | 3 | 1 | 33.33% | 50.00% |
| LIST_REMINDERS | 3 | 3 | 3 | 100.00% | 100.00% |
| TIMER | 9 | 9 | 2 | 22.22% | 36.36% |
| ALARM | 8 | 8 | 2 | 25.00% | 40.00% |
| TEMPERATURE | 9 | 9 | 7 | 77.78% | 87.50% |
| BRIGHTNESS | 6 | 6 | 5 | 83.33% | 90.91% |
| COLOR | 6 | 6 | 3 | 50.00% | 66.67% |
| CREATE_REMINDER | 9 | 9 | 1 | 11.11% | 20.00% |

Only **30/31 typed commands** and **62/93 exact normalized verbatims** have original-human holdout coverage. The missing typed command is generic `LIGHT_OFF` (`light.all`); the two original LIGHT_OFF positives were named-target extensions. Those recordings cannot establish OptionB generic-light-off recall. No training/development recording is borrowed to fill this cell.

Among the 62 observed exact verbatims, raw correctness is **62/62** but correct acceptance is **30/62**. Literal keyword or exact-phrase coverage is therefore not the principal demonstrated limitation on these recordings. All unobserved cells remain explicit in the all-93-verbatim export.

## Padding/context robustness

Same 80 in-scope positive parents; all 14 negative parents retained. No clipping or trimming.

| Waveform condition | Full raw correct | OptionB-catalog raw correct | Correct accepted, either catalog | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Original | 79/80 | 79/80 | 39/80 | 48.75% | 65.55% |
| Leading 1s silence | 77/80 | 78/80 | 14/80 | 17.50% | 29.79% |
| Trailing 1s silence | 78/80 | 78/80 | 34/80 | 42.50% | 59.65% |
| Both ends, 1s each | 77/80 | 78/80 | 16/80 | 20.00% | 33.33% |

All observed accepted precisions are 100% and task-negative false accepts are 0/14 in these derived trials. These are paired artificial probes, not independent evidence or actual spoken-unrelated-word tests. Catalog restriction does not repair their acceptance loss. In the restricted leading-silence trial, 27 correct raw commands have wrong intent-head winners and 36 have matching heads below .95; the corresponding original counts were 13 and 26.

## Dataset, supervision and segregation checks

[Fit reference audit](../experiments/speaker-optionb-acceptance-v1/fit-reference-audit.json) reads the **actual selected full model's frozen reproduction manifest**, whose digest matches the full recipe, instead of assuming the original GPU cache exists on this workstation. The original `.cache/speaker-synthetic-completion-v1/prepared-v1` cache is absent; no replacement fitting input was fabricated.

- Historical selected-model fitting view: **23,295 rows**, including **1,869 human** and **21,426 synthetic**. This is not the newer 1,889-human research-recovery view and does not mean those recovered examples have trained this model.
- Train/development/calibration/test/holdout ID separation is verified. Human train versus each reserved split has zero observed ID, declared-speaker, source-audio SHA or PCM SHA overlap. Physical speaker aliases/session independence remain unverified.
- All **1,869 normalization contributors** belong to the original train split, with no reserved-view ID overlap. The shipped normalization is preserved, not refitted.
- All **23,295 fitting** and **1,251 development** rows have present semantic/target indices consistent with their complete references. The graph recipe's semantic order matches the runtime contract. No class-index swap or coupled-target error is demonstrated in this actual full-model fitting view.
- Calibration/test/holdout reproduction rows have **no supplied training-head indices**. The preserved initial audit naively counted those absent values as discrepancies; the [field-availability companion check](../experiments/speaker-optionb-acceptance-v1/reference-field-check.json) explicitly distinguishes absence from an actual present-label disagreement. These are not newly proven mislabeled recordings. Actual prediction-head metrics always use the diagnostic predicted head, never these reference-training fields.
- Existing code uses a human foundation followed by class/command/speaker-balanced mixed sampling with half the mixed batch human. Raw corpus ratios do not by themselves prove that synthetic examples dominate every batch. The .3 intent and .15 target loss weights are inspected recipe facts, not demonstrated optimal choices.
- Human fitting evidence remains sparse for CALL (10 recordings), CREATE_REMINDER (11) and absent for MESSAGE. This is a descriptive coverage limitation, not a causal ablation. Existing independent-listening and disputed-reference gaps remain open; internal index consistency does not verify spoken truth.

No fitting manifest, reference, split membership, graph, confidence policy or normalization file was edited. The new catalog exists only as a versioned research file and an in-memory replay alternative. It is not installed into the shipping bundle.

## Bounded next experiment

The evidence supports prioritizing the semantic heads and context invariance, not adding a keyword-only acceptor:

1. Keep all 31 complete OptionB commands and required values/targets. Retain unknown, negation, extra-action, wrong-value and out-of-scope contrasts rather than deleting competition to raise acceptance.
2. Use admitted training records for intent/target supervision and retain masks for missing lexical targets. Investigate the head/finite-candidate disagreement and human source/wording coverage on development data; no holdout example, transcript-derived alias or failure-specific rule enters fitting.
3. If changing acceptance cutoffs, select on declared OptionB development/calibration only, preserving intent agreement and slot/target guards, then lock before another evaluation. A lower cutoff alone cannot fix the 13 wrong intent-head winners and could expose raw false accepts.
4. Test matched context-position/preprocessing training alternatives on development, preserving whole commands and labels. Do not replace inference normalization or remove speech based on these holdout errors. Actual connected-human unrelated-word robustness needs separate recordings; silence/text fixtures do not establish it.
5. Use a new untouched speaker/session population for blind improvement claims. This exposed corpus can continue to serve as regression evidence.

The exact upstream cause of the head generalization/position weakness is not isolated by this catalog-only ablation. No controlled semantic-head retraining, attention-pooling ablation, quantization comparison or new calibration is claimed here.

## Evidence and verification

- [Results](../experiments/speaker-optionb-acceptance-v1/results.json): separate original-human/synthetic and dependent stress metrics for all three reporting arms.
- [Per-intent CSV](../experiments/speaker-optionb-acceptance-v1/per-intent.csv): 108 strata; [per-command CSV](../experiments/speaker-optionb-acceptance-v1/per-command.csv): 186 strata; [all-verbatim CSV](../experiments/speaker-optionb-acceptance-v1/per-verbatim.csv): 558 strata. These are reporting rows, not new recordings.
- Raw original/stress predictions and gate profiles: `.cache/speaker-optionb-acceptance-v1/replay-01/`; its receipt verifies original/stress baseline parity, unchanged semantic heads and all protected identities.
- Canonical tested helper: `scripts/test/speaker_optionb_acceptance.py`; regression tests: `scripts/test/test_speaker_optionb_acceptance.py`. Fourteen new cases cover complete-command scope, retained negatives/aliases, omitted benchmark commands, evaluation-only guards, wrong-slot FP/FN, undefined metrics, masked gate failures and decision-replay tampering.
- Broadened focused suite: **122 passed**. This is not a full-repository test/type-check or Pi-quality pass. The workstation language server does not resolve the root research environment consistently; no clean full type-check is claimed.
- [Independent verifier](../experiments/speaker-optionb-acceptance-v1/verify-results.py) rederives TP/FP/FN/precision/recall/F1, reference scope, every per-intent/command/verbatim table and gate profile, checks parent original/stress decisions and protected inputs, and writes the [final receipt](../experiments/speaker-optionb-acceptance-v1/verification.json).

Executed entry points from the project root:

```text
.venv/Scripts/python.exe experiments/speaker-optionb-acceptance-v1/replay-driver.py freeze
.venv/Scripts/python.exe experiments/speaker-optionb-acceptance-v1/replay-driver.py replay
.venv/Scripts/python.exe experiments/speaker-optionb-acceptance-v1/reference-field-check.py
.venv/Scripts/python.exe experiments/speaker-optionb-acceptance-v1/verify-results.py --write-receipt
```

Creation is exclusive so existing selections/runs/results are never overwritten. The verifier can be rerun without `--write-receipt`. No commit, push, model installation, live microphone, network recognition or physical device operation occurred.
