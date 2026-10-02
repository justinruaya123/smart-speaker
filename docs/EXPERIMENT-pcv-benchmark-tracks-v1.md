# PCV dataset benchmark tracks v1

Status: **primary results regrouped; supplied-test identity awaiting clarification**. The user's new benchmark grouping is by dataset membership, not a human-versus-synthetic split. This is a reporting/evaluation follow-up to the immutable `pcv-release-train-v1` result, addressing ME-03/04 and all command/slot requirements. No model fitting, calibration, acoustic inference, package replacement or installed-model change has occurred in this follow-up.

## Two headline tracks

| Track | Membership and boundary | Evaluation |
|---|---|---|
| Strict primary dataset | Only the pinned HF `airimonda/ai231-me2-voice-commands` snapshot, including both human and supplied-synthetic recordings | Preserve published training/test/holdout membership. Use one combined primary score, plus per-intent/slot/phrase and provenance diagnostics. Keep the separate holdout descriptive. |
| Primary dataset + supplied test | Primary dataset plus the exact additional test corpus identified by the user | Deduplicate recordings and retain scores for the primary portion and added portion, alongside any combined score. Identification is pending: the user's 415 baseline/custom-slot definitions versus the HF published test split. Definitions are not themselves recorded audio. |

The HF published test is already part of the strict primary evaluation. If that is the intended supplied test, it must not be counted twice. If the intent is adaptation using previously labelled test recordings, first reclassify the adaptation subset as training and freeze an independent evaluation subset; do not claim those adapted-on recordings as held-out evidence. Human and synthetic remain provenance diagnostics, not the headline track names.

The existing dataset-source restriction remains in force until the added corpus is identified. No new TTS is generated. The previous three-engine study has 415 Qwen recordings and 124 Dia recordings, but their existence alone does not establish that they are the intended supplied test or that they are clean against the latest HF training source.

## Current strict-primary result

Reaggregation of the existing frozen model gives 4,418 published test rows: 4,004 supported positives, 229 supported rejection references and 185 ambiguous references kept outside exact-command scoring.

| Metric | Combined primary result |
|---|---:|
| Raw complete commands correct / positives | 3,839/4,004 (95.88%) |
| Raw complete-command precision | 3,839/4,146 (92.60%) |
| Raw wrong commands on negative inputs | 160/229 (69.87%) |
| Diagnostic threshold-filtered exact / positives | 3,170/4,004 (79.17%) |
| Diagnostic threshold-filtered precision | 3,170/3,182 (99.62%) |
| Diagnostic threshold-filtered wrong commands on negatives | 1/229 (0.44%) |
| Actual accepted exact / positives | 0/4,004; existing policy remains disabled |

The separate published holdout has 178/183 raw positive commands correct, with 8/11 wrong commands on negative references and two ambiguous references excluded. It is not pooled into the primary test. A combined score is not evidence that the two origins perform equally or that the existing acceptance gates have passed. The model/policy remains unchanged; this report is not a fresh blind test or a new qualification decision.

## Overfitting checks

1. Audit sample IDs, exact file/decoded PCM duplicates, conflicting labels, speaker identity and synthesis voice/anchor overlap across fitting, development, calibration and each evaluation corpus. Unknown voice lineage remains unknown.
2. Compare primary and added-test performance with the same frozen model and thresholds, including per-intent/slot/phrase denominators. A combined average alone cannot establish generalization.
3. If adaptation is intended later, report adaptation-set versus held-out performance, primary-regression differences, and results on recordings from unseen speakers/voices. Repeated known command wording is expected in a finite VCM; label it separately from audio/speaker leakage and from unseen slot combinations.
4. Freeze any new adaptation split, model-selection rule and evaluation criteria before fitting or inspecting its held-out predictions. Preserve the existing observed test receipts; regrouping does not reset prior test exposure.

Evidence: [regrouped primary metrics and source binding](../experiments/pcv-benchmark-tracks-v1/primary-results.json), [original immutable model results](../experiments/pcv-release-train-v1/runs/release-v1/results.json), [source/split audit and delivery report](EXPERIMENT-pcv-release-train-v1.md).
