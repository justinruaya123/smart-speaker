# Mixed real/synthetic comparison

Two independent acoustic VCM experts start from random initialization. The synthetic expert fits only `tts` training audio; the human expert fits only original human training recordings. Both have the same fixed intent and parameter vocabulary. Development within the respective source chooses checkpoints.

The mixed system combines their probabilities, using a separate `real_weight` for every intent. Zero chooses synthetic evidence, one chooses human evidence, and intermediate values blend them. A per-intent log-odds bias adjusts the family prior. The winning VCM family is selected before its finite parameters are considered. That family's source weight also blends readiness, operation, amount and color probabilities. A parameter cannot select a different intent.

This is a source-specific ensemble. It is not described as a single model trained on a concatenated dataset. The human-only expert is an additional baseline inside this directory; the main requested comparison is synthetic versus mixed.

Optuna's seeded TPE search tunes source preference, confidence bias, intent threshold, slot threshold and margin using human development only. It rewards correct accepted commands subject to at least 95% precision, at most 1% false acceptance on negative development speech and at least ten accepted examples for the emitted family. Two coordinate passes use 100 trials per intent each: 1,800 trials per system.

After all competing family choices are set, the complete system is checked again. Unsupported families are disabled. Brightness and volume SET, INCREASE and DECREASE each receive an additional development support/precision check, so a successful relative-command score cannot hide failed absolute numbers. This refines the existing precision/support gate; it does not use final test errors.

The selection receipt locks models, manifest hashes and every threshold before any final test inference. Final evaluation reports original-human and held-out synthetic-voice tests separately, along with each intent, absolute/relative number strata and each value's observed support. It does not test new names or a newly expanded corpus. Counts are this study's filtered subsets, not complete publisher benchmark results.

The unchanged original human test speakers are excluded from training. The prior supplemental SmartLights numeric holdout is also preserved. Root calibration recordings are excluded. Audio hashes and speaker identity are disjoint across train/dev/test, paired microphones are deduplicated, and conflicting duplicate labels are quarantined.

Human labels remain source annotations plus the existing strict grammar, hence silver rather than adjudicated ground truth. The human test contains 73 absolute brightness commands over 11 values and two absolute volume commands over two values. The TTS manifest spans all integers 0-100, but that cannot prove complete human acoustic coverage.

Fixed output classes remain fixed. Learning a new spoken synonym requires new acoustic evidence or a validated phonetic encoder, not merely an alias in JSON. A future enrollment bank can bind new examples to existing intent/parameter IDs, followed by development recalibration. No new-word acoustic generalization is claimed in this phase.

The human-only control is calibrated with the same development precision/support targets and a fixed real_weight of one. It supplies a pure-human comparison. The final global false-accept check considers 21 shared threshold offsets and rechecks support after pruning.

Post-scoring limitation: the coordinate search changes competing normalized family scores and does not guarantee preserving the strongest pure-source policy. The mixed system retained only 12 correct development accepts, versus 146 for the human-only control. Consequently, this run does not isolate the causal value of generated data. Its failed result is preserved. A later protocol should keep pure-source incumbent policies and compare a jointly fitted mixed model as well; neither modification was made after this test.
