# OptionB holdout, labels and data loading — 2026-10-22

## Scope and decision

Study `speaker-optionb-holdout-v1` addresses ME-01/02/03/04/08 and the existing finite UC-01 through UC-10 contracts. User authorization: test holdout, investigate targeted OptionB cues with minor wording/context resilience, and prioritize datasets, pretraining, preprocessing, loading, segregation and labels.

The experiment freezes the existing full submission and verified v7 compact comparison before inference. This is an already-examined published-holdout regression, not a fresh blind evaluation or a newly trained human-only result. No weights, thresholds, runtime catalog, publisher inputs or split memberships change. No synthetic training injection, device action, live microphone, installation, commit or push occurs.

**Main finding: the full model's raw complete command is usually right, but human acceptance and padding invariance are weak.** Tight cue labels are useful for data organization and contrast collection; a bag-of-keywords acceptor would discard negation/context and cannot be justified by these results. Keep the full extension catalog and every required typed parameter.

## Frozen experiment and original holdout

[Selection before holdout](../experiments/speaker-optionb-holdout-v1/selection-before-holdout.json) binds 78 input identities, both models, policy settings, the 93-case cue plan and three silence probes. Both primary views retain intent .95 and required-slot/target .90 demo thresholds. The compact relaxed view uses its existing bound policy, with no new fitting.

- Full: `submission/models/full`, model SHA256 `3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e`, 2,066,831 parameters.
- Compact comparison: `experiments/speaker-targeted-v7`, SHA256 `e8a2c5aca69b03eacf1092a68b6b23a541c3a7f51fdc5a34a52214dc9ebe0017`, 935,567 parameters. This is not the currently identity-blocked compact submission bundle.
- Source: primary revision `25111444af3adff7588d86ab27c304bb080895cf`, all 196 frozen holdout records.
- Separate populations: 83 human positives, 11 true human negatives, two ambiguous human references and 100 synthetic positives.
- Only two declared speakers supply eligible human positives. Alias/session independence and independent listening are not established.

Every original WAV and decoded PCM is rehashed before inference. No expected command or wording is passed to the recognizer. The complete catalog competes normally. Strict qualified outputs remain zero; tables below are explicitly demo-policy diagnostics.

| Model / policy | Raw exact / 83 | Correct accepted / 83 | Accepted precision | Recall | F1 | UNKNOWN false accepts / 11 |
|---|---:|---:|---:|---:|---:|---:|
| Full fixed | 81 | 39 | 100.00% | 46.99% | 63.93% | 0 |
| Compact v7 fixed | 79 | 17 | 94.44% | 20.48% | 33.66% | 1 |
| Compact v7 existing relaxed | 79 | 41 | 97.62% | 49.40% | 65.60% | 1 |

Both fixed models correctly accept 99/100 synthetic positives, with one wrong positive accept each. There are no synthetic negatives here, so synthetic FAR is undefined, not zero. Human/synthetic distributions are not matched source ablations.

The full model rejects 42 raw-correct human commands. Its raw candidate is nevertheless non-null on 10/11 true human negatives: bypassing confidence gates or accepting any detected keyword would be unsafe. Zero observed accepted-negative errors has a recording-level Wilson 95% upper bound of 0.258833, not proof of a 1% FAR. Speaker-cluster intervals remain conditional on this tiny observed speaker pool.

The full 39/83 result agrees with the previously recorded Windows benchmark; it is not a new improvement over historical 38/83 on another execution context.

### Per-intent human holdout results

| Intent | Positive n | Full raw correct | Full correct accepted | Full accepted with 1s leading silence | Compact fixed accepted | Compact existing relaxed accepted |
|---|---:|---:|---:|---:|---:|---:|
| PLAY_MUSIC | 3 | 2 | 0 | 0 | 0 | 0 |
| WEATHER | 3 | 3 | 1 | 0 | 0 | 2 |
| TIME | 3 | 3 | 1 | 0 | 0 | 2 |
| LIGHT_ON | 3 | 2 | 2 | 2 | 0 | 1 |
| LIGHT_OFF | 2 | 2 | 0 | 0 | 0 | 0 |
| PAUSE | 3 | 3 | 3 | 2 | 3 | 3 |
| STOP | 3 | 3 | 1 | 0 | 0 | 1 |
| NEXT | 3 | 3 | 3 | 0 | 1 | 3 |
| VOLUME | 4 | 4 | 2 | 1 | 1 | 2 |
| CALL | 3 | 3 | 2 | 1 | 0 | 1 |
| MESSAGE | 3 | 3 | 1 | 0 | 0 | 1 |
| LIST_REMINDERS | 3 | 3 | 3 | 0 | 1 | 2 |
| TIMER | 9 | 9 | 2 | 0 | 2 | 3 |
| ALARM | 8 | 8 | 2 | 0 | 2 | 5 |
| TEMPERATURE | 9 | 9 | 7 | 5 | 3 | 5 |
| BRIGHTNESS | 6 | 6 | 5 | 1 | 3 | 5 |
| COLOR | 6 | 6 | 3 | 2 | 0 | 2 |
| CREATE_REMINDER | 9 | 9 | 1 | 0 | 1 | 3 |

These are small, dependent strata, not per-intent qualification. [Full results and confusion/slot diagnostics](../experiments/speaker-optionb-holdout-v1/holdout-results.json) and [54 positive-family metric rows](../experiments/speaker-optionb-holdout-v1/holdout-per-intent.csv) preserve exact counts, precision, recall, F1 and uncertainty.

## Padding and minor-wording evidence

Silence probes were declared before replay: 16,000 zero-valued samples before, after, or at both ends, with no trimming or clipping. Each model runs 282 derived trials from 94 eligible human parents. Derived trials are marked `constructed_silence_stress`, not synthetic/human original recordings; stale feature paths/hashes are cleared. Parent identity, derived PCM identity and padding offsets are retained. These trials are dependent and cannot enlarge the independent human sample size.

| Input | Full raw / 83 | Full correct accepted / 83 | Full accepted UNKNOWN / 11 | Compact fixed raw / 83 | Compact fixed correct accepted / 83 |
|---|---:|---:|---:|---:|---:|
| Original | 81 | 39 | 0 | 79 | 17 |
| +1s prefix silence | 79 | 14 | 0 | 70 | 5 |
| +1s suffix silence | 80 | 34 | 0 | 77 | 12 |
| +1s both ends | 79 | 16 | 0 | 68 | 4 |

The full prefix probe loses 26 formerly correct accepts and gains one. All 26 losses report `intent_confidence`. The full both-ends probe loses 24/gains one; suffix loses five/gains zero, likewise through intent confidence. This isolates a padding effect under fixed artifacts, not its unique training/preprocessing cause.

These are complete-utterance file tests, not a wake/VAD microphone test. The actual capture path may remove some leading context. Do not fix this by blindly removing one second or clipping quiet onset/negation/units.

Natural human holdout wording groups, defined from supplied metadata and typed references:

| Group | Positive n | Raw correct | Correct accepted |
|---|---:|---:|---:|
| Exact normalized OptionB wording | 62 | 62 | 30 |
| Conservative courtesy/function-word variant | 1 | 1 | 0 |
| Other wording or extension | 20 | 18 | 9 |

This does not establish broad minor-wording resilience: the courtesy group has one example, and other-wording comparisons are not matched speaker/command ablations. No spoken unrelated-word injection test was manufactured. Word-injection acoustic resilience remains unmeasured.

## Labels: actual frozen policy versus obsolete rules

A first metadata comparison used the older `pcv_updated_prepare.annotate` base rules. It reports 258 differences from stored labels. **Those are not 258 newly discovered labeling errors:** the base rules omit the release's approved aliases. First-pass evidence is preserved as a versioned negative diagnostic.

The authoritative follow-up verifies `annotation-approval.json` against its original data-freeze SHA256, then passes that immutable policy explicitly to the existing release annotation API. Across 3,416 loaded references (admitted human views plus the complete mixed holdout), it finds:

- **Zero semantic-reference disagreements** against the actual frozen release policy.
- 288 training and 55 development transcripts are ASR-derived and not independently reviewed.
- Three dev/two calibration publisher slot/family conflicts remain flagged.
- No label changes, no new admissions and no independent listening claims.

The two excluded holdout references have supplied transcripts “Decrease sound levels” and “Turn volume down”. They are plausible paraphrase-review candidates, not verified heard labels. They remain excluded and must never be moved into training or used to inflate the locked score.

A separate reproduction problem is real: the live proposal file hashes to `70d40fe5...`, while the default policy builder pins `e1c0fba7...`; calling it fails `Frozen identity mismatch`. CRLF/BOM do not explain the mismatch. The projected supported maps (110 entries) and ambiguous maps nevertheless equal the archived policy, so semantic mapping drift is not demonstrated. No checksum guard or source file is changed. [Verified frozen-policy recheck](../experiments/speaker-optionb-holdout-v1/frozen-label-recheck.json) and [source rebuild diagnostic](../experiments/speaker-optionb-holdout-v1/annotation-source-rebuild-check.json) separate these facts.

## Dataset, loader and preprocessing improvements

### Per-verbatim data and cue contracts

The offline plan contains every one of the 93 OptionB phrases, typed slots, candidate cue tokens and benign/negation/quotation contrasts. It does not restrict runtime competition to 93 phrases, rescue rejected commands or use a transcript parser at inference. All current extension commands remain preserved.

| Human split | Exact normalized metadata wordings / 93 | Publisher variation buckets / 93 |
|---|---:|---:|
| Train | 31 | 59 |
| Dev | 16 | 44 |
| Calibration | 11 | 34 |
| Test metadata only | 71 | 85 |
| Holdout | 62 | 80 |

Publisher buckets and supplied transcripts are not independently verified verbatims. Cue-compatible, exact and typed-only evidence remain separate. The 930-row wording export includes empty/unprovided strata explicitly; synthetic training/test views were not loaded for this wording audit and must not be reported as empty corpora.

The cue-rule fixture checks pass 279/279 benign text forms and 186/186 negation/quotation contrasts. These are offline rule tests, not acoustic trials. The [93-case human collection plan](../experiments/speaker-optionb-holdout-v1/human-keyword-collection-plan.json) creates zero recordings and no invented keyword-phone targets.

### Verified research feature loading

`scripts/test/speaker_optionb_holdout.py:125` implements the new fail-closed audit loader. The actual run loads all 1,889 admitted human training records: 1,863 CTC-supervised, 26 semantic-only and 373 target-supervised, while checking 1,427 reserved human rows for split overlap. It verifies:

- Explicit original-human train eligibility and no ambiguous/auxiliary records.
- Unique IDs/PCM and declared speaker/hash/pair/available-group split separation.
- Exact normalization contributor membership, without refitting on dev/test/holdout.
- Exact feature paths and SHA256, finite float32 40-bin arrays and CTC feasibility.
- Coupled intent/target indices, strict integer types, valid head masks and unavailable-target masking.

It does not replace or rerun historical trainers. A read-only legacy-path audit reconfirms 118 target inversions: LIGHT_ON 32, LIGHT_OFF 11, BRIGHTNESS 6, COLOR 69. The corrected human pool does not contain those inversions. A constructed fixture also shows the old targeted loader can substitute a different same-basename file, and its pool accepts a contrived dev row if its eligibility flag is flipped. These demonstrate guard weaknesses, not observed leakage or wrong-file loading in the frozen full model. The new loader rejects these cases.

### Batch padding is not actual waveform silence

The model already masks length-dependent batch padding. A real full-architecture random-initialization software probe compares a short example alone versus in a padded batch: maximum valid-output difference is below 1e-6, without a model change.

Actual zero-valued PCM is different: with the frozen full-model frontend it produces mean normalized features -2.931221 (range -3.311410 to -2.332311), not normalized zero. It also shifts absolute positions. Frozen full-model augmentation uses normalized-zero SpecAugment, not waveform prefix/suffix jitter/noise/reverb. This is a measured mismatch between mechanisms and a candidate explanation to test, not a proven single cause. Existing inference normalization remains unchanged.

## Next bounded data/pretraining experiment

1. Preserve joint command labels: intent plus operation/value/unit/meridiem/target/task. Cues may be auxiliary supervision, never permission to collapse 20/60/100, on/off, AM/PM or task/contact names.
2. Use the verified human view and exact normalization membership; retain semantic-only masks and genuine UNKNOWNs. Finish independent listening for ASR-derived/conflicting examples rather than trusting prompts as audio truth.
3. Compare train-only, label-preserving context/position variation against the unchanged foundation recipe on development. Natural connected human wording changes are preferred; any artificial augmentation is a separately labeled arm. Do not invent cue phone sequences or unverified cue spans.
4. Include semantic near-misses: negation, quotation, changed values/targets, unsupported names, extra actions and unrelated numbers. Arbitrary unrelated words are not universally harmless padding.
5. Keep architecture, catalog and policy fixed for a one-factor data/sampling/preprocessing comparison. Select only on development/calibration, then use a new untouched speaker/session-disjoint evaluation for a generalization claim. The inspected holdout is now diagnostic/regression evidence.

No full new fit or recognition improvement is claimed by this task. Lowering thresholds on this holdout, replacing the model with a keyword-only acceptor or dropping UNKNOWN safeguards is not an accepted remedy.

## Verification and reproduction evidence

Executed driver snapshots live under `experiments/speaker-optionb-holdout-v1/`: `replay-driver.py`, `data-audit-driver-v1.py`, `frozen-label-recheck-driver.py`, `format-results-driver.py`. They require the pinned local cache inputs and exclusive new run paths; they preserve existing runs rather than overwrite them. Raw original/stress predictions and PCM/audit provenance remain under `.cache/speaker-optionb-holdout-v1/`.

**108 focused tests pass**, including new cue, wording, label-policy and safe-loader cases. Compileall and git diff --check pass. These are software checks, not a repository-wide or release-quality pass. The final verification receipt reconciles output hashes/counts, per-intent totals, derived-parent identities, original versus constructed provenance and protected inputs. Earlier base-rule implementation snapshots remain available for their frozen hashes; the authoritative label recheck uses the explicit immutable approval.
