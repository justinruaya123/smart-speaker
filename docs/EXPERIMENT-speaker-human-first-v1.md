# Human-first, per-intent data audit and repair v1

Date: October 22, 2026. Status: **audit and preparation implemented; full human-only training and accuracy improvement are not established**.

Experiment: `speaker-human-first-v1`. Requirements: ME-01/02/03/04/08 and UC-01 through UC-10 within the existing finite ontology. Actual repository: `smart-speaker-fresh`, branch `codex/final-submission`, baseline `babf00329abd43cf81dea3ba4d117cae8e5c2f6a`, with pre-existing untracked experiments/prototype preserved. The surrounding `me2` worktree is a different Git boundary; it is not the model repository's revision.

## Deliverables

- `scripts/train/speaker_human_data.py`: executable, research-only human admission/audit; immutable split/intent reference views; waveform/PCM/feature integrity and CTC feasibility; full-identity publisher metadata join; hierarchical human sampler; dependent semantic-target reconstruction; train-only normalization refit; honest human metrics; explicit readiness gate; one-step CPU optimizer smoke.
- `scripts/test/test_speaker_human_data.py`: regression coverage for the new path, including unsafe inputs, ambiguity, missing classes, joint command coverage, pinned metadata joins, immutable outputs and real gradient updates.
- `experiments/speaker-human-first-v1/protocol.json`: scope, pinned baseline, proposed comparison and evidence boundaries.
- `experiments/speaker-human-first-v1/intent-fixes.json`: a separate targeted treatment for each of the 18 finite intents and UNKNOWN, plus shared segmentation/preprocessing/pretraining/split rules.
- `experiments/speaker-human-first-v1/results.json`: measured counts, per-intent and per-slot coverage, exact-wording coverage, source provenance, development replay and endpoint diagnostics.
- `experiments/speaker-human-first-v1/collection-queue.json`: **55 baseline command/split cells** below proposed pilot speaker floors, and **887 full-catalog command/split cells** below those floors. These are two overlapping request tiers, not counts of new recordings or distinct people.

Mutable original-audio references, 95 split/intent views, exclusions, per-recording acoustic diagnostics, refitted normalization and optimizer receipts are under `.cache/speaker-human-first-v1/`. No recording is duplicated merely to segregate intents.

## Most consequential new bug: target-label inversion

The existing `speaker_targeted_train.apply_corrections_and_filter` sets generic `light.all` to target index **1**, and named lights to **0**. Runtime semantics are the opposite: **0 = generic; 1 = named**. Applying this legacy correction stage to the admitted human train view changes **118 otherwise consistent target labels**:

| Intent | Inverted target labels |
| --- | ---: |
| LIGHT_ON | 32 |
| LIGHT_OFF | 11 |
| BRIGHTNESS | 6 |
| COLOR | 69 |

A further row's expected command changes from UNKNOWN to LIST_REMINDERS. That semantic change is retained as a review finding, not automatically adopted.

**Implemented fix in the new version:** do not replay the legacy proposal loader. Rebuild intent and target indices from the existing typed command using the deployed index convention. The safe sampler reconstructs these indices without mutating its input. It preserves the parent expected labels and phone sequences; a future transcript change must rebuild sequence supervision too. This avoids re-inverting already-correct prepared labels. The old trainer, original manifests and previously fitted models remain unchanged as historical evidence.

This is an observed supervision inconsistency, not a measured estimate of its accuracy cost. A fresh controlled fit is required to attribute a gain to repairing it.

## What was actually audited

Five command manifests contain 29,364 rows, including 3,475 original-human rows. The additional normalization-only records in the original portable recipe are not counted as independent command-split rows.

| View | Admitted human rows | Interpretation |
| --- | ---: | --- |
| Train | 1,863 | 1,412 positive commands and 451 true-UNKNOWN references |
| Development | 256 | 193 positives and 63 true UNKNOWN |
| Calibration | 210 | 122 positives and 88 true UNKNOWN |
| Test | 865 | 636 positives and 229 true UNKNOWN; 185 ambiguous references retained separately |
| Holdout | 94 | 83 positives and 11 true UNKNOWN; 2 ambiguous references retained separately |

The audit checked original WAV bytes, decoded PCM identities and cached feature hashes/shapes/finiteness for **all 2,119 admitted human train/development recordings**. It checked repeated-label CTC frame feasibility. None failed those integrity/feasibility checks. Calibration, test and holdout **audio was not opened** by this work; their membership/label/source metadata was inspected only.

The cached inputs are the existing frozen-hash feature replay. Checking those images is not a new claim of source-only, bitwise Windows/Linux frontend reproducibility. A future preprocessing comparison must bind the numerical stack/frontend settings and verify its own training/runtime parity; the prior reproduction investigation remains separate.

All four primary Parquet shards were hash-bound before metadata joins. Audio and secondary ASR-transcript columns were not read during those joins. Missing original-human recordings were not replaced with fabricated examples.

### Coverage is much smaller than family counts suggest

| Coverage view | Human train | Human development | Human calibration |
| --- | ---: | ---: | ---: |
| Baseline typed commands | 25/31 | 22/31 | 15/31 |
| Complete finite-catalog commands | 55/309 | 34/309 | 23/309 |
| Exact baseline metadata wordings | 31/93 | 16/93 | Not inspected for wording selection |

Exact wording is a normalized publisher-transcript match, not independently verified speech. Typed-command, wording, marginal-slot and joint-slot/target coverage are different measurements. A model cannot claim human coverage of every percentage, room, task or song because its text catalog contains those values.

## Per-intent gaps and targeted fixes

Counts are original-human rows; training speakers are declared identities, not independently voice-verified people. Dev/calibration columns count positive references, except UNKNOWN.

| Intent | Train rows / speakers | Dev / calibration rows | Observed gap and targeted fix |
| --- | ---: | ---: | --- |
| PLAY_MUSIC | 42 / 11 | 8 / 4 | Human train covers only generic playback, 1/5 catalog commands; dev and calibration each have one positive speaker. Separate generic/title/title+artist pools; collect original connected utterances for approved songs; reject unknown names instead of silently falling back to generic playback. |
| WEATHER | 47 / 19 | 10 / 2 | Calibration has one speaker; unchanged v7 development replay accepts only 5/10 correctly. Expand source/microphone/paraphrase coverage and time/thermostat/query negatives. Do not lower thresholds using weather test confidences. |
| TIME | 98 / 39 | 16 / 11 | Only 11/16 development positives are correctly accepted. Balance short and connected queries; distinguish timers/alarms and unsupported date/location qualifiers. Preserve short utterances and initial consonants. |
| LIGHT_ON | 78 / 26 | 7 / 17 | Named-room cells remain small, and 32 target labels are inverted by the legacy correction stage. Rebuild generic/named indices; sample target/source/speaker cells; retain on/off polarity, room names and negation. |
| LIGHT_OFF | 92 / 34 | 11 / 14 | Kitchen and bathroom each have only two training speakers; 11 legacy target inversions. Repair target supervision, fill named-room cells, and contrast light-off with music-stop. |
| PAUSE | 89 / 46 | 16 / 9 | Development raw recognition is 16/16 but correct accepts are 14/16. Keep single-word commands and pause/stop semantics; audit short-command capture and calibrate only on human development/calibration. |
| STOP | 135 / 54 | 20 / 13 | Development has 18/20 raw correct and 14/20 correct accepts. Separate playback-stop from cancel-timer, unrelated stop requests and negation; source/speaker balance prevents abundant clips masking other intents. |
| NEXT | 89 / 31 | 13 / 8 | Development has 12/13 raw correct and 9/13 correct accepts. Contrast Next/Skip/Play next song with generic play; protect final consonant clusters and the next cue. |
| VOLUME | 238 / 61 | 27 / 21 | Only 7 absolute-set training utterances, no mute positives, 7/104 catalog commands covered. Sample operation then value; collect complete human numeric commands and fifteen/fifty-type contrasts; keep absolute-to distinct from relative-by. |
| CALL | 10 / 2 | 0 / 0 | Home has nine training rows from one speaker; Mom has one row from one speaker. Fill independent fitting/dev/calibration speakers; distinguish contacts and incomplete/unknown names. Do not infer a contact from a family prediction. |
| MESSAGE | 0 / 0 | 0 / 0 | No eligible empty-slot human positives. There are 36 MESSAGE-labelled coarse-source rows currently treated as unsupported in train; recipient/body-bearing speech must not automatically become an empty-slot command. Review original metadata/audio, then collect supported human requests if needed. Missing positives remain a blocker. |
| LIST_REMINDERS | 37 / 12 | 5 / 0 | No human calibration positives; one legacy proposal would resurrect a currently negative row. Review that row rather than silently overriding it; collect list/show/read variants and contrast create/list/delete semantics. |
| TIMER | 54 / 14 | 11 / 2 | No ten-second train positive; one thirty-second training row; calibration has only two one-speaker recordings of a different duration. Fill connected baseline duration/unit cells, adjudicate disputed slot metadata, and preserve number/unit pauses. |
| ALARM | 117 / 25 | 9 / 7 | Eight-AM is absent from dev; nine-PM absent from calibration. Fill clock-cell/speaker coverage and meridiem contrasts; preserve quiet AM/PM endings. Do not infer missing meridiem or add unsupported recurrence/times. |
| TEMPERATURE | 72 / 3 | 20 / 0 | Only three train speakers and two dev speakers; no calibration positives. Add independent human speakers/sources and calibration; audit numeral/unit/bounds and distinguish thermostat settings from weather queries. |
| BRIGHTNESS | 40 / 9 | 1 / 4 | Train percentages are 20, 50 and 80; all generic-light commands. Baseline 60/100 are absent; only one dev positive; six legacy target inversions. Fill connected percentage/target cells, repair target supervision and expand independent dev/calibration evidence. |
| COLOR | 163 / 35 | 17 / 10 | Only red/blue/green, 10/48 joint catalog cells; 69 legacy target inversions. Development accepts only 1/17 correctly and eight rows fail target confidence. Correct target supervision; balance color×room; preserve ambiguous activation-versus-color proposals for review; collect remaining approved colors/targets. |
| CREATE_REMINDER | 11 / 1 | 2 / 0 | Only task.study and one training speaker; drink-water/exercise absent; no calibration. Review existing unsupported task metadata, then fill all approved task/speaker cells; distinguish create/list and retain the full task phrase. Arbitrary dictated reminders remain unsupported. |
| UNKNOWN | 451 / 107 | 63 / 88 | Must not contain ambiguous or unintelligible references merely because expected is null. Stratify verified unsupported speech by family/rejection/source/speaker; preserve negation, unknown targets and appended unsupported content; keep capture faults separate. |

Each row has a detailed collection, annotation-review, hard-contrast, segmentation and auxiliary/pretraining treatment in `intent-fixes.json`. Floors of five training, three development and three calibration speakers per typed command are **proposed pilot collection floors**, not evidence of adequate release accuracy. New blind speakers/sessions and predeclared uncertainty-aware gates are still required.

## Additional cross-cutting gaps

### 1. Previous negative and test-selection accounting is not trustworthy as blind evidence

The legacy targeted evaluator treats every human `expected=None` row as a negative. This combines **229 true test UNKNOWNs with 185 ambiguous references**, producing the previously reported 414 denominator. The new partitioner/metrics keep those groups separate. The old false-accept numerator cannot simply be divided by 229; stored decisions must be rescored against the proper reference groups.

The earlier session also adjusted WEATHER/ALARM/TIMER thresholds against test outcomes. The current policy still carries `test_used=false` and older fitting-view summaries. Those flags do not undo the actual test exposure. Existing test/holdout results are regression evidence, not a fresh independent human validation. No historical result or policy file was rewritten here.

The existing trainer reads the test manifest in its fitting entry point, injects synthetic coverage selected for test signatures, and awards a +1 mixed-phase selection bonus. The new human path does none of these. Its proposed model selection uses human complete-command macro-intent/macro-speaker recall with precision/FAR constraints; a mixed-stage label is not a reason to prefer a checkpoint.

### 2. Segmentation needs actual endpoint evidence, not blanket silence trimming

The energy screen flags 198/2,119 train/dev clips with an internal low-energy run of at least 400 ms. It flags 931 active starts and 923 active ends. These are review flags, not verified truncation or corrupted recordings; some sources are already tightly segmented.

A separate replay of **all 256 human development clips** used the actual Silero models, the deployed .5 command-start threshold/four-frame preroll and the actual .35/400 ms `EndpointCommand`. Every clip produced a speech start and an endpoint, but **57/256 had later high-probability VAD activity after the first endpoint**, including 44 positive references and 13 UNKNOWNs. That can indicate premature endpointing, background/extra speech or VAD error; independent listening and boundary labels are needed to distinguish them. It is not proof that all 57 intended commands were truncated. Wake detection and live microphones were not tested.

Implemented preparation retains original PCM and produces only a padded outer-boundary proposal. It never splices internal pauses or silently removes short commands. A future segmentation candidate must use reviewed command/slot boundaries, a versioned feature cache and matched deployment behavior; compare whole and endpointed commands on development before freezing it.

### 3. Human audio is not equivalent to human-verified labels

Training transcript origins are 1,425 publisher-dataset labels, 288 ASR-derived (`whisper`), 78 folder phrases and 72 prompts. None was independently listened to by this audit. Folder names and intended prompts can be wrong; an ASR transcript can miss accents, negation or numbers. Preserve these as separate provenance strata, prioritize disputed labels and obtain independent adjudication instead of calling the entire set pristine.

A further **547 human rows in the original training shards are absent from the admitted command views**. This is an existing review/recovery pool, not 547 automatically valid extra examples. Preserve source split/speaker exclusions, source terms and exact row/audio identities before recovering anything. Some absent rows are deliberately uncertain or unsupported; some belong to development/calibration populations. No row was automatically moved into train.

### 4. Speaker-disjoint metadata does not prove unseen-person/session generalization

No cross-split collisions were found for available declared speakers, publisher source-speaker keys, paired identities, file hashes or PCM hashes. No duplicate PCM rows were found within admitted human train. However, all 2,119 inspected train/dev rows lack explicit session IDs, parent-recording IDs and independently adjudicated speaker-alias groups. Physical-person aliasing, alternate takes/channels, near-duplicates, room/microphone balance and session leakage remain unverified. Repeated wordings are expected for this finite task and are not themselves leakage.

Use connected components of verified speaker aliases/parents/pairs when freezing a future split; do not row-random-split or move already-examined evaluation speakers into train. Group-aware split methods can enforce non-overlapping declared groups, but cannot discover missing physical identities.[4]

### 5. Normalization membership is inconsistent, but its effect is not proven

The installed normalization used 1,869 human rows; this compact training view admits 1,863. The new smoke refits on exactly those 1,863. Maximum absolute differences are 0.009555 in bin means and 0.004923 in standard deviations. This is a membership/version mismatch, **not proof that normalization caused the human accuracy gap**.

Keep train-only global normalization as the baseline. A per-utterance CMVN alternative is a separately versioned matched train/dev/deployment ablation, not an inference-only repair to existing weights. Neither method is universally superior. Fitted transforms must exclude evaluation data and be applied consistently.[1]

### 6. Pretraining/auxiliary and augmentation boundaries

The new smoke starts from random weights and uses only original human recordings; no pretrained teacher, inherited mixed-data checkpoint, synthetic examples, feature masking, gain/noise perturbation or word concatenation is used. Existing v7/v10 mixed/synthetic-exposed weights are not a human-only foundation.

The isolated numeral/word bank remains auxiliary. Before any human-only pretraining experiment, verify that each candidate is original human audio, exclude evaluation speakers/aliases/parents/hashes, and use trustworthy role-specific loss masks. Isolated numeral recognition is not connected percentage/duration evidence. Do not force an uncertain transcript into a CTC target or count isolated tokens as complete-command coverage. Sequence lengths and padding masks must follow the actual model stride; the CTC objective constrains valid input/target lengths.[5]

Regional-pronunciation/lexicon mismatch is a hypothesis to audit with original speech and reviewed variants, not a demonstrated cause here. No new acoustic classes or runtime transcriber is introduced.

An existing untracked `speaker_human_pipeline.py` prototype was preserved and excluded from the new path. It contains numeric-suffix pruning/reclassification, imports absent legacy helper names, reads `intent_idx`/`target_idx` rather than the prepared `intent_index`/`target_index`, and treats spoken negative features as ambient-noise material. These are not a verified human-data repair, and its stated counts are not adopted.

### 7. Current compact submission artifact fails its identity guard

`submission/models/compact` has graph/catalog/normalization hashes matching its recipe, but its `policy.json` binding disagrees with the recipe. `CompactRuntime` raises **Bundle identity mismatch**. The v7 research directory verifies and was used for the development replay, with its exact model/policy hashes retained in the machine evidence. The submission bundle was not repaired or rebound: copying a stale policy binding would not establish that it was calibrated for the current weights.

## Verified development diagnostic, not improvement evidence

Using the verified unchanged v7 research bundle and its unchanged demo thresholds:

- Human positives: 193; true UNKNOWNs: 63.
- Raw complete-command correct: 158/193.
- Correct accepted commands: 112/193; wrong positive accepts: 13.
- UNKNOWN false accepts: 1/63.
- Accepted precision: 112/126 = 88.89%; macro-speaker accepted recall: 46.92%.
- CALL and MESSAGE have zero development positives; their recall is undefined, not zero-evidence success.

These are diagnostic development results on unreviewed publisher references, not an untouched test and not a comparison proving the repair improves recognition. Particularly small denominators—BRIGHTNESS 1/1, CREATE_REMINDER 1/2—must not be presented as robust human accuracy. Per-intent raw/correct-accept/wrong-accept/rejection breakdowns and the fixed .95/.90 comparison are retained in `results.json`.

## Execution and verification

Actual preparation command, from the model repository, using the existing root environment:

```bash
.venv/Scripts/python.exe -m scripts.train.speaker_human_data \
  --prepared .cache/submission/reproductions/frozen-replay-compact-01/.cache/reproduction/prepared \
  --metadata-root .cache/frontend-repro-20261019/source-view/.cache/pcv-source-refresh-v1/source \
  --output .cache/speaker-human-first-v1/a-new-run --smoke-fit
```

Existing output paths are immutable; choose a new run path. For a fail-closed preparation gate, replace `--smoke-fit` with `--require-ready`. The actual `readiness-01` execution returned **exit 2**, correctly refusing full-human-readiness claims for missing coverage. The ordinary audit and explicitly labelled smoke completed with exit 0.

The real CPU plumbing smoke instantiated the unchanged **935,567-parameter** architecture, refit normalization on 1,863 admitted human train rows and performed **one optimizer update** on a human-only batch. Losses were finite (CTC 23.127934; intent 2.763643; target 0.820538) and weights changed. No checkpoint was installed/exported and no accuracy improvement is claimed.

Focused verification: **79 tests passed** across the new human-data tests and existing targeted/final training, compact-policy, human data preparation, CPU-device and reproduction checks. These include all four light-intent target conventions and a full-catalog readiness guard that cannot ignore extension or UNKNOWN gaps. An initial broad unittest attempt encountered three missing-`pytest` imports; installing `pytest==8.4.2` into the existing active root `.venv` enabled the proper pytest run. Only its missing test-runner dependencies were added; no project manifest/lockfile or model dependency was changed. The entire repository test suite was not claimed as passing.

Protected baseline graph, policies, recipe, catalog, normalization, source trainers and installed metadata were rehashed unchanged during preparation. Final verification receipts bind the new source/tests/protocol and generated metadata. No commit, push, deployment, device action, dataset move, original-audio mutation, new synthetic injection or full training run was performed.

## Decision and next experiment

1. Use the safe human-only path, not the legacy target-inverting correction loader or unverified pruning prototype.
2. Review/recover eligible original-human material from existing sources, prioritizing the 547 absent rows and weak label-provenance strata; never manufacture labels or change old test denominators silently.
3. Fill the baseline request tier first: MESSAGE, CALL, missing reminder tasks/calibration, brightness 60/100, timer duration/unit cells, temperature speakers/calibration and named light/color targets. Collect new blind speakers/sessions separately.
4. Freeze adjudicated labels, split components, frontend/normalization, catalog, loss masks and selection criteria. Then run the proposed matched human-only comparison, retaining full-command precision/recall and per-intent/source/speaker failures. The target-index bug needs a controlled repair ablation, not an inferred accuracy claim.
5. Only after that human baseline is evaluated should a separate synthetic-injection experiment be considered. Preserve its untouched human evaluation and measure gains and regressions per intent; more acceptances alone are not success.

## Sources

[1] https://scikit-learn.org/stable/common_pitfalls.html
[4] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
[5] https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
