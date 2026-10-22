# Original-human recovery and head-masked supervision

Date: October 22, 2026. Experiment: `speaker-human-recovery-v1`. Requirements: ME-01/02/03/04/08; finite UC-01 through UC-10.

## Outcome and scope

**Recovered 26 original recordings for intent-head supervision; kept 521 unresolved recordings outside training. Full acoustic comparison remains blocked. No recognition-accuracy improvement is established.**

The research training view grows from 1,863 to **1,889** recordings. Four additions are PAUSE positives; 22 are metadata-supported finite-scope negatives. All 26 have CTC supervision explicitly disabled: an unavailable pronunciation is not an assertion that the recording contains an empty transcript. These are existing publisher/dataset labels, **not independently listened ground truth**. No synthetic speech, word splicing, acoustic augmentation, pretrained checkpoint, model installation, threshold change, device action or test-prediction replay was used.

The unchanged 935,567-parameter architecture completed **one** finite CPU optimizer update with four intent-supervised examples and two CTC-supervised examples. This validates the new loss path, not model quality. The full comparison was not started because human class/calibration coverage and disputed-label adjudication remain incomplete. The readiness invocation exited **2**, as required by the frozen protocol.

## Evidence and reproducibility

- [Frozen protocol and 45 input/implementation bindings](../experiments/speaker-human-recovery-v1/protocol.json).
- [Measured results and all-19-label counts](../experiments/speaker-human-recovery-v1/results.json).
- [Per-intent treatments, inherited collection/contrasts, and new targeted follow-ups](../experiments/speaker-human-recovery-v1/intent-fixes.json).
- [Reference-only review queue: 521 rows](../experiments/speaker-human-recovery-v1/review-queue.jsonl).
- [Blank training-review templates: 428 rows](../experiments/speaker-human-recovery-v1/review-template.jsonl).
- [Pre-existing prototype compatibility findings](../experiments/speaker-human-recovery-v1/compatibility-findings.json).
- [Previous human-first audit and collection priorities](EXPERIMENT-speaker-human-first-v1.md).

Baseline: dirty branch `codex/final-submission`, commit `babf00329abd43cf81dea3ba4d117cae8e5c2f6a`. The tracked-diff digest and hashes of otherwise untracked implementation files are recorded in the protocol. Source payload: primary corpus revision `25111444af3adff7588d86ab27c304bb080895cf`, verified against the parent audit's complete repository/revision/shard/row identities. No shortened numeric-ID matching is used.

Run output: `.cache/speaker-human-recovery-v1/prepared-01/`. The separate `readiness-01/` invocation confirms the failing readiness gate without an optimizer update. The current environment is the project-root Python 3.13 environment with PyTorch `2.10.0+cpu`; CUDA is unavailable in that interpreter. No accelerator package was installed for this task. Missing human evidence, not GPU availability, blocks the quality comparison.

## What the omitted pool actually contains

All **547** omitted original-human training-shard rows were accounted for:

| Original exclusion / omission | Rows | Treatment |
|---|---:|---|
| Ambiguous reference | 504 | Keep untrained pending adjudication; never convert ambiguity to UNKNOWN |
| Missing pronunciation | 26 | Recover existing supported/unsupported semantic labels only after whole-audio, split and duplicate checks; no fabricated phones |
| Six disputed timer references | 6 | Keep untrained; complete transcript and slot metadata disagree |
| Missing-speaker silence recordings | 11 | Keep calibration-reserved and metadata-only; do not invent speaker identities or move them into training |

Original assignments are **454 train, 40 development and 53 calibration**. Recovery admitted only train rows. The unresolved queue therefore contains **428 train, 40 development and 53 calibration** recordings. An apparent absence from a fitted manifest is not permission to reassign its speaker or call the row a negative.

The four recovered PAUSE recordings are “pause the audiobook,” from two declared speakers. They do not establish two new independent evaluation speakers or a newly validated decoder phrase. The 22 recovered negative labels originate from seven coarse PLAY_MUSIC, two LIGHT_ON, six CALL and seven OUT_OF_SCOPE recordings. These already have unsupported finite references: **they are not new named-music, light-target or contact positives**. They represent 16 declared speaker IDs, not an alias-resolved independence claim.

Among the remaining queue, **153** rows have ASR-derived transcript provenance. Ninety training recordings with publisher unintelligibility flags remain unresolved, rather than becoming 90 extra true negatives. The six Timers and Such conflicts retain both the one-hour-plus-one-minute transcript hypothesis (3,660 seconds) and one-minute metadata; neither is declared correct without listening.

## Per-intent state and targeted next action

Train counts include metadata-supported semantic-only additions. Pending-review counts below are **training rows only**, grouped by publisher family; they are not promised recoverable positives. The machine-readable treatments also preserve each intent's collection, contrast, segmentation and auxiliary/pretraining rules.

| Intent | Human train before / now | Dev / cal | Pending train review | Targeted treatment |
|---|---:|---:|---:|---|
| PLAY_MUSIC | 42 / 42 | 8 / 4 | 1 | Use seven recovered unknown-name/artist requests only for rejection; collect approved title and title-plus-artist speech |
| WEATHER | 47 / 47 | 10 / 2 | 79 | Review actual query semantics and wording; diversify calibration speakers rather than tune against test confidences |
| TIME | 98 / 98 | 16 / 11 | 8 | Preserve short queries and extra location/date qualifiers; separate time, timer and alarm |
| LIGHT_ON | 78 / 78 | 7 / 17 | 12 | Retain generic=0/named=1 targets; review named/unknown-target semantics without invented aliases |
| LIGHT_OFF | 92 / 92 | 11 / 14 | 15 | Fill scarce kitchen/bathroom speaker cells; preserve polarity and full target wording |
| PAUSE | 89 / 93 | 16 / 9 | 38 | Four semantic-only audiobook positives recovered; do not guess pronunciations or merge pause with stop |
| STOP | 135 / 135 | 20 / 13 | 0 | No dormant STOP pool found; collect negation, conversation and pause/stop contrasts, preserving short clips |
| NEXT | 89 / 89 | 13 / 8 | 13 | Review next/skip/play-next cues and preserve final consonants; do not collapse to generic playback |
| VOLUME | 238 / 238 | 27 / 21 | 36 | Separate up/down/mute/set and connected numbers; missing mute/absolute-value evidence remains missing |
| CALL | 10 / 10 | 0 / 0 | 4 | Six recovered unsupported-contact requests teach rejection only; independent Home/Mom dev/cal positives are still absent |
| MESSAGE | 0 / 0 | 0 / 0 | 0 | No omitted candidate exists here; collect genuine empty-slot requests without deleting recipients or message bodies |
| LIST_REMINDERS | 37 / 37 | 5 / 0 | 14 | Review listing versus creation/unsupported scopes; collect original-human calibration positives |
| TIMER | 54 / 54 | 11 / 2 | 12 | Adjudicate six duration conflicts; preserve connected numbers, units and internal pauses |
| ALARM | 117 / 117 | 9 / 7 | 20 | Adjudicate meridiem/time wording; preserve quiet AM/PM endings and fill missing clock cells |
| TEMPERATURE | 72 / 72 | 20 / 0 | 2 | Two incomplete ASR-derived cases do not add verified commands or speakers; diversify Celsius speakers and calibration |
| BRIGHTNESS | 40 / 40 | 1 / 4 | 22 | Review dim/deem/green-like transcript uncertainty without restoring a verb from its prompt; collect connected percent × target combinations |
| COLOR | 163 / 163 | 17 / 10 | 56 | Resolve activation-versus-color ambiguity; preserve two-word colors and target identity, then fill joint cells |
| CREATE_REMINDER | 11 / 11 | 2 / 0 | 6 | Preserve finite task semantics and appended content; collect independent task and calibration speech |
| UNKNOWN | 451 / 473 | 63 / 88 | 90 | Recover 22 existing finite-scope negatives with no CTC; keep unintelligibility, capture faults and silence provenance separate |

No new baseline typed positive cell, MESSAGE example, or independent development/calibration class was filled. The [existing collection queue](../experiments/speaker-human-first-v1/collection-queue.json) still governs missing baseline and broader catalog cells. Repeated sampling is not additional human coverage.

## Implemented pipeline changes

1. **Admission and segregation.** The new preparation validates pinned shard hashes, exact historical exclusion identity/status, original-human provenance, unchanged speaker assignment, whole-recording audio quality and duplicate/reserved identities. Existing frozen references and exclusions participate in cross-split checks. Nontraining views are copied byte-for-byte, not relabelled or moved.
2. **Head-specific loss masks.** The 26 lexical recoveries use `ctc=false, intent=true, target=false`; their phone arrays remain empty. Full-supervision rows retain their actual integer phone IDs and dependent semantic targets. Each head's loss is normalized by its own labelled-example count, avoiding dilution merely because another head lacks annotations. Software tests verify microbatch equivalence and reject unsafe evaluation rows before weight changes. This is a deliberate missing-label treatment, not PyTorch's meaningful zero-length-target case; CTC alignment constraints remain checked rather than hidden with zeroed infinite loss.[4]
3. **Preprocessing and boundaries.** All 454 omitted training recordings were decoded and energy-screened without changing their audio. There were no decode failures. Thirty have internal-pause endpoint-risk flags; 280/267 have activity at the outer start/end under the energy heuristic. These are review signals, **not proof of speech truncation or verified VAD boundaries**. No trimming, internal-pause splicing or automatic slot removal occurs. New feature images use the unchanged 40-bin frontend and disclosed one-thread BLAS extraction, with payload/PCM/feature hashes. Existing frozen feature images are not numerically rewritten.
4. **Normalization.** The optimizer plumbing run fits equal-utterance-weight statistics on exactly the 1,889 admitted training recordings. It neither uses evaluation examples nor replaces any deployed checkpoint's normalization. Train-only fitting and matched transforms avoid leakage from preprocessing.[2]
5. **Adjudication.** Reviews must identify distinct listeners, attest independent full listening, bind the complete pinned source and audio SHA-256, and agree on transcript, typed decision and complete boundaries. Duplicate reviewers, mismatched audio or inconsistent verdicts fail validation. Disagreement and ambiguity remain unresolved. Even consensus produces only a semantic **proposal**, never an automatically rewritten training set. No real independent reviews were supplied or completed in this run.
6. **Pretraining and splits.** The smoke starts from random weights. Synthetic-exposed historical checkpoints are not renamed human-only. Isolated numerals remain outside connected-command coverage and require a separately audited auxiliary experiment. Published test/holdout remain previously exposed regression evidence; a fresh speaker/session holdout is still needed. Group-separated validation is necessary when recordings depend on shared speakers/sessions.[3]

Implementation: `scripts/train/speaker_human_recovery.py`; tests: `scripts/test/test_speaker_human_recovery.py`. The source-map entry explicitly has `runtime=false`.

## Additional compatibility gaps found

The pre-existing `speaker_human_pipeline.py` was preserved, not silently repaired or run as a full trainer. Read-only probes established three separate problems:

- Its import of `load_corrections` from the current targeted trainer fails.
- Its Dataset adapter filters already-integer phone IDs through a string-keyed mapping, yielding empty phone targets in the isolated software fixture.
- Its `in_channels`/class-count constructor call is incompatible with the imported `PooledContextVCM` constructor.

The isolated Dataset class was loaded via AST because the complete module could not import; this is not a successful execution of its training loop. The current compact recipe's vocabulary keys **were present**, so that particular check did not identify a gap. The executed recovery path uses the real model API and preserves actual integer targets instead.

The legacy release annotation-policy loader also failed its frozen checksum guard, and the project-local CMUdict file is absent. Neither guard nor dictionary was patched around. Recovery uses the original finite rules plus exact historical exclusion/status evidence, not an assertion that the legacy release policy was replayed successfully.

## Verification

- **94 focused tests passed** across eight relevant modules; the new recovery module contributes 15 test cases. This is not a full-repository test pass.
- `compileall` and `git diff --check` passed.
- All 547 omissions, all 521 unresolved references, all 26 admitted additions, and all 19 intent/UNKNOWN treatment entries were counted and reconciled programmatically.
- New declared split-overlap findings: none. Physical-speaker aliases, near-duplicates and missing session/parent metadata remain unverified.
- All four nontraining human-view hashes match the parent exactly. Protected model, policy, recipe, source and implementation bindings remain unchanged.
- Actual masked optimizer losses were finite, weights changed, and normalization bound 1,889 training IDs. No full fit, quality selection, exported candidate or accuracy gain is claimed.
- The complete-human readiness command exited 2; it did not trigger a full comparison or bypass missing classes.

## Reproduce preparation and independent review

From the project root, use the existing `.venv/Scripts/python.exe`. Output directories must be new; the commands do not overwrite prior runs.

```bash
.venv/Scripts/python.exe -m scripts.train.speaker_human_recovery prepare --audit .cache/speaker-human-first-v1/audit-02 --source-root .cache/frontend-repro-20261019/source-view/.cache/pcv-source-refresh-v1/source --output .cache/speaker-human-recovery-v1/NEW-RUN --smoke-fit
```

For the qualification guard, use a separate new output path and replace `--smoke-fit` with `--require-ready`; the expected current exit code is 2. The command offers no full-fit bypass.

The 428-row review template is **blank**, not a set of completed annotations. Reviewers should produce separate JSONL records only for clips they actually listened to, explicitly completing reviewer ID, independent/listened attestations, full transcript, boundary status, decision and typed command. The 40 dev/53 calibration omissions stay metadata-only in this packet and must not be moved into training.

```bash
.venv/Scripts/python.exe -m scripts.train.speaker_human_recovery adjudicate --queue .cache/speaker-human-recovery-v1/prepared-01/review-queue.jsonl --reviews PATH-TO-COMPLETED-REVIEWS.jsonl --output .cache/speaker-human-recovery-v1/NEW-ADJUDICATION
```

This writes adjudication outcomes, not training data. Incomplete/disputed review exits 2. A subsequent admitted-data version must repeat original-audio, split, duplicate, provenance and coverage checks. The next meaningful step is actual listener adjudication plus the missing human command/calibration recordings; synthetic augmentation remains deferred.

## Sources

[2] https://scikit-learn.org/stable/common_pitfalls.html
[3] https://scikit-learn.org/stable/modules/cross_validation.html
[4] https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.md
