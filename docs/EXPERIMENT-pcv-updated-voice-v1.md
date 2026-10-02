# PCV updated AI231 voice dataset v1

Current continuation: the completed [final qualification report](EXPERIMENT-pcv-final-qualification-v1.md) selects a larger endpoint-context INT8 research candidate after the user authorized an offline pretrained teacher. It adds the already approved `call mother` alias, uses no acoustic cache states, and improves held-out results, but still fails release gates. The residual architecture and 967-entry catalog below describe this original frozen v1 experiment and remain historical evidence.

## Scope and execution plan

The user's updated primary dataset is [airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands), initially pinned to revision `e3e203b360d5c3205d59f99ad65cd380f9c0f517`. The accompanying [publisher audit](https://claude.ai/artifact/PPNHMWd5rx9qcXTdcukV7s) is a reference to check against the actual files, not independent human verification. This study addresses ME-01/02/03/04/06/07/08 and all ten use-case categories, while preserving the schema-v3 contract and installed bundle.

The user explicitly selected this sequence: (1) human recordings first, with properly separated human test recordings; (2) synthetic recordings already supplied in this Hugging Face dataset. The later source-only instruction narrows fallback permission: other recordings may be used only when no suitable recording in this dataset matches the required training input. A development failure alone no longer authorizes our own augmentation. No older generated corpus, external recording, inherited neural checkpoint or distillation teacher is included in this version.

Hypothesis: training primarily on the supplied command recordings improves the prescribed baseline and local-speaker coverage over the previous public-corpus model. Measure the incremental value of supplied synthetic recordings over a matched human-only baseline on the same untouched human test.

1. Acquire the exact publisher revision with per-file hashes. Cache audio-bearing Parquet files under `.cache/pcv-updated-voice-v1/`; retain compact source receipts, exclusions and protocols under `experiments/pcv-updated-voice-v1/`. Existing ignore rules cover payloads and expose reproducibility metadata. No repository maintenance, environment changes or installation.
2. Audit every row: source and synthetic flags, speaker/voice identity, source licenses, actual audio format/readability/duration, exact file and decoded-PCM duplication, paired recordings, transcript provenance, command/slot mapping and exact baseline coverage. Check utterance boundaries and unusable audio without splicing phrases. Compare published counts with the audit and README. Phrase-class membership is not proof that the exact phrase was spoken. Publisher Whisper checks are automated metadata; this task does not run ASR.
3. Preserve official test and holdout membership. Derive speaker-disjoint development and calibration subsets only from training speakers; keep the numeral pool as separately identified acoustic support, never as complete-command success. Exclude conflicting/ambiguous labels from full-command supervision rather than inventing slots. Start new candidates from random weights so previously used public-source evaluation recordings cannot leak through an inherited checkpoint.
4. Freeze typed labels, catalog, splits, model/optimizer schedule, matched update budgets, development selection rule, rejection gates and source hashes before fitting. Train the human foundation first, then compare an equal-budget human-only continuation against a continuation using the source's existing synthetic training recordings. Human validation/test membership remains unchanged. Report human, Filipino-group, other-source and synthetic strata separately.
5. Select using fitting/development data, lock weights and calibration before test, then evaluate complete typed commands, all 93 exact baseline cells, OOV/unknown-name behavior, per-intent/slot results and missing coverage. Preserve the separate Pi holdout; do not use it to tune. Verify ONNX, endpoint-buffered frontend/state parity, size and desktop timings without claiming Pi or microphone evidence.
6. If development assessment after the source-synthetic stage shows remaining gaps, first search the supplied dataset for usable matching recordings and audit their labels/quality. External fallback requires a documented absence of suitable source examples and a separate experiment version. Distillation, if investigated later, must preserve the same source and split restrictions; it cannot establish the true label of an unintelligible clip. Retain earlier baselines and do not tune on test failures.

## Initial findings

Revision metadata declares train 10,682, test 4,371, holdout 186 and numerals 66,390 rows. README prose still lists 10,603 train examples; the linked audit contains older 4,418-test and 196-holdout counts alongside newer figures. Actual pinned files determine authoritative denominators.

The publisher balances semantic phrase classes using substantial pre-existing synthetic speech. Its definition permits different wording with the same command and value. This study independently distinguishes exact wording, semantic command coverage, original-human evidence and synthetic evidence. All 93 phrase classes being represented does not establish all 93 exact spoken phrases in human recordings.

## 2026-10-02 acquisition and preparation

All ten pinned Parquet files (3,025,410,306 bytes) match their published LFS SHA-256 hashes. All 81,629 waveforms decode as mono 16 kHz PCM16. The independently computed audit found zero exact file/PCM duplicates and zero declared speaker overlaps across published splits. Paired-file identifiers and speakers also have no overlap after the training-only development/calibration partition.

Published counts are 2,877 human + 7,805 synthetic training, 1,003 human + 3,368 synthetic test, 84 human + 102 synthetic holdout, and 66,390 human isolated numerals. Every published test and holdout recording retains its original membership.

Preparation retains 1,215 supported human training commands, 537 unsupported human requests, 48,831 human isolated numerals and 5,973 supplied synthetic training commands. Numerals have masked semantic loss and a bounded batch allocation, so their much larger count does not dominate command learning. Development and calibration use only original training/numeral speakers. The two main local training speakers remain in fitting; the separately declared secondary speaker is development-only. No test speaker enters fitting, normalization, catalog construction, weight selection or calibration.

The frozen annotation audit records 813 ambiguous references, 83 missing-pronunciation exclusions and 448 waveform-quality exclusions across all source splits. These counts are reasons, not evidence that every excluded utterance is inaudible. Test retains 207 ambiguous human references, separately reported rather than scored as known OOV. Its exact-command evaluation has 609 supported and 187 unsupported references. Eleven clear slot-metadata conflicts are recorded, including compound durations such as one hour plus one minute, whose correct target is 3,660 seconds rather than 60. Offline finite annotation never supplies a runtime transcript.

Exact human baseline wording is present for only 32/93 cells in training, 16/93 in development, 13/93 in calibration, 71/93 in test and 67/93 in holdout. These are metadata-derived exact normalized wording counts, not listening verification. Supplied synthetic partitions each cover 93/93. Human fitting has no positive empty-slot MESSAGE examples; human calibration lacks CALL, MESSAGE, CREATE_REMINDER, LIST_REMINDERS and TEMPERATURE. The experiment must disclose these gaps even if synthetic baseline scores are strong.

Publisher utterance boundaries are preserved. Waveform checks cover duration, level, clipping and edge silence; no phrase splicing, cropping or independent human segmentation/listening claim is made. The 967-entry finite catalog uses predeclared schema definitions and eligible fitting text only. Original waveforms remain inside immutable Parquet files; derived features are cached by PCM hash.

Evidence: [acquisition](../experiments/pcv-updated-voice-v1/acquisition-complete.json), [waveform audit](../experiments/pcv-updated-voice-v1/audio-audit.json), [annotation/coverage audit](../experiments/pcv-updated-voice-v1/data-audit.json), [exclusions](../experiments/pcv-updated-voice-v1/annotation-exclusions.json), [data freeze](../experiments/pcv-updated-voice-v1/data-freeze.json).

Training starts from random weights with the existing 366,653-parameter residual VCM, 40 log-mel channels, a finite candidate scorer, intent head and explicit-light-target head. The declared budget is 32 foundation epochs and 16 epochs for each matched continuation (96 updates/epoch, effective batch 32, microbatch 8). Human batches allocate 24 positive commands, four negatives and four numerals; supplied-synthetic continuation replaces eight positive-human places with eight HF synthetic examples. Both continuations share the selected foundation and update budget. Development selects candidates every four epochs. No fitting/test success is claimed until the corresponding receipts are complete; installation remains unauthorized.

## Final architecture contract

The selected architecture for this experiment is endpoint-buffered, direct finite command recognition. Wake/VAD collects one complete utterance; the frozen frontend computes 40-channel features once; the residual ONNX graph scores acoustic cues, intent and explicit light targeting; the finite command scorer chooses among compiled typed candidates; the frozen confidence policy and schema validator either emit a command or reject. No unconstrained text decoder, transcript parser, LLM, cloud recognition or device-action executor is in this path. Training annotations and pronunciation compilation are offline preparation only. Recognition of CALL/MESSAGE never places a call or sends a message.

| Intent | Typed slots and defaults |
| --- | --- |
| PLAY_MUSIC | Empty for generic playback; optional `song_id`, with optional `artist_id` only when a song is supplied |
| WEATHER, TIME, PAUSE, STOP, NEXT, MESSAGE, LIST_REMINDERS | Empty slots; MESSAGE means request recognition only |
| LIGHT_ON, LIGHT_OFF | `light_id`; omitted name means `light.all`; an unresolved explicit name never falls back to all lights |
| VOLUME | `operation` in up/down/mute/set; set additionally requires integer `percent` 0–100 |
| CALL | `contact_id`; omission defaults to `contact.home`; Mom/mother resolves separately to `contact.mom` |
| TIMER | Integer `duration_seconds` 1–86,400; compound durations retain every term |
| ALARM | `time` in HH:MM and the fixed benchmark `timezone` Asia/Manila |
| TEMPERATURE | Integer `degrees` −50–60 and `unit` C; unmarked 70-degree recordings are not silently converted from Fahrenheit |
| BRIGHTNESS | `light_id` and integer `percent` 0–100; the named-light forms retain the named target |
| COLOR | `light_id` and a supported finite color |
| CREATE_REMINDER | `task_id` from the configured finite task list; no free-form dictation |

The output envelope is schema version 3 with `accepted`, `command` and `reason`. Successful commands contain exactly `intent` and `slots`; rejection uses null command and OOV, invalid_argument, capture_fault or unqualified_model. Uncertain intent becomes OOV; confidently recognized intent with unresolved/invalid arguments becomes invalid_argument. A failed qualification policy stays disabled and is reported as such; all-reject is never a baseline pass.

The custom-slot JSON contains stable light/song/artist/contact/task IDs and textual names/aliases. Catalog compilation is deterministic for definitions; spoken recognition remains statistical. New names need pronunciation coverage and new competition/unknown-name calibration. Adding a string does not manufacture a matching recording or establish a high confidence score. The inherited mother contact alias appears in the 415-case reporting contract; this candidate catalog does not add a new mother phrase unless it occurs among eligible fitting definitions. This absent acoustic cell must remain visible.

Model swapping is atomic over graph, eleven cache-state shapes, 40-bin frontend/normalization, label order, catalog, custom slots, schema and policy hashes. A policy calibrated for another graph or catalog is rejected. The candidate is not interchangeable with the older six-state dummy adapter. Keep the prior bundle as the recovery point; graph parity, endpoint replay and measured qualification precede any later installation. This experiment creates candidate bundles only.

The entire 415-case requested reporting contract remains in scope, but only recordings from the pinned HF source may supply its rows in this experiment. Missing custom-name or percentage cells have a zero denominator and cannot pass. The older Qwen/Dia/Parler queue remains stopped for this task; its completed recordings are retained as separate historical evidence. Dia was local, and no remote TTS service is involved. That separate three-engine benchmark is incomplete and is not relabeled as a completed HF-source evaluation.
