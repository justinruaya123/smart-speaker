# VCM-only removal and Option B transfer audit

## Decision and scope

The independent character-transcription component was actually removed in commit
`8607d1519228ae6d1d6f307956acbea3da121c48`, after real ONNX fixture and packaging
checks. The branch is `research/vcm-final-suite-visualizer`; main was not changed.
The one-time data-acquisition workflows have now been removed, and the migration
workflow is replaced by read-only regression CI. It cannot auto-commit changes.

The root v12 VCM family encoder, finite heads and frontend are unchanged. The
separate `phone_s53` research CNN continues to recognize constrained phonemes for
number/catalog decoding. It is not renamed general-purpose transcription.
No external pretrained weights were downloaded. Neither path executes actions.

**The adapted research checkpoint is not promoted.** It improved raw external
brightness matching from 0/180 to 69/180 but regressed the retained V6 test from
645/706 to 628/706, losing 26 correct requests and gaining nine. The default
remains `phone_s53`, with automatic acceptance disabled.

## Removed component

Executable notebook source no longer contains `tinyvcm.asr`,
`tinyvcm.asr_runtime`, transcription-session loading, `transcribe`,
`transcribe_features`, `fit_asr`, transcription validation, a character decoder,
or a custom-media transcript fallback. Old runtime/training/test switches fail
explicitly. Unresolved custom media returns `catalog_decoder_required`.

The publisher's model allowlist is exactly `bundle.json` and `vcm.fp32.onnx`.
A test plants a stale local `asr.fp32.onnx` and verifies it is not packaged.
Previous metadata declared that excluded graph as 5,319,980 bytes. Old ZIPs must
be rebuilt; ignored workstation files are not deleted by this source change.
No full trained v12 publication or v12 accuracy rerun was performed: its original
local trained weights and referenced recordings are absent from this checkout.
The tests use explicitly labelled random-weight software fixtures instead.

Root v12 remains 398,422 parameters. Research `phone_s53` remains 113,064
parameters, 458,230 ONNX bytes and 32,064 neural-cache bytes. These are different
checkpoints, not two models loaded together. Historical `validation.json` is
labelled historical; its old results are not new measurements after removal.
No latency/RSS gain is claimed from removing a formerly lazy/disabled fallback.

## What the linked source actually contains

Pinned reference: `markandrian30/AI231@ed245708c2573188eb769096aba76588f79b8de1`,
`MEX2/OptionB`, subtree `5ecea2a846815c358bb54b1b71ca8f75950d8a54`.

The README describes a DGX2 inventory of 150 reference speakers and 27,956
active files. The complete committed Git tree, manifest and dataset_design.txt
instead agree on **18,375 WAVs, 100 speaker IDs and 96 phrase texts**. Preserve
this distinction; do not report the larger inventory as the audited checkout.

| Split | Recordings | Speaker IDs | Clean | Noisy | Phrase texts |
|---|---:|---:|---:|---:|---:|
| Train | 14677 | 80 | 7344 | 7333 | 96 |
| Validation | 1856 | 10 | 926 | 930 | 96 |
| Test | 1842 | 10 | 922 | 920 | 96 |

The 19 intents occupy 32 intent/slot folders. No duplicate manifest paths,
missing/extra manifest WAVs, cross-split speaker IDs, clean/noisy parent groups,
or cross-split identical file blobs were found. All 96 phrases recur in each
split. Thus speaker-ID grouping is present, but this is not unseen-phrase,
unseen-value or dynamic-catalog validation. Exact hash tests do not exclude
near-duplicates or establish independent human-command recording provenance.

Decoded and hash-verified **1,845 WAVs**: all 1,787 brightness recordings plus
58 coverage controls. All inspected files are mono; 603 are 16 kHz and 1,242
are 12 kHz. No nonfinite PCM, duplicate decoded PCM, samples above the audit's
clipping threshold, or duration discrepancies over 1.1 ms were found. Durations
span 0.76-4.52 seconds. This subset audit is not manual transcript adjudication
and the rate counts do not describe every WAV in the full repository. Whole
recordings were correctly resampled; no old frontend resampling bug was claimed.

**No Python trainer, model notebook, architecture, optimizer, epoch schedule,
selection record or accuracy report is published in this Option B subtree.**
Its dataset design can be reviewed; an unpublished training recipe cannot be
reconstructed. Offline transcription-similarity filtering at 0.80 is described
in the README, not a runtime transcription dependency.

Brightness values are only 20, 60 and 100; volume has up/down commands rather
than numeric settings. The templates contain no custom light names or
song-title/BY/artist tuples, and the published 19 labels contain no UNKNOWN or
background class. The source uses the term reference speakers; final-command
recording/generation provenance is incomplete. No repository-level LICENSE was
present in the inspected root/OptionB. No third-party WAVs are committed or
redistributed in the supplement.

## New experiments, not the external author's training recipe

Three training trials reused this project's own CNN without adding layers.
All used seed 709, six epochs, batch 24, AdamW learning rate 0.00008, weight decay
0.0001, cosine minimum LR 0.00002 and gradient clipping 3. Input normalization
and phoneme inventory were fixed. Training used 1,427 external brightness files
plus 850 own synthetic train recordings; development used 180 external and 149
own recordings. Test did not select training epochs or coefficients.

The parent was an eligible epoch-zero candidate. The provisional development
criterion minimized mean domain phoneme error rate (PER), with replay PER at
most two percentage points above the parent. Ordinary mixed CTC failed that
guard. Own-model replay distillation with coefficient 5 also retained the
parent. Coefficient 20 selected epoch 6 for testing: external PER 87.88% ->
46.46%; replay PER 25.80% -> 27.02%. The frozen teacher is training-only, not
an extra runtime network or a downloaded pretrained model.

Three complete brightness templates found in training rows were compiled for
all 0..100 values. Original forms/IDs remain intact; forms rose from 1,878 to
2,181, with a new graph hash and no inherited acceptance policy. With unchanged
weights, this did not improve validation (1/180 either way).

| Held-out external brightness test | Original grammar | Expanded grammar |
|---|---:|---:|
| Retained parent | 0/180 | 0/180 |
| Adapted candidate | 51/180 | 69/180 |

Expanded candidate: 37/90 clean and 32/90 noisy. Expansion gained 24/lost six
relative to the adapted candidate's original grammar. These are complete raw
commands against source labels, not accepted precision or full-range evidence.

The original V6 regression was re-rendered with all 754 PCM hashes matching.
It is previously inspected regression evidence, not a newly sealed test.

| Category | Parent | Adapted |
|---|---:|---:|
| Light names | 34/40 | 36/40 |
| Named device + value | 27/40 | 33/40 |
| Absolute numbers | 375/404 | 358/404 |
| Relative values | 24/24 | 24/24 |
| Title by artist | 100/108 | 92/108 |
| Title only | 85/90 | 85/90 |
| Total valid commands | 645/706 | 628/706 |

Twenty-six prior successes were lost, nine gained. Grammar expansion did not
change those totals. PER alone was not a sufficient selection safeguard.
Automatic acceptance stayed disabled; rejecting everything is not a successful
negative-speech benchmark. No adapted checkpoint replaces runtime assets.

## ONNX Runtime and verification

Actual ONNX Runtime 1.23.2 CPU execution now supplements OpenCV/PyTorch checks.
Both 40 ms and 80 ms graphs carried nonzero caches. The parent was checked on
180 validation and 180 test records; the candidate on the same 180 test records.
Maximum logit differences were 0.00004578, 0.00005341 and 0.00003624 respectively.
Both chunk sizes produced zero changed frame paths or complete command/tie
results. Agreement does not mean the recognized command is correct.

Matched single-threaded x86 timing, 300 measured calls after warmup, random
feature inputs and the same 1,878-form graph:

| Chunk/backend | Neural median ms | Neural + decoder median ms | Combined p95 ms |
|---|---:|---:|---:|
| 40/OpenCV | .124 | 1.779 | 2.264 |
| 40/ORT | .138 | 1.847 | 2.239 |
| 80/OpenCV | .146 | 3.228 | 3.922 |
| 80/ORT | .160 | 3.247 | 4.117 |

Includes session/cache transfers and, for combined timing, decoding/ranking.
Excludes frontend, microphone waiting, endpoint waiting, actions and startup.
ORT was not faster in median time here. No Pi speed, power, thermal or RSS
claim is made.

GitHub removal verification passed 45 tests with one explicit absent-payload
skip. The integrated local checkout (actual migrated remote source plus restored
full research payload and new tests) passed **430 Python, 18 JS/WASM and eight
Chromium scenarios**, plus the existing seven-WAV browser/PyTorch/OpenCV math
checks. Source-only CI deliberately does not imply acoustic qualification.

## Reproduction and next decision

The downloadable `VCM_NO_ASR_RESEARCH.zip` supplement contains exact experiment
scripts, protocols, histories, summaries and per-recording predictions. It needs
the previous VCM_FINAL_PAYLOAD.zip for this project's baseline artifacts and
runtime. It does not include third-party WAVs or the rejected adapted weights.
The branch keeps the audit/compiler methods, tests and concise results; the
full local experiment payload is separate, as in prior research deliveries.

Keep one small causal CNN. Require per-family complete-command development
checks, not PER alone, before another promotion. Preserve speaker/session and
clean/noisy-parent grouping; review external recording provenance and transcripts;
collect full connected numbers, names, BY pairs and background/out-of-list
examples. Neither model has earned automatic device control or all-ten-use-case
human/Pi qualification.

### Source links

- [Pinned README](https://github.com/markandrian30/AI231/blob/ed245708c2573188eb769096aba76588f79b8de1/MEX2/OptionB/README.md)
- [Committed dataset design](https://github.com/markandrian30/AI231/blob/ed245708c2573188eb769096aba76588f79b8de1/MEX2/OptionB/dataset_design.txt)
- [Label set](https://github.com/markandrian30/AI231/blob/ed245708c2573188eb769096aba76588f79b8de1/MEX2/OptionB/labels.json)
- [Removal commit](https://github.com/justinruaya123/smart-speaker/commit/8607d1519228ae6d1d6f307956acbea3da121c48)
- [Real removal/ONNX fixture verification](https://github.com/justinruaya123/smart-speaker/actions/runs/36542903217)
