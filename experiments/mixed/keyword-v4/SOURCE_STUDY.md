# Fluent Speech corpus and Speech-to-Intent-Micro study

The requested sources are now incorporated into the fixed-command study. Fluent contributes original human training recordings. Speech-to-Intent-Micro informed an independently implemented small Conv2D baseline; its published score is not a measured result of this project.

## Dataset acquisition and use

Downloaded [tommyngx/fluent-speech-corpus on Kaggle](https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus), version 1. The archive is 1,548,762,021 bytes, SHA-256 `1f0de5928db4d53b4a4fd5810974a5f79bcfe8417fb59bea1dae821373d717e9`. The receipt is [fluent-incorporation.json](fluent-incorporation.json). It records the publisher splits and the precise incorporation counts.

| Publisher split | Recordings | Speakers |
|---|---:|---:|
| Train | 23,132 | 77 |
| Validation | 3,118 | 10 |
| Test | 3,793 | 10 |
| Total | 30,043 | 97 |

The existing workspace already contained part of this corpus. Sixty-four overlapping WAV hashes were compared and matched. The importer added 5,820 missing WAV files and 9,399 additional intent-training records after excluding held-out speakers and duplicate audio. Some newly used records referenced files already present locally. It retained the publisher action/object/location fields, speaker, transcript, source and audio hash. Existing complete-command references remain unchanged. Locations are contextual words, not new device-name classes. The packaged Fluent Speech Commands Public License is retained with the source provenance.

FSC primarily strengthens action and object context. It does not supply a 0–100 percentage-value slot, so incorporating it does not establish numeric-command coverage. Our historical test is a bounded, mapped multi-corpus subset, not the official full FSC benchmark.

| Active v4 supervision | Training recordings | Use |
|---|---:|---|
| Existing human complete-command manifest | 7,919 | Intent, readiness, ordered cues and finite command labels |
| Existing `tts` training partition | 2,688 | Whole synthetic commands with intended text and fixed labels |
| Original isolated cue/number words | 6,921 | Fixed-word CTC supervision |
| Additional original intent-labeled speech | 10,688 | Intent and cues; readiness loss masked |
| Additional original connected cue speech | 2,455 | CTC only; no invented intent label |
| Total mixed training records | 30,671 | Weighted sampling, not synthetic word concatenation |

The 9,399 additional FSC examples are a subset of the 10,688 intent-only records. Human development is 1,869 recordings; historical human test is 1,803. Synthetic development and test each contain 672 recordings. New audio is confined to training. Speaker IDs and audio hashes are disjoint from the historical human development/test partitions. The checks cover 35,687 manifest records including evaluation partitions.

## What the linked repository actually does

Studied [Speech-to-Intent-Micro](https://github.com/AIWintermuteAI/Speech-to-Intent-Micro) at commit `7e6674117c55145d0071d39768c183e63e917479`. Source copies and hashes are retained in `.cache/keyword-study-sources/` and `source-study-receipt.json`. The repository uses Apache-2.0 licensing.

Its [model implementation](https://github.com/AIWintermuteAI/Speech-to-Intent-Micro/blob/7e6674117c55145d0071d39768c183e63e917479/models.py) combines small 2D convolutions, global max pooling, a 32-unit dense layer, an intent head and two categorical slot outputs. Its [data implementation](https://github.com/AIWintermuteAI/Speech-to-Intent-Micro/blob/7e6674117c55145d0071d39768c183e63e917479/data_utils.py) uses MFCCs and a fixed 150-frame input. The README reports 87.5% joint intent/slot accuracy and a 29 KB INT8 model. Those are upstream claims under its task and evaluation, not reproduced local scores.

This is a useful design for a finite action/object/location ontology. Global pooling does not itself demonstrate ordered number parsing, negation handling, wake-boundary detection, or 101 distinct brightness values. The upstream training recipe also uses augmentation. The present study keeps original human recordings and the separately labeled existing TTS corpus; it did not import that augmentation recipe.

## Local architectural comparison

All local acoustic candidates started from random weights. Development selected checkpoints; no candidate was chosen from final test outcomes.

| Candidate | Parameters | Selected epoch | Human development macro intent recall | Role |
|---|---:|---:|---:|---|
| Earlier small temporal 1D CNN (`keyword-v2`) | 385,931 | 41/48 | 73.54% | Retained intent candidate |
| Conv2D/BiLSTM with expanded human corpus (`keyword-v4`) | 857,899 | 29/48 | 57.87% | Ordered keyword reader and shared-intent candidate |
| Micro-inspired Conv2D (`keyword-v4`) | 91,338 | 11/40 | 54.66% | Independently implemented intent baseline |

These are not controlled architecture ablations: the earlier CNN had less auxiliary data, the objectives differ, and model selection rules differ. They establish the measured candidate ordering, not that one architecture is universally superior. The final policy compares complete-command recall after development confidence gates. It selected the earlier CNN as the authoritative intent model and the expanded-data BiLSTM as the fixed keyword reader. The Micro-inspired candidate is preserved as a negative result.

The keyword reader produces a sequence over 158 fixed tokens including blank and filler. Numeric decoding scores the ordered word representations of 0–100 plus a missing-number competitor from the same CTC acoustic posteriors. It is neither general STT nor an automatic second-model fallback. It cannot change the accepted family. Numeric and directional operations have separate development gates.

## Post-wake command path

```text
external wake-end sample
  -> remaining mono 16 kHz PCM
  -> energy VAD endpoint
  -> small VCM intent + confidence gate
  -> accepted intent's ordered keyword/number reader
  -> finite validation + command confidence gate
  -> typed JSON or abstention
```

For an intended phrase such as `SET BRIGHTNESS TO FIFTY PERCENT`, the reader preserves the relative cue order, requires brightness context before the number, distinguishes `TO` from `BY`, and rejects conflicting or malformed values. Text grammar support is not proof that every spoken percentage is recognized. CTC frame emissions are approximate cue positions, not verified word boundaries.

The CLI accepts an externally supplied `--wake-end-ms`; no wake detector was trained. Live microphone capture can be armed by an upstream assistant, but neither live wake behavior nor a physical Pi was validated here. See the measured command results and numeric-operation tables in [REPORT.md](REPORT.md), including disabled operations and false accepts.

The final comparison reuses historical test recordings. It is not a fresh untouched confirmation. Frozen policies, raw predictions, source receipts and the abandoned v3 preparation are preserved so that this limitation remains visible.
