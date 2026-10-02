# Fixed-keyword VCM: completed comparison

**Human intent recognition and gated command recall improved, but the all-intent and numeric quality targets still failed.** The selected mixed candidate is an experimental temporal CNN intent model followed by an ordered fixed-word reader. Brightness, color and absolute volume remain disabled by development gates. It is not installed as the default.

The requested Kaggle corpus was acquired and incorporated into expanded-data candidates. Speech-to-Intent-Micro was studied at a pinned commit and a small independent Conv2D baseline was trained. See [the source study](SOURCE_STUDY.md), [the mixed report with a table for every intent](REPORT.md), and [the synthetic report](../../synthetic/keyword-v4/REPORT.md).

## Human comparison on original, presegmented recordings

| Experiment | Raw intent accuracy | Exact accepted / valid | Exact recall | Output precision | Negative false accepts |
|---|---:|---:|---:|---:|---:|
| Prior mixed v1 | 65.45% | 14/1175 | 1.19% | 100.00% | 0/628 |
| Prior human-only control | 69.94% | 178/1175 | 15.15% | 89.00% | 3/628 |
| Mixed keyword v4 | 78.20% | 446/1175 | 37.96% | 97.59% | 3/628 |
| Synthetic keyword v4 | 17.03% | 0/1175 | 0.00% | n/a | 0/628 |

The presegmented mixed result improves raw intent accuracy from 65.45% to 78.20%, and exact accepted recall from 1.19% to 37.96%. It also exceeds the earlier human-only control (15.15% recall at 89% precision). These experiments change data, architecture and calibration together; this is not an attribution of the gain to FSC alone.

**The raw command result did not improve:** before confidence gates, the earlier mixed model had 54.47% exact recall; the new ordered reader has 44.60% (524/1,175). It improves precision before gates from 67.30% to 85.90%. Better intent classification and the replacement of the earlier failed coordinate calibration both contribute to the accepted-result improvement. Ordered cue validation still loses correct commands when cues are missing.

## Every intent, including the actual endpoint path

| Intent | Valid human commands | Raw intent recall | Original-file exact recall | Original-file precision | VAD exact recall | VAD output precision |
|---|---:|---:|---:|---:|---:|---:|
| UNKNOWN | 0 | 73.01% | n/a | n/a | n/a | n/a |
| POWER_ON | 232 | 85.66% | 56.47% | 100.00% | 41.38% | 100.00% |
| POWER_OFF | 268 | 83.28% | 57.46% | 96.86% | 52.61% | 95.92% |
| BRIGHTNESS | 213 | 74.38% | 0.00% | n/a | 0.00% | n/a |
| COLOR_SET | 106 | 51.94% | 0.00% | n/a | 0.00% | n/a |
| MEDIA_PLAY | 73 | 92.22% | 69.86% | 98.08% | 67.12% | 100.00% |
| MEDIA_RESUME | 37 | 89.19% | 78.38% | 100.00% | 70.27% | 100.00% |
| TIME_QUERY | 50 | 68.00% | 24.00% | 92.31% | 0.00% | n/a |
| DATE_QUERY | 59 | 81.36% | 38.98% | 100.00% | 8.47% | 100.00% |
| VOLUME | 137 | 90.00% | 33.58% | 92.00% | 33.58% | 82.14% |

Raw intent recall uses all examples bearing that family label, including unresolved requests; command recall uses only valid references. Output precision groups by the emitted family and includes cross-family mistakes. UNKNOWN has no valid commands: 476/478 are rejected in the original-file path and 478/478 through VAD.

The mixed VAD path emits 379 outputs, 363 correct, for 30.89% recall and 95.78% precision. It falsely accepts 3/628 negative or unresolved recordings (0.48%). Time-query recall is zero and volume precision is 82.14%; aggregate precision does not qualify every intent for use. The policy was calibrated on original development utterances, not VAD-transformed development. That deployment mismatch is now measured and must be addressed in a new development-selected version, not by tuning this test.

The original-file runtime agrees with the batched evaluation on every acceptance/command decision for both historical test sets. VAD loses 123 previously correct human commands and gains 40. Among lost commands, no-speech rejection is not the main explanation: changed family/confidence and missing keyword/readiness/command evidence account for the losses. See [post-test diagnostics](post-test-diagnostics.json). No thresholds were changed after these observations.

## Numeric and directional coverage

| Operation | Human valid references | Correct before gates | Correct accepted | Accepted output precision |
|---|---:|---:|---:|---:|
| BRIGHTNESS/SET | 73 | 3 | 0 | n/a |
| BRIGHTNESS/INCREASE | 75 | 5 | 0 | n/a |
| BRIGHTNESS/DECREASE | 65 | 5 | 0 | n/a |
| VOLUME/SET | 2 | 0 | 0 | n/a |
| VOLUME/INCREASE | 70 | 35 | 19 | 95.00% |
| VOLUME/DECREASE | 65 | 36 | 27 | 90.00% |

There are only 179 original human absolute-brightness training commands covering 13 distinct values, and 31 absolute-volume commands covering 11 values. The existing synthetic training data covers all 101 values for each family, and isolated human number words add acoustic supervision. Neither proves connected human-number recognition. Absolute volume has only three development positives, below the ten-accept minimum for enabling an operation. There is no human acoustic evidence here supporting all 0–100 values.

The grammar passes 404 textual cases (101 values × two numeric families × optional percent), plus conflict/order tests. Those are software checks. The new mixed acoustic system gets only 3/75 absolute numeric test commands correct before gates, compared with 2/75 previously; both accept zero after calibration. The requested reliable “set brightness to 100 percent” behavior has not been achieved.

## Ordered-cue case study

The path takes an external wake-end sample, endpoints following speech, selects an intent, then reads ordered cues only after acceptance. `SET -> BRIGHTNESS -> TO -> FIFTY -> PERCENT` represents an intended cue sequence, not a successful measured example. The finite reader distinguishes TO from BY, bounds values to 0–100, treats bare UP/DOWN as +/-10 percentage points, and rejects misplaced/conflicting cues.

A real development recording (`smartlights/2623/far`, “Put the brightness in the shed at fifty.”) illustrates the remaining failure. The classifier identifies BRIGHTNESS. Offline diagnostics decode brightness/to/twenty and infer 20 instead of 50. The frozen runtime rejects at the disabled brightness intent gate before invoking the reader. Another recording of the same phrase lacks brightness context in the decoded cues. Preserving order cannot repair missing or wrong acoustic words. See [ordered examples](ordered-cue-examples.json) and [development diagnostics](development-diagnostics.json).

In another development phrase (“turn up the lights in here please”), the intent model selects POWER_OFF while the reader detects turn/up/lights. The context check rejects the conflict; it does not change the accepted family to rescue the command. VCM ownership is preserved.

## Both synthetic and mixed experiments

| Model | Test speech | Raw intent accuracy | Exact recall before gates | Accepted / correct | Gated exact recall | Gated precision |
|---|---|---:|---:|---:|---:|---:|
| mixed | human | 78.20% | 44.60% | 457/446 | 37.96% | 97.59% |
| mixed | synthetic | 99.26% | 62.33% | 83/79 | 13.72% | 95.18% |
| synthetic | human | 17.03% | 2.38% | 0/0 | 0.00% | n/a |
| synthetic | synthetic | 91.22% | 11.46% | 0/0 | 0.00% | n/a |

The synthetic-only model still fails transfer to human speech and all its human-development gates remain disabled. The selected mixed intent model is the earlier 1D CNN; the newly added FSC recordings train the selected keyword reader and the evaluated shared/Micro-inspired intent candidates. The earlier CNN already used the pre-existing FSC subset. This distinction matters when interpreting the data addition.

## Failure cascade for each valid human intent

| Intent | Valid | Correct family | Pass intent gate | Valid reader + ready | Exact before final gate | Exact accepted |
|---|---:|---:|---:|---:|---:|---:|
| POWER_ON | 232 | 209 | 167 | 131 | 131 | 131 |
| POWER_OFF | 268 | 232 | 207 | 155 | 155 | 154 |
| BRIGHTNESS | 213 | 162 | 0 | 0 | 0 | 0 |
| COLOR_SET | 106 | 62 | 0 | 0 | 0 | 0 |
| MEDIA_PLAY | 73 | 68 | 62 | 51 | 51 | 51 |
| MEDIA_RESUME | 37 | 33 | 32 | 29 | 29 | 29 |
| TIME_QUERY | 50 | 34 | 25 | 12 | 12 | 12 |
| DATE_QUERY | 59 | 48 | 38 | 23 | 23 | 23 |
| VOLUME | 137 | 126 | 101 | 74 | 64 | 46 |

## Implementation, validation and limits

- Canonical implementation: `scripts/models/keyword_vcm.py`, `keyword_intent.py`, `scripts/train/keyword_data.py`, `keyword_training.py`, `scripts/runtime/keyword_runtime.py`, and `scripts/test/keyword_eval.py`. Registered as research-only in `scripts/modules.json`.
- The selected mixed bundle loads 1,243,830 parameters across two FP32 graphs totaling 5,029,167 bytes. It is a CPU ONNX experimental command path, not the repository’s 29 KB INT8 model.
- Identical 60 preloaded development recordings, two CPU threads: p50 3.18 ms, p95 5.28 ms; session loading 33.79 ms. Timing includes frontend and conditional reader/validation. It excludes disk, speech duration, VAD and the 400 ms endpoint wait; many rejected intents skip the reader.
- Passed variable-length/batch/padding ONNX parity, ordered numeric/conflict tests, rejection-before-reader checks, all 35,687 manifest-record hashes, speaker separation, root runtime/parser tests, and preservation of installed artifacts and prior result JSONs.
- Replayed all 1,803 human and 672 synthetic historical test recordings through both frozen models using both original-file and VAD paths. A separate software fixture verifies that an externally supplied 500 ms wake boundary excludes preceding samples and preserves following speech. No wake detector, live microphone, device action or physical Pi was tested.
- All 48 keyword epochs and exports completed. Both training launchers returned code 1 without an independently captured native Python exit code. Separate CPU exports returned 0 and graph checks passed. The 40-epoch Micro candidate returned 0. See `execution-receipt.json`; do not infer a specific CUDA fault from the launcher code.
- Human labels are inherited silver annotations. TTS intended transcripts and requested Filipino accents are not acoustically adjudicated. The prior test has already been used historically, so this is a comparison rather than an untouched confirmation.
- No online vocabulary expansion was tested. Adding a word to JSON does not add an acoustic output class. Names, artists, new-corpus tests and automatic STT fallback remain outside this phase. Installed VCM/STT defaults are unchanged.

## Run the isolated pipeline

```powershell
.venv/Scripts/python.exe scripts/keywords.py predict --mode mixed --audio path/to/command.wav
.venv/Scripts/python.exe scripts/keywords.py predict --mode mixed --audio path/to/recording.wav --wake-end-ms 900 --vad
.venv/Scripts/python.exe scripts/keywords.py list-mics
.venv/Scripts/python.exe scripts/keywords.py listen --mode mixed --mic 2 --vad-debug
```

The microphone command must be armed by the assistant; it is not a wake listener. Outputs are typed JSON or abstentions and never execute an action. Use `--mode synthetic` for the corresponding frozen synthetic control.

Analysis commands are `scripts/keywords.py check --mode both` and `scripts/keywords.py report --mode both`; the comparison is rendered by `experiments/mixed/keyword-v4/write_comparison.py`. Fitting, calibration and evaluation refuse to overwrite completed/frozen evidence. New training or endpoint calibration requires a new study version. Saved checkpoint histories record the greedy-CTC selection used during fitting; the final finite-number decoder was added before acceptance calibration and test.

The next justified experiment is to collect/adjudicate connected human brightness, direction and numeric commands from independent speakers, then fit and calibrate through the same endpoint path. A controlled same-data comparison should separate the effect of the intent architecture, ordered reader, FSC expansion and calibration. These test outcomes must not choose the next thresholds.
