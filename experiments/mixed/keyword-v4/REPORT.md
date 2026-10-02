# Mixed ordered fixed-keyword VCM

A temporal 1D CNN first selects and gates the VCM intent. A separate fixed Conv2D/BiLSTM keyword reader runs only after acceptance. The accepted intent owns parsing; the keyword reader cannot select another family. The numeric grammar accepts only 0-100. This is an experimental pipeline with typed JSON and no action executor.

This version incorporates the requested Kaggle Fluent Speech corpus and studies Speech-to-Intent-Micro. The same historical human/synthetic test is reused for comparison; it is not an untouched confirmation after prior research iterations. Development selected all current weights and gates before this round of test inference.

Selected intent candidate: cnn. Loaded graphs: 1,243,830 total parameters and 5,029,167 ONNX bytes. Keyword model selected epoch: 29.

The keyword-model checkpoints were selected using greedy CTC development decoding. The finite-number forward decoder and confidence policy were subsequently selected/calibrated on human development before test inference.

## Each intent

| Intent | Human examples | Valid human commands | Raw intent recall | Gated intent recall | Accepted command precision | Exact command recall |
|---|---:|---:|---:|---:|---:|---:|
| UNKNOWN | 478 | 0 | 73.01% | 0.00% | n/a | n/a |
| POWER_ON | 251 | 232 | 85.66% | 66.53% | 100.00% | 56.47% |
| POWER_OFF | 287 | 268 | 83.28% | 73.17% | 96.86% | 57.46% |
| BRIGHTNESS | 242 | 213 | 74.38% | 0.00% | n/a | 0.00% |
| COLOR_SET | 129 | 106 | 51.94% | 0.00% | n/a | 0.00% |
| MEDIA_PLAY | 90 | 73 | 92.22% | 78.89% | 98.08% | 69.86% |
| MEDIA_RESUME | 37 | 37 | 89.19% | 86.49% | 100.00% | 78.38% |
| TIME_QUERY | 50 | 50 | 68.00% | 50.00% | 92.31% | 24.00% |
| DATE_QUERY | 59 | 59 | 81.36% | 64.41% | 100.00% | 38.98% |
| VOLUME | 180 | 137 | 90.00% | 67.22% | 92.00% | 33.58% |

## UNKNOWN

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 478 | 73.01% | n/a | 0/0 | n/a |
| synthetic | 64 | 93.75% | n/a | 0/0 | n/a |

UNKNOWN has no valid commands. Its relevant measure is rejection and false acceptance; command recall is undefined.

## POWER_ON

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 251 | 85.66% | 62.93% | 131/232 | 100.00% |
| synthetic | 12 | 100.00% | 83.33% | 10/12 | 100.00% |

## POWER_OFF

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 287 | 83.28% | 60.45% | 154/268 | 96.86% |
| synthetic | 12 | 100.00% | 100.00% | 12/12 | 100.00% |

## BRIGHTNESS

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 242 | 74.38% | 6.10% | 0/213 | n/a |
| synthetic | 250 | 100.00% | 59.40% | 0/234 | n/a |

## COLOR_SET

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 129 | 51.94% | 9.43% | 0/106 | n/a |
| synthetic | 36 | 100.00% | 86.11% | 0/36 | n/a |

## MEDIA_PLAY

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 90 | 92.22% | 73.97% | 51/73 | 98.08% |
| synthetic | 12 | 100.00% | 100.00% | 12/12 | 100.00% |

## MEDIA_RESUME

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 37 | 89.19% | 78.38% | 29/37 | 100.00% |
| synthetic | 12 | 91.67% | 83.33% | 10/12 | 100.00% |

## TIME_QUERY

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 50 | 68.00% | 28.00% | 12/50 | 92.31% |
| synthetic | 12 | 100.00% | 66.67% | 8/12 | 100.00% |

## DATE_QUERY

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 59 | 81.36% | 42.37% | 23/59 | 100.00% |
| synthetic | 12 | 100.00% | 100.00% | 12/12 | 100.00% |

## VOLUME

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 180 | 90.00% | 51.82% | 46/137 | 92.00% |
| synthetic | 250 | 100.00% | 53.42% | 15/234 | 78.95% |

## Numeric operations

| Operation | Human positives | Correct before gates | Correct accepted | Accepted precision |
|---|---:|---:|---:|---:|
| BRIGHTNESS/SET | 73 | 3 | 0 | n/a |
| BRIGHTNESS/INCREASE | 75 | 5 | 0 | n/a |
| BRIGHTNESS/DECREASE | 65 | 5 | 0 | n/a |
| VOLUME/SET | 2 | 0 | 0 | n/a |
| VOLUME/INCREASE | 70 | 35 | 19 | 95.00% |
| VOLUME/DECREASE | 65 | 36 | 27 | 90.00% |

## Overall

| Speech | Accepted | Correct | Precision | Exact recall | Negative false accepts |
|---|---:|---:|---:|---:|---:|
| human | 457 | 446 | 97.59% | 37.96% | 3/628 |
| synthetic | 83 | 79 | 95.18% | 13.72% | 0/96 |

## Reproduce and use

```powershell
.venv/Scripts/python.exe scripts/keywords.py predict --mode mixed --audio path/to/command.wav
.venv/Scripts/python.exe scripts/keywords.py predict --mode mixed --audio path/to/recording.wav --wake-end-ms 900 --vad
.venv/Scripts/python.exe scripts/keywords.py list-mics
.venv/Scripts/python.exe scripts/keywords.py listen --mode mixed --mic 2 --vad-debug
```

The wake end time is supplied by an external wake detector; no wake-word detector was trained or validated. The listen command processes endpointed speech and must be armed by the assistant integration. CTC emission intervals are approximate cues, not verified acoustic word boundaries.

Thresholds are selected exactly on development, with a shared negative-error budget and independent numeric-operation gates. No accepted intent is replaced by a parameter prediction. Unsupported groups abstain. See selection-before-test.json and development-search.json. Test accuracy below those targets is a failed generalization gate, even when development passed.

The TTS accents remain requested rather than verified. Human semantic labels are silver annotations, and generated transcript fidelity is unadjudicated. Names, artists and new-word tests are outside scope. CPU/replay checks do not establish live-microphone or Pi performance.
