# Synthetic ordered fixed-keyword VCM

A small Conv2D/BiLSTM model predicts VCM intent, readiness, and a fixed sequence of command cues. The accepted intent owns parsing; the keyword reader cannot select another family. The numeric grammar accepts only 0-100. This is an experimental pipeline with typed JSON and no action executor.

This version incorporates the requested Kaggle Fluent Speech corpus and studies Speech-to-Intent-Micro. The same historical human/synthetic test is reused for comparison; it is not an untouched confirmation after prior research iterations. Development selected all current weights and gates before this round of test inference.

Selected intent candidate: shared. Loaded graphs: 857,899 total parameters and 3,457,545 ONNX bytes. Keyword model selected epoch: 47.

The keyword-model checkpoints were selected using greedy CTC development decoding. The finite-number forward decoder and confidence policy were subsequently selected/calibrated on human development before test inference.

## Each intent

| Intent | Human examples | Valid human commands | Raw intent recall | Gated intent recall | Accepted command precision | Exact command recall |
|---|---:|---:|---:|---:|---:|---:|
| UNKNOWN | 478 | 0 | 34.94% | 0.00% | n/a | n/a |
| POWER_ON | 251 | 232 | 13.94% | 0.00% | n/a | 0.00% |
| POWER_OFF | 287 | 268 | 2.44% | 0.00% | n/a | 0.00% |
| BRIGHTNESS | 242 | 213 | 11.16% | 0.00% | n/a | 0.00% |
| COLOR_SET | 129 | 106 | 5.43% | 0.00% | n/a | 0.00% |
| MEDIA_PLAY | 90 | 73 | 0.00% | 0.00% | n/a | 0.00% |
| MEDIA_RESUME | 37 | 37 | 13.51% | 0.00% | n/a | 0.00% |
| TIME_QUERY | 50 | 50 | 4.00% | 0.00% | n/a | 0.00% |
| DATE_QUERY | 59 | 59 | 0.00% | 0.00% | n/a | 0.00% |
| VOLUME | 180 | 137 | 31.67% | 0.00% | n/a | 0.00% |

## UNKNOWN

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 478 | 34.94% | n/a | 0/0 | n/a |
| synthetic | 64 | 68.75% | n/a | 0/0 | n/a |

UNKNOWN has no valid commands. Its relevant measure is rejection and false acceptance; command recall is undefined.

## POWER_ON

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 251 | 13.94% | 7.33% | 0/232 | n/a |
| synthetic | 12 | 91.67% | 58.33% | 0/12 | n/a |

## POWER_OFF

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 287 | 2.44% | 1.87% | 0/268 | n/a |
| synthetic | 12 | 58.33% | 58.33% | 0/12 | n/a |

## BRIGHTNESS

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 242 | 11.16% | 0.00% | 0/213 | n/a |
| synthetic | 250 | 98.80% | 2.56% | 0/234 | n/a |

## COLOR_SET

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 129 | 5.43% | 0.00% | 0/106 | n/a |
| synthetic | 36 | 91.67% | 36.11% | 0/36 | n/a |

## MEDIA_PLAY

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 90 | 0.00% | 0.00% | 0/73 | n/a |
| synthetic | 12 | 83.33% | 83.33% | 0/12 | n/a |

## MEDIA_RESUME

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 37 | 13.51% | 13.51% | 0/37 | n/a |
| synthetic | 12 | 83.33% | 66.67% | 0/12 | n/a |

## TIME_QUERY

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 50 | 4.00% | 2.00% | 0/50 | n/a |
| synthetic | 12 | 41.67% | 41.67% | 0/12 | n/a |

## DATE_QUERY

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 59 | 0.00% | 0.00% | 0/59 | n/a |
| synthetic | 12 | 75.00% | 58.33% | 0/12 | n/a |

## VOLUME

| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |
|---|---:|---:|---:|---:|---:|
| human | 180 | 31.67% | 0.00% | 0/137 | n/a |
| synthetic | 250 | 94.80% | 1.28% | 0/234 | n/a |

## Numeric operations

| Operation | Human positives | Correct before gates | Correct accepted | Accepted precision |
|---|---:|---:|---:|---:|
| BRIGHTNESS/SET | 73 | 0 | 0 | n/a |
| BRIGHTNESS/INCREASE | 75 | 0 | 0 | n/a |
| BRIGHTNESS/DECREASE | 65 | 0 | 0 | n/a |
| VOLUME/SET | 2 | 0 | 0 | n/a |
| VOLUME/INCREASE | 70 | 0 | 0 | n/a |
| VOLUME/DECREASE | 65 | 0 | 0 | n/a |

## Overall

| Speech | Accepted | Correct | Precision | Exact recall | Negative false accepts |
|---|---:|---:|---:|---:|---:|
| human | 0 | 0 | n/a | 0.00% | 0/628 |
| synthetic | 0 | 0 | n/a | 0.00% | 0/96 |

## Reproduce and use

```powershell
.venv/Scripts/python.exe scripts/keywords.py predict --mode synthetic --audio path/to/command.wav
.venv/Scripts/python.exe scripts/keywords.py predict --mode synthetic --audio path/to/recording.wav --wake-end-ms 900 --vad
.venv/Scripts/python.exe scripts/keywords.py list-mics
.venv/Scripts/python.exe scripts/keywords.py listen --mode synthetic --mic 2 --vad-debug
```

The wake end time is supplied by an external wake detector; no wake-word detector was trained or validated. The listen command processes endpointed speech and must be armed by the assistant integration. CTC emission intervals are approximate cues, not verified acoustic word boundaries.

Thresholds are selected exactly on development, with a shared negative-error budget and independent numeric-operation gates. No accepted intent is replaced by a parameter prediction. Unsupported groups abstain. See selection-before-test.json and development-search.json. Test accuracy below those targets is a failed generalization gate, even when development passed.

The TTS accents remain requested rather than verified. Human semantic labels are silver annotations, and generated transcript fidelity is unadjudicated. Names, artists and new-word tests are outside scope. CPU/replay checks do not establish live-microphone or Pi performance.
