# Mixed frozen endpoint replay

Original historical files with wake boundary zero, using the unchanged frozen policy. There are no annotated wake words in these files. This supplemental evaluation was specified after the original-file results were seen and does not select a model or threshold.

| Speech | Accepted outputs | Correct outputs | Exact / valid commands | Exact recall | Output precision | Negative false accepts |
|---|---:|---:|---:|---:|---:|---:|
| human | 379 | 363 | 363/1175 | 30.89% | 95.78% | 3/628 |
| synthetic | 55 | 50 | 50/576 | 8.68% | 90.91% | 0/96 |

## UNKNOWN

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 0 | 0 | n/a | n/a |
| synthetic | 0 | 0 | n/a | n/a |

## POWER_ON

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 232 | 96 | 41.38% | 100.00% |
| synthetic | 12 | 11 | 91.67% | 100.00% |

## POWER_OFF

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 268 | 141 | 52.61% | 95.92% |
| synthetic | 12 | 9 | 75.00% | 100.00% |

## BRIGHTNESS

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 213 | 0 | 0.00% | n/a |
| synthetic | 234 | 0 | 0.00% | n/a |

## COLOR_SET

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 106 | 0 | 0.00% | n/a |
| synthetic | 36 | 0 | 0.00% | n/a |

## MEDIA_PLAY

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 73 | 49 | 67.12% | 100.00% |
| synthetic | 12 | 12 | 100.00% | 100.00% |

## MEDIA_RESUME

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 37 | 26 | 70.27% | 100.00% |
| synthetic | 12 | 7 | 58.33% | 100.00% |

## TIME_QUERY

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 50 | 0 | 0.00% | n/a |
| synthetic | 12 | 0 | 0.00% | n/a |

## DATE_QUERY

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 59 | 5 | 8.47% | 100.00% |
| synthetic | 12 | 3 | 25.00% | 100.00% |

## VOLUME

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 137 | 46 | 33.58% | 82.14% |
| synthetic | 234 | 8 | 3.42% | 61.54% |

All original-file single-recording decisions matched the batched evaluator. Endpoint exact correctness requires one correct command and no extra accepted outputs. UNKNOWN recall is undefined; see the raw JSON for false accepts.

The software wake-boundary fixture passed; it does not establish wake-word recognition. Neither live microphone nor Pi results are claimed. See [the full comparison](../../mixed/keyword-v4/COMPARISON.md).
