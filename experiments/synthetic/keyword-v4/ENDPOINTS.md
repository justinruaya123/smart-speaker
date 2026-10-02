# Synthetic frozen endpoint replay

Original historical files with wake boundary zero, using the unchanged frozen policy. There are no annotated wake words in these files. This supplemental evaluation was specified after the original-file results were seen and does not select a model or threshold.

| Speech | Accepted outputs | Correct outputs | Exact / valid commands | Exact recall | Output precision | Negative false accepts |
|---|---:|---:|---:|---:|---:|---:|
| human | 0 | 0 | 0/1175 | 0.00% | n/a | 0/628 |
| synthetic | 0 | 0 | 0/576 | 0.00% | n/a | 0/96 |

## UNKNOWN

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 0 | 0 | n/a | n/a |
| synthetic | 0 | 0 | n/a | n/a |

## POWER_ON

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 232 | 0 | 0.00% | n/a |
| synthetic | 12 | 0 | 0.00% | n/a |

## POWER_OFF

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 268 | 0 | 0.00% | n/a |
| synthetic | 12 | 0 | 0.00% | n/a |

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
| human | 73 | 0 | 0.00% | n/a |
| synthetic | 12 | 0 | 0.00% | n/a |

## MEDIA_RESUME

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 37 | 0 | 0.00% | n/a |
| synthetic | 12 | 0 | 0.00% | n/a |

## TIME_QUERY

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 50 | 0 | 0.00% | n/a |
| synthetic | 12 | 0 | 0.00% | n/a |

## DATE_QUERY

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 59 | 0 | 0.00% | n/a |
| synthetic | 12 | 0 | 0.00% | n/a |

## VOLUME

| Speech | Valid commands | Exact accepted | Exact recall | Output precision |
|---|---:|---:|---:|---:|
| human | 137 | 0 | 0.00% | n/a |
| synthetic | 234 | 0 | 0.00% | n/a |

All original-file single-recording decisions matched the batched evaluator. Endpoint exact correctness requires one correct command and no extra accepted outputs. UNKNOWN recall is undefined; see the raw JSON for false accepts.

The software wake-boundary fixture passed; it does not establish wake-word recognition. Neither live microphone nor Pi results are claimed. See [the full comparison](../../mixed/keyword-v4/COMPARISON.md).
