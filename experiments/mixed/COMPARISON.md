# Synthetic and mixed VCM: completed experiment, failed quality target

Both requested systems were fitted from random initialization, exported to CPU ONNX, calibrated on development data, and scored once on the frozen tests. Neither recognizes all fixed commands reliably on human speech. These are research artifacts; the installed VCM/STT was preserved.

Read the [synthetic report](../synthetic/REPORT.md) and [mixed report](REPORT.md) for per-intent results, selected hyperparameters, coverage and timing. The machine-readable results and individual predictions are alongside each report. This comparison was written after scoring; it changes no model, threshold, label, test or prediction.

## What was built

The `tts` dataset contains 4,032 whole generated utterances, about 2.49 hours, across 12 original designed voices. Voice IDs are disjoint between training (8), development (2) and test (2). The text inventory is fixed across those voice splits; this tests new generated voices, not new words or unseen command wording. English voices include requested US, UK and Australian varieties and Filipino-accented English prompts. Accent authenticity remains unverified. Intended transcripts and keyword character spans are labels, not independently verified acoustic transcripts or word timestamps.

The public Qwen3-TTS 1.7B VoiceDesign and Base snapshots were downloaded from Hugging Face. VoiceDesign creates original anchors and Base generates each complete utterance from them. On the actual RTX 4080 SUPER 16 GB, batch 16 with SDPA and bfloat16 used at most 8.31 GiB active and 13.71 GiB reserved GPU memory in the recorded batches after cache cleanup. Successful generation batches totaled 35.18 minutes; acquisition, anchor creation, smoke runs and restarts are excluded from that total. See [TTS selection and provenance](../synthetic/TTS-SELECTION.md), `tts-performance.json` and `generation-adjustments.json` in the synthetic directory.

The acoustic recognizer is a small depthwise residual CNN with attended intent, readiness, operation, amount and color heads, plus auxiliary keyword supervision. The number head combines tens/units and finite-value logits. Each expert has 323,685 parameters. Recognition uses no STT or TTS model. The fixed output inventory covers the nine supported intent families, UNKNOWN, absolute integers 0-100, finite colors and relative level commands. Bare UP/DOWN represents ten percentage points. Explicit relative-by recordings cover 1, 10, 25, 50 and 100; parser validity does not establish acoustic coverage of all relative combinations.

The synthetic expert was trained on 2,688 generated training utterances. The human expert used 7,919 original training recordings; the full human manifest has 11,591 records from FSC, SLURP-real and SnipsSmartLights. The mixed system combines these two separately fitted experts using per-intent real-data weights and confidence biases. It is an ensemble, not a single model trained on a concatenated corpus. Human semantic labels are programmatic silver labels derived from the original annotations and existing strict grammar.

## Frozen operating points on human speech

The human test has 1,803 recordings: 1,175 valid commands and 628 negative or unresolved examples. Exact command recall is correct accepted commands divided by valid commands. Precision is correct accepted commands divided by all accepted outputs. A wrong command emitted on a valid utterance reduces precision even though it is not counted as a negative-speech false accept.

| System | Accepted outputs | Correct accepted | Accepted precision | Exact command recall | False accepts on 628 negatives |
|---|---:|---:|---:|---:|---:|
| Synthetic | 0 | 0 | Undefined | 0.00% | 0 |
| Mixed | 14 | 14 | 100.00% | 1.19% | 0 |
| Human-only control | 200 | 178 | 89.00% | 15.15% | 3 |

The synthetic system failed development support/precision gates for every intent. Mixed retained only DATE_QUERY; its 14 correct test outputs are a small set and do not validate other intents. Both systems disabled all brightness/volume operations. The control retained POWER_OFF, but its test precision fell below the intended 95% target.

Optuna ran 1,800 development trials per system, plus a bounded shared-threshold check. Per-intent source weights, log-odds biases and confidence/margin thresholds were selected using human development only. The constraints were at least 95% accepted precision, at most 1% negative false acceptance, and at least ten development accepts per enabled intent/operation. Test labels did not choose checkpoints or policies. Selection receipts were written before scoring.

The mixed optimizer did not outperform the human-only control. Its coordinate search changes competing normalized family scores and does not guarantee retaining the best pure-human endpoint. It already yielded fewer correct accepts on development (12 versus 146), before the final test. This is a limitation of this search and ensemble recipe, not evidence that synthetic augmentation can never help. A future version should retain the pure-source baselines as incumbent candidates and jointly validate changes to the competing family scores on development. No such change was applied after seeing this test.

## Diagnostic results before confidence gates

These diagnostics retain the selected source weights and biases, predicted readiness and finite-value validity. They remove confidence, margin and disabled-operation gates. They are not another calibrated operating point and should not be deployed.

| System | Test speech | Exact command recall | Output precision | False accepts on negative speech |
|---|---|---:|---:|---:|
| Synthetic | Human | 273/1,175 = 23.23% | 40.44% | 143/628 = 22.77% |
| Synthetic | Generated, held-out voices | 551/576 = 95.66% | 95.99% | 3/96 = 3.13% |
| Mixed | Human | 640/1,175 = 54.47% | 67.30% | 90/628 = 14.33% |
| Mixed | Generated, held-out voices | 99/576 = 17.19% | 26.76% | 24/96 = 25.00% |

The synthetic expert learned the generated task substantially better than the human task. Mixing improved human diagnostic recall relative to the synthetic expert, but its errors still prevent broad acceptance and its generated-speech performance deteriorated. Turning gates off exposes those errors; it does not resolve them.

## Percentages and relative levels

| System | Human absolute commands correct | Generated absolute commands correct | Human relative commands correct | Generated relative commands correct |
|---|---:|---:|---:|---:|
| Synthetic, before gates | 2/75 = 2.67% | 384/404 = 95.05% | 16/275 = 5.82% | 62/64 = 96.88% |
| Mixed, before gates | 2/75 = 2.67% | 21/404 = 5.20% | 127/275 = 46.18% | 21/64 = 32.81% |
| Either requested system, gated | 0/75 | 0/404 | 0/275 | 0/64 |

The natural absolute test contains 73 brightness and only two volume examples. Brightness covers 11 distinct test values and volume two; this does not establish all 101 values in human speech. Human development has 47 absolute brightness examples and three absolute volume examples. More natural numeric coverage is necessary. The 404 text-parser checks and 816 finite-decoder checks are software tests, not successful spoken examples.

## New wording and next study

[NEW-WORDS.md](../synthetic/NEW-WORDS.md) describes query-by-example enrollment: add a few labeled acoustic exemplars to a prototype bank, each bound to an existing intent and finite parameter value. A metric-trained encoder and calibrated similarity/margin rejection are needed before activation. This keeps semantic outputs fixed while permitting new wording. Text-only enrollment would instead need a validated phonetic encoder and reviewed pronunciations. Neither enrollment approach is implemented or validated here, and no new-word corpus, named device, artist or song test was added.

The next fixed-command study should broaden whole-utterance voice and wording diversity, independently review synthesized numbers and requested accents, improve original human numeric coverage, compare a jointly trained mixed model against the ensemble and both pure-source baselines, and use a development selection procedure that cannot discard a stronger incumbent unnoticed. These are follow-up hypotheses, not measurements from this run. Changes require a new version and preserved test evidence.

## Verification and boundaries

All 4,032 generated and 11,591 human recording hashes and split identities passed checks. ONNX variable-length/batch parity, finite decoding and existing runtime regression checks passed. Both training processes completed normally. File prediction emitted typed JSON or abstained, without device actions. These software results do not repair the failed acoustic quality gates.

With identical preloaded development PCM and two ONNX threads, synthetic inference was 5.03/6.98 ms p50/p95 and mixed was 7.18/9.76 ms. Session loading was 54.40 ms for synthetic and 147.37 ms summed for mixed. These include the frontend, inference and finite decoding; they exclude disk, speech duration, VAD and the 400 ms endpoint wait. No physical Pi or live-microphone performance was measured.

This is one seed and a bounded corpus subset, with silver human labels, unadjudicated TTS transcripts, two held-out synthetic voices and no Filipino-human benchmark. The installed artifacts and pipeline configuration are checked in `installed-preservation.json`. Neither candidate is approved for installation or shipping.
