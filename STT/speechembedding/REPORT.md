# Speech embeddings for dynamic music-name catalogs

20 September 2026. Research-only, original human recordings, random initialization.

**Recommendation: use a pronunciation-aware catalog reader for text-only additions. These new compact embeddings demonstrate the mechanism, but their failed name-quality gates prevent deployment. BC-ResNet is a promising alternative audio backbone; its fixed-word paper does not establish dynamic artist recognition.**

Entry point: [experiments/speechembedding](../../experiments/speechembedding/README.md). Paper investigation: [BC-ResNet](BCRESNET.md). Model/dataset rationale: [research notes](RESEARCH.md).

## What was built

Two candidates share a temporal depthwise residual acoustic encoder and a separate convolutional pronunciation encoder. Each returns a normalized 96-dimensional vector. Four ordered temporal bins preserve coarse sequence order; ten fixed feature windows offer weak name localization within an intact command. The model never generates a transcript. A catalog is embedded once and queried with cosine similarity. Windows are not verified word boundaries.

`dual` learns audio/pronunciation contrastive matching. `dual_kws` adds a fixed-label keyword loss. Both use the same seed, initial weights and sampling recipe, so the keyword loss is the intended ablation. Duplicate phrases are positives rather than in-batch negatives. All acoustic and pronunciation-encoder weights start randomly. CMUdict supplies only textual pronunciations. No pretrained weights, TTS, generated commands, splicing or acoustic augmentation are used.

The study additionally compares whole-command versus window matching, one/three/five-speaker audio prototypes, and a predeclared 50:50 audio/text score where an audio prototype exists. Audio enrollment embeds whole original commands because reliable name-only timestamps are unavailable. Missing audio retains text only in the explicitly selected hybrid experiment.

## Data and evaluation boundaries

The exact earlier parameter-study corpus is frozen again: 13,762 train, 412 enrollment, 2,537 development, 2,602 test recordings. There are 166 artist and 99 song training targets, versus 6,341 number targets. Test music targets are 77 artists and 45 songs; reserved unseen-name targets are 16 artists and only one song. These are curated public-source subsets, not full published-dataset benchmark scores.

Training/enrollment speakers are disjoint from development and test, and file hashes are deduplicated. Enrollment may use training speakers; reserved-name enrollment recordings never enter gradient updates. The prior test recordings have already been reported in earlier studies, so this is fresh evaluation on a preserved holdout, not a newly blind test. No test labels or errors select these weights, thresholds or catalog lists.

Slot roles are supplied from annotations. This measures conditional name/parameter retrieval, not automatic artist-versus-title parsing or complete-command accuracy. The data contain artist/song names, not a validated collection of custom playlist names. Small “playlist” catalogs are offline fixtures selected by training frequency. There are no Spotify requests, user-library accesses, playback actions or Home Assistant operations.

The development-selected text reader is **dual/text_global/expanded**. Each model/catalog/role needs at least 95% development precision over at least ten emissions and at most 1% unknown false accepts. Failure disables that role. These thresholds are operating-point estimates, not statistical guarantees; cosine scores are not probabilities. Weights, graphs, catalogs and thresholds were written to `selection-before-test.json` before test inference.

## Same-query comparison: expanded catalogs

All rows below score the same 77 artist and 45 song references against 127 artist / 109 song catalog entries. Historical rows are imported from preserved predictions and their recording/reference identities are checked. Phonetic v2 had additional real-English training, so it is a practical comparator rather than a controlled encoder-only ablation.

| Reader | Artists: raw exact | Songs: raw exact | Accepted artist / song correct |
| --- | ---: | ---: | ---: |
| Earlier fixed KWS (preserved) | 0/77 (0.0%) | 1/45 (2.2%) | See preserved gates; name roles disabled |
| Earlier audio enrollment, 5 speakers (preserved) | 4/77 (5.2%) | 1/45 (2.2%) | See preserved gates; name roles disabled |
| Earlier phoneme greedy matcher (preserved) | 5/77 (6.5%) | 6/45 (13.3%) | See preserved gates; name roles disabled |
| Phonetic v2 wide phrase graph (preserved) | 36/77 (46.8%) | 24/45 (53.3%) | See preserved gates; name roles disabled |
| dual / text_local | 2/77 (2.6%) | 6/45 (13.3%) | 0 / 0 |
| dual / text_global | 4/77 (5.2%) | 5/45 (11.1%) | 0 / 0 |
| dual_kws / text_local | 12/77 (15.6%) | 8/45 (17.8%) | 0 / 0 |
| dual_kws / text_global | 6/77 (7.8%) | 8/45 (17.8%) | 0 / 0 |
| dual_kws / kws | 8/77 (10.4%) | 2/45 (4.4%) | 0 / 0 |
| dual_kws / audio_1 | 7/77 (9.1%) | 3/45 (6.7%) | 0 / 0 |
| dual_kws / audio_3 | 9/77 (11.7%) | 6/45 (13.3%) | 0 / 0 |
| dual_kws / audio_5 | 11/77 (14.3%) | 6/45 (13.3%) | 0 / 0 |
| dual_kws / hybrid_3 | 10/77 (13.0%) | 8/45 (17.8%) | 0 / 0 |

Top-five retrieval is diagnostic only; returning a candidate list does not establish a correct accepted command.

| Preferred reader role | Top 1 | Top 5 | Representable references | Unknown false accepts / trials, gated |
| --- | ---: | ---: | ---: | ---: |
| artist | 4/77 (5.2%) | 8/77 (10.4%) | 60/77 | 0/320 |
| song | 5/45 (11.1%) | 7/45 (15.6%) | 44/45 | 0/351 |
| device | 19/79 (24.1%) | 32/79 (40.5%) | 79/79 | 0/300 |

A zero false-accept count alongside zero recall means the role is disabled; it is not a successful recognizer. Unknown trials include ordinary speech and commands for the other music role. The same background recordings are repeated across roles and are not independent new recordings.

## Adding unseen names and optional recordings

| Reader, expanded catalog | Reserved artists | Reserved songs | Accepted correct, both roles |
| --- | ---: | ---: | ---: |
| dual/text_local/expanded | 0/16 (0.0%) | 0/1 (0.0%) | 0 |
| dual_kws/text_local/expanded | 0/16 (0.0%) | 0/1 (0.0%) | 0 |
| dual_kws/audio_1/expanded | 4/16 (25.0%) | 1/1 (100.0%) | 0 |
| dual_kws/audio_3/expanded | 3/16 (18.8%) | 1/1 (100.0%) | 0 |
| dual_kws/audio_5/expanded | 5/16 (31.2%) | 1/1 (100.0%) | 0 |
| dual_kws/hybrid_3/expanded | 0/16 (0.0%) | 1/1 (100.0%) | 0 |

These names were excluded from all gradient training, including transcript matches. Adding catalog text changes the text vectors, not acoustic classes. An unavailable dictionary pronunciation remains unavailable unless explicitly supplied in the existing phone vocabulary. Audio-only matching can represent names with recordings but cannot create prototypes from text. One held-out song is far too little evidence for generalization.

The reserved-name set above is distinct from naturally unseen names that occur only outside training. The preferred text reader gives:

| Role | Names seen in training: raw exact queries | All names unseen in training: raw exact queries |
| --- | ---: | ---: |
| artist | 4/33 (12.1%) | 0/44 (0.0%) |
| song | 0/19 (0.0%) | 5/26 (19.2%) |

| Enrollment request | Artists with zero / fewer than requested speakers | Songs with zero / fewer than requested speakers |
| --- | ---: | ---: |
| 1 | 44 / 44 of 127 | 36 / 36 of 109 |
| 3 | 44 / 107 of 127 | 36 / 99 of 109 |
| 5 | 44 / 119 of 127 | 36 / 108 of 109 |

## Smaller playlist fixtures

These lists are fixed before test from training frequency. An omitted name is an unknown-name trial; the evaluator never inserts the answer into a per-query shortlist.

| Catalog / role, preferred text mode | Catalog entries | References inside catalog | Raw correct / inside catalog | Gated absent-name false accepts |
| --- | ---: | ---: | ---: | ---: |
| playlist10 / artist | 10 | 16/77 | 5/16 (31.2%) | 0/61 |
| playlist10 / song | 10 | 7/45 | 0/7 (0.0%) | 0/38 |
| playlist25 / artist | 25 | 17/77 | 5/17 (29.4%) | 0/60 |
| playlist25 / song | 25 | 11/45 | 0/11 (0.0%) | 0/34 |
| fixed / artist | 70 | 33/77 | 5/33 (15.2%) | 0/44 |
| fixed / song | 66 | 19/45 | 0/19 (0.0%) | 0/26 |
| expanded / artist | 127 | 77/77 | 4/77 (5.2%) | 0/0 |
| expanded / song | 109 | 45/45 | 5/45 (11.1%) | 0/0 |

## Other roles remain separate

| Preferred reader group | Raw exact | Accepted correct |
| --- | ---: | ---: |
| isolated_number | 976/1272 (76.7%) | 768/1272 |
| natural_number | 0/10 (0.0%) | 0/10 |
| word | 143/145 (98.6%) | 140/145 |
| device | 19/79 (24.1%) | 0/79 |

Number labels are included to expose corpus imbalance, not to claim a new numeric grammar. The reader does not implement connected volume or brightness validation. Room/place labels remain device-name proxies. Artist, song, number and room results must not be pooled into one headline accuracy.

## Model size and measured cost

| Model | Total parameters | Audio / text ONNX bytes | Selected epoch / epochs run | Fit seconds |
| --- | ---: | ---: | ---: | ---: |
| dual | 211,542 | 656,095 / 251,909 | 14 / 23 | 204.9 |
| dual_kws | 211,542 | 656,095 / 251,909 | 20 / 28 | 767.0 |

Timing uses identical preloaded PCM from 48 original music commands, CPU ONNX with two intra-op threads, one inter-op thread and one BLAS thread. One warmup pass and three measured passes include frontend, acoustic model and matching; disk, speech duration, session/catalog construction and the default 400 ms endpoint wait are excluded. These are Windows workstation measurements, not Pi or live-microphone results.

| Reader | p50 / p95 ms | Session load ms | Catalog construction ms |
| --- | ---: | ---: | ---: |
| dual/text_local | 11.93 / 19.24 | 47.68 | 759.56 |
| dual/text_global | 13.15 / 17.48 | 47.68 | 701.84 |
| dual_kws/text_local | 13.12 / 16.97 | 34.38 | 910.62 |
| dual_kws/kws | 12.72 / 16.12 | 34.38 | 853.84 |
| dual_kws/audio_3 | 12.70 / 16.66 | 34.38 | 943.99 |
| dual_kws/hybrid_3 | 13.05 / 17.24 | 34.38 | 910.67 |
| historical/fixed | 13.11 / 17.35 | 49.75 | 838.28 |
| historical/prototype | 12.57 / 16.13 | 45.15 | 830.69 |
| historical/phonetic_wide_graph | 18.15 / 26.23 | 46.26 | 953.50 |

Text graph inference occurs at catalog construction, not on every utterance. Reported construction includes pronunciation lookup and text-graph session setup. Separate catalog-only scaling uses generated dictionary **text** distractors, not generated speech or a large-catalog accuracy benchmark.

| Names | Matching p50 / p95 ms | Catalog construction ms | Vector matrix bytes |
| --- | ---: | ---: | ---: |
| 100 | 0.033 / 0.033 | 912.9 | 38,400 |
| 1000 | 0.200 / 0.264 | 1018.1 | 384,000 |
| 5000 | 1.943 / 2.615 | 1706.9 | 1,920,000 |

## Interpretation and next experiment

This study does not identify a universally best model. It tests a compact, single-seed, from-scratch recipe with very sparse name supervision. Its acoustic and text spaces can be compared, but simple cosine retrieval remains vulnerable to surrounding command words, accent variation, phonetic neighbors and insufficient training coverage. Those are plausible failure contributors, not experimentally separated causal estimates. Dictionary/enrollment coverage and per-query outputs are saved so data gaps remain visible.

For text-only music catalogs, the next architecture to test is a compact phonetic or audio/text keyword reader with a sequence verifier. Use a cheap embedding search to retrieve candidates, then verify the candidate phone sequence against local acoustic frames. BC-ResNet can be tested as an alternative acoustic backbone inside that system. A fixed KWS head remains useful for stable command words, while VCM retains intent ownership. Optional original recordings can provide additional pronunciation evidence for unusual names. Neither pooled word accuracy nor a fixed-word paper result establishes open-name recognition.

The next version should first add original connected artist/song/playlist recordings from new speakers, including confusable names, accents, background speech and names absent from the active catalog. Do not tune this version on its test errors. Evaluate a new, preserved protocol with several seeds and enough positive/unknown trials per role.

## Verification and artifacts

ONNX parity covers variable durations, batch sizes and padding. Original file hashes, preprocessing parity, duplicate-positive contrastive loss, homophones, explicit pronunciations, stable IDs, rejected intent, invalid thresholds and catalog-version invalidation are checked separately from acoustic accuracy. The adapter returns research diagnostics and a typed decision; failed gates abstain. Its caller must supply an already accepted MEDIA_PLAY intent. No device actions are executed.

An initial frontend check failed a tight tolerance with the default 24 BLAS threads (maximum absolute difference 0.0000019223). The diagnostic reproduced exact equality with the cached training features at one BLAS thread. The entry point now pins one BLAS thread for calibration, test, timing and file queries; the original feature cache and trained weights were not changed. See `feature-parity-diagnostic.json` and the preserved initial check log.

Candidate checkpoints, ONNX graphs, fitting histories, development/test predictions, enrollment IDs, selection lock, checks and timing are in [.cache/speechembedding-v1](../../.cache/speechembedding-v1/). The frozen manifest, protocol and catalogs are in [data/speechembedding-v1](../../data/speechembedding-v1/). Implementation is registered as `runtime: false` in the shared source map. Installed VCM/STT files, configuration and previous study results remain hash-identical.
