# Human-only speech parameter alternatives

19 September 2026. Final audited feasibility experiment.

**Recommendation: retain VCM for intent and stable finite controls; treat expanding names as a separate parameter-reading problem. None of these four small, from-scratch candidates establishes reliable natural numeric commands or arbitrary music/device names.** The code now retires the legacy convolutional ASR model. The installed experimental word-STT brightness checkpoint remains unchanged; the four candidates are isolated and were not promoted.

A vocabulary list is a label specification, not acoustic training data. A fixed classifier needs recordings and new output weights for new classes. Audio enrollment can add a name from real recordings; a phonetic or character reader can use an updated text catalog without changing its acoustic weights, but only if it already recognizes the relevant speech well.

## Four approaches actually trained and tested

| Approach | Parameters | FP32 ONNX bytes | Selected epoch / epochs run | CPU p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Fixed KWS / direct slot head | 152,262 | 628,372 | 27 / 28 | 3.50 / 6.28 ms |
| Audio-enrollment embedding | 152,262 | 628,372 | 19 / 28 | 3.54 / 6.34 ms |
| Phoneme CTC + keyword matching | 186,665 | 756,723 | 23 / 28 | 3.75 / 7.24 ms |
| Compact character CTC STT | 185,117 | 750,528 | 26 / 28 | 3.92 / 7.86 ms |

Fixed KWS uses a temporal depthwise residual encoder and a fixed softmax label set. Audio enrollment uses a separately initialized copy of that encoder with classification plus supervised contrastive loss, then normalized mean prototypes. Phoneme and character models use separate width-64, two-layer bidirectional GRU CTC networks with convolutional inputs. The phonetic reader uses CMUdict pronunciations, greedy CTC and exact substring edit-distance matching; the character reader applies the same matching idea to decoded text. These are small feasibility implementations, not reproductions of published SOTA scores.

All acoustic weights started randomly, using one seed per approach (31–34). No legacy weights, pretrained acoustic encoder, cloud service, LLM, TTS or joined-word audio was used. The CTC models received three isolated-word warmup epochs, then the mixed human corpus. Maximum fitting was 28 epochs, with development selection and patience 9. This limited-data pilot is not a hyperparameter sweep or general-English pretraining.

## Exact parameter recognition on original human recordings

These are **raw value/name matches before confidence gating**, including abstentions in the denominator. The slot role is supplied from annotations, so this is a conditional parameter test. It does not establish automatic artist-versus-title selection, word boundaries, complete-command correctness or end-to-end VCM accuracy. Room/place entities are proxies for Home Assistant names, not the user’s actual device inventory.

| Approach | Isolated number words | Connected numeric commands | Artists | Songs | Room/place names |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed KWS / direct slot head | 1044/1272 (82.1%) | 0/10 (0.0%) | 0/77 (0.0%) | 1/45 (2.2%) | 3/79 (3.8%) |
| Audio-enrollment embedding | 774/1272 (60.8%) | 1/10 (10.0%) | 4/77 (5.2%) | 1/45 (2.2%) | 7/79 (8.9%) |
| Phoneme CTC + keyword matching | 1063/1272 (83.6%) | 1/10 (10.0%) | 5/77 (6.5%) | 6/45 (13.3%) | 16/79 (20.3%) |
| Compact character CTC STT | 1067/1272 (83.9%) | 1/10 (10.0%) | 2/77 (2.6%) | 1/45 (2.2%) | 6/79 (7.6%) |

Only **10 connected numeric commands** survive the frozen test protocol, covering values 0, 2, 7, 8, 12, 20, 50. There are many more isolated number words. These results cannot establish all 101 values, optional-percent robustness, coarticulation or microphone reliability. Counting successful text parsing of `brightness 50%` would not fill that acoustic gap.

Per-role thresholds were chosen on development data for at least 95% precision and 10 emissions. A role that could not meet that rule was disabled with an unreachable threshold. The rule is an operating-point heuristic, not a statistical precision guarantee; the test checks whether it transfers.

| Approach | Accepted isolated numbers / all | Connected numeric values accepted correctly | Artist/song/device roles passing development gate | All accepted parameters: correct / emitted |
| --- | ---: | ---: | --- | ---: |
| Fixed KWS / direct slot head | 873/1272 (68.6%) | 0/10 (0.0%) | None | 1012/1079 (93.8%) |
| Audio-enrollment embedding | 70/1272 (5.5%) | 0/10 (0.0%) | None | 147/166 (88.6%) |
| Phoneme CTC + keyword matching | 73/1272 (5.7%) | 0/10 (0.0%) | None | 114/120 (95.0%) |
| Compact character CTC STT | 658/1272 (51.7%) | 0/10 (0.0%) | None | 691/732 (94.4%) |

The pooled accepted-parameter column is dominated by number/context words and must not be interpreted as name accuracy. The 300 original background recordings are queried separately for each of five roles, producing 1,500 negative role trials, not 1,500 independent recordings.

| Approach | Raw / gated false number matches on 300 backgrounds | Raw / gated false artist matches on 300 backgrounds |
| --- | ---: | ---: |
| Fixed KWS / direct slot head | 9 / 1 | 5 / 0 |
| Audio-enrollment embedding | 300 / 0 | 300 / 0 |
| Phoneme CTC + keyword matching | 200 / 0 | 187 / 0 |
| Compact character CTC STT | 247 / 0 | 213 / 0 |

The high raw false-match rates of nearest-prototype and fuzzy catalog matching show why “choose the nearest name” is insufficient. Gating suppresses errors but also rejects useful speech. No failed gate was relaxed to improve the reported recall.

## Updating a corpus without retraining

The fixed fixture contains 70 artists, 66 songs and 24 room/place labels. The expanded fixture contains 127, 109 and 41, respectively. Both have the 0–100 text grammar. These lists were declared from the dataset ontology before fitting, not customized to each query. No Spotify/playlist/filepath integration was implemented.

19 entity labels were reserved using training counts and a deterministic hash. All training utterances containing those phrases or carrying those entity labels were withheld from gradient updates; their original audio could be used only for enrollment. Enrollment and test speakers are disjoint. Zero reserved labels overlap the final acoustic training classes.

| Method | Raw correct reserved-name queries without update | Raw correct after update | Correct accepted after update |
| --- | ---: | ---: | ---: |
| Fixed KWS / direct slot head | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| Audio-enrollment embedding (up to 1 enrollment speakers) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) |
| Audio-enrollment embedding (up to 3 enrollment speakers) | 0/25 (0.0%) | 3/25 (12.0%) | 0/25 (0.0%) |
| Audio-enrollment embedding (up to 5 enrollment speakers) | 0/25 (0.0%) | 5/25 (20.0%) | 0/25 (0.0%) |
| Phoneme CTC + keyword matching | 0/25 (0.0%) | 3/25 (12.0%) | 0/25 (0.0%) |
| Compact character CTC STT | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) |

The reserved-name test consists of 16 artist, 1 song and 8 room/place query occurrences. It is too small for a general music-name claim, especially for songs. Enrollment uses up to the requested number of distinct available speakers; some labels have fewer. Whole original commands are embedded because no verified name-only audio boundaries are available.

The fixed classifier’s acoustic class set cannot expand by editing JSON. The audio-enrollment mechanism works without acoustic retraining but requires recordings. Phoneme/character catalog changes also leave the acoustic graph unchanged. Those mechanisms are technically feasible; their recognition quality here is insufficient.

The phonetic catalog has 51 unpronounceable-by-this-dictionary entries in the expanded fixture. Proper names, spelling variants, tokenization and foreign names need explicit pronunciations or a separately tested G2P model. Dictionary presence still does not imply that the acoustic model recognizes a name. Homophone/tied matches abstain.

## Does VCM suffice?

For stable intent families and a finite vocabulary, **yes in principle**. The current VCM already has 103 amount outputs: NONE, OTHER and every integer 0–100. However, all 101 numeric value thresholds are 1.1, so its current policy disables them. More text labels will not fix that. Proper connected human training and per-value calibration could make a finite numeric head useful without general transcription.

For names, its target head has 11 entries including NONE/OTHER, and its media head is only NONE/OTHER/GENERIC. A newly added artist or device ID is not a new acoustic output. Keep VCM for intent, and augment the parameter stage for changing catalogs.

Recommended architecture:

```text
PCM → existing VAD → VCM accepts intent
    → fixed finite heads for supported stable controls
    → otherwise an approved parameter reader + versioned catalog/grammar
       → typed percent/delta or entity ID, calibrated confidence, or abstention
    → shared validation → command JSON
```

A future music grammar must resolve artist/title roles and multiple slots from the utterance rather than receiving the annotation role used by this study. Percentages need the explicit 0–100 grammar, distinction between absolute and relative changes, and rejection of conflicting values. A reader must never change VCM intent or execute device actions.

My next architecture choice is **small phonetic or learned audio/text KWS with a command grammar**, after expanding real phrase-level training. For user-enrolled aliases, keep audio-template/embedding enrollment as a separate option. A fifth researched approach, [OHF Speech-to-Phrase](https://github.com/OHF-voice/speech-to-phrase), shows how command templates and device names can form a constrained grammar; its downloadable models were not trained or installed here. See [the dataset/model survey](DATASETS_AND_MODELS.md) for DS-CNN, TC-ResNet, BC-ResNet, MatchboxNet, few-shot KWS and audio/text alternatives.

## Dataset, preprocessing and audit

19,313 original human recordings were collated. Training uses 13,762 (6.90 hours), development 2,537 (1.28 hours), test 2,602 (1.22 hours), and held-out enrollment 412 (0.30 hours). Not every recording carries a studied parameter. The feature protocol accepts 0.1–8 seconds, converts to mono 16 kHz and uses per-utterance 80-bin CMVN. There is no audio augmentation or synthetic composition.

| Human source | Train | Enrollment | Dev | Test |
| --- | ---: | ---: | ---: | ---: |
| FSDD | 2000 | 0 | 500 | 500 |
| FluentSpeechCommands | 2172 | 273 | 317 | 350 |
| GoogleSpeechCommandsV2 | 2812 | 0 | 311 | 461 |
| LibriSpeech | 28 | 0 | 0 | 0 |
| MLEndSpokenNumerals | 1124 | 0 | 290 | 294 |
| MSWC-English | 985 | 0 | 176 | 162 |
| SLURP-real | 4362 | 129 | 939 | 831 |
| SnipsSmartLights | 279 | 10 | 4 | 4 |

The corpus retains all chosen parameter examples and a deterministic bounded sample of other speech. It is heavily weighted toward isolated numeric words; only a small portion is general English or individual music names. Root calibration recordings were excluded. Speaker/recording hashes were audited, paired microphones were reduced to one recording and conflicting duplicates were quarantined. MLEnd targets retain their canonical-category caveat. Public data and matching metadata can still contain errors; these are not manually reviewed ground-truth command labels.

A first pilot exposed a source inconsistency: a birthday reminder had unrelated music entity annotations. A uniform punctuation-insensitive entity/transcript consistency rule quarantined three records across all splits. Reserved-name exclusion now also checks explicit entity labels. All four models were retrained from random weights with unchanged seeds and recipes. The superseded pilot is retained in `.cache/parameter-study/superseded-v1-annotation-audit/`. Final test scoring is a rerun after this data-quality repair, **not a newly collected blind test**; no test error was used to choose hyperparameters or confidence thresholds.

Initial GPU evaluation also failed a tight confidence parity check against CPU. Final development calibration and test scoring use the actual CPU ONNX engine, with 96 matched single-versus-batched task checks per model. Variable-length Torch/ONNX and padding parity are recorded in each training JSON. Model/threshold/catalog choices are persisted before final test scoring. This avoids reporting CUDA confidence estimates as deployed CPU behavior.

## Recognition cost and expanding-list cost

Main timing uses the same 96 original held-out waveforms for each candidate: one warmup pass, three measured passes, CPUExecutionProvider, two intra-op threads and one inter-op thread. It includes the frontend, acoustic inference, decoding and catalog matching; excludes file reads, speech duration, capture and the default 400 ms VAD endpoint wait. The model graph itself takes less time than the complete recognition path. These are Windows workstation measurements, not Raspberry Pi or live-microphone results.

| Reader | 100 text names p50 / p95 | 1,000 names | 5,000 names |
| --- | ---: | ---: | ---: |
| Phoneme CTC + keyword matching | 1.59 / 2.19 ms | 8.26 / 13.92 ms | 38.16 / 66.31 ms |
| Compact character CTC STT | 1.74 / 2.30 ms | 7.78 / 12.28 ms | 33.52 / 55.15 ms |

Scaling is a separate timing-only test on eight original artist-query recordings with deterministic generated **text** distractors. It generates no audio and establishes no large-catalog accuracy. The simple matching implementation scans candidates; a large playlist needs a better index/grammar or learned retrieval and new false-accept calibration. An audio-enrollment catalog cannot acquire new acoustic prototypes from text alone.

The retired model had 1,321,757 parameters and a 5,319,980-byte graph; the new candidate graphs are substantially smaller. Its old latency numbers used a different, constructed-command test set, so they are not a valid matched speed comparison against this human-only test. It was not restored for this study. Smaller graphs alone are not evidence of reliable or faster end-to-end commands.

## Runtime changes and verification

Removed the retired model implementation, continuation trainer, runtime loader, legacy switches/custom-media fallback, active checkpoint/graph and ASR/dual-model release ZIPs. The transcription dataset and audio remain. New brightness preparation is human-only under `data/stt-human/`; the old constructed dataset/report are historical. `training.update_artifacts=false` prevents automatic replacement by new experiments. The installed VCM and word-STT graph hashes are unchanged.

Runtime component, VAD/capture simulation, intent ownership, retirement and brightness grammar checks passed, including 404 numeric parser forms. Full existing validation audited 39,951 original recordings and replayed 4,314 held-out examples. Its quality gate remains **failed**: the current full pipeline emitted 87/91 correct complete finite commands, with 1783 finite references. These are existing-system replay results, separate from the new parameter pilot. No live mic, physical Pi, Spotify or Home Assistant test was performed.

## Reproduce and extend

```powershell
.\.venv\Scripts\python.exe scripts/parameter_study.py prepare
.\.venv\Scripts\python.exe scripts/parameter_study.py train --epochs 28
.\.venv\Scripts\python.exe scripts/parameter_study.py evaluate
.\.venv\Scripts\python.exe scripts/parameter_study.py scale
.\.venv\Scripts\python.exe STT/alternatives/write_report.py
.\.venv\Scripts\python.exe train.py --tests-only --no-setup
.\.venv\Scripts\python.exe publish.py
```

Use the source map and contracts in [AGENTS.md](../../AGENTS.md). `--models fixed prototype phoneme character` selects training candidates. Frozen manifests/catalogs are in `data/parameter-study/`; checkpoints, graphs, per-query predictions, per-value results, training histories, selection locks and timings are in `.cache/parameter-study/`. Exact provenance is recorded in `audit.json`, the pronunciation receipt and `reproducibility.json`.

The next data priority is a larger, speaker-stratified connected-command benchmark using more available natural commands, followed by real recordings for missing 0–100 values and confusable/out-of-range negatives. This study’s deterministic split placed most SmartLights examples in training, so its small numeric holdout must not be confused with the full public corpus. MSWC/LibriPhrase and a larger real English phonetic pool can then strengthen representations. Add independent human names in natural commands and real enrollment examples for device aliases. Synthetic speech for rare names/numbers is a later, separately labeled roadmap experiment; it must keep the real-human test untouched. The [survey](DATASETS_AND_MODELS.md) distinguishes locally used datasets, surveyed sources, access-gated data and excluded synthetic corpora.

No candidate was promoted on these results. The best approach depends on the contract: finite KWS for stable controls, audio enrollment when recordings can be supplied, and phonetic/audio-text matching plus a grammar when text catalogs must expand. The present bare datasets do not yet demonstrate the desired natural-command reliability.
