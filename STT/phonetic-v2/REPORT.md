# Phonetic keyword reader and constrained grammar

19 September 2026. Follow-up to the four-approach human-speech study.

**Implemented and measured in isolation. The shipping VCM/STT graphs, default, configuration and prior results are unchanged.** This report separates recognition quality from software checks. New reader outputs are conditional on an accepted intent and slot role; they are not complete-command or full-pipeline accuracy.

**The best observed raw absolute-brightness result is 36/52 (69.2%), at 60.20 ms median CPU time.** The same acoustic model with greedy matching gets 12/52 (23.1%) at 7.35 ms. No combination passes the connected-number or name development gate, so the gated readers abstain on those roles. No candidate is ready to replace the installed reader.

Text-catalog expansion improves the larger graph reader's reserved-name retrieval from 0/25 (0.0%) to 14/25 (56.0%) without acoustic retraining. None is accepted after confidence gating. This demonstrates the update mechanism, while leaving recognition reliability unresolved.

## What was implemented

A small phoneme CTC acoustic model feeds a shared-prefix Viterbi phrase graph. Brightness uses a bounded sentence grammar with 0–100 values and optional percent. Artists, songs and room-name proxies use keyword/filler phone graphs. The graph retains frame scores, unlike the earlier greedy-phone/edit matcher. A versioned catalog provides stable entity IDs, aliases and optional explicit pronunciations. Adding names recompiles that graph without altering acoustic weights; a changed catalog invalidates the adapter's calibration. Entries without a usable pronunciation are excluded. Tied homophones, unsupported intent/role pairs and disabled confidence gates abstain.

The typed adapter preserves VCM intent and returns a percent or entity ID; it executes no action. Only absolute brightness has a connected numeric grammar here. This is an independent phonetic speech-recognition implementation, not a claim that renaming ASR eliminates speech-recognition work. The retired model was not restored. No Spotify, filepath source connector or Home Assistant integration was implemented.

## Real data and split

The publisher-listed LibriSpeech train-clean-100 archive was downloaded and verified: 6,387,309,499 bytes, MD5 `2a93770f6d5c6c964bc36631d331a522`, SHA-256 `d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2`. [LibriSpeech](https://www.openslr.org/12/) provides the human recordings and [publisher checksums](https://www.openslr.org/resources/12/md5sum.txt). This run uses a bounded selection of its original utterances, not the entire official benchmark.

The frozen manifest has **29,307 original recordings**; no generated or joined audio. The manifest assigns 37.06 hours to training, 2.36 to development and 2.42 to test. General English is capped at 30 additional training hours and one hour each for development/test. Per-utterance 80-bin CMVN features are float32, with mono conversion/resampling; training English may last up to 20 seconds, while command audio retains the eight-second protocol.

| Source | Train | Enroll | Dev | Test |
| --- | ---: | ---: | ---: | ---: |
| FSDD | 2000 | 0 | 500 | 500 |
| FluentSpeechCommands | 2172 | 273 | 317 | 350 |
| GoogleSpeechCommandsV2 | 2812 | 0 | 311 | 461 |
| LibriSpeech | 28 | 0 | 0 | 0 |
| LibriSpeech-expanded | 8839 | 0 | 290 | 289 |
| MLEndSpokenNumerals | 1124 | 0 | 290 | 294 |
| MSWC-English | 985 | 0 | 176 | 162 |
| SLURP-real | 4362 | 129 | 939 | 831 |
| SnipsSmartLights | 485 | 25 | 102 | 261 |

Annotated speaker-ID intersections: 0; reserved-name training overlap: 0; previous test recordings preserved: 2,602. All original file hashes are checked before reuse.

The supplemental SmartLights holdout selects two well-covered speakers using training-side transcript counts and a fixed hash, one for development and one for test. These speakers were previously seen by historical models, so **only new v2 models may be evaluated on the supplemental holdout**. The old ten-command subset is retained and reported separately for a fair historical comparison. This is a protocol revision using existing recordings, not newly collected blind speech.

| Split | Connected numeric examples | Distinct values | Values covered |
| --- | ---: | ---: | --- |
| train | 123 | 15 | 0, 1, 2, 5, 8, 10, 12, 20, 22, 39, 50, 55, 70, 73, 90 |
| dev | 38 | 14 | 2, 5, 7, 10, 12, 20, 22, 39, 40, 50, 55, 73, 80, 90 |
| test | 60 | 12 | 0, 2, 5, 7, 8, 10, 12, 20, 22, 39, 50, 73 |

These counts do not establish 101-value natural-speech coverage. MLEnd remains canonical word labeling; room/place annotations are proxies for Home Assistant names. Source transcriptions and entities remain publisher/programmatic labels, not newly human-adjudicated truth. [Sonos SmartLights](https://github.com/sonos/spoken-language-understanding-research-datasets) has research-only source terms. [LibriPhrase](https://github.com/gusrud1103/LibriPhrase) was reviewed again; the published test package was kept out of training.

## Random-initialization training

| Model | Parameters | ONNX bytes | Best epoch / run | Development phone error: word / command / general |
| --- | ---: | ---: | ---: | --- |
| small | 186,665 | 756,723 | 29 / 36 | 12.1% / 43.5% / 38.1% |
| wide | 387,497 | 1,560,051 | 33 / 36 | 10.6% / 38.7% / 32.6% |

Both use two GRU layers, a two-epoch word warmup, maximum 36 epochs, fourfold sampling of natural parameter commands, and mean development phone error across three groups for checkpoint selection. No pretrained acoustic weights are loaded. Transcripts without a complete dictionary pronunciation, or too many target phones for their frame count, are excluded from CTC fitting; the two runs each used 22,583 eligible training recordings. Held-out name-enrollment rows are not used by these text-catalog models. The width comparison uses one distinct seed per model, so it is not a multi-seed estimate of a pure capacity effect.

## Exact matches before confidence gating

| Candidate | Isolated numbers | Connected numbers | Artists | Songs | Room/place names |
| --- | ---: | ---: | ---: | ---: | ---: |
| Width 64 + greedy matching | 1070/1272 (84.1%) | 16/60 (26.7%) | 11/77 (14.3%) | 14/45 (31.1%) | 32/79 (40.5%) |
| Width 64 + CTC graph | 1107/1272 (87.0%) | 38/60 (63.3%) | 27/77 (35.1%) | 21/45 (46.7%) | 51/79 (64.6%) |
| Width 96 + greedy matching | 1076/1272 (84.6%) | 14/60 (23.3%) | 16/77 (20.8%) | 12/45 (26.7%) | 38/79 (48.1%) |
| Width 96 + CTC graph | 1117/1272 (87.8%) | 40/60 (66.7%) | 36/77 (46.8%) | 24/45 (53.3%) | 41/79 (51.9%) |

The connected-number pool contains 52 text-valid absolute brightness requests and eight other numeric utterances (volume, relative changes or unsupported intent/grammar). The table above preserves the broad conditional number task; it does not mean the adapter supports all those operations. The brightness-specific view is:

| Candidate | Absolute brightness value: raw | Correct after the connected-number gate |
| --- | ---: | ---: |
| Width 64 + greedy matching | 14/52 (26.9%) | 0/52 (0.0%) |
| Width 64 + CTC graph | 33/52 (63.5%) | 0/52 (0.0%) |
| Width 96 + greedy matching | 12/52 (23.1%) | 0/52 (0.0%) |
| Width 96 + CTC graph | 36/52 (69.2%) | 0/52 (0.0%) |

## Confidence-gated performance

Each group/catalog/decoder has its own development threshold: at least 95% observed precision and ten emissions, otherwise disabled. Connected numbers have a separate gate from isolated numbers. These development rules are not a statistical guarantee. All choices were locked before test inference.

| Candidate | Isolated numbers accepted correctly | Connected numbers accepted correctly | Artist / song / room accepted correctly | Raw / gated false number matches on 300 backgrounds |
| --- | ---: | ---: | --- | ---: |
| Width 64 + greedy matching | 720/1272 (56.6%) | 0/60 (0.0%) | 0/77 (0.0%) / 0/45 (0.0%) / 0/79 (0.0%) | 213 / 0 |
| Width 64 + CTC graph | 983/1272 (77.3%) | 0/60 (0.0%) | 0/77 (0.0%) / 0/45 (0.0%) / 0/79 (0.0%) | 300 / 0 |
| Width 96 + greedy matching | 760/1272 (59.7%) | 0/60 (0.0%) | 0/77 (0.0%) / 0/45 (0.0%) / 0/79 (0.0%) | 201 / 0 |
| Width 96 + CTC graph | 1021/1272 (80.3%) | 0/60 (0.0%) | 0/77 (0.0%) / 0/45 (0.0%) / 0/79 (0.0%) | 300 / 0 |

| Candidate | Accepted isolated-number trials: correct / emitted | Test precision | Background false accepts / 300 |
| --- | ---: | ---: | ---: |
| Width 64 + greedy matching | 720/761 | 94.6% | 0/300 |
| Width 64 + CTC graph | 983/1050 | 93.6% | 0/300 |
| Width 96 + greedy matching | 760/790 | 96.2% | 1/300 |
| Width 96 + CTC graph | 1021/1099 | 92.9% | 0/300 |

Only the width-96 greedy decoder maintains at least 95% observed precision on accepted isolated-number test clips. The graph decoders improve raw number recognition but their isolated-number calibration does not transfer at that target. This isolated-word result does not establish connected brightness reliability.

Negatives are the same original recordings reused as separate role trials; they are not independent speech samples for each role. Both graph readers select an allowed numeric candidate on all 300 background queries before gating. The current score does not separate correct natural commands from those forced matches well enough. A stronger explicit unknown/background alternative and better connected-command evidence are needed; merely expanding the allowed grammar does not solve rejection. Full per-value, per-intent and unknown-role results and every prediction are in `.cache/phonetic-v2/`.

## Catalog expansion without acoustic retraining

| Candidate | Fixed catalog: reserved names raw | Expanded: reserved names raw | Expanded: gated | Expanded using stale fixed thresholds |
| --- | ---: | ---: | ---: | ---: |
| Width 64 + greedy matching | 0/25 (0.0%) | 3/25 (12.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| Width 64 + CTC graph | 0/25 (0.0%) | 10/25 (40.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| Width 96 + greedy matching | 0/25 (0.0%) | 5/25 (20.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| Width 96 + CTC graph | 0/25 (0.0%) | 14/25 (56.0%) | 0/25 (0.0%) | 0/25 (0.0%) |

Every fixed/expanded pair uses exactly the same acoustic graph. The stale-threshold ablation is reported for diagnosis; the adapter refuses to treat it as calibrated. Correct raw retrieval alone does not justify adding a reader to the pipeline. The small reserved-name sample still limits conclusions about arbitrary playlists.

The 25 reserved-name trials cover 11 distinct role/name pairs: 16 artist recordings, 1 song recording and 8 room/place recordings. The width-96 graph reader retrieves 9/16 artists, 1/1 songs and 4/8 rooms correctly before gating. One held-out song recording is insufficient to establish general playlist-title recognition.

| Candidate | Previously cataloged names: raw before / after expansion | Gated before / after |
| --- | --- | --- |
| Width 64 + greedy matching | 47/114 (41.2%) / 41/114 (36.0%) | 0/114 (0.0%) / 0/114 (0.0%) |
| Width 64 + CTC graph | 72/114 (63.2%) / 65/114 (57.0%) | 0/114 (0.0%) / 0/114 (0.0%) |
| Width 96 + greedy matching | 56/114 (49.1%) / 45/114 (39.5%) | 0/114 (0.0%) / 0/114 (0.0%) |
| Width 96 + CTC graph | 67/114 (58.8%) / 63/114 (55.3%) | 0/114 (0.0%) / 0/114 (0.0%) |

## Preserved legacy numeric comparison

The earlier phonetic reader correctly matched **1/10** connected numeric commands before gating and **0/10** after gating. On those same ten recordings, the new candidates give:

| Candidate | Raw | Gated |
| --- | ---: | ---: |
| Width 64 + greedy matching | 3/10 (30.0%) | 0/10 (0.0%) |
| Width 64 + CTC graph | 5/10 (50.0%) | 0/10 (0.0%) |
| Width 96 + greedy matching | 2/10 (20.0%) | 0/10 (0.0%) |
| Width 96 + CTC graph | 4/10 (40.0%) | 0/10 (0.0%) |

The historical model was not tested on the supplemental holdout, where its previous training would invalidate the comparison.

## CPU recognition cost

| Candidate | Mixed commands p50 / p95 | Absolute brightness p50 / p95 | Catalog compilation |
| --- | ---: | ---: | ---: |
| Width 64 + greedy matching | 5.13 / 8.48 ms | 5.53 / 8.36 ms | 739.7 ms |
| Width 64 + CTC graph | 53.66 / 77.97 ms | 59.91 / 89.12 ms | 129.9 ms |
| Width 96 + greedy matching | 6.79 / 10.89 ms | 7.35 / 10.83 ms | 909.1 ms |
| Width 96 + CTC graph | 46.51 / 79.33 ms | 60.20 / 90.76 ms | 245.3 ms |

Warm timing uses 86 identical preloaded natural-command recordings, including all 52 absolute brightness requests, one warmup and three measured passes, CPU ONNX with two intra-op/one inter-op threads and one BLAS thread. It includes frontend, graph inference and catalog/grammar decoding; excludes disk, utterance duration, cold construction and the 400 ms endpoint wait. This mixed-role timing is separate from the previous pilot's mostly-word timing, so those earlier latencies are not a matched comparison. These are workstation results, not physical Pi or live-microphone measurements.

Model session creation took 20.67 ms for the smaller model and 9.61 ms for the larger model. The catalog column measures object construction: the graph reader reuses an already loaded dictionary, while the unchanged greedy constructor rereads it. Those setup figures are implementation costs, not a controlled comparison of total cold startup.

| Candidate | 100 names p50 / p95 | 1,000 names | 5,000 names |
| --- | ---: | ---: | ---: |
| Width 64 + greedy matching | 3.73 / 4.59 ms | 17.53 / 24.48 ms | 75.27 / 113.78 ms |
| Width 64 + CTC graph | 4.43 / 6.11 ms | 11.10 / 14.52 ms | 36.04 / 46.89 ms |
| Width 96 + greedy matching | 4.71 / 5.88 ms | 19.15 / 24.39 ms | 82.35 / 110.15 ms |
| Width 96 + CTC graph | 5.38 / 6.89 ms | 12.05 / 16.61 ms | 37.92 / 48.50 ms |

This separate scaling experiment uses eight original artist recordings and deterministic dictionary-word text distractors, with no generated audio and no large-catalog accuracy claim. Different audio from the main timing table means the two tables should not be directly compared. Construction time, phone-graph state counts and dictionary exclusions are retained in `catalog-scaling.json`.

## Verification, limitations and promotion

The decoder passed 30 exhaustive full-CTC cases and 9 keyword/filler cases; the parser passed 404 numeric forms and rejected 8 malformed forms. Catalog tampering/version changes, disabled gates, homophone abstention and VCM intent ownership are checked. Torch/ONNX and padded/single parity include 20-second inputs; development CPU task parity is recorded per model.

Both 36-epoch CUDA fitting runs completed and wrote checkpoints, export metadata and parity results, then their Windows processes exited with native code 3221226505. A small GRU reproducer isolates the issue to the cuDNN path on this build: disabling cuDNN makes that reproducer exit normally. The selected checkpoints were independently re-exported in a fresh CPU process with exit code zero, and separately compared with PyTorch CPU outputs. The workflow now separates export into a CPU worker and provides an explicit `export` command. An optional `--no-cudnn` fitting path is available, but a full 36-epoch run using that slower path was not performed. Original exit statuses and diagnostics remain in the evidence folder.

An initial development-only CPU batching check exceeded a strict absolute 1e-4 bound by 2.64e-5 on one recording, at logit magnitude 17.3. There were no frame argmax changes. Final comparison uses absolute tolerance 1e-4 plus relative tolerance 1e-5 and separately requires identical decoded values and gated decisions. The recognition precision target was not relaxed; this numerical correction occurred before any test scoring.

**No candidate was installed.** The shipping checkpoint and its existing experimental quality limitations remain. A constrained grammar improves neither missing training coverage nor unknown rejection by itself; it can force unsupported audio toward an allowed phrase. The next acceptance decision must use natural-command precision/recall and per-value/name coverage, not text parser tests or pooled isolated words. Connected volume parsing, automatic artist/title role extraction, verified word timing, actual device names and Philippine-English coverage are not established here.

Synthetic speech remains a future experiment only. First add missing real connected values and difficult unknown/multiple-number recordings, then evaluate any synthetic augmentation against a separate untouched human test. Wider open-vocabulary models and learned audio/text retrieval remain alternatives if this phonetic approach cannot pass the required operating points.

## Reproduce

```powershell
.\.venv\Scripts\python.exe STT/phonetic-v2/acquire.py
.\.venv\Scripts\python.exe scripts/phonetic.py prepare
.\.venv\Scripts\python.exe scripts/phonetic.py train --models small wide --epochs 36
.\.venv\Scripts\python.exe scripts/phonetic.py export --models small wide
.\.venv\Scripts\python.exe scripts/phonetic.py check
.\.venv\Scripts\python.exe scripts/phonetic.py evaluate
.\.venv\Scripts\python.exe scripts/phonetic.py scale
.\.venv\Scripts\python.exe STT/phonetic-v2/write_report.py
```

See [PROTOCOL.md](PROTOCOL.md) for the declared design, [AGENTS.md](../../AGENTS.md) for the architecture contract, and [the previous study](../alternatives/REPORT.md) for the earlier four approaches. Generated data manifests live under `data/phonetic-v2/`; weights, graphs, features, thresholds and predictions live under `.cache/phonetic-v2/`. Research modules are excluded from shipping.
