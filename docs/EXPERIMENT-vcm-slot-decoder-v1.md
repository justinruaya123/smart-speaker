# E19: tiny voice slot decoder POC

Date: 2026-09-22. **Decision: keep isolated; do not integrate.** The executable direct audio/pronunciation embedding POC works mechanically but fails useful acceptance for connected numbers, clock times and dynamic names. Cheap lookup does not solve acoustic identity or rejection. This is the first feasibility probe of the supplied handoff, not completion of its full research program.

This single report combines research, results and runnable instructions, following the repository's flat-document convention. Code is in canonical `scripts/models/` and `scripts/test/`; [the requested experiment directory](../experiments/vcm_slot_decoder/) contains a thin entry point and compact receipts. It introduces no environment, model training, downloads, audio synthesis, shipping source-map entries, installation, packaging or device actions.

## Scope and evidence

Addresses ME-03/04/06/07/08 and conditional UC-01/03/04/06/08 slot research. The installed VCM remains authoritative. It does not currently provide verified slot boundaries or an alarm class. This POC takes full recorded utterances and an annotation/caller-supplied slot role; it never claims complete-command, endpoint, wake-word or Pi accuracy.

The handoff explicitly permits phoneme CTC, whereas [REQUIREMENTS.md](../REQUIREMENTS.md) currently excludes narrow/phonetic transcription readers. A research-only exception was requested during this task and remains unresolved in this receipt. **Only the non-transcribing embedding arm was newly executed.** Historical phonetic results are read-only comparisons; no general ASR control, phoneme arm or CTC hybrid was run or added to the pipeline. An authorized follow-up needs its own frozen protocol/selection receipt.

The probe reuses E06's frozen `dual_kws` models, originally trained from random weights. That is a predeclared existing-model control, not a newly selected best architecture. Prior results have already been inspected: this is historical-holdout regression, not a new blind confirmation. E08 trained on more speech than E06, so their accuracy differences cannot isolate architecture.

Evidence:

- [Frozen protocol and source/model hashes](../experiments/vcm_slot_decoder/protocol.json), [original-audio audit](../experiments/vcm_slot_decoder/audit.json), [recording references](../experiments/vcm_slot_decoder/recordings.json), [conditional trials](../experiments/vcm_slot_decoder/queries.json).
- [Catalogs](../experiments/vcm_slot_decoder/catalogs.json), [development selection written before test inference](../experiments/vcm_slot_decoder/selection-before-test.json), [results with denominators](../experiments/vcm_slot_decoder/results.json).
- [Desktop scaling and dynamic updates](../experiments/vcm_slot_decoder/benchmark.json), [post-hoc failure accounting](../experiments/vcm_slot_decoder/diagnostics.json), [hard-name index controls](../experiments/vcm_slot_decoder/hard-negatives.json).
- Raw predictions: `.cache/vcm-slot-decoder-v1/{dev,test}-predictions.json`; unchanged-source hashes and cache prediction hashes are recorded in the compact receipts. Never use these test outcomes to retune this version.

## What was built

`16 kHz mono -> existing 80-bin per-utterance CMVN log-Mel frontend -> frozen depthwise temporal audio encoder -> ten 96-dimensional window embeddings -> cosine comparison with catalog pronunciation embeddings -> ID/value -> score and margin gates -> structured result or abstention`.

The frontend is reused from a historically named `stt_runtime` module; no transcription function or token decoder is called. The shared bootstrap registers existing module locations lazily. The POC itself has no PyTorch or network inference dependency. Pronunciations enter from catalog text, not from decoding the recording. This is phonetic identity matching, not semantic search: related artists, rooms or genres are not interchangeable identities.

The candidate inventory composes number names for 0-100 and includes percent forms and “a hundred.” One existing encoder scores these candidates; no 101-class model was trained. It is **enumerated compositional text enrollment, not demonstrated compositional acoustic generalization**. Hyphenation is a text convention, not an acoustic distinction; aliases in the fixture use spaces. The minimal time inventory covers hours 1-12, minutes 00/15/30/45, and AM/PM/unspecified. Missing meridiem returns clarification, never an invented morning/evening choice. Other minutes, durations, dates, relative times and quarter-to/past forms are unsupported in this POC.

Aliases and explicit phone alternatives can be indexed without retraining. Stable IDs, catalog fingerprints, unavailable pronunciations, duplicate-ID checks and score/margin abstention are implemented. Any catalog change invalidates the previous gate. Raw top-five candidates are clearly labeled diagnostics, never emitted commands. A rejected or incompatible caller intent cannot produce an accepted slot.

## Data and protocol

905 original recordings were freshly hashed and decoded for readability: 490 dev and 415 test. They produce 1,758 conditional trials before applying the two catalog variants; the same negative recording can be queried for several roles and is not counted as several independent recordings. No selected audio hash or annotated speaker overlaps model training or the other evaluation split. Existing train/dev/test assignments were preserved, and only recordings with consistent original and phonetic-study splits were eligible.

All natural eligible name/number queries are included. Isolated numerals are capped at five recordings per value/split by stable ID, making their 145-test denominator intentionally different from the older 1,272-test corpus. Each role sees 80 deterministic unrelated utterances per split. Names absent from the fixed catalog become out-of-catalog trials; expanded catalogs add the older held-out names. “Unseen” here means the exact normalized name does not occur in the union of both existing model training transcript pools. It is stricter than a label-only check, but does not certify absent subwords, homophones or aliases.

| Human source reused | Selected dev / test | Relevance and limitations |
| --- | ---: | --- |
| SLURP-real | 327 / 247 | Connected assistant speech, artist/song and room/place labels, new manually reviewed time labels. Real audio is CC BY-NC 4.0; text has separate terms. |
| FSDD | 50 / 50 | Isolated digits only; cannot establish connected 0-100 or time coverage. |
| MLEnd Spoken Numerals | 94 / 95 | Isolated number categories; labels are not independently verified transcripts. |
| Fluent Speech Commands | 19 / 20 | Unrelated/negative command speech; academic-research license. |
| Snips SmartLights | 0 / 3 | Original connected lighting numbers; Sonos research-only license. |

Source receipts remain in [data/sources](../data/sources/) and the existing [parameter-study audit](../data/parameter-study/audit.json). FSDD/MLEnd selected-row license fields point to earlier acquisition receipts rather than naming a license: that unresolved provenance field is retained, not silently inferred. No redistribution or new acquisition occurred.

Six dev and five test clock labels were reviewed from existing transcript annotations before inference. All are on-the-hour AM/PM expressions; labels and exact text are in the audit. No acoustic segmentation or independent listening-based label adjudication is claimed. “Tomorrow” and meeting context remain unresolved complete-command fields.

Development selects a score and best-versus-second margin per catalog and role/group: at least 10 emissions, at least 95% precision, and at most 1% false acceptance over at least 20 unknown trials. Otherwise the group is disabled. Nine connected-number dev positives and six time positives cannot meet the minimum-emission rule even with perfect recognition; their failed gates reflect insufficient support as well as weak raw accuracy. Test never changes these policies.

## Fresh audio results

Expanded catalogs: 101 number values, 144 time/meridiem candidates, 127 artists, 109 songs, 41 room/place labels. Room names are **proxies**, not proof of exact device identities.

| Conditional slot task | Raw top-1 | Top-3 | Top-5 | Correct accepted / all emitted | Unknown false accepts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isolated numbers | 90/145 (62.1%) | 128/145 | 136/145 | 40/46 (87.0% precision) | 0/80 |
| Connected numbers | 3/10 (30.0%) | 3/10 | 3/10 | 0/0; disabled | 0/80 |
| Clock times | 0/5 | 1/5 | 1/5 | 0/0; disabled | 0/80 |
| Artists | 12/77 (15.6%) | 16/77 | 19/77 | 0/0; disabled | 0/80 |
| Songs | 8/45 (17.8%) | 11/45 | 16/45 | 0/0; disabled | 0/80 |
| Room/place names | 25/79 (31.6%) | 37/79 | 43/79 | 0/0; disabled | 0/80 |

Disabled groups reject every known query too. Their zero false accepts are not evidence of a useful recognizer. The only active policy, isolated numbers, fell from 48/49 development precision to 40/46 test precision and fails the 95% target. Zero errors in 80 unknown trials also cannot certify a population false-accept rate below 1% (one-sided 95% upper bound is about 3.7%).

| Expanded-catalog name task | Seen-name top-1 | Unseen-name top-1 | Top-20 recall ceiling for any reranker |
| --- | ---: | ---: | ---: |
| Artist | 12/39 | 0/38 | 35/77 (45.5%) |
| Song | 4/26 | 4/19 | 24/45 (53.3%) |
| Room/place | 24/65 | 1/14 | 65/79 (82.3%) |

This directly challenges an embedding-first top-20 hybrid: even a perfect downstream scorer cannot recover candidates omitted by the shortlist. It does not rule out a better embedding model or a larger shortlist.

Failure attribution is bounded. Dictionary coverage is missing for 17/77 artist queries and 1/45 song queries. But 48/60 representable artists and 36/44 representable songs are still wrong. All ten connected numbers and five clock targets have representations, so those failures are not dictionary OOV. Connected numbers are 1/7 in SLURP and 2/3 in SmartLights; the small denominators prevent a source-general conclusion. Training data diversity, context contamination, window localization and encoder/objective quality remain hypotheses requiring ablations, not established causes. No microphone/capture failure is implicated by these file replays.

The held-out connected values are 0, 2, 7, 8, 12, 20 and 50. Results include 2/3 for zero and 1/1 for twelve; other supported values are 0. There is no natural held-out 75/37/58/91 evidence. A legal text inventory covering them must not be counted as passing speech tests.

## Historical controls and research investigation

Primary literature was checked on 2026-09-22. Inclusion: original papers, author/publisher pages and official dataset documentation relevant to compact acoustic identity, constrained sequence scoring, unseen words or evaluation. No paper's published score is imported as project evidence, and no exhaustive systematic-review claim is made.

| Handoff baseline | Evidence available | Assessment |
| --- | --- | --- |
| Phoneme + edit distance | Historical E08 wide: 16/77 artists, 12/45 songs; connected numbers 14/60 | Loses acoustic alternatives during greedy collapse. No fresh run here. |
| Phoneme + constrained Viterbi graph | Historical E08 wide: 36/77 artists, 24/45 songs; 40/60 connected numbers; all those acceptance groups disabled | Stronger existing raw result, still fails reliable rejection/acceptance. A shared-prefix graph, not a new WFST-library benchmark. |
| Direct embedding retrieval | Fresh E19 table above; 908,004 model bytes | Fast dynamic indexing, inadequate unseen-name and connected-slot quality. |
| Embedding top-K + sequence reranker | Reranker unrun; fresh top-20 ceilings above | Current shortlist already prevents high recall. More components are not yet justified. |

The 60 E08 connected examples include supplemental speaker holdouts that the old embedding model previously saw. Do not compare 40/60 against E19 3/10 as a matched architecture experiment. The name recordings are historically matched, but model data/objectives differ. Historical figures come from [E08's preserved report](../STT/phonetic-v2/REPORT.md) and [E06 interpretation](../STT/speechembedding/EVIDENCE_NOTES.md), not fresh scoring.

- [CTC, Graves et al., ICML 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf) supplies alignment-free sequence supervision. Restricting a phone graph changes allowed paths; it does not by itself establish compliance with this project's strict no-transcription requirement, rejection quality, or calibrated probability.
- [Multi-view Recurrent Neural Acoustic Word Embeddings, He et al., ICLR 2017](https://arxiv.org/abs/1611.04496) learns paired acoustic/character representations with contrastive objectives. It motivates cross-modal identity matching; independent generic semantic encoders do not provide that alignment automatically.
- [Yuan et al., Interspeech 2018](https://www.isca-archive.org/interspeech_2018/yuan18_interspeech.html) studies acoustic word embeddings with surrounding context for spoken query search. It motivates controlled localization/context experiments, not assuming that the VCM already provides accurate crops.
- [Flexible Keyword Spotting, Nishu et al., 2024](https://machinelearning.apple.com/research/flexible-keyword) explicitly addresses audio/text representation mismatch and confusable keywords. Its different model/data and EER/AUC tasks do not certify entity selection or this POC's operating point.
- [MM-KWS, Ai et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/ai24_interspeech.html) combines text and spoken templates, with pretrained multilingual features and augmentation. It motivates a separately declared enrollment comparison; it is not evidence that our scratch human-only model will transfer equally well.
- [TC-ResNet](https://arxiv.org/abs/1904.03814) and [Hello Edge](https://arxiv.org/abs/1711.07128) support investigating small temporal/depthwise encoders. Their fixed-keyword tasks and target hardware differ from dynamic names and Pi4; no Pi latency is inferred from their scores.

[SLURP's official repository](https://github.com/pswietojanski/slurp) provides connected assistant utterances, entity annotations and recording metadata. [LibriSpeech](https://www.openslr.org/12) provides broader English acoustic coverage but read-speech domain mismatch and no complete smart-device command guarantee. [Sonos's released command corpora](https://github.com/sonos/spoken-language-understanding-research-datasets) provide task-related speech under their source terms. Speech Commands and FSDD can supplement short words/digits but cannot be the sole arbitrary-name/time corpus. These sources already exist locally in overlapping studies; no duplicate dataset was created.

The choice between generic acoustic pretraining and targeted fitting remains unanswered by a controlled experiment. E08's additional LibriSpeech and better raw name ranking are suggestive but confounded. TTS may later expand training name coverage under a separately authorized synthetic protocol; human-only study rules and real-speaker final evaluation must remain separate. This task generated text catalogs only, never speech.

## Dynamic updates, hard negatives and scaling

Both ONNX sessions remained loaded while candidates changed, and graph hashes were unchanged afterward. Adding Adele to the smaller fixed catalog made it rank first on 1/3 tested held-out Adele recordings. In the full expanded catalog, unseen artists remain 0/38: the extra competition matters. These are raw ranks; the changed catalog has no passing gate. Adding Bedroom Lamp placed it third on the available training recording, behind Bedroom and Bathroom. The exact phrase was absent from this held-out pool, so this is a failed seen-audio smoke demonstration, not novel-device validation.

The requested Bedroom Light/Lamp/Lights, Bed Lamp and Bedside Lamp fixtures all index. “The Weeknd” and “ACDC” require explicit pronunciation handling in this dictionary. A controlled Adele/Adell pair with identical supplied phones produces a tie and abstention. That is an index/decision check using a catalog vector, not an acoustic confusable-name benchmark. Accented, non-English, acronym, digit-name and possessive recordings remain unmeasured. Identical pronunciations cannot identify two distinct entity IDs without additional context or clarification.

Desktop Windows 11 AMD64, root Python 3.12.13, ONNX CPU two intra-op/one inter-op threads, one BLAS thread. Four original dev artist queries, five repeats each per size. Entries are deterministic dictionary-word text distractors; their presence does not create 10,000 spoken examples.

| Entities | Index build ms | Rank p50 / p95 ms | Dense index bytes |
| ---: | ---: | ---: | ---: |
| 20 | 1.16 | 0.010 / 0.022 | 7,680 |
| 100 | 3.87 | 0.034 / 0.075 | 38,400 |
| 500 | 17.45 | 0.166 / 0.250 | 192,000 |
| 1,000 | 35.42 | 0.337 / 0.356 | 384,000 |
| 5,000 | 171.64 | 1.957 / 3.431 | 1,920,000 |
| 10,000 | 347.02 | 4.175 / 4.305 | 3,840,000 |

Ranking includes similarity, alias aggregation and full sorting. Index size is 384 bytes per one-alias 96-float32 vector, excluding Python objects, ID strings and serialized metadata; aliases add rows. The 10k catalog's JSON alone adds 763,133 bytes. Model graphs total 908,004 bytes, with 211,542 reported parameters across audio and text models. Session/dictionary startup is about 480 ms. The four fresh waveform frontend+encoder observations range 1.30-2.94 ms; this small sample is not a reliable system p95. Post-benchmark process RSS is 221,278,208 bytes (211.0 MiB), **not peak memory**, not a minimal deployment image, and not the full wake/VCM pipeline. CPU utilization, minimal-runtime peak RAM and actual Pi values are unmeasured.

No brute-force lookup crossover is observed through 10k on this desktop. This does not prove Pi performance. A top-K index is not needed for the measured desktop lookup cost; a sequence reranker would be an accuracy experiment, with the shortlist-recall limitation above. Disk read, speech duration, wake/VCM cost and the endpoint wait are excluded.

## Reproduction and querying

Run from the repository root using the existing configured environment. No dependency changes are required. If the environment is absent, use the repository's existing setup command rather than creating an experiment environment:

The tested research environment uses `threadpoolctl==3.7.0` for repeatable thread limits and `psutil==7.2.2` for the benchmark memory sample, in addition to the root runtime packages. Both were already installed. The root runtime lock alone does not pin these two research helpers; a fresh-machine setup has not been validated. Exact observed package versions are in [verification.json](../experiments/vcm_slot_decoder/verification.json).

```powershell
python train.py --tests-only
.venv/Scripts/python.exe experiments/vcm_slot_decoder/run.py check
.venv/Scripts/python.exe experiments/vcm_slot_decoder/run.py prepare
```

The first command uses the existing root setup-and-check workflow; it is not necessary in the current prepared checkout and was not rerun for this task. `prepare` verifies frozen inputs if receipts already exist. It references existing recordings, checks selected audio hashes/readability and freezes labels/catalogs; it does not download or copy audio. Fresh-run commands used for this receipt were:

```powershell
.venv/Scripts/python.exe experiments/vcm_slot_decoder/run.py evaluate
.venv/Scripts/python.exe experiments/vcm_slot_decoder/run.py benchmark
.venv/Scripts/python.exe scripts/test/vcm_slot_decoder_diagnostics.py
```

Completed selection, evaluation, benchmark and diagnostics refuse to overwrite their evidence. A changed protocol or policy requires a new experiment version; do not delete locks to rerun. `evaluate` builds the index and performs preprocessing from original PCM; derived acoustic outputs stay in the ignored cache. No training command was run: the input model hashes are the reproducible checkpoint specification. Historical training recipes remain under `experiments/speechembedding/run.py` and its frozen protocol, and must not be rerun over the old candidate as part of reproducing E19.

Example audio query (choose an actual recording path from `recordings.json`):

```powershell
.venv/Scripts/python.exe experiments/vcm_slot_decoder/run.py query --audio "data/audio/slurp/slurp_real/audio-1497862421-headset.flac" --intent SET_ALARM --role time
```

`--intent` is a conditional probe supplied by the caller, not a claim that VCM recognized that family. Results include the locked gate's abstention, raw top-five diagnostics and per-call desktop timings. Use `--role number --intent BRIGHTNESS` or `--role artist --intent MEDIA_PLAY` for the other families. Current connected/time/name gates intentionally abstain.

A custom catalog passed with `--catalog` is a JSON list, or an object with an `entities` list:

```json
[{"id":"device:bedroom_lamp","role":"device","name":"Bedroom Lamp","aliases":["bedroom lamp","bed lamp"]}]
```

Aliases use lower-case dictionary spellings with spaces. Optional `pronunciations` is a list of phone sequences in the frozen CMU-derived inventory, with `|` word boundaries. No new acoustic class is created by an edit. Unknown spellings are listed as unavailable; custom catalog fingerprints invalidate stored acceptance calibration.

## Handoff acceptance tests and answers

| Requested test | POC finding |
| --- | --- |
| A: brightness 75 | Software candidate exists; no matching untouched human test audio here. Not passed. |
| B: unseen 37/58/91 composition | Compositional candidate construction passes; acoustic generalization untested. |
| C: six thirty | Legal pilot candidate with mandatory meridiem clarification; no matching held-out audio. Five other clock recordings fail top-1. |
| D: newly added Bedroom Lamp | Index updated without model changes; seen-audio raw match failed, no held-out evidence. |
| E: newly added artist | Adele became top-1 in one smaller-catalog trial; no passed gate and expanded unseen artists 0/38. |
| F: confusable names | Text-index/OOV and homophone-abstention controls executed; full acoustic confusable benchmark missing. |
| G: out-of-catalog rejection | Fixed-catalog unseen-name and unrelated-speech trials scored; command/name policies disabled, so rejection alone is uninformative. |

1. **Reliable 0-100 without ASR?** Not established; 3/10 connected slot values, no accepted connected group, and missing values dominate coverage.
2. **Reliable clock times?** No: 0/5 raw matches, six dev examples, no half/quarter-hour acoustic coverage.
3. **Dynamic additions without retraining?** Index mechanics yes; robust new-name recognition no.
4. **Phones or embeddings superior?** Historical phones have stronger raw name results, but unmatched training and failed gates prevent an architectural winner or deployment recommendation.
5. **Brute-force limit?** Not reached for embedding ranking at 10k on this desktop; Pi limit unknown.
6. **Two-stage index at 1k/5k/10k?** Not justified by current desktop lookup cost; top-20 recall is too low for a reliable embedding-first hybrid.
7. **Unseen names?** Artist 0/38, song 4/19, room 1/14 in the expanded catalog.
8. **Accents?** No audited accent-stratified result; do not infer from speaker count.
9. **Noise?** No controlled noise/far-field evaluation or synthetic corruption was performed.
10. **Failure modes?** Wrong raw identities, missing pronunciations, shortlist omissions, homophones, unverified localization, insufficient connected/time coverage and invalidated gates after updates.
11. **Pi4 latency?** Unmeasured. No desktop-to-Pi multiplier is defensible from this probe.
12. **Smallest model meeting accuracy?** None demonstrated. The tested embedding pair is under 1 MB but fails; larger historical phones fail acceptance too.
13. **Share what?** Acoustic frontend, a validated acoustic encoder when feasible, immutable catalog IDs/hashes, and common metric/abstention infrastructure.
14. **Separate what?** Numeric validity/composition, time ambiguity/date semantics, entity enrollment/indexing, and per-role/catalog acceptance calibration.
15. **Robust enough to avoid shipping ASR?** No architecture here is demonstrated robust enough to ship. This is not permission to introduce ASR; the direct-command requirement remains unchanged.

## Concrete next architecture to test

For the strict existing contract, retain one small direct temporal VCM encoder with finite numeric/time/validity heads and a 96-dimensional **spoken-enrollment** entity branch. Bind genuine enrollment recordings to user-selected IDs; use cosine retrieval with separate rejection and clarification. Share the frontend/encoder only after a matched experiment shows no regression. Keep numeric/time semantics separate from dynamic entity identity. This respects the previously accepted playlist-enrollment milestone and does not claim that text-only unseen artist recognition is solved.

Current measured reference budgets are 0.908 MB for this separate embedding pair and 3.84 MB of dense vectors per 10k one-alias names. A future shared model's bytes/RAM, enrollment cost, full pipeline cost and Pi4 latency are **unmeasured**, not these numbers copied forward. Start with a small household catalog and real spoken enrollment; support at 10k is timing-only evidence.

The strongest evidence against this recommendation is that E06/E18's earlier audio-enrollment studies also failed their acceptance gates. Therefore the next decisive experiment is a newly frozen, real connected-command corpus with disjoint names and speakers, separate enrollment/query sessions, confusable/out-of-catalog negatives, complete numbers/time coverage and explicitly annotated slot boundaries for a diagnostic comparison. Compare whole-utterance versus true-slot acoustic embeddings and direct finite heads under matched training. This separates representation/localization limitations from data coverage. If the research-only phoneme exception is granted, add the existing small/wide CTC graph controls and hybrid to that same frozen probe; do not alter installed inference.

Eight behavioral checks passed before acoustic evaluation and again at completion. Three real-audio CLI queries returned the expected locked-gate abstention; their raw failures remain in [query-smoke.json](../experiments/vcm_slot_decoder/query-smoke.json). Original audio, frozen source/model hashes and dev/test isolation were verified. No task result closes a quality or physical Pi gate. The [final preservation receipt](../experiments/vcm_slot_decoder/verification.json) verifies 22 frozen input files, all 20 installed artifact files, and the three protected config/publisher/source-map files with zero mismatches. Source syntax and report local links passed. No commit was made.
