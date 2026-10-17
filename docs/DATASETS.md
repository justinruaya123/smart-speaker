# Dataset directory, citations and selection history

Prepared 2026-10-16 (Asia/Manila). This is a documentation audit for shipping preparation, not a new dataset version or permission to redistribute recordings. Existing datasets, frozen protocols and active work remain in place. Counts below come from named receipts; this task did not decode the full recording pool again. Missing provenance is explicitly unresolved.

Publication preparation and the proposed final structure: [shipping plan](PLAN-shipping-and-huggingface.md). A [machine-readable snapshot](dataset-provenance.snapshot.json) binds inspected receipts and source code to their current SHA-256. Historical [DATASET.md](../DATASET.md) describes the semifinal only, not the final refinement.

## Submitted 935k and 2M models: verified input membership

Updated 2026-10-17 (Asia/Manila). Use this section for the two models supplied for `airimonda/vcm-benchmark`. The [citation ledger](DATASET-CITATIONS.md) supplies original publisher references, paper-versus-dataset DOIs and access terms. The [submission and reproduction guide](REPRODUCIBILITY-935k-2m.md) maps the supplied slide fields to evidence. A fresh [machine-readable audit](model-dataset-usage.json) records manifest hashes, aggregate source counts, durations, declared identities, selected weights and locally verified archive/graph hashes. It supplements the earlier final-refinement snapshot rather than changing any frozen experiment.

| Item | 935k submission | 2M submission |
| --- | --- | --- |
| Exact experiment | `speaker-pipeline-repair-v1`, saved mixed epoch 28 | `speaker-final-refinement-v1`, control mixed epoch 32 |
| Parameters / exported graph | 935,567 / FP32, 3,864,875 bytes | 2,066,831 / INT8, 3,173,089 bytes |
| Eligible fitting pool | **23,289 = 1,863 human + 21,426 synthetic** | **23,295 = 1,869 human + 21,426 synthetic** |
| Fitting duration from stored sample metadata | 14.033144 h | 14.038888 h |
| Declared fitting identities | 173 human source-qualified IDs; 70 synthetic voice IDs | 179 human source-qualified IDs; 70 synthetic voice IDs |
| Shared development | 1,251 = 256 human + 995 synthetic | Same |
| Human / synthetic calibration | 210 / 1,197; never pooled for qualification | Same |
| Published test / holdout | 4,418 / 196; typed-scoring eligibility 4,233 / 194 | Same |

These are **eligible sampler pools**, not a record proving every example was drawn in optimization. The compact fit excludes six unresolved `TimersAndSuch` slot-metadata-conflict rows that remain in the 2M control pool. It also excludes all new `qsc1/` training recordings, as does the 2M control sampler. Compact normalization deliberately reuses the parent statistics fitted on 1,869 human training utterances, rather than silently refitting on 1,863. The repaired compact release reuses a saved checkpoint from the completed compact fit; it does not add a new training run. Older descriptions of compact epoch 32 refer to its preserved predecessor, not this submission.

The following table is recounted from the actual manifests. `source` labels are preserved as published, with attribution identities in the citation ledger. Test/holdout columns include uncertain references; do not substitute these counts for metric denominators. Calibration combines the two origin columns for inventory display only.

| Original source / published label | 935k fit | 2M fit | Shared dev | Calibration | Test | Holdout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SLURP (`SLURP`) | 910 | 910 | 108 | 72 | 574 | 0 |
| Fluent Speech Commands (`FluentSpeechCommands`) | 180 | 180 | 32 | 74 | 153 | 20 |
| SNIPS / SmartLights (`SNIPS`) | 256 | 256 | 29 | 52 | 107 | 0 |
| Common Voice (`CommonVoice_en`; composite card identifies release 19 English) | 18 | 18 | 3 | 10 | 14 | 3 |
| Timers and Such (`TimersAndSuch`) | 17 | 23 | 6 | 2 | 2 | 0 |
| Multi-Sensor Voice Command (`xela_Multi-Sensor`) | 4 | 4 | 0 | 0 | 11 | 0 |
| Student/group recordings (`real_voice`) | 438 | 438 | 58 | 0 | 179 | 73 |
| Xela temperature recordings (`xela_SET_TEMPERATURE_REAL`) | 40 | 40 | 20 | 0 | 10 | 0 |
| Existing group synthetic commands (`group_synthetic`) | 5,982 | 5,982 | 811 | 1,013 | 3,368 | 100 |
| Filipino synthetic supplement (`fil50_persona`) | 15,444 | 15,444 | 0 | 0 | 0 | 0 |
| New Qwen designed completion (`qsc1/`) | **0** | **0** | 184 | 184 | 0 | 0 |
| **Total** | **23,289** | **23,295** | **1,251** | **1,407** | **4,418** | **196** |

There is no independent original-source release/consent receipt for `xela_SET_TEMPERATURE_REAL`; it must remain a separately identified unresolved source, rather than being attributed to Multi-Sensor by its name alone. LibriSpeech and SilencioPH are declared voice-reference sources for the **existing synthetic** pool, not additional original-human fitting rows. Their exact reference-to-output map is not recovered by these manifest counts.

### Hugging Face inputs to preserve

| Input | Immutable identity | Exact role |
| --- | --- | --- |
| [AI231 ME2 Voice Commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/tree/25111444af3adff7588d86ab27c304bb080895cf) | `25111444af3adff7588d86ab27c304bb080895cf`, local config label `primary` (Hub config `default`) | 7,845 compact / 7,851 2M eligible fitting rows; source of shared original dev/calibration and all published test/holdout rows |
| [Filipino supplement](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data/tree/230c8b8d58caf78e4bc90886d9c6fa77fe6024b8) | `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8` | Both fits use `default/train` 14,120, QA-eligible `gap_fill/train` 1,320 of 1,849, and `numeral_wordings/train` 4; 529 failed-QA rows are excluded |
| Local Qwen completion | [Generation protocol](../experiments/speaker-synthetic-completion-v1/protocol.json), [row manifest](../experiments/speaker-synthetic-completion-v1/generated.jsonl) | 184 dev + 184 synthetic calibration rows affect shared diagnostics/selection context. The other 184 train rows belong to the losing completion comparison, not either winning fit |
| CMUdict | `74790861f652b15e4ac49015a90074ad62a27690` | Text pronunciations and project overrides; not acoustic pretraining or waveform data |

The primary corpus DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) is an upstream citation. Always pair it with the fitted Hub commit; no DOI-to-that-commit binding or DOI for the prepared derivative release is established here. Do not reuse the upstream DOI as the identifier of `justinruaya123/ai231-smart-speaker`.

### Available data that did not train these two models

The separate auxiliary pool has **66,390 original-human numeral clips = 32,029 MLEnd + 34,361 Speech Commands v2**, 21.796344 h from stored durations. Neither final command-fitting pool contains those rows. FSDD, MSWC, separately acquired LibriSpeech human speech, DEMAND and MS-SNSD have no confirmed direct fitting rows in these manifests. Historical training studies, voice-cloning references, wake/VAD pretraining and the target Hub's extra negative configurations are distinct provenance roles. Cite them if distributing those resources, without describing them as direct fitting data for these models.

### Audit result and remaining gaps

The 2026-10-17 metadata audit freshly hashed eight input manifests, 37 evidence/source files, both selected checkpoints and both ZIPs; it verified each packaged ONNX graph against its model binding. It also compared all ten pairs of command train/dev/calibration/test/holdout sets: zero overlap in example ID, declared speaker ID, pair ID and stored PCM SHA-256. Compact is a verified subset of the 2M fitting pool. This does **not** establish real-world alias-resolved unseen speakers, absence of near-duplicates, fresh waveform integrity or independent listening. Source consent, clip-level ancestry and distribution conditions remain open for the named unresolved inputs. Both models remain unqualified; physical Pi and live-microphone evidence remains unmeasured.

## Current experiment inputs

| Input | Exact version and local authority | Use and redistribution status |
| --- | --- | --- |
| AI231 ME2 Voice Commands, airimonda (2026) | `airimonda/ai231-me2-voice-commands` at `25111444af3adff7588d86ab27c304bb080895cf`; [source receipt](../data/sources/ai231-primary.source.json), [saved source card](../experiments/pcv-source-refresh-v1/source-card.txt); [dataset citation, DOI 10.57967/hf/10723](https://doi.org/10.57967/hf/10723) | Primary mixed human/synthetic corpus. Every recording retains its upstream terms. Restricted speech and consent-limited student recordings require individual inclusion decisions. |
| Filipino synthetic supplement, martinnavs (2026) | `martinnavs/ai231-fil-supplemental-data` at `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`; [receipt](../data/sources/fil-supplemental.source.json), [saved notice](../data/sources/fil-supplemental-NOTICE.txt), [pinned publisher card](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data/blob/230c8b8d58caf78e4bc90886d9c6fa77fe6024b8/README.md) | Synthetic training addition; its upstream pool is train only. The publisher prohibits redistribution while reference rights are unresolved. Its voices are not independent human speakers. |
| Local synthetic completion v1 | [protocol](../experiments/speaker-synthetic-completion-v1/protocol.json), [generation completion](../experiments/speaker-synthetic-completion-v1/generation-complete.json), [results/preparation receipt](../experiments/speaker-synthetic-completion-v1/results.json) | 552 generated commands; 184 each in train/dev/calibration. Six designed voice conditions, 92 prompts each. Intended text and waveform checks are not independently verified speech labels. Output distribution terms need a separate decision. |
| CMUdict | Revision `74790861f652b15e4ac49015a90074ad62a27690`; [receipt](../data/sources/cmudict.source.json), [preserved license](../data/sources/cmudict-LICENSE.txt), [original repository](https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690) | Text pronunciations, no human audio or pretrained acoustic weights. BSD-style CMU dictionary terms. Preserve the project lexicon overrides separately. |

Completion generation used Qwen3-TTS [VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/tree/5ecdb67327fd37bb2e042aab12ff7391903235d3) at `5ecdb67327fd37bb2e042aab12ff7391903235d3` and [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/fd4b254389122332181a7c3db7f27e918eec64e3) at `fd4b254389122332181a7c3db7f27e918eec64e3`. The [asset receipt](../data/sources/qwen-synthetic-completion.source.json) records Apache-2.0 model assets. That asset license alone is not a rights audit of every generated recording. TTS is offline data construction and is absent from VCM runtime and initialization.

## Upstream datasets and resources requiring attribution

This table covers directly acquired datasets, historical studies and sources declared by the composite corpus/target card. Inclusion in this table does not mean every source was used in final fitting. Composite rows must retain their original source identity; a corpus-level citation alone is insufficient. Original per-record release IDs are unresolved where only an aggregate card identifies the source.

| Dataset / original citation | Local identity, selection and terms evidence |
| --- | --- |
| Fluent Speech Commands — Lugosch et al. (2019), [Speech Model Pre-training for End-to-End Spoken Language Understanding](https://arxiv.org/abs/1904.03670) | [Receipt](../data/sources/fluent.source.json): mirror revision `c4cc0e2c898585329319835ed55f47a6ab9e0d92`, archive SHA-256 `c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b`. Natural connected commands, historical semifinal and parameter studies, also inherited through primary. [Publisher terms](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) restrict use to academic research; do not treat a mirror license tag as redistribution clearance. |
| Speech Commands v0.02 — Warden (2018), [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209) | [Receipt](../data/sources/speech_commands.source.json): archive SHA-256 `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58`; [dataset authority](https://www.tensorflow.org/datasets/catalog/speech_commands). CC BY 4.0. Historical isolated controls/number supervision and primary numeral pool; isolated words do not prove connected-command accuracy. |
| SLURP — Bastianelli et al. (2020), [SLURP: A Spoken Language Understanding Resource Package](https://aclanthology.org/2020.emnlp-main.588/) | [Annotation receipt](../data/sources/slurp.source.json): `8eb16545762be97ace75334109d73824217311f1`; [audio mirror receipt](../data/sources/slurp_audio_mirror.source.json): `91b0abfee2e735282967ee00d631d6d5f0fb7ff9`. **Publisher distinction: text CC BY 4.0, audio CC BY-NC 4.0**, per [upstream license section](https://github.com/pswietojanski/slurp#license). Composite cards and some historical local descriptions label SLURP audio CC BY 4.0; resolve that mismatch before shipping audio, including smoke samples. Do not rewrite frozen receipts to conceal it. |
| LibriSpeech — Panayotov et al. (2015), [LibriSpeech: An ASR Corpus Based on Public Domain Audio Books](https://www.openslr.org/12) | [Receipt](../data/sources/librispeech.source.json) hashes `train-clean-100`, `dev-clean`, `test-clean` archives. CC BY 4.0. Historical parameter/phonetic subset and upstream synthetic voice references. Internal speaker holdouts are not the official LibriSpeech benchmark; reference speech and cloned outputs need distinct provenance. |
| Multilingual Spoken Words Corpus — Mazumder et al. (2021), [Multilingual Spoken Words Corpus](https://mlcommons.org/datasets/multilingual-spoken-words/) | [Receipt](../data/sources/mswc_lighting.source.json): `0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218`, 31 requested English lighting words, official train shards only. CC BY 4.0. Automatically aligned Common Voice word crops; internal speaker-separated keyword splits, not connected-command test evidence. |
| SNIPS / SmartLights — Saade et al. (2019), [Spoken Language Understanding on the Edge](https://github.com/sonos/spoken-language-understanding-research-datasets) | [Receipt](../data/sources/smartlights.source.json): mirror `d1a41a6e3c09b45bbfdc64220dff7b3176679964`. Sonos Crowdsourced Data License governs audio; mirror MIT metadata does not override it. Upstream terms restrict republishing to unmodified datasets under those terms; review a merged/transformed release separately. Historical brightness and room-name proxy selections are not native transcripts for arbitrary device names. |
| MLEnd Spoken Numerals — [Queen Mary MLEnd dataset page](https://mlenddatasets.github.io/spoken_numerals/) | Primary card identifies 32,029 numeral clips. Historical parameter audit selects 1,708 numeral recordings (1,124 train / 290 dev / 294 test). Original archive/release hash and full publisher citation are not in the top-level source receipts: unresolved. Categories are canonical labels, not listener-verified verbatim transcripts. |
| Free Spoken Digit Dataset (FSDD) — [Jackson et al., original repository](https://github.com/Jakobovski/free-spoken-digit-dataset) | Historical parameter/phonetic data: 3,000 clips, 2,000 train / 500 dev / 500 test. `STT/prepare.py` explicitly holds out speakers instead of the source's overlapping-speaker split. Top-level source acquisition revision/hash is missing; exact historical input must be recovered before redistributing/rebuilding it. |
| Common Voice 19 English — Ardila et al. (2020), [Common Voice: A Massively-Multilingual Speech Corpus](https://arxiv.org/abs/1912.06670) | Composite card identifies release 19 English and CC0. Original archive checksum and selected upstream row IDs require reconstruction from publisher metadata; the paper is not a release-19 checksum. MSWC crops are a separate derived source and must not be conflated with this input. |
| Timers and Such — Lugosch et al. (2021), [Timers and Such: A Practical Benchmark for Spoken Language Understanding with Numbers](https://arxiv.org/abs/2104.01604), [dataset DOI](https://doi.org/10.5281/zenodo.4623772) | Inherited from primary and target evaluation extras. Composite card says `other-open`; preserve the actual publisher license and resolve clip-level identity before export. No direct source receipt in this checkout. |
| Multi-Sensor Voice Command — Rusci, Van hamme and Tuytelaars (2024), [dataset DOI 10.48804/IEKKVZ](https://doi.org/10.48804/IEKKVZ) | Inherited through Xela's VCM. Publisher custom terms combine CC BY 4.0 with download-tracking requirements for full/partial derivatives. Original version/record mapping and a compliant hosting decision remain unresolved. |
| Group student recordings / Xela S1–S5 | Identified by [primary card](../experiments/pcv-source-refresh-v1/source-card.txt). No standalone scholarly citation or public license is established. Preserve contributor attribution and consent receipts privately; publishing identifiable metadata/audio needs the relevant consent scope. |
| Group synthetic set — markandrian30/AI231, MEX2/Data | Composite card attribution only; exact immutable repositories/releases and generation receipts are not established here. LibriSpeech and SilencioPH references, synthetic voices and human speakers remain distinct. SilencioPH identity/license/consent is unresolved. |
| Filipino supplement / CosyVoice2 references | Pinned supplement above. Default 14,120; gap-fill 1,849; numeral-wordings 4. Seventeen cloned reference voices; reference consent, license and alias overlap unresolved. Transcription QA does not settle these questions. |
| DEMAND noise | Declared by existing target card for synthetic negatives; [original release](https://doi.org/10.5281/zenodo.1227121). Target states CC BY-SA 3.0. Source-file and mixing lineage, actual license copy and derivative obligations must be retained; final-refinement training use is not established. |
| Microsoft MS-SNSD — Reddy et al. (2019), [A Scalable Noisy Speech Dataset and Online Subjective Test Framework](https://www.isca-archive.org/interspeech_2019/reddy19_interspeech.html) | Declared by target card; [publisher dataset licenses](https://github.com/microsoft/MS-SNSD#dataset-licenses). **Correction:** MIT covers code; audio retains upstream terms (including Freesound CC0 and DEMAND CC BY-SA 3.0 for noise). The target's blanket MIT audio description is insufficient. Exact selected-file origins/revisions/checksums are unresolved. |
| Historical local Qwen / Dia / Parler synthetic studies | [Synthetic v2 protocol](../experiments/pcv-synthetic-v2/protocol.json), [v3 protocol](../experiments/pcv-synthetic-v3/protocol.json), [Qwen acquisition](../experiments/synthetic/tts-acquisition.json). These are separate constructed datasets. v3 regenerates Parler audio after codec-weight verification and retains parent Qwen/Dia assignments. No claim that all historical generators feed final refinement; recover their exact model/license receipts if including those configurations. |

MLEnd Spoken Words, if encountered in legacy files, is a separate identity requiring verification; do not silently cite Spoken Numerals for it. Pretrained wake/VAD graphs and Python wheels are software/model assets, not recording datasets; their separate [Pi-assets receipt](../data/sources/speaker-final-pi-assets.source.json) must travel with runtime notices.

## Selection and mutation ledger

### Primary upstream construction

The pinned saved primary card describes a 93-variation bounded task, source/speaker-stratified train/test, dedicated holdout voices, source-priority balancing and a separate numeral pool. It describes an out-of-scope mixture with noise, Filipino speech, near-miss requests and general speech. Upstream ASR-assisted checks and relabeling are annotation provenance, not a project runtime transcriber or independent listening. Preserve `transcript_source`, original labels, checks and notes; do not upgrade them to verified transcripts.

The local source refresh records 81,686 primary rows: train 10,682; test 4,418; holdout 196; auxiliary numerals 66,390. See [acquisition](../experiments/pcv-source-refresh-v1/acquisition-complete.json) and [revision difference](../experiments/pcv-source-refresh-v1/revision-difference.json). Eighteen former-training PCM identities and ten former-training speakers moved into newer evaluation membership: older fitted models are not clean unseen-test candidates for that revision.

### Shared installation: speaker-dataset-v1

[Installer](../scripts/train/speaker_dataset_install.py), [audit](../experiments/speaker-demo-v2-data/data-audit.json), [protocol](../experiments/speaker-demo-v2-data/protocol.json).

The installer combines those 81,686 primary rows with 15,973 supplemental rows, for 97,659 inventory rows (70,412 human / 27,247 synthetic). It preserves original WAV bytes, validates mono 16 kHz PCM16, and records container and suite-compatible decoded-PCM hashes. Audio remains in `data/audio/ai231-pool-v1/`; functional views reference it.

Primary test/holdout membership stays unchanged. Primary train rows use `assign_split` in [pcv_updated_prepare.py](../scripts/train/pcv_updated_prepare.py): source-qualified speaker key, SHA-256 of `pcv-updated-v1/` plus key, first 12 hex digits modulo 100; `<75` train, `<88` dev, otherwise calibration. Explicit exceptions: `real_voice/202322013` and `real_voice/202521746` train; `real_voice/202322013_speaker2` dev. Retain these exceptions in reconstruction metadata. All imported supplemental rows first enter the training pool.

Finite annotation uses existing source supervision, number/text normalization and exact reviewed corrections in `load_policy`. Preserve raw metadata and derived command, reason, ambiguity and conflict status. Reuse the frozen annotation receipt, including its documented CRLF recovery; do not normalize and overwrite its original bytes.

| Installed view | Rows | Meaning |
| --- | ---: | --- |
| human/train | 1,895 | Eligible original human fitting pool before pronunciation filtering |
| human/dev | 256 | Human development |
| human/calibration | 210 | Human-only calibration; five required intents absent |
| synthetic/train | 21,426 | Existing synthetic training pool, including eligible supplement |
| synthetic/diagnostic | 1,824 | Primary synthetic dev/calibration; not imported supplemental holdouts |
| evaluation/test | 4,418 | Every published test row retained |
| evaluation/holdout | 196 | Every published holdout row retained |
| auxiliary | 66,390 | Isolated numerals, outside the final complete-command fitting views |
| excluded | 1,044 | 504 ambiguous references, 11 missing-speaker rows, 529 failed supplemental QA |

The audit records no cross-split declared-speaker, pair or PCM groups and no conflicting PCM labels. This is not an alias-resolved human/voice identity audit or near-duplicate check. Gap-fill eligibility requires `qa_passed OR qa_rescued`. Fitting excludes audio errors, ambiguity, missing identities, duplicate/conflicting PCM and held-out/cross-split identities. Evaluation retains ambiguity explicitly; metric eligibility is a separate flag.

### Synthetic completion and final fitting views

[Preparation implementation](../scripts/train/speaker_synthetic_prepare.py), [generation plan](../experiments/speaker-synthetic-completion-v1/generation-plan.json), [voices](../experiments/speaker-synthetic-completion-v1/voices.json), [generation receipt](../experiments/speaker-synthetic-completion-v1/generation-complete.json).

New synthetic commands retain native waveform hashes, derived waveform/PCM hashes, voice/anchor identity, prompt, model revisions, seeds and attempts. Generation uses Qwen-TTS 0.1.1 / Torch 2.10.0+cu128; SHA-256-derived per-batch/per-attempt seeds, bfloat16, SDPA, temperature .75 and top-p .95. Native-to-16-kHz conversion is a derived-audio operation; preserve parameters from the generator and both identities. Six anchor recordings are additional generation assets, not six extra command examples. Exact generated bytes must be preserved: seeded GPU regeneration is not a promise of byte identity.

The preparation retains all 1,031 parent catalog entries and adds declared light/color pairs for 1,079 entries. It expands supervised numeric text and dictionary pronunciations, filters 26 human training rows for missing pronunciation/text, and preserves the exclusion receipt. Normalization uses **1,869 eligible human training utterances only**, with equal utterance weight. Feature extraction covers fitting/development, never held-out evaluation during preparation.

| Final view / comparison | Human | Synthetic | Total |
| --- | ---: | ---: | ---: |
| Control fitting | 1,869 | 21,426 | 23,295 |
| Completion fitting | 1,869 | 21,610 | 23,479 |
| Shared development | 256 | 995 | 1,251 |
| Calibration, reported separately by origin | 210 | 1,197 | 1,407 |

Only 184 of the 552 new commands enter completion fitting. The other 368 enter synthetic dev/calibration. Synthetic diagnostic evidence never replaces missing human calibration coverage. Full published evaluation membership stays 4,418 test and 196 holdout; final typed scoring uses 4,233/194 eligible rows with 185/2 uncertain references separate.

Final [training protocol](../experiments/speaker-final-refinement-v1/protocol.json) uses seed 53 and grouped intent → command → declared-speaker → recording sampling. Both arms use training-only conservative feature masking (frequency width at most 4; time at most 10 frames and 5% of length; each probability .5; no noise/reverb/speed/splicing). Masks are online stochastic transforms, not additional independent recordings. Evaluation has no augmentation. The [development selection](../experiments/speaker-final-refinement-v1/selection-before-calibration.json) and [pre-test lock](../experiments/speaker-final-refinement-v1/selection-before-test.json) precede the final regression replay. Previously examined test data are not a new blind holdout.

**The selected shipping candidate is the control arm, epoch 32**, checkpoint SHA-256 `ff6a3af63115a8f92a0106b20a17ee57c3e84f263177b89aaedb9cce326fce9b`. Its fitting pool is 23,295, excluding the 184 new synthetic training rows: the sampler classifies IDs beginning `qsc1/` as `new_synthetic`, whose control batch count is zero. The 23,479-row prepared manifest is the superset for the comparison, not the selected candidate's fitting membership. New completion recordings still influence the shared development comparison; keep their lineage and the losing arm's evidence. The completion arm did not win development and must not be credited as the source of the final model's improvement.

### Historical selections preserved separately

| Version / evidence | Selection and mutation boundary |
| --- | --- |
| Semifinal v1: [DATASET.md](../DATASET.md), [protocol](../reports/semifinal-v1/protocol.json) | All 30,043 FSC originals plus 2,500 deterministic Speech Commands controls; train/dev/calibration/test totals 25,132 / 1,583 / 1,835 / 3,993. Original FSC validation speakers split into five dev/five calibration. CMUdict textual supervision. No TTS/splicing. Historical features lacked pre-fit byte hashes; later numerical audit does not erase the strict audit failure. Frozen manifest ZIP remains authoritative historical evidence. |
| Constructed brightness: [audit](../data/stt/brightness_audit.json), [construction receipt](../data/stt/context_receipt.json) | Historical spliced/constructed human-word phrases must remain labeled constructed and excluded from human-only claims. Not part of the pure-VCM shipping inference path. |
| Human brightness: [audit](../data/stt-human/brightness_audit.json) | Original human recordings, separate frozen manifest; no synthetic concatenation. Exact source-manifest hash and selection counts live in the audit. |
| Parameter study: [audit](../data/parameter-study/audit.json) | 19,313 originals: train 13,762 / dev 2,537 / test 2,602 / enrollment 412. Eight named source groups; exclude inconsistent annotations (3), root calibration (4,105), duration (471). Source-qualified speaker split and reserved names; one microphone per paired utterance. Text normalization and 80-bin per-utterance CMVN are historical study-specific transforms. |
| Phonetic v2: [audit](../data/phonetic-v2/audit.json) | 29,307 originals: train 22,807 / dev 2,925 / test 3,148 / enrollment 427. Preserves 2,602 legacy tests; adds LibriSpeech internal speaker holdout and SmartLights supplemental speakers. Omits dictionary OOV (10,826) and reserved-name (589) rows. No synthetic audio; not an official LibriSpeech benchmark. |
| Fixed synthetic v1: [protocol](../data/tts/protocol.json), [human audit](../data/tts/human-audit.json) | Planned synthetic train/dev/test 2,688/672/672; separate 11,591 original human references with source-label/grammar supervision, not human adjudication. Planned counts do not prove all generated payloads exist. |
| PCV synthetic v1/v2/v3 | [v1 report](EXPERIMENT-pcv-synthetic-v1.md), [v2 report](EXPERIMENT-pcv-synthetic-v2.md), [v3 report](EXPERIMENT-pcv-synthetic-v3.md). Preserve generation, voice partitions, source weights/codec checks and negative results per version. Do not merge examples by prompt text or count voices as people. |
| Older full benchmark | [Protocol/report](EXPERIMENT-pcv-full-benchmark-v1.md): primary revision `6947f13073e57eb6ae67e7e2fc3680700b82aa13`, 6,768 evaluation recordings including supplemental-synth and negatives. It has different source membership from final refinement. |

Historical numeral selection is additionally specified in [STT/acquire_mlend.py](../STT/acquire_mlend.py): seed `20260919`, values at most 100, speaker split before sampling, caps 35 train / 12 dev / 12 test per numeral. It expects `data/raw/mlend/revision.json` and writes `evidence/mlend_acquisition.json`; those acquisition files were absent in this checkout at inspection, so do not claim the exact historical revision was recovered. [STT/prepare.py](../STT/prepare.py) puts FSDD `yweweler` in test, `theo` in dev, other speakers in train. Preserve original selections and audio hashes instead of inferring a release from today's repository contents.

## Existing target contents are a different snapshot

Read-only inspection found target commit `3ed018c1bc731124a1b2ac5bd6653b5cd01bf576`. Its [pinned card](https://huggingface.co/datasets/justinruaya123/ai231-smart-speaker/blob/3ed018c1bc731124a1b2ac5bd6653b5cd01bf576/README.md) declares default train/test/holdout/numerals **10,733 / 4,443 / 202 / 66,390**, `supplemental_fil` 14,120, `supplemental_synth` 5,856, and `synthetic_negatives` train 1,000 / test 250 / noise 88 / tune_oos 400. These are card declarations, not an independent audio recount. Its prose still contains older 10,682/4,418/196 counts and upstream loading examples: reconcile those before release.

That card attributes upstream removal/replacement of 743 unslotted-command rows, generated out-of-scope additions, and seed-0 negative generation: equal shares noise, babble, reversed, truncated and near-silence. It preserves `voice_split` for unused synthetic commands and `source_files` for negative construction. Recover original script revision, source hashes, parameters and per-row decisions before claiming these mutations reproducible locally. Do not infer these extra target configurations trained the final refinement. The final candidate instead binds the earlier primary revision and the separate supplemental `gap_fill`/`numeral_wordings` inputs, plus local completion.

No target files were changed by this task. Existing publication is not evidence of permission to republish, complete provenance or equivalence to the experiment.

## Remaining reproducibility and attribution work

Freeze portable manifests with source-revision/shard/row IDs, immutable source hashes, source-qualified identity, original and experiment splits, raw and normalized supervision, command/reference status, per-row license and inclusion/exclusion reason. Preserve frozen original bytes alongside a separately hashed portable representation and path map. Resolve original per-clip releases for composite sources and missing historical FSDD/MLEnd pins. Audit alias/near-duplicate leakage, consent, label quality and restricted-source distribution explicitly. Save generated audio and anchors under an appropriate distribution decision or state that the public release cannot reconstruct those bytes.

2026-10-16 audit entry: all nine installed view manifests (97,659 rows total) and three prepared manifests (train, dev, excluded-training) were freshly rehashed and row-counted against their receipts. The snapshot preserves per-arm ID-set hashes and 58 source/evidence/code file identities. This does not re-audit audio bytes, speaker aliases or acoustic labels. Documentation and metadata snapshots only; no downloads of recording payloads, dataset mutation, feature extraction, fitting, installation or publication. Citation/terms discrepancies remain visible pending source review.
