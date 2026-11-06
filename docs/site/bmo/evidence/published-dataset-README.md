---
language:
- en
task_categories:
- audio-classification
license: other
license_name: source-specific-research-terms
license_link: README.md#terms-and-redistribution
pretty_name: Treated finite-command data for retained baseline A
---

# Treated baseline A dataset — shared-demo-a-v1

This version is the minimal treated input state for `shared-demo-slots-v1 / heads_s53 / epoch38`: **175,653 parameters, 34 complete commands, 102 canonical wordings, 18 intent families**. It preserves the original labels, ordered rows, splits, source identities, feature hashes and human-TRAIN-only normalization. It does not relabel speech, add a new vocabulary, establish independent listening truth or qualify a production system.

## Contents and use

| Ordered view | Rows | Human | Synthetic | Role |
|---|---:|---:|---:|---|
| train | 25,958 | 1,889 | 24,069 | Optimizer inputs only |
| original-dev | 1,692 | 256 | 1,436 | Original development selection |
| new-dev | 440 | 0 | 440 | Demo development selection |
| test | 440 | 0 | 440 | Previously evaluated reserved synthetic voices; regression reproduction |
| official | 202 | 96 | 106 | Exposed publisher benchmark, not a new blind test |

`manifest.json` pins every version file, row count and upstream shard. Gzipped JSONL records keep treated source truth separately from the restricted-demo projection (`expected` versus `demo_expected`, plus explicit action/percentage/alarm targets). Legitimate percentage zero is not missing. Unsupported values are never rounded. Row order is retained because it affects deterministic sampling. Only TRAIN/DEV rows carry predeclared feature identity expectations; TEST/official features are computed after model selection is locked.

`authored-synthesis.tar` contains **3,964 exact locally generated WAVs**: 1,764 Microsoft System.Speech utterances (1,323 TRAIN, 441 DEV) and 2,200 Microsoft Edge speech utterances (1,320 TRAIN, 440 DEV, 440 TEST). These are synthetic voices speaking authored public command prompts, not recordings of a new human cohort. Prompts, source/voice/rate, declared split, original WAV SHA-256 and decoded-PCM SHA-256 remain in the rows. Stochastic regeneration is not a way to recover these exact bytes.

**Restricted third-party audio and its derived feature arrays are not newly mirrored here.** Original recordings are acquired by immutable repository/revision/shard/row locators from their publishers. The 184 previously published designed-Qwen DEV WAVs are referenced at their existing exact paths on this dataset's main history rather than uploaded again. No numerals-only pool, unrelated experiment, calibration-fitting data, old fit cache, model checkpoint, private capture or credential is included. The fixed argmax decision policy requires no new threshold calibration.

The executable source and original comparison model are in [smart-speaker main](https://github.com/justinruaya123/smart-speaker/tree/main). The `shared_demo_a.py` reproduction entrypoint resolves this dataset's main revision once, verifies the pinned manifest/files, fetches original upstream payloads, recomputes all 28,090 TRAIN/DEV feature identities and the 1,889-human normalizer, and performs the original 48-epoch/4,608-update CUDA heads fit from random initialization. Downloading the original model does not count as retraining. See that repository's reproduction report for **actual execution status**; this dataset card alone makes no successful-reproduction claim.

## Treatment and evidence limitations

The source annotations and historical OptionB/demo projections are frozen. Human reference text includes folder/publisher and ASR-derived annotations that have not all been independently listened to. A metadata agreement is not listener-verified truth. Prior exclusions, unsupported old values and incomplete human wording coverage remain limitations; the dataset does not invent replacements for them. Public publisher datasets and these previously evaluated voices are exposed regression evidence.

Six new TRAIN, two DEV and two TEST Edge ShortNames were reserved within this study; these are not the entire historical voice population or independently verified human demographics. Historical cloned-voice aliases/reference overlap remain unresolved. Exact synthetic TEST recognition and context-added TEST recognition must be reported separately. Missing evidence is not zero accuracy or qualification.

## Provenance and citations

1. **airimonda (2026), AI231 ME2 Voice Commands.** Dataset DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723). Actual inherited TRAIN/DEV source: [revision 25111444af3adff7588d86ab27c304bb080895cf](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/tree/25111444af3adff7588d86ab27c304bb080895cf); official202 regression: [revision bfca0bbebfe17e00d6a685d1da573615d29ecdce](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/tree/bfca0bbebfe17e00d6a685d1da573615d29ecdce). The dataset DOI is the upstream composite's identifier, **not a DOI for this derivative**. Recorded shard hashes and row indices, not the DOI alone, identify the actual inputs.
2. **martinnavs (2026), AI231 ME2 Filipino-accented synthetic commands.** [Revision 230c8b8d58caf78e4bc90886d9c6fa77fe6024b8](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data/tree/230c8b8d58caf78e4bc90886d9c6fa77fe6024b8). The original card declares research/education only and no redistribution of these voices because reference-recording provenance/consent/license is unconfirmed. Download from that publisher; do not treat the synthetic labels as proof of independent people or accent validation. No new dataset DOI is asserted.
3. **Microsoft System.Speech and Microsoft Edge speech service**, generation-only tools for the authored synthesis above. [System.Speech.Synthesis API](https://learn.microsoft.com/en-us/dotnet/api/system.speech.synthesis.speechsynthesizer), [edge-tts project](https://github.com/rany2/edge-tts). Provider ShortNames, engine provenance, prompt and rate are preserved. The edge-tts software license is **not** a license for Microsoft voices or generated recordings. Service/output redistribution terms have not been independently legally audited. These assets are labelled synthetic research artifacts without an invented MIT/CC0/Apache recording license.
4. **Qwen3-TTS designed completion DEV inputs**, 184 existing generated recordings, not human audio or an additional foundation model at inference. Cite Hangrui Hu et al. (2026), *Qwen3-TTS Technical Report*, [10.48550/arXiv.2601.15621](https://arxiv.org/abs/2601.15621). The Apache-2.0 generator-model card does not itself settle every generated-output or reference right. These already published DEV files are acquired by exact original SHA-256; they do not enter TRAIN for A.

The composite's inherited scholarly sources require their own attribution. The preserved `citations.bib` and `INHERITED-CITATION-AUDIT.md` distinguish paper citations, dataset DOIs, exact acquisition gaps and source-specific terms. The latter is the original audit for earlier models, **not a new assertion of A's fitting counts or current legal clearance**:

| Inherited source | Citation | Recorded limitation |
|---|---|---|
| Fluent Speech Commands | Lugosch et al., Interspeech 2019, [10.21437/Interspeech.2019-2396](https://www.isca-archive.org/interspeech_2019/lugosch19_interspeech.html) | Academic/noncommercial and no-derivatives publisher restrictions; no new mirror |
| SLURP | Bastianelli et al., EMNLP 2020, [10.18653/v1/2020.emnlp-main.588](https://aclanthology.org/2020.emnlp-main.588/) | Audio CC BY-NC; text CC BY; inherited original-archive mapping incomplete |
| SNIPS | Saade et al., *Spoken Language Understanding on the Edge*, EMC2/NeurIPS 2019, [publisher repository/citation](https://github.com/sonos/spoken-language-understanding-research-datasets) | Noncommercial research; modified-dataset publication restrictions; no unverified DOI inserted |
| Common Voice | Ardila et al., LREC 2020, [official proceedings](https://aclanthology.org/2020.lrec-1.520/) | Acquisition-era release/clip/terms mapping incomplete; present publisher terms are not retroactive clearance |
| Timers and Such | Lugosch et al., NeurIPS Datasets and Benchmarks 2021; dataset [10.5281/zenodo.4623772](https://zenodo.org/records/4623772) | Paper and dataset identifiers differ; inherited exact clip mapping incomplete |
| Multi-Sensor Voice Command Dataset | Rusci, Van hamme and Tuytelaars, dataset [10.48804/IEKKVZ](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/IEKKVZ) | Publisher derivative-download tracking terms; inherited original version unresolved |
| Class/student speech and group synthesis | Attribute the pinned airimonda card and its original contributors | Public speaker consent and cloned-reference lineage not independently established; no new raw-audio mirror |

## Terms and redistribution

This is a **source-specific research/education dataset**, not a uniformly open-licensed audio collection. Code, model weights, metadata, voice-generator software and recording/output rights are separate. Citations do not grant permission. Preserve publisher terms and consent restrictions; obtain required original-source access independently. Do not redistribute restricted voices or upstream audio/features from this locator release, use it commercially, infer identities or claim independent human/accent validation. No new DOI or legal-clearance certification is claimed.

Existing unrelated root dataset files and earlier dataset versions are preserved by this additive publication; their presence is not a new rights endorsement.
