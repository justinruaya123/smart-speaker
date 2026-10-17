# Dataset citations and publisher terms

Verified 2026-10-17 (Asia/Manila). This bibliography supports the 935,567-parameter `speaker-pipeline-repair-v1` and 2,066,831-parameter `speaker-final-refinement-v1` reproducibility documentation. [DATASETS.md](DATASETS.md) records local versions, selections and receipts. A source citation establishes attribution; it does not establish that every recording in that source trained either model or that a derivative may be republished.

The fitting manifests contain the six scholarly recording sources below, student recordings, `group_synthetic` and `fil50_persona`. The separate MLEnd/Speech Commands numeral pool did not fit either winning model. Local Qwen completion recordings belong to the shared synthetic development/calibration comparison, not either winning fit. Keep these distinctions in the Hugging Face card and slide references. Downloadable [BibTeX](dataset-citations.bib) includes the main proceedings entries plus the pinned supplement, dictionary, generator and reference-only citations. [Submission fields](REPRODUCIBILITY-935k-2m.md) map those citations to the supplied slide template.

## Dataset-location field for the slides

**Upstream dataset:** airimonda (2026), *AI231 ME2 Voice Commands*, Hugging Face, dataset DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723). The [publisher citation](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands#citation) lists this DOI. The actual experiment input is revision `25111444af3adff7588d86ab27c304bb080895cf`, bound by the [source receipt](../data/sources/ai231-primary.source.json) and [saved card](../experiments/pcv-source-refresh-v1/source-card.txt). Cite both DOI and revision: the DOI alone does not identify these exact bytes.

**Additional fitting source:** martinnavs (2026), *AI231 ME2 Filipino-accented synthetic commands*, revision `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`; see the [publisher card](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data), [pinned source receipt](../data/sources/fil-supplemental.source.json) and [saved notice](../data/sources/fil-supplemental-NOTICE.txt). No dataset DOI was found in these sources.

**Access terms:** research/education with per-source restrictions; the supplement's publisher prohibits redistribution while reference-recording terms remain unconfirmed. This is not an MIT-licensed recording collection. An upstream DOI is not a DOI for a newly prepared `justinruaya123/ai231-smart-speaker` derivative. No new derivative DOI or publication was created by this audit.

## Scholarly recording sources confirmed in fitting

| Manifest source | Citation and identifier to use | Publisher terms and remaining provenance |
| --- | --- | --- |
| `FluentSpeechCommands` | Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar and Yoshua Bengio (2019). Interspeech, pp. 814–818. **Paper DOI:** [10.21437/Interspeech.2019-2396](https://www.isca-archive.org/interspeech_2019/lugosch19_interspeech.html). No separate dataset DOI found in the inspected publisher sources. | The [Fluent.ai dataset page](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) states academic research only and CC BY-NC-ND 4.0. Its linked [Fluent Speech Commands Public License](https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf), section 2, limits use to noncommercial academic purposes and excludes commercial training, testing, benchmarking, industrial research and product development. Preserve the complete license; an aggregate card or mirror tag does not grant broader terms. |
| `SLURP` | Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski and Verena Rieser (2020). EMNLP, pp. 7252–7262. **Paper DOI:** [10.18653/v1/2020.emnlp-main.588](https://aclanthology.org/2020.emnlp-main.588/). | The [original repository license section](https://github.com/pswietojanski/slurp#license) distinguishes **text: CC BY 4.0; audio: CC BY-NC 4.0**. The composite card's blanket CC BY 4.0 description of SLURP does not match the publisher's audio terms. The exact original audio archive/version of the inherited rows remains to be mapped; the paper DOI is not that archive identity. |
| `SNIPS` | Alaa Saade et al. (2019), *Spoken Language Understanding on the Edge*, Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing, colocated with NeurIPS. Use the [official conference paper](https://www.emc2-ai.org/assets/docs/neurips-19/emc2-neurips19-paper-19.pdf) and the [publisher's requested citation](https://github.com/sonos/spoken-language-understanding-research-datasets#license-summary). | [Sonos Crowdsourced Data License](https://github.com/sonos/spoken-language-understanding-research-datasets/blob/master/LICENSE), sections 1–2: noncommercial academic/research use; redistribution of unmodified datasets under the same terms, with the full paper citation; modified dataset publication prohibited. Access is requested through the publisher's form. The inherited rows need exact original dataset/configuration mapping. **Candidate paper DOI, not primary-verified in this audit:** `10.1109/EMC2-NIPS53020.2019.00021`; publisher/Crossref endpoints were inaccessible. No dataset DOI found in inspected primary sources. |
| `CommonVoice_en` | Rosana Ardila et al. (2020), *Common Voice: A Massively-Multilingual Speech Corpus*, LREC, pp. 4218–4222; [official proceedings](https://aclanthology.org/2020.lrec-1.520/). The ACL record lists no proceedings DOI. The [arXiv identifier](https://arxiv.org/abs/1912.06670) identifies the paper, not release 19. | The composite card identifies **Common Voice 19 English**; Mozilla's [release announcement](https://discourse.mozilla.org/t/common-voice-19-0-dataset-release/135857) is dated 18 September 2024. Current [Mozilla terms](https://commonvoice.mozilla.org/dav/terms), effective 31 October 2025, state CC0 and request access through Mozilla Data Collective without external reposting/mirroring. Record that current publisher statement separately from the still-missing acquisition-era terms, original archive checksum and clip IDs; this audit does not establish a retroactive change to acquired CV19 rights. No release-19 dataset DOI found in inspected primary sources. |
| `TimersAndSuch` | Loren Lugosch, Piyush Papreja, Mirco Ravanelli, Abdelwahab Heba and Titouan Parcollet (2021). [NeurIPS Datasets and Benchmarks proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/34173cb38f07f89ddbebc2ac9128303f-Abstract-round1.html). **Dataset archive DOI:** [10.5281/zenodo.4623772](https://zenodo.org/records/4623772), *Timers and Such v1.0*, creator Loren Lugosch, 19 March 2021. | The [author paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/34173cb38f07f89ddbebc2ac9128303f-Paper-round1.pdf), section 1, states **CC0** for the dataset. This corrects the composite card's nonspecific `other-open` entry. The archive DOI and paper citation are distinct. No proceedings DOI found in the inspected official record. The inherited clip-to-v1.0 mapping remains unverified. The proceedings HTML duplicates Parcollet's name; the PDF lists five authors, used here. |
| `xela_Multi-Sensor` | Manuele Rusci, Hugo Van hamme and Tinne Tuytelaars (2024), *Multi-Sensor Voice Command Dataset*, KU Leuven RDR. **Dataset DOI:** [10.48804/IEKKVZ](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/IEKKVZ). Related paper: Rusci, Van Hamme and Tuytelaars (2025), *Self-Incremental Training for Personalized Voice Command Recognition in a Wireless Audio Sensor Network*, ICASSP, **paper DOI:** `10.1109/ICASSP49660.2025.10887592`, listed in the dataset's publication metadata. | Publisher custom terms: CC BY 4.0 **plus tracking partial and full downloads of derivatives**. The page presents version 1.2 and its citation uses V1; neither proves which original version Xela's inherited recordings use. Preserve the publisher download/guestbook terms and original record mapping. The dataset DOI and related paper DOI must not be interchanged. |

## Additional fitting data and text resource

| Source | Attribution, terms and missing evidence |
| --- | --- |
| Student `real_voice`, `xela_SET_TEMPERATURE_REAL` and other Xela recordings | Attribute AI231 ME2 contributors/Xela through the [pinned primary card](../experiments/pcv-source-refresh-v1/source-card.txt). The card limits sharing outside the class to recordings with the relevant speaker consent. No independent dataset DOI, complete contributor citation or public-release consent receipt was established. Preserve any consent records privately; do not expose student identifiers merely to fill an authors field. |
| `group_synthetic` | The same card attributes `markandrian30/AI231` and `MEX2/Data`, with voices cloned from LibriSpeech and SilencioPH references. Exact immutable generator repositories, reference identities and output terms are missing. No independent dataset DOI found in the inspected card. Cite the upstream composite and identified original references without treating synthetic voice IDs as independent people. |
| `fil50_persona` | Attribute martinnavs and the pinned supplement above. Its [publisher terms](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data#sources-and-licenses) record missing source/consent/license for 17 reference recordings and an unverified CosyVoice2 weights license, with research/education only and no redistribution/commercial use. The saved notice is the authority for the acquired revision. Current model metadata cannot fill the missing reference provenance. |
| CMUdict pronunciations | Cite Carnegie Mellon University, *The CMU Pronouncing Dictionary*, revision `74790861f652b15e4ac49015a90074ad62a27690`; [original repository](https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690), [publisher license](https://github.com/cmusphinx/cmudict/blob/74790861f652b15e4ac49015a90074ad62a27690/LICENSE), [local receipt](../data/sources/cmudict.source.json). BSD-style terms require retaining notices/conditions/disclaimer. No DOI found in inspected sources. This is a text lexicon, not a recording corpus or pretrained acoustic model. |

## Auxiliary, reference and historical sources

These citations retain upstream attribution without adding the sources to either model's fitting denominator.

| Resource and role | Citation / identifier and qualification |
| --- | --- |
| MLEnd Spoken Numerals — auxiliary pool only | Queen Mary University of London / MLEnd, *MLEnd Spoken Numerals*, [publisher page](https://mlenddatasets.github.io/spoken_numerals/) and [download documentation](https://mlend.readthedocs.io/en/latest/api.html). No dataset DOI or explicit recording license found on the inspected primary pages; do not substitute an unrelated MLEnd dataset or the Python library's license. The composite's 32,029 rows are auxiliary, not fitted complete commands. Historical acquisition revision remains missing. |
| Google Speech Commands v0.02 — auxiliary digits only | Pete Warden (2018), *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*. **Paper/preprint DOI:** [10.48550/arXiv.1804.03209](https://arxiv.org/abs/1804.03209); [official TensorFlow citation](https://www.tensorflow.org/datasets/catalog/speech_commands). No separate dataset DOI found. The saved primary card declares CC BY 4.0; retain the acquired archive's own license alongside its [source receipt](../data/sources/speech_commands.source.json). Its 34,361 composite digit rows did not fit these winners. |
| LibriSpeech — inherited synthetic voice references; historical studies | Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015), *LibriSpeech: An ASR Corpus Based on Public Domain Audio Books*, ICASSP, pp. 5206–5210. **Paper DOI:** `10.1109/ICASSP.2015.7178964`, also reproduced in the KU Leuven publisher metadata above. [OpenSLR SLR12](https://www.openslr.org/12) states CC BY 4.0. No separate dataset DOI found in the inspected release page. The original recording license does not independently establish the terms or consent lineage of cloned outputs. |
| Qwen3-TTS — local development/calibration generation assets | Hangrui Hu et al. (2026), *Qwen3-TTS Technical Report*. **Paper/preprint DOI:** [10.48550/arXiv.2601.15621](https://arxiv.org/abs/2601.15621). The [pinned VoiceDesign card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/blob/5ecdb67327fd37bb2e042aab12ff7391903235d3/README.md) states Apache-2.0; the [asset receipt](../data/sources/qwen-synthetic-completion.source.json) pins VoiceDesign and Base. This is a generator-model citation, not a dataset DOI or a rights determination for generated audio/anchors. These new recordings were excluded from both winning fits. |
| Free Spoken Digit Dataset — historical studies only | Zohar Jackson, César Souza, Jason Flaks, Yuxin Pan, Hereman Nicolas and Adhish Thite (2018), *Jakobovski/free-spoken-digit-dataset: v1.0.8*, [archived release DOI 10.5281/zenodo.1342401](https://zenodo.org/records/1342401). The [original repository](https://github.com/Jakobovski/free-spoken-digit-dataset) states CC BY-SA 4.0. This is an available archived citation, **not a recovered identity of the historical 3,000-clip local input**. No confirmed fitting rows in these winners. |

MSWC, DEMAND and MS-SNSD have no confirmed fitting rows for these two winners. They remain historical or later composite-card sources in [DATASETS.md](DATASETS.md); do not place them in a slide's winning-model training list merely because they appear elsewhere in this repository. Likewise, pretrained wake/VAD assets require their own model/software attribution, not a recording-dataset row.

## BibTeX for the main dataset and verified proceedings citations

These entries preserve dataset versus paper identity. The SNIPS candidate DOI is deliberately omitted until primary metadata can be retrieved. For the remaining sources, the tables above provide the complete citation links and exact version qualification.

```bibtex
@misc{ai231_me2_voice_commands,
  author = {airimonda},
  title = {AI231 ME2 Voice Commands},
  year = {2026},
  publisher = {Hugging Face},
  doi = {10.57967/hf/10723},
  url = {https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands},
  note = {Experiment revision 25111444af3adff7588d86ab27c304bb080895cf; per-source terms}
}

@inproceedings{lugosch2019fluent,
  author = {Lugosch, Loren and Ravanelli, Mirco and Ignoto, Patrick and Tomar, Vikrant Singh and Bengio, Yoshua},
  title = {Speech Model Pre-Training for End-to-End Spoken Language Understanding},
  booktitle = {Interspeech 2019},
  year = {2019},
  pages = {814--818},
  doi = {10.21437/Interspeech.2019-2396},
  url = {https://www.isca-archive.org/interspeech_2019/lugosch19_interspeech.html}
}

@inproceedings{bastianelli2020slurp,
  author = {Bastianelli, Emanuele and Vanzo, Andrea and Swietojanski, Pawel and Rieser, Verena},
  title = {{SLURP}: A Spoken Language Understanding Resource Package},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year = {2020},
  pages = {7252--7262},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2020.emnlp-main.588},
  url = {https://aclanthology.org/2020.emnlp-main.588/}
}

@inproceedings{saade2019edge,
  author = {Saade, Alaa and Coucke, Alice and Caulier, Alexandre and Dureau, Joseph and Ball, Adrien and Bluche, Th{\'e}odore and Leroy, David and Doumouro, Cl{\'e}ment and Gisselbrecht, Thibault and Caltagirone, Francesco and Lavril, Thibaut and Primet, Ma{\"e}l},
  title = {Spoken Language Understanding on the Edge},
  booktitle = {Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing, colocated with NeurIPS},
  year = {2019},
  url = {https://www.emc2-ai.org/assets/docs/neurips-19/emc2-neurips19-paper-19.pdf}
}

@inproceedings{ardila2020commonvoice,
  author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Kohler, Michael and Meyer, Josh and Henretty, Michael and Morais, Reuben and Saunders, Lindsay and Tyers, Francis and Weber, Gregor},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  year = {2020},
  pages = {4218--4222},
  publisher = {European Language Resources Association},
  url = {https://aclanthology.org/2020.lrec-1.520/}
}

@inproceedings{lugosch2021timers,
  author = {Lugosch, Loren and Papreja, Piyush and Ravanelli, Mirco and Heba, Abdelwahab and Parcollet, Titouan},
  title = {Timers and Such: A Practical Benchmark for Spoken Language Understanding with Numbers},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  volume = {1},
  year = {2021},
  url = {https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/34173cb38f07f89ddbebc2ac9128303f-Abstract-round1.html}
}

@misc{lugosch2021timersdata,
  author = {Lugosch, Loren},
  title = {Timers and Such v1.0},
  year = {2021},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.4623772},
  url = {https://zenodo.org/records/4623772},
  note = {Archived dataset citation; inherited clip-to-version mapping remains unverified}
}

@misc{rusci2024multisensor,
  author = {Rusci, Manuele and Van hamme, Hugo and Tuytelaars, Tinne},
  title = {Multi-Sensor Voice Command Dataset},
  year = {2024},
  publisher = {KU Leuven RDR},
  doi = {10.48804/IEKKVZ},
  url = {https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/IEKKVZ},
  note = {Publisher citation V1; inherited recording version remains unverified}
}
```

Before marking the slide's “dataset licensed and citable” item complete, retain the per-record original source/version mapping and publisher notices, resolve supplement/reference and student consent evidence, and decide which payloads can be distributed under their recorded terms. The repository code license and model-weight license are separate fields. Missing evidence in this ledger is not a finding that permission exists or that no DOI exists anywhere.
