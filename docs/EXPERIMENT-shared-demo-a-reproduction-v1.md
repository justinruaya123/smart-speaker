# Baseline A treated-data publication and fresh-main reproduction v1

## Result

**Completed actual retraining from freshly acquired GitHub/Hugging Face main.** The retained **175,653-parameter heads_s53 model selected epoch38 again**. All41selected tensors, final-epoch tensor fingerprints, all48loss/development curve events excluding elapsed time, ordered draws, per-recording sample counts and source quotas match the original. The newly exported ONNX graph is **byte-identical** to the original graph, and all642original evaluation decisions match.

This reproduces the existing scientific model; it is not an acoustic improvement, new model selection, installation, action enablement or production qualification.

## Exact inputs and source

| Item | Actual identity |
|---|---|
| Fresh fitting source, GitHub main | `64c3ea19bb0a0f0bd52866ad04f6c8b41b5a92e4` |
| Fresh dataset acquisition, Hugging Face main | `13d388c31985be7f42af56b46570281728ecc054` |
| Dataset version | `shared-demo-a-v1/` |
| Dataset manifest SHA-256 | `98ea697e4cea02157a04835843ce1f9d8ca6f8b1420eecff4180dc7498aeed5a` |
| Original model | `shared-demo-slots-v1 / heads_s53 / baseline A` |
| Neural parameters | 175,653 |
| Scope | 34complete commands,102canonical wordings,18intent families; no Kitchen/Ceiling names |
| Scientific environment | New CPython3.13.14 environment, Torch2.10.0+cu128, NumPy2.3.5, CUDA12.8/BF16, RTX3080 |

The source was pulled through the remote main branch into a separate clone, not copied from the dirty research checkout. The39-package environment has no inherited site-packages; `uv pip check` passed. Hugging Face payloads and original-source shards were acquired into a new empty run-owned download cache. Existing feature/fit caches and old trained weights were not training initializers. Original checkpoint loading is only for post-fit equality comparison.

The historical fitter and minimal import closure are preserved byte-for-byte under `submission/shared-demo-a/source/`. Main's newer producer implementation and the consumer application were not replaced. The earlier source publication is `86f2630`; two additive engineering corrections are represented by the actual fitting revision above. Later report-only commits do not change the consumed fitting source.

## Necessary treated dataset and citations

[Dataset main](https://huggingface.co/datasets/justinruaya123/ai231-smart-speaker/tree/main/shared-demo-a-v1) publishes the frozen ordered labels, complete typed source references and demo projections, split and voice identities, original WAV/PCM locators, expected feature hashes and unchanged human-TRAIN-only normalization. New payloads are only the3,964necessary locally authored synthetic WAVs:1,764System.Speech and2,200Edge utterances. The184existing designed-Qwen DEV WAVs are referenced at their already published main-history paths, not uploaded again.

| View | Rows | Human | Synthetic |
|---|---:|---:|---:|
| TRAIN | 25,958 | 1,889 | 24,069 |
| Original DEV | 1,692 | 256 | 1,436 |
| New DEV | 440 | 0 | 440 |
| Synthetic TEST | 440 | 0 | 440 |
| Exposed official regression | 202 | 96 | 106 |

All12declared version files were downloaded and hashed; nine original publisher Parquet shards were independently rehashed. All28,090TRAIN/DEV feature arrays match the frozen shape/dtype/content hashes. The normalizer exactly matches the original1,889-human equal-utterance TRAIN values and membership. The compact audit reconciles all28,732treated reference rows across the five views.

No restricted third-party audio or derived feature corpus is newly mirrored. No numerals-only auxiliary pool, unrelated experiment, checkpoint, private capture or credential is in the new dataset version. Calibration is not fitted: the original fixed action-authoritative argmax policy remains unchanged.

The [dataset card](https://huggingface.co/datasets/justinruaya123/ai231-smart-speaker/blob/main/shared-demo-a-v1/README.md), [BibTeX](https://huggingface.co/datasets/justinruaya123/ai231-smart-speaker/blob/main/shared-demo-a-v1/citations.bib), pinned publisher-card hashes and preserved citation audit distinguish:

- Upstream AI231 dataset DOI `10.57967/hf/10723` from this derivative, which has no new DOI; immutable revisions/row locators identify actual bytes.
- martinnavs supplemental research/education-only, nonredistribution and unresolved cloned-reference consent/license terms.
- Original scholarly recording sources and their dataset-versus-paper identifiers and per-source restrictions.
- Offline Microsoft synthesis software/service provenance versus unverified output-rights assumptions; no MIT/CC0/Apache recording license is invented.
- Qwen generator-model attribution versus generated-output/reference rights and independent human evidence.

Citations do not establish legal clearance. Human annotation/listening gaps, opaque historical voice aliases, consent limitations and synthetic demographic/accent uncertainty remain explicit.

## Actual optimizer execution and equality

The unchanged heads recipe executes6human-foundation +42mixed epochs,96updates/epoch, batch32: **4,608updates and147,456draws**. A separate four-update smoke completed before the full run; it is not included in the full-fit count. Actual full fitting plus per-epoch development evaluation took1699.97seconds on this host; download/preparation/export/audit time is excluded.

| Realized full-fit sampling group | Draws |
|---|---:|
| human_positive | 46,080 |
| human_negative | 20,736 |
| synthetic_positive | 32,256 |
| demo_slot_positive | 32,256 |
| synthetic_negative | 16,128 |

Initialization, quotas, RNG/dropout reseeding, augmentation, optimizer/LR trajectory, labels, human-retention floor and DEV-only selection are unchanged. The flat control was not retrained: the task reproduces the already selected best heads arm, not a new architecture comparison. No TEST-driven fitting or threshold selection occurred. The reproduced selection lock preceded TEST/official feature extraction and inference.

| Identity check | Outcome |
|---|---|
| All41selected tensors | Bit-identical |
| Selected epoch and selected DEV metrics | Identical, epoch38 |
| Final48th-epoch tensor fingerprint | Identical |
| Full48-event loss/DEV curve, excluding elapsed time | Identical |
| Ordered sample sequence, source quotas and per-ID counts | Identical |
| Original and reproduced ONNX bytes | Identical |
| Torch versus ONNX logits/decisions | Passed; max absolute logit difference7.62939453125e-6 |
| Original versus reproduced complete decisions | Identical on all642recordings |

Original/reproduced ONNX SHA-256:

`9b98acefa8bad6610d98501ff0310c9b2a2ec1594619b4b5b7d2c7a80dc888a6`

**Serialization distinction:** the entire native `.pt` files are not byte-identical. The selected tensors and every non-weight metadata value match; configuration-dictionary insertion order differs after portable JSON reconstruction. Original native SHA-256 is `0164568747c67657836659dc1322947d6282d829717b93efe218269b2b456069`; reproduced native SHA-256 is `8e759891e55cc98f0381f20bf6299d796b12bf8cc0f2aefdccdb14a740cc29a5`. The newly written normalizer JSON differs only by CRLF versus LF; all parsed values and membership match. These serialization differences are reported, not disguised as matching file hashes. The original trusted bundle is preserved unchanged.

## Same outcomes, including limitations

| Cohort | Correct supported commands | Rejected supported | Wrong accepted supported | Negative false accepts |
|---|---:|---:|---:|---:|
| Exact synthetic TEST | 203/204 | 1 | 0 | N/A |
| Context-added synthetic TEST | 98/204 | 105 | 1 | N/A |
| All synthetic TEST | 301/408 | 106 | 1 | 0/32 |
| Exposed human supported commands | 71/77 | 3 | 3 | Separate below |
| All official202, restricted-demo scope | 155/162 | 4 | 3 | 8/40 |

The40official task negatives include formerly supported values now outside this restricted demo as well as original UNKNOWN. They are not silently removed. The full96-row human official stratum contains77supported and19task-negative cases; its false accepts remain5/19. Existing reports retain the original-UNKNOWN versus formerly-supported breakdown. Exact synthetic success does not erase the severe context failure, wrong supported commands or open-set false accepts. These are exposed regression populations, not fresh independently reviewed human recordings.

## Engineering failures and software checks

- The first isolated dependency command rejected a uv-only index strategy written inside the requirements file. It was moved to the CLI; the fresh installation then completed with39compatible packages.
- The first fresh acquisition verified the full new dataset files, then stopped before features/optimization because the verifier rejected a legitimate Hugging Face snapshot-to-blob symlink with correct bytes. The wrapper also anchored relative outputs to its isolated source working directory. The failed attempt is preserved and pinned by `reproduction-attempt1.json`; no optimizer updates occurred there.
- The correction accepts only hash-verified cache links whose resolved target remains inside the declared cache, and anchors outputs inside repository-root `.cache`. Four new RED-to-GREEN tests cover valid/corrupt/escaping cache links and output paths. No original data/feature hash or fitting gate was relaxed. The successful attempt used a new empty cache, not repaired old feature/fit output.
- Final scoped software verification: **198passed,4skipped**, including26reproduction guards. The earlier whole-repository suite had621passed/14failed/8skipped versus untouched main606passed/the same14failed node IDs/8skipped. The15-test difference was the initial guard set; later added guards passed separately. All inherited failure names remain in `publication-tests.json`.
- GitHub's model-only boundary CI passes. The separate Iris publication workflow still fails its existing UC-01browser-layout assertion and wake acoustic gate; Pages deployment is skipped. No all-green repository/website/acoustic claim is made.
- The external reviewer failed at its model provider and supplied no verdict. Direct review, RED-to-GREEN guards, actual complete fitting and independently recomputed primitive accounting are the evidence; external approval is not claimed.

## Evidence and commands

[Portable reproduction commands](REPRODUCIBILITY-shared-demo-a.md). Compact actual receipts live in `submission/shared-demo-a/reproduction-v1/`: source/data acquisition, preparation, full curve, training summary/comparison, pre-TEST lock, predictions, evaluation, environment and independent audit. `frozen-local-audit.py` preserves the exact machine-local audit invocation/source by hash; it is historical verification evidence, not a second portable training implementation. The supported portable entrypoint remains `python scripts/shared_demo_a.py all --output .cache/NEW_RUN`.

The successful untouched checkout is `D:/School/AI231/smart-speaker/.cache/hf-main-reproduction-v1/reproduce`; run artifacts are `.cache/fresh-main-02/`, including `heads_s53/best.pt`, `heads_s53/last.pt`, `reproduced.onnx`, exact prepared features and full raw receipts. No duplicate graph needs publishing because the reproduced ONNX is the exact graph already committed at `submission/models/shared-demo-a/model.onnx`.

Runtime/default configuration, microphones, speaker/echo processing, provider actions, physical Pi and fresh-human qualification were not changed or exercised. Qualification and action permissions remain false.
