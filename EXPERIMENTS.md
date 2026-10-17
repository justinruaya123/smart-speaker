# Experiment directory

## Iris release v1

Status: the six-fit acoustic comparison and locked evaluation are closed. The release retains the predeclared compact V3 VCM after the new candidate failed its UNKNOWN nonregression guard. The release addresses ME-01–08 and UC-01–10 with a finite direct VCM, Hey Iris wake, independent experimental speaker gate, UI/actions, reproducible workflow and Pi package. Strict acoustic qualification has not passed. Original human wake-pair validation and physical Pi qualification remain operator work.

- [Final experiment report](docs/EXPERIMENT-iris-release-v1.md): architecture, methodology, comparisons, failed gates, resource costs and software evidence.
- [Protocol](experiments/iris-release-v1/protocol.json), [voice design/splits](experiments/iris-release-v1/voices.json), [input/reconstruction receipt](experiments/iris-release-v1/reproduction-inputs.json).
- [Owner protocol](experiments/iris-release-v1/owner-protocol.json), [owner data audit](experiments/iris-release-v1/owner-dataset-audit.json), [reservation receipt](experiments/iris-release-v1/owner-reservations.json).
- [Training execution](experiments/iris-release-v1/training-execution.json), [locked selection](experiments/iris-release-v1/selection-before-test.json), [command results](experiments/iris-release-v1/recognition-results.json), [promotion decision](experiments/iris-release-v1/release-choice.json).
- [Wake results](experiments/iris-release-v1/wake-results.json), [speaker results](experiments/iris-release-v1/owner-results.json), [actual runtime speaker proxies](experiments/iris-release-v1/owner-export.json), [wake-duration diagnostic](experiments/iris-release-v1/owner-wake-duration-diagnostic.json).
- [Archive receipt](experiments/iris-release-v1/publication-archive.json), [exact named cleanup manifest](experiments/iris-release-v1/archive-manifest.json), [documentation lifecycle](docs/DOCS.md).

The measurable hypothesis is improved human macro-speaker **correct acceptance**, within frozen development error limits, from source/command/speaker balancing and CE versus CE + 0.1 cross-speaker contrastive loss (temperature 0.1). Two widths × two losses at seed 53 are followed by both losses at the development-selected width at seed 67. Weights/policy/catalog/normalization lock before held-out evaluation. At least 30/36 accepted-correct personal commands plus human/official nonregression gates govern promotion. A failed candidate keeps its evidence; the predeclared same-width V3 baseline supplies runtime. No test-driven policy adjustment is permitted.

The speaker model starts from random weights and predicts a 128-D identity embedding plus overlap score, independently of command embeddings. Short human pairs, synthetic wake pairs and constructed mixtures are labelled proxies. They do not prove real human wake ownership. Recognition contains no transcription/LLM/model fallback.

Measured findings:

- The candidate is compact, CE + contrastive loss, seed 67, epoch 3. Personal correct acceptance improved from 31/36 to 35/36; wrong accepted positives changed from one to zero and negative accepts from 1/52 to 0/52. Both cohorts are synthetic.
- Fresh designed-voice correct acceptance improved from 259/444 to 320/444, wrong accepted positives from nine to three, and negative accepts from 3/120 to 1/120. These are requested synthetic conditions, not verified human demographic coverage.
- Human correct accepts improved from 165/636 to 171/636 and wrong accepted positives from 25 to 15, but UNKNOWN accepts increased from 12/229 to 14/229. That failed guard vetoed promotion. The unchanged 196-row accepted exact score improved from 138/196 to 147/196. Test failures did not cause threshold changes.
- The shipped VCM therefore remains compact V3: 969,928 parameters, 31/36 personal correct accepts, 172/196 raw and 138/196 accepted official exact commands. Low broad human acceptance and false accepts remain unresolved.
- Expanded wake training selected its new checkpoint on calibration. Held-out synthetic wake recall is 29/32, with zero duplicates and zero false wakes in 660 negative clips (1,160.81 seconds). The zero-event upper 95% false-wake bound is 9.29/hour; this is inadequate household exposure and contains no original human wake positives.
- Speaker raw proxies have 279/1005 same-speaker rejections, 72/1005 impostor accepts and 280/1005 missed overlaps. The full runtime policy rejected 62/80 same-speaker short-human proxies. Only 1/32 new standalone synthetic wake references met the locked 0.8-second voiced-speech requirement. Ownership remains experimental with substantial usability limits.

Replay caches now require complete ordered IDs and hashes for data, model, policy and scorer inputs. An independent review reproduced truncated-cache reuse before the fix. Metadata serialization and evaluation-cache fixes occurred after fitting, before held-out scoring; [the source compatibility receipt](experiments/iris-release-v1/reproduction-prefit-compatibility.json) proves the fitting and selection logic stayed equivalent. Original source locks, failed attempts and optimizer/RNG states remain preserved.

## Preserved historical findings

The complete prior experiment directory, detailed protocols, split audits, source code and raw compact evidence are available at [the immutable research snapshot](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/EXPERIMENTS.md). No legacy test population, frozen protocol or report was rewritten. Selected reusable implementation and current reproduction inputs remain in main; obsolete experiments live on the archive branch.

| Study | Finding retained | Immutable evidence |
|---|---|---|
| VCM direct V2 | Project encoder/direct-head continuation; masking and priority sampling. Personal correct acceptance 0/36 compact, 23/36 full. | [Results](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/experiments/vcm-mvp-hey-iris-v2/results.json) |
| VCM direct V3 | Development-only per-class policy; same V2 weights. Personal 31/36 compact, 30/36 full, one wrong accepted positive each; negatives 1/52 and 0/52. Official accepted 138/196 and 147/196. Human correct 165/636 and 192/636; UNKNOWN accepts 12/229 and 15/229. Broad qualification failed. | [Results](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/experiments/vcm-mvp-hey-iris-v3/results.json) |
| Own Hey Iris v1 | 82,114 parameters, synthetic test wake 12/16; zero events in 421.53 seconds of negative speech gives insufficient exposure for a strong false-wake/hour claim. No original human wake positives. | [Wake evidence](https://github.com/justinruaya123/smart-speaker/tree/ae1f1d7c67616e4efbae4619ac75620495e6073a/experiments/vcm-mvp-hey-iris-v1) |
| Earlier contextual/keyword/slot studies | Family accuracy, parser support and dynamic text catalogs did not establish complete-command recognition. Numeric/name/owner gaps and failed studies remain evidence. Earlier transcription readers are excluded from current runtime. | [Historical directory](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/EXPERIMENTS.md) |

Historical public/academic source licenses and consent restrictions continue to govern the data. The [dataset directory](docs/DATASETS.md) separates original humans, supplied synthetic data, our designed voices, ambiguous references and constructed speaker mixtures. Synthetic conditions do not establish human demographic coverage.
