# Draft dataset card additions — 935k and 2M reproducibility evidence

Prepared 2026-10-17 (Asia/Manila). Target: `justinruaya123/ai231-smart-speaker`. This is a proposed **addition** to its existing root README, not an upload-ready replacement. Preserve existing configurations and merge their source notices after review. Do not invent final license metadata, audio configs, counts or a release commit before export/verification. See [shipping plan](PLAN-shipping-and-huggingface.md), [dataset selections and mutations](DATASETS.md), [publisher citations and DOIs](DATASET-CITATIONS.md), and [submission/reproduction evidence](REPRODUCIBILITY-935k-2m.md). Relative links in this local draft must be remapped to the actual Hub release layout before upload.

## Dataset description and intended use

This research bundle documents the exact data used by **both** submitted finite Voice Command Models: `speaker-pipeline-repair-v1` (935,567 parameters, saved mixed epoch 28, FP32) and `speaker-final-refinement-v1` (2,066,831 parameters, control mixed epoch 32, INT8). Recognition emits a structured command or abstention on device; it contains no runtime ASR/STT, LLM or remote inference. These are complete-utterance endpoint models; the streaming-causal/KV-cache diagram in the assignment template does not describe them. Upstream annotation metadata includes source transcription checks; those are disclosed as label provenance, not independently validated speech transcripts.

Intended use is bounded English smart-device command and rejection research, with real and synthetic origin reported separately. The finite schema has 18 supported command intents plus UNKNOWN, 93 official baseline wordings and 415 project phrase definitions. Dictionary/template coverage does not establish acoustic coverage or successful device actions. This release does not establish arbitrary transcription, unrestricted names or human-safe deployment.

The proposed release contains frozen source identities, example selections, exclusion reasons, train/development/calibration/test/holdout assignments, finite command catalogs, normalization/frontend identity, dependency locks and development selection locks. Recording distribution is source dependent; the initial evidence release does not promise to mirror every recording. If an input requires permission or is unavailable, reconstruction must stop with a precise missing-source report rather than silently substitute audio.

## Sources and attribution

Primary: airimonda (2026), *AI231 ME2 Voice Commands*, [upstream dataset DOI 10.57967/hf/10723](https://doi.org/10.57967/hf/10723), Hub revision `25111444af3adff7588d86ab27c304bb080895cf`. Supplemental: martinnavs (2026), *AI231 ME2 Filipino-accented synthetic commands*, revision `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`; no verified dataset DOI in this audit. Do not cite the upstream primary DOI as this derivative release's DOI, or assume it resolves to the historical fitted commit.

Direct fitting sources inherited through the primary are **SLURP, Fluent Speech Commands, SNIPS/SmartLights, Common Voice English (publisher card identifies v19), Timers and Such, Multi-Sensor Voice Command, student recordings, separately labeled Xela temperature recordings and existing group synthetic commands**. The supplement supplies Filipino-persona synthetic commands. Original paper references, actual dataset DOIs where available and per-source row counts are in the accompanying citation/source ledger. Original release/clip mappings inside the composite remain incomplete for some sources.

Local completion comprises six designed Qwen3-TTS voice conditions and 552 generated commands, with generation receipts and separate synthetic splits. **Zero new completion command rows enter either selected model's fitting pool**; 184 participate in shared development and 184 in synthetic calibration. The losing 2M completion arm has the other 184 training rows. Existing group/supplement synthetic recordings are separate generated sources with their own reference ancestry; this exclusion does not mean the selected models were trained without synthetic audio. CMUdict supplies text pronunciations at revision `74790861f652b15e4ac49015a90074ad62a27690`.

Audio keeps its original terms. SLURP text is CC BY 4.0 and audio CC BY-NC 4.0. FSC and SNIPS have publisher restrictions; Timers and Such's publisher identifies CC0. Student/Xela consent, cloned-reference rights and Multi-Sensor download-tracking conditions require source-specific handling. The pinned supplement explicitly withholds redistribution while reference rights are unresolved. Generator-code/model licensing alone does not settle output or reference-voice rights. Do not assign one permissive license to this merged corpus or infer audio terms from mirror tags. Preserve unresolved audio and private metadata outside public staging. Access and attribution evidence is documented in the citation ledger; a DOI is not a redistribution permission.

## Exact experiment selection

The source installation inventory is 97,659 rows; inventory size is not the model training denominator. The 2026-10-17 manifest audit gives:

| View | 935k model | 2M model |
| --- | --- | --- |
| Eligible fitting | **23,289: 1,863 human + 21,426 synthetic** | **23,295: 1,869 human + 21,426 synthetic** |
| Stored fitting duration | 14.033144 hours | 14.038888 hours |
| Declared fitting identities | 173 human IDs + 70 synthetic voice IDs | 179 human IDs + 70 synthetic voice IDs |
| Shared development | 1,251: 256 human + 995 synthetic | Same |
| Human calibration | 210 | Same |
| Synthetic calibration/diagnostic | 1,197 | Same |
| Published test / holdout | 4,418 / 196 | Same |

These are eligible sampler pools, not evidence that each row was drawn. Durations are summed from stored decoded sample metadata, without new audio decoding. IDs are source-qualified declared identities, not verified independent people. Twenty-six human training examples were excluded during preparation for pronunciation/text issues; compact fitting additionally excludes six unresolved Timers and Such slot-metadata conflicts. Both reuse the unchanged normalization fitted on the parent's 1,869 eligible human training utterances. Do not silently harmonize this six-row difference in a reconstruction.

The 23,479-row completion comparison manifest is a **superset**, not either winner's fitting set. The **66,390 isolated numerals = 32,029 MLEnd + 34,361 Speech Commands v2** remain auxiliary, outside both final complete-command fitting pools. FSDD, MSWC and the target's extra noise/negative configurations have no confirmed direct fitting rows for these models; historical experiments and synthetic reference corpora are distinct roles.

Published source test and holdout retain 4,418 and 196 rows respectively, including 185 and two uncertain references outside final typed scoring (eligible totals 4,233 and 194). Preserve these denominators and eligibility flags. Test has 1,050 human and 3,368 synthetic rows before reference exclusions; holdout has 96 human and 100 synthetic. The official `vcm-benchmark` replay separately scores all 196 published holdout references. Its denominator and semantics must not be mixed with the project typed-scoring view. Prior evaluation exposure means these are locked regression results, not a new blind holdout.

Training uses source-qualified speaker partitions, explicit identity exceptions and deduplication from the frozen installer, original human-only normalization, grouped sampling and training-only feature masks. The fresh metadata comparison finds zero intersections across all ten pairs of command train/dev/calibration/test/holdout sets for stored example IDs, speaker IDs, pair IDs and PCM hashes. This does not audit speaker aliases, cloned-reference overlap or near-duplicate acoustics. Preserve exact row assignments rather than relying only on a seed. The protocol and pre-test selection receipts identify every data/catalog/model choice. Generated labels represent intended prompt text and have not been independently listened to.

## Data instances and reconstruction fields

Retain stable `id`, upstream `source`/`source_revision`/Hub configuration/shard/row, `published_split`, experimental `split`, human/synthetic `kind`, source-qualified `speaker_id`, `pair_id`, original and derived audio SHA-256, duration/sample format, raw `publisher_metadata`, derived typed `command`, reference status/reason, `metric_eligible`, slot-conflict and QA/exclusion flags. Record original and normalized supervision separately. Export local paths in a separately hashed portable representation with a reversible map; do not edit frozen absolute-path manifests in place. A path or a SHA-256 string alone is not downloadable audio.

## Relationship to the existing Hub data

The target's pre-existing commit `3ed018c1bc731124a1b2ac5bd6653b5cd01bf576` contains different primary counts and additional supplemental/negative configurations. Those are not interchangeable with the final experiment inputs. The release evidence must map each selected example to its exact upstream revision and configuration. Loading the target's `default` at `main` does not reproduce this experiment.

## Reproduction contract

The final published bundle must provide an immutable Hub commit, code commit/source archive hash, dependency locks, original source shard hashes, portable example IDs, separate exact 935k/2M fitting memberships, frozen split manifests, original and derived audio hashes, frontend/normalization, catalog and selection locks. The local `model-dataset-usage.json` already binds eight inspected input manifests, 37 evidence/source files, selected checkpoints, archive/ONNX hashes, declared-identity counts and stored duration totals. It is aggregate evidence, not a portable audio export or complete source archive. Byte-preserved generated commands and anchors are needed for exact synthetic input recovery; seed-only regeneration is insufficient. CUDA CTC fitting is not bitwise deterministic.

An evidence-only release is downloaded at its fixed commit and used to reconstruct allowed inputs from upstream before fitting. It is not exposed as an audio dataset through generic manifest discovery. Any later audio configuration must have explicit README `configs`, cleared byte-backed audio and a clean-machine load check.

Before publication, replace this paragraph with the actual tested export/reconstruction command, clean-machine environment, immutable source and derivative-dataset commits, included configurations and their verified row/hash counts. These public-release artifacts and a clean-clone one-command reproduction are not established by the current documentation audit. Existing study-specific commands require the prepared local inputs and must not be advertised as a portable fresh-clone workflow.

## Limitations and qualification

Neither model is release-qualified. Human calibration lacks CALL, CREATE_REMINDER, LIST_REMINDERS, MESSAGE and TEMPERATURE; synthetic completion does not close those human evidence gaps. Qualified acceptance remains disabled. Demo thresholds are separate application rules and do not establish qualification; compact uses a frozen per-intent recall policy and 2M uses fixed .95 intent/.90 slot gates. Alias-resolved speaker/voice overlap, near-duplicate audio, listener-adjudicated labels, independent unseen-human testing and physical Pi/live microphone measurements remain incomplete.

Report per-intent, complete-command and rejection results with their actual denominators and preserve failed gates. Dataset release, reproducible software execution and Raspberry Pi model quality are separate claims.

## Dataset citation

Credit the original composite corpus and every selected upstream dataset as listed in the accompanying source ledger, including its reusable BibTeX entries. For slide attribution use the upstream primary DOI plus the two pinned Hub repository revisions; add the original dataset papers in notes/references. A citation for this derivative release must include the confirmed author list, release title, year, permanent repository URL and actual published commit/DOI once assigned. Hugging Face supports version-specific DOI generation through repository settings; request it for the reviewed released revision, following the [official DOI documentation](https://huggingface.co/docs/hub/doi). No derivative DOI or completed publication is claimed by this draft.
