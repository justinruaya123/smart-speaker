# Iris datasets and audit directory

The release uses immutable original recordings and explicitly labelled generated or constructed examples. [reproduction-inputs.json](../experiments/iris-release-v1/reproduction-inputs.json) is the machine-readable index of pinned files, source revisions, SHA-256 values, audio reconstruction mapping and permitted publication. `python train.py` verifies these inputs before fitting; `--skip-train` verifies the selected artifacts and rebuilds/evaluates the delivery.

## Command corpus

Original source views retain recording IDs, labels, speaker assignments, audio hashes and train/development/calibration/test membership. Historical absolute feature-cache fields were removed from the portable snapshot; recordings and labels did not change. Hash-bound manifests remain in the published input capsule, not a second audio pool in Git.

| Existing view | Recordings | Human | Supplied/generated synthetic | Speakers/voice IDs |
|---|---:|---:|---:|---:|
| Train | 23,739 | 1,889 | 21,850 | 252 |
| Development | 1,339 | 256 | 1,083 | 47 |
| Calibration | 1,495 | 210 | 1,285 | 37 |
| Test regression | 4,418 | 1,050 | 3,368 | 121 |
| Fixed benchmark holdout | 196 | 96 | 100 | 5 |
| Personal command cohort | 88 | 0 | 88 | 2 |

The human regression has 636 supported positives, 229 negatives and 185 ambiguous references. Ambiguous references are retained but excluded from accuracy denominators. The fixed 196-row scorer has 186 positives and ten negatives; its audio shard SHA-256 is `9917e336ec596eaaddb476e266f36451326e3f8e5a2cb3dafbb675c33be133aa`. The personal cohort is unchanged: 36 positives and 52 negatives, from two held-out designed synthetic voices. It is not human personal-command validation.

The [primary source receipt](../data/sources/ai231-primary.source.json) pins revision `25111444af3adff7588d86ab27c304bb080895cf`. The [supplemental receipt](../data/sources/fil-supplemental.source.json) pins `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`. Research/education restrictions, original recording consent and upstream component conditions still apply. Supplemental synthetic/model licensing is unresolved. **No original restricted or supplemental audio is newly uploaded by this release.** Existing public dataset history is preserved; this release adds documentation, manifests and permitted original-designed examples without expanding those old redistribution rights.

## New designed voice conditions

[voices.json](../experiments/iris-release-v1/voices.json) fixes 24 designed identities: 12 train, four development, four calibration and four independent test. Each identity has 141 completed clips: 93 baseline wordings, 18 personal commands, 12 near-wake negatives, ten invalid commands and eight Hey Iris positives. All 3,384 clips were generated and hash-verified; [the completion receipt](../experiments/iris-release-v1/generation-complete.json) records 3,319.23 seconds of generation. Speaker/voice IDs never cross folds. The new command-only test is 564 recordings: 444 supported commands and 120 negatives, including wakes treated as out-of-scope commands. Wake evaluation uses its separately labelled subset and additional fixed negative speech.

The combined frozen VCM views contain 25,431 train, 1,903 development, 2,059 calibration, 4,418 original regression, 196 fixed benchmark, 88 personal and 564 new-voice test rows. The [pre-fit file audit](../experiments/iris-release-v1/audio-audit-before-fitting.json) verified all 34,659 distinct paths and file hashes. The [split audit](../experiments/iris-release-v1/data-audit.json) records zero cross-fold recording-ID, speaker, PCM-hash and pair-ID overlap. These audits verify declared identities and bytes, not perceptual speaker identity or transcript quality.

The generator is the pinned [Qwen3-TTS VoiceDesign/Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) source in [the source receipt](../data/sources/qwen-synthetic-completion.source.json), used offline only. The release preserves prompts, design seeds, anchors, batch seeds, generation settings, audio/PCM hashes and objective duration/energy/clipping QA. A prompt asking for a particular accent, age-like presentation or voice style does **not** verify those characteristics or establish human demographic coverage. No independent listener/transcript verification is claimed. Human and synthetic metrics remain separate.

The VCM trains on complete utterance labels, with balanced human/synthetic batches, command/speaker balancing, UNKNOWN examples, bounded training-only gain and feature masks. The generator is not installed on the Pi. Completed published generated data can be hash-verified and reused without CUDA or a TTS download.

## Independent speaker and overlap corpus

[Owner protocol](../experiments/iris-release-v1/owner-protocol.json), [dataset audit](../experiments/iris-release-v1/owner-dataset-audit.json), [readability audit](../experiments/iris-release-v1/owner-readability-audit.json) and [historical reservations](../experiments/iris-release-v1/owner-reservations.json) define the independent identity/overlap study.

The selected pool contains 9,429 original-human recordings, 24.264 hours: 7,001 train, 640 development, 210 calibration and 1,578 test. It combines 3,405 VCM human recordings with 6,024 deterministic selections from [LibriSpeech train-clean-100](https://www.openslr.org/12/) (CC BY 4.0). LibriSpeech contributes waveform/speaker labels only; no transcript task, decoding or text target is used. The complete source archive is hash-verified; selected paths are immutable. Declared cross-fold speaker and file-hash overlap is zero. Historical reserved assignments are preserved; missing historical identity lists remain an explicit audit limitation.

Training constructs overlap mixtures from different speakers **within the same assigned split**, preserving original singles and labelling constructed mixtures as such. Constructed switch/overlap trials are proxies, not real captured overlapping speakers. There are 192 separately reserved synthetic proxy recordings; they do not fit identity weights or feature normalization. Short human utterance pairs and synthetic wake pairs support experimental calibration only. Original human wake-to-command pairs, real room overlap and physical Pi capture remain unmeasured.

## Reconstruct and extend

Use `python train.py` for the frozen comparison or `python train.py --skip-train` for the pinned selected release. Upstream restricted payloads are acquired from their original pinned repositories when access permits; embedded original waveform bytes are restored under their original hashes. Permitted own-generated payloads and project initializers use the immutable published input capsule. The audio map translates historical relative paths to hash-addressed local payloads without changing frozen manifest labels/splits. Shared source recordings are never moved or modified.

Do not add test examples to fitting, relabel ambiguous references as known, count TTS voices as humans, silently replace a manifest, synthesize inside an archived human-only study, or upload restricted audio. A new name/value changes the acoustic competition and needs a new experiment/version, development calibration and unknown-name tests. Text entity bindings cannot establish acoustic coverage. Record further additions, licenses and audit results here, with detailed immutable machine receipts beside the new protocol.
