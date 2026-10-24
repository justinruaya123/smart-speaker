# Iris citation and rights audit

Scope: the **original 258,912-parameter OptionB epoch 44** VCM and its publication UI/deck. This is not a blanket license for code, models, voices or recordings. No root license is introduced. Public availability is not consent to redistribute a recording or a cloned voice.

## Primary datasets: cite both the DOI and actual revision

| ID | Inspected primary source | Membership in this model |
|---|---|---|
| D1 | [AI231 ME2 voice-command dataset](https://doi.org/10.57967/hf/10723), `airimonda/ai231-me2-voice-commands` | TRAIN 1,889 human +5,982 supplied synthetic rows; original TEST/HOLDOUT populations remain separately exposed |
| D1v | [Pinned primary card](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/blob/25111444af3adff7588d86ab27c304bb080895cf/README.md), revision `25111444af3adff7588d86ab27c304bb080895cf` | Exact upstream Parquet identities and admitted membership in [snapshot/inputs](../submission/optionb-tiny/snapshot.json) |
| D2 | [Pinned supplemental card](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data/blob/230c8b8d58caf78e4bc90886d9c6fa77fe6024b8/README.md), revision `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8` | TRAIN 15,444 synthetic rows, including admitted gap-fill/numeral-wording subsets; **not** every upstream row |
| D3 | `justinruaya123/ai231-smart-speaker@0fc3ac3ff257838d02000b6a8ce78225c262b9fa`, `reproduction/auxiliary` |184 exact synthetic DEV audio files in the original source recipe, not new human TRAIN data |

D1's DOI was resolved during this audit to the primary dataset. Its pinned card and D2's pinned card were fetched directly, not inferred from current-head descriptions. The supplemental DOI is **not asserted** because none is established by the inspected source. The [local supplemental notice](../data/sources/fil-supplemental-NOTICE.txt) retains the card's caveats.

## Original sources: attribution follows actual TRAIN membership

Recounted from the hash-verified `train.jsonl.gz`, grouping human rows by their literal `publisher_source`. These are **publisher labels**, not independent adjudication or proof of all derivative rights. Synthetic rows do not expose this field; their repository/kind counts remain separate.

| Literal TRAIN publisher label | Admitted human rows | Terms reported by the pinned composite card |
|---|---:|---|
| `SLURP` |929|CC BY4.0; credit original authors under upstream terms |
| `real_voice` |438|Group recordings; do not share outside class without each speaker's consent |
| `SNIPS` |258|Follow original SNIPS spoken-language-understanding dataset terms |
| `FluentSpeechCommands` |180|Fluent Speech Commands Public License; non-commercial academic use |
| `xela_SET_TEMPERATURE_REAL` |40|Group-recording provenance; individual consent/class restrictions are not waived |
| `CommonVoice_en` |23|Composite card identifies Common Voice19(en), CC0 |
| `TimersAndSuch` |17|“other-open”; inspect original dataset license, not an assumed MIT grant |
| `xela_Multi-Sensor` |4|CC BY4.0; original DOI10.48804/IEKKVZ and download-tracking/GDPR obligation reported by composite card |

These eight counts reconcile to the 1,889 admitted human TRAIN rows. The card also mentions Speech Commands, MLEnd and other sources in the wider dataset; this audit does **not** claim that those names fitted this tiny VCM when its admitted TRAIN publisher labels do not establish that membership. The older 935k/2M citation inventory is a historical ledger, not proof of this model's training inputs.

## Synthetic voice and consent boundaries

- D1 describes a group synthetic set with LibriSpeech/SilencioPH reference lineage. Its source/consent obligations remain, including CC BY where applicable. A synthetic clip is not an independent human recording.
- D2 explicitly says reference recordings' source, speaker consent and license are **not recorded**, and its CosyVoice2 model-weights license is unverified. Its stated boundary is research/education only; do not redistribute voices or use commercially while these rights are unresolved.
- D2's17 requested reference voices are not 17 independently qualified synthetic accents or demographic subjects. Accent has not been measured; reference-person overlap with the primary synthetic pool has not been verified.
- Build-time wake/cue TTS is separately identified by the runtime's versioned generator/voice manifest. It does not turn historical corpus cloning into a resolved license, prove human demographic coverage or create a cloud recognition dependency.
- The deck contains no raw dataset audio, private home coordinates, tokens, copied reference voices or human recordings. Its output surfaces are labeled storyboard examples, not fabricated provider receipts.

## Model, code and component notices

| Component | Provenance / notice | Claim boundary |
|---|---|---|
| Original tiny VCM | [Frozen snapshot](../submission/optionb-tiny/snapshot.json), [original source](https://github.com/justinruaya123/smart-speaker/tree/3c05b3beb46f06c3108089e0d8c79474670c6cb7), five-file hashes in the [new card](MODEL-CARD-iris-epoch44.md) | No blanket weight/data license inferred; qualification remains false |
| Native Hey Iris | Project's separate `artifacts/iris/wake/iris.json`, graph hash/footprint in publication JSON |82,114 parameters are **additional** to VCM; no original-human/live qualification |
| Owner/overlap | `artifacts/iris/owner/owner.json` |129,985 runtime parameters, experimental; not a qualified biometric system |
| Silero VAD | [Upstream v5.1 graph](https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/src/silero_vad/data/silero_vad.onnx), [upstream project](https://github.com/snakers4/silero-vad); local `wake-models.json` records MIT notice/hash | Separate companion, not inside258,912 |
| ONNX Runtime | [Microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | Runtime software terms are separate from model/recording rights |
| SpeakerUI | Existing project Svelte/Fastify implementation; [Svelte](https://github.com/sveltejs/svelte), [Fastify](https://github.com/fastify/fastify) upstream notices | Screenshot is actual compiled base-main UI with local action-disabled fixture, not a live acoustic/provider claim |
| Manim | [ManimCommunity/manim](https://github.com/ManimCommunity/manim), authoring pin0.20.1 | CPU-rendered geometric animations, not CSS mislabeled as Manim |
| manim-slides | [jeertmans/manim-slides](https://github.com/jeertmans/manim-slides), authoring pin5.7.0 | Real `Slide` scenes and offline Reveal export; not part of the scientific environment |
| Reveal.js / embedded theme fonts | Offline dependencies downloaded by actual manim-slides export | Retain upstream notices and the distributed licenses in `docs/site/assets/notices/` |
| FFmpeg / x264 | Actual authoring host's FFmpeg4.2.11 GPL/libx264-enabled build | Used to encode/probe posters/media; no executable is redistributed by Pages |

The Mono scene labels render installed Consolas glyphs into images/video; no Microsoft font file is distributed. HTML uses the actual Iris Inter/system fallback stack with no external font request. The offline Reveal theme embeds its own upstream font assets; their license remains separate.

## Evidence, not borrowed numbers

The [publication JSON](site/assets/iris.json) is generated from verified frozen payloads and named actual receipts. [Historical evidence manifest](../presentation/evidence/manifest.json) records exact source paths, bytes and SHA-256 from `3c05b3b…`; training curve/summary and original/later PC receipts are not synthesized. New measurements must keep their own source/runtime, load conditions, timing scope and hardware. No969k/2M metric or Pi number is copied into an epoch 44 result. `pi:null` means unmeasured.

The [existing pinned reproduction instructions](REPRODUCIBILITY-optionb-epoch44.md) preserve the original exact CPU comparison failure and the separately repeatable CUDA epoch 28 evidence. Artifact verification, source reconstruction, independent GPU repeatability and original CPU exact retraining are different claims.

## Presentation reference and transformation

The [tiny-alexa presentation](https://charrienacor.github.io/tiny-alexa/) informed sparse chapter rhythm, a staged architecture inspector and compact metric tables. The Iris deck does **not** reuse its brand palette, model architecture, two-head design, parameter counts, device numbers, copy or screenshots. The palette and screenshot come from the actual Iris `SpeakerUI/src/ambient/Iris.svelte` at base `c0628940400642f204ff98e26908549b1c07e41c`.

Authoring API references: [Manim Slides quickstart](https://manim-slides.eertmans.be/latest/quickstart.html), [HTML conversion](https://manim-slides.eertmans.be/latest/reference/html.html), [CLI](https://manim-slides.eertmans.be/latest/reference/cli.html). Actual rendering found Manim0.22.0 incompatible with manim-slides5.7.0 (`ManimConfig.write_to_movie` removed), so the exercised authoring pair is 0.20.1/5.7.0. That fallback is recorded rather than inventing a successful0.22 render.

## Publication infrastructure

[Official Pages custom-workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) defines the artifact/permission contract. Action refs were resolved from each official `actions/*` repository with `git ls-remote`, not copied from a blog:

| Action | Exact inspected ref |
|---|---|
| `actions/checkout` v6 | `d23441a48e516b6c34aea4fa41551a30e30af803` |
| `actions/setup-python` v6 | `ece7cb06caefa5fff74198d8649806c4678c61a1` |
| `actions/setup-node` v6 | `249970729cb0ef3589644e2896645e5dc5ba9c38` |
| `actions/configure-pages` v6 | `45bfe0192ca1faeb007ade9deae92b16b8254a0d` |
| `actions/upload-pages-artifact` v5 | `fc324d3547104276b827a68afc52ff2a11cc49c9` |
| `actions/deploy-pages` v5 | `368f82528645a54fb793d4d04e342629a3f51346` |

PR validation is read-only. Static-site checks are distinct from the **required acoustic-release job**, which currently fails on the preserved failed wake receipt. Deployment needs both jobs; a green site validator never turns the native 1/32 or unpromoted candidate21/32 result into a passing acoustic release. No production Pages deployment is allowed while that gate fails. Only a validated push to main may upload/deploy `docs/site`; no dataset/private state/scientific environment is included. Deployment success is not asserted by authoring these files; the release owner must read back the actual PR checks, merged commit and public site.

## Audit method and project citation

Repository facts above are grounded in the immutable snapshot/compressed membership and actual receipts. External data/rights wording is attributed to the fetched pinned cards, not presented as a new legal adjudication. Provenance lives in this authorized in-artifact register and hash-bound evidence, not a separate citation-ledger cache; the standalone ledger/evidence verifier was not run.

For the software/model publication, cite **Iris project contributors (2026), Iris: original OptionB epoch 44 finite-command smart speaker**, the repository revision actually used, model graph hash, and dataset DOI + revisions above. No project DOI is invented. The final README contains a project BibTeX entry; cite original data authors as required by each source's terms.
