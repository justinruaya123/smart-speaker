# Semifinal data and rights

The bounded experiment uses original FSC connected speech plus isolated Speech Commands negatives, and a textual CMU pronunciation dictionary. It uses no TTS, splicing, external neural initialization or synthetic training augmentation. Dataset procurement is hash-pinned in `data/sources/*.source.json`; payloads and derived features stay ignored.

| Source | Pinned identity | Purpose and terms |
|---|---|---|
| Fluent Speech Commands | revision `c4cc0e2c898585329319835ed55f47a6ab9e0d92`; archive SHA-256 `c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b` | 30,043 original connected utterances. Academic research only; the publisher describes CC BY-NC-ND 4.0 and prohibits commercial training/testing/product development. |
| Speech Commands v0.02 | SHA-256 `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58` | 2,500 deterministic isolated-word controls, CC BY 4.0. These are not complete-command or continuous unknown-speech coverage. |
| CMUdict | revision `74790861f652b15e4ac49015a90074ad62a27690` | Text pronunciation data only, BSD-style CMU license. No learned parameters. |

Source authorities: [Fluent dataset and terms](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/), [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands), [pinned CMUdict](https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690). The downloaded FSC archive and CMU license are preserved locally; no corpus recordings are redistributed in the runtime ZIP.

| Partition | FSC | Speech Commands | Total | Distinct source-prefixed speakers |
|---|---:|---:|---:|---:|
| Train | 23,132 | 2,000 | 25,132 | 1,167 |
| Development | 1,483 | 100 | 1,583 | 63 |
| Calibration | 1,635 | 200 | 1,835 | 88 |
| Test | 3,793 | 200 | 3,993 | 128 |

Original FSC training/test speakers are preserved. Sorted original FSC validation speakers are split into five development and five calibration speakers. Speech Commands keeps its official train/validation/test speaker assignment; validation speakers are deterministically divided between development and calibration. Selection uses the first 64 development identities after SHA-based sorting, fixed before fitting. Normalization uses only the mixed training pool. No selected candidate used test results for fitting, epoch selection or thresholds.

Every prepared row records source/revision, original speaker ID, transcript/annotation provenance, sample format, original file and decoded PCM hashes, explicit structured reference or unsupported valid request, pronunciation target, split and catalog identity. Session identity is unknown. CTC feasibility and finite mono 16 kHz PCM are checked. Source archive extraction rejects traversal and links. Speaker and exact PCM cross-split overlap are zero. Labels were inherited from source CSV/folder metadata; they were not independently listened to. Near-duplicate similarity auditing and manually verified acoustic numeric/color/name coverage remain absent.

The audit and exclusions are in [fsc-audit.json](reports/semifinal-v1/fsc-audit.json) and [data-audit.json](reports/semifinal-v1/data-audit.json). The original frozen manifests live under `.cache/semifinal-v1/mixed-v2/`; their identities are in [protocol.json](reports/semifinal-v1/protocol.json). They contain this run's resolved local paths; regenerated paths on another host change manifest bytes and require a new run receipt. Raw input archive hashes and the deterministic preparation rules remain portable.

The core bootstrap profile also fetched the pinned SLURP annotation files. They were not included in this bounded model's training or evaluation. No SLURP audio or unrelated local research dataset was used as a hidden input.

The final review identified that the original frozen manifests did not hash cached feature bytes. New preparations record feature SHA-256 and frontend-spec identity; normalization and fitting reject missing or mismatched provenance. The completed run is not rewritten. The [strict feature audit](reports/semifinal-v1/feature-audit/summary.json) found identical audio and PCM hashes for all 32,543 recordings, but exact feature equality for only 17. The [numerical follow-up](reports/semifinal-v1/feature-audit-numerical/summary.json) measured maximum absolute feature difference 1.90735e-6; train-only normalization recomputed from the original cached arrays is byte-value exact, while WAV-recomputed means/std differ by at most 8.59e-11. These observations are consistent with small floating-point differences; their cause was not established. The follow-up's 1e-5 tolerance is exploratory and does not replace the failed strict audit. Historical cache provenance remains limited because it was not hashed before fitting. Reproduction requires freshly prepared manifests under the corrected protocol. The [original manifest archive](reports/semifinal-v1/frozen-data-manifests.zip) retains all four frozen manifests and normalization byte-for-byte, without audio payloads.
