# ME2 — Voice Controlled Smart Device

**Author/account:** justinruaya123  
**Card revision:** 15 October 2026  
**Stage:** Accuracy and correctness first; Raspberry Pi validation deferred.  
**Current verdict:** BLOCKED — selected A100 checkpoint and final acoustic receipts
are not available to this audit. No passing final-training result is asserted.

## Model

| Item | Filled value |
|---|---|
| Input and frontend | Mono 16 kHz PCM; existing 40-bin log-mel, 400-sample window, 160-sample hop, FFT 512; uncentered framing; train-only normalization |
| Encoder | Complete-utterance temporal CNN; width 128, stride-2 stem, eight residual depthwise temporal blocks; dilations 1,2,4,8,1,2,4,8; per-frame LayerNorm; dropout 0.1 |
| Streaming / KV cache | Not a causal streaming neural encoder; no KV cache. Always-on wake/capture is a separate subsystem. |
| Heads | 19 acoustic intent outputs including OOV; 15 role queries with four role states; four request-validity states; categorical finite values; five 64-dimensional entity projections |
| Typed labels | 18 schema intents plus OOV; 19 external OptionB labels map into the schema. These are different inventories. |
| Parameters | 280,818 = 0.280818 million, from the settled implementation |
| Nominal FP32 parameter storage | 1,123,272 bytes (calculated as parameters x 4); not final exported bundle/process size |
| Final exported weights | Not received; measure the selected ONNX/ncnn files and complete bundle |
| Output | Authoritative typed accepted command or explicit rejection; intent >=0.95 and every supplied slot >=0.90; no ASR/STT, LLM or cloud recognition |
| Runtime | ncnn primary; ONNX Runtime and PyTorch reference parity, FP32 first |
| Actuator | Disabled for replay; no provider/device actions authorized by this closeout |

## Dataset

| Item | Filled value |
|---|---|
| Source | Existing approved, revision-bound project sources; primary `airimonda/ai231-me2-voice-commands`; actual A100 source/manifests must be attached to final receipts |
| Hours / utterances used in final A100 training | Not verified; derive from the actual final manifest, not from this card or old receipts |
| Reference-view human counts | Last committed audit: train 1,895; dev 256; calibration 210; historical test 1,050; historical holdout 96. These are not claimed as current A100 counts. |
| Reference-view speakers | 337 declared source-qualified human IDs across all views; aliases/physical identity not fully established. Final-training speaker counts pending. |
| Labels and scope | Existing 93 OptionB verbatims and all personal extensions; last recorded 414 definitions and 1,031 selected catalog entries. Preserve exact grammar, defaults and typed outputs. |
| Dataset DOI / access terms | Final citable dataset release and rights/access receipt not verified; do not substitute the TTS model DOI |
| Split integrity | Require file/PCM hashes, person/alias and recording-group separation; train-only normalization; no final-test person in fitting, calibration or enrollment |

## Training on the A100 cluster

| Item | Filled value |
|---|---|
| Cluster | A100 training is user-reported; live node, scheduler/job ID and allocation not inspected here |
| GPUs / wall clock | Not verified; populate from the actual job receipt |
| Objective | Masked intent/role/value/request-validity classification plus eligible entity metric and reviewed-span localization losses; no runtime transcription or CTC fallback |
| Optimizer recipe | AdamW; learning rate 0.001; weight decay 0.0001; gradient clipping 1; batch 64 |
| Planned comparison | Human-only uniform vs complete-command/person-balanced sampling; seeds 53, 71, 89; 30 x 96 updates per run |
| Actual steps / final loss / selected epoch | Not received. Report actual completed updates, train/dev curves and the development-selected epoch; do not fill with nominal budget. |
| Selection and calibration | Human development selects the checkpoint; separate human calibration; final entity memory rebuilt against exact artifact identities; freeze before final testing |
| Training-quality verdict | Not established for the active candidate. Missing supervision, nonfinite losses, unresolved data leakage or failed independent accuracy gates block completion. |

## Accuracy and correctness — final candidate only

| Test / metric | Current filled status |
|---|---|
| Official held-out command audio, strict typed exactness | Not measured for final A100 checkpoint |
| Official intent/slot score and pinned parser compatibility | Final-candidate replay pending; not a completed live Pi benchmark |
| Fresh unseen-human accepted exact-command recall | Not measured; preserve pilot goal >=95%, with denominators and per-person results |
| Accepted precision / negative false-accept rate | Not measured; precision is undefined if nothing is accepted, not 100% |
| Per-intent / supplied-slot correctness | Not measured for final A100 checkpoint; all advertised scope must remain counted |
| Added six-voice synthetic sanity | Protocol and runnable harness committed; model download/generation/scoring not executed successfully in this session |
| TTS composition | Female presets af_heart, af_bella, bf_emma; male presets am_fenrir, am_michael, bm_george; two American and one British preset per group |
| TTS coverage and local pass criterion | >=558 OptionB voice/phrase trials plus all preserved extensions/catalog cases; every positive accepted exactly, every specified negative correct, no missing outputs/faults; independent QA for every WAV |
| Final-model engine parity | Pending real selected-model PyTorch/ORT/ncnn comparison; fixture parity is not sufficient |
| Genuine same-speaker ownership | Not measured; separate acoustic backend/pairs required; command-WAV TTS does not test ownership |

**Software evidence in this session:** 76 tests passed (49 existing package
fixtures re-executed, 27 new numerical/accounting tests). These are not trained
model or speech accuracy measurements. No full repository/cluster-suite pass is
claimed. Previous broad failures remain for the agent to rerun and report.

**Historical, not current-A100 evidence:** the earlier numeral-only/unqualified
checkpoints had human accepted exact recall 0/636 test and 0/83 holdout; a separate
numeral probe reached 449/696 raw category accuracy. Neither belongs in the final
candidate result cells above.

## Validation on the Raspberry Pi 4 — deferred

Keyword/wake accuracy, physical false-wake rate, p50/p95 latency, RTF, cold load,
peak RSS, CPU, microphone/endpoint path, temperature, throttling and network-disabled
physical operation are all **deferred by the user**. No desktop or synthetic timing
will be inserted as a Pi result. Training is on A100; the Pi is an inference and
validation target, not the training target.

## To be submitted / reviewer checks

| Item | Filled status |
|---|---|
| GitHub repository | `justinruaya123/smart-speaker`; last live check private, no detected root license. Public/MIT submission condition not yet met; visibility/license not changed. |
| One-command reproduction | Existing project pipeline plus `final/README.md` execution sequence; final artifact-backed end-to-end reproduction pending |
| Dataset licensed and citable | Final release DOI and access/license receipts pending verification |
| A100 record | Actual node/GPU/seed/wall-clock/log/update/checkpoint receipt pending |
| Model weights / license / release URL | Final checkpoint/export and immutable authorized release not supplied; no fabricated link |
| Training logs + final checkpoint evidence | Existing historical receipts only; preserve final logs and checkpoint outside ordinary Git with durable hash-bound release/registry references |
| Pi4 latency reproduced by posted script | Deferred; no claimed measurement |
| Held-out unseen speakers | Fresh final-candidate evaluation pending; synthetic presets are not human speakers |
| Comparable-size baseline | Matched uniform/balanced comparison is an ablation, not an independent architectural baseline; comparable baseline result still requires honest matched evidence |

## Evidence references

The settled specification, protocol and experiment receipts at parent
`5be3a7f5f5ac38ba6c4fcc14d7fd99de478b0bbd`, the supplied current handoff, staged
`VCM_FINALIZATION.md` and `verification.json`, and this folder's `tts-protocol.json`
and `verification.json` support the filled fields. Replace pending values only
with the actual selected candidate's matching immutable receipts. Do not copy
old metrics, synthetic model-card claims or slide placeholders into final results.
