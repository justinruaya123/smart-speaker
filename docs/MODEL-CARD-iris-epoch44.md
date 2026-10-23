# Iris epoch 44 model card

**Selected VCM:** original OptionBTinyVCM-1, width 160, **258,912 parameters**, mixed epoch 44, FP32 ONNX. **Qualification remains false. The new synthetic wake smoke gate FAILED; no replacement was promoted.** This is an unqualified research build, not a working acoustic release. This publication selects the existing VCM artifact; it is not a new fit, threshold search, live-human qualification or Raspberry Pi benchmark.

This card supersedes product-default descriptions for this publication only. The bound historical [root model card](../MODEL_CARD.md), [dataset directory](DATASETS.md), earlier 969,928-parameter Iris experiments and their receipts remain historical evidence, not measurements of this model.

## Identity and intended use

| Property | Frozen value |
|---|---|
| Model | `submission/models/optionb-tiny/` |
| Recovery checkpoint | `submission/checkpoints/optionb-tiny-selected.pt` |
| Model parameters / graph | 258,912 / 1,082,940 bytes |
| Whole five-file bundle | 1,088,682 bytes; excludes wake/VAD/owner |
| Output | One32-way complete-command/UNKNOWN head; UNKNOWN is class 0 |
| Positive scope | 31 complete intent+slot outcomes,18 intent families,93 wording definitions |
| Input | Whole mono16 kHz utterance;40-bin log-mel features plus valid lengths |
| Policy | Non-UNKNOWN, posterior≥.50 **and** top-minus-second margin≥.10 |
| Scientific source | `3c05b3beb46f06c3108089e0d8c79474670c6cb7` |
| Snapshot | [Original membership/architecture/recipe/hashes](../submission/optionb-tiny/snapshot.json) |

| Bundle file | SHA-256 |
|---|---|
| `model.onnx` | `c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce` |
| `commands.json` | `71017cf22366d15557e8878636135f5ff2d0ac81fc4dfd6f058fc88f0f2d4b5d` |
| `normalization.json` | `28f7ef1b2c926c587d611d17eef91184c2c7e2de66abe352d1fc2365e45d69ed` |
| `policy.json` | `b9242e46fb093642a41023e8ace3c0cadc2fe51ffede512aa09e399622f355f0` |
| `recipe.json` | `020c3deccb0f8cece0f23b8c016b6a63bdd0359321bb06cbd00e091966d46621` |

Checkpoint SHA-256: `d45d71b3858b4182967d3f41dad8145150028ca17401c73d203fba75d6bda687`. `python -m scripts.optionb_epoch44 verify --load` independently pins and loads the original bytes; one[1,32] feature fixture proves loading, not acoustic accuracy.

Intended use is a research/education finite-command smart-speaker demonstration. It is not an unrestricted assistant, transcription service, reliable security/biometric system, safety-critical controller or a human-qualified household product.

## Architecture and I/O

The unchanged frontend uses a causal400-sample Hann window,160-sample hop,512-point FFT,40 HTK mel bins spanning20–7600Hz, natural log with floor1e-8, and no centered padding. Equal-utterance per-bin normalization is fitted on **1,889 human TRAIN recordings only**, never DEV/calibration/test/holdout or supplied synthetic speech.

Features are subsampled at stride2, then processed by a causal kernel 5 stem and four width 160 residual blocks. Each block has two depthwise/pointwise stages, kernel 7, dilation 1/2/4/8 respectively, per-frame LayerNorm and SiLU; training dropout is.1. Valid-length masked mean and maximum pooling produce a concatenated320-feature vector. One linear32-way head produces logits; softmax and the frozen margin policy decide the complete command.

ONNX consumes `features` and `lengths`, returns `logits`, and is run with CPUExecutionProvider in this publication. There is no independent intent/slot acceptance gate, transcript, ASR/STT, LLM, semantic rescue, extra acoustic class, slot rounding or arbitrary-name enrollment. Metadata transcripts in the source dataset do not become a runtime input.

`commands.json` is the authoritative class order. Examples of model outputs are `{"intent":"TIMER","slots":{"duration_seconds":10}}` and `{"intent":"TEMPERATURE","slots":{"degrees":22,"unit":"C"}}`; they are complete fixed classes, not words assembled into slots.

## Finite values

| Family | Acoustically defined values |
|---|---|
| Music | Generic PLAY_MUSIC; no arbitrary song/artist name recognition |
| Questions | WEATHER and TIME only; no general search |
| Lights | `light.all`; on/off; brightness20/60/100%; red/blue/green |
| Timers | 10/30/60 seconds |
| Alarms | 06:00/08:00/21:00, `Asia/Manila` |
| Temperature | 18/22/26°C |
| Media | PAUSE, STOP, NEXT, VOLUME up/down |
| Reminders | `task.drink_water`, `task.study`, `task.exercise`; list |
| Calls/messages | CALL `contact.home`; preset MESSAGE |

Touch controls and provider bindings can be broader, but do not teach the model a new spoken value. The 31 classes are allocated exactly once to the ten categories by the publication generator; all 93 wording definitions are exported for inspection. A definition is not proof every human speaker's wording was correctly recognized.

## Data and source membership

The generator verifies compressed manifests against the immutable snapshot and recounts kinds and TRAIN repositories. Counts are admitted manifest rows, not unique optimizer samples or independent synthetic people.

| Split | Human | Synthetic |
|---|---:|---:|
| TRAIN | 1,889 | 21,426 |
| DEV | 256 | 995 |
| Calibration | 210 | 0 |
| TEST | 1,050 | 3,368 |
| HOLDOUT | 96 | 100 |

TRAIN membership:

- `airimonda/ai231-me2-voice-commands@25111444af3adff7588d86ab27c304bb080895cf`:1,889 human +5,982 supplied synthetic rows. Composite dataset DOI **10.57967/hf/10723**.
- `martinnavs/ai231-fil-supplemental-data@230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`:15,444 supplied synthetic rows. Includes admitted gap-fill and numeral-wording members; not every published supplemental row is used.
- The separate public auxiliary source `justinruaya123/ai231-smart-speaker@0fc3ac3ff257838d02000b6a8ce78225c262b9fa` supplies184 exact synthetic DEV audio files, not new human or TRAIN membership.

Original-source TRAIN publisher labels are recorded in the [citation audit](IRIS-CITATIONS.md). Do not import an older 935k/2M source inventory as proof those sources fitted this VCM. Original human/synthetic manifests and publisher labels are not independent listening/adjudication. Human holdout's declared speakers/recording/audio identities are separated from human fitting/selection inputs in the historical checks, but synthetic reference-voice alias separation has not been independently established.

Rights are heterogeneous. Group recordings retain individual consent/class-sharing restrictions; some upstream datasets are non-commercial academic. Supplemental cloned-voice reference source/consent/license and cloning-weights license remain unresolved. “Filipino” describes source reference metadata, not measured accent/demographic coverage. No raw recordings or feature tensors are redistributed in the static site. See [source-specific terms](IRIS-CITATIONS.md).

## Original training and reproducibility

The original random-initialized CPU recipe uses seed 53 and four Torch threads;6 human +42 mixed epochs;96 batches/epoch;32 samples/batch;4,608 optimizer updates; AdamW lr.002, weight decay.0001, label smoothing.02. Original wall clock was2,987.9094288999986 seconds. The selected checkpoint is epoch 44 chosen on DEV, not the last epoch 48. [Historical curve](../presentation/evidence/training-curve.json) and [summary](../presentation/evidence/training-summary.json) are exact copies bound to the pinned source commit by the [manifest](../presentation/evidence/manifest.json).

Sampling is balanced by command/negative scope, then declared speaker, with replacement. Human-only batches use24 positive/8 negative human rows; mixed batches use12 human positives/8 human negatives/12 supplied synthetic positives. Admitted corpus rows, draws and unique sampled examples are different populations; the full recipe's147,456 draws are not 147,456 unique recordings.

| Claim | Evidence boundary |
|---|---|
| Exact artifact integrity/loading | Verified independently by the existing main verifier |
| Source/data/frontend reconstruction | Hash-bound original-source receipts; use its pinned environment |
| GPU-to-GPU repeatability | Two historical full CUDA fits matched; selected epoch 28 and fresh.55/.30 policy |
| Exact original CPU epoch 44 retraining | **Not verified.** The GPU-vs-original comparison failed; absent success lock is preserved |
| Fresh human generalization / Pi | Unmeasured, not inferred from software or synthetic tests |

Use the [pinned original-source runbook](REPRODUCIBILITY-optionb-epoch44.md), noting that its earlier “opt-in default” prose describes its historical main integration. This publication selects the original artifact; it does not redefine the runbook's failed exact-fit comparison. Never replace epoch 44 with a repeatable epoch 28 GPU result, forge `selection-before-test.json`, or claim `reproduce.py` trained the exact selected weights. [Getting started](IRIS-GETTING-STARTED.md) separates artifact benchmarking from fitting.

## Historical human regression and failure modes

These are already exposed cohorts, **not a fresh blind test**. Wrong accepts below already include the listed negative false accepts.

| Cohort | Raw correct positive commands | Policy correct accepts | Wrong accepts (all scored rows) | Negative false accepts | Ambiguous/unscored |
|---|---:|---:|---:|---:|---:|
| HOLDOUT | 76/80 | 72/80 | 3 | 1/14 | 2 |
| TEST | 469/609 | 393/609 | 59 | 24/256 | 185 |

The exact metadata-wording subsets score58/62 holdout and 173/232 test correctly accepted, not full 93-wording human coverage. Twenty-one wording definitions lack exact human evaluation-reference evidence. Ambiguous rows stay outside scored denominators rather than becoming UNKNOWN. Historical errors include “Play a song”→NEXT,18°C→22°C, and out-of-scope “More heat”→PAUSE. The strict qualification gate has not passed; speed does not repair these errors.

## Timing, hardware and companion costs

[Original PC receipt](../presentation/evidence/pc-original.json): **4.82/8.41 ms p50/p95**. [Later preserved PC repeat v2](../presentation/evidence/pc-repeat-v2.json): **5.356650/8.691980 ms p50/p95**. These are separate historical runs, not a newly measured current gateway or a fastest-run selection. Both use1,146 original human recordings and 3,438 timed calls (three repeats, ten warmups); accuracy is counted once per original recording.

Hardware: Intel Core i7-11700, Windows 11, Python 3.13.14, CPUExecutionProvider, one ONNX and one frontend BLAS thread, unchanged process affinity. Scope is external wall time around `recognize(PCM)`: PCM validation, unchanged frontend/normalizer, CPU ONNX, softmax/margin and command validation. It excludes WAV decode/hash, speaking duration, wake/VAD/endpoint waiting, worker/gateway/UI, audio feedback and providers. It is neither graph-only nor full microphone interaction latency. The separate 256-DEV working-set probe is not memory evidence for these full-PC runs; no full-PC memory value is invented here.

**Current product-lock adapter receipt (2026-10-24):** p50 **5.3902499967080075 ms**, p95 **10.56562999983725 ms**, peak whole-process working-set RSS **84.6796875 MiB**; NumPy2.5.3 / ONNX Runtime1.30.0 / Python3.13.14. This actual run was **not quiesced**. Memory includes libraries/result storage/runtime, not model-only RAM; direct VCM timing excludes wake/owner/gateway/UI/audio/providers. [Actual public receipt](../presentation/evidence/desktop-product.json) retains raw counters, resources, source/metrics/trial hashes and independent parity proof: every1,146 original WAV and header-bound decoded PCM plus all3,438 raw/accepted/full typed outputs matched frozen gold. The product result does not establish wake quality or change the failed wake gate.

The publication JSON has separate `historical_desktop`, `reference_desktop`, `desktop` and **`pi:null`** fields. The independently read-back **2026-10-24 reference-driver measurement is 9.338650/15.342980 ms p50/p95 under unquiesced publication load**, with the same 1,146 originals/3,438 calls and unchanged decisions. It is not a new-adapter, idle-host or end-to-end measurement; [its own receipt](../presentation/evidence/pc-reference-under-load.json) remains separate from historical timing. A future current adapter receipt must bind the same graph/policy and state hardware, scope, original/call denominators and measurement time. Pi timing/RAM/thermals/live microphone are unmeasured; PC results are never relabeled as Pi.

The 258,912 count covers the VCM only:

| Companion | Additional parameter / graph cost |
|---|---|
| Native Hey Iris | 82,114 parameters; separate `artifacts/iris/wake/iris.onnx` |
| Owner / overlap | 129,985 runtime parameters;542,023-byte ONNX; experimental |
| Silero VAD | Separate2,327,524-byte ONNX; parameter count not asserted |

**New completed synthetic wake study:** the existing native detector detected **1/32** positives with **0/68** false wakes; a separately trained same 82,114-parameter companion detected **21/32** with **2/68** false wakes. Both failed the predeclared gate; **no replacement was promoted**. [Actual compact study receipt and immutable source binding](../presentation/evidence/wake-smoke-status.json). This is separate from older calibration, not live human evidence. Static-site/software QA cannot declare an acoustic release pass.

Actual current graph byte counts/hashes are exported in `iris.json`. Original native wake metadata records calibration 17/32 synthetic positives and 0/660 negative clips at its locked policy. This is **not** a reliable household false-wake or original-human qualification claim. A separately versioned wake study must keep its own graph/receipt identity; no unobserved new gate is called passed by this card. The original-human wake-to-command owner check remains experimental and may veto an accepted VCM command.

## Application and synthetic provenance

Only final accepted, explicitly enabled live microphone commands can reach existing actions, after the independent owner/overlap and typed validation guards. Tentative events, WAVs, checks and benchmarks remain non-actuating. Recognition output and later application veto/outcome are separate facts.

Recognized light commands bind to `light.hue_color_lamp_1`. Recognized CALL opens `camera.dining_area_main_camera_tc70_mainstream`, **video only**, not a telephone call. WEATHER is OpenWeather-first; a cache keeps its actual retrieval/observation time, and synthetic fallback has no invented current timestamp. TEMPERATURE changes a **synthetic target only**, never an HA climate service; a real configured sensor is separately observed/read-only.

An unavailable integration may produce an explicitly synthetic outcome with `provider_accepted:false`, `device_state_confirmed:false`, a visible source badge and “Synthetic — no device changed.” Unknown real-provider outcomes must remain unknown, without an automatic retry or synthetic-success substitution. Local timers/alarms/reminders are real persisted server jobs in the app, not provider-state evidence. All static deck outputs are explicitly examples; they create no jobs or actions.

One host-owned audio path plays one prebaked wake acknowledgment, the accepted-intent beep, then success/failure only after a final dispatcher result. Cue delivery itself is not device confirmation. Replay and benchmarks stay silent. Build-time synthetic speech is not a runtime recognition service or real-human evidence.

## Sources and notices

The [citation audit](IRIS-CITATIONS.md) records exact dataset revisions, observed primary sources, unresolved rights, companion and presentation tooling notices. No root license is introduced and no blanket MIT grant is made for weights, recordings, voices or upstream datasets. Use the immutable scientific snapshot to reproduce the original experiment; use this publication's source/receipts to identify the application integration.
