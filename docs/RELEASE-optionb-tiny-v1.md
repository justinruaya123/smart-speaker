# Final OptionB architecture and release-candidate freeze

Status: **architecture final; adapter parity verified; not promoted or installed.**
Baseline: `justinruaya123/smart-speaker@a130ce88a6c76fda7ba6cfb6f54d940d2336b558`.
Scope: ME-02/03/04/05/06/07/08, the accepted finite OptionB milestone, not arbitrary speech or completion of the broader assignment.

**Current benchmark scope, 2026-10-23:** the user clarified that the present benchmark primarily concerns **accuracy and latency on the PC**. Pi/microphone work is deferred and is **not a blocker for reporting or completing this PC benchmark**. Prior frozen protocols retain their historical device-gate language; this scope clarification does not change the model/policy, enable actions or claim fresh-blind or hardware qualification.

## Final model — no further encoder search

Ship the existing **width-160, 258,912-parameter OptionBTinyVCM-1** only after the remaining promotion checks pass:

- Complete mono 16 kHz utterance; unchanged causal 400-sample Hann window, 160-sample hop, 512-point FFT, 40 HTK log-mel bins, 20–7600 Hz, log floor `1e-8`.
- Original equal-utterance HUMAN TRAIN per-bin normalizer, fitted on exactly 1,889 admitted recordings. No inference-only normalization change.
- Feature stride two. Causal kernel-five stem; four width-160 residual blocks, each with two depthwise/pointwise stages, kernel seven, dilations 1/2/4/8, per-frame LayerNorm and SiLU. Training dropout remains 0.1.
- Valid-length masked mean and maximum pooling, concatenated to 320 features; **one 32-way complete-command/UNKNOWN head**. Class zero rejects. No independent intent/slot head, transcript, cue rescue, extra encoder or slot rounding.
- Preserve original mixed **epoch 44**, original class order and FP32 ONNX. Neither failed fine-tune is a replacement.
- Preserve the separately selected **posterior ≥0.50 AND top-minus-second margin ≥0.10** demo policy. This is not the original strict policy, new acoustic training or release qualification.

The width-320 alternative was instantiated **only to count** its 927,392 parameters. It was not optimized, exported or shipped. No measured advantage justifies it.

Exact five-file identities, source/data bindings and promotion boundaries are in [the current v3 verification freeze](../experiments/speaker-optionb-release-v1/protocol-v3.json). The graph SHA-256 is `c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce`. The original v1/v2 protocols and evidence remain preserved, not silently rewritten.

## Data and evidence are not changed

The original sampler remains command/negative-scope balanced, then declared-speaker balanced, then recording sampled with replacement. Human-only batch quotas remain 24 human positives /8 human negatives; mixed quotas remain 12 human positives /8 human negatives /12 supplied synthetic positives. The original fitting source and normalization memberships are byte-bound; this work performs zero fitting updates and no policy selection.

Do not import evaluation negatives, promote ambiguous references to UNKNOWN, alter source labels, round unsupported numbers/targets into one of the 31 commands, or hide failed commands. Synthetic recordings do not fill independent-human coverage gaps. Historical source-lineage/voice-alias caveats and the two failed continuation studies remain intact. All old experiment source, receipts and per-command tables are preserved.

## SpeakerUI path — one recognition decision, no actuation

[The dedicated worker](../scripts/runtime/optionb_worker.py) is an **explicit opt-in**, not a root/submission default replacement. It pins all five candidate files, including the recipe itself, against the epoch44/demo identities before loading ONNX. A self-consistent but different bundle cannot silently replace the candidate.

The adapter reuses the existing WAV validation, capture, wake/endpoint and JSONL transport machinery. It forwards the tiny runtime's `output` and `application` unchanged, rejecting inconsistent accepted packets rather than applying the historical `.95/.90` or other independent intent/slot gates. Gate-level capture faults remain rejections. `qualified_output` rejects and `actions_enabled=false` remains explicit for every source, including microphone capture.

SpeakerUI's worker allowlist now accepts `scripts.runtime.optionb_worker`. For the explicit atomic OptionB contract, the gateway honors worker-side `actions_enabled=false` **before local commands and external provider calls**. This veto is scoped to the OptionB preview contract: legacy local time/weather/timer/reminder behavior retains its original meaning when only device actions are disabled. The gateway does not recompute model acceptance. Existing WAV non-actuation, provider configuration, typed validation and duplicate-event protection remain. Failed capture/control operations publish error state before their failed reply and attempt stream/gate cleanup even when cleanup itself fails. No physical provider action was tested or enabled.

After the normal SpeakerUI build, select this worker using the existing gateway's environment contract:

| Variable | Preview value |
| --- | --- |
| `INFERENCE_MODE` | `onnx` |
| `SMART_SPEAKER_WORKER_MODULE` | `scripts.runtime.optionb_worker` |
| `SMART_SPEAKER_ROOT` | Absolute repository root |
| `SMART_SPEAKER_PYTHON` | Absolute existing root `.venv` Python |
| `SMART_SPEAKER_BUNDLE` | Absolute `.cache/speaker-optionb-tiny-demo-v1/run-01/bundle` |
| `SMART_SPEAKER_THREADS` | `1` |
| `SMART_SPEAKER_TOKEN` | A fresh local launch token, not a provider credential |
| `ACTIONS_ENABLED`, `HA_ENABLED`, `SPOTIFY_ENABLED`, `OPENWEATHER_ENABLED` | `false` |
| `SUBMISSION_RESPONSE_PLAYBACK` | `false`; the explicit OptionB preview also stays silent if a previous launch left this `true` |
| `SMART_SPEAKER_WAKE_ROOT` | Optional explicit root containing the separately verified `wake/` assets, required before wake/microphone operation |

The wake root does not select a larger VCM: only its existing Hey Jarvis/Silero assets are loaded. Wake/VAD footprint must be reported separately. This adapter does not establish speaker-owner verification or overlap recovery.

## What actually passed

[Current artifact/adapter results](../experiments/speaker-optionb-release-v1/parity-results-v3.json):

- Exact selected checkpoint → FP32 ONNX parity for variable frame lengths and a mixed-length batch; largest measured absolute logit difference `3.814697265625e-06`, within the frozen `3e-5` checks.
- Six capture-fault cases reject without actions.
- **1,146 original human WAV/PCM replays** through the adapter preserve every original test/holdout decision and exact cohort identity. Normalizer, labels, ontology and policy remain unchanged.
- Holdout's three declared speakers are disjoint from HUMAN TRAIN/DEV/calibration; recording IDs, original WAV digests and PCM digests also have zero overlap. Declared identity separation is not independent voice-alias adjudication.

[Current native SpeakerUI parity](../experiments/speaker-optionb-release-v1/speakerui-parity-v3.json): **96 original human holdout WAVs** pass through the actual ONNX worker, compiled gateway HTTP route and actual UI event reducer, with exact output/application/slot parity. Its real Python preflight binds consumed/canonical manifests and reference predictions, verifies original WAV and decoded-PCM identities, and supplies the independently frozen human cohort. The native harness rehashes the exact WAV bytes submitted and rechecks the proof after replay. Empty, missing, duplicate or substituted cohorts cannot issue a verified receipt. This is software and recorded-audio evidence, not a browser-layout, live-microphone or Pi test. The total 77 accepted events includes ambiguous unscored rows; do not call all of them correct.

The v2 focused suite recorded **68 passing Python tests**; one unrelated optimizer smoke is deliberately deselected because no fitting is in scope. SpeakerUI type checks report zero errors/warnings, all **115 offline tests** pass, and the frontend/backend production build succeeds. The preserved original full demo verifier and metadata-wording addendum also pass again, including their 399 historical bindings.

### Preserved independent-review failure and additive correction

[The first independent review](../experiments/speaker-optionb-release-v1/independent-review-v1.json) failed on four concrete defects: unbound consumed manifest counterparts/self-referential cohort checks; missing native-harness cohort/WAV/PCM identity proof; a preview veto that also disabled legacy local operations; and stale capture state after failed re-listen. The first numerical outputs were not shown to be altered, but the verifier's claims needed stronger evidence.

The original eight source snapshots, protocol and receipts are preserved through [the byte-bound snapshot manifest](../experiments/speaker-optionb-release-v1/review-v1-source/manifest.json). The additive v2 protocol binds **51 actual inputs**, including all consumed manifests, independently pinned canonical counterparts, corrected source and compiled native gateway files. Regression tests exercise dropped/duplicated/substituted/empty cohorts, changed reference/WAV/PCM identities, legacy events carrying the real disabled-device flag, failed re-listen/cleanup and bounded JSONL EOF/shutdown. The v2 rerun preserves every original complete-command decision and the exact per-wording table digest; it does not retrain, relabel or reselect anything. v1 receipts remain historical observations, not the authoritative corrected verification. [Independent v2 closure](../experiments/speaker-optionb-release-v1/independent-review-v2.json) passes with no remaining security or logic findings; it independently checks the actual 51 bindings and cohort consistency. This closes software review only, not the pending fresh-human/Pi promotion gates.

### Preserved failures and coverage

| Human regression cohort | Correct accepts / supported positives | Wrong accepts | Negative false accepts |
| --- | ---: | ---: | ---: |
| Holdout | 72/80 | 3 | 1/14 |
| Broader test | 393/609 | 59 | 24/256 |

Exact full-reference wording results remain **58/62** on holdout (one wrong, three rejected) and **173/232** on test (four wrong, 55 rejected). These exposed cohorts are **not a fresh blind check**. Twenty-one of the 93 definitions still lack exact human evaluation-reference evidence; metadata matching is not independent listening. The 185 ambiguous test references and two ambiguous holdout references remain unscored, not true UNKNOWN negatives.

Known holdout mistakes remain visible: “Play a song” → NEXT; 18°C →22°C; true UNKNOWN “More heat” →PAUSE. The unchanged [per-command table](../experiments/speaker-optionb-tiny-demo-v1/per-command.csv) remains authoritative for the other command-level failures. Lowering posterior rejection does not make these mistakes disappear.

## Current PC accuracy and latency benchmark

[The original PC protocol](../experiments/speaker-optionb-release-v1/pc-benchmark-protocol.json) and [original PC results](../experiments/speaker-optionb-release-v1/pc-benchmark-results.json) keep epoch44/.50/.10 and all source identities unchanged. The fresh CPU-only process replays all1,146 original human test/holdout recordings three times after ten warmups:3,438 timed calls, with accuracy counted once per original recording. All calls execute the actual graph and reproduce the frozen raw/accepted decisions. Ambiguous references remain unscored; exact wording and broader commands remain separate.

Host: **Intel Core i7-11700, Windows11, Python3.13.14, CPUExecutionProvider**, one ONNX thread and one frontend BLAS thread, process affinity unchanged. Overall recognition latency is **4.82ms p50 /8.41ms p95**. This is external wall-clock time around `Runtime.recognize`, including PCM checks, frontend/normalization, CPU ONNX and complete-command decision. It excludes file loading/decoding/hash checks, speaking time, wake/VAD waiting, worker/gateway/UI, TTS and provider overhead. It is neither graph-only latency nor a full microphone interaction delay. Cold runtime construction is26.04ms, with20.43ms ONNX session loading; neither is full application-launch time.

Broader human accepted results remain holdout72/80 (90% recall,96% precision,3wrong accepts including1/14 negatives) and test393/609 (64.53% recall,86.95% precision,59wrong accepts including24/256 negatives). Exact wording remains58/62 and173/232. Speed is measured; the accuracy limitations are not hidden or repaired by changing the policy. No synthetic repeats are counted as human evidence, no timing repeats are counted as independent acoustic examples, and no thresholds are selected from these replays.

The PC driver has seven passing timing/output-guard tests; the combined focused Python suite passes **75 tests**, with the unrelated optimizer fixture excluded. [Independent PC benchmark review](../experiments/speaker-optionb-release-v1/pc-benchmark-review.json) passes: all3,438 raw timing rows, canonical recording/repetition counts, split/overall percentiles, RTF, frozen accuracy/wording counts and4 PC +51 release bindings reconcile. The suggested additional mocked driver-failure fixtures are non-blocking hardening, not claimed completed tests.

**Fresh human evidence:** [the prompt plan](../experiments/speaker-optionb-release-v1/fresh-check-plan.json) remains pending because the user confirms there are no new recordings. Its93 wording definitions +13 authored rejection prompts for three speakers (318 planned takes) are not recordings or adjudicated audio gold. This limits unseen-speaker claims; it does not prevent reporting the completed exposed-cohort PC benchmark. Future full references and recording manifests must be independently adjudicated/frozen before inference, with ambiguity separate and no refitting/reselection.

**Pi/microphone evidence:** deferred by user for now, not a current PC benchmark blocker. No SSH target or Pi operation is requested. Any later physical-device/live demo remains distinct evidence; PC timing must never be relabeled as a Pi or microphone result. The broader assignment's eventual hardware deliverable and existing action/qualification boundaries are not silently waived.

## Reproduction of the completed bounded checks

The current entry points below preserve the frozen model and policy. For the final non-Pi delivery and the fresh-recording limitation, see [the completion handoff](#non-pi-handoff-completion--2026-10-23).

From the repository root, using the existing environment (on Windows replace `python` with `.venv/Scripts/python.exe`):

```sh
python -m scripts.runtime.optionb_worker --bundle .cache/speaker-optionb-tiny-demo-v1/run-01/bundle --check
python -m scripts.test.optionb_release_check --protocol experiments/speaker-optionb-release-v1/protocol-v3.json --output .cache/speaker-optionb-release-v1/NEW_RUN
python -m scripts.test.optionb_pc_benchmark --protocol experiments/speaker-optionb-release-v1/pc-benchmark-protocol-v2.json --output .cache/speaker-optionb-release-v1/NEW_PC_RUN
```

From `SpeakerUI/`, after its normal check/test/build, using the locally pinned Node24 executable:

```sh
node scripts/optionb-parity.mjs .cache/speaker-optionb-release-v1/NEW_RUN/speakerui-parity.json
```

The parity script resolves its receipt path relative to the repository root, and creates it exclusively. Its preflight uses the current v3 protocol. Never overwrite `run-01`, `run-02`, `handoff-run-01`, `handoff-run-02` or original study outputs. Exact external model/cache/audio files are prerequisites; the local caches and recordings are not newly committed or bundled here. No installation, root default swap, dependency-lock change, commit or push is implied by these checks.

## Non-Pi handoff completion — 2026-10-23

The user requested continuing the outstanding handoff **except Pi tests**. This follow-up completes and locally commits the non-Pi source/software/evidence handoff; it does not authorize fitting, threshold changes, installation, provider actions, a push, or a release-quality waiver. The frozen v1/v2 protocols, earlier reviews and measurements remain byte-preserved.

[The corrected additive handoff verification](../experiments/speaker-optionb-release-v1/handoff-verification-v2.json) records the actual repeat commands and output identities. The [first handoff verification](../experiments/speaker-optionb-release-v1/handoff-verification-v1.json) remains historical evidence from before the final review found H1.

[That failed handoff review](../experiments/speaker-optionb-release-v1/handoff-review-v1.json) found one additional integration defect: an inherited `SUBMISSION_RESPONSE_PLAYBACK=true` could send an unsupported `speak` request to the OptionB worker and stop capture. The explicit guard now returns the existing no-speech **preview** outcome instead of **disabled**. The real gateway regression first failed, then passed; it verifies zero speech requests, speech errors or capture-state changes, while a legacy local-response control still speaks exactly once. This is mocked gateway/lifecycle evidence, not physical capture/playback. [Five pre-correction source snapshots](../experiments/speaker-optionb-release-v1/review-v2-source/manifest.json), all failed reviews and old receipts are retained. The v3 lock binds **62 inputs**, including the corrected source and built gateway, preserved snapshots, previous locks/review and new tests. The [current PC lock](../experiments/speaker-optionb-release-v1/pc-benchmark-protocol-v2.json) binds **five inputs**. CLI defaults have separate red/green regression checks and select these current locks. No acoustic decision, source label or policy changed.

- **77 focused Python tests pass**, with only the unrelated optimizer fixture deselected. **117 offline UI/server tests pass**, type checks have zero errors/warnings, and frontend/backend production builds succeed using the existing pinned Node 24.13.0 and root Python environment.
- A fresh `handoff-run-02` verifies checkpoint/ONNX parity, all six capture faults and all **1,146 original human adapter replays**. Every selected decision, cohort metric and wording-table digest remains unchanged. The **96 original native-worker → compiled-gateway → UI-reducer replays** preserve the exact v2 recognition decisions/digest; only verification-lock metadata changes.
- Both original demo verifiers pass again: **4,896 canonical original/stress cases**, 32 graph spot replays, all 138 policy ranks and **399 historical bindings**; the metadata-wording addendum stays reporting-only.
- The corrected `pc-handoff-run-02` repeats all **3,438 timed CPU calls**. Its recognition latency is **5.36 ms p50 /8.69 ms p95**, with unchanged accuracy and wording results. The earlier repeat (**4.96/8.02 ms**) and original benchmark (**4.82/8.41 ms**) remain preserved. These are separate repeat-run observations, not an acoustic improvement or a favorable-run replacement. Timing scope/exclusions are unchanged.

The generated **`dist/optionb-pc-handoff-v2/optionb-pc-handoff.zip`** is a source/model **review handoff**, not an installed or qualified distribution. Its `REVIEW-MANIFEST.json` identifies the actual delivery commit, exact Git blobs and explicitly included local inputs. The adjacent `archive-verification.json` records archive integrity and extracted execution. The previous v1 archive is retained. No raw recordings, feature tensors, source Parquet, credentials, environment or dependency payloads are included. Running the extracted checks reuses existing dependencies; it is not a clean-machine installation claim. Cached manifests retain their original paths; full raw-corpus replay still needs the original external inputs. Exact candidate loading, software fixtures and independently reconciled packaged timing/decision evidence are distinct checks.

**The only remaining acoustic check in the non-Pi plan is blocked on real inputs:** [the unchanged fresh-check plan](../experiments/speaker-optionb-release-v1/fresh-check-plan.json) requires three genuinely new independent human speakers, each recording the 93 supported wording prompts and 13 rejection prompts. Its **318 planned takes are not recorded or adjudicated examples**. Preserve the original WAVs, speaker/session/source identities and audio/PCM hashes; independently listen and adjudicate full references without predictions, then freeze the actual manifest before running the unchanged model once. Keep disputed references separate. No existing test/holdout replay, synthetic voice or authored prompt closes this requirement.

[Independent handoff closure](../experiments/speaker-optionb-release-v1/handoff-review-v2.json) **passes**, with no blocking security or logic findings. The reviewer reran all nine gateway/boundary cases and two CLI-default cases, traced the speech path and independently verified all **62 release +5 PC bindings**, predecessor identities and preserved source snapshots. This closes H1/software delivery review, not release promotion. Earlier optional JSONL, exact-buffer and PC-driver failure-path hardening suggestions remain unimplemented and nonblocking.

**Pi/live-microphone tests are excluded from this continuation, not passed.** Qualification remains false, actions remain disabled, and the candidate remains opt-in rather than replacing installed defaults. A local commit/archive closes the durable software handoff only; fresh-human reliability and a later real-device demonstration remain distinct evidence.
