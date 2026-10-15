# VCM live final v3 — implementation and completion handoff

> For Codex: execute this handoff with the existing project environment and A100 allocation. Use `superpowers:executing-plans` when available. This is an implementation/training/qualification task, not another research proposal. Complete the checkboxes with actual evidence; an unchecked item is not a passing result.

**Goal:** a small offline voice model that accepts the preserved commands from previously unseen people, including natural pronunciation and modest wording variation, while rejecting uncertain or unsupported speech rather than returning a wrong intent or slot.

**Baseline inspected:** `56372e2fa30e1c6dde21c51034105cba4530d072`, branch `codex/vcm-intent-slots-v1`, 15 October 2026. This commit adds failure diagnostics to candidate `6aee6bb`; it does not contain newly improved weights. **Status of this document: execution handoff only; no v3 model has been trained or qualified by its creation.**

**Architecture direction:** retain the compact acoustic encoder initially, but make a jointly recognized complete command the primary target for the finite catalog. Decode that command atomically to its existing typed JSON. Use explicit rejection and measured correctness, not fifteen independent absence gates or a fixed confidence number, to determine acceptance.

**Stack:** existing Python/PyTorch training, NumPy/audio frontend, ONNX/ncnn export, SpeakerUI and resident runtime. No new service platform or research framework is needed.

## 1. Latest user authorization and immutable boundaries

This handoff records the user's subsequent authorization to change the model, data preparation, training, calibration, decoder, service integration and **gates outside `vcm-benchmark`**. The old 0.95 intent / 0.90 slot thresholds, all-role absence gates, mandatory acoustic-span review, six-arm recipe and CPU-inference hold are **not requirements for v3**. Replace them where appropriate. CPU preprocessing, numerical tests, export, ncnn/ORT parity and resident inference are authorized; use the allocated GPU for substantial training. Do not interrupt another agent's job.

The following remain fixed:

- The pinned upstream `airimonda/vcm-benchmark` implementation, references, defaults, scoring and trial accounting. Its complete run is the **first formal acceptance gate** for each frozen release candidate. Unit tests, fitting and development diagnostics necessarily precede that gate. There is no benchmark-aware recognition mode or benchmark-only threshold.
- Every existing benchmark and custom verbatim, its expected intent, typed slots, defaults, IDs and negative outcome. Preserve all current catalog/grammar-supported commands, not just the 93 benchmark phrases. Additional meaning-preserving **training variants** are allowed; do not replace or delete the scored originals.
- Offline direct audio-to-command inference, small enough for the Pi. No ASR/STT, transcript/phonetic fallback, LLM, cloud recognition or pretrained command-recognition weights. Existing authorized wake/VAD models may remain. Training-time TTS is allowed.
- No fitting, selection, calibration or speaker adaptation using the eventual live test participants, including the user. Entity examples from other voices are not permission to enroll the test person's voice. Zero-shot here means **unseen speaker**, not arbitrary unconfigured names.
- Preserve previous artifacts, reports and failed outcomes. Create a versioned v3 run; do not rewrite v2 receipts to make old failures pass. Preserve dirty work and other agents' outputs. Keep physical device, messaging and calling actions disabled during qualification.

This is scoped completion authorization, not permission for unrelated cleanup, destructive resets, dependency churn or external action integrations.

## 2. What the new diagnostics already establish

Sources: [failure review](HANDOFF-vcm-mixed-final-v2-error-review.md), [numeric receipt](../experiments/vcm-mixed-final-v2/post-handoff-diagnostics-20261015.json), [v2 results](EXPERIMENT-vcm-mixed-final-v2.md).

| Evidence | Consequence for this task |
| --- | --- |
| All six fitted slot calibrators have maximum possible scores below 0.90 | The old slot acceptance policy is unreachable; retire it for v3, not the required commands. |
| All 1,895 human training rows lack explicit role-state labels; their entity targets are masked | Use trusted complete-command labels directly and repair auxiliary supervision. Human examples must not silently lose their useful targets. |
| Synthetic training has only two voice IDs | Expand voice and acoustic variation rather than repeating the same small grid of seeds. |
| Fresh-human raw intent: 43/93; final exact accepted: 0/93 | This is also an acoustic recognition problem, not only a threshold problem. |
| Original-human raw intent: 352/636; final exact accepted: 25/636 | Lowering gates alone is not an acceptable fix. |
| `Send my message` was accepted as `WEATHER` | Count wrong accepts on supported speech, not only false accepts on negatives. |
| Alarm-minute loss denominator is zero | Any retained learned minute path needs actual training targets. A legal whole-hour default does not prove spoken-minute recognition. |

The original synthetic test score remains 19/3,368 exact accepted commands. Passing export parity or generating a package does not change recognition accuracy.

## 3. Task A — freeze the target and create the smallest useful tests

**Use:** `scripts/runtime/pcv_schema.py`, `scripts/runtime/vcm_intent_slots_contract.py`, the selected catalog, benchmark adapter and current source map. Put compact v3 protocol/evidence under `experiments/vcm-live-final-v3/`; put payloads in the existing ignored shared-cache layout.

- [ ] Inspect the latest branch, local status, running jobs and actual artifact bindings. Reconcile any newer work without resetting or replacing it. Register v3 in `EXPERIMENTS.md` and record the new authority above so old recipe assertions do not silently block execution.
- [ ] Freeze a manifest of every original positive and negative phrase, spoken rendering and expected typed JSON. Hash the benchmark code, default data/cache, grammar and catalog separately. Use the actual pinned loader output, including duplicates, rather than substituting the README's historical count.
- [ ] Build a stable codebook of **distinct complete typed commands**, not separate classes for synonymous phrases. Derive it from all preserved catalog entries and advertised grammar forms. A baseline/custom phrase must map to the same result before and after this change. Do not turn merely valid JSON ranges into an unsupported claim of acoustic coverage, or remove an already advertised form.
- [ ] Add failing tests for original-reference equality, conflicting labels, every required command's decoder reachability, rejection output, numeric/name minimal pairs, generic defaults versus explicit unknown names, and train/evaluation overlap. A deliberately correct test vector must produce its exact command; reject-all must fail qualification.

If an advertised compositional or enrolled-entity path cannot be represented by the finite codebook, preserve and test its structured acoustic path inside the same VCM rather than deleting it. The selected fixed catalog is the immediate completion target; do not start an unrelated open-vocabulary project.

## 4. Task B — build the training scaffold, not a synthetic-only success claim

**Use:** the current mixed corpus/data preparation and shared human recordings. Create v3 views rather than changing frozen v2 manifests.

- [ ] Use the trusted human utterance reference and existing typed command to supervise the joint command target. Derive finite values and applicable role presence from an unambiguous reference/approved phrase mapping. Record derived-label provenance. An omitted/default field is not evidence that an entity was acoustically spoken. Ambiguous or incomplete annotations remain masked only for the affected auxiliary target, not for a known whole-command label.
- [ ] Report unique recordings, people/voice IDs, label coverage and effective loss denominators per command, value and source. Fail preparation for a required learned output with no examples. Do not treat a manual acoustic-span annotation campaign as necessary for whole-utterance classification; localization remains optional unless a retained model actually requires it.
- [ ] Synthesize the **exact supported strings first**, including every custom light/song/contact/task and every supported numeric value. Add a smaller set of verified meaning-preserving variants: articles, polite prefixes, contractions and approved word-order variants. Keep numbers, negation, entities and requested actions semantically correct. Changing twenty to thirty requires a different label.
- [ ] Default scaffold: target 12 available licensed English training voice presets and speeds 0.9/1.0/1.1; reserve different presets for development, calibration and evaluation. Use a second already available licensed TTS family when feasible, but do not stall the run on new infrastructure or claim preset IDs are independent humans. Keep the previously reserved final-TTS voices/audio out of fitting. Record actual counts if available assets differ.
- [ ] Use training-only gain, moderate room response/background noise, small timing variation and mild feature masking. Retain clean examples. Do not crop away a number, negation or final entity token. Changed or noisy waveforms inherit the original recording/voice split; splitting derived files does not create unseen speakers.
- [ ] Include close hard negatives: missing required numbers, unknown names after valid prefixes, out-of-range values, contradictory requests, negation, unrelated speech, silence and overlapping commands. Include confusable supported pairs such as fifteen/fifty, thirteen/thirty, on/off, AM/PM and the actual catalog names.
- [ ] Start with batch sampling mass of 40% human positives, 40% synthetic positives and 20% negatives, with negatives from both sources where available. Within positive sources, balance intent first, then command/value, then speaker/recording. Do not let hundreds of percentage labels overwhelm the other intents. Record actual sampling mass and any development-justified adjustment.

Training text may match fixed evaluation verbatims; evaluation waveforms, their augmentations, participants and reserved voice presets may not enter fitting. Cache by waveform/frontend identity and fit normalization on training only. Perform deterministic waveform/label QA and a stratified listening spot-check where listening is available. Never claim listening occurred when it did not. A generation fault stays a generation fault, not a recognition result; version corrections before scoring and never replace takes based on recognition outcomes. Exhaustive listening to every synthetic take is not a blocker for diagnostic training.

## 5. Task C — train one direct complete-command model

**Starting point:** `scripts/models/vcm_intent_slots.py` and the current mixed trainer. Reuse the width-128 temporal encoder, padding masks and attention pooling initially. Put v3-specific implementation in focused canonical modules, not inside the evidence folder.

- [ ] Add a `K + 1` joint output over the frozen complete-command codebook plus rejection. Reuse the pooled acoustic features, including role-pooled information where useful for values. Every trusted human/synthetic command contributes to this target. For a finite codebook prediction, the command ID selects one existing JSON atomically; do not independently choose an intent, number and name and assemble a plausible combination.
- [ ] Retain useful intent/value heads as auxiliary training signals or measured consistency checks. Start with loss `CE(command) + 0.25 * CE(intent) + 0.25 * mean(observed finite-head CE)`. Mask only genuinely unavailable auxiliary labels and report each denominator. Do not gate runtime acceptance on all fifteen unrelated absent-role confidences. Do not ship unused heads merely to satisfy an old architecture receipt.
- [ ] Before a full run, overfit a small trusted, multi-command diagnostic training subset without augmentation. Require at least 99% raw complete-command accuracy on that subset and correct singleton/batched behavior. This is a software/optimization test, not generalization evidence. Failure means debug labels, masks, features and loss; do not launch six seeds.
- [ ] Train from random initialization on the existing A100. Start with AdamW, learning rate 0.001, weight decay 0.0001, gradient clipping 1, batch 64, warmup plus cosine decay, at most 60 epochs and early stopping after eight development evaluations without improvement. Define an epoch from the actual new corpus/sampler coverage, not a permanently hard-coded 96 updates. Record actual updates, losses and human/synthetic development curves.
- [ ] Start with one seed. Preserve the best raw development checkpoints, then compare a small shortlist using the actual final decision policy fitted on separate calibration data and evaluated on human development. Synthetic development is a scaffold/debug signal, not a substitute for human selection. Validate a second seed only after the first recipe is useful.
- [ ] If training accuracy is high but unseen-human development is poor, fix coverage, label quality and domain variation first. If the verified model underfits, allow one measured capacity/pooling revision. Do not start a broad architecture sweep or package a poor candidate just because the training budget ended.

The recommended model is a testable starting design, not a claim that a new head alone produces 95% human accuracy. Preserve compositional/enrollment functionality that the frozen contract actually requires. Count all shipped heads and models in the Pi resource report.

## 6. Task D — replace old gates with an error/coverage operating point

**Use:** the current calibration, decoder and runtime interfaces. Version the new policy; preserve old policy bytes.

- [ ] The runtime consumes audio and configured catalog context only. It must never receive the expected phrase, trial ID, playback order, source filename label or benchmark mode as a recognition feature.
- [ ] Return one exact supported command or `accepted=false, command=null`. Preserve only the defaults already defined by the contract, such as the recognized generic-light command and schema constants. Explicit unknown/missing arguments must not become generic commands. No nearest-name substitution, snapping an unsupported number to a supported number, or filling a value from a likely intent.
- [ ] Fit a small full-command correctness/abstention policy using held-out calibration examples, including supported errors and hard negatives. Candidate features are joint score, top-two margin, rejection score and available acoustic value/name evidence. Use global or adequately supported intent-group policies instead of hundreds of unstable per-phrase calibrators. Choose thresholds to maximize accepted exact recall subject to the declared wrong-command limits, then check them on development.
- [ ] Confidence 0.95 is not a guarantee. Thresholds may change outside the benchmark; do not inflate reported probabilities or relabel a heuristic as a calibrated probability. Check reliability, achievable score range and actual coverage. A policy that cannot meet both error and recall requirements is a failed candidate, not a reason to manipulate the score scale.
- [ ] Validate all required runtime paths from the contract, including ones absent from calibration observations. Fixed unit/timezone values need no invented acoustic confidence head, but the complete command still requires reliable recognition. Unsupported/ambiguous acoustics must retain a genuine rejection path even though the valid output catalog is finite.

For each evaluation, show raw intent, raw complete command, accepted exact command, rejection reasons, wrong accepted positives and negative false accepts separately. This prevents calibration and acoustic failures from being conflated again.

## 7. Task E — use these acceptance gates, in this order

The numerical targets below are **v3 project release criteria**, not claims that upstream imposes these thresholds. Freeze them before candidate evaluation; do not weaken them after seeing failures.

| Gate | Required evidence and passing target |
| --- | --- |
| 1. Unmodified `vcm-benchmark` | Full default pinned run and its original reports first. At least 95% command-level accuracy separately for human and synthetic groups, zero observed wrong-command misfires and zero observed negative false accepts. Keep upstream intent/conditional-slot/false-wake metrics and denominators intact. Also compute strict typed exactness from the same actual application outputs; it must not conceal wrong custom fields that upstream does not score. |
| 2. Preserved catalog synthetic scaffold | Every original benchmark/custom form, supported value and declared negative, using reserved voices and pre-result QA. Accepted exact supported-command recall at least 99% overall and at least 97% for each voice; zero observed accepted wrong commands or negative false accepts. List every miss. Report generation coverage/faults separately; a missing generated case is not a passed case. |
| 3. Unseen-human command qualification | Speaker-disjoint human audio covering all supported intents, and an action-free live pilot with at least five previously unused people, at least 30 supported trials and ten negative trials each. Cover all intents and custom entity categories collectively; randomize a balanced phrase schedule before capture. Accepted exact supported-command recall at least 95% overall and at least 90% per person and per intent; zero observed wrong accepted commands and negative false accepts. Count first attempts, including rejections. |
| 4. Same deployed service | PyTorch/ORT/ncnn decision parity on real waveforms; the selected runtime works through SpeakerUI, mic, wake/VAD, endpointing and the resident worker. Benchmark the actual Pi for latency/resources and preserve the required upstream reporting. No hidden ASR/cloud dependencies, runtime fallback to a different model, duplicate command events or execution faults. |

Accepted exact recall = correctly accepted supported commands / all supported trials. Accepted precision = correctly accepted commands / all accepted commands, counting both wrong positives and negative false accepts as errors. Empty acceptance makes precision undefined, never 100%. Report human and synthetic groups separately. Add at least 500 predeclared hard-negative trials across held-out human and synthetic sources outside the small upstream negative set, keeping source-specific denominators and outcomes visible.

A finite test with zero errors cannot guarantee zero errors for every future voice. The release statement is **zero observed wrong commands on named tests with measured recall**, not universal perfection. Do not promise all accents, rooms or overlap conditions without evidence.

Use the already pinned upstream revision `ab39857cf73dd067d068fed388c321ab160dc191` and matching bound default cache unless a new official version is explicitly adopted and both versions are reported. Keep native recovery/retry accounting. No quick/limit/sim run, selected subset, best-of-repeated-takes score or changed reference is a release pass. Never tune on benchmark or live-test waveforms. Historical test cohorts already inspected remain regression evidence; repeated evaluation does not make them fresh.

**Physical benchmark availability:** the official benchmark plays wake-plus-command through a loudspeaker into the Pi microphone. An offline command-WAV replay of its pinned set is a useful first provisional check, not that physical benchmark. Run all available software/training/export work now. If hardware or people are unavailable, leave the physical/live gate explicitly pending and provide the executable test service; do not fabricate a pass or stop earlier software work waiting for a recording ceremony.

**Wake-owner compatibility:** upstream prepends a user's wake recording to commands spoken by other voices. A mandatory same-speaker wake/command gate would conflict with that test. Keep identity ownership outside the default recognition qualification profile; use the same declared wake/VAD profile for benchmark and initial live pilot. Do not silently disable ownership only when a benchmark is detected, or claim same-speaker isolation is qualified. Evaluate that separate optional capability on genuine paired recordings later.

## 8. Task F — integrate and finish, instead of delivering another plan

- [ ] Use the same frontend and authoritative application decision for replay, SpeakerUI and the resident microphone service. Test short commands, padding, gain, endpoint boundaries, trailing numbers/names, silence, overlapped speech and repeated wake/command sessions. Keep a pre-roll and do not truncate tokens with a training-only crop.
- [ ] Export the exact selected checkpoint and bind codebook, normalization, calibration and graph hashes. Run real ncnn and ORT parity; the old CPU hold is lifted for this work. Keep diagnostics out of the benchmark-discovered log: it must contain only the actual final command/rejection and measured timing, not expected labels or rejected candidate intents.
- [ ] Add focused tests to the repository test suite, update `scripts/modules.json` where needed, and run the relevant broader SpeakerUI/runtime/export regressions. Preserve unrelated known failures distinctly. Software fixtures do not count as speech-accuracy results.
- [ ] Implement one canonical v3 entry point with prepare/train/calibrate/evaluate/export/serve stages and a full acceptance runner. Reuse existing implementations. Document and run the exact real commands; do not publish invented commands or a plan that cannot launch. Use the existing root environment and runtime lock.
- [ ] Commit source, tests, the versioned protocol, compact actual results, every missed verbatim and the selected artifact manifest. Put large model/audio/log payloads in the existing artifact/release mechanism with verified hashes, not ordinary Git. Update `EXPERIMENTS.md` and the active handoff pointer. Preserve rollback to the previous installed bundle.
- [ ] Promote the selected bundle only after available accuracy gates pass; prepare the same bundle for physical acceptance. Use explicit status names such as `training_failed`, `accuracy_failed`, `offline_qualified_live_pending`, or `live_qualified`. Only `live_qualified` means the required live tests were actually passed. No report/card/ZIP existence check can set this status.

The final Codex response must name the actual commit, selected model/bundle, exact benchmark and live commands, measured human/synthetic accepted exact recall, wrong accepted commands, false accepts, per-intent/per-person failures, and any genuinely unexecuted physical step. Do not answer only with a new architecture suggestion or another generic handoff. Execute the authorized work and report what the evidence supports.

## Source map for implementation

- [Schema, defaults, baseline and custom forms](../scripts/runtime/pcv_schema.py)
- [Current compact acoustic model and losses](../scripts/models/vcm_intent_slots.py)
- [Human/auxiliary target construction](../scripts/train/vcm_intent_slots_train.py)
- [Mixed training and entity masking](../scripts/train/vcm_intent_slots_mixed_train.py)
- [Current calibration fitting](../scripts/train/vcm_intent_slots_calibration.py)
- [Current acceptance decoder](../scripts/runtime/vcm_intent_slots_decode.py)
- [Current runtime](../scripts/runtime/vcm_intent_slots_runtime.py)
- [Current pipeline stages](../scripts/vcm_intent_slots.py)
- [Finalization evidence and existing package](../final/README.md)
- [Upstream pinned benchmark](https://github.com/airimonda/vcm-benchmark/tree/ab39857cf73dd067d068fed388c321ab160dc191)
