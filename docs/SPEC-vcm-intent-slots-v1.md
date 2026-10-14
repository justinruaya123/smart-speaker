# VCM intent and slots v1: target architecture

Date: 2026-10-14. Status: user-selected architecture recorded for implementation;
not an implemented, trained, calibrated or qualified model.

Experiment ID: `vcm-intent-slots-v1`. Repository: `justinruaya123/smart-speaker`.
Source baseline: `codex/speaker-demo-v1` at
`5d872e3a249e5876aaa8130b76f6e3b31622355f`.
Implementation branch: `codex/vcm-intent-slots-v1`.

Read the [execution handoff](HANDOFF-vcm-intent-slots-v1.md) and
[design protocol](../experiments/vcm-intent-slots-v1/protocol.json) together.
The [requirements](../REQUIREMENTS.md) and [agent contract](../AGENTS.md) govern.
This commit records the architecture the user selected in conversation. It does
not authorize changing deployed weights, thresholds, providers or physical devices.

## 1. Goal and preserved scope

Build one small local acoustic model that recognizes an intent and all required
finite slots from complete spoken commands, including unfamiliar speakers. It
must return the existing typed command envelope or reject. No runtime transcript,
ASR/STT parser, LLM, cloud recognition, speaker-specific command model or
93-sentence-only replacement is allowed.

The preserved scope is the OptionB benchmark plus every existing personal-demo
extension. Report them separately using the same weights, frontend, normalization,
catalog and decision policy. Keep the 0.95 intent threshold and 0.90 threshold for
every supplied slot. A new score definition still needs versioned calibration;
unchanged threshold numbers do not make new scores equivalent to old ones.

Use one accepted wake for one command, then release ownership. An unfamiliar
person may initiate the next session. Speaker ownership is an additional acoustic
verification gate, not a routing rule for choosing the command model. Initially,
reject overlap and unverifiable handovers; do not claim recovery through overlap.

The supplied research handoff is evidence, not a measured result of this design.
Its uploaded `Pasted text.txt` had SHA-256
`c8fdc47c9bfa5ce45113315cf492688a434ebd11b8d06011edba36f3b4f54e40`.
It reports incomplete human coverage, examined historical tests, and missing local
shipping graphs. Those are source observations, not a fresh audit of the executor's
machine. The executor must inspect current availability and preserve other agents'
uncommitted preparation. A remote branch cannot include their uncommitted files.

## 2. Selected network

The user-selected architecture is:

`PCM -> existing log-mel frontend -> time-preserving CNN/TCN -> joint intent and
role-specific slot evidence -> finite typed assembly -> application gates`.

A separate identity/overlap backend gates the result for an owner-aware session.
There is no second command recognizer that rescues a rejected intent.

| Component | Target specification |
| --- | --- |
| Frontend | Reuse `vcm.frontend.SPEC` and `features`: mono 16 kHz, 40 log-mel bands, 400-sample window, 160-sample hop, FFT 512, uncentered framing. Preserve its mel/window/log details. |
| Normalization | Frozen per-feature training-only statistics; identity-bound to the graph and frontend. Never fit on development, calibration, test or enrollment queries. |
| Stem | Conv1d 40 -> 128 channels, kernel 5, stride 2, symmetric padding 2; one output per approximately 20 ms. |
| Encoder | Eight residual depthwise-separable temporal blocks, width 128, kernel 5, dilations `[1, 2, 4, 8, 1, 2, 4, 8]`. Keep the time axis. |
| Intent | Masked learned attention pooling over all valid encoded frames; linear classifier with 19 outputs, ordered OOV then the 18 `pcv_schema.INTENTS`. |
| Slot readouts | Learned query per semantic role, separate attention over the same sequence; explicit absent, present, conflicting and unsupported role states. |
| Finite values | Classification over schema/catalog-bound value sets; never scalar regression for exact integer slots. |
| Names | Content-trained acoustic embeddings matched to versioned enrollment examples in the relevant entity namespace, with open-set rejection. |
| Decision | Typed validation, evidence/completeness checks, unchanged numerical application gates; final accepted command or explicit rejection. |

The following are implementation defaults, not tuned performance claims: within
each temporal block use depthwise convolution, per-frame channel LayerNorm, ReLU,
pointwise 128 -> 128 convolution, training-only dropout 0.1 and residual addition.
Use 64-dimensional attention keys/queries and normalized 64-dimensional entity
embeddings. Mask padding before and after each block and in every pooling/loss.
LayerNorm is across channels of each frame, not across padded time. The stem and
blocks use symmetric context: this first model is complete-utterance, not a causal
streaming model. Variable-length padded training must agree with unpadded inference.

Reuse [the frontend](../vcm/frontend.py), rather than replacing it with a library's
similarly named mel transform. Retain the existing supported capture duration and
endpoint behavior for the first comparison. A graph with these new outputs must
have its own versioned interface; it is not a drop-in three-head PCV graph.

## 3. Intent, roles and command assembly

[The current schema](../scripts/runtime/pcv_schema.py) is authoritative. At the
pinned source it maps 19 OptionB labels to 18 runtime intents because VOLUME_UP and
VOLUME_DOWN share VOLUME with different operations. The benchmark contains 93
phrase definitions and 31 distinct positive typed outputs; these are not the same
as acoustic class counts. Generate and validate the complete existing catalog,
including extensions and negatives, before fitting. Preserve any larger selected
shipping catalog rather than reducing it to the case generator's output.

| Semantic role | Representation and constraints |
| --- | --- |
| Brightness / absolute volume | Separate role queries, categorical values 0..100. VOLUME operation is up/down/mute/set; percent is present exactly when operation is set. |
| Timer | Separate amount and unit readouts. Freeze their finite spoken value inventories from the declared grammar; convert only recognized valid pairs to integer seconds. The schema's 1..86400-second bound is not evidence for every spoken duration. |
| Alarm | Hour, minute and meridiem under the declared time grammar; normalize to HH:MM with Asia/Manila. Whole-hour and other implicit fields must follow an explicitly versioned grammar/default rule, not ad hoc guesses. |
| Temperature | Discrete values within the schema's -50..60 bound; C remains the declared unit. Values without training/qualification coverage stay visibly unqualified. |
| Color / operation | Small categorical heads covering the full existing declared inventory, not just the three benchmark colors. |
| Light / song / artist / contact / task | Role-conditioned content embedding plus enrolled candidates; maintain stable typed IDs, catalog versions and missing-name rejection. |

The current PLAY_MUSIC schema supports `song_id` and `artist_id`, and requires a
song when an artist is present. It has no `playlist_id`; do not silently add that
field or claim existing playlist support. The acoustic memory is designed to admit
a later playlist namespace, but its command schema/protocol must be explicitly
versioned before implementation. Preserve current song/artist extensions now.

Required distinctions:

- The intent head remains authoritative. Rejected intent, intent/slot disagreement
  or unsupported input cannot be rescued by a high-confidence number or name.
- Preserve `light.all` for supported generic-light requests and `contact.home` for
  supported bare CALL. An explicitly spoken unknown name must reject, not fall
  back to those defaults. MESSAGE remains intent-only compose/request-input, not
  dictated text.
- Default/constant fields need provenance. `timezone`, declared C, `light.all` and
  bare-CALL `contact.home` are not independently heard names or units. Their gate
  evidence comes from the qualified request/default rule, not a singleton softmax
  assigned confidence 1. Preserve every required-slot gate in the final envelope.
- Reject missing, conflicting, extra, malformed, out-of-range, negated or
  unsupported arguments. Validate the proposed combination rather than choosing
  a different low-scoring combination until the validator accepts something.
- Predict role state/validity before restricting candidate competition. A legal
  value's softmax score alone is not evidence that a slot was actually spoken.
- When a command contains two values or two targets, do not let attention silently
  discard the conflicting one. Include a complete-request validity objective and
  dedicated negative training/evaluation; its effectiveness needs acoustic tests.

Attention weights are not verified temporal boundaries. Add reviewed span labels
for a subset of connected numeric/name training recordings and a training-only
role-localization loss. Unknown span annotations are masked, not invented. Learned
slot spans are not speaker-continuity evidence and cannot be supplied as verified
owner coverage without separate validation.

## 4. Acoustic enrollment

Keep three to five original enrollment recordings per entity as a starting
protocol, not a demonstrated sufficient count. Preserve multiple examples instead
of treating a single average as mandatory. Bind each example to entity ID, role,
recording provenance/hash, verified speaker metadata where available, encoder,
frontend and normalization hashes. Display text is metadata, not the recognition
mechanism. Rebuild embeddings when their bindings change.

Train the content representation using different speakers saying the same entity
and hard negatives saying similar or different entities. Include isolated-name
enrollment and names inside connected commands. Do not reuse the ownership
embedding. Enrollment/query recordings must be disjoint; report person-disjoint
query performance and separately evaluate entities withheld from metric training.

For the first matcher, normalize embeddings; use the maximum cosine match over
retained examples for each entity, then compare entities. Exact duplicate examples
must not inflate a score. Require both absolute match evidence and a margin where
competitors exist, with calibration against nonmatching speech and near names.
A one-entity catalog still needs an absolute rejection criterion. Empty catalogs,
unknown names, ambiguous/homophonic alternatives and invalid/stale embeddings
reject. Catalog changes require a new binding and competition regression;
uncalibrated catalog-size regimes are not automatically qualified.

Classify entity presence before retrieval. The model must distinguish generic
PLAY_MUSIC from a failed named-song request, and generic lights from failed
named-light evidence. A matched artist cannot manufacture a missing song.

## 5. Joint training and controlled comparisons

Start the new command model from random initialization. The baseline objective
contains intent cross-entropy, masked role-state and categorical-value losses,
complete-request validity, entity metric learning and reviewed-span localization
where annotation exists. Average finite-value losses over observed valid roles,
not over padded or unknown labels. Log effective denominators for every loss.
A batch with no eligible entity pair or reviewed span skips that term safely.

For the initial implementation, use unit weights for normalized supervised terms,
0.1 for eligible entity metric loss and 0.1 for span localization. Training-only
whole-command contrastive and acoustic CTC terms are disabled in the first arm.
These are fixed experimental defaults, not empirically best hyperparameters.

Use the existing v1 optimizer budget as the reproducible initial baseline:
AdamW, learning rate 0.001, weight decay 0.0001, gradient norm limit 1.0, batch 64,
30 epochs, 96 updates per epoch, seeds 53/71/89. This is a new full-command
architecture study; do not confuse it with the existing small DemoKeywordNet
pilot or reinterpret that pilot's metrics as full-slot results.

First perform frozen reference/catalog/crop attribution where the shipping graph
and original PCM are available. Missing shipping payloads block reproduction and
comparative claims, not safe implementation or software tests of the new model.
Never substitute older recurrent metadata or a newly fitted graph for the recorded
shipping baseline.

Within the new architecture compare uniform and complete-command/person-balanced
sampling with identical optimizer updates, frontend, data eligibility and losses.
For balanced sampling, draw a complete-command class, then a speaker within that
class, then a recording; keep the negative-class treatment explicit and identical
across seeds. Report macro-person and full typed-command results. Isolated numerals
must not enter full-command fitting through misleading generic role metadata.

The first arm is original-human-only, without synthesis, concatenation, splicing
or synthetic audio augmentation. Use the pinned prepared human views only after
validating their actual availability and bindings. QA-eligible supplied synthetic
augmentation is a separately versioned arm; preserve its sampling cap and report
actual training mass. New synthesis or new pretrained speaker weights are not
implicitly authorized. Distinct source IDs are not proof of distinct people.

Later, test exact-complete-command cross-speaker contrastive supervision separately.
Twenty and sixty percent, different targets and different reminder tasks must not
be collapsed just because they share an intent. Existing finite acoustic CTC may
be a separately declared training auxiliary; it must not become an automatic
runtime fallback. Keep the old finite-CTC runtime and evidence unchanged.

## 6. Calibration and evidence requirements

Keep train, development, calibration, historical regression and genuinely fresh
human evaluation roles separate. Source-qualified IDs, PCM hashes, near-duplicate
review and clone ancestry need explicit audits. Preserve ambiguous records and
source exclusions; never relabel failed positive trials as true OOV.

The supplied handoff reports human training/dev/calibration coverage of 25/22/15
of 31 benchmark outputs, respectively. Those metadata observations identify
coverage work; they do not establish label correctness or a capacity diagnosis.
Do not report isolated numerals or synthetic voices as connected-human coverage.
Produce a per-command/value/person acquisition list rather than fabricate labels,
listening or owner pairs. Already examined tests remain regression evidence.

Checkpoint selection uses human development data. For cross-arm screening, use
macro-person accepted exact-command recall with the unmodified numerical gates,
then fewer false accepts, then micro recall; retain earlier epochs on ties. This
is a declared change from v1's raw-recall selection. Fit any additional score
calibration only on the separate calibration partition after checkpoint selection;
write its identity before test and do not refit after test exposure. Report both
pre-calibration and locked calibrated scores. Missing calibration classes prevent
qualification claims for those classes, not honest partial measurements.

For composed slots, bind a documented calibrated score to the typed value rather
than presenting a product/minimum of constituent scores as a proved joint
probability. Report reliability, missing/unknown rejection and all constituent
scores in diagnostics. Retain 0.95 intent and 0.90 for every supplied slot; no
threshold lowering, artificial confidence saturation or test-specific masking.

Retain the proposed v1 pilot quality targets as unachieved targets: 95% human
accepted exact recall, 99% accepted precision, 90% adequately sampled worst-person
recall, at most 1% observed wrong-speaker acceptance, and 95% genuine retention.
These are not official assignment limits. Report OOV rates with denominators and
person/session-aware uncertainty. Promotion also requires the governing project
limits and numerical physical-resource gates frozen before qualification. No
resource gate or missing threshold is to be invented from desktop timing.

## 7. Runtime integration and benchmark ownership

Create a versioned experimental runtime for the new graph. Do not weaken
`pcv_final_runtime.verify_bundle` to accept incompatible heads. Keep the existing
baseline loaded only in an explicit comparator, not as a fallback. Export FP32
and verify PyTorch/ONNX Runtime/ncnn parity before any quantization arm. Test real
variable durations, padding, very short inputs, tail frames and near-threshold
accepted/rejected decisions, not only logit closeness on random tensors.

Compute one declared server-side final application decision. Reuse the existing
[benchmark adapter](../scripts/runtime/pcv_benchmark_adapter.py) and extend its
interface under regression tests if needed. It currently reads `event.output`,
whereas demo acceptance can differ. Do not feed diagnostic candidates to the
benchmark, rescue rejection, insert expected slots or conceal extension misfires.
Capture faults remain separately counted faults, not successful OOV rejections.
Dynamic task-name projection must bind the actual catalog rather than silently
use `DEFAULT_CUSTOM`. Keep richer typed outputs for personal-demo scoring.

Official tester-wake/prerecorded-command trials can pair different people. Keep
three named tracks: official command-only with ownership explicitly disabled;
official owner-enabled with mismatched-pair failures retained; and original
same-person wake/command sessions plus impostor, handover and overlap trials.
Never silently disable the owner gate per trial to improve the official score.

## 8. Separate owner subsystem and deployment gates

Reuse the existing [session policy](../scripts/runtime/speaker_session.py), but
supply real, separately validated identity, overlap and temporal-coverage evidence.
A policy fixture or arbitrary embedding does not constitute that backend. First
implement the interface and failure behavior; train/evaluate a small separate
speaker encoder only with suitable authorized human identity data. Command-content
embeddings must not be used as owner identities. No new pretrained speaker or
separation weights are authorized by the wake/VAD exception.

Keep a bounded pre-wake buffer, isolate the actual accepted wake span, bind an
immutable owner/session ID, and verify all command-bearing PCM consumed by the
VCM. A second wake cannot replace an active owner. Never adapt the reference from
an unverified command. Completion, rejection, timeout or capture fault consumes
the session; reject stale results. Combine VCM, owner and capture decisions
server-side. Missing backend or unknown overlap/coverage fails closed in the
owner-enabled track; command-only evaluation is separately named.

Calibrate using disjoint human reference/command pairs, stratified by both voiced
reference and command duration, noise and interference. The existing 0.8/0.4/0.25
second settings are policy parameters, not established validity for Hey Jarvis,
Stop or Next. Silence padding does not add identity evidence. Similar voices and
different people saying the same command must appear in impostor evaluation.
Reject contaminated commands rather than cutting out and joining fragments.
Target-conditioned extraction is deferred to a separate supervised experiment.

Count every shipped frontend/model/head/catalog/decoder and owner-reference cost.
Measure actual RPi4/5 p50/p95, endpoint wait, utterance duration, RTF, peak RSS,
CPU, thermal/throttling behavior and network-disabled recognition. Catalog-size
scaling and decoder cost matter. Desktop timing and exported bytes are not Pi or
live-microphone evidence. Keep `training.update_artifacts=false`, installation
blocked and provider actions disabled throughout this candidate handoff.

## 9. Source and documentation boundary

This design is grounded in the user-approved architecture conversation, the
provided research handoff, and the pinned repository sources above. Additional
implementation defaults are explicitly identified; no new literature result or
acoustic improvement is asserted. The source branch was read remotely; the
executor's dirty files, ignored corpora, environments and hardware were not read.

The existing remote branch uses additive `register.patch` files for experiment
registration. This documentation commit follows that pattern to avoid overwriting
shared `EXPERIMENTS.md`. The executor must review and apply/merge the supplied
[registration patch](../experiments/vcm-intent-slots-v1/register.patch) before
fitting. No repository cleanup or documentation migration is requested.
