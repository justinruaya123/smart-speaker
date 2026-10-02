# Causal Slot-TCN-96 v2: implementation and qualification handoff

**Status:** one architecture locked; source and numerical prototypes tested. No
new speech-trained model, ONNX export, physical Pi result, source separator,
qualified wake detector, or qualified speaker verifier is supplied.

**Start on:** `codex/pcv-streaming-v2`, descended from
`codex/pcv-strict-final-v1` at `ebc3279f4928111bb2f8d1d1a86506ad270eba5d`.
Preserve the original semifinal branch and installed artifact bundle.

**Codex kickoff sentence:**

> Execute docs/HANDOFF-pcv-streaming-v2.md on codex/pcv-streaming-v2: keep the locked Causal Slot-TCN-96 architecture, train all learned components from scratch using only the approved data, prioritize unseen-human and connected-slot accuracy, build the two specified exports, and qualify streaming, wake-speaker ownership, and Raspberry Pi latency without ASR/STT, pretrained weights, hidden demo overrides, or replacing the current model before every applicable gate passes.

## Charter

Goal: train one small acoustic architecture that accepts the 93 base verbatims
across unfamiliar voices, supports explicit typed slots in the demo, and processes
new microphone frames incrementally. Return exactly one accepted typed command
or a rejection per turn. Keep ownership and recognition evidence distinct.

Priority is accurate human acceptance, not minimal model milliseconds. The user
reports the current ONNX/ncnn latency around 200 ms is acceptable. That is user
observation, not a new measurement here. The old model remains the empirical
accuracy reference; the new architecture is a selected candidate, not an
accuracy winner established by random-weight timing.

Scope: the primary Hugging Face dataset, the existing official benchmark, the
user's named optional HF dataset after audit, the provided source/report snapshot,
and separately tracked local evaluation/enrollment. No outside acoustic corpora,
pretrained initialization, teacher, VAD, wake model, neural TTS invocation, general
phone/word recognizer, ASR/STT output, or cloud recognition.

## Canonical sources

1. User constraints, Option B CSV and October 2 meeting agreement.
2. Supplied repository-only handoff at original `7d8296c`: historical model,
   6,768-clip replay, data partitions, SpeakerUI and missing artifact facts.
3. GitHub parent `ebc3279f`: existing source, earlier 37-test audit and untrained
   noncausal candidates. The new tests verify the reused frontend and earlier
   TCN bytes against their Git blob identities in the verification receipt.
4. Primary HF `airimonda/ai231-me2-voice-commands`, known evaluation revision
   `6947f13073e57eb6ae67e7e2fc3680700b82aa13`; selected historical model trained on
   `25111444af3adff7588d86ab27c304bb080895cf`. Current HF main was NOT verified.
   Container DNS and web HF access failed; no audio was acquired in this chat.
5. Class benchmark `airimonda/vcm-benchmark` at the prior verified
   `ab39857cf73dd067d068fed388c321ab160dc191` pin. Do not silently follow main.
6. Primary research/official documentation consulted: streaming KWS
   (Rybakov et al., Interspeech 2020, arXiv:2005.06720); acoustic query-by-example
   (Settle et al., Interspeech 2017, arXiv:1706.03818; Yuan et al.,
   arXiv:1806.03621); Personal VAD (Ding et al., Odyssey 2020,
   arXiv:1908.04284); VoiceFilter-Lite (Wang et al., Interspeech 2020,
   arXiv:2009.04323); short-utterance speaker verification (Jung et al.,
   arXiv:1810.10884); GE2E (Wan et al., arXiv:1710.10467); SpecAugment
   (Park et al., arXiv:1904.08779); official ONNX Runtime threading/quantization
   documentation and Raspberry Pi 4 specifications. These are method references,
   not permission to import their models/data or transplant reported accuracy.
   Landing pages/abstracts/docs were read; no full-paper reproduction is claimed.

## Findings ledger

| ID | Evidence-backed finding | Consequence |
|---|---|---|
| F1 | Earlier TCN uses symmetric convolutions; future perturbation changes prior outputs. | Previous model-only timing did not prove streaming. |
| F2 | New causal implementation passes prefix isolation, irregular chunk, warm state, reset, masked-padding and independent online/full pooling checks. | Streaming mechanics are demonstrable without learned speech accuracy. |
| F3 | New complete module: 115,825 parameters, 463,300 FP32 parameter bytes, 103,592 explicit state bytes. | Weight/state memory is not a 4 GB bottleneck; process/UI RSS still needs Pi measurement. |
| F4 | Old primary human raw exactness was 498/613; thresholds retained 251/613. | Human discrimination and calibration, not export success, block acceptance. Historical only. |
| F5 | Class benchmark prepends independently recorded wake clips to heldout commands. | A correct strict owner check conflicts with this artificial trial construction. |
| F6 | Template duplication does not add independent evidence; equal embeddings for different IDs are ambiguous. | Max-per-ID scores, runner-up margin, unknown rejection and binding checks are mandatory. |
| F7 | Ownership tests reject intrusive/uncertain windows rather than averaging them away; reference cannot silently adapt to command speech. | Turn owner must be anchored, not inferred from whichever voice becomes loudest. |
| F8 | Current source files do not contain the current checkpoint/audio; ONNX packages are absent here. | New acoustic, ONNX and Pi qualification remain unexecuted. |

## Scope filter

Do not interpret KWS or ASR paper scores as this VCM's accuracy. Borrow causal
caching, contrastive objectives, augmentation and calibration methods only.
No ASR or phone transcription is reintroduced under an alternative name.
Pitch perturbation does not create a Filipino or American accent, and TTS voices
are not independent human speakers. Subject identity is not inferred from gender.

## Object filter

Tensor and fabricated-embedding tests are implementation evidence, not acoustic
recognition, speaker verification or source-separation evidence. Desktop
TorchScript timings are not ONNX, ncnn or Pi results. Existing exposed test audio
is regression evidence, not a newly blind qualification set. The prototype
policy helpers are NOT drop-in replacements for the existing runtime/UI.

## Conclusions and locked design

### 1. One architecture, not another model tournament (F1–F4)

Lock `Causal-Slot-TCN-96-v2` in `scripts/models/pcv_streaming_v2.py`:

- Reuse the existing mono 16 kHz, 40-bin, noncentered 25 ms/10 ms log-mel frontend.
  Fit and freeze normalization using eligible training humans only.
- Causal kernel-5 stem 40 -> 96; six depthwise-separable residual blocks, kernel 5,
  dilations 1/2/4/8/16/32. Frame-local channel LayerNorm, SiLU and training dropout
  0.1. No bidirectional layers, temporal normalization, time downsampling, KV
  attention or whole-window re-encoding in streaming inference.
- Five independently learned online attention summaries: complete command,
  percentage, light entity, song and artist. The online stable max/sum/weighted
  sum recurrence has an independently computed full-sequence reference test.
- Semantic head: REJECT + 31 baseline outcomes + VOLUME_SET = 33 routes.
  Percent head: missing/invalid + values 0..100 = 102 classes.
- Demo heads: 19 coarse intents, five volume operations, three target states,
  finite light/song/artist classifiers, three 64-D entity query embeddings and
  a 64-D acoustic feature sequence. No phone or character output inventory.
- Fixture classifier counts are 5 lights (unknown/all/kitchen/bedroom/bathroom),
  3 songs (unknown/Dynamite/Butter) and 2 artists (unknown/BTS). Actual connected
  catalogs are not available here. Freeze their IDs and class order before
  fitting; do not pretend fixture labels are the user's actual devices.
- Independent 12,480-parameter speaker tower: 40 -> 32 causal kernel-5 stem,
  three width-32 separable blocks at dilation 1/2/4, then 64-D frame embeddings.
  It shares mel inputs, NOT weights, with content recognition. Tests confirm
  speaker loss cannot update the content encoder and vice versa.

The two objectives are different: content embeddings should match the same
command spoken by different people; speaker embeddings should match different
words spoken by the same person. Never use content similarity, pitch alone or
an accent label as proof of ownership.

The content receptive field is 257 feature frames (2.585 s including the 25 ms
window). This is past context, NOT required lookahead or startup waiting time.
Default new-frame cadence is four frames/40 ms. Real 1–3-frame final fragments
use the same stateful ABI. Fourteen state tensors are explicit; command state
resets at the wake/arming boundary. Speaker-reference history is separate.

Do not increase/decrease width or change architecture after inspecting test
errors. A failed qualification is a failed release, not permission to tune on
heldout audio. Dataset/loss corrections use a new version and development only.

### 2. Exactly two delivery artifacts (F2, F5)

**`vcm-benchmark.zip`:** one benchmark ONNX graph from the selected checkpoint,
no SpeakerUI/Node, no dynamic entity catalog, no pretrained wake/VAD graphs.
Run the unchanged 93-phrase official track with its 19 external labels. Absolute
volume 0..100 is an additional explicitly separate exact-JSON track in this same
artifact; it is NOT an extra class silently inserted into the official scorer.
The two volume directions still map to VOLUME_UP/VOLUME_DOWN externally. Do not
map absolute volume to a direction to obtain a better official score.

**`vcm-demo.zip`:** the demo export of the SAME checkpoint, compiled SpeakerUI,
finite connected-light/music catalog, explicit volume values and separate
extension benchmarks. Active runtime loads one graph, not both. The baseline
subheads must have numerical/prediction parity across exports when owner mode is
disabled for the comparison. Each graph/profile/catalog has its own calibrated
policy binding. There is no third combined/full artifact.

Benchmark semantics for a named-light/music extension are REJECT in the strict
semantic head, while the separate demo intent/entity heads can have valid
supervision. This resolves shared-checkpoint profile conflicts. Auxiliary
number-only clips have semantic loss masked. A named-command label cannot be
silently reused as an unqualified generic/all-devices label.

The demo does not silently add arbitrary timers, alarm times, temperatures,
contacts, message dictation or general questions. Baseline values persist for
those families. The newly scoped extras are names for lights/music and complete
percentages; baseline colors remain red/blue/green unless separately authorized,
labeled and qualified. Formatting-valid IDs are not catalog authorization.

### 3. Connected lights first; acoustic enrollment second (F4, F6)

Pre-bake the initial demo light IDs into a supervised role classifier. Train on
connected utterances such as `turn on the kitchen lamp`, `turn off the kitchen
lamp`, and `set kitchen lamp brightness to seventy three percent`, with matched
examples that change only the action, entity, or percentage. An isolated name
or the presence of a JSON alias does not prove slot localization. Require
counterfactual heldout combinations; do not let a percentage identify the room.

Known demo names are the first release path. New user-provided WAV exemplars
are a separately gated enrollment path, not online weight changes:

1. Verify 16 kHz mono finite unclipped PCM, ownership/permission, source metadata,
   duration, complete intended name and associated stable entity ID.
2. Encode reviewed name spans/full contextual examples with the frozen acoustic
   encoder. Build role-specific prototypes from the same representation used by
   the runtime query. Learn name/role separation from labeled connected examples.
3. Store encoder/frontend/catalog/exemplar hashes, not just a filename. Cap a
   role bank at 96 exemplars; compare max-per-ID cosine similarity and a distinct
   ID runner-up margin. A singleton catalog has no rival evidence.
4. Unknown, acoustically indistinguishable or duplicate names must reject or
   request clarification. Never turn an explicitly unresolved light into all.
5. New enrollment invalidates relevant calibration. A high match cannot bypass
   acoustic/intent validity, explicit-target resolution, endpoint, ownership,
   capability or qualification checks.

`ExemplarBank` tests prove these score/policy mechanics only. Query localization,
training, real waveform enrollment and acoustic improvement are still required.
Do not expose its raw cosine similarity as a probability. Do not add a large
prior bonus to make an unrecognized name win.

Compare no enrollment, reviewed human exemplars, and supplied TTS WAV exemplars
on the SAME separate unseen-human set. Human contextual examples are the default
first attempt; whether they beat TTS here is empirical. TTS is not forbidden
as already supplied data, but no pretrained TTS engine is invoked in this phase.
One TTS voice does not teach all accents, and one person's enrollment may create
speaker dependence. Preserve per-accent regressions and unknown-name errors.

Dynamic time warping is a diagnostic alignment alternative, not the first runtime
path. Avoid reinstating thousands of full-command CTC hypotheses merely for a
handful of lights. Optional semantic-tag CTC may be studied as a training-only
alignment loss in a NEW protocol; no general phone/word reader ships.

### 4. Same wake speaker and shared-room testing (F5, F7)

Speaker ownership is ephemeral per turn, not a whitelist of enrolled people.
Every unfamiliar person can become the owner by issuing the supported wake phrase.
An owned turn is not an authentication or anti-replay security system.

The completed helpers take assumed speaker embeddings/overlap evidence. They do
NOT detect overlap, learn speaker embeddings, or isolate a source waveform. The
speaker tower has random weights until trained. Keep its policy disabled until
its own unseen-speaker tests pass.

Production sequence:

- Keep a short microphone ring and identify the wake segment. Reject an
  overlapping/uncertain wake; derive an immutable reference from voiced wake
  frames. Do not average the following command into the reference.
- Analyze a sliding voiced command window with the independent speaker tower.
  Speaker training pairs must include same person/different phrase and different
  person/same phrase, using only allowed fitting speakers.
- Attribute NEW nonoverlapping evidence spans once; overlapping analysis windows
  cannot multiply evidence duration. Missing/too-short evidence is uncertainty,
  not a match. A short `pause` may not yield enough evidence for a strict gate;
  measure that separately rather than inventing confidence or silently bypassing.
- Reject the entire turn on an intruding speaker, uncertain ownership, detected
  overlap, sequence loss or invalid endpoint. Do not delete foreign frames and
  concatenate the rest: that can manufacture a command no one actually said.
- Release the owner after final/reject/timeout/mute. Do not follow the loudest
  subsequent voice. Partial UI output is non-actionable.

A mono mixture is not uniquely invertible from the mixture alone: a+b equals
(a+d)+(b-d). The fixture test demonstrates that identity, not a universal speech
separation impossibility theorem. Reliable preservation of target speech during
arbitrary overlap needs more evidence/modeling than this simple verifier.
Use conservative rejection/retry first. A microphone array with deterministic
spatial processing is an optional separately measured aid, not a guaranteed
solution. Do not quietly install VoiceFilter, ECAPA, Silero or another pretrained
component. Personal VAD/VoiceFilter literature motivates the distinction between
speaker gating and separation; no component from those papers is included.

**Critical official-benchmark exception:** its recorded wake and heldout command
usually belong to different voices. Provide explicit
`owner_mode=official_spliced`, log `owner_enforced=false`, and do not claim that
run measures speaker ownership. Demo defaults to `same_speaker`; ownership tests
use A-wake/A-command, A-wake/B-command, and actual overlap. No filename-based or
benchmark-process-based automatic bypass is allowed.

Several devices sharing the same wake word can all validly hear the same owner.
Speaker verification cannot select which device they intended. Use unique device
wake phrases or explicit touch arming during multi-device tests. Touch arming is
not counted as a passed hands-free wake benchmark.

No supplied wake-positive corpus has been verified. Initial integration may use
clearly labeled touch arming plus deterministic signal activity detection. For
hands-free release, use reviewed permitted wake/enrollment recordings and train
or enroll a wake detector without pretrained weights. Three benchmark wake
recordings do not by themselves establish a general unseen-speaker detector.

### 5. Accuracy, accent, confidence and prefix contracts (F4, F8)

A base verbatim remains the same valid command regardless of accent, perceived
gender, pitch or tone. Acoustic tolerance is NOT semantic tolerance: `fifteen`
versus `fifty`, `on` versus `off`, `to` versus `by`, another light and negation
remain different. Approve lexical paraphrases/aliases before evaluating them;
keep official publisher labels unchanged and discrepancy counts visible.

Use human-first speaker-balanced fitting; preserve the source's train/test/
holdout and supplemental voice reservations. Audit exact PCM, known speaker,
TTS voice lineage, clean/noisy variants and derived parents across partitions.
Only training groups may generate augmented fitting examples. No test voices,
heldout exemplars or repeated test errors select thresholds or checkpoints.

Base schedule, fixed before fitting: seeds 1102/2203/3304; AdamW 8e-4, weight decay
1e-3, cosine to 2e-5, gradient clip 5, batch 32, 96 updates/epoch, up to 80 epochs,
development every four epochs, early stop after eight evaluations without
improvement. Human-only foundation first 24 epochs; continuation can use supplied
training synthetics with a minimum 50% original-human sampling share. Sample
classes then speakers, not random clips dominated by numerals/TTS. Missing human
categories are coverage failures, not synthetic replacements.

Content loss uses complete-command cross entropy weight 1.0; applicable percent
and known-name losses weight 0.5 each; coarse intent/operation/target auxiliary
losses weight 0.2 each. Divide each head by its eligible example count. Add
supervised contrastive content loss 0.1 for same full typed outcome across
speakers, and role contrastive loss 0.1 for same entity across actions/values.
Mask unlabeled roles; do not assign a placeholder entity label. Train the speaker
tower separately with same-person/different-text verification pairs and
GE2E/supervised contrastive loss; no gradient reaches the content tower.

Bounded training-only waveform gain/noise/rate/reverb and mild feature masking
may use allowed data. Keep a clean control. Avoid masking most of a short number
or on/off word. Pitch/rate transforms do not certify an accent. Speaker training
must not use identity-destroying pitch transformations as unquestioned positives.
No model is run to generate new transcripts or repair test labels.

Tune checkpoints on unseen-training-group human macro complete-command recall,
subject to development false-accept constraints. Fit calibration on a separate
speaker-disjoint calibration partition, then freeze once. Do not average near-99%
synthetic performance with weak human performance for selection. Don't train a
final-command label on every incomplete prefix. UI partials can change from 70 to
73; a stable intermediate class must never authorize early execution.

The final acceptance policy is joint: complete command, all required values,
explicitness/identity, owner when applicable, endpoint and calibrated confidence.
A 0.95 softmax or cosine value is not a 95% correctness claim. Temperature scaling
or lowering a threshold cannot repair wrong raw candidates. All-reject policies
must fail recall. Expose raw candidate accuracy, accepted precision, accepted
recall, valid-command misfire rate and unknown FAR separately.

### 6. Streaming, SpeakerUI and latency contract (F2, F3)

Use one resident ONNX Runtime CPU session, eval/inference mode, sequential
execution, one intra-op thread initially, no per-turn model load. Benchmark
1/2 threads on actual Pi before choosing. No x86 optimized graph is shipped as
an ARM-specific artifact. FP32 first; INT8/ncnn only after real waveform/state/
policy parity and measured target benefit. There is no silent engine fallback.

Audio capture only enqueues contiguous PCM into a bounded queue; do not block the
capture callback on inference/JSON. Verify sample indices, missing chunks,
overflows and resampler continuity. No repeated encoding of a growing utterance.
Add queue-age limits and cancel stale turns instead of processing a backlog.

Reuse the resident worker + Node JSONL/WebSocket path. Add non-actionable
`command_partial` events at at most 10 Hz; final events take priority. Carry
turn ID, sequence/sample end, graph/profile/policy identity and separate timing
stages. Drop stale UI revisions; final commands are idempotent once per turn.
Do not request cloud/STT or wait for a completed WAV upload on the live path.

The new helper event format is not the old nested schema-v3 runtime envelope.
Write an explicit versioned adapter. Keep authoritative output separate from
candidate UI data. Map ownership failure to legacy OOV with structured diagnostic
reason, or introduce a reviewed schema version; never silently add enum values
the existing UI validator rejects. Existing demo-confidence overrides MUST NOT
promote a candidate rejected by qualification/ownership.

Endpoint silence is not removed by streaming. Prototype endpoint proposals:
200–300 ms after unambiguous complete commands, and 450–600 ms for extendable
numeric/name prefixes. Determine values only from development with real pauses.
A valid `seventy` can become `seventy three`; final execution cannot know future
speech. Prefer correctness to claiming 200 ms for every ambiguous prefix.

Target gates on Pi 4 4 GB, 64-bit OS, actual microphone and UI: p95 new-chunk
processing <=10 ms for a 40 ms cadence; p99 <40 ms with no accumulating backlog;
ordinary speech-end-to-final-visible-UI p95 <=400 ms; ambiguous numeric/name
turns separately reported and targeted <=700 ms. Inference worker RSS <=256 MiB;
whole assistant including local Chromium/Node <=1 GiB. These are proposed budgets,
NOT measurements. Run a 30-minute thermal/concurrency soak, report throttling,
queue depth/dropouts, p50/p95/p99 and cold start. Recognition-only timing cannot
substitute for speech-end-to-UI or speaker-verifier latency.

## Implementation steps and acceptance criteria

### A. Verify and freeze source/data

Check clean tracked status/branch, preserve untracked files, inspect existing
AGENTS/requirements and ignore rules. Never clean/reset/stash unrelated work.
Run the checked-in focused tests below. Acquire pinned permissible data through
existing audited download code; do NOT call old release training that eagerly
prepares WavLM. Audit drift against all historical partitions before fitting.
Keep raw audio/features/checkpoints in ignored cache, not Git or public bundles.
Optional martinnavs data remains disabled until revision/rights/overlap are audited.

### B. Finish source-only training and assembly

Create `scripts/train/pcv_streaming_fit_v2.py` and a preparation module using the
existing frontend/manifests. These entry points are NOT yet implemented. Produce
frozen maps, speaker-disjoint train/dev/calibration/confirmation IDs, applicable
loss masks, real per-cell coverage, normalization and a source-only receipt.
Use the locked module; do not copy the old endpoint model or pretrained teacher.
Add CPU fixture trainer tests, train on local GPU, retain all seed histories and
negative results. Checkpoint selection uses development; confirmation is once.

### C. Qualify known demo slots and optional enrollment

Implement waveform-to-query enrollment and connected-slot decoding. Test all
known light IDs under on/off/brightness; known songs with and without artists;
unknown and homophonous aliases; unknown explicit lights versus omitted targets;
0/100 and teens/tens; number-containing names; supported role combinations.
Only valid catalog song/artist pairs may emit. Enrollment cannot change baseline
head outputs. Freeze separate known-name and new-name enrollment evidence.

### D. Train and validate activation/ownership

First establish allowed wake-positive data or explicit touch-only limitation.
Train the independent speaker tower from allowed multi-utterance speaker IDs;
verify against unfamiliar people and different text. Implement voiced-window
aggregation, segmentation and separately validated overlap/uncertainty handling.
`OwnerGuard` is only the policy consumer. A random speaker head or a cosine
fixture must never be represented as a working verifier or separator.

Test A wake/A command, A wake/B same command, A wake/B different command, B cutting
into A mid-word, overlapping wake, different people alternating fragments, replay,
same speaker changing pitch/loudness, very short commands and several devices
with the same wake word. Report owner wrong-accept and genuine-owner rejection
separately from semantic accuracy. Measure 0/5/10 dB SIR and negative SIR stress
conditions; unsupported conditions stay visible, not relabeled as negatives.

### E. Export and integrate both profiles

Create `scripts/test/pcv_streaming_onnx_v2.py` and a selected-checkpoint exporter;
these do NOT exist yet. Use opset 17 initially, one dynamic new-frame axis, flat
state inputs/outputs, raw logits only once. Compare full reference, stepped
PyTorch and ORT on random AND original audio, 1/3/4/7-frame chunks, warmed states,
long streams, resets and both profiles. Raw tensor tolerance target 2e-5; require
exact candidate/final-decision parity on the recorded verification corpus.
Quantization tolerance and acceptable decision changes must be frozen separately.
Never package a random graph. Reject missing checkpoint/provenance instead of
initializing silently. Bind the whole bundle and preserve rollback.

Implement `scripts/runtime/pcv_streaming_runtime_v2.py`, add an explicit runtime
kind in the existing worker, and update `SpeakerUI/server/inference/python-runtime.ts`
and client event handling without touching provider actions. Actions stay off
in all benchmarks. Real lights/music service actions require separate scoped
configuration and action tests, not just a recognized command.

### F. Freeze qualification, then publish

Development qualification proposal: accepted human complete-command recall >=95%
overall, >=90% per advertised intent and adequately populated accent cohort;
accepted precision >=99%; unknown FAR <=1%. Also report confidence intervals,
bootstrap by speaker, subgroup sample sizes and worst-stratum results. Do not
claim every subgroup is certified by the same pooled interval.

Perfect-demo gate is a FINITE trial specification: all 93 verbatims accepted
correctly, including every required slot, on first attempts in a predeclared
microphone matrix; zero wrong accepted commands. Prefer at least 12 new speakers,
with at least six Filipino-English and six American-English speakers if those
are the named claimed groups, and voluntary gender/voice-range diversity. Two
first-take repetitions each would be 2,232 trials. This is a collection target,
not evidence those speakers/recordings currently exist. Do not infer accent or
gender automatically; missing groups remain untested. Define microphone,
distance, room/noise, speaking levels and allowed pauses before recording.
Store every attempt, not a chosen best take. Enrollment/test speakers are
separate for the claimed speaker-independent track.

Source-only benchmark remains the primary headline. Additional local recordings
are explicitly named external evaluation/enrollment tracks, not silently added
to fitting. Need for more training speech beyond approved sources is a recorded
data blocker requiring authorization, not permission to import a new corpus.

For a strong <=1% FAR claim, 0/250 negatives is insufficient for a one-sided 95%
upper binomial bound; 0/299 is approximately sufficient under independent trials.
Correlated clips or TTS voices invalidate a naive independence interpretation.
Require enough separate unknown-name, different-owner and noise/near-miss trials;
do not merge these into one flattering negative denominator.

Unqualified categories/owner mode must block publication as a hands-free qualified
demo. Never achieve perfect recall by disabling rejection or achieve zero FAR by
rejecting everything. Repeat the unchanged official pipeline with explicit
spliced-owner setting and the separate same-owner demo pipeline. Publish exactly
the two artifacts only after their applicable recorded-audio, software and Pi
receipts pass. Current model remains installed until then.

## Commands that exist now

From the child branch root, using the configured environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q scripts/test/test_pcv_streaming_v2.py \
  scripts/test/test_pcv_enrollment_owner_v2.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.test.pcv_streaming_preflight_v2 \
  --output .cache/pcv-streaming-v2/new-local-preflight.json
```

The preflight refuses to overwrite its output. TorchScript in that probe is
measurement tooling, not a third deployment engine. Full training, ONNX export,
waveform enrollment, speaker/overlap detection, production UI adaptation and Pi
qualification above remain substantive work. Do not describe them as completed
or merely packaging.
