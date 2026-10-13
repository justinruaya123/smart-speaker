# Codex handoff: unfamiliar-speaker demo and wake-owner validation

## Start here

Continue `speaker-demo-v1` on `codex/speaker-demo-v1`, based on
`codex/semifinal-handoff` at `7d8296c96e19e1e65fc3f0cd1b82d2391df33dbc`.
Read `AGENTS.md`, `REQUIREMENTS.md`, `EXPERIMENTS.md`, this handoff and
`experiments/speaker-demo-v1/protocol.json` before execution. Do not reset, clean,
stash, relocate datasets or stage unrelated work. Preserve all installed artifacts.

The user authorized improvements and commits, followed by mostly script-based
large-scale testing. The deliverable is a **measured improvement in the actual
accepted typed command for unfamiliar human speakers**, plus a separately measured
wake-owner policy. A high softmax score or a passing software test is not success.

Use the existing root interpreter: `.venv/Scripts/python.exe` on Windows or
`.venv/bin/python` on Linux. Below, `python` denotes that interpreter. Use existing
locked dependencies. Do not make a new environment per experiment.

### What is implemented and tested

- `scripts/runtime/speaker_session.py`: one-wake/one-command ownership, immutable
  reference, session binding, timeout, model identity, finite embedding validation,
  speech-duration checks, overlap rejection and optional required continuity
  windows. `gate_application` can reject but cannot rescue a VCM rejection.
- `scripts/models/speaker_demo.py`: a randomly initialized, masked temporal
  depthwise-convolution keyword candidate; class/speaker balancing; cross-speaker
  supervised contrastive loss. UNKNOWN is never a contrastive anchor/positive.
  The 11-class, width-64 configuration has 22,027 parameters. It is not a replica
  of BC-ResNet or the current 2M-parameter full-command model.
- `scripts/train/speaker_demo_train.py`: human-only fitting, original VCM frontend,
  train-only mean/std, feature hashes, matched update budgets, macro-speaker dev
  selection, immutable run directories and an unqualified checkpoint receipt.
- `experiments/speaker-demo-v1/run_grid.py`: nine sequential, matched training runs;
  dry-run by default, explicit `--execute`, logs and fail-fast stage receipts.
- `scripts/test/speaker_demo_suite.py`: strict manifest audit, deterministic trial
  planning, hash-bound replay locks, action-free **existing ncnn runtime** replay,
  expected-label/owner/coverage validation, and denominator-preserving metrics.
- `scripts/test/speaker_demo_stress.py` and `test_speaker_demo_v1.py`: numerical
  stress, optimization smoke and focused regression tests.

Local evidence: 41 focused tests passed; 10,000 numerical ownership cases passed;
12,000 planned fixture trials retained their missing-result denominators; an
80-step numerical optimization smoke decreased its loss. These are **not acoustic
results**. The complete project checkout, root environment, model bundles, corpus,
ncnn, ONNX Runtime and a physical microphone/Pi were not available to that run.
See `experiments/speaker-demo-v1/verification.json` for measured details.

### What is NOT implemented or established

The helper is **not wired into the shipping microphone path**. There is no trained
speaker embedding/overlap backend, calibrated voice threshold, automatic coverage
proof for continuity windows, or verified speaker-exclusive live session. The
keyword candidate is not acoustically trained or exported and is not supported by
the existing three-head PCV `Runtime` without a separate explicit adapter. Do not
put its checkpoint in `models/default` and claim compatibility.

These integration tasks are part of this handoff. Execute the existing scripts
first, then implement only the listed adapters/calibration/capture hooks. No ASR,
transcript parser, LLM, cloud inference or confidence-relaxation fallback is allowed.
Do not silently deploy pretrained speaker weights; the existing wake/VAD exception
does not automatically authorize another pretrained model.

## 1. Establish the checkout and register the study

Inspect branch/status/staged paths. Preserve the installed graph, catalog,
normalization, policy and ncnn graph hashes before changes. The pre-existing
`.gitignore` already covers `.cache/`, audio and weights; no new ignore rules are
needed for this experiment. Do not commit data, embeddings, credentials or weights.

The root experiment directory was not replaced remotely. Apply the supplied
append-only registration patch after checking it, or add the same short entry
manually if concurrent edits changed the context:

```sh
git apply --check --unidiff-zero experiments/speaker-demo-v1/register.patch
git apply --unidiff-zero experiments/speaker-demo-v1/register.patch
python -m pytest -q scripts/test/test_speaker_demo_v1.py
python -m scripts.test.speaker_demo_stress --cases 10000 --model-steps 80 --output .cache/speaker-demo-v1/local-stress-01.json
```

Register the new source modules in `scripts/modules.json` as research-only where
appropriate. Do not alter the shipping allowlist/defaults before qualification.
The registration patch and source-map update are deliberate remaining integration
steps, not changes claimed to have been applied by the source-only run.

On constrained hosts, set OMP/MKL/OpenBLAS thread counts to one before the numerical
smoke. One initial combined test/stress invocation hit a 45-second tool limit;
a separate one-thread rerun completed. Its cause was not isolated as a benchmark.

## 2. Freeze data and the actual demonstration catalog

Reuse the existing acquisition/source receipts and immutable recordings. The
previous release report references `airimonda/ai231-me2-voice-commands` revision
`25111444af3adff7588d86ab27c304bb080895cf`; verify the local receipt rather than
pulling a moving dataset head. Historical reports/test results already viewed are
regression evidence, not a fresh blind holdout. Never recover labels by running ASR.

Create `.cache/speaker-demo-v1/manifest.jsonl` by explicitly mapping actual source
fields. Do not guess which column is a global speaker identifier or a typed command.
Each row must contain:

```json
{
  "id": "source/speaker/recording",
  "audio": "data/audio/source/recording.wav",
  "sha256": "64-character SHA256 of original file bytes",
  "pcm_sha256": "64-character digest from speaker_demo_suite.pcm_digest",
  "speaker_id": "globally-resolved-pseudonymous-speaker-id",
  "source": "pinned-source-name",
  "license": "verified-source-license-or-restricted-research-terms",
  "kind": "human",
  "split": "train",
  "role": "command",
  "command": {"intent": "STOP", "slots": {}}
}
```

This is a schema illustration, not a valid ready-to-run manifest. Compute both
hashes, retaining the suite's exact PCM hashing convention. Audio must already be
mono 16 kHz floating PCM in [-1,1] after decoding. Reuse the project's existing
conversion/preparation utilities and record parent/derived hashes when needed.
Do not silently resample inside an evaluator.

Allowed splits: train, dev, calibration, test, holdout. Allowed roles: command,
wake. Wake rows and true out-of-vocabulary command rows have `command: null`.
Keep natural human and supplied synthetic provenance distinct. Resolve speaker
aliases across sources before splitting; prefixing every dataset's IDs does not
prove that speakers are disjoint. The audit checks declared identities, not voices.
It rejects duplicate recording IDs, file/PCM duplicates and cross-split speakers.
Near-duplicate acoustic matching and independent label listening remain required.

Create `.cache/speaker-demo-v1/catalog.json` as
`{"commands": [<approved complete typed command>, ...]}`. Choose the catalog from
the already approved demonstration definitions **before viewing new test results**.
Do not invent parameter defaults to turn a keyword into a complete slotted command.
The narrow candidate treats other commands as UNKNOWN only in its explicit demo
mode; the broader ten-use-case requirement and full-model benchmark remain intact.

Every demo class and UNKNOWN must have human train and dev examples. The contrastive
arm requires at least two training speakers per positive class. A missing class
must block fitting, not be silently omitted. Fit uses train/dev only; calibration
and test audio may be integrity-audited but never feed checkpoint optimization.
No TTS, splicing, or waveform augmentation is introduced in this first human-only
ablation. PCEN/augmentation follow-ups require a separately frozen experiment.

```sh
python -m scripts.test.speaker_demo_suite audit --manifest .cache/speaker-demo-v1/manifest.jsonl --check-audio --output .cache/speaker-demo-v1/audit-01.json
```

For the fresh human holdout, target at least 20 previously unseen people with every
chosen demo command represented. Ten commands x five repeats x twenty speakers
is 1,000 positive trials. Keep distance, speaking rate, room and microphone metadata
in extra manifest fields. Obtain several clean wake recordings per person.
Missing wake recordings must block the paired track; do not replace them with
command recordings or TTS and call that wake-to-command verification.

## 3. Reproduce the real baseline before replacing anything

Use the corrected `publication-v2-shipping` bundle, not a superseded archive with
the ncnn buffer-lifetime defect. Verify its release hashes and extract it to
`.cache/speaker-demo-v1/shipping/` using the existing validated package workflow.
The following path assumes the verified extraction contains `models/default`.
If the bundle is absent, locate the current source-bound selected checkpoint and
publisher receipts; report missing payloads rather than creating a substitute.

```sh
python -m scripts.test.speaker_demo_suite plan --manifest .cache/speaker-demo-v1/manifest.jsonl --split dev --command-only --output .cache/speaker-demo-v1/dev-commands.jsonl
python -m scripts.test.speaker_demo_suite lock --manifest .cache/speaker-demo-v1/manifest.jsonl --trials .cache/speaker-demo-v1/dev-commands.jsonl --bundle .cache/speaker-demo-v1/shipping/models/default --protocol experiments/speaker-demo-v1/protocol.json --output .cache/speaker-demo-v1/baseline-dev-lock.json
python -m scripts.test.speaker_demo_suite replay --lock .cache/speaker-demo-v1/baseline-dev-lock.json --output .cache/speaker-demo-v1/baseline-dev-01
```

The runner instantiates the existing ncnn runtime once, calls its actual
`application_decision` with 0.95 intent / 0.90 for every supplied slot, and imports
no action gateway. A correct supported command rejected by the policy is a miss.
It records the decision before the optional owner gate, timing, final application
output, and exceptions. Errors return nonzero status; they are never successful
negative rejections. Missing predictions remain in denominators. Partial results
can be scored without pretending the run completed:

```sh
python -m scripts.test.speaker_demo_suite score --trials .cache/speaker-demo-v1/dev-commands.jsonl --predictions .cache/speaker-demo-v1/baseline-dev-01/predictions.jsonl --output .cache/speaker-demo-v1/baseline-dev-rescore.json
```

Locks bind bytes, decoded-audio hashes, runtime sources, bundle files, policy and
backend files. Run from the same root; regenerate a new lock after relocating
inputs or intentionally changing code. A lock is a reproducibility receipt, not
signed protection against a malicious operator rewriting the lock itself.

## 4. Run the nine matched training jobs

The grid consists of uniform CE, class/speaker-balanced CE, and balanced CE plus
cross-speaker contrastive loss; seeds 53, 71 and 89; identical width, epochs,
updates, frontend and optimizer. Inspect the dry-run command plan first:

```sh
python experiments/speaker-demo-v1/run_grid.py --manifest .cache/speaker-demo-v1/manifest.jsonl --catalog .cache/speaker-demo-v1/catalog.json --output .cache/speaker-demo-v1/grid-01 --device cuda
python experiments/speaker-demo-v1/run_grid.py --manifest .cache/speaker-demo-v1/manifest.jsonl --catalog .cache/speaker-demo-v1/catalog.json --output .cache/speaker-demo-v1/grid-01 --device cuda --execute
```

Use `--device cpu` when CUDA is unavailable. On CUDA, set the deterministic cuBLAS
workspace configuration required by the installed PyTorch version before running;
do not silently disable deterministic-algorithm errors. The driver runs sequentially
and stops on failure, leaving logs. An existing run directory is never overwritten.
A partial grid needs a new run ID or explicit separately reviewed resume support.

The trainer saves raw feature hashes before training, train-only normalization,
per-epoch summaries and selected checkpoint hashes. Selection prioritizes dev
macro-speaker positive recall, then fewer negative errors, then micro recall;
ties retain the earlier checkpoint. Log the fraction of anchors with usable
cross-speaker positives. Sparse pairs require a newly declared batch-sampler
ablation, not a hidden change during a run.

Compare the three arms at matched capacity. Compare to the existing full PCV only
on the same declared demo subset and separately retain its broader regression.
Do not attribute any difference from the 2M-parameter PCV solely to contrastive
learning: capacity, objective and output space differ.

## 5. Finish the candidate adapter and export gate

This is a remaining code task, not an implemented CLI option. The current replay
runner intentionally supports the existing PCV ncnn runtime only. Add an explicit
fixed-demo model-family adapter; never activate it as a fallback after PCV rejects.
Keep raw logits, typed-command decisions and calibrated confidence distinct.

Export the selected candidate through the project's established PyTorch/pnnx/ncnn
workflow. The classifier has one intent/complete-command output, not the PCV's
CTC/intent/target head layout. Do not alter the frozen PCV verifier to accept the
wrong graph. For single-utterance inference, make padding/mask handling explicit;
compare valid-length batches to unpadded inference before export.

Require real-audio and generated-feature parity across short/long/odd frame lengths,
repeat-order stability, no native buffer lifetime defect, and full typed-decision
agreement. Test before and after quantization separately. Measure the complete
frontend + model + decision path. Do not label desktop latency as Pi latency.
Use fresh human calibration for confidence after selecting weights. Preserve the
0.95 / 0.90 application contract; a fixed catalog's singleton slot probability is
not proof that a value was acoustically recognized. Either provide defensible
slot evidence or limit this explicit pilot to valid no-slot keywords. Record that
scope; never add a fabricated slot to pass schema validation.

## 6. Implement and calibrate the separate speaker backend

Do not reuse the keyword embeddings: their training objective removes speaker
variation. Use a separately trained text-independent speaker encoder. Start with
a small TDNN/CNN with mean/std statistics pooling and a normalized embedding;
train from random weights on the eligible human speaker pool. Positive training
pairs must include different phrases, not just repetitions of one wake phrase.
Count its parameters, frontend and memory separately. A larger pretrained encoder
may be an explicitly labelled research control, not a silently shipped dependency.

Implement `module:factory` with this contract:

```python
def factory(binding):
    # binding["files"]: absolute model/config paths -> verified SHA256
    # binding["model_id"]: digest of that complete binding
    return backend

# Both methods consume only PCM and sample rate, never expected labels or IDs.
backend.extract(pcm, 16000)  # -> SpeakerEvidence
backend.windows(pcm, 16000)  # -> sequence[SpeakerEvidence] or None
```

The embedding identity must bind weights, normalization and frontend, not just a
model nickname. `extract` supplies actual voiced duration, clipping and overlap
evidence. Unknown overlap must stay `None`, which rejects. `windows` must cover
all command-bearing speech using verified segmentation. Returning the whole
utterance repeatedly does not establish continuity. The policy helper checks
supplied windows; it does not discover handovers or prove their coverage.

A missing overlap/continuity backend is an explicit blocker for the strict track.
An exploratory whole-utterance comparison may use a separately labelled ablation,
not pretend that it enforces single-speaker speech. Very short words may contain
insufficient identity evidence; retain those as false rejects of supported commands.
Do not turn silence-padding into voiced duration or learn the owner from an
unverified command. Do not update the reference during a session.

Calibrate on held-out calibration speakers using genuine and impostor wake/command
pairs, varied distances, duration buckets and same-command/different-speaker hard
negatives. Select the cosine threshold with predeclared wrong-speaker acceptance
and genuine-retention targets. It is not the 0.95 intent probability threshold.
Write `.cache/speaker-demo-v1/speaker-policy.json` containing only constructor
fields of `SpeakerPolicy`; the null threshold in the protocol is deliberately
not runnable. Save candidate scores, selected threshold, counts and binding hashes
before any final test. Do not calibrate on the test failure list.

Generate the large paired plan after the manifest is frozen:

```sh
python -m scripts.test.speaker_demo_suite plan --manifest .cache/speaker-demo-v1/manifest.jsonl --split calibration --impostors 10 --output .cache/speaker-demo-v1/calibration-pairs.jsonl
python -m scripts.test.speaker_demo_suite plan --manifest .cache/speaker-demo-v1/manifest.jsonl --split test --impostors 10 --output .cache/speaker-demo-v1/test-pairs.jsonl
```

After backend implementation, use `lock` with `--speaker-backend module:factory`,
`--speaker-policy` and **all** `--backend-files` (encoder, frontend/normalization,
overlap/continuity artifacts). These are real integration arguments; replace the
module name with the implemented module, not an assumed installed package.
Then run `replay --lock ... --output ...` as above. The speaker helper is deliberately
not enabled merely because a backend factory can be imported.

With 1,000 command recordings, one command-only and one genuine-owner trial plus
ten distinct impostor-owner speakers per command gives up to 12,000 planned trials.
This does not mean 12,000 independent people/recordings. Always report unique
speakers, unique audio, pair reuse, human/synthetic strata, per-speaker results,
per-command results, and uncertainty respecting speaker clusters. The supplied
scorer intentionally makes no independent-binomial confidence claim.

## 7. Wire the policy into actual capture only after backend validation

Relevant existing files: `experiments/pcv-synthetic-v2/wake_gate.py`,
`scripts/runtime/pcv_final_runtime.py`, `scripts/runtime/speaker_worker.py`, and
`scripts/runtime/speaker_confidence.py`.

Add a bounded rolling PCM buffer before wake detection. Extract the actual wake
span without including earlier bystanders or later command speech. Arm one owner
at the accepted wake event. Preserve the existing release/command-ready protocol
unless a separately tested segmentation change is introduced. Include wake and
command capture failures in the full-session denominator.

Check ownership on the same command PCM passed to recognition. Carry an explicit
wake-session ID and consume it for exactly one completed/rejected command. A second
wake must not overwrite an active owner. Reset after timeout, malformed audio,
stream reset or completion. Enforce the conjunction server-side before any action:
VCM acceptance AND speaker verification AND audio integrity. Preserve diagnostics
without allowing demo mode to reinterpret a speaker rejection as acceptance.
Keep owner reference embeddings in RAM, clear them at reset, and omit them from logs.

Tests must include A-wake/A-command, unseen B starting a new valid session,
A-wake/B-command, A-start/B-finish, simultaneous speech, noisy/short/clipped wake,
owner silence followed by B, timeout/rearm, capture overflow, replay ordering,
repeated command dispatch attempts and stale-session reuse. For initial deployment,
reject overlap rather than adding a source separator. Voice matching is not
absolute identity/authentication, especially under replayed audio.

## 8. Locked test, continuous stress, Pi and release decision

Freeze winner, frontend/catalog, backend, thresholds and source hashes before
running final test/holdout. Preserve the complete supported-command denominator.
Use the supplied scorer's missing/error counts; all-reject is never qualification.
Evaluate naturally recorded conversations separately from deliberately constructed
mixtures or handovers. If adding constructed scenes, retain constituent hashes,
SNR/SIR, offsets, clipping checks, seed and explicit constructed provenance. Verify
that both speakers actually speak in each labelled interference interval.

Measure on the physical Pi: full endpoint-to-decision p50/p95, model processing,
peak RSS, load time and session overhead. Record the Pi model, OS, CPU threads and
temperature/throttling; do not copy the workstation's numbers. Run action-free
replay first. Physical light/music actions remain outside this test authorization.
A human microphone campaign is an unavoidable separate data-collection step,
not something synthetic script execution can replace.

Report proposed targets separately from measurements: 95% correct accepted human
recall, 99% accepted-command precision, 90% worst-speaker recall where adequately
sampled, at most 1% observed wrong-speaker acceptance, and at least 95% retention
of otherwise correct genuine commands. Give denominators and uncertainty; no
confidence or safety guarantee follows from an observed zero-error sample.
The existing full-model release requirements remain in force.

Final outputs: update `docs/EXPERIMENT-speaker-demo-v1.md` and the root experiment
entry; commit source/tests/protocols and compact selection/verification receipts;
leave raw predictions, audio, features and candidate weights in `.cache/`.
Provide exact commands, per-speaker/command tables, rejected candidates, error
reasons, missing coverage, graph hashes and measured Pi results. Install nothing
when quality, export or backend evidence is missing or a gate fails.

## Research basis, not project results

The cross-speaker objective is an adaptation of supervised contrastive learning,
not a reproduced speech benchmark: https://arxiv.org/abs/2004.11362.
Speaker-conditioned frame decisions are motivated by Personal VAD:
https://research.google/pubs/personal-vad-speaker-conditioned-voice-activity-detection/.
These references do not establish this project's speaker accuracy or Pi performance.
