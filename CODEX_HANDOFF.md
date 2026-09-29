# Codex handoff: finalize the VCM and publish one ONNX Runtime package

## 1. Assignment and release status

Work in `justinruaya123/smart-speaker`, on the existing branch
`research/vcm-final-suite-visualizer`.

**Invocation assumption:** the user will run this handoff only after freshly
pulling this branch into a clean checkout. Treat the checked-out repository as
the complete starting state. Do not search parent directories, ChatGPT sandboxes,
old local project folders, download caches, or other branches for missing
artifacts. Do not require untracked files. If an input is not present in the
fresh branch and cannot be reproduced by checked-in code plus version-pinned
dataset pulls, mark it unavailable and rebuild the experiment reproducibly.

Do not reset to an older hash, force-push, switch to `main`, or modify `main`.
Historical commit hashes in this document are evidence references, not checkout
targets.

The user requests a clean semi-final VCM: a small, genuinely streaming acoustic
model, reproducible dataset/training/validation scripts, typed parameters and
catalog IDs, complete success/failure CSVs, a terminal inspector, and a portable
ONNX Runtime package for Raspberry Pi 4. They explicitly delegate finalization
and removal of unnecessary historical working files to Codex.

**This handoff is not a final model release.** The interrupted semi-final run's
workspace, checkpoint, calibrated policy, and confirmation outputs were not
recovered. Do not reconstruct its results from conversational claims. There is
no verified new candidate to promote or commit. The previous project-trained
`phone_s53` is a recovery reference, not a qualified winner.

Keep the independent character-transcription component removed. Do not introduce
Whisper, general sentence transcription, an inference-time LLM, or externally
pretrained acoustic weights. Existing project-trained weights may be continued
with their provenance intact. External pretrained systems are benchmark-only,
never initialization, automatic fallbacks, or runtime dependencies. A licensed
pronunciation dictionary is data, not a pretrained neural model. The retained
research path recognizes constrained phonemes; describe this honestly.

No Home Assistant service call, Spotify playback, GPIO action, call, or message
is authorized. Use fictional catalogs and inert action previews. Do not claim
that a recognizable intent implements an external service.

## 2. What is actually recoverable

Read `docs/finalization/RECOVERY_MANIFEST.json` first. It records byte counts,
SHA-256 identities, the verified source tree, and absent semi-final artifacts.
**Repository-only rule:** `VCM_FINAL_PAYLOAD.zip` and
`VCM_NO_ASR_FOLLOWUP.zip` are not in this project and are not required inputs.
Do not ask Codex to find them, do not fail because they are absent, and do not
depend on any ChatGPT sandbox path. The checked-out branch is the only guaranteed
starting point. The manifest keeps historical hashes only so a separately supplied
archive can be identified later; such an archive is optional recovery evidence,
not part of the finalization procedure.

If a historical graph/checkpoint/raw recording mentioned below is not tracked in
Git, treat it as unavailable. Reconstruct it only from checked-in source,
version-pinned dataset pull scripts, and recorded configuration/provenance. If
that cannot be done reproducibly, record the artifact as missing and train/evaluate
a new candidate under the frozen protocol. Never download similarly named model
weights or infer old accuracy from reports as a substitute.

| Existing path | Meaning / required treatment |
|---|---|
| `Architecture.ipynb`, `train.py`, `publish.py` | Executable notebook-based v12 finite-head pipeline. It is not the smaller streaming research model. |
| `pipeline.toml` | Still defaults to historical root validation/publication behavior; publication is not yet a qualified-release gate. |
| `artifacts/`, `validation.json` | Root v12 metadata and historical results. Trained v12 weights and its full recordings are local prerequisites, not in the Git source snapshot. |
| `research/no_asr_review/` | Removal verification, Option B audit, and failed transfer results. |
| `research/no_asr_followup/` | First-layer adaptation, paired command-selection guard, and failed-candidate report. |
| `final_suite/` | Evidence consolidation, typed exactness, scope checks and source tests. Payload-only historical checks are optional legacy evidence and should be migrated or retired rather than made release prerequisites. |
| `visualizer/` | Development-only inspector. Preserve its source and recipe compatibility; never ship it in the Pi inference ZIP. |
| Historical `experiments/vcm_streaming_v3/` paths | Reports describe the own-trained `phone_s53` lineage. These files are not guaranteed to be tracked. If absent, regenerate a new candidate; do not make them a bootstrap dependency. |
| Historical `experiments/vcm_services_v6/` paths | Reports describe fictional Home Assistant/music catalogs and compiler logic. Recreate only the canonical catalog fixtures needed by the final package from checked-in specifications/tests. |
| Historical `experiments/vcm_acceptance_v4/` and `vcm_acceptance_v5/` paths | Historical gate/failure evidence only. Never require these directories or reuse stale thresholds in the final runtime. |

Historical results to preserve, not newly measured here: `phone_s53` matched
645/706 valid V6 synthetic commands, but 0/180 external brightness test commands
in the last replay. Its six ambiguous and 42 negative V6 cases are separate.
Acceptance was disabled. The newer first-layer trial lost existing development
successes and was rejected. Do not cherry-pick the externally improved earlier
candidate while hiding its V6 regression.

The root v12 model has 398,422 parameters; the research CNN has 113,064. They are
alternatives, not an ensemble. The old root publisher still packages v12 unless
you explicitly migrate and verify it. Renaming a file does not migrate a model.

## 3. First execution: audit the freshly pulled branch

Assume the user has already fetched and freshly pulled this branch. Do not
perform recovery searches outside the checkout. Confirm the handoff really
started from the expected clean branch before changing anything:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -8 --oneline
git ls-files
```

Expected branch: `research/vcm-final-suite-visualizer`. Expected worktree:
clean. If either condition is false, stop and report the mismatch; do not stash,
discard, merge, reset, switch branches, or search another checkout on the user's
behalf. All subsequent work must be reproducible from this branch plus the
version-pinned public/manual dataset acquisition procedures that you create or
retain in the repository.

Do **not** run an archive restore as a prerequisite. Start from tracked source:

```bash
python -m pytest -q
python final_suite/run_all.py --qualify
```

Run optional browser/full-evidence checks only when their declared tracked inputs
exist. A missing historical payload is an expected repository state, not a reason
to search for `VCM_FINAL_PAYLOAD.zip` or `VCM_NO_ASR_FOLLOWUP.zip`.

Next inventory what the branch can actually regenerate:

```bash
git ls-files
find . -maxdepth 4 -type f \( -name '*.onnx' -o -name '*.pt' -o -name '*.jsonl' -o -name '*.csv' \) -print
```

For every absent model or dataset artifact, trace the checked-in producer and
configuration. Convert that producer into the canonical `scripts/`, `configs/`,
`train.py`, and `validate.py` flow described below. If the old `phone_s53`
binary cannot be recreated from tracked inputs, keep its reported numbers as
historical comparison only and train the semi-final candidate anew. Promotion
must use fresh waveform replay of artifacts produced by the repository itself.

The existing qualification check is expected to fail while human/Pi/calibration
evidence is missing. Record skipped tests and missing dependencies distinctly
from passes. Saved predictions can establish regression history, but cannot be
represented as a fresh waveform replay.

Do not search for the interrupted semi-final workspace outside this repository.
It is not a required input. Treat it as `not_recovered` and restart the bounded
experiment from tracked source and reproducible data pulls. If the user later
supplies a separately hashed artifact, evaluate it as an additional candidate
under the same frozen protocol; never make it a hidden dependency.

## 4. Minimal target layout and migration

The following is the desired final layout, not a claim that these interfaces
already exist. Prefer moving proven functions over rewriting working code.

```text
README.md
Architecture.ipynb          # one readable report importing the same vcm package
MODEL_CARD.md
DATASET.md
RESEARCH_SUMMARY.md         # small retained successes, failures, and archive IDs
recipe.json                # authoritative frontend/model/decoder/decision contract
train.py                   # explicit training, own initialization only
validate.py                # frozen-policy waveform replay and exhaustive CSVs
benchmark.py               # real runtime timings and paired model comparisons
publish.py                 # portable single-graph runtime bundle, strict allowlist
vcm/                       # shared source for scripts, notebook, and Pi runtime
  frontend.py
  model.py                 # training-only torch code, excluded from Pi package
  decoder.py
  catalog.py
  acceptance.py
  runtime.py
  tui.py
scripts/                   # dataset pull, audit, split, export, optional quantize
configs/                   # pinned data/model/selection/benchmark protocols
catalogs/                  # fictional lights, tracks/artists, pronunciations
models/                    # one selected deployment graph plus bound metadata
reports/                   # selected release evidence and complete result CSVs
tests/                     # portable unit, regression, integration, package tests
visualizer/                # optional development tool, excluded from deployment
docs/finalization/         # recovery and cleanup evidence, excluded from deployment
```

First extract notebook implementation cells into an importable package with
behavioral parity tests. Change root scripts and the report notebook to import
that package. Only then retire the old dynamic notebook module loader. Preserve
one current notebook with architecture Mermaid, recipe, datasets, payloads,
coverage, all successes/failures, and reproduction commands. Archive the redundant
review notebook after that content has moved; do not leave two competing sources
of truth. Prevent `train.py` from overwriting the curated report with stale metrics.

Implement ordinary CLI entry points with `--help`; the same implementation must
serve the notebook, WAV replay, terminal UI, tests, and deployed application.
Training is opt-in. Missing artifacts must fail with actionable errors, not fall
back to random weights. A test fixture must never enter a deployment directory.

## 5. Candidate and ONNX contract

Use the historical causal CNN design as the first bounded candidate, not a larger architecture search. If its exact project-trained binary is absent, reproduce the architecture and train a new candidate from project/audited data rather than blocking on recovery:

- 16 kHz mono; 400-sample periodic Hann window; 160-sample hop; FFT 512.
- 40 HTK mel bins, 20-7,600 Hz; log floor 1e-8; no centered windows or future
  normalization. Reuse the recovered training mean/std exactly for its weights.
- Stem Conv1d 40->64, kernel 3; five Conv1d 64->64 blocks, kernel 5, dilations
  1/2/4/8/16; stride one, biases, ReLU; 64->40 linear 1x1 phoneme projection.
- 39 phones and blank index zero; complete-command constrained CTC decoding.
- The recovered 40 ms graph has `features [1,40,4]` plus six cache inputs
  `state_0..state_5`; outputs are `ctc` and `next_state_0..next_state_5`.
  Cache shapes are `[1,40,2]`, `[1,64,4]`, `[1,64,8]`, `[1,64,16]`,
  `[1,64,32]`, `[1,64,64]`. Verify output semantics against source; do not apply
  normalization or log-softmax twice.
- Default inspection cadence is four new frames; compare eight frames on the Pi.
  Ship one chosen graph per deployment package, not both loaded simultaneously.

The reference has 113,064 parameters, a 458,230-byte FP32 ONNX graph and 32,064
bytes of neural caches. Those are not process RSS or end-to-end latency. Changing
phone inventory, input normalization, color supervision, or model dimensions
requires a new graph, metadata, regression replay, and acceptance calibration.

Retain standard ONNX operators and one reusable `CPUExecutionProvider` session.
Begin target measurements with sequential execution, one intra-op thread and
spinning disabled; compare 1/2/4 threads rather than assuming all cores are faster.
Do not ship an x86-optimized serialized graph as a portable ARM artifact.
FP32 is the reference. Static QDQ INT8 is optional only after representative
nonzero-cache calibration and Pi benchmarking; keep it only if accuracy and
acceptance survive and measured target cost improves. Do not assume FP16 is fast
on Pi CPU. Current ONNX Runtime references are listed in section 15.

## 6. Commands, parameters, colors, and catalog identity

Publish a versioned intent schema and explicitly map both the ten legacy labels
and the ten application use cases. Required application rows are music,
questions/search, lights on/off, brightness/color, timers, alarms, thermostat,
media/volume, reminders/lists, and calls/messages. Mark missing capabilities as
unsupported, count their valid requests as misses in the full-scope benchmark,
and do not relabel them as negatives to raise recall. A fixed timer example is
not proof of all durations. Ask for the original rubric only where it changes
coverage; do not invent open-ended text capability under a finite catalog.

Numbers are typed, complete values: integer 0..100 for absolute brightness/volume;
signed percentage-point deltas for declared relative forms. Do not clamp invalid
input or match a convenient substring. Preserve `to` versus `by`, name tokens
such as `lamp two`, and numeric song titles. Test teens/tens, appended numbers,
decimals, negatives, missing units/values, multiple commands, and negation.
A provisional 70 can become 73; no action or final acceptance occurs before the
endpoint. Do not confuse symbolic parser tests with waveform recognition tests.

Colors use a reviewed pronunciation/alias list and labeled speech for this
project's model, not a separate pretrained color recognizer. Explicitly test
unseen speakers, all advertised colors, ambiguous aliases, and devices that
cannot change color. Listing a color in JSON does not establish acoustic support.

Home Assistant-style vocabulary joins stable entity IDs, exposed names, aliases,
areas, and capability metadata. Exclude unexposed/disabled entities. Keep exposed
unavailable names recognizable but block their action previews. Resolve duplicate
aliases only with explicit area context; otherwise clarify. Separate mutable
state from lexical graph identity. Never execute a real Home Assistant request.

For fictional music use `<title> [by <artist>]`, optionally preceded by declared
play verbs. BY is a reserved whole-word delimiter, not a new homophone subsystem.
Compile only title/artist pairs present in the catalog. Preserve duplicate-title
ambiguity and actual multi-artist relations. Retain explicit spoken aliases for
titles containing the reserved delimiter. Real Spotify data/voice-control terms
need a separate current policy review before any live integration.

Prefer the existing type-filtered pronunciation/CTC scorer for identity matching.
Levenshtein or weighted phone edit distance may be a small diagnostic/reranking
baseline, not an acceptance probability or a mechanism for forcing unknown names
onto the nearest entity. Learn any confusion costs using training data only.
Do not add semantic embeddings unless a matched experiment shows a necessary
benefit. This task matches existing IDs; it is not semantic recommendation.

Bind model, vocabulary, frontend, grammar, catalog, and policy identities together.
A changed ID/alias/pronunciation invalidates the corresponding acceptance policy.
Check raw dynamic-list matching separately from calibrated automatic acceptance.

## 7. Dataset pulling, provenance, and cross-validation

Keep dataset acquisition separate from runtime. Pull only version-pinned sources
with manifest hashes, sizes, licenses, source IDs and restartable downloads.
Validate archive paths and checksums; never execute downloaded dataset scripts.
Keep a manual-local-source path where redistribution/permissions are unclear.
No raw third-party recordings belong in Git or the Pi ZIP by default.

The audited external reference moved from `MEX2/OptionB` at
`ed245708c2573188eb769096aba76588f79b8de1` to `MEX2/Data` at
`79d7d524aee9b46a110ca87a36ddcc6bcbd37599`. Preserve those pinned revisions.
The audited committed corpus has 18,375 WAVs, 100 reference IDs and 96 texts;
the README's larger local inventory is not the audited checkout. Brightness is
only 20/60/100; numeric volume, dynamic names/BY pairs and background labels are
missing. No published model trainer was found in that inspected subtree.
The reference-speaker IDs do not establish independent human-command provenance.
Do not claim an unpublished training recipe or human quality from its labels.

Use existing `data/sources/` ledgers as candidates, not automatic inclusion:
FSC for human finite-command controls; real SLURP for assistant commands; optional
LibriSpeech for locally trained phonetic breadth; Speech Commands for keyword
controls. Audit their available content, rights, and actual target-label mapping.
Synthetic material is useful for coverage tests but cannot replace the independent
human benchmark. Keep the known failing external transfer domain visible.

Every example needs: source/revision, recording ID and raw/normalized PCM hashes,
speaker/session, text or annotation provenance, structured reference, sample
rate/channels, synthetic flag, augmentation parent, partition, and catalog version.
Missing annotations are not UNKNOWN or NONE. Audit CTC target feasibility,
truncation, clipping, durations, exact/near duplicates, label errors, and leakage.
Do not label 12 kHz samples as 16 kHz. Resample full WAVs or use a stateful streaming
resampler; independent callback resampling can create discontinuities.

Use train, development, calibration and sealed-test partitions grouped by
speaker/session/source/augmentation parent. Keep paired close/far recordings and
clean/noisy versions together. Add disjoint complete-name and value-composition
probes; distinguish these from all-value coverage. All previously inspected tests
are regression sets. Choose a small, recorded speaker-group CV protocol on the
training pool only; fit normalization in each training fold. Never tune on a CV
test fold and then call it an independent final test. Do not average synthetic
voice variants as independent people. Report speaker-clustered uncertainty where
data support it, and disclose small denominators.

## 8. Bounded training and model selection

Close the previous failed trials with reasons in `RESEARCH_SUMMARY.md`; do not
repeat them indefinitely. Start with one audited mixed-data recipe. Use the historical project checkpoint as epoch zero only if its exact tracked/reproducible artifact is available; otherwise use genuinely random initialization or another project-produced checkpoint with complete provenance. Record seeds, data/graph hashes, objective,
optimizer, schedule, batch composition, augmentation and checkpoint selection.
Use project-only continuation or genuinely random initialization; never an
external checkpoint. Preserve phonetic evidence for short number/negation words.

Select on development complete-command results and critical per-family behavior,
not phoneme error alone. Reuse the paired identity/type-aware regression guard
from `research/no_asr_followup/selection.py` after migrating its tests. The current
conservative rule rejects loss of an already correct replay-development command;
any revised tolerance must be justified and frozen before the experiment.
Include raw errors, wrong accepts, valid rejection and per-speaker results.

Run at most one primary candidate and one justified training ablation, with two
seeds each, before reviewing outcomes. Keep all candidates' histories, including
failures. Fix the protocol before using test results. If no candidate meets the
criteria, retain the reference, record the blocker, and produce only a diagnostic
semi-final bundle. Do not call a rejected run the best model to satisfy a deadline.

Fit compact acceptance only on designated calibration data and freeze it before
confirmation. Optimize correct accepted recall subject to precision/negative
constraints; never lower thresholds on inspected test failures. Do not accept
partial numbers, hidden extra words, negation, unsupported scope, or outside-list
names just because they are close to a legal candidate. Evaluate new gates with
the deployed endpoint and decoder, not reference-transcript-assisted spans.

## 9. Required release tests and explicit metrics

Keep fast software checks separate from acoustic and physical-hardware gates.
Required suites:

1. Typed JSON exactness, all integer levels, relative semantics, missing/extra
   slots, bool-versus-int, nonfinite values, catalog ties and capability checks.
2. Arbitrary chunk boundaries, 40/80 ms state carry, causality, frontend parity,
   all cache resets, valid-frame counts for the final short block, and long input.
3. Capture dropout/overflow/NaN/device failure, sticky fault until reset, no
   duplicated samples, no stale prior-prefix finalization, supported sample rates.
4. CTC blanks/repeated phones, underflow, native/reference decoder agreement,
   out-of-list and wrong title/artist combinations, negation and appended numbers.
5. Real waveform replay of every available regression, new human confirmation,
   robustness/noise/far-field, critical number pairs, and all advertised intents.
6. ONNX checker and real ONNX Runtime CPU execution with nonzero caches; compare
   PyTorch/reference outputs, decoded commands, ties and final decisions. Freeze
   tolerances before testing. Do not merely pass graph loading or a random tensor.
7. Clean-install publication smoke tests with network disabled during inference;
   no PyTorch/ASR/model download or research-folder dependency; stale-file poisoning
   test; every documented command tested from the extracted ZIP.
8. Physical Pi4 and microphone: cold/warm timings, frontend/encoder/decoder/gate/
   endpoint separately, sustained runs, RSS, CPU, temperature/throttling, overruns,
   response after speech ends, and bounded queues at the chosen catalog size.

Use the existing proposed targets explicitly as gates, not accomplished facts:
clean complete accuracy >=95%; realistic 10 dB >=90%; correct accepted recall
>=90% overall and per advertised family; observed accepted precision >=99%;
negative false accepts <=1%; weights <1 MiB; peak RSS <256 MiB; Pi p95 processing
<=100 ms, real-time factor <1, and response p95 <=1 s after speech ends. Freeze
any justified changes before confirmation. Report confidence bounds and sample
counts; passing a point estimate is not proof of population reliability. Missing
human, hardware, calibration or required-intent evidence prevents qualification.

Generate from actual replay records, not handwritten success examples:

- `reports/all_commands.csv`: every evaluated input and decision.
- `reports/accepted_commands.csv`: final accepted AND exactly correct commands.
- `reports/wrong_acceptances.csv`: accepted but incorrect, including negatives.
- `reports/failed_commands.csv`: valid requests not correctly accepted, including
  correctly ranked but rejected requests and unsupported required use cases.
- `reports/negative_results.csv`: all negative/ambiguous input outcomes.

Columns: run/protocol/model/frontend/decoder/catalog/policy hashes, partition,
source, recording and PCM IDs, speaker/session, synthetic flag, verbatim,
expected JSON, raw candidate JSON, final output JSON, accepted boolean,
exact-correct boolean, reason, latency fields and evidence type. Preserve quoted
newlines and Unicode safely. An empty accepted CSV must remain empty with a header.
Do not mix diagnostic 'would accept' results with deployment acceptance.
Also emit machine-readable summaries and reconciliation tests: counts must agree
with the report, and every evaluated recording must be accounted for exactly once
within its run. Repeated evaluations across studies are not new recordings.

## 10. TUI and runtime behavior

Offer WAV replay and optional microphone input through the same runtime. A small
terminal UI can show audio position, energy/VAD, current candidate intent, number,
entity/track IDs, margin, queue depth, computation time and final rejection reason.
Labels must say `tentative` while speech is incomplete. Keep microphone capture
lightweight; decoding/rendering belongs outside the real-time capture callback.
Use a bounded queue and explicit sample sequence numbers; overflow invalidates
that utterance, never silently discards evidence. Reset frontend/resampler/model/
decoder/endpoint together between utterances and on model/catalog replacement.

Use push-to-talk initially. If an endpoint is enabled, validate its preroll and
silence behavior with pauses inside numbers, negation and names. Do not replace
missing wake-word functionality with VAD. No always-on false-activation claim
without a continuous-audio benchmark. Replayed audio can be slowed for inspection;
live capture stays at 1x. Step/debug visualization does not authorize an action.

## 11. Publication contract: what ONNX Runtime must be able to run

An ONNX graph alone is insufficient: publish its exact frontend, vocabulary,
cache/reset contract, catalog decoder, policy and hashes together. Emit a recipe
with graph input/output names and shapes, dtypes, opset/IR, required ORT version,
selected execution provider, normalization/grammar identities, and qualification
state. Reject mismatches before opening the microphone.

Required final command interfaces, to implement and verify before documenting
as available:

```text
python train.py --config configs/train.json --output runs/<run-id>
python validate.py --bundle models/selected --manifest <frozen.jsonl> --output reports/<run-id>
python benchmark.py --bundle models/selected --manifest <benchmark.jsonl> --output reports/pi4
python publish.py --bundle models/selected --report reports/<run-id>/release.json --output dist/vcm-rpi4.zip
python publish.py --bundle models/selected --diagnostic --output dist/vcm-diagnostic.zip
python -m vcm.tui --bundle models/selected --wav <file.wav>
python -m vcm.tui --bundle models/selected --mic <device-index>
```

Publication without `--diagnostic` must fail unless required evidence is present
and passes for exactly the shipped identities. Diagnostic publication must embed
`qualified=false` and disabled acceptance/action execution in unambiguous metadata.
Do not let a config switch or stale passing JSON bypass qualification. A software
smoke pass must not set an acoustic release flag.

Use a deterministic allowlist for runtime source, one selected graph, frontend
statistics, phone/intent schema, reviewed catalog/policy, small licensed samples,
third-party notices, requirements and optional hash-pinned ARM64 wheels. Exclude
ASR, pretrained checkpoints, torch/training code, notebooks, visualizer, all raw
corpora, caches, rejected weights, test captures and credentials. A standard C
helper may be packaged only as verified ARM64 code with its source/build recipe;
never copy a workstation binary and label it Pi-ready.

Test installation on a supported 64-bit Pi OS/Python combination. Separate OS
package prerequisites from optional offline Python wheel installation. Require
all wheel hashes and correct architecture tags; test fresh installation. Never
commit tokens, live home entity exports or personal music history. Publish model
assets via an explicit versioned artifact/release with durable hashes when not in
Git; the final fresh checkout must not depend on expiring chat attachments.

## 12. Cleanup: remove clutter, not unfavorable evidence

Perform cleanup after migration and baseline preservation, before the final
fresh-install test. Produce `reports/cleanup_manifest.csv` with exact path,
tracked/untracked state, action, reason, replacement, archive reference, and hash.
First run it as a dry run. Do not broadly run `git clean -fdx`, `rm -rf experiments`,
force-add the entire checkout, or rewrite history.

| Existing content | Final treatment and safety condition |
|---|---|
| Root notebook implementation and v12-only loaders | Migrate or retire only once the selected runtime replaces them and parity/package tests pass. Preserve comparison provenance. |
| `research/no_asr_review/`, `research/no_asr_followup/` | Move useful audit/selection helpers and tests into canonical modules; preserve a compact result ledger and immutable archive; then remove duplicate experiment scripts from active tree. |
| Restored `experiments/vcm_*` | Extract the selected training/runtime components and mandatory regressions first; archive failed trials; then remove runtime dependencies on these directories. |
| `final_suite/` | Migrate reusable checks and source evidence; retire payload-only loaders only after the final source/artifact distribution is self-contained. |
| `tools/retire_asr.py` | Remove the completed one-time migration after ASR-exclusion tests stand independently. Do not remove the regression tests with it. |
| Historical `artifacts/*.json`, `validation.json` | Archive as historical; replace current metadata only with a fresh result for the selected bundle. Do not carry old passing flags into a new release. |
| `data/`, source ledgers and raw recordings | Keep versioned manifests/licenses/pulling scripts. Move raw data outside Git. Delete unreferenced local data only by explicit manifest and recoverable archive, not by arbitrary age. |
| Old `.pt`/`.onnx`, policies and deployment ZIPs | Preserve one rollback/reference in a verified archive, keep one selected deployment artifact, remove rejected duplicates from active package paths. Never ship dormant ASR files. |
| `.cache/`, temporary features, logs, bytecode | Remove regenerable items only after confirming they are not sole evidence. Preserve selection and final test logs. |
| Duplicate report notebooks | Consolidate into one current notebook; preserve full command CSVs and archive identifiers before deleting old copies. |
| `visualizer/` | Keep requested source and selected recipe compatibility as an optional development extra; exclude from the Pi package. |
| CI workflows | Consolidate no-ASR, data-integrity and release checks; remove obsolete artifact-acquisition/migration workflows, never replace quality checks with unconditional success. |

Tracked historical source is recoverable from the baseline commit, but ignored
weights/raw evidence are not. Archive those with a hash inventory and verify a
restore before deleting their only working copy. Do not redistribute third-party
WAVs without permission. Low accuracy is not a criterion for deleting a test.
Only corrupt, invalid, duplicate or out-of-scope training examples may be removed
with documented rules; preserve their exclusion ledger. Keep hard negatives and
all sealed-test failures. Update every import, manifest, README command and CI
path after moving files. Test with old directories physically absent.

## 13. Work order and small commits

1. Recovery inventory, dataset/source rights, baseline replay and requirement map.
2. Canonical shared runtime extraction, preserved removal tests and ONNX parity.
3. Audited data pulling/splits, bounded own-model training and guarded selection.
4. Frozen calibration, exhaustive CSVs, confirmation and explicit qualification.
5. Terminal runtime and self-contained ONNX Runtime publisher.
6. Manifest-driven cleanup and fresh-clone/extracted-package verification.
7. One final notebook/report update, artifact checksums, CI and branch commit.

Write a failing regression test before each behavioral change; run it before and
after the fix. Keep commits focused and use ordinary fast-forward pushes to the
same branch. Never have CI auto-promote weights or force-push on your behalf.
If shared remote history advances, fetch and reconcile without discarding it.

## 14. Final response required from Codex

Return the actual commit SHA and branch, selected model/data/policy hashes,
complete changed/removed-file summary, restoration path for archived evidence,
exact fresh-install commands, tests passed/failed/skipped, qualification status,
and links/paths to all CSVs, notebook, model card and ONNX package. Show measured
acceptance/precision/negative failures per family with examples. Identify evidence
that could not be obtained, including physical Pi/microphone tests when absent.
A successful diagnostic build may be reported as such, never as a qualified final
assistant. Do not claim new accuracy from old reports or a successful export.

## 15. Source and evidence boundaries

Baseline repository evidence: `README.md`, `data/contract.json`,
`final_suite/final_protocol.json`, `research/no_asr_review/RESULTS.md`,
`research/no_asr_followup/RESULTS.md` and its `reports/summary.json`.
Archive recovery identities and the checks performed when writing this handoff
are in `docs/finalization/RECOVERY_MANIFEST.json`. Historical quantitative results
above come from those retained reports, not a new training run in this handoff.
Sections 4-14 specify required future implementation and verification.

Primary runtime references checked 2026-09-29; pin implementation versions and
recheck compatibility rather than upgrading by implication:

- https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- https://onnxruntime.ai/docs/
- https://docs.pytorch.org/docs/main/onnx_export.html

ONNX Runtime documents the thread/spinning tradeoff, recommends static quantization
for CNNs, and warns that quantization can reduce accuracy or performance. The
proposed Pi settings are a starting benchmark configuration, not verified Pi
results. Keep deployment small by measuring actual runtime cost, not assuming
that fewer parameters or operations make a model faster.
