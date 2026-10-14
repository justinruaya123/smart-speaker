# VCM intent and slots v1 implementation plan and Codex handoff

> **For agentic workers:** Use `superpowers:executing-plans` task-by-task when
> available, with test-driven implementation and verification before completion.
> Do not spawn additional agents without the applicable explicit authorization.

**Goal:** Implement and evaluate the user-selected shared acoustic intent-plus-slot
candidate without replacing the deployed model or narrowing the personal catalog.

**Architecture:** 40-bin log-mel -> width-128, eight-block residual temporal CNN ->
intent, role-state, finite-slot, request-validity and acoustic-entity heads -> typed
validation/rejection. Speaker identity is a separate one-wake/one-command gate.

**Tech stack:** Existing root Python environment, PyTorch training, einops tensor
operations, existing frontend/schema, ONNX Runtime parity and ncnn target runtime.
No new environment, transcriber, cloud inference or pretrained speaker exception.

**Spec:** [SPEC-vcm-intent-slots-v1.md](SPEC-vcm-intent-slots-v1.md).
**Protocol:** [protocol.json](../experiments/vcm-intent-slots-v1/protocol.json).
**Baseline source:** `5d872e3a249e5876aaa8130b76f6e3b31622355f`.
**Work branch:** `codex/vcm-intent-slots-v1`.

## Global constraints

- One shared command model; preserve benchmark and all existing extension outputs.
- Keep 0.95 intent and 0.90 for every supplied slot; bind calibration and score meaning.
- No runtime ASR/STT, transcript parser, LLM, cloud model or automatic fallback.
- Preserve installed graphs/configuration, the finite-CTC comparator, historical
  results, frozen protocols and other agents' dirty/staged/untracked work.
- Use existing source locations, root environment and shared read-only recording
  pool. Keep generated payloads in `.cache/vcm-intent-slots-v1/<run-id>/`.
- New implementation is experimental. `training.update_artifacts=false`;
  `installation_allowed=false`; `provider_actions_allowed=false`.
- Missing data/weights/calibration/Pi blocks the affected evidence claim, not
  unrelated safe software work. Never fabricate a substitute measurement.

## Review focus

Pin these cases in the tasks that own them: an unknown named light must not become
`light.all`; an empty/singleton/expanded catalog must not force a name acceptance;
padding must not change a command; conflicting or missing numeric evidence must
not be filled by the schema; stale/mismatched owner evidence must never dispatch.
These complement ordinary positive and shape tests.

## Execution boundary and local preflight

This commit contains design/handoff material only. Proposed new source files and
CLI subcommands below DO NOT exist yet and are not claimed tested. The pasted
kickoff prompt authorizes executing this candidate plan, not installing it.

Read `AGENTS.md`, `REQUIREMENTS.md`, `EXPERIMENTS.md`, both new documents and the
existing `docs/HANDOFF-speaker-demo-v1.md` before implementation. If present, read
another agent's `docs/PLAN-speaker-demo-v2.md` and preparation receipts without
rewriting them. Do not assume they were committed to the remote source branch.

Start with read-only status and environment discovery:

```sh
git status --short
git diff --cached --name-only
git branch --show-current
git rev-parse HEAD
```

Do not switch over dirty work, stash it, clean it, reset it or stage it. Reuse an
already isolated workspace or create an isolated worktree according to the project
rules. Record the actual source SHA and relevant dirty-diff provenance. Fetch this
branch only through the configured origin; do not alter remotes or credentials.

Select the existing root `.venv/bin/python` on POSIX or
`.venv/Scripts/python.exe` on Windows. Record imports, device availability and
versions before fitting; do not install dependencies into another environment.
Check ignore rules for a cache checkpoint, WAV, protocol JSON and both documents.
Do not make an empty ignore commit when current rules already suffice.

Apply/merge `experiments/vcm-intent-slots-v1/register.patch` only after inspecting
its context and existing directory changes. `git apply --check` must pass before
applying automatically. If the title/context changed, insert only its new section
without replacing other work. Keep the original patch as the scope receipt.

## Task 1: freeze the actual contract, provenance and prerequisites

**Create:** `scripts/runtime/vcm_intent_slots_contract.py`,
`scripts/train/vcm_intent_slots_data.py`,
`scripts/test/test_vcm_intent_slots_contract.py`.
**Update within scope:** experiment protocol/status and compact preflight/coverage
receipts; the relevant `EXPERIMENTS.md` entry. Do not rewrite source datasets.

**Interfaces:** `build_slot_contract(cases, catalog_entries, custom) -> dict` emits
versioned intent order, required/optional roles, finite value inventories, typed
assembly rules, default provenance and input-content hashes.
`audit_views(view_paths, source_receipts, output_dir) -> dict` emits role/person/
phrase coverage and exclusions, retaining original reference identities.

- [ ] Write `test_optionb_and_extensions_preserved`: compare all 93 baseline
  definitions and all existing extensions/negatives to canonical schema outputs.
  Bind the selected full catalog; do not force historic 414/415/968 objects equal.
- [ ] Write tests for VOLUME operation/percent consistency, song/artist dependency,
  default versus explicit-unknown targets, time/unit defaults, numeric bounds,
  missing/extra fields and absent `playlist_id` support.
- [ ] Write provenance tests rejecting cross-partition PCM/speaker leakage, unknown
  or failed eligibility, numeral-pool contamination and missing input receipts.
  Preserve ambiguous evaluation rows separately rather than relabeling OOV.
- [ ] Run these tests red, implement the contract/audit, rerun green and commit
  only these files and the reviewed experiment registration/receipts.

**Completion:** protocol binds actual inputs or lists explicit blockers. Missing
shipping payloads have a recovery request with original hashes. No baseline replay
is claimed from metadata. Missing label/span/enrollment data has a coverage-driven
acquisition list. Build software contracts even when acoustic inputs are absent.

## Task 2: implement the joint time-preserving model

**Create:** `scripts/models/vcm_intent_slots.py`,
`scripts/test/test_vcm_intent_slots_model.py`.

**Interfaces:** `IntentSlotsVCM(contract: dict)`;
`forward(features, lengths) -> dict[str, Tensor]`. Inputs are float features
`[B,40,T]` and valid-frame lengths `[B]`. Output keys are `intent_logits [B,19]`,
`role_state_logits [B,R,4]`, `request_validity_logits [B,4]`, a `value_logits`
mapping by finite role, `entity_embeddings [B,E,64]`, and training-only
`role_frame_logits [B,T2,R+1]`; `T2=ceil(T/2)` for the specified stem. Request
validity classes are valid/missing/conflicting/unsupported; OOV remains an intent.
Ordering/offsets and applicable labels are recorded in the frozen contract.

- [ ] Write tests for all specified dimensions, odd/even/minimum valid lengths,
  padding invariance, all-padding rejection, finite logits and gradients, and
  invariance to another padded batch member at evaluation.
- [ ] Implement the exact spec's stem, eight blocks, masked intent/role attention,
  role-state/value/validity heads and normalized entity projection. Null/absent
  handling must avoid an all-masked attention softmax. Do not infer durations
  from padding or let padded states enter later convolutions.
- [ ] Add loss tests: unknown values/spans are masked; no eligible metric pairs
  yields a finite zero contribution; different complete commands sharing an intent
  are not contrastive positives. Numeric 20/60/100 remain different labels.
- [ ] Run red/green tests; record parameter counts by encoder/head from the actual
  implementation; commit without describing numerical fixtures as speech accuracy.

**Completion:** a random-initialized full typed-command model, not another fixed
sentence classifier. Do not integrate it by weakening old runtime shape checks.

## Task 3: implement entity enrollment and typed acceptance

**Create:** `scripts/runtime/vcm_intent_slots_entities.py`,
`scripts/runtime/vcm_intent_slots_decode.py`,
`scripts/test/test_vcm_intent_slots_decode.py`.

**Interfaces:** `enroll_examples(examples, encoder_binding, catalog) -> dict`
produces the versioned acoustic memory; `match_entity(embedding, role, memory,
policy) -> dict` returns typed ID or rejection with raw/calibrated diagnostics.
`decide(prediction, contract, catalog, policy) -> dict` returns an application
verdict with accepted/command/reason and per-supplied-slot evidence/provenance.
No function accepts the expected trial/intent/slot as inference context.

- [ ] Test zero/one/many entities, exact duplicate examples, stale encoder/catalog
  bindings, invalid embeddings, unknown and tied names, role-prefix mismatch,
  aliases and catalog expansion. A singleton never bypasses absolute rejection.
- [ ] Test generic music/lights, named-song/light failure, artist without song,
  bare CALL and unknown contact. An unknown spoken target cannot default.
- [ ] Test absent, conflicting, negated, extra and out-of-bound slot evidence;
  rejected intent must remain rejected even when every slot score is high.
- [ ] Implement matching and typed assembly without transcript decoding. Keep
  confidence provenance for constants/defaults. Do not present singleton-softmax
  1.0 or an uncalibrated composite score as measured correctness probability.
- [ ] Test 0.95/0.90 boundary behavior and nonfinite/missing confidence values;
  every supplied slot remains gated. Run red/green tests and commit.

**Completion:** software decision semantics and explicit calibration prerequisites.
Numerical mocked embeddings prove policy behavior only, not acoustic name recall.

## Task 4: make fitting, attribution and evaluation reproducible

**Create:** `scripts/train/vcm_intent_slots_train.py`,
`scripts/test/vcm_intent_slots_eval.py`,
`scripts/test/test_vcm_intent_slots_training.py`,
`scripts/vcm_intent_slots.py` as the thin orchestrator.
**Register:** reviewed new entries in `scripts/modules.json` as research-only;
reuse existing samplers/audits where their contracts actually match.

**CLI contract to implement:** `preflight`, `audit`, `diagnose`, `train`, `calibrate`,
`evaluate`, `export`, `parity`, `owner-replay`, `summarize`, with `--protocol`,
explicit run IDs and reproducible cache outputs. Commands with missing bindings
must fail clearly rather than silently synthesize, download or fit defaults.
`evaluate --split test|holdout` requires a complete `selection-before-test.json`.

- [ ] Test complete-command/person sampling, preserved negatives, fixed seeds,
  deterministic update budgets, interrupted-run identity, and no calibration/test
  access during fitting or checkpoint selection. Match all non-sampler settings.
- [ ] Implement masked joint losses and the spec's initial training recipe; log
  actual sampling mass and supervision denominators. Do not allow the 66,390-row
  numeral pool or supplied synthetic rows into the human-only default arm.
- [ ] When available, use identical original PCM and the exact restored baseline
  for full-crop versus wake/VAD-crop attribution. Log raw intent/slot errors,
  confidence rejection, capture/wake/endpoint faults and catalog gaps separately.
- [ ] Fit uniform and balanced arms at seeds 53/71/89 only when audited authorized
  inputs exist. Whole-command contrastive, CTC auxiliaries, synthetic inclusion,
  capacity changes and extraction remain separate follow-up arms, not bundled fixes.
- [ ] Select on human development, then fit/freeze score calibration separately.
  Record all source/model/catalog/normalization/policy hashes before scoring fresh
  tests. Retain examined test sets as named regression tracks.
- [ ] Test trial accounting: supported rejects are failures, faults are not correct
  OOV rejections, and human/synthetic and OptionB/extension denominators are separate.
  Include per-person and per-role counts, missing cells and dependent-pair uncertainty.
- [ ] Run focused tests and feasible acoustic work. Commit source and compact real
  receipts; keep payloads ignored. An unavailable acoustic comparison stays blocked.

**Completion:** a repeatable pipeline and truthful results/blocker report. No
promise of a successful acoustic run merely because the script completes.

## Task 5: export a distinct candidate and verify decision parity

**Create:** `scripts/train/vcm_intent_slots_export.py`,
`scripts/runtime/vcm_intent_slots_runtime.py`,
`scripts/test/test_vcm_intent_slots_export.py`.

**Interfaces:** `export_candidate(checkpoint, contract, output_dir) -> dict`
produces a hash-bound new-format bundle; `IntentSlotsRuntime(bundle, engine,
threads).recognize(pcm, sample_rate=16000) -> dict` returns the experimental event.
The export wrapper uses fixed ordered tensors plus role/value offsets in the
recipe, not a Python dictionary as an ONNX output. Runtime batch is one, variable
unpadded duration; training-only localization/auxiliary outputs need not ship.

- [ ] Test missing/mixed binding files, wrong frontend/intent order/head layout,
  malformed PCM, empty/short/overlong input and corrupt or nonfinite graph outputs.
- [ ] Verify CPU PyTorch -> FP32 ONNX -> ncnn for actual recorded durations,
  short words, tail frames, supported/unknown slots and near-threshold decisions.
  Record frontend and final accepted typed-command agreement, not only average
  tensor error. Calibration may not hide backend disagreement.
- [ ] Keep the old three-head verifier and bundle unchanged. Unsupported export
  operators are a blocker to solve explicitly, not permission to relabel an ORT
  result as ncnn. Quantization is a separate arm after FP32 parity.
- [ ] Record actual bytes, model/head parameters, load and per-stage desktop timing,
  memory and catalog-size scaling. Commit metadata/source; do not install graphs.

**Completion:** a distinct experimental bundle with measured parity or a precise
export blocker. Desktop execution does not establish physical Pi feasibility.

## Task 6: integrate final-decision logging and action-free application replay

**Modify only after focused tests:** `scripts/runtime/speaker_worker.py`,
`scripts/runtime/pcv_benchmark_adapter.py`, source-map/configuration adapters needed
for an explicit experimental selection. Preserve existing defaults.
**Tests:** extend `scripts/test/test_pcv_full_benchmark.py` and add
`scripts/test/test_vcm_intent_slots_integration.py`.

- [ ] Add failing tests for demo/qualified/final-application decision selection,
  rejected diagnostic candidates, capture faults, unmapped extensions and dynamic
  task catalogs. Reuse the canonical adapter, not another formatter.
- [ ] Expose a named experimental path that computes one server-side application
  verdict, then projects that verdict into the benchmark stream. Keep raw and
  diagnostic candidates out of accepted-output serialization.
- [ ] Exercise every baseline definition and all extension/negative contract cases
  against the pinned benchmark parser `ab39857cf73dd067d068fed388c321ab160dc191`.
  Stipulated-output fixtures prove adapter compatibility only.
- [ ] Replay immutable WAVs through the actual application decision at 0.95/0.90
  with all provider actions mocked/disabled. Retain richer personal-command JSON.
- [ ] Run regressions and commit only the reviewed interface changes. No default
  model selection, dispatch, provider action or publication is allowed.

## Task 7: owner interface now; acoustic ownership only with real evidence

**Reuse:** `scripts/runtime/speaker_session.py`.
**Create if no equivalent exists:** `scripts/runtime/speaker_evidence.py`,
`scripts/test/test_vcm_intent_slots_owner.py`.
**Integrate within scope:** action-free worker/replay adapter; keep default runtime
behavior unchanged until an explicitly qualified owner-enabled path is selected.

**Interface:** the backend consumes actual wake/command PCM with exact sample-span
and session bindings; it returns identity, voiced-duration, overlap and verified
continuity coverage. Missing/unvalidated evidence cannot be converted to an accept.
The model/threshold identity travels with every observation. Do not serialize raw
speaker embeddings in application logs.

- [ ] Test A-wake/A-command, new B-wake/B-command, A-wake/B-command, A-start/B-slot,
  A-silent/B-command, overlap, short/clipped reference, unavailable coverage, timeout,
  rearm, second wake, sequence discontinuity, overflow and stale results.
- [ ] Implement the server-side intersection with the existing application verdict.
  Consume ownership after completion/reject/fault/expiry. A rejected VCM command
  cannot be rescued by valid speaker evidence or a subsequent wake.
- [ ] Use separately trained/authorized identity and overlap backends when real
  identity data exist. Otherwise complete software tests and leave acoustic
  qualification blocked. No command embedding or fabricated window list substitutes.
- [ ] Calibrate genuine/impostor sessions on disjoint people by reference and command
  duration; retain short-command rejection as a recall failure. Report repeated-pair
  dependence, genuine retention and wrong-person accepted commands.
- [ ] Keep official command-only, official owner-enabled mismatched-pair outcomes,
  and original paired-human sessions distinct. Initial overlap rejection is not
  speech extraction or owner-command recovery.

**Completion:** verified software contracts plus separately labeled acoustic
findings or missing evidence. New pretrained identity/separation weights and actual
extraction require additional explicit scope; they are not automatic next steps.

## Task 8: qualification, commits and final receipt

- [ ] Run all new focused tests, relevant existing runtime/adapter/owner tests,
  formatting/PEP8 checks and full-suite collection where feasible. Preserve prior
  unrelated failures and exact command/exit-status logs; do not claim a full pass
  from a filtered run. Tests should not require the entire research corpus.
- [ ] Require the complete benchmark and personal superset evidence, calibrated
  rejection and owner retention before any qualification claim. Missing required
  human classes, fresh people or owner pairs remain explicit blocked coverage.
- [ ] Freeze numerical Pi resource limits from the governing approved protocol
  before physical qualification. Use the actual named Pi only when accessible and
  authorized; otherwise output the exact measurement commands and state unmeasured.
  Never substitute desktop timing or hypothetical estimates for physical results.
- [ ] Update `EXPERIMENTS.md`, this handoff's progress and compact protocol/selection/
  verification receipts without overwriting other studies. Recheck source/bundle
  preservation and the exact staged file list before every commit.
- [ ] Report commit SHAs, files, commands, test counts, actual human/synthetic and
  benchmark/extension outcomes, failed gates, missing inputs, export status and
  physical/ownership evidence. Do not declare the model final from software tests.

## Command sequence after implementing the CLI

These commands specify the new CLI acceptance contract; they are NOT available or
executed in the documentation commit. On POSIX, from the repository root:

```sh
PY=.venv/bin/python
PROTO=experiments/vcm-intent-slots-v1/protocol.json
$PY scripts/vcm_intent_slots.py preflight --protocol "$PROTO"
$PY scripts/vcm_intent_slots.py audit --protocol "$PROTO"
$PY -m pytest -q -p no:cacheprovider scripts/test/test_vcm_intent_slots_*.py
$PY -m pytest -q -p no:cacheprovider scripts/test/test_speaker_demo_v1.py
$PY -m pytest -q -p no:cacheprovider scripts/test/test_pcv_full_benchmark.py -k 'adapter or live_log'
$PY scripts/vcm_intent_slots.py diagnose --protocol "$PROTO" --run-id baseline-attribution
$PY scripts/vcm_intent_slots.py train --protocol "$PROTO" --arm uniform --seeds 53 71 89
$PY scripts/vcm_intent_slots.py train --protocol "$PROTO" --arm balanced --seeds 53 71 89
$PY scripts/vcm_intent_slots.py calibrate --protocol "$PROTO" --run-id selected
$PY scripts/vcm_intent_slots.py export --protocol "$PROTO" --run-id selected
$PY scripts/vcm_intent_slots.py parity --protocol "$PROTO" --run-id selected --engines onnxruntime ncnn
$PY scripts/vcm_intent_slots.py evaluate --protocol "$PROTO" --run-id selected --split test
$PY scripts/vcm_intent_slots.py owner-replay --protocol "$PROTO" --run-id owner-evaluation
$PY scripts/vcm_intent_slots.py summarize --protocol "$PROTO"
```

Do not run later commands blindly when prerequisites failed. The initial design
protocol intentionally lacks local data/checkpoint/calibration bindings; preflight
must identify and freeze them before affected work proceeds. The old shipping
baseline may remain unavailable while implementation and new-model software tests
continue. Fresh tests require a selection lock; owner replay requires genuine
session evidence. No command may dispatch actions or install a model.
