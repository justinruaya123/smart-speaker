# Speaker demo v2 dataset and training implementation plan

Compact export fallback, 2026-10-16: the bounded 935,567-parameter fit is complete. The original INT8 development gate failed (human raw negative candidates 39->41, limit <=1 additional); no calibration/test inference followed that failed stage. Preserve it without changing its guard. Reuse the same numerically verified FP32 graph (3,864,875 bytes), with a separately frozen four-MiB FP32 resource allowance before human policy fitting/test. The user parameter ceiling remains 1,200,000; no retraining, architecture change, quantizer search or test-driven tuning. The new continuation wrapper uses the reviewed policy/report/package helpers. Its source/graph/protocol and human thresholds must be locked before the single published regression replay. Report the failed INT8 attempt and larger FP32 footprint explicitly, and verify the actual extracted FP32 archive. Physical Pi speed remains unmeasured.

Compact-recall authorization, 2026-10-16: the user approved the compact pass and explicitly wants fewer rejections of correct commands. Execute `speaker-compact-recall-v1` inline. This supersedes the earlier stop rule for this separate study and permits development/calibration-selected demo intent thresholds; preserve the original .95/.90 rule as a comparison and preserve strict qualification gates.

1. Use the existing pooled context model at width 128, six blocks, four heads, feedforward 256, dropout .1: 935,567 neural parameters. Enforce the 1,200,000-parameter ceiling before fitting/export and a two-MiB INT8 graph limit. Reuse the current frontend, 1,079-entry full catalog, human-training normalization, prepared features and original evaluation membership.
2. Audit supplied supervision without inventing listener verification. Exclude the six prepared training rows carrying unresolved slot-metadata-conflict flags and exclude new Qwen training rows from this control fit. Preserve every exclusion and its source; do not relabel or remove evaluation rows. Existing normalization remains based on the parent 1,869 human training utterances, including the six excluded supervision rows; it remains training-only and identical for comparison.
3. Reuse the frozen balanced trainer through an isolated module instance: one seed 53, 24 human-foundation epochs and 32 mixed-control epochs, 96 updates/epoch, batch 32/microbatch 8, existing losses/optimizer and conservative masking. Keep development checkpoint selection under the parent fixed .95/.90 rule; no new synthesis, teacher, arm or architecture search.
4. Fit a separate demo policy using original human development and calibration predictions only, after locking weights and INT8 export. Per intent, try [.50, .60, .70, .80, .85, .90, .95], keeping slot and light-target checks at .90 and retaining intent agreement/schema validation. Require positive references in both human views before relaxing that intent. Select the lowest rejection point with no increase in wrong accepted commands or UNKNOWN false accepts on either view relative to .95; ties retain the higher threshold. This is a finite diagnostic policy fit, not proof of unseen-human safety. Missing classes retain .95. Strict .99 precision/.90 recall/.01 FAR/all-human-intent qualification remains separate and unchanged.
5. Freeze graph/catalog/normalization, both policies, source and evaluation identities before one published test/holdout regression replay. Report raw, original fixed-rule, relaxed-demo and qualified outcomes separately, including wrong accepts, UNKNOWN accepts, per-intent and exact OptionB wording counts. Do not revise thresholds from those results.
6. Package the compact CPU ONNX runtime with the existing offline Pi assets and a small explicit demo/qualified decision adapter. Preserve the underlying qualified output; actions stay disabled. Verify the extracted CLI, source/dependency closure and actual decision parity. Document methods/citations, negatives and physical Pi limits, run focused and repository checks, and obtain one final fresh-context review. Preserve all prior artifacts; no commit/push/install/device action or cleanup.

Final-refinement authorization, 2026-10-16: the user approved continuing as the last refinement, with citations, explicit augmentations, grounded studies, unchanged architecture and a practical Raspberry Pi 4 handoff. This supersedes the earlier preparation-only execution boundary for this new study only. Execute `speaker-final-refinement-v1` inline in the existing feature checkout; preserve installed artifacts and every prior study.

Final bounded execution:

1. Add and test one research trainer consuming the completed synthetic preparation. Freeze the existing `PooledContextVCM` configuration (192 width, six layers, four heads, 384 feedforward, dropout .1; 2,066,831 parameters), frontend, 1,079-entry finite catalog, normalization and input hashes. Keep the existing output ABI and decoder.
2. Run seed 53, with one shared random-init human foundation (24 epochs), then two matched mixed arms (32 epochs each), 96 updates/epoch, batch 32, microbatch 8. Foundation uses 24 human positives / 8 human negatives. Both mixed arms use 12 human positives / 4 human negatives; control adds 16 earlier synthetic records, completion adds 12 earlier / 4 new Qwen records. Hierarchical intent/complete-command/declared-speaker sampling and matched per-group random streams prevent synthetic row counts dominating human examples. This is one bounded comparison, not a seed-robust result.
3. Apply the same conservative training-only SpecAugment adaptation in both arms: independently apply at most one frequency mask (0–4 of 40 bins) and one time mask (0–min(10, floor(.05*T)) frames), each with probability .5, replacing normalized features with zero. No time warping, speed/noise/reverb synthesis, splicing, teacher training or new heads. Keep AdamW, existing CTC + .3 intent + .15 target losses, cosine learning-rate schedule, weight decay .001 and gradient clip 5. Use .0008 foundation / .00025 mixed learning rates, minimum .00002.
4. Evaluate full frozen development every four epochs using true unpadded output lengths and the existing .95 intent / .90 every-slot application rule. Select by feasible human precision/FAR, macro-human-speaker accepted complete-command recall, precision, false accepts, raw recall, then synthetic accepted recall; retain earlier epoch/control on ties. Feasible means nonzero accepted correct commands, precision >=.99 and human UNKNOWN FAR <=.01. If no checkpoint is feasible, retain the best diagnostic candidate and record failure. Do not select on calibration/test/holdout. Stop after the declared comparison; no result-driven training extensions.
5. Lock the winner, export FP32/dynamic INT8 through the existing exporter, check numerical/typed development parity and the existing four-MiB INT8 limit. Calibrate only on original human calibration and retain missing-class failures. Lock before one published test/holdout evaluation, report human/synthetic/ambiguous strata and fixed demo gates separately, and preserve all errors. These previously examined published sets are regression evidence.
6. Reuse existing CPU endpoint runtime and Pi packaging; target the existing Linux ARM64 / CPython 3.12 contract, one inference thread, no training/TTS/PyTorch in the archive. Prepare an immutable Pi4 handoff with exact target/dependencies, citations, source/model hashes, measured desktop limits and physical Pi test commands. Keep unqualified/demo status explicit. Perform final focused checks and one independent review. No architecture search, service integration, physical-device action or unrelated cleanup.

Final-refinement outcome: the bounded comparison and locked evaluation are complete. Control epoch 32 won the declared human development selection; additional Qwen recordings improved synthetic development but did not improve the human selection metric. The unchanged 2,066,831-parameter model exports to a 3.03 MiB INT8 graph. Human test raw complete commands are 516/636; correct accepts at .95/.90 are 274/636, with 274/297 precision and 3/229 UNKNOWN false accepts. Qualification remains disabled. The completed diagnostic archive, extracted desktop replay, exact augmentation/citation record and physical Pi commands are documented in [the final report](EXPERIMENT-speaker-final-refinement-v1.md) and [verification](../experiments/speaker-final-refinement-v1/verification.json). No further training or test-driven refinement follows. The earlier preparation/future-task sections below remain historical plans, not instructions to extend this final run.

Earlier steering, 2026-10-16: the user has no new human recordings and selected synthetic generation. The original v2 preparation protocol and its blocked result remain preserved. Continue through the separate `speaker-synthetic-completion-v1` research version below. Synthetic development/calibration supports research only; it does not close the original human qualification gates.

GPU execution note: the SSH host `ai-n002.hpc.coe.upd.edu.ph` exposes eight A100-SXM4-40GB GPUs outside the command sandbox. Use an approved host command for CUDA stages, and verify an actual allocation before choosing hardware. Use one available GPU, leaving other workloads alone. The CPU-only conclusion from a sandboxed `nvidia-smi` was incorrect.

Synthetic-completion implementation steps (inline, existing workspace):

1. Add `scripts/train/speaker_synthetic_data.py` and `scripts/test/test_speaker_synthetic_data.py`. First test finite prompt/schema validation, exhaustive declared light/color combinations, disjoint designed-voice splits, duplicate/source contamination and preserved synthetic provenance. Implement `build_prompts(definitions, custom)`, `build_plan(prompts, voices)` and `audit_generated(rows, plan, existing)`; never manufacture human labels or slots.
2. Pin Qwen VoiceDesign/Base to the existing verified acquisition receipt, add their canonical source ledger, and extend the root environment using `experiments/speaker-synthetic-completion-v1/requirements-synthesis.lock` without replacing CUDA PyTorch or numerical packages. Freeze model/code/voice/prompt identities before generation. Generate whole commands on one A100 with distinct designed train/dev/calibration voices. Preserve native and derived hashes, failed attempts and immutable completed rows. Focus new audio on the five sparse families, all declared light/color pairs and existing rejection definitions; reuse existing eligible synthetic recordings.
3. Implement and test `prepare_run(protocol, installed, generated, output)` in `scripts/train/speaker_synthetic_prepare.py`, separate from the generator so frozen generation code remains unchanged while preparation is developed. Preserve the old audit/views, original human and published test/holdout membership. Use existing separately labelled synthetic dev/calibration recordings plus the new disjoint generations. Extend the *new* catalog with the full predeclared room/color matrix, independently of any individual calibration error. Freeze train/dev features and human-training-only normalization; do not decode published test/holdout for preparation. Allow research fitting with missing human classes while recording `human_qualified=false` and retaining the original failed human gates.
4. Run the new and inherited checks, actual generation/audit/preparation and independent review. Register sources as research-only and record actual counts, failures, timings, hardware and remaining human/wake/Pi limitations in `EXPERIMENTS.md` and compact experiment receipts. Keep the frozen generator unchanged; a separately versioned recovery helper must preserve unjournaled failed waveforms as unaccepted evidence and consume only the interrupted declared attempt. Subsequent matched model fitting uses the existing full typed-command architecture and fixed gates; no installation or physical actions follow from this data authorization.

Synthetic-completion outcome: all 552 planned commands completed on GPU 0 in 898.22 seconds, with 138 batch attempts and zero waveform-QA failures. Actual preparation produced 24,756 feature files, 23,479 eligible training rows (1,869 human / 21,610 synthetic) and 1,251 development rows (256 human / 995 synthetic). Twenty-six human training rows with unavailable pronunciation metadata remain in an exclusion manifest. Normalization uses only the 1,869 eligible human training utterances. The 1,079-entry catalog preserves all 1,031 parent entries. Original human calibration and published test/holdout audio were not opened; generated calibration was checked for integrity only. See [results](../experiments/speaker-synthetic-completion-v1/results.json) and [verification](../experiments/speaker-synthetic-completion-v1/verification.json). Fitting remains a subsequent task under the original scope below.

These commands require the exact hash-bound ignored installation and pinned model assets. Restoring a fresh installation with a different audit timestamp/path requires a new protocol version. Run generation only before a completion receipt exists; reuse the completed receipt for this corpus. Preparation always requires a new output directory:

```sh
# CUDA stage requires host GPU visibility; the default command sandbox hides it.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.train.speaker_synthetic_data generate
.venv/bin/python -m scripts.train.speaker_synthetic_prepare --output .cache/speaker-synthetic-completion-v1/my-preparation
```

The completed run needed no recovery. For a future interrupted run, first ensure its generation process has stopped, then identify its exact voice, zero-based batch offset and attempt. `speaker_synthetic_recovery` v1 atomically publishes a separate recovery journal and missing attempt receipt, never accepts unjournaled audio, and preserves failed arrays/native/derived files. The original generator then skips that consumed attempt and uses the next frozen attempt; exhausted attempts still require a child study version. Do not invoke recovery on a complete or active run.

```sh
.venv/bin/python -m scripts.train.speaker_synthetic_recovery --voice train-en --batch-offset 0 --attempt 0
```

> Execution scope: prepare datasets and the training plan first, as requested on 2026-10-13. Training, model installation and device actions are not part of this execution. Future implementation should use superpowers:executing-plans inline; do not delegate without authorization.

**Goal:** improve correct accepted complete commands on unfamiliar human speakers while preserving rejection quality and all requirements in `REQUIREMENTS.md`.

**Architecture:** install the pinned primary corpus and Filipino synthetic supplement into one immutable audio pool. Keep an unchanged human evaluation denominator, compare matched human-only and synthetic-augmented fitting, and require explicit export, calibration and physical Pi evidence before any final-model decision.

**Tech stack:** root Python environment, PyArrow, SoundFile, NumPy; existing VCM frontend, schema and reviewed offline annotations. Future training uses the established PyTorch and ncnn/ONNX workflows.

**Spec:** `REQUIREMENTS.md`, `docs/HANDOFF-speaker-demo-v1.md`, `experiments/speaker-demo-v1/protocol.json`, and the pinned supplemental source card.

## Global constraints

- No runtime ASR/STT, transcript parser, LLM or cloud model calls.
- Preserve installed models, existing frozen protocols and test/holdout membership.
- Keep the application thresholds at 0.95 intent and 0.90 for every required slot. Evaluate accepted typed commands, not only raw intent accuracy.
- The Filipino supplement remains training-only augmentation. The separately authorized synthetic-completion version also uses disjoint synthetic development/calibration conditions for research. Synthetic clips are never human speaker evidence or new test/holdout data.
- Exclude failed supplemental QA unless the publisher explicitly records rescue; preserve excluded rows and their reasons.
- Fit normalization on eligible human training audio only. Select checkpoints on development, voice/confidence thresholds on calibration, and use locked test only after selection.
- Random initialization for new acoustic candidates; no silent pretrained speaker encoder or source-fitted checkpoint reuse.
- All ten use cases and required typed parameters remain in scope for the final model. The 22,027-parameter demo classifier is a pilot, not proof of full specification compliance.

## Review focus

1. A primary fitting recording duplicates a test/holdout recording or declared speaker: exclude it from fitting and retain evaluation membership.
2. Supplemental failed-QA, ambiguous or invalid audio: preserve it in inventory and exclude it from supervised fitting.
3. A synthetic reference voice aliases a human evaluation speaker: unresolved aliases block unfamiliar-speaker claims; obtain provenance or a fresh independently recruited human holdout.
4. A slotted command lacks a verified value: do not manufacture a default or convert an ambiguous reference into UNKNOWN.
5. Dataset addition changes competition, calibration or export behavior: create a new version, measure parity and retain the old result.

## Dataset installation

- [x] Pin primary `airimonda/ai231-me2-voice-commands` at `25111444af3adff7588d86ab27c304bb080895cf`; verify all ten Parquet shards.
- [x] Pin supplement `martinnavs/ai231-fil-supplemental-data` at `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`; download all six shards, including gap-fill and numeral-wording configurations.
- [x] Add `scripts/train/speaker_dataset_install.py` and meaningful regression tests in `scripts/test/test_speaker_dataset_install.py` for QA exclusion, split contamination, duplicate/conflicting supervision, atomic immutable audio publication, shard-directory identity, source-receipt completeness, and hash-checked historical CRLF recovery.
- [x] Extract original embedded WAV bytes into `data/audio/ai231-pool-v1/`; retain file and suite-compatible PCM hashes and all publisher metadata in `.cache/speaker-dataset-v1/verified/inventory.jsonl`. Two complete runs produced identical inventory and view hashes; the second uses the reviewed installer.
- [x] Produce clean human fitting views, a separately labelled synthetic training view, isolated-numeral inventory, and complete published evaluation views. Save compact audit/environment receipts in `experiments/speaker-demo-v2-data/`.
- [x] Verify every decoded recording, source hash and view identity; register the dataset installer as research-only and add the dataset-preparation entry to `EXPERIMENTS.md`.

Measured preparation: 81,686 primary plus 15,973 supplemental rows, 97,659 unique original WAV/PCM identities, and 16 verified source shards (4,137,836,220 bytes). Human fitting views contain 1,895 train, 256 dev and 210 calibration records; the separate synthetic training view contains 21,426 records. 66,390 isolated numerals stay auxiliary. Published test/holdout retain all 4,418/196 rows. Fitting excludes 504 ambiguous records, 529 failed-QA records and 11 missing-speaker records. Exact PCM and declared speaker/pair cross-split groups are zero; source aliases, voice identity, near-duplicates and independent label listening remain unverified. Synthetic data does not close the missing human intent coverage listed in Task 1.

## Future task 1: freeze the comparison before fitting

**Files:** new `experiments/speaker-demo-v2/protocol.json`, new `scripts/test/test_speaker_demo_v2_data.py`, new `scripts/train/speaker_demo_v2_data.py`.

**Interfaces:** consume the installation inventory and hash-bound views; produce immutable run-specific manifests, catalog, human-training normalization, feature receipts and `selection-before-test` inputs.

Adapter contract: `prepare_run(protocol: Path, installed: Path, output: Path) -> dict`. Return a receipt with `manifests`, `catalog`, `normalization`, `source_bindings` and `implementation_bindings`; each artifact records its path and SHA-256. Refuse an existing output directory or any source/view identity mismatch.

Continuation, 2026-10-16: the preparation adapter and regression tests are implemented in `scripts/train/speaker_demo_v2_data.py` and `scripts/test/test_speaker_demo_v2_data.py`, registered as research-only. [The preparation protocol](../experiments/speaker-demo-v2/protocol.json) binds the existing installation audit, unchanged 1,031-entry release catalog and all 415 benchmark/superset definitions. This executes the preparation portion of the plan; collecting missing human data and freezing actual fitting features remain incomplete.

The actual corpus check exits **2** before feature extraction or training. Human train/dev/calibration cover 25/31, 22/31 and 15/31 OptionB typed commands respectively; synthetic train covers 31/31. Missing human intents remain as listed below. A further catalog mismatch is now explicit: calibration contains `COLOR(color=red, light_id=light.bathroom)`, absent from the frozen release catalog. It is preserved as a supported out-of-catalog reference, not silently changed to UNKNOWN or added from calibration. Resolve it through a separately reviewed catalog/protocol version. [Readiness and exact recording requests](../experiments/speaker-demo-v2/readiness.json) retain the missing intent cells, typed values, declared-person counts, source bindings and published evaluation denominators. The requests identify absent cells; they are not a sufficient sample-size prescription.

The adapter rejects synthetic records in human train/dev/calibration, bad QA, ambiguous fitting labels, duplicate/cross-split identities, source/view changes and incomplete typed commands. It preserves the original mixed-origin published test/holdout manifests and reports their kinds separately. On complete WAV fixtures, it freezes original manifest bytes, raw feature hashes and normalization from human training only, while never opening calibration/test/holdout audio. These are software checks, not measured acoustic gains. Corpus feature extraction remains blocked. Existing test exposure, unresolved voice aliases and independent listening still limit unfamiliar-speaker claims.

Reproduce the gate using a new output directory (exit 2 is the expected blocked result):

```sh
.venv/bin/python -m scripts.train.speaker_demo_v2_data --output .cache/speaker-demo-v2/my-preflight
.venv/bin/python -m pytest -q scripts/test/test_speaker_demo_v2_data.py scripts/test/test_speaker_dataset_install.py scripts/test/test_speaker_demo_v1.py scripts/test/test_pcv_updated.py
```

- [ ] Collect independently reviewed human MESSAGE recordings for train/dev, CALL recordings for dev, and human CALL, MESSAGE, LIST_REMINDERS, TEMPERATURE and CREATE_REMINDER recordings for calibration. Obtain disjoint people for each partition; additional TTS cannot close these human gates. Freeze a new manifest version before any candidate comparison. Preserve the existing published test/holdout.

- [x] Write failing tests that reject synthetic rows in human fitting/dev/calibration, failed-QA input, cross-split file/PCM/speaker/pair leakage, fabricated slot defaults and missing human class coverage. Preserve the original mixed-origin published test/holdout and its human/synthetic denominators.
- [x] Implement the versioned preparation adapter using existing reviewed annotations and `vcm.frontend.features`. Reuse audio paths rather than copying recordings. Never modify v1's human-only trainer to accept TTS silently. The complete-corpus preparation receipt remains blocked until coverage and catalog gaps are resolved.
- [ ] Freeze approved full-command catalog and the complete-command/UNKNOWN denominators before inference. Separate metadata-ambiguous records from true negatives.
- [ ] Resolve reference-speaker aliases. If resolution is unavailable, record that limitation and recruit at least 20 fresh human speakers for a separate blind holdout before making unfamiliar-speaker claims.
- [ ] Run `.venv/bin/python -m pytest -q scripts/test/test_speaker_demo_v2_data.py` and the existing preparation tests. Write source, data and environment hashes before training.

## Future task 2: run matched acoustic comparisons

**Files:** new `scripts/train/speaker_demo_v2_train.py`, new `experiments/speaker-demo-v2/run_grid.py`, new `scripts/test/test_speaker_demo_v2_train.py`; reuse `scripts/train/speaker_demo_train.py` and `scripts/train/pcv_release_training.py` as separate architecture baselines.

**Interfaces:** consume Task 1's frozen fitting/dev data; produce unqualified checkpoints, histories and development-only selection receipts. No test results enter the trainer.

Trainer contract: `fit_run(protocol: Path, prepared: Path, output: Path, *, arm: str, seed: int, device: str) -> dict`. Return selected checkpoint/normalization/catalog paths and hashes, history path, development selection metrics and `status="unqualified_candidate"`. The grid calls this function sequentially for the three named data arms and three seeds and stops on failure.

- [ ] Test identical update budgets and exclusion of calibration/test/holdout, plus deterministic sampling and correct reporting of synthetic sampling mass.
- [ ] Run seeds 53, 71 and 89 with matched optimizer/frontend/batch size and update budgets: human-only balanced fitting; the same fitting plus default QA-passed Filipino supplement; and the same fitting plus QA-passed/rescued gap-fill and numeral wordings. Cap synthetic sampling to 50% of each batch and balance its voice/wording cells; keep the remaining mass human.
- [ ] Use development macro-human-speaker accepted complete-command recall for selection, subject to rejection/precision constraints. Report worst-speaker, per-intent/slot, source and Filipino-accent strata with denominators and speaker-cluster uncertainty.
- [ ] Treat the small fixed-demo classifier as an ablation. For the final all-use-case model, run the same data comparison with the full typed-command architecture, or a newly approved causal model from `codex/pcv-streaming-v2`; do not claim the sibling branch is already integrated.
- [ ] Run training integration tests on CPU first, then the actual grid on available compute. Preserve partial/failing runs; do not overwrite run directories.

## Future task 3: export, calibration and locked human evaluation

**Files:** existing `scripts/test/speaker_demo_suite.py`, `scripts/test/speaker_engine_replay.py`, `scripts/test/pcv_release_qualification.py`; new versioned candidate adapter and parity tests as required by the selected architecture.

- [ ] Test explicit model-family compatibility, valid-length/padding behavior, repeated inference ordering and typed-decision parity before and after quantization.
- [ ] Export the selected candidate through the established ncnn/ONNX path; bind weights, frontend, normalization, catalog, policy and runtime sources.
- [ ] Calibrate using human calibration only. Keep 0.95/0.90 application gates and separately evaluate the model's frozen qualification policy.
- [ ] Lock selection before evaluating the preserved human test/holdout. Report correct accepted command recall, accepted precision, UNKNOWN false accepts, per-slot metrics and missing coverage; retain all errors in denominators.
- [ ] Apply existing full-model qualification gates (precision 0.99, recall 0.90, human false acceptance at most 0.01, all 18 intents in human calibration), and separately report the proposed speaker-demo targets (0.95 human accepted recall, 0.90 worst-speaker recall where sampled). These are targets, not measured results.

## Future task 4: wake ownership and physical Pi qualification

**Files:** `scripts/runtime/speaker_session.py`, `scripts/runtime/speaker_worker.py`, a new separately trained speaker/overlap backend and its calibration/test receipts.

- [ ] Obtain real wake and command recordings from disjoint fitting/calibration/evaluation speakers. Command-only supplemental TTS does not supply verified wake-owner pairs.
- [ ] Test A-wake/A-command, A-wake/B-command, handovers, overlap, short/clipped audio, expiry, overflow and stale/repeated dispatch.
- [ ] Train and calibrate speaker verification separately from speaker-invariant command embeddings; require actual voiced duration and coverage of command-bearing windows.
- [ ] Measure endpoint-to-decision p50/p95, peak RSS, load time, temperature/throttling and accuracy on a physical RPi4/5 with live speech. Preserve desktop results as desktop evidence.
- [ ] Install a model only after acoustic, export, wake-owner and hardware gates pass and installation is authorized. Keep failed gates and unsupported use cases explicit.

## Research basis and limits

### Charter

Goal: decide how to install and use the supplied supplement without invalidating the human comparison. Scope/object/subject: local VCM accuracy / speaker-demo data preparation / primary snapshot plus Filipino supplement. Type: descriptive; the task is dataset preparation and a future training protocol. Questions: Q1 provenance and intended split; Q2 audio, QA and label integrity; Q3 applicability to final-model qualification. Knowledge depth: working project knowledge, source-specific facts verified against publisher cards and raw Parquet inspection.

### Canonical sources

1. [Pinned supplemental card](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data/blob/230c8b8d58caf78e4bc90886d9c6fa77fe6024b8/README.md): provenance, configurations, QA and terms.
2. [Pinned primary card](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/blob/25111444af3adff7588d86ab27c304bb080895cf/README.md): source/split contracts and licenses.
3. Local `REQUIREMENTS.md`, v1 protocol/handoff and actual installation audit: project requirements and observed file/PCM identities.

### Findings ledger

| ID | Claim | Evidence | Rank | Question |
|---|---|---|---|---|
| F1 | Supplement is cloned synthetic training data; never human test/holdout | Pinned supplemental card, loading/caveats | 1 | Q1 |
| F2 | Gap-fill includes failed QA and exposes pass/rescue flags | Pinned supplemental card, gap-fill section | 1 | Q2 |
| F3 | Reference-speaker overlap, consent/license and accent authenticity remain unverified | Pinned supplemental card, caveats/terms | 1 | Q1/Q3 |
| F4 | v1 is human-only; candidate/export/speaker backend/live integration remain pending | v1 protocol and handoff at `5d872e3` | 3 | Q3 |
| F5 | Final acceptance requires typed parameters, unknown rejection, all use cases and Pi evidence | `REQUIREMENTS.md` | 3 | Q3 |
| F6 | Installed pool contains 97,659 valid original WAV files; 529 failed-QA fitting clips are excluded | `experiments/speaker-demo-v2-data/data-audit.json` and hash-bound inventory | 3 | Q2 |
| F7 | Full-model human fitting/calibration coverage is incomplete | Same audit, human coverage tables | 3 | Q3 |

### Scope filter

Synthetic training may help command robustness; it cannot supply real-human speaker counts, verified wake-owner trials, or unseen-human accuracy evidence.

### Object filter

The publisher's previously observed gains also changed other training data, so no isolated improvement from this supplement is established. A matched, locked comparison is required.

### Conclusions

Install and label the supplement separately, use explicit QA filtering and training-only views (F1/F2/F6), preserve v1 (F4), collect missing human coverage (F7), and require human/hardware evidence before final-model claims (F3/F5).

### Next steps / risks

Resolve synthetic reference aliases/rights; independently review uncertain labels; collect fresh human wake/command data and physical Pi measurements. Existing historical test exposure prevents calling those old splits a new blind holdout.

## Reproduction

The installed sources are described in `data/sources/ai231-primary.source.json` and `data/sources/fil-supplemental.source.json`. On a clean tracked checkout that includes these changes, acquire missing pinned payloads with `python scripts/bootstrap_semifinal.py --source ai231-primary --source fil-supplemental --apply --allow-restricted`. The flag acknowledges the supplied research-use source terms; it does not establish unresolved source-speaker consent or grant redistribution rights. Bootstrap deliberately refuses tracked/staged changes.

For another preparation run using the local payloads:

```sh
.venv/bin/python -m scripts.train.speaker_dataset_install --output .cache/speaker-dataset-v1/new-run --workers 12
.venv/bin/python -m pytest -q scripts/test/test_speaker_dataset_install.py
```

Choose an output directory that does not already exist. Windows uses `.venv/Scripts/python.exe`. Current dataset dependencies and exact versions are recorded in `experiments/speaker-demo-v2-data/environment.json`; this is a dataset-only environment and does not claim that the runtime/training lock is installed.

The installer accepts the historically verified CRLF representation of the annotation review only when reconstructed bytes match its original SHA-256. It does not change the frozen file or disable integrity checks. Older preparation tests still fail their original direct file-hash checks on the LF Git representation. The repository-wide test run also has collection blockers from unavailable model/test dependencies and existing test-module conflicts; see `experiments/speaker-demo-v2-data/verification.json` for the exact failures. No full-suite pass or acoustic improvement is claimed.
