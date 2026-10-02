# Codex handoff: strict-source final VCM campaign

## Scope and current state

Start on `codex/pcv-strict-final-v1`, branched from `codex/semifinal-handoff` at `7d8296c96e19e1e65fc3f0cd1b82d2391df33dbc`. Read `EXPERIMENT-pcv-strict-final-v1.md` and `experiments/pcv-strict-final-v1/protocol.json` before older handoffs. For this campaign, the user's new dataset restriction overrides old bootstrap profiles and old instructions selecting another branch.

**Completed in chat:** source inspection; exact hash verification of the decoder/context/pooling references; 37 passing targeted tests; schema expansion; independent CTC parity and padding probes; four untrained CPU cost comparisons; rejection sample-size calculations; two compact candidate implementations and an explicit-head export wrapper.

**Not completed:** fresh HF acquisition/audit, any new acoustic training, replay of the selected checkpoint, ONNX export or parity, full repository regressions, physical Pi/live microphone testing. The large previously reported acoustic benchmark is historical evidence. The compact models have random initial weights. Do not advertise this branch as a trained or qualified model.

Only the primary HF dataset and the optional explicitly named HF supplemental dataset may provide recordings. No Whisper/general transcription, LLM, pretrained command model, pretrained teacher, or other acoustic dataset. Existing wake/VAD remains a separately recorded pipeline component, not a command-weight initialization. No real calls/messages, Spotify playback, Home Assistant/GPIO actions or installed-artifact replacement before qualification and explicit authorization.

## Deliverable

One selected ONNX command model, one versioned strict-class label map, one frozen policy, one source/split receipt, complete success/failure CSVs and the existing portable serving/benchmark package. The custom-slot profile is separately versioned and disabled unless its own acoustic acceptance contract passes. Final report must state `QUALIFIED`, `FAILED`, or `INSUFFICIENT_EVIDENCE` per profile; never fill absent measurements with estimates.

## 1. Verify starting state and repeat the completed checks

```sh
git status --short
git branch --show-current
git rev-parse HEAD
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q scripts/test/test_pcv_strict_final.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.test.pcv_strict_preflight \
  --output .cache/pcv-strict-final-v1/local-preflight-01.json
```

Preserve unrelated and untracked work. Do not reset, clean, stash or force-push. Existing `.cache` and `dist` rules should keep payloads out of Git; verify before downloading. Store durable receipts under `experiments/pcv-strict-final-v1`; mutable audio/features/weights/predictions under the corresponding `.cache` directory. Add the report's status/link to the central `EXPERIMENTS.md` without copying the whole report. No repository reorganization is authorized.

The tests require the three pinned parent source blobs. A changed blob must be recorded as a new experiment revision; do not disable integrity checks. The AST loader is a test-only dependency isolation tool, never a runtime loader.

## 2. Acquire and freeze the permitted data, from zero cache

Reuse source acquisition primitives in `scripts/test/pcv_benchmark_source.py` and `scripts/train/pcv_release_data.py`, but do not invoke a legacy workflow that requires missing older caches or fetches an external teacher. Build a new source-only preparation entry point under `scripts/train/pcv_strict_prepare.py`. This entry point is **not implemented by the chat audit**.

First acquire the known primary revision `6947f13073e57eb6ae67e7e2fc3680700b82aa13`. Record fetched card, every payload SHA-256, schema, configs, row count, duration, provenance and acquisition command. Live HF main was not reachable in chat. A newer revision requires a new source lock and comparison against the known snapshot **before fitting**; never silently follow main.

The optional `martinnavs/ai231-fil-supplemental-data` source stays off until its immutable revision, terms, speaker identities and cross-source overlap have been audited. Do not require it for a primary-only baseline.

Preparation requirements:

- Keep `default/test` and `default/holdout` out of all fitting, normalization, architecture selection and calibration. Treat their final replay as a repeated benchmark.
- For `supplemental_synth`, route by `voice_split`, not the train filename. For `synthetic_negatives`, preserve the separate train/test reservations and resolve `source_files` to parent identities before an internal split.
- Group exact PCM, known speakers, clean/noisy or near-duplicate families and derived-audio parents into connected components. A cross-reservation component is a leakage error. Unknown speaker IDs are unknown, not evidence of independence.
- From publisher train groups, allocate approximately 70/10/10/10 to fit/development/calibration/confirmation, with constrained intent/provenance coverage. Freeze the allocation before fitting. If rare intents lack enough independent groups, report insufficient coverage; never borrow test examples or split near duplicates to fill cells.
- Preserve original metadata and official labels. Write a separate strict label column tied to the uploaded schema. Produce a disagreement CSV where official metadata, strict wording, transcript, slot or uncertainty flags conflict. Review training-only conflicts without model predictions; do not silently alter official evaluation labels.
- Expand the schema with `expand_schema`; freeze the class order explicitly as OOS plus a declared stable order of the 31 semantic keys. Bind the mapping to the model and policy. Retain all 93 phrase IDs. The differing class order in the benchmark is handled by explicit names/mapping, never guessed indices.
- Fit the existing 16 kHz, 40-mel frontend normalization on eligible fit humans only. Check sample format, clipping, duration and empty recordings. Retain evaluation anomalies with flags rather than improving denominators by deleting failures.
- Semantic training can use valid label-only rows even when phone targets are missing. Mask uncertain CTC targets. Numerals are auxiliary, never fabricated full commands; check repeated-token feasibility on the **post-stride-two** output length.

Observable completion: all manifests/hashes exist; leakage audits pass; official-versus-strict conflicts are counted; every split has per-human-intent/per-phrase/per-value support tables; no teacher cache or unapproved recording source is required.

## 3. Run the bounded acoustic comparison

Implement `scripts/train/pcv_strict_training.py` to consume the new frozen manifests and `StrictTinyVCM`. This training orchestration remains work for Codex; do not pretend the old CUDA/WavLM release command already implements it.

First run the direct arm for TCN and CRNN with matched data, three seeds (1102, 2203, 3304), optimizer, sampler and loss. The fixed configuration is in protocol JSON: width 96, batch 32/microbatch 8, AdamW 8e-4, weight decay 1e-3, clip 5, 5-epoch warmup, cosine to 2e-5, at most 80 epochs and 12-epoch development patience. Sample 16 human positives, 8 publisher synthetic positives and 8 train OOS per batch, balancing positives by semantic outcome. Missing pools block a claimed matched experiment; they do not authorize held-out borrowing.

Loss: command CE + 0.2 phrase CE. Unknown/uncertain phrase targets are masked, not guessed. The command target is the full strict semantic outcome, so wrong brightness/temperature/timer values receive direct discriminative supervision. Do not use numeric MSE regression for the three allowed values.

Choose on human macro exact-command development recall subject to OOS constraints, then worst-intent recall, then measured one-core ONNX cost, then size. Never select on a pooled score dominated by synthetic voices. Save every seed and failure, not just a favorable seed.

Only after the direct comparison, test the selected backbone with 0.1 auxiliary CTC on verified targets and a separately declared numeral batch plan capped at 12.5%. This is an ablation, not a pretrained teacher. Auxiliary heads are removed from the strict export. If both compact candidates fail, run the specified direct-command contextual-head control to separate a backbone limitation from a CTC-versus-classification objective effect; this contextual-head control is not implemented in the chat branch.

A0 is the existing hard-label PooledContextVCM. Its actual weights must be obtained from the user's verified local bundle or regenerated from provenance. Report it unavailable otherwise. A catalog-restricted A0 run requires new profile calibration. A0 versus a new direct model changes multiple factors; only matched new-head/backbone comparisons justify a causal architecture statement.

## 4. Rejection calibration and untouched confirmation

Fit direct-head temperature and an OOS/runner-up-margin threshold on calibration only; no per-test-case adjustments. Freeze model, normalization, label order, profile, threshold and source hashes before confirmation. Selection must not inspect confirmation failures to choose among tried thresholds.

For the existing CTC reference, explicitly test 0/200/400/800 ms trailing-silence perturbations and gain/channel conditions derived only from permitted fit data. The chat probe demonstrated score/T duration dependence; it did not prove which recorded negatives caused it. Measure before changing VAD/cropping. A different catalog, frontend, endpoint or quantization invalidates the prior calibration.

Use the new `gate` helper only as a minimum check, not a complete qualification implementation. It requires honest counts and stronger one-sided precision/FAR evidence. Add per-intent/phrase coverage, provenance and runtime evidence. Insufficient sample counts or correlated TTS parents cannot be turned into a statistical pass through duplication.

Operational demo metrics and statistical-assurance metrics must both be shown. A class demonstration may meet an observed error target without supporting a 95% population bound; distinguish these statuses instead of moving a gate after seeing results.

## 5. ONNX runtime integration is an explicit ABI change

The existing verifier requires outputs `ctc,intent,target`; the proposed strict graph outputs only `command_logits [1,32]`. **Do not rename the new graph to model.onnx and put it into the old recipe.** Add an explicit runtime kind and a new `scripts/runtime/pcv_strict_runtime.py`, selected through a tested runtime factory. Preserve the old runtime and its frozen bundles.

Reuse existing frontend, typed-output validation, resident worker, gateway, benchmark adapter and package integrity machinery. Strict logits map through a frozen 31-outcome table into the existing typed command payload. Map VOLUME_UP/DOWN to internal VOLUME + operation and back to the official 19-intent log. CALL/MESSAGE without a named recipient are finite intent recognition, not authority to choose a real person. All actions remain inert.

Use `StrictExport(model, extensions=False)` for strict export. Verify dynamic lengths 1,3,16,37,80,121,200,400,800 and the maximum supported duration; include clean/noisy/short/long examples and classifier decisions, not just tensor tolerances. ONNX tests were not run in chat because ONNX/ORT were absent. CRNN's no-length batch-one inference path avoids padded bidirectional recurrence; test it against the length-aware PyTorch path.

FP32 is the reference. Compare static calibrated INT8 for the convolutional candidate and appropriate dynamic/static alternatives for CRNN, using only calibration-fit recordings. Keep quantization only if exact-command/rejection parity and physical Pi cost improve. No assumption that INT8 or FP16 is faster on Pi 4.

ORT settings to test: CPUExecutionProvider, sequential execution, one inter-op thread, intra-op 1/2/4, spinning disabled as an initial configuration, explicit recorded CPU affinity for the one-core comparison. Bind frontend, label map, model, policy and profile hashes; package no torch, datasets, pyarrow, transformers, training cache or unused models.

Observable completion: unit tests, dynamic ONNX parity, authoritative CLI/HTTP/WebSocket parity, extracted-package no-network/no-training-dependency checks, and one canonical output per command.

## 6. Run three separate final suites

1. Official class pipeline: pin `airimonda/vcm-benchmark` at `ab39857cf73dd067d068fed388c321ab160dc191`; use a pinned local holdout manifest. Do not allow its default moving-HF cache to silently change the corpus. Run genuine laptop-speaker to Pi-microphone trials with wake word, shuffled order and authoritative logs.
2. Isolated VCM replay: unchanged original primary test/holdout, supplemental reserved test/holdout, and synthetic-negative test as separate tracks. Record raw candidate versus accepted exact, human/synthetic/accent/source, all 19 intents, all 93 phrases, unconditional slot accuracy and conditional slot distances. Never add the five track percentages into one headline hiding provenance.
3. Extended profile: 0–100 value generalization, named lights, finite song/contact aliases, unknown aliases, duplicate names, unavailable devices, on/off, to/by, teen/tens, missing units, number-bearing names, negation, truncation, multiple commands and extra fires. Missing waveform cells remain untested; parser tests do not fill them.

Measure Pi model-only and full inference latency separately from wake/endpoint/transport. Record p50/p95/p99, RTF, whole-assistant RSS/CPU, temperatures, frequency and throttling, model bytes and exact ORT version. Proposed budgets: inference p95 <=250 ms, end-of-speech response p95 <=750 ms, RTF p95 <=0.2, process RSS <=256 MiB, graph <=4 MiB, no duplicate fires or throttling. These are engineering proposals, not invented class rubric thresholds.

## 7. Optional custom synthesis: last, isolated, reversible

Keep a frozen primary-only winner and failure report. Branch a new experiment/run for custom augmentation; never overwrite this baseline. Use the repository generator only after the strict baseline and source audits exist. Generate primarily unsupported extension vocabulary/compositions and controlled strict-OOS near misses, from training-only text/identities. No exposed test-error prompt harvesting, held-out voice cloning or generated official-test replacements.

Track generator/config/voice/seed/text/parent IDs and clean/noisy families. Compare zero versus bounded low synthetic contribution with identical human batches and selection. Retain it only if it improves human development exactness without harming OOS or strict-profile recall, then lock and evaluate confirmation. TTS coverage never substitutes for human spoken coverage or creates independent speakers.

## Stop conditions and final response

Never enable a failed policy. Stop the affected stage on source drift, unresolved cross-split parents, insufficient calibration groups, unsupported ONNX operators, profile binding mismatch or failed confirmation. Keep partial results and give exact counts/reproduction commands. Do not stop an unrelated local agent or delete its work.

Return the selected architecture/checkpoint hashes, all matched-arm results, per-intent blockers, calibration/confirmation receipts, official pipeline report and measured Pi budgets. Where no candidate qualifies, say so and retain the diagnostic baseline. The local training and physical measurements are the substantial remaining evidence; packaging alone cannot make this a final model.
