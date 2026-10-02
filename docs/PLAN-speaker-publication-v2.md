# Speaker publication v2 implementation plan

> Execution: inline, as requested by the user. Use executing-plans, test-driven-development and verification-before-completion. No commits or physical device actions.

Goal: build three shippable profiles, live microphone action execution at configurable 0.95 intent / 0.90 per-slot confidence, shared private/public configuration, and integrated debug UI.

Spec: [SPEC-speaker-publication-v2.md](SPEC-speaker-publication-v2.md). User's subsequent instruction supersedes the proposed 0.80 default: intent 0.95, each slot 0.90. Inference is the shipping application; benchmark produces Raspberry Pi evidence.

Architecture: canonical publisher chooses explicit profile inventories. Python resolves root TOML/dotenv configuration and supplies the gateway. Frozen recognition output remains unchanged; application decisions and action outcomes are additional fields.

## Constraints and review focus

- Preserve dirty source, installed graphs, frozen policies, source receipts and datasets. Use root interpreter and existing tools. Work in current authorized checkout; a clean HEAD worktree would omit the active untracked runtime.
- Credentials never enter bundles, browser responses, or logs. Editable configuration does not weaken immutable code/model checks.
- Slot confidence is per-value finite-catalog acoustic competition, not an empirical correctness guarantee. Duplicate aliases must not inflate confidence.
- One server dispatcher processes each utterance once, independently of browser subscriber count. WAV and synthetic fixtures never execute provider actions.
- Benchmark/full include verified evaluation data and both recorded and guided Pi test entry points. Inference excludes corpus payloads.

## Tasks and progress

- [x] Configuration and confidence: tests in `scripts/test/test_speaker_publication.py`; implement `speaker_config.py` resolver and template, `speaker_confidence.py` scoring and demo decision; attach diagnostics in `pcv_final_runtime.py` without changing frozen decision outputs. Test defaults, precedence, invalid input, alias invariance, intent and all-slot boundaries, no-slot intents and negative/capture rejection.
- [x] Publication and startup: extend `speaker_publish.py`, `pcv_final_delivery.py`, `speaker_start.py`, `speaker_package.py`; test profile inventories, data independence, mutable config and model updates. Include full recorded benchmark and guided microphone scripts/report generation; preserve standalone aliases.
- [x] Actions and capture: worker emits independent application decisions; gateway dispatches validated commands through existing providers, explicitly reports unsupported capabilities, deduplicates utterances, and exposes sanitized runtime errors. Test source/threshold gating, missing mappings, action disable, unsupported and duplicate requests with mocked providers.
- [x] UI: move RuntimeDebug inside existing debug panel, add bottom-left live readout, keep full history inspectable, perform local timer/alarm/reminder actions once, and show mode/threshold/action outcomes. Update relevant event/browser tests for intentional behavior changes.
- [x] Verify/deliver: focused Python tests, UI type checks/offline tests/optimized build; browser checks at 800x480 and 1024x600; generate all three fresh archives, verify hashes/inventories, extracted replay/benchmark entry points and write durable receipts. Review changed files and fix material findings. Record physical Pi evidence as pending.

## Execution ledger

- Initial investigation: current launcher hardcodes actions off, gateway repeats that override, inference view floats at bottom right, UI hides device-list exceptions, existing package always includes samples. Model provides intent probability and full finite-command acoustic scores but no slot-confidence head.
- Ruling: implement slot confidence by restoring sequence log likelihood from per-frame scores, taking the best alias per distinct slot value and normalizing competing values within the predicted intent. Each slot must independently pass. This is catalog-conditional confidence and is documented as uncalibrated.
- Ruling: user's explicit instruction to perform changes authorizes implementation now; proceed inline without another design approval cycle. Preserve the approved threshold and packaging intent in the spec.

- Engine choice: every profile ships ncnn (default, selected FP32 checkpoint) and ONNX Runtime (existing INT8 graph). No fallback. Source publication installs export-only torch/pnnx; neither is included in runtime ZIPs.
- Review fixed benchmark-profile WAV precedence, configurable server transport/origin/TLS, and missing exporter bootstrap dependencies. Regression tests cover explicit benchmark-profile inference, connection metadata, and environment binding overrides.
- Repeated real-audio inference found a native input-buffer ownership bug missed by the initial random-input parity check. Keeping both NumPy buffers and native Mats alive through output extraction fixed it. Earlier `publication-v2` and `publication-v2-release` archives and initial replay reports are superseded negative evidence; **use only `dist/speaker/publication-v2-shipping/`**.
- After the fix: 90 extracted-runtime inferences (15 recorded files, three alternating-order repeats, two engines) were exactly stable within each engine. Candidate agreement was 15/15; this does not promise general FP32/INT8 equivalence. ncnn vs selected PyTorch real-audio tensor maximum absolute error was 9.7752e-6 over 15 files. Eight synthetic feature lengths (1–1499) also passed with maximum error 1.0491e-5 at tolerance 2e-5.
- Offline software verification: 69 Python tests plus three subtests; 88 JavaScript tests; Svelte/TypeScript checks with zero errors/warnings; optimized production build. Browser verification: 22 checks, including actual WAV replay at 800x480/1024x600, bottom-left readout/history navigation, and all 18 typed command fixtures. Provider integrations were disabled; no physical actions were performed.
- Every shipping archive inventory and runtime source was checked. Both engines are present, benchmark data/UI exclusion matches profile, no real .env, editable config.toml, or PyTorch checkpoint is shipped, mutable config changes pass immutable-file verification, and all 20 baseline installed artifact hashes remain unchanged.

## Reproduction and receipts

Fresh publication: `python publish.py --no-setup --benchmarks skip --target all --output dist/speaker/NEW_DIRECTORY` in the configured root environment. Omit `--no-setup` for dependency bootstrap. Selected checkpoint and original ONNX identities are in each `models/default/ncnn/engine.json`; package hashes are in release.json.

Focused tests: `.venv/Scripts/python.exe -m pytest scripts/test/test_speaker_publication.py scripts/test/test_speaker_start.py scripts/test/test_speaker_worker.py scripts/test/test_pcv_full_benchmark.py scripts/test/test_pcv_final_delivery.py`.

Repeat check: `.venv/Scripts/python.exe -m scripts.test.speaker_engine_replay --package EXTRACTED_FULL_ZIP --output NEW_RESULT.json`.

Raw task receipts: `.cache/speaker-publication-v2/` (shipping-archive-checks.json, shipping-engine-replay.json, real-audio-fp32-parity.json, browser-stable/, ncnn-stable-full-replay/). Earlier directories lacking `stable` or `shipping` are superseded where the engine lifetime defect applies. Desktop timing includes concurrent package/browser activity and is not an isolated performance comparison or Pi measurement.

Final corrected full ncnn replay completed all 6768 recordings. Desktop one-logical-CPU inference p50/p95: 44.484/115.600 ms; whole-process high-water RSS 130.34 MiB. Recorded reports and CSV generation passed. These are desktop measurements with other task activity, not Pi results. Frozen acceptance remained disabled. Demo-threshold replay is a separate postprocessing receipt, with no device execution or threshold tuning.

Durable compact receipt: [verification.json](../experiments/speaker-publication-v2/verification.json). All five implementation items are complete. Pi/live microphone/provider verification remains a deployment step. No commits or installed-model replacements were made.
