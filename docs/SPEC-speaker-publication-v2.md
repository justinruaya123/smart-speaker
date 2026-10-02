# Speaker publication v2

Status: implemented and desktop/package verification complete, 2026-10-02. User authorized shipping packages, microphone demo execution, 0.95 intent / 0.90 every slot, and both engines with ncnn default.

## Agreed outcome

Publish one release directory with three independently runnable ZIPs built from the same selected model. Put inference controls inside the existing debug panel, retain a compact bottom-left inference readout, and share one configuration system between the Python runtime and SpeakerUI.

The user explicitly selected demo voice execution with a configurable threshold. Inference and full packages therefore default to actions enabled and demo execution. This is an application execution policy; it does not change the model's frozen qualification policy, benchmark results, or qualification status.

## Package profiles

Default publication creates `dist/speaker/<release-id>/` containing:

| Archive | Included | Default entry behavior |
| --- | --- | --- |
| `smart-speaker-inference.zip` | Selected command model and required metadata, wake/VAD models, inference runtime, compiled SpeakerUI/gateway, dependency locks, config templates, notices | `python3 start.py` serves SpeakerUI and inference |
| `smart-speaker-benchmark.zip` | Same model and CLI inference, benchmark runner, pinned evaluation recordings, expected outputs, manifests and source/license receipts | `python3 start.py` runs the CLI benchmark |
| `smart-speaker-full.zip` | Union of inference and benchmark profiles | `python3 start.py` serves SpeakerUI; benchmark runs explicitly |

The inference profile excludes benchmark recordings, benchmark report generators, corpus manifests and submission slides. Small sample-command text fixtures remain available in the debug panel. The benchmark profile excludes SpeakerUI and Node. Required model metadata and third-party notices are included in every profile.

Use explicit inventory selection in the canonical publisher before reading benchmark payloads. Inference publication must work without the benchmark corpus present. Benchmark/full publication fails clearly when required pinned data is unavailable; neither silently downloads substitute data nor emits an incomplete success ZIP.

Expose profile selection through the existing launcher and publishing entry points; default is all three profiles. Keep old target options as documented compatibility aliases where their meaning is unambiguous (`standalone` to benchmark, `pi` to full, `both` to all). Retain immutable release directories and ZIPs. Write a compact release receipt listing profile, archive size/hash, model identity and entry command.

## Shared configuration

Each extraction uses root `config.toml` for ordinary settings and root `.env` for private settings/overrides. Both Python and the gateway consume one resolved configuration; do not independently interpret divergent defaults.

Precedence: explicit CLI options > existing process environment > root `.env` > root `config.toml` > documented profile defaults. CLI defaults must not accidentally override file values. Empty, malformed, unknown and out-of-range settings receive field-specific validation. Relative paths resolve from the installation root.

The canonical Python launcher parses TOML and dotenv without executing shell expressions and forwards a validated configuration to the gateway. Provider credentials remain server-side. Browser health/debug responses expose only an explicit public subset. Environment substitution and command execution are not part of dotenv parsing.

Configuration groups:

- Inference: engine selection (ncnn default or onnxruntime), CPU threads, execution mode, intent and per-slot confidence thresholds.
- Audio: microphone index; existing wake/VAD pipeline remains fixed.
- Server/UI: host, port, origin, debug panel and bottom-left readout visibility.
- Actions: enabled flag, provider enablement, explicit model-slot-to-provider entity/media bindings.
- Benchmark: output directory and CPU affinity; CLI arguments select alternate validated datasets.
- Providers: ordinary URLs/IDs in TOML if desired; tokens, client secrets and private overrides in `.env`.

Ship immutable `config.example.toml` and `.env.example`. On first startup create `config.toml` from the example only if absent. Editable config is separate from immutable manifest entries, so changes do not break installation verification. Validate configuration every startup. Never ship a real `.env` or overwrite an existing private file.

For existing installations, detect a legacy `SpeakerUI/.env` and give an explicit migration message when root `.env` is absent. Preserve the legacy file; do not silently combine conflicting files. Model-only updates preserve root configuration and credentials. Runtime/UI changes require a complete new extraction.

## Demo execution and command routing

Modes are `demo` and `qualified`. Inference/full defaults: `actions.enabled=true`, execution mode `demo`, minimum intent confidence `0.95` and every-slot confidence `0.90`. This initial threshold is a configurable application default, not a measured guarantee of command correctness. Benchmark execution does not read application demo thresholds and never dispatches actions.

Retain the frozen model output verbatim. Add an application decision and execution outcome rather than replacing its rejection with a claim of qualified acceptance. Qualified mode uses only the model's accepted typed command. Demo mode can use the typed diagnostic candidate even when the frozen output is `unqualified_model`.

Demo checks require a finite candidate, valid exact command schema and slots, finite scores, agreement between candidate intent and winning intent head, and intent probability at or above the configured threshold. Preserve existing named/all-light target consistency checks. Every predicted slot must independently meet its threshold. Null candidates, capture faults, invalid arguments, unresolved bindings and unsupported actions do not dispatch.

Display intent probability as intent confidence and command score/margin by their actual names. Neither is described as a calibrated probability of the entire command being correct.

Route finalized live microphone commands through one application dispatcher with a unique utterance ID. Prevent duplicate execution on repeated events, multiple browser clients, reconnects and replayed history. Provider actions execute once on the server. Use existing HA and Spotify implementations with explicit target/media bindings; do not guess a provider entity from a friendly model name. Local UI operations use the existing handlers with equivalent event deduplication. Missing capabilities return `unsupported` or `not_configured`, never fabricated success.

WAV replay and sample-command fixtures remain diagnostic by default and report their source. They do not silently trigger provider actions. Live microphone commands are the default demo execution source. Providers require their configured credentials/enablement; an enabled action flag alone does not invent a connection.

Every result records the model decision, application mode/decision, command, confidence, source and execution outcome. Changes to settings take effect after restart in this version; no settings editor is required.

## Inference UI

Add an Inference option alongside sample commands and aura controls in the existing debug surface. Move microphone discovery/selection, Listen/Mute, WAV replay, errors, mode/threshold summary and detailed JSON there. Remove the separate bottom-right inference menu.

Show compact bottom-left text on the main screen: latest intent, typed slot command, intent confidence and application execution status. Keep full score/margin/model rejection details in the debug panel/history. Use the existing receipt area where practical, preserving touch controls and avoiding overlap with call-resume controls and notifications. Receiving a voice result must not force the full history view open.

Populate microphones when opening Inference and retain an explicit refresh action. A successful empty list gets a clear no-input-devices message. Worker/audio failures surface a bounded actionable message rather than `internal_error`; provider credentials and unrelated exception details must not leak. Report whether failure occurred during device discovery or stream opening. The current Pi failure remains undiagnosed until its device-list output is available.

Maintain the existing palette, aura, gestures and layout at 800 x 480 and 1024 x 600. No new UI framework or animation dependency.

## Implementation scope and evidence

Modify the existing publisher, launcher/configuration, worker/gateway event routing, debug/readout components, focused tests and current documentation. Preserve other agents' dirty changes. This authorization changes application defaults and package structure; it does not authorize new training, dataset migration, a model qualification claim, commits or physical device operations.

Validation must cover:

1. All three ZIP inventories, matching model hashes, archive verification and extracted startup paths. Inference build succeeds without benchmark data; benchmark/full contain their declared pinned data.
2. Configuration precedence, field validation, default profile behavior, editing config without integrity failure, private-file exclusion and preservation during updates.
3. Demo threshold boundaries, invalid/null candidates, capture faults, qualified mode, actions disabled, missing bindings, provider outcomes and duplicate-event suppression. Use mocked providers, never live devices.
4. Benchmark execution remains independent of demo settings and produces the original model acceptance results.
5. Microphone listing success/empty/error cases; WAV replay does not execute actions; detailed failures reach the debug UI.
6. SpeakerUI type checks, offline tests and optimized build; browser layout/interaction checks at both display sizes, including bottom-left readout, debug navigation, full history and retained animation. Report desktop resource measurements as desktop evidence.

Deliver source changes, fresh package paths/sizes/hashes, verification results and explicit remaining Pi/live-microphone limitations. Existing physical-device performance and the user's current Pi capture failure must not be reported as fixed solely from desktop checks.

## Design review

The three package inventories, defaults, editable configuration boundary and application-versus-model decisions are explicit. Publication reuses the existing model exporter. The approved 0.95 intent / 0.90 slot thresholds are application settings and can be changed without rewriting qualification evidence. Missing action capabilities remain visible. No pending architecture alternative is required to begin implementation planning after user review.

## Engine implementation

All three archives contain the selected INT8 ONNX graph and an ncnn FP32 conversion of the same selected checkpoint, plus a conversion/hash/parity receipt. The ncnn dynamic-length rewrite is checked against the original PyTorch model at eight lengths (1–1499 feature frames). Neither engine falls back to the other. Wake/VAD retain their ONNX Runtime sessions. Export dependencies (PyTorch and pnnx) are workstation-only; runtime ZIPs contain neither. Native ncnn CPU packing is disabled for the verified numerical path. Physical ARM performance remains to be measured.

The ncnn conversion is FP32, so identical outputs to the INT8 engine are not promised. Reports label the engine and its actual model hash. Conditional finite-catalog slot probabilities are not a calibration guarantee. Model qualification remains unchanged.

Implementation sources: [official ncnn conversion guidance](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md) and [pnnx](https://github.com/pnnx/pnnx). Runtime version is pinned to ncnn 1.0.20260526, with an ARM64 CPython 3.12 wheel present in uv.lock. The wheel availability check does not replace physical Pi testing.
