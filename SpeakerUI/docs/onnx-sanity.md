# First real ONNX sanity check

Status: **not run; real service contract and model fixtures are not supplied.** `pnpm inference:sanity` exits 2 deliberately. The transport mock produces a fixed intent for any nonempty recording, even silence; it cannot establish model correctness.

`pnpm run setup` installs the native `onnxruntime-node` CPU package and verifies that it imports. Keep supplied models and supporting files under the ignored `models/` directory (`ONNX_MODEL_DIR` in `.env.example`). Importing the runtime does not validate model loading or inference.

Before implementing the real adapter, obtain:

- Service protocol, authentication, endpoint, audio encoding/rate, message examples, finality rules, cancellation, and error behavior.
- Model identity/version/checksum, label map, ONNX Runtime version/provider, tensor names/shapes/dtypes, preprocessing and decoding rules. ONNX Runtime itself does not define this application's network API.
- One short known-command recording plus expected intent and parameters; one silence fixture and one non-command/noise fixture. Keep recordings outside Git unless explicitly approved.
- Score semantics (logits, normalized score, calibrated probability, or another measure) and whether parameter confidence is available. Do not interpret a raw logit or ASR score as intent confidence.

With actions disabled, perform only the first sanity pass:

1. In the owning inference service, load the exact model once using the CPU provider. Verify tensor metadata and preprocessing against the training/export contract.
2. Execute a known input and verify finite outputs, expected dimensions, label decoding, intent and required parameter values. A successful model load alone is insufficient.
3. Send the known-command, silence, and noise fixtures through the actual service adapter. Silence/noise must not become an accepted actionable command. Verify normalized events and utterance IDs; partial results must remain provisional.
4. Record model/runtime versions, fixtures, expected and observed results. Keep `ACTIONS_ENABLED=false`. Fail explicitly if any prerequisite is unavailable.

Only after sanity passes, measure Pi inference RSS separately from gateway/browser RSS, warm inference latency, end-of-utterance-to-final latency, and repeated-session stability. Start with one persistent model session and bounded concurrency. Compare thread settings and INT8 against the original model using held-out commands, noise and confusing intents. Set `INTENT_MIN_CONFIDENCE` from those results; no universal threshold is implied by the diagnostic mock's 0.9 cutoff.

References: [CPU/ARM64 support](https://onnxruntime.ai/docs/get-started/with-javascript/node.html), [thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
