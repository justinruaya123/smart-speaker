# Signal Lab — actual VCM tensor inspection

Generate an inspector from the selected model, then serve the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://localhost:8000/visualizer/` when `assets/` has been exported, or open
the generated standalone HTML. No CDN, npm install, external font or speech
service is required. Browser microphone support needs localhost/HTTPS and permission.
The PCV synthetic study is still running; an artificial fixture is not the selected
trained model. Current architecture and evidence status: [PCV model](../docs/PCV-MODEL.md).

## What runs

The frontend computes real log-mel features in JavaScript. `kernel.c` supplies
float32 causal convolution and double-precision log-domain CTC in WebAssembly.
`weights.bin` contains the own research checkpoint's exact float32 weights.
`model.onnx` remains the reference artifact. The PCV exporter compares the checkpoint
with CPU ONNX Runtime using nonzero caches; `tests/check_pcv.py` compares the real
WASM/frontend tensors with the same ONNX graph on a specified waveform.

The scene draws real activation samples with WebGL, or projected 3D Canvas when
WebGL is unavailable. This is GPU rendering, **not WebGPU neural inference**.
It is not ONNX Runtime Web and not a generic ONNX loader. No learned neural layer
was added just for visualization.

## Controls

Load a supplied waveform or a mono 16 kHz PCM16/float32 WAV. Inputs over 7.8 seconds
are rejected, reserving the fixed 100 ms inspection tail inside an eight-second
bound. `prepare_wav.py INPUT OUTPUT` resamples a recording once outside streaming.

Run/Pause processes chunks. Step layer executes one real convolution; Step chunk
completes a neural/CTC update. 0.25–3x changes the scheduler, not PCM, pitch or model
inputs. Compute can limit the effective display cadence. Partial numbers never
execute actions. Select layers, channels and frames to inspect all tensor values,
min/max, cache sizes, bias, and the largest measured input-times-weight terms.
The 3D bars sample channels; heatmaps/downloads contain full tensors.

Live microphone code uses AudioWorklet, a 16 kHz context, energy onset, preroll,
400 ms silence endpoint and an eight-second bound. It rejects sequence gaps,
sample-rate changes and excessive backlog. It always runs at 1x. Physical microphone
behavior, including pauses inside numbers, is not yet qualified. Audio is not uploaded.

## Model hot-swap

Select compatible recipe JSON, float32 weights and compiled catalog together.
The optional ONNX file is hash checked against the recipe. The recipe binds layer
shapes, offsets, state lengths, vocabulary, normalization and filter coefficients.
Supported layers are bounded causal stride-one Conv1d/ReLU chains. Unsupported
architectures are rejected, not approximated. Weights and the entire catalog body
are hashed; swaps reset frontend, neural caches, CTC state and clock.

All returned decisions remain `needs_clarification`. Catalog matching can be wrong
on out-of-list names; the largest CTC score is not a calibrated probability or
permission to act. No Home Assistant/Spotify calls occur here.

## Rebuild and verify

Once the selected portable bundle and checkpoint exist:

```bash
python visualizer/export_assets.py --pcv-runtime --bundle <portable-bundle> --checkpoint <selected.pt> --output visualizer/assets --evidence <demo-results.json>
python visualizer/build_kernel.py
python visualizer/build_standalone.py
node --test visualizer/tests/core.test.mjs visualizer/tests/evidence.test.mjs
python visualizer/tests/check_pcv.py --module-dir visualizer --assets visualizer/assets --wav <mono-16k.wav> --output <new-parity-receipt>
```

Export into a new directory: existing assets and receipts are not overwritten.
Build the kernel after exporting when using the same assets directory. Alternate
locations are supported by `build_kernel.py --output-dir ... --module ...` and
`build_standalone.py --assets-dir ... --kernel-module ... --output ...`.
The kernel compiler can be Clang with wasm32 support or `--compiler <zig-executable>`.
These commands never download a pretrained model. Browser component tests execute local fixtures;
physical GPU, WebGPU and microphone tests are separate. The 32 MiB WASM arena and
DOM/graphics/evidence memory are not the model's 32,064-byte neural cache. Run the
visualizer on a development computer, separately from the Pi inference process.

The recorded-results table verifies its evidence file hash plus source/deployment
bindings. A new PCV model cannot silently display the historical V4/V5 table.
Changing the model clears old results. Empty sample lists are supported: load a
WAV explicitly. Browser microphone inspection uses its own diagnostic energy
endpoint; authoritative acceptance uses the packaged Hey Jarvis + CPU ONNX pipeline.
The older `final_suite/verify_math.py` checks historical V3 assets and is not the
selected PCV parity command.

## References

Interaction inspiration: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/).
No code, model, figure or asset was copied from it. Renderer, UI, frontend and WASM
kernels were written for this project.

API references: [WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly),
[AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet), and
[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).
See `THIRD_PARTY.md` for offline training/export/verification dependencies.
