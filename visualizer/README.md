# Signal Lab — actual VCM tensor inspection

Restore the supplied payload, then serve the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://localhost:8000/visualizer/`. No CDN, npm install, external font, speech
service or pretrained model is required. `VCM_SIGNAL_LAB.html` is the downloadable
self-contained build. Browser microphone support needs localhost/HTTPS and permission.

## What runs

The frontend computes real log-mel features in JavaScript. `kernel.c` supplies
float32 causal convolution and double-precision log-domain CTC in WebAssembly.
`weights.bin` contains the own research checkpoint's exact float32 weights.
`model.onnx` remains the reference artifact; tests independently execute it with
OpenCV CPU and compare it with WASM and PyTorch.

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

```bash
python visualizer/build_kernel.py
python visualizer/export_assets.py
python visualizer/build_standalone.py
node --test visualizer/tests/core.test.mjs
node visualizer/tests/run_parity.mjs
python final_suite/verify_math.py
python final_suite/browser_check.py
```

Clang with wasm32 is needed only to rebuild the included kernel. These commands
never download a pretrained model. Browser component tests execute local fixtures;
physical GPU, WebGPU and microphone tests are separate. The 32 MiB WASM arena and
DOM/graphics/evidence memory are not the model's 32,064-byte neural cache. Run the
visualizer on a development computer, separately from the Pi inference process.

## References

Interaction inspiration: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/).
No code, model, figure or asset was copied from it. Renderer, UI, frontend and WASM
kernels were written for this project.

API references: [WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly),
[AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet), and
[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).
See `THIRD_PARTY.md` for offline training/export/verification dependencies.
