# VCM final review and Signal Lab

This separate research branch leaves the production notebook, training scripts,
pipeline configuration and artifacts unchanged.

**Reference:** `phone_s53`, one causal 64-channel CNN, 113,064 parameters,
458,230-byte FP32 ONNX and 32,064-byte neural cache. **No release winner.**
The model performs constrained phonetic recognition, not unrestricted transcription.
V6 remains uncalibrated and executes no actions. No new acoustic training was
performed while finalizing this suite.

## Deliverables

`VCM_Final_Review.ipynb` contains TL;DR, Mermaid architecture/dataset diagrams,
exact shapes, sample payloads, all ten original application use cases and their
verbatim/expected/actual outputs, the separate legacy ten-label contract, and
exhaustive command ledgers. The Git edition is compact; restore the payload to
execute it. The accompanying download also provides an executed self-contained
notebook with all tables embedded.

`visualizer/` is a real tensor inspector: JavaScript acoustic frontend, WASM CNN
and CTC, real intermediate activations, channel/time probes, weighted contribution
inspection, waveform stepping and 0.25–3x inspection speed. WebGL is rendering,
not WebGPU inference. Canvas 3D is the fallback. Compatible recipe/weights/catalog
bundles are hot-swappable; this is not an arbitrary ONNX parser. The original
ONNX graph is preserved and independently used for parity checks.

## Restore and run

Download `VCM_FINAL_PAYLOAD.zip` supplied with this review and verify its checksum.
From a checkout of `research/vcm-final-suite-visualizer`:

```bash
python final_suite/restore_payload.py /path/to/VCM_FINAL_PAYLOAD.zip --repo .
python final_suite/restore_payload.py /path/to/VCM_FINAL_PAYLOAD.zip --repo . --apply
python -m pip install -r final_suite/requirements.txt
python final_suite/run_all.py --browser
python -m http.server 8000 --bind 127.0.0.1
```

The first import is a dry run. Production root files are outside the allowlist;
conflicting binary/data assets are never overwritten. Large models, waveforms,
archived research code and generated evidence travel in the ZIP instead of being
silently committed through the original repository's ignore rules.

Open `http://localhost:8000/visualizer/`. The supplied `VCM_SIGNAL_LAB.html` is
also a self-contained offline build. Microphone use requires a supported secure
context and explicit permission; it remains physically unverified.

```bash
python final_suite/run_all.py --qualify
```

This deliberately exits 2 while release evidence is missing. Software success is
not recognition quality. For notebook regeneration after evidence rebuilding:

```bash
python final_suite/review.py
python final_suite/dataset_audit.py
python final_suite/refresh_notebook.py
```

The executed copy is written under `final_suite/reports/`; it does not overwrite
production code or silently select/train a model.

## Measured results and limitations

- 397 Python tests, 18 JavaScript tests, and eight Chromium UI scenarios passed.
- Actual WASM execution on seven WAVs was compared with full PyTorch and serialized
  ONNX executed by OpenCV CPU. No changed frame-argmax paths, commands or ties.
- The 44-waveform full-scope diagnostic matched 10/40 positive requests, with four
  other negative/ambiguous inputs. Unsupported families remain explicitly shown.
- V6's narrower historical in-grammar task matched 645/706 raw commands. Neither
  figure is accepted recall: the V6 acceptance policy refuses every final command.
- The ledger contains 36,972 rows from 79 reports, including training/development,
  failed alternatives and repeated evaluations. It records 3,566 incorrect raw
  positive rows, 251 wrong diagnostic accept rows, 24,830 correct raw rows and 8,576
  negative/ambiguous rows. Categories overlap; do not pool them into one accuracy.
- Only 19,118 rows retain PCM hashes, representing 6,036 distinct recorded hashes.
  Most historical WAVs are absent. Archived predictions are not new waveform tests.
- All 48 supplemental V3 test IDs were resolved through the OOV manifest.

All acoustic rows are synthetic. Inspected tests are regression sets, not sealed
selection sets. The browser test harness used exact local in-memory assets and
Canvas 3D because navigation/WebGL were restricted. Physical mic, hardware GPU,
WebGPU compute, ONNX Runtime Web, and Pi4 remain unqualified. The visualizer's
32 MiB WASM arena plus DOM/graphics is separate from the 32 KiB neural cache; keep
the viewer on a development computer, not inside the Pi inference daemon.

The GitHub workflow checks source contracts only and explicitly skips unavailable
payload evidence. It must not be presented as acoustic or Pi qualification.
