# Final submission integration — 2026-10-17

Purpose: give a professor three portable entry points and inspectable evidence for the user-selected 2,066,831-parameter final-refinement model, retaining the 935,567-parameter comparison. The user changed the shipping choice to 2M during implementation. This is additive submission work, not a repository maintenance window. Existing protocols, installed artifacts, running experiments and user changes remain preserved.

The user's requested design is the implementation brief. `reproduce.py` bootstraps uv, reconstructs pinned inputs in a separate destination, runs the original seed-53 recipe, freezes development selection before test, compares regenerated weights/results and renders training curves plus the submission card. An optional seed creates a separately identified run. CUDA fitting is seeded but has no bitwise guarantee. A smoke run cannot establish full reproduction.

`inference.py` bootstraps the minimal CPU runtime and offers microphone/speaker debug CLI without speech playback, or the existing compiled SpeakerUI and native runtime. The submitted ONNX and pretrained wake/VAD graphs are directly inspectable. Fixed response clips are rendered ahead of time, stored separately on the Hugging Face inference branch, and downloaded without training data or a runtime synthesizer.

`benchmark.py` asks for local PC or Raspberry Pi SSH address, transfers only the runtime and pinned evaluation audio, bootstraps dependencies remotely, silently scores recordings and retrieves immutable results. Hardware identity, timing scope, thread count, graph hashes, raw/demo/qualified decisions, UNKNOWN denominators and human/synthetic results remain explicit. It does not measure wake or microphone performance. The user will operate the physical Pi from another PC.

Writable scope: the three entry points, focused `scripts/submission/` helpers/tests, `submission/` evidence/assets, flat submission documentation, an additive README entry, scoped ignore exceptions and an inference ZIP under ignored `dist/`. Shared model/training code is reused without modifying frozen implementations. New release packaging delegates to the existing publisher where compatible and preserves its integrity format. Public release preparation uses explicit file inventories; private credentials, captures, audio datasets and cache paths are excluded.

Execution and verification:

- [x] Deliver and test the portable PC/SSH benchmark first; immediately notify the user when ready.
- [x] Export sanitized row-level reconstruction maps, histories, selected checkpoint, graphs and source/environment identities.
- [x] Implement reconstruction, original fitting, development selection, export/calibration and locked test in a fresh source/output folder; record actual comparison and separate fresh uv setup verification.
- [x] Render deterministic responses and prepare the Hugging Face inference branch independently of its existing main branch.
- [x] Build and test CLI/SpeakerUI launchers and an inference-only ZIP containing compiled UI and production dependencies.
- [x] Render evidence-bound README graphs/card, verify extracted source/inference packages and record unresolved public access, DOI, licensing and physical-device requirements.

Verification includes meaningful failure-path tests for corrupt/missing assets, unsafe SSH targets, existing-output preservation and false hardware claims, actual recorded ONNX inference, clean-folder execution and archive inventory checks. No physical Pi result, exact regeneration, public release URL or dataset DOI is invented. Only the required ignore-first change is committed automatically; public publication is the final reviewable step.
