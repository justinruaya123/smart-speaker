# Desktop baseline A: one-command rebuild and replay

From a fresh checkout of `smart-speaker/main` on the Windows RTX 3080 host, `python all_in_one.py` must prepare the pinned scientific Python environment, create a fresh run directory, invoke the existing frozen baseline-A `all` pipeline (original sources, real training, evaluation and export), and only on success invoke the repository's `inference.py` on the published brightness WAV with the **newly exported** graph. The caller needs Git, Python, uv, network, a compatible CUDA driver and enough local disk. Do not overwrite or silently resume previous runs. Print the run path, exact commit and verification result; fail nonzero if a stage or expected command fails.

## Approaches

- **Chosen:** rebuild from random initialization on the RTX 3080; compare the model and decisions, then exercise the desktop command-line decoder. This tests reproduction, not just package usability.
- **Not chosen:** replay the bundled weights without fitting. Useful as a quick inference check, but cannot establish training reproduction.

## Acceptance and boundaries

Require the existing scientific gates (48 epochs/4,608 updates, selected epoch 38, matched tensors, 642 matching decisions) and an explicit SHA-256 match of the new ONNX export to the retained graph. Replay `docs/site/bmo/data/sample.wav` through the new graph using baseline A's verified preprocessing, command order and decoder; require `BRIGHTNESS`, `light.all`, 50%. Keep `inference.py` independently usable from the CLI with a WAV. No UI, Pi variant, device actions, microphone capture or SpeakerUI imports. Narrow the producer boundary test to permit this offline model CLI, not the application runtime. Push only the workflow, tests and documentation to producer main; leave unrelated dirty files, training data and generated artifacts out. For a future clean-clone test, clone the remotely verified commit on a disk with enough space and run the one command above. The delivery request explicitly defers that training run; publication is not itself a fresh fit.
