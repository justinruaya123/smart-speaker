# Iris fresh-clone reproduction

This engineering verification addresses ME-01/02/03/04/06/07/08. Its hypothesis is that a clean GitHub checkout can install pinned dependencies, obtain permitted data from immutable Hugging Face revisions, actually run the six frozen VCM continuation fits, evaluate them and reconstruct the selected inference delivery without the development machine's caches. It changes no acoustic vocabulary, split, loss, calibration or promotion rule. The [protocol](../experiments/iris-reproduction-v1/protocol.json) is frozen before this run.

GitHub main owns the recipe, executable sources, reports and selected inference bundle. Hugging Face holds data and private model/checkpoint artifacts. Original restricted recordings remain at their permitted upstream sources. Upstream benchmark code is fetched from its pinned GitHub commit and hash-checked; it is not relicensed or mirrored as project code.

`python reproduce.py --device cuda` is the DGX training and benchmark entry point; choose one available GPU with `CUDA_VISIBLE_DEVICES`. CPU is the default. The command installs the root environment and creates a unique ignored source workspace. Completed VCM fit checkpoints and evaluation caches are deliberately absent there. Published project initializers and the fixed fallback are inputs; wake and speaker checkpoints are verified companions. The original Git checkout and historical results remain unchanged. `--skip-train` is an explicitly different selected-artifact reconstruction.

The complete historical initializer chain has not been replayed. A test veto can retain the already trained baseline; that outcome must be reported separately from the newly trained candidate. No result here establishes fresh human ownership, microphone or physical Raspberry Pi acceptance.

Execution, measured timings, artifact comparisons and fresh-clone evidence will be recorded below after the run.
