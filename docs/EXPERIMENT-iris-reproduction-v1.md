# Iris fresh-clone reproduction

This engineering verification addresses ME-01/02/03/04/06/07/08. Its hypothesis is that a clean GitHub checkout can install pinned dependencies, obtain permitted data from immutable Hugging Face revisions, actually run the six frozen VCM continuation fits, evaluate them and reconstruct the selected inference delivery without the development machine's caches. It changes no acoustic vocabulary, split, loss, calibration or promotion rule. The [protocol](../experiments/iris-reproduction-v1/protocol.json) is frozen before this run.

GitHub main owns the recipe, executable sources, reports and selected inference bundle. Hugging Face holds data and private model/checkpoint artifacts. Original restricted recordings remain at their permitted upstream sources. Upstream benchmark code is fetched from its pinned GitHub commit and hash-checked; it is not relicensed or mirrored as project code.

`python reproduce.py --device cuda` is the DGX training and benchmark entry point; choose one available GPU with `CUDA_VISIBLE_DEVICES`. CPU is the default. The command installs the root environment and creates a unique ignored source workspace. Completed VCM fit checkpoints and evaluation caches are deliberately absent there. Published project initializers and the fixed fallback are inputs; wake and speaker checkpoints are verified companions. The original Git checkout and historical results remain unchanged. `--skip-train` is an explicitly different selected-artifact reconstruction.

The complete historical initializer chain has not been replayed. A test veto can retain the already trained baseline; that outcome must be reported separately from the newly trained candidate. No result here establishes fresh human ownership, microphone or physical Raspberry Pi acceptance.

Execution, measured timings, artifact comparisons and fresh-clone evidence will be recorded below after the run.

## Download failure and correction

The first exact-main attempt at `3d17eddd79880d26e0c8eebe49d36fed02aaeb74` installed an independent CPU environment, then stopped **before fitting** after 1,381 designed recordings: the pinned Hub client exhausted the Xet read-token API quota (HTTP 429, 1,000 requests per five minutes). No candidate was trained or selected in that attempt. The [failure receipt](../experiments/iris-reproduction-v1/preflight-failure-xet.json) preserves source identity and denominators.

The corrected recipe sets the client's documented [HF_HUB_DISABLE_XET](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubdisablexet) option before importing it. A real HTTP probe retrieved the same pinned 38,444-byte recording and matched its declared SHA-256. Immutable Hub objects are cached within a clone; run workspaces and VCM fitting states remain separate. Data, splits, normalizer, grammar, losses, seeds, selection and test gates are unchanged. The failed attempt remains preserved.

The initial installer redirect failure was also corrected by installing the same checksum-pinned uv 0.12.5 wheel from official PyPI. A separate cold dependency probe took 556.04 seconds; the exact-main clone installed its own environment in 186.47 seconds using only that probe's newly downloaded dependency wheels. Neither used a development model/audio cache.
