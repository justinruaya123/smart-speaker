# Iris fresh-clone reproduction

This engineering verification addresses ME-01/02/03/04/06/07/08. Its hypothesis is that a clean GitHub checkout can install pinned dependencies, obtain permitted data from immutable Hugging Face revisions, actually run the six frozen VCM continuation fits, evaluate them and reconstruct the selected inference delivery without the development machine's caches. It changes no acoustic vocabulary, split, loss, calibration or promotion rule. The [protocol](../experiments/iris-reproduction-v1/protocol.json) is frozen before this run.

GitHub main owns the recipe, executable sources, reports and selected inference bundle. Hugging Face holds data and private model/checkpoint artifacts. Original restricted recordings remain at their permitted upstream sources. Upstream benchmark code is fetched from its pinned GitHub commit and hash-checked; it is not relicensed or mirrored as project code.

`python reproduce.py --device cuda` is the DGX training and benchmark entry point; choose one available GPU with `CUDA_VISIBLE_DEVICES`. CPU is the default. The command installs the root environment and creates a unique ignored source workspace. Completed VCM fit checkpoints and evaluation caches are deliberately absent there. Published project initializers and the fixed fallback are inputs; wake and speaker checkpoints are verified companions. The original Git checkout and historical results remain unchanged. `--skip-train` is an explicitly different selected-artifact reconstruction.

The complete historical initializer chain has not been replayed. A test veto can retain the already trained baseline; that outcome must be reported separately from the newly trained candidate. No result here establishes fresh human ownership, microphone or physical Raspberry Pi acceptance.

## Executed setup and data checks

The independent checkout is GitHub commit `0692fb02750a9537198d7396da3bf701f8cc649c`, tree `dfb6502b11beab0b315d552cf913ca9399d25abd`. This host lacks native GitHub HTTPS credentials while the repository remains private. Authenticated GitHub Git-data APIs supplied the exact objects to a verified local mirror, followed by a native shallow Git clone and fast-forward. Commit/tree identities and `git fsck` matched. There are no object alternates or development model/audio-cache links. The initially empty clone created its own environment and input cache. Its Python bootstrap reused only dependency wheels newly downloaded by the separate cold bootstrap probe. Its first UI installation reused the global pnpm package store; the additional empty-store verification below measures that dependency path separately.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python reproduce.py --device cpu --run-id dgx-cpu-http-20261017
```

This is a **CPU run on the DGX host**. All eight GPUs had active foreign jobs; none were used or interrupted. The independent environment is Python 3.13.9 with PyTorch/Torchaudio 2.10.0+cpu. GPU timings from the original experiment remain separate historical measurements.

Preflight passed in **1,220.17 seconds**: 4,582 required input files verified, 52 completed VCM artifacts/evaluation inputs excluded, and 34,659 mapped recordings verified. Of these recordings, 30,219 were reconstructed from pinned upstream Parquet bytes. Designed recordings were independently downloaded from the project's pinned dataset revision. The original protocol, audio hashes, labels, splits and catalog remain unchanged. No online synthesis is required to repeat the fits.

In the independently installed environment, the full source Python suite passed **219 tests** and the UI passed **99 tests**, type checking and its production build. These setup checks precede completion of the training workflow; they are not a claim that the six fits, locked evaluation or final artifact comparison have finished.

An additional [empty-cache UI check](../experiments/iris-reproduction-v1/cold-ui-verification.json) copied the tracked source into another ignored workspace within this clone and used empty, dedicated npm/pnpm/XDG stores and freshly downloaded local Node tools. It downloaded all **170 packages**, reused zero packages, then passed all 99 UI tests, type checking and production compilation in **174.37 seconds**. It shared only this clone's new Python environment. This establishes the cold UI dependency path; it does not assert byte-identical CSS bundles across build contexts or measure browser/device performance.

The six-fit workflow is running. Its first compact CE epoch completed in 125.81 seconds after feature preparation. Final selections, held-out results, complete pipeline timings and bundle comparisons remain pending.

## Download failure and correction

The first exact-main attempt at `3d17eddd79880d26e0c8eebe49d36fed02aaeb74` installed an independent CPU environment, then stopped **before fitting** after 1,381 designed recordings: the pinned Hub client exhausted the Xet read-token API quota (HTTP 429, 1,000 requests per five minutes). No candidate was trained or selected in that attempt. The [failure receipt](../experiments/iris-reproduction-v1/preflight-failure-xet.json) preserves source identity and denominators.

The corrected recipe sets the client's documented [HF_HUB_DISABLE_XET](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubdisablexet) option before importing it. A real HTTP probe retrieved the same pinned 38,444-byte recording and matched its declared SHA-256. Immutable Hub objects are cached within a clone; run workspaces and VCM fitting states remain separate. Data, splits, normalizer, grammar, losses, seeds, selection and test gates are unchanged. The failed attempt remains preserved.

The initial installer redirect failure was also corrected by installing the same checksum-pinned uv 0.12.5 wheel from official PyPI. A separate cold dependency probe took 556.04 seconds; the exact-main clone installed its own environment in 186.47 seconds using only that probe's newly downloaded dependency wheels. Neither used a development model/audio cache.

The recorded training dependency set requires Python 3.13, including `audioop-lts==0.2.2`. The independent clone uses Python 3.13.9. The README now states this training prerequisite explicitly; the separate inference package retains its Python 3.11–3.13 runtime contract. No frozen dependency version was changed.

## Selected delivery and offline replay

The selected-artifact workflow `python train.py --skip-train --publish --run-id iris-github-final` completed separately from the fresh fitting experiment. Its [workflow receipt](../experiments/iris-release-v1/reproduction-run-iris-github-final.json) records current Torch/ONNX/ncnn checks, resource measurement, 216 default workflow Python tests, all 99 UI tests, type checking, production build and package verification. The [private publication receipt](../experiments/iris-release-v1/publication-hf-iris-github-final.json) verifies immutable downloaded hashes and model privacy. GitHub and dataset visibility were preserved.

The [extracted-package replay](../experiments/iris-reproduction-v1/final-package-replay.json) checks the selected 14,140,040-byte ZIP (`efbb3cc9df013a39ecc051329204f6d6c5b9d6db29ae2faac73a9198fbc7442d`). All 215 manifest payload hashes survived extraction and replay; all 17 approved model payloads matched Git. The standalone audit installed its network-denial hook before importing runtime code, imported package modules from the extracted directory, and observed zero network attempts. Both engines agreed on PCM, raw commands and typed outputs for 88 packaged sources × three repeats = 264 observations. Each measured 31/36 correctly accepted personal positives and 1/52 negative accepts. Actions, wake and speaker ownership were disabled. These synthetic command diagnostics do not establish complete wake-to-action or human acceptance.

The supported command is now in Git:

```bash
.venv/bin/python -m scripts.test.iris_package_replay \
  --package /path/to/iris-pi.zip --sha256 PUBLISHED_SHA256 \
  --output .cache/iris-reproduction-v1/offline-replay
```

It requires a new evidence directory and preserves partial/previous output. The two rejection guards were observed failing before implementation, then passing; the positive flow used the actual pinned ZIP. The expanded [source suite](../experiments/iris-reproduction-v1/source-software-verification.json) passed 221 tests (17 existing export warnings). This suite includes three plot checks and the two new offline guards in addition to the 216 default workflow checks. The standalone replay measured ncnn p50/p95 5.98/11.77 ms and ONNX Runtime 4.97/8.52 ms on the shared DGX CPU host during training; these are recorded-command processing times, not physical Pi measurements.

Fresh-clone run receipts, this report and the central experiment directory are late verification evidence outside the frozen selected-component closure. Their Git commits preserve their identities. Training/runtime code, scientific protocols, data receipts, selected weights, acceptance policy and static model evidence remain hash-bound by the release receipt. Updating verification findings therefore does not change the model package being measured.

The final private model revision is `9ed418197914496de33b7720fb988b7447167c41`; the dataset documentation remains at `44a1a23081a8b042c5d43a73cf233c03c8151049` with unchanged visibility. The final ZIP differs from the prior selected-artifact package only in its release/host measurement evidence and manifest. Inference and UI payloads are identical. The local Pi transfer folder was refreshed and all 220 transfer-file hashes verified; its restricted benchmark companion remains unchanged and local. Earlier package/publication receipts remain preserved.

## Manifest fixture failure and correction

The first independent selected-only run (`dgx-selected-20261017`, source `0692fb02750a9537198d7396da3bf701f8cc649c`) passed acoustic evaluation, parity, resource checks and the UI build, then failed one of 216 Python tests. The wrapper correctly sets a run-specific `IRIS_REPRODUCTION_INPUTS`; the audio-map unit fixture wrote its own default manifest without scoping that environment variable. The same test failed independently when invoked with the wrapper's override. The runtime resolver correctly refused a missing fixture recording.

The [fix receipt](../experiments/iris-reproduction-v1/manifest-fixture-fix.json) preserves failure/log hashes. The fixture now explicitly selects its own manifest and tests both default and custom paths, including changed-audio rejection. The focused check changed from one failure to two passes; the entire source suite passed **222 tests** with the inherited override (17 existing export warnings, 62.46 seconds). Runtime, fitting, calibration and scoring code did not change.

The ongoing six-fit workspace received only this test-module patch, recorded by its before/after hashes. All five fitting/frontend/policy component hashes and frozen protocol/data/input hashes were checked unchanged; the source checkout remains unchanged. This is a disclosed software-harness correction, so that run is not presented as an entirely unmodified clean-checkout execution. A separate clean checkout of the corrected main is used for selected-artifact reconstruction and publication. The failed selected-only run remains preserved.
