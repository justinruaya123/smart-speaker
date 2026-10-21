# Windows reproduction and PC benchmark — 2026-10-19

## Scope and baseline

User requested setup, reproduction and benchmarks of `justinruaya123/smart-speaker`. Fresh isolated clone: `codex/final-submission`, commit `af58d9f7ed952c30b9f5f65dfdd84a64a6d9d56b`, under `D:/School/AI231/.worktrees/me2/smart-speaker`. Study ID: `windows-reproduction-benchmark-20261019`. Addresses ME-02/03/04/06/07/08; no new UC coverage is claimed. Hypothesis: the frozen full-model seed-53 recipe and recorded benchmark execute on this Windows host without changing data, thresholds or selected artifacts. Allowed changes: local setup, ignored run outputs, this receipt, experiment summary, and the download-cleanup portability fix/test. No actuation, microphone, Pi deployment, model installation, commits or pushes.

## Setup and verified environment

Windows 11 build 26200, AMD64, 16 logical CPUs, NVIDIA RTX 3080 10 GiB, driver 591.59. Root `.venv` uses CPython 3.13.14. `bootstrap(group='training')` and benchmark setup succeeded with locked dependencies. The Windows PyPI torch wheel was CPU-only; replaced only the local installed torch with official `torch==2.10.0+cu128` from `https://download.pytorch.org/whl/cu128` using `uv pip install --python .venv/Scripts/python.exe --no-deps`. CUDA 12.8 availability and an actual GPU allocation/sum passed. This is an environment deviation from the default locked Windows wheel; lockfiles were not changed. Subsequent bootstrap/sync can restore CPU-only torch, so inspect CUDA availability before fitting.

Installed NumPy 2.3.5 loads OpenBLAS 0.3.30, default SkylakeX kernel. Original artifacts/normalization/row-map verification passed: 29,370 rows, full graph SHA-256 `3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e`. `.venv`, `.runtime`, and `.cache` ignore probes passed before output creation; no ignore change/commit was needed.

## PC benchmark results

Commands, run sequentially without fitting:

```sh
.venv/Scripts/python.exe benchmark.py --target pc --no-setup --output .cache/submission/benchmarks/windows-full-01
.venv/Scripts/python.exe benchmark.py --target pc --no-setup --model-choice compact --output .cache/submission/benchmarks/windows-compact-01
```

Both completed all 196 frozen recordings and 588 repeated timing observations. Typed denominators exclude two uncertain references: 83 human positives, 100 synthetic positives, and 11 human negatives. Previously examined holdout; not blind evaluation.

| Metric | Full 2M INT8 | Compact 935k FP32 |
|---|---:|---:|
| Parameters | 2,066,831 | 935,567 |
| Raw human correct | 81/83 | 77/83 |
| Raw synthetic correct | 99/100 | 99/100 |
| Raw UNKNOWN accepts | 10/11 | 9/11 |
| Demo human correct accepts | 39/83 | 26/83 |
| Demo synthetic correct accepts | 99/100 | 94/100 |
| Demo wrong positive accepts | 1 | 1 |
| Demo UNKNOWN accepts | 0/11 | 1/11 |
| Qualified accepts | 0 | 0 |
| p50 inference ms | 92.03 | 82.14 |
| p95 inference ms | 201.99 | 171.19 |
| p95 real-time factor | 0.04202 | 0.03443 |
| Session loading ms | 153.30 | 169.03 |

One ONNX thread and one BLAS thread. Windows runner does not set CPU affinity, so this is not a pinned-core measurement. Includes frontend, ONNX, finite catalog decoding, typed validation and demo policy; excludes file decoding, speech duration, wake/VAD, endpoint wait and setup. Peak RSS is null because the current worker uses the Unix `resource` API, unavailable on Windows; memory was not measured. No Pi/live/wake/thermal evidence. Both models remain unqualified. Export precision and bound demo policies differ; not a controlled size-only ablation.

Evidence: `.cache/submission/benchmarks/{windows-full-01,windows-compact-01}/{metrics.json,REPORT.md,trials.jsonl,timing.jsonl,telemetry.json}` and programmatically verified aggregate `.cache/submission/benchmarks/windows-summary.json`.

## Reproduction attempts and blocker

Recipe: full model, seed 53, random initialization, 24 human + 32 mixed epochs. Intended 5,376 optimizer updates; **zero executed**, because frozen preparation checks block fitting.

1. `windows-seed53-01`: source shard downloads and 29,186 recordings verified before an auxiliary download error. Windows attempted to unlink an open temporary file and masked the download exception with WinError 32. A deterministic regression test failed on this exact cleanup behavior. `fetch()` now closes the temporary stream before exception cleanup; the regression passes. Direct auxiliary download retry succeeded.
2. `windows-seed53-02`: all 29,370 rows reconstructed. Original payload and PCM checks passed; 24,546 feature hashes differed, so fitting refused. Preserve this run.
3. `windows-seed53-03`: same exact sources, with `OPENBLAS_CORETYPE=Haswell`; all source/PCM reconstruction completed, but 26 frozen feature hashes still differ. Fitting refused. Source receipt and `run.log` are under this workspace. Three initial diagnostic recordings became exact matches under Haswell with 16/64 BLAS threads. For the remaining 26, separate Haswell and Sandybridge probes at 1/16/64 threads all remained mismatched. Kernel choice explains much of the observed difference, but the residual cause is unresolved; no numeric tolerance or substituted inputs were authorized.

Workspaces: `.cache/submission/reproductions/windows-seed53-{01,02,03}`. Downloads were hardlinked between runs but hash-checked by the reconstruction routine; prepared features and output directories were separate. Residual row list: `.cache/frontend-mismatches.json`; probes: `.cache/frontend-probe-{haswell,sandybridge}.json`. No new checkpoint, ONNX export, calibrated model or weight-equality claim exists. Preinstalled selected models are preserved.

A Linux alternative was checked: `wsl.exe -d Ubuntu2 -- bash -lc ...` fails with `HCS_E_HYPERV_NOT_INSTALLED` and reports disabled virtualization. No OS/firmware settings were changed. Next step is a matching usable Linux/CUDA environment (or a separately versioned, explicitly authorized portability protocol); do not suppress frozen feature checks and advertise exact reproduction.

## Tests and source changes

`python -m unittest discover -s scripts/test -p 'test_submission*.py' -v`: 85 tests, 83 passed, 2 errors. Errors are existing simulated-SSH tests whose expected temporary remote-home paths do not resolve on this Windows run; no actual Pi connection was tested. New download-cleanup regression and all eight reproduction tests pass. `git diff --check` passes. Full repository pytest was not run.

Source edit: `scripts/submission/common.py`, three lines closing the temporary stream before cleanup. New test: `scripts/test/test_submission_fetch_windows.py`. No threshold, frontend algorithm, frozen protocol, training loop, graph, corpus membership or private settings changed.

## Download audit and Linux/Windows evidence boundary

The original recipe was trained and tested on Linux. These Windows failures do not invalidate those Linux results or establish a defective model. The unresolved issue is exact cross-platform numerical reproduction; the remaining feature mismatch cause is not yet established.

A fresh audit of `windows-seed53-05` confirms all 10 pinned Parquet shards and all 184 auxiliary clips match their required SHA-256 hashes. The recipe row map, normalization and catalog verify; all 443 source-receipt files match. Reconstructed ordered IDs and counts match every recipe view: train 23,295, development 1,251, calibration 210, test 4,418, holdout 196 and normalization 1,869. Views overlap by design. No required dataset download is missing or corrupted. The compact JSON audit is retained in [WINDOWS-REPRODUCTION-download-audit.json](WINDOWS-REPRODUCTION-download-audit.json).

`windows-seed53-04` encountered a connection reset during an auxiliary download. The cached `windows-seed53-05` retry avoids that network failure but still exits at the same 26-feature guard before fitting. Never interpret the progress line at 29,186 upstream recordings as a missing final 184 clips; auxiliary extraction runs afterward and the audited prepared manifests contain the full membership.

### uv commands and remaining compatibility work

Use uv from the repository root. Locked setup currently chooses a CPU-only torch wheel on this Windows host. The tested local CUDA override and diagnostic are:

```sh
uv sync --locked --inexact --python 3.13 --group training --group benchmark
uv pip install --python .venv/Scripts/python.exe --no-deps 'torch==2.10.0+cu128' --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run --no-sync python benchmark.py --target pc --no-setup
```

`--no-sync` deliberately preserves the locally installed CUDA build. This is not a permanent lockfile fix; a later sync may restore the CPU wheel. Proper accelerator-specific uv source configuration and a reviewed regenerated lock remain follow-up work. CUDA forward/backward passed on the RTX 3080, so GPU functionality is not the active reproduction blocker. Neither CUDA setup nor CPU benchmarking requires WSL.

The tested Haswell override is a diagnostic, not an exact-reproduction workaround:

```sh
# Git Bash/POSIX shell; use a new output directory for every attempt.
OPENBLAS_CORETYPE=Haswell uv run --no-sync python reproduce.py --seed 53 --yes --no-benchmark --no-setup
```

It still fails on 26 frozen feature hashes. Do not describe it as a successful training command. The original Linux environment remains the demonstrated training path; a portable feature contract and any tolerance-based policy need investigation and explicit versioning before support is claimed.

## User-authorized CPU training fallback

After reviewing the Windows findings, the user requested merging this PR and relaxing PyTorch's CUDA-only requirement. The portable reproduction entry point and shared final-model fitting stages now select CUDA when available and CPU otherwise. Device selection is carried through model placement, optimization and development scoring; CUDA memory setup/cleanup and GPU receipt fields are conditional. The portable run's protocol, and both entry points' checkpoint metadata and selection receipts, record the selected device; the historical trainer keeps its frozen protocol unchanged. CPU GPU/memory fields are null rather than fabricated.

Three added tests cover CPU configuration without CUDA setup, retained CUDA configuration, and an actual one-epoch/one-optimizer-update CPU smoke run of a reduced test architecture. The smoke run verifies changed weights, CPU tensors, checkpoint writing, CPU development dispatch and no CUDA cleanup. Together with the existing eight reproduction tests and download regression, all 12 focused tests pass. The expanded submission suite runs 88 tests: 86 pass and the same two Windows simulated-SSH cases fail (one error and one assertion failure on the latest run). This is real fitting-path evidence, not a full 2M training run or accuracy claim. macOS is untested; no MPS support is claimed. The exact-feature, source-audio, split and qualification guards remain unchanged, so the 26-feature Windows blocker is still present before a full fit can start.

## Decision

PC benchmark execution is complete and measured. Automatic CPU training fallback is regression-tested; CUDA configuration remains preferred when available. Exact Windows training reproduction remains **blocked**, not complete. Preserve failed evidence and obtain a compatible numerical environment before full fitting. Physical Pi and live microphone acceptance remain unmeasured.

## Frozen-input replay repair — 2026-10-19

Follow-up authorization: address the preprocessing reproducibility issue. Baseline for this repair is `b6c1d09e58e008d4155aa4c61677fee5e1d11b80`, branch `codex/final-submission`, initially clean. Scope is reproduction source/tests, this existing report, README/experiment navigation, and separate ignored/external artifacts. Addresses ME-02/03/04/07/08. No repository reorganization, frozen-evidence rewrite, full-model fitting, installation, publishing, device action, commit or push is authorized by this repair. The historical failures and decision above remain evidence for source-only execution.

### Measured cause and remaining uncertainty

- The original six-recording loop fails on the default Windows stack. Haswell reproduces 24,520/24,546 bound feature images exactly, leaving 26.
- Windows NumPy 2.3.5 bundles OpenBLAS 0.3.30 with `MAX_THREADS=24`; `threadpool_limits(limits=64)` actually reports 24. Tests at every available limit from 1 through 24 do not resolve the residuals. NumPy SIMD disabling, array-layout, FFT, magnitude and log probes did not establish a complete source-only solution.
- An isolated, hash-verified conda-forge OpenBLAS 0.3.30 diagnostic DLL supports 64 requested threads (`MAX_THREADS=128`). Under Haswell, that actual 64-thread path reproduces all 25 long-recording residual hashes. It is not installed into NumPy, added to the lockfile, shipped in the feature archive, or used by production inference.
- The final supplemental recording differs at exactly one float32 element: frame 206, channel 33, current `-5.161544322967529` versus frozen-image `-5.1615447998046875`. Searching neighboring single-element representations recovers an entire NPY payload with the **pre-existing** SHA-256 `424ebe460a9393141af02d3cab9db8ee3ce9f627f080ac94f566524786e59325`. This cryptographically verifies the recovered image, but does not establish the numerical cause. It is explicitly recorded as digest-verified reference-image recovery, **not source rederivation or a general frontend correction**. No row-specific correction exists in production code.

### Implemented contract

`--frozen-features ZIP` is opt-in; the default source-only path and its strict mismatch guard remain unchanged. Bound rows read `features/<original-features-sha256>.npy` from that archive. The original row map, not the archive's manifest, supplies the expected digest. Missing/corrupt images fail closed. Size is bounded before decompression; payload SHA-256, original PCM-derived shape, float32 dtype and finite values are required. No pickle loading, path extraction, numerical tolerance or expected-hash substitution is used.

Original WAV and decoded PCM checks still run for every reconstructed recording, as do train/evaluation separation, ordered view reconstruction and the unchanged equal-human-utterance normalization check (`atol=1e-12`, `rtol=0`). Unbound external-evaluation features remain source-derived and the existing separate one-thread evaluation routine is unchanged. New preparation audits distinguish `frozen_hash_bound_replay` from `source_rederived`, record the archive hash and reused-row count, and are bound into new fitting protocols. No new fit is claimed here.

The local archive contains 24,546 distinct original-hash images: 24,520 rederived under native Haswell/24-thread NumPy, 25 rederived under the isolated Haswell/64-thread diagnostic library, and the one disclosed digest-recovered image. Every image's full-file SHA-256 was checked while building and independently checked again after closing the archive. The archive is **773,356,621 bytes**, SHA-256 `f98dc02ff2296a649485bf1239c4b27e6026641f71886c0bcce081c1a324e0bf`. It is external, not committed and not public-hosted. Another computer needs this exact archive transferred separately; no automatic download endpoint is claimed.

Local artifact directory: `C:/Users/Justin/AppData/Local/hermes/outputs/feature-replay-20261019/`. `archive-receipt.json`, `image-recovery.json`, `verification.json`, copied preparation audits, test records and the evidence bundle preserve the provenance. Diagnostic probes are under `.cache/frontend-repro-20261019/`; failed broad-auditor attempts are preserved, not reported as passes.

### Executed acceptance checks

Root environment was prepared through `uv sync --locked --inexact --python 3.13 --group training --group benchmark`; commands deliberately use `uv run --no-sync` and the existing root interpreter. These are dirty-source-copy replay runs with verified cache reuse, not new clean-clone or source-only acceptance claims.

```sh
uv run --no-sync python reproduce.py --prepare-only --yes --no-setup \
  --model-choice full \
  --frozen-features 'C:/Users/Justin/AppData/Local/hermes/outputs/feature-replay-20261019/frozen-training-features.zip' \
  --source-cache '.cache/frontend-repro-20261019/source-view' \
  --workspace '.cache/submission/reproductions/frozen-replay-full-01'
```

Exit 0. The compact command differs only in `--model-choice compact` and a new `frozen-replay-compact-01` workspace; it also exits 0. Existing destinations are never overwritten. The read-only source view hardlinks 10 independently hash-checked pinned Parquet shards and 184 verified auxiliary WAVs from the earlier downloaded cache; original source files were not modified. Without `--source-cache`, the existing downloader retrieves those pinned sources.

| Check | Full replay | Compact replay |
|---|---:|---:|
| Original source/PCM rows verified | 29,370 | 29,370 |
| Bound feature images matching original SHA-256 | 24,546 | 24,546 |
| Feature hash mismatches | 0 | 0 |
| Ordered fitting rows | 23,295 | 23,289 |
| Development rows | 1,251 | 1,251 |
| Parent human normalization rows | 1,869 | 1,869 |
| Normalization check | pass | pass |
| New full-model optimizer updates | 0 | 0 |

Compact still retains the six training-excluded parent rows for normalization ancestry; neither recipe's membership or labels changed. Programmatic postchecks compare every view's ordered IDs and original hashes/labels/split metadata, verify complete image coverage for both row maps, and verify that selected checkpoints, ONNX graphs, catalogs and normalization files retain their expected hashes. Both `reproduce.py --verify` commands also exit 0.

Fourteen reproduction tests pass, including six added replay tests. The full submission suite executes 94 tests: 92 pass, one existing simulated-SSH test fails and one errors because the Windows simulated remote home does not resolve to the intended disposable home (existing `test-corrupt`/`test-roundtrip` directories). These unrelated failures were not hidden or repaired in this task. The suite includes a tiny CPU optimizer/checkpoint smoke test; it is not a fit of either selected model. `git diff --check` passes. Full repository pytest, actual SSH/Pi execution and live microphone tests were not run.

Two independent broad file-read audit attempts timed out at 180 seconds while run concurrently; their empty output folders remain preserved. Completed checks are the original source/PCM/per-file hash and normalization guards inside both CLI runs, all replay-manifest postchecks, and the independent sequential rehash of every archived image. Do not label those timeouts as completed dataset audits.

**Current decision:** exact frozen training-input replay is verified on Windows for both submitted models and no mismatch guard is weakened. Source-only Windows numerical rederivation remains incomplete; the last one-ULP cause and a portable general frontend remain open. This removes the frozen-input preparation blocker in explicit replay mode, not the need to execute/measure a full Windows fit, compare resulting weights/metrics, or satisfy the existing quality and physical-Pi gates. No new quality gain or bitwise CPU/CUDA weight-equality claim exists.
