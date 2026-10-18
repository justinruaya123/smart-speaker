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

## Decision

PC benchmark execution is complete and measured. Setup works for CPU inference and CUDA allocation after the documented local torch override. Exact Windows training reproduction remains **blocked**, not complete. Preserve failed evidence and obtain a compatible numerical environment before fitting. Physical Pi and live microphone acceptance remain unmeasured.
