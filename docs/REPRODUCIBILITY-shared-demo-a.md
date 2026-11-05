# Baseline A: treated data and fresh-main training reproduction

Target: the retained **175653-parameter `shared-demo-slots-v1 / heads_s53`**, selected epoch38. This is not the older 214592-parameter atomic model or the named-device-expanded child. Scope:34 complete commands,102 wordings,18 intents; Kitchen/Ceiling unsupported. Qualification/actions remain disabled.

## Inputs and provenance

The versioned [Hugging Face dataset](https://huggingface.co/datasets/justinruaya123/ai231-smart-speaker/tree/main/shared-demo-a-v1) contains only the necessary treated references, original identities/splits, expected feature hashes, the exact TRAIN-only normalizer and3,964 locally authored synthetic WAVs. Its card and BibTeX distinguish upstream dataset/paper citations, source-specific research restrictions, unconfirmed consent and Microsoft synthetic-output terms. This is not a blanket open-licensed audio corpus. No restricted third-party audio/features, numeric auxiliary pool, private recordings, old fit caches or unrelated studies are newly uploaded.

TRAIN:25,958 rows (1,889human/24,069synthetic). DEV:2,132 (256human/1,876synthetic). Synthetic TEST:440; exposed official regression:202. Calibration is not used because the frozen decision policy is fixed argmax. The inherited184Qwen-designed DEV WAVs are already present on dataset main and referenced without another copy. Third-party recordings are downloaded from their immutable original publisher revisions, with WAV/decoded-PCM checks. All28,090TRAIN/DEV feature arrays and the1,889-human equal-utterance normalizer must match before fitting.

`submission/shared-demo-a/dataset.lock.json` pins the published manifest. Acquisition resolves **Hugging Face main** once to an immutable revision, verifies its files, and records that revision. Third-party historical revision pins remain necessary: substituting today's upstream main would change the experiment. GitHub acquisition must use main and record its actual commit. Both download caches and training outputs must be new.

## Commands

Use a fresh main clone, CPython3.13.14, uv and an RTX3080-compatible CUDA host. The historical scientific dependency versions are separate from the newer application runtime lock:

```sh
git clone --branch main --single-branch https://github.com/justinruaya123/smart-speaker.git baseline-a-reproduction
cd baseline-a-reproduction
uv venv --python 3.13.14 .venv
uv pip install --python .venv/Scripts/python.exe --index-strategy unsafe-best-match -r submission/shared-demo-a/requirements-training.txt
.venv/Scripts/python.exe -B scripts/shared_demo_a.py all --output .cache/baseline-a-reproduction-01
```

On POSIX use `.venv/bin/python`; Windows commands above select the actually measured scientific backend. Cross-OS or CPU equality is not assumed. The explicit CUDA recipe fails closed if CUDA/BF16 is unavailable; it never silently switches to CPU.

`all` performs real source/WAV/PCM/feature preparation, a separate four-update optimizer smoke, **48 full epochs/4608 updates from random initialization**, exact checkpoint/curve/draw comparison, a selection lock, then exposed TEST/official evaluation and export parity. Original checkpoints are comparison evidence, never training initializers. The unchanged historical fitter and minimal import closure are isolated under `submission/shared-demo-a/source/`; removed app/runtime modules are not restored to producer main.

Stages are also callable separately with the same output path: `prepare`, `smoke`, `train`, `evaluate`. `prepare` requires a previously nonexistent output directory. Existing runs are never overwritten or automatically restarted. A failed full-fit equality comparison preserves its completed fit and blocks the success/selection lock; it is not repaired by forcing epoch38 or tuning on TEST.

## What counts as reproduction

Required: every selected and final model tensor, selected epoch, all48loss/DEV curve events excluding elapsed time, original initial-encoder identity, all4608updates/147456draws, ordered-draw digest, per-ID counts and source quotas. Source/feature/normalizer identities must match before fitting. Selected model/policy/source/data bind before held-out features. All642original synthetic TEST/official decisions must match; wrong slots count as wrong commands. Torch/ONNX parity is checked separately from original complete-command parity. A new export may differ in serialization while representing identical tensors/decisions; its hash is reported, not substituted for the original graph.

The original observed results remain203/204 exact synthetic commands,98/204 context-added commands, and71/77 supported exposed-human regression commands. These are **targets for readback comparison, not a claim the fresh fit has already passed**. The detailed actual execution receipt is published only after real fitting/evaluation.

## Actual execution status

Hugging Face dataset main is published at `13d388c31985be7f42af56b46570281728ecc054`; the exact manifest was downloaded and verified after publication. Full payload readback and retraining remain pending.

The 22 new guard tests pass; the final scoped model-boundary/export/publication suite returned **194 passed, 4 skipped** in the fresh environment. The earlier full publication-checkout suite (including the first 15 guards) returned 621 passed, 14 failed, 8 skipped. An independently cloned untouched main at `4028871f4fe9c07a7d9d7b946de3bf3512e0fdcd` returned 606 passed, the exact same 14 failed node IDs, and 8 skipped. No new failing node was introduced. The complete inherited-failure list is preserved in `submission/shared-demo-a/publication-tests.json`; missing removed application modules, missing ncnn, old ARM64/README expectations, old plot/recipe checks and Windows line-ending fixtures are not repaired as part of this model reproduction. An initial shallow-clone test run also lacked the required historical Git oracle; fetching main history resolved those setup errors without source changes. The independent reviewer failed at its provider without a verdict; direct integrity review added seven RED-to-GREEN gold-reference and prepared-source guards rather than claiming external approval.

Original frozen model files are unchanged. No fresh-human, microphone, Pi, echo cancellation, provider action, consumer-default switch or acoustic qualification is part of this task. Software guard tests and dataset upload do not establish successful retraining.
