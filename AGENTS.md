# Iris agent contract

Read [REQUIREMENTS.md](REQUIREMENTS.md), [EXPERIMENTS.md](EXPERIMENTS.md), then the current [release report](docs/EXPERIMENT-iris-release-v1.md). This is a local finite-command smart speaker for Raspberry Pi 4/5. No ASR/STT, transcription reader, phonetic decoder, LLM, remote recognition or model fallback belongs in runtime.

## Working boundaries

- Inspect branch, staged/dirty paths and artifact hashes first. Preserve unrelated worktrees, datasets and running jobs. Never reset, clean, stash or stage another task's work; never use `git add .` or `git add -A`.
- The final-release maintenance window authorizes the named paths in [archive-manifest.json](experiments/iris-release-v1/archive-manifest.json), on the release branch only. Historical research is recoverable from the immutable archive recorded in [DOCS.md](docs/DOCS.md). This does not authorize cleanup of shared caches, physical devices or other worktrees.
- Use the root `.venv`, pinned environment receipts and one implementation. Source, protocols, source/license receipts and compact results stay in Git. Audio, features, checkpoints, downloads, generated packages and private state stay ignored. Verify scoped ignore rules before creating outputs.
- New durable Markdown belongs directly in `docs/`; root requirements, experiment directory, model card and README are entry-point exceptions. Keep one current report per experiment version.

## Experiment integrity

Freeze new protocol versions before changing datasets, splits, normalization, labels, objectives, catalog competition or acceptance policy. Record hypothesis, requirement IDs, actual source revision/diff, data hashes, seed, environment and allowed outputs. Preserve failed experiments. Historical human-only studies prohibit synthesis/splicing/augmentation and remain archived rather than silently redefined.

Fit on train, choose weights on development, fit thresholds on development/calibration and write a hash-bound lock before test. Test may veto promotion to a predeclared baseline; it may never drive threshold retuning. Report raw accuracy, correct accepts, wrong accepts, unknown accepts, complete slots, sources/speakers and denominators separately. Synthetic voice prompts do not prove human demographic coverage.

Use one GPU by default. A permitted second run must poll every five seconds and checkpoint/stop when a foreign job appears or any other GPU rises at least 20 utilization points above the initial baseline. Never interrupt another user's job.

## Runtime and actions

Microphone → Hey Iris → command VAD → direct VCM and independent speaker check → typed validation → executor → receipt. One wake owns one subsequent command; overlap/mismatch/missing evidence rejects, then resets. No persistent enrollment. Ownership remains experimental until original human wake/command tests exist.

Tentative previews use the same VCM at most twice per second, with only the latest pending prefix. They never execute. Final microphone commands require VCM acceptance, owner acceptance, valid slots and configured actions. WAVs, fixtures and benchmarks are always non-executing. Unmapped targets are unconfigured. No substitute entity or silent provider fallback. Deduplicate by session/utterance; never retry an unknown outcome. Persist jobs and execution identifiers atomically.

## Supported workflow and verification

```bash
python reproduce.py                  # isolated actual continuation fits from a clean clone
python reproduce.py --device cuda    # one explicitly selected available GPU
python train.py
python train.py --skip-train
python train.py --skip-train --publish
```

The default is the frozen continuation comparison using verified project initializers. `--from-scratch` is a separately identified random-init reconstruction study, not a claim of identical historical weights. Publish only permitted data. Check downloads by immutable revision and SHA-256. One package writer is `scripts/archive_io.py`; runtime assets come from `scripts/speaker_publish.py` and `scripts/mvp_delivery.py`. [scripts/modules.json](scripts/modules.json) maps current source.

Run meaningful Python, UI, acoustic, export/decision-parity, extracted-package, offline and reconstruction checks before claiming completion. Keep physical Pi, live microphone, real HA actions and original human wake-pair evidence distinct from software tests. Packaging does not establish model qualification. Consult the Pi runbook for operator acceptance steps.
