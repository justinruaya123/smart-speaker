# Start Codex here

This branch contains the PC-v1 preliminary implementation. **Do not apply the old chat patch and do not reset to the original semifinal hash.** Work from the current branch head and preserve unrelated files.

Read, in order:

1. `experiments/phrase-contract-v1/START_CODEX_HERE.md`
2. `experiments/phrase-contract-v1/CODEX_FINAL_EXECUTION.md`
3. `experiments/phrase-contract-v1/PRELIMINARY_RESULTS.md`
4. `experiments/phrase-contract-v1/PROGRESS.md`
5. `experiments/phrase-contract-v1/HANDOFF_FROM_PRELIMINARY.md` only for historical context.

Section 0 of `CODEX_FINAL_EXECUTION.md` predates this repository commit; ignore its instruction to apply a supplied patch. The repository is now the source of truth. Regenerate the exact finite contract before running tests:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
python experiments/phrase-contract-v1/run.py init-spec
python -m pytest experiments/phrase-contract-v1/tests -q
python experiments/phrase-contract-v1/run.py check
```

Use `python experiments/phrase-contract-v1/run.py --help` as the authoritative CLI. In older handoff command examples, supply any arguments now required by the CLI, notably `--endpoint-config experiments/phrase-contract-v1/configs/endpoint.proposed.json` for export and `--audio-root <original-audio-root>` for calibration/evaluation.

Continue the fixed architecture: existing causal 113,064-parameter CNN + 39 phones/blank + exact shared-prefix finite-command CTC + endpoint-only typed command output. No architecture search, ASR, LLM, cloud recognizer, external pretrained acoustic initialization, free transcription, or service execution.

Implement the remaining code in this order before expensive fitting: (1) fixed three-fold train-speaker CV runner, (2) four-value development-only endpoint threshold selector, (3) bounded live microphone capture adapter sharing `PCVRuntime.feed`, (4) target-specific dependency/package verification, (5) physical-evidence-verifying qualified publisher. Add tests first for every behavior change and keep diagnostic publication fail-closed.

Then obtain the original human recordings specified by the recording plan, freeze train/dev/calibration/sealed-confirmation speakers and hashes, run rights/near-duplicate/manual label review, train the primary two seeds, run only the prescribed hard-pair ablation if development requires it, calibrate once, evaluate the sealed set, and perform Pi4/Pi5 microphone qualification. Preserve all failed candidates and confirmation failures. Do not reduce the 570-phrase scope or the 99% precision / 90% complete-command recall / 1% negative-FAR gates to manufacture a release.

Call the model final only if the exact shipped model/catalog/frontend/endpoint/policy binding passes every phrase, every use case, sealed human confirmation, clean-package checks, and each advertised physical Pi/microphone profile. Otherwise commit the precise blocked state and evidence.
