# Continue PC-v1 implementation from the measured decoder milestone

## What is actually committed

Commit `1a4caaf3263a0106653b8bdd33b8e3f4a4ccac27` contains six additive files: exact shared-prefix decoder, strict command helper, independent decoder tests, benchmark receipt, and milestone documentation. The broader implementation was built and tested in the source delivery, but is **not yet fully committed to this branch**. Do not describe this milestone as a trained or qualified speech model.

The original selected disabled model, research, root documentation, and existing workflows are unchanged. No dataset-upload workflow or external execution is part of this milestone.

## Restore the accompanying implementation

The user-facing delivery supplied with this handoff is:

- `VCM_PCv1_Implementation_and_Results.zip`, SHA256 `500e58c5190300927883ad898c8d408cfaa433ddc1c3e9f04817e9234487aab8`, 270851 bytes, 98 files.
- `VCM_PCv1_remaining_implementation.patch`, SHA256 `e0c63ff501a88fe3153849a99de0668ebb3ceff7c27dfb1beae62921f79a4797`.

These are source/results artifacts, not a model bundle. Obtain the explicit supplied files from the user; do not search arbitrary caches or treat a similarly named archive as equivalent. The patch was applied to an isolated mirror of the milestone files and all resulting bytes were checked against the delivery. It adds only files under `experiments/phrase-contract-v1/`.

From a clean checkout of this branch containing the decoder milestone:

```bash
git status --short
git diff --exit-code
git diff --cached --exit-code
git -c core.whitespace=cr-at-eol apply --check /path/to/VCM_PCv1_remaining_implementation.patch
git -c core.whitespace=cr-at-eol apply /path/to/VCM_PCv1_remaining_implementation.patch
python -m pytest experiments/phrase-contract-v1/tests -q
python experiments/phrase-contract-v1/run.py check
git add experiments/phrase-contract-v1
git commit -m "Add PC-v1 training runtime evidence pipeline and fixed execution handoff"
git push origin codex/semifinal-handoff
```

Preserve unrelated untracked files. Do not force-push/reset/stash/clean or overwrite conflicting files. Some historical Markdown uses intentional trailing spaces; do not rewrite the frozen proposal to silence whitespace warnings. Generated spec files are reproduced by `run.py init-spec` from the committed generator and verified against the six-file binding receipt.

Then read `experiments/phrase-contract-v1/CODEX_FINAL_EXECUTION.md` from the restored source delivery. It fixes the remaining procedure and identifies executable versus unfinished integration steps. No further architecture search is requested.

## Actual preliminary results

The full locally delivered implementation passed 48 tests; the five independent decoder tests in the smaller committed milestone also passed. These are not a claim that the complete original repository suite was run.

For 770 complete positive/rejection phone sequences and 250 frames, five paired desktop runs measured padded CTC median 289.508021 ms and shared-prefix median 81.717550 ms: 3.542789x faster, maximum score discrepancy 4.884981308350689e-15. Probability-state storage fell from 572880 to 115888 bytes; this excludes index arrays, encoder and process RSS.

Noiseless generated phone logits produced 770/770 exact outputs. Severe artificial logit noise (SD6) produced 541/770; all 229 failures are retained. This is not speech accuracy, accepted recall, negative FAR, an audio-SNR test, or a Pi measurement.

A 12-update generated-feature gradient smoke test instantiated the real 113064-parameter baseline and reduced CTC loss from 52.181038 to 12.338114. No deployment checkpoint was saved. Initial oneDNN whole/chunk difference 2.288818e-5 failed the fixed 2e-5 bound. Explicit native CPU convolution produced 1.907349e-5 and passed without changing the bound; both results remain saved. ONNX was not executed.

Original human recordings and trained weights were unavailable. Direct dataset/pip downloads failed in the sandbox; the connected source artifact contained neither. No new human acoustic precision/recall/FAR is established. CUDA, ONNX/ORT execution, microphone captures and physical Pi qualification remain absent.

## Fixed candidate and acceptance contract

Use the original causal CNN/frontend and the exact-prefix decoder, not a replacement transcriber. One graph, 39 phones plus blank, finite whole commands, 570 literal positive wordings and 402 complete typed commands across all ten use cases. Keep all integer 0..100 brightness/volume rows and explicit finite name/content IDs. Preserve 200 rejection prompts. Runtime emits intent plus every parameter only after its real endpoint. No LLM, cloud inference, external pretrained acoustic initialization, general sentence transcript, or service/action execution.

The delivered primary training configuration uses random initialization, seeds53/71, AdamW .001, weight decay .0001, clip5, batch64, at most40 epochs, 80% human positives/20% real negatives, hierarchical use-case/phrase/speaker balancing. One prescribed hard-pair sampling ablation is available before calibration. Retain all epochs and failed trials. Fixed development selection uses every required phrase, worst-use-case performance and complete typed exactness. The total protocol remains bounded at200 epoch-equivalents and24 GPU-hours, including training-only CV and reproducibility checks; no endless retries.

New independent human speech is still the major missing input. The executable recording plan contains48800 unrecorded slots across positive/rejection/natural-unknown partitions. Positive per phrase quotas are train24/12speakers, dev8/4, calibration8/4, new sealed confirmation20/10; speakers and utterance parents are disjoint. The full handoff specifies four conditions, source/consent metadata, hashes, near-duplicate review and an independently held final commitment frozen before fitting. Never replace these with generated audio, spelling tests, augmented copies or repeatedly replayed utterances.

Calibration and final requirements remain99% accepted precision,90% complete accepted recall,1% negative false accepts, independently for every use case. Every final required phrase needs18/20 exact acceptances,10 speakers, zero wrong accepted commands, and4/5 in each condition. Wrong slots, IDs, abstentions and tentative rankings fail positives. Missing use cases cannot be compensated by aggregate scores. An all-reject policy fails. Old examined tests remain regressions; failed new confirmation is retained and cannot be reused as untouched final evidence.

The remaining integration tasks are explicitly specified in the complete handoff: train-speaker CV runner, four-value development-only endpoint-threshold selection, bounded live capture adapter, target-specific dependency/package validation, and a physical-evidence-verifying qualified publisher. Current publisher permits diagnostics only with acceptance disabled. No handwritten passing booleans may enable final release.

Run actual training, calibration, fresh sealed confirmation and physical Pi4/Pi5 microphone tests on available resources. Final graph<1MiB, RSS<256MiB, processingp95<=100ms, RTF<1, responsep95<=1s, sustained30-minute capture and three armed negative hours/profile are required. Do not extrapolate x86 timings to Pi. Only call the model final when the exact shipped bundle passes all applicable gates. Otherwise return the precise blocked status and preserve every unsuccessful result.
