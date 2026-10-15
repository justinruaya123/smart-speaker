# Final selected VCM and Pi handoff

**Status: blocked / Pi-deferred.** Non-Pi qualifications have failed or remain blocked; the actual diagnostic experiments and their complete evidence are preserved. Physical Raspberry Pi testing is deferred.

Download [vcm-pi-handoff-20261015-01.zip](vcm-pi-handoff-20261015-01.zip). It contains the exact selected ONNX model, checkpoint reference, train-only normalization, separately fitted calibration, entity memory, existing command-WAV resident worker, semi-final cards, every missed verbatim, portable evidence and pinned current202 benchmark/default cache. No backend, affinity or memory limit was added.

ZIP SHA-256: `7e86166167560cce4724d2a569e614a6bb2ddaaa7803d452871ceb10e9351dd5`

ZIP bytes: `86039578`

The selected checkpoint is uniform seed 89, epoch 28 after 2,688 checkpoint updates, SHA-256 `bcfc0b85d5e61238e9f93008af78e98df00c8d7c838e2d6c4df10834b6c6b976`. Recipe SHA-256 is `40f2d027e81393ca10edc8200f2a4c6631425c21b807206c63ec9cc6e546218b`; contract hash is `ff0a789114327f031599b198bfdcb74523eafb4c3075588ab8c313c9fecc6ac8`. Grammar, catalog, typed outputs, and the 0.95 intent / 0.90 supplied-slot thresholds remain fixed.

## Results and source-bound evidence

- [Comprehensive experiment report](../docs/EXPERIMENT-vcm-mixed-final-v2.md)
- [Scope, exact failures and observed missed commands](FINALIZATION_NOTES.md)
- [JSON card](card.json), [Markdown card](MODEL_CARD.md), [editable HTML](cards/MODEL_CARD.html), [PowerPoint](cards/MODEL_CARD.pptx), [PDF](cards/MODEL_CARD.pdf)
- [Every missed verbatim](missed-verbatims.csv), [actual learning curves](curves/training-curves.pdf)
- [Final verification](final-verification.json), [Assembly receipt](assembly-receipt.json), [rendering receipt](cards/rendering-receipt.json), [evidence manifest](evidence-manifest.json), [Pi ZIP receipt](pi-handoff-receipt.json)
- [Sealed final-TTS technical quarantine evidence](final-TTS-technical-quarantine-lock.json)
- [Selective archive and cleanup receipt](../archive/vcm-finalization-20261015/archive-manifest.json)

Fresh human accuracy is 0/93 exact supported commands. Original human test accuracy is 25/636; original human holdout is 5/83. The unchanged mandatory defaults score 5/186 overall (human 2/86, synthetic 3/100). Every supplied-slot score is zero. All cohorts have zero false accepts on negatives; original-test retains two overlong ambiguous capture faults and one accepted wrong positive. These cohorts are not pooled: only the 106 first-pass FSC recordings are fresh by declared source identity.

All six training arms completed 30 epochs, with 17,280 actual updates overall. Finite losses/gradients, actual parameter updates, complete curves, train-only normalization and development-only selection were verified. Full CUDA PyTorch/ONNX waveform parity passed for 4,920 valid trials; both capture faults matched. Maximum head error is 1.1444091796875e-5 under 2e-5; all 141 profiled neural nodes were CUDA.

The pinned final Kokoro grid attempted 18,666 unique first takes and preserved 2,017 generation faults. No final-TTS recognition result exists: independent acoustic listening and the qualified supplemental lock are incomplete, and selected ncnn/resident tests remain blocked under the CPU inference hold. Final audio was excluded from fitting, selection, calibration and enrollment. Do not score a selected valid subset or replace failed takes. Preserve failures and version any subsequent correction.

## Physical Pi procedure

Unzip into a fresh directory. Verify all packaged members without loading a model:

```sh
python final/package_pi.py --verify /absolute/path/vcm-pi-handoff-20261015-01.zip
```

Follow the package's README-Pi.md to install the unchanged runtime lock and use its existing ONNX Runtime CPU JSONL command-WAV service. The worker does not supply live microphone, wake or owner qualification. The pinned official benchmark is a separate live laptop-speaker-to-Pi-microphone procedure. Keep all defaults, references, duplicates and the packaged .cache/holdout.parquet unchanged:

```sh
python benchmark/vcm-benchmark/benchmark.py
```

Do not use simulation, quick, limit, selected clips or changed references as the final benchmark. SSH uses the packaged pinned agent. For HTTP, use the packaged pinned pi_agent.py explicitly: the unchanged upstream default agent URL points to main. No physical result, hardware limit or Vulkan capability is claimed here.

After actual Pi measurements, create a physical_pi_benchmark JSON receipt using the exact card bindings, actual hardware/environment, complete supported/negative/fault/missing counts and boolean measured gates as described in README-Pi.md. Update deterministically from the frozen packaged baseline:

```sh
python final/pi_finalize.py --baseline final/card.json --receipt results/pi-receipt.json --output results
```

This writes derived results/card.json and results/MODEL_CARD.md, preserving the packaged baseline and every non-Pi result/failure. Repeating identical input bytes and paths gives identical output bytes. The status becomes blocked / Pi-passed or blocked / Pi-failed for this candidate; actual Pi measurements do not erase non-Pi failed gates. Optional editable/PDF regeneration uses the exact report dependencies:

```sh
python -m pip install -r final/report-requirements.txt
python final/render_card.py --card results/card.json --output results/cards
```

Pi latency, RTF, RSS, CPU, wake, thermal, power and hardware fields remain deferred until an actual bound receipt is supplied.

## Archive and original package

The prior instructions/card are preserved byte-for-byte in archive/vcm-finalization-20261015. Closed own-worktree package-import and software-fixture residuals moved into the ignored archive with compatibility symlinks and complete hashes. All source audio, raw native TTS, failed takes, selected/all-arm checkpoints, logs and references remain preserved. Other worktrees and the unrelated local integration branch were not changed.

The original supplied final/vcm-finalization-package.zip was hash checked at 1f11a6188cbaa27bb4199e7aca728a3a502a7d387c1306ac5c47283f1187e491 and retained. Its preserved README requires actual accuracy gates to pass before deletion; they have not passed. No blanket cache, data or final-directory deletion is authorized by this cleanup.
