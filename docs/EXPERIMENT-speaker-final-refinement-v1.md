# Final fixed-architecture VCM refinement and Raspberry Pi 4 handoff

Study: `speaker-final-refinement-v1`, authorized 2026-10-16. The final bounded training comparison and locked evaluation are complete; the model is not release-qualified. Source, data and selection identities are in [the protocol](../experiments/speaker-final-refinement-v1/protocol.json); primary references and their exact role are in [references.json](../experiments/speaker-final-refinement-v1/references.json).

## Fixed model and deployment scope

The model remains `PooledContextVCM`: six 192-channel context blocks, four attention heads, feedforward width 384, dropout .1 and learned semantic pooling; 2,066,831 parameters. The existing 40 acoustic outputs (39 phones plus blank), 19 intent outputs and two target outputs, 40-bin frontend, output stride, 1,079-entry finite catalog and typed decoder are unchanged. Normalization remains fitted on 1,869 eligible original human training utterances. Runtime emits finite commands or rejection; no transcript parser, ASR/STT fallback, LLM, cloud inference or TTS ships.

The target is the existing **Raspberry Pi 4, 64-bit Linux ARM64, CPython 3.12, glibc >=2.28** package contract. CPU ONNX inference uses one thread. Target wheels are included for offline installation. A 32-bit OS or another Python ABI needs a different dependency bundle. Desktop execution and a verified archive do not establish physical Pi performance.

The retained ARM64 lock pins ONNX Runtime 1.30.0 and NumPy 2.5.3; the SSH export/replay environment uses ONNX Runtime 1.23.2 and NumPy 2.3.5. Exact wheel hashes and their complete dependency closure are verified against the [existing runtime lock](../experiments/phrase-contract-v1/configs/runtime-cp312-aarch64/dependencies.json). ARM execution and cross-version numerical behavior remain part of the physical Pi check.

## Grounded study rationale

The project's [release-v1 study](../experiments/pcv-release-train-v1/runs/release-v1/results.json) selected hard-label training over its distilled arm on development (159/176 versus 157/176 human exact commands). Its published-test raw scores were 502/636 human and 3,337/3,368 synthetic, with failed acceptance gates. This motivates retaining the established architecture and hard-label losses while investigating the data mix. Historical-versus-new differences are not an isolated treatment effect because fitting membership and training schedule differ.

[Park et al., SpecAugment, Interspeech 2019](https://arxiv.org/abs/1904.08779) introduce masking in filterbank feature space. We use a conservative adaptation during training only. Their ASR results do not establish a benefit for this finite-command model. Both comparison arms receive identical augmentation, so this study estimates the added-data effect under that recipe; it does not independently estimate the masking effect.

[Warden, Speech Commands, 2018](https://arxiv.org/abs/1804.03209) motivates a task-specific dataset and reproducible command-recognition evaluation. This project uses complete typed commands and rejection cases, rather than claiming the paper's isolated-word benchmark. No Speech Commands recordings are added. Existing declared speaker splits, human/synthetic strata and published denominators remain fixed; prior test exposure is disclosed.

[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) distinguish confidence calibration from classification accuracy. We retain the existing score interfaces and test actual accepted commands. No new temperature-scaling component is introduced. Conditional finite-catalog slot scores are not empirical accuracy probabilities, and synthetic calibration cannot supply missing human evidence.

[ONNX Runtime quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) supports dynamic quantization as a candidate for transformer-style models and requires evaluating accuracy changes. We reuse the existing constant-weight MatMul INT8 exporter, check numerical export and typed development fidelity, and retain the full-model four-MiB graph limit. Pi speed and thermal behavior require measurement on the device.

## Data, sampling and augmentations actually used

Control fitting contains 23,295 eligible examples: 1,869 human and 21,426 earlier synthetic. Completion fitting adds 184 newly generated training recordings, for 23,479 total. The other 368 new recordings belong to synthetic development/calibration, with different declared designed conditions. Both arms use the same 1,251 development examples (256 human / 995 synthetic) and the same full catalog. The original human coverage gaps remain visible. Generated text is intended supervision, without independent listening or verified accent/voice-identity evidence.

The added recordings use pinned Qwen3-TTS VoiceDesign anchors and Base whole-utterance generation, following the upstream [voice-design/clone workflow](https://github.com/QwenLM/Qwen3-TTS). They contain 92 prompts per condition: 30 sparse-family commands, the 48 declared light/color pairs and 14 rejection definitions. Two designed conditions belong to each partition. Native 24 kHz and derived 16 kHz recordings are preserved; no original human reference voice is cloned for this generation. Exact model revisions, prompts, seeds, waveform checks and retry bounds remain in [the generation protocol](../experiments/speaker-synthetic-completion-v1/protocol.json) and [source ledger](../data/sources/qwen-synthetic-completion.source.json). The synthesizer is an offline data tool and is excluded from the Pi archive.

| Setting | Shared human foundation | Control mixed arm | Completion mixed arm |
|---|---:|---:|---:|
| Epochs | 24 | 32 | 32 |
| Updates per epoch | 96 | 96 | 96 |
| Human positive / UNKNOWN per batch | 24 / 8 | 12 / 4 | 12 / 4 |
| Earlier / new synthetic per batch | 0 / 0 | 16 / 0 | 12 / 4 |
| Batch / microbatch | 32 / 8 | 32 / 8 | 32 / 8 |

Seed is 53. The selected human foundation initializes both mixed arms. Sampling selects intent, complete command, declared speaker and then recording; independent per-group random streams match the human draws across arms. Mixed batches remain 50% human regardless of the corpus row imbalance. This is a single-seed comparison, not evidence of seed-robust gains.

Training uses AdamW with weight decay .001, gradient clipping 5, existing CTC + .3 intent cross-entropy + .15 target cross-entropy losses, cosine learning rates .0008 for the foundation and .00025 for mixed fitting, and minimum .00002. GPU TF32 is enabled in both arms. CUDA CTC backward is not bitwise deterministic; seeded sampling does not remove that limitation.

| Augmentation | Exact bounded policy |
|---|---|
| Frequency masking | At most one mask; width uniformly 0–4 of 40 bins; applied with probability .5 |
| Time masking | At most one mask; width uniformly 0–min(10, floor(.05 × valid frames)); applied with probability .5 |
| Mask fill | Zero in human-training-normalized feature space |
| Application | Fresh feature copy per sampled training item, both arms; never normalization, development, calibration, test or holdout |

No time warping, speed perturbation, noise mixing, reverberation, word splicing, new teacher, new head, architecture search or hyperparameter grid is added. Original audio and cached features remain unchanged. The small time-mask cap avoids erasing much of a short command; this is a design choice to test, not a published optimum.

## Selection and stop rule

Development is evaluated every four epochs, using true input lengths and cropped downsampled CTC outputs. Selection first favors a nonempty set of correct human accepts with precision >=.99 and human UNKNOWN false acceptance <=.01, then macro-human-speaker accepted complete-command recall, precision, fewer false accepts, raw macro recall, synthetic accepted recall and earlier epoch. Control wins an arm tie. If no checkpoint is feasible, the best diagnostic candidate is retained with failed gates disclosed.

The application rule stays .95 intent confidence and .90 for every required slot, with existing intent/target agreement checks. Qualification additionally requires .99 precision, .90 recall, <=.01 human false acceptance and all 18 intents in original human calibration. The latter still lacks CALL, CREATE_REMINDER, LIST_REMINDERS, MESSAGE and TEMPERATURE. Synthetic examples do not clear that requirement.

The winner is locked before original-human calibration. INT8 development may lose at most one human exact command, one synthetic exact command and one additional human negative relative to FP32, while staying below four MiB. These compression bounds are separate from qualification gates. Model, policy and catalog are locked before a single published test/holdout regression replay. Ambiguous references retain their own counts; capture faults remain failures. No new training is selected from test errors, and there is no result-driven extension after the declared comparison.

## Outcome and device handoff

The complete GPU fitting took 1,990.47 seconds excluding input loading, on one A100, with peak allocated CUDA memory 703,036,416 bytes. Both mixed arms selected epoch 32. The declared human-focused rule selected **control**; the additional Qwen data improved synthetic development results but did not improve the human selection metric in this single-seed comparison. The completion arm and its negative result remain preserved.

| Development result | Control | Completion |
|---|---:|---:|
| Human raw correct commands | 166/193 (86.01%) | 161/193 (83.42%) |
| Human correctly accepted commands | 108/193 (55.96%) | 106/193 (54.92%) |
| Human accepted precision | 108/115 (93.91%) | 106/113 (93.81%) |
| Human UNKNOWN false accepts | 2/63 | 1/63 |
| Macro human-speaker accepted recall | 53.86% | 50.43% |
| Synthetic raw correct commands | 914/967 (94.52%) | 964/967 (99.69%) |
| Synthetic correctly accepted commands | 847/967 (87.59%) | 924/967 (95.55%) |

Neither arm met the predeclared human precision/FAR feasibility constraint. Control is therefore the selected **diagnostic candidate**, not a qualified winner. The paired comparison does not support claiming improved human robustness from the added Qwen recordings.

INT8 export is **3,173,089 bytes (3.03 MiB)**. Compared with FP32 on all 1,251 development rows, it loses zero raw human and synthetic correct commands, with one fewer raw human negative error; 1,247/1,251 candidate commands agree exactly. Numerical and declared compression checks pass; exact cross-engine agreement is not universal. The earlier shared export helper's two-MiB flags do not override this full model's predeclared four-MiB limit.

| Locked INT8 regression | Human test | Synthetic test | Human holdout | Synthetic holdout |
|---|---:|---:|---:|---:|
| Raw correct commands | 516/636 (81.13%) | 3,335/3,368 (99.02%) | 81/83 (97.59%) | 99/100 (99.00%) |
| Correct accepts at .95/.90 | 274/636 (43.08%) | 3,219/3,368 (95.58%) | 38/83 (45.78%) | 99/100 (99.00%) |
| Accepted precision at .95/.90 | 274/297 (92.26%) | 3,219/3,232 (99.60%) | 38/38 (100%) | 99/100 (99.00%) |
| UNKNOWN false accepts at .95/.90 | 3/229 (1.31%) | No negatives | 0/11 | No negatives |

All 4,418 test and 196 holdout rows remain in their source views. Typed scoring retains 4,233/194 eligible rows, with 185/2 ambiguous references explicitly separate; no capture faults occurred. Human holdout positives represent only two declared speakers and are previously examined data. Compared descriptively with the earlier release's 502/636 human raw correct commands, this model has 14 more (+2.20 percentage points); changed fitting membership, sampling, augmentation and catalog prevent attributing that difference to one cause. It is not a fresh blind improvement estimate.

Human-only calibration selects a diagnostic point with 20/122 correct positives (16.39% recall), 20/20 precision and 0/88 false accepts; five required intents remain missing. Qualification stays **disabled**, so actual qualified-mode accepted commands are zero. The nonzero .95/.90 figures above are the separate existing demo application rule and do not establish release qualification. Its human precision, recall and false-accept rate fail the release requirements.

Model SHA-256: `3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e`. Complete source-bound results are in [results.json](../experiments/speaker-final-refinement-v1/results.json) and [export-development.json](../experiments/speaker-final-refinement-v1/export-development.json). No further training or test-driven tuning follows this final comparison.

The Pi archive includes this methods/acoustic-results snapshot, exact citations, model bindings, offline ARM64 wheels, wake/VAD, the existing CPU runtime and original CC BY SLURP smoke recordings only. Group/private-source audio, TTS weights, trainers, PyTorch and feature caches are excluded. Its post-build verification receipt records extracted archive execution and desktop resources; these cannot establish physical Pi performance. Source-checkout links in this copied report refer to the repository; core protocol/reference/result copies are also under archive `evidence/`.

On the Pi, install the declared interpreter and system audio libraries, then extract the archive into a new directory. Do not overlay an existing installation:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3.12 pi_test.py --verify-only
python3.12 pi_test.py --setup --engine onnxruntime --threads 1 --output results/replay-01
.venv/bin/python pi_test.py --list-mics
.venv/bin/python pi_test.py --engine onnxruntime --threads 1 --guided --mic INDEX --output results/live-01
vcgencmd measure_temp
vcgencmd get_throttled
```

`--setup` creates a local environment and installs the bundled hash-locked wheels with `--no-index`. PortAudio/libsndfile must already be available (`libportaudio2` and `libsndfile1` on Debian/Pi OS). The interpreter is a prerequisite; stock Pi OS Python may have another ABI. Use a fresh output directory for each measurement. Preserve replay and live results separately and record actual Pi identity, latency, peak RSS, temperature/throttling and failed capture trials.

Run the two `vcgencmd` probes before and after replay/live testing. Their meanings are documented in the [official Raspberry Pi OS utilities reference](https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd). Throttling history and undervoltage are distinct from current throttling; retain the raw value with the report.

Wake/VAD does not establish wake-owner identity or recovery during overlap. No trained speaker-verification backend is added here. A disabled qualification policy produces diagnostic candidates while rejecting commands; such an archive is not a qualified action-executing deployment. Physical Pi/live microphone evidence remains pending until the device commands above are actually run.

## Post-build verification in the source checkout

The immutable diagnostic ZIP is 46,316,255 bytes (44.17 MiB); its SHA-256 and exact target are in [package.json](../experiments/speaker-final-refinement-v1/package.json). All 74 payload identities passed verification before extraction. The extracted `pi_test.py --verify-only` and actual replay CLI both exited successfully in fresh processes, with source imports resolved inside the archive and no training/TTS imports. All 15 raw candidates and fixed-application decisions agree with the already locked test predictions. This smoke subset contains 13 positive intents and two UNKNOWN recordings, not full acoustic coverage of the 415 definitions or all 18 intents.

On the x86-64 SSH host, restricted to one allowed CPU core with ONNX and numerical-library thread counts set to one, the 15-recording replay measured 97.13/228.11 ms p50/p95 inference, 0.0633/0.0767 real-time factor, 197.93 ms session load and 67.63 MiB peak process RSS. The first inference is included. Timing includes frontend, model and finite decoder; it excludes WAV loading, utterance duration, wake listening and endpoint silence. The replay did not load wake/VAD sessions; their additional live memory/compute is unmeasured. This shared host core was not reserved, and these numbers do not predict Pi4 timing.

The focused suite passes 168 tests. Full-repository pytest still stops during collection at the historical duplicate `test_catalog` module names and missing `train.register_sources` import. Final review found two deployment/evaluation identity gaps; four RED-to-GREEN regressions cover the fixes, with no changes to frozen training code. [verification.json](../experiments/speaker-final-refinement-v1/verification.json) preserves post-build measurements, review rulings, hashes and remaining limitations. The archive carries the pre-build methods/acoustic-results snapshot; this post-build receipt is separate to avoid a self-referential archive hash.
