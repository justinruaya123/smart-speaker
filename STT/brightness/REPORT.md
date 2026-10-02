> Historical experiment. The ASR model and its shipping options were retired on 19 September 2026. Constructed speech below is excluded from the current [human-only alternatives study](../alternatives/REPORT.md). Commands and package links in this archived report describe the earlier state.

# Brightness STT versus ASR

Fresh experiment, 19 September 2026. **STT is the experimental default; ASR remains selectable.** Both are local parameter readers behind the existing VCM. No music, playlist, filepath integration or device execution was added.

STT was **23.4% faster** at warm median parameter recognition on this workstation. It recognized 716/1010 (70.89%) constructed commands before confidence gating, versus 44/1010 (4.36%) for ASR. **This did not transfer to reliable natural speech:** STT recognized 0/3 shared held-out natural numeric commands, versus 1/3 for ASR. Three positive recordings are insufficient for a general natural-speech comparison.

## Architecture and switching

Microphone/file → common VAD → VCM intent acceptance → selected STT or ASR → shared brightness/target grammar → typed command or clarification. VCM weights and thresholds are unchanged. Rejected intents never reach a parameter model. Neither reader may change intent or call the other as fallback. Existing accepted relative brightness retains finite-head routing. The new reader handles absolute values only, with whole-utterance timing explicitly marked unverified.

STT here means finite-word CTC; ASR means the existing character CTC implementation. Both perform speech recognition. The STT word vocabulary is fixed by training; merely appending words to its vocabulary is not a supported model update.

| Model | Architecture | Parameters | ONNX bytes | Initialization |
| --- | --- | ---: | ---: | --- |
| STT | 80-mel CMVN, convolution, 2-layer bidirectional GRU width 128, 226-word CTC | 707,554 | 2,840,297 | Random, freshly trained |
| ASR | 40-mel checkpoint normalization, width-256 convolutional character CTC | 1,321,757 | 5,319,980 | Existing epoch-19 weights, frozen |

```bash
python train.py --tests-only --no-setup
python train.py --tests-only --no-setup --backend asr
python publish.py
python publish.py --backend asr --output dist/smart-speaker-asr.zip
python publish.py --include-alternative --output dist/smart-speaker-both.zip
python3 pipeline.pyz --backend asr --wav recording.wav
```

The last command requires an ASR-inclusive package. `[parameters].backend` controls brightness; the separate legacy `runtime.asr` switch stays false. Default shipping excludes the ASR graph. See [the repository architecture contract](../../AGENTS.md) and [README](../../README.md).

## Training data and method

The new manifest contains natural brightness speech, unrelated human speech for rejection, isolated number/context words, and constructed phrases. Existing VCM/ASR manifests and weights were preserved. Public sources include SLURP, Snips SmartLights, MLEnd Spoken Numerals, Free Spoken Digit Dataset and MSWC-English. A fresh pinned MSWC acquisition supplied 222 “brightness” and 201 “percent” clips; duplicate files already present were deduplicated. Every retained original recording is hashed and speaker splits are audited.

| Kind | Train | Development | Test |
| --- | ---: | ---: | ---: |
| natural | 599 | 180 | 108 |
| negative | 1199 | 250 | 250 |
| word | 3321 | 1324 | 1387 |
| constructed | 6060 | 606 | 1010 |

Manifest rows: 16,294; vocabulary: 226; cross-split speaker intersections: 0. The trainer excluded 195 rows outside its 0.1–8.0-second feature-length window, leaving 10,984 train and 2,360 development examples. The command benchmark contains 1,368 test rows; isolated words are not counted as commands. Full-path inference separately enforces its existing duration limits.

Constructed phrases cover every integer 0–100, with and without a spoken “percent.” The value 100 uses “hundred” in these audio compositions; the “one hundred” variant is covered by text-parser tests but not separately by this acoustic test. Each split uses only components assigned to that split. Recordings are concatenated with brief silence and can mix speakers within an utterance. They do not reproduce natural coarticulation. Test compositions reuse component recordings, so 1,010 compositions are not 1,010 independent speakers or natural utterances.

SmartLights was already used to train the incumbent ASR. Its held-out-for-STT samples are therefore an explicitly biased diagnostic, separate from the shared natural holdout. Root calibration recordings were excluded from STT fitting and selection. References are programmatic and require human review before deployment.

Training used seed 17, AdamW, CTC, isolated-word warmup followed by full-command training. It ran 32 epochs, selected epoch 22 using development data only, and took 413.7 seconds including export/checks after features were cached. No earlier STT checkpoint initialized the model. GPU training and CPU inference are distinct. Checkpoint selection equally weighted development word, constructed, natural and negative accuracy; natural/negative abstentions are included in that selection score, so it is not a positive-command recall metric.

## Accuracy: raw recognition and deployed acceptance

“Raw” applies the common brightness grammar to greedy text before confidence gating. Recall counts correct numeric values out of absolute-brightness references. Precision counts correct values out of emitted values. These measure the numeric slot; the existing pipeline regression additionally checks complete target/action/argument JSON.

| Test stratum | Positive / total | STT raw correct values | ASR raw correct values |
| --- | ---: | ---: | ---: |
| Constructed, held-out components | 1010/1010 | 716/1010 (70.89%) | 44/1010 (4.36%) |
| Natural, shared held-out | 3/87 | 0/3 (0.00%) | 1/3 (33.33%) |
| Natural, ASR training overlap | 3/21 | 0/3 (0.00%) | 3/3 (100.00%) |
| Unrelated human commands | 0/250 | 0/0 (n/a) | 0/0 (n/a) |

| Operating point, all 1,368 command examples | STT | ASR |
| --- | ---: | ---: |
| Raw value recall | 716/1016 (70.47%) | 48/1016 (4.72%) |
| Raw emitted-value precision | 73.36% | 24.37% |
| Confidence-gated value recall | 113/1016 (11.12%) | 0/1016 (0.00%) |
| Confidence-gated precision | 94.17% | n/a |
| Full VCM + reader value recall | 42/1016 (4.13%) | 0/1016 (0.00%) |
| Full VCM + reader precision | 95.45% | n/a |

Thresholds were locked before test evaluation. STT's development-selected minimum CTC-token confidence is 0.999163924; it achieved 83/87 correct development emissions. ASR could not meet the same ≥95% precision with ≥20 development emissions, so its brightness acceptance threshold is 1.1: **its selected brightness path abstains**, while raw scores above expose its actual recognition performance. This does not disable ASR transcription diagnostics or change its weights.

On test, STT emitted 120 values, of which 113 were correct; the 94.17% precision missed the 95% target. VCM gating reduced correct emitted values to 42/1016. This is a functioning experimental integration with low accepted recall, not a reliable brightness controller. The natural VAD replay also recovered 0/3 numeric commands with STT. Do not infer natural-speech or all-101-value reliability from parser coverage.

| Constructed phrase form, confidence gated | STT | ASR |
| --- | ---: | ---: |
| brightness <x> | 64/505 (12.67%) | 0/505 (0.00%) |
| brightness <x> percent | 49/505 (9.70%) | 0/505 (0.00%) |

## Recognition speed

Measured on Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, CPUExecutionProvider, two ONNX intra-op threads and one inter-op thread. Both readers receive identical preloaded PCM in alternating order. The final timing set contains 202 utterances (all 101 values and both suffix forms), repeated three times after warmup.

| CPU parameter recognition | STT | ASR |
| --- | ---: | ---: |
| Warm p50 | 2.255 ms | 2.944 ms |
| Warm p95 | 3.274 ms | 4.283 ms |
| Median real-time factor | 0.00125 | 0.00161 |
| First session construction (one observation) | 72.81 ms | 15.71 ms |

STT saved 0.688 ms at the median. Timing includes frontend, ONNX, greedy decoding and parsing; it excludes file reads, speaking time and VAD waiting. The default 400 ms endpoint wait is much larger than this model-speed difference. Session construction is separate and includes Python/model initialization with unspecified filesystem cache state; it is not a cold-device boot benchmark. Neither Pi4 nor live microphone latency was measured.

## Checks and reproducibility

The exported STT passed Torch/ONNX parity at input lengths 100, 201 and 411, plus a mixed-length batch [301, 173, 100]; maximum absolute error was 1.2e-05. Single/batch valid-frame outputs also agree. Shared tests exercise all 404 textual numeric forms (digits/spoken words, with/without percent), malformed inputs, UNKNOWN intent, low-confidence abstention and prevention of cross-model fallback. Routing fixtures are labeled separately from real graph execution.

```bash
python scripts/brightness.py fetch
python scripts/brightness.py prepare --refresh-data
python scripts/brightness.py train
python scripts/brightness.py benchmark
python scripts/brightness.py install
```

The manifest, recordings and acquisition receipt must be retained to reproduce the run. Training uses the root environment. Add a source with a documented JSONL overlay and explicit `--refresh-data`; examples and boundaries are in `AGENTS.md`. New recordings should cover full natural phrases, independent speakers, 0/100, tens versus teens, target names, noise and percent omission. Refit only from training/development data and reserve a new untouched test set before further iteration.

Evidence: [training](../../.cache/stt-brightness/training_result.json), [split audit](../../data/stt/brightness_audit.json), [locked selection](../../.cache/stt-brightness/selection_locked_before_test.json), [full benchmark including all 101 values](../../.cache/stt-brightness/benchmark.json), [per-recording predictions](../../.cache/stt-brightness/test_predictions.json), [artifact checks](../../artifacts/brightness-validation.json).

## Sources

- [SLURP dataset and official tooling](https://github.com/pswietojanski/slurp)
- [Snips research datasets](https://github.com/snipsco/snips-research-datasets)
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- [MLCommons Multilingual Spoken Words Corpus](https://mlcommons.org/datasets/multilingual-spoken-words/)
- [Pinned MSWC English source](https://huggingface.co/datasets/MLCommons/ml_spoken_words/tree/0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218)
- [Earlier dataset survey and MLEnd provenance](../REPORT.md)

These findings support selectable local model shipping and a bounded vocabulary experiment. They do not establish satisfactory natural brightness-command accuracy.

## Completed integration and shipping verification

STT-only, ASR-only and dual-model ZIPs passed isolated executable checks. The single-model packages contain VCM plus only their selected parameter graph. Both runtime overrides passed in the dual package, and selecting an absent backend failed explicitly. No training checkpoint is included. VCM and ASR graph hashes match the pre-change snapshot.

Full repository validation checked 39,951 original audio files and replayed 4,314 held-out recordings. All selected software checks passed. The full pipeline emitted 91 commands; 87 matched complete reference JSON, giving 95.60% precision and 4.88% finite-command recall. **The existing overall quality gate remains failed.** These full-dataset metrics cover all command families and include VAD, unlike the isolated numeric-slot benchmark above.

Artifacts: [default STT ZIP](../../dist/smart-speaker-rpi4.zip), [ASR ZIP](../../dist/smart-speaker-asr.zip), [dual-model ZIP](../../dist/smart-speaker-both.zip). Each has a matching `.sha256` file and `.build.json`. [Package-switch evidence](../../.cache/stt-brightness/package-checks/results.json) and [full validation](../../.cache/validation/latest.json) are local generated evidence. ARM64 wheels are bundled; execution was verified on this Windows workstation, not a Raspberry Pi.
