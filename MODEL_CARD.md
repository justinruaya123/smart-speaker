# Semifinal constrained-phoneme VCM

Status: **diagnostic, not qualified; automatic acceptance disabled**. This is a new randomly initialized project model, not recovered `phone_s53`, not the installed historical v12 model, and not a general sentence transcriber. No externally pretrained acoustic weights, LLM, cloud inference or action executor is used.

The model has 113,064 parameters and one 468,395-byte FP32 ONNX graph. It consumes 40 causal HTK log-mel channels at 100 frames/second and emits 39 phone logits plus CTC blank. Six explicit neural caches require 32,064 bytes. The frontend uses 16 kHz mono PCM, a periodic 400-sample Hann window, 160-sample hop, FFT 512, 20–7600 Hz mel bands and floor 1e-8. Training-only mean/std are frozen. Whole-command CTC likelihood ranks finite, typed catalog entries; there is no free sentence output or device execution.

Two primary seeds (53, 71; 12 epochs) and one duration ablation (same seeds; 36 epochs) were run. A fixed 64-recording development subset selected checkpoints, with no loss of previously correct development commands permitted. Seed 71, epoch 18 was selected at 47/64. This small development subset is a limitation; speaker-group training cross-validation was not run. Full histories, rejected epochs, protocol and selection receipts are in [reports/semifinal-v1](reports/semifinal-v1).

Calibration used 1,835 separate recordings: 1,635 FSC requests and 200 isolated-word negatives. The best tested thresholds satisfying observed precision >=99% and negative false accepts <=1% emitted 301 commands, 298 correct, but recalled only 18.23% of valid requests. The required >=90% recall failed. The locked deployed policy rejects every command; diagnostic candidates remain visible.

Fresh waveform replay of the existing corpus test partitions covered 3,993 recordings. Raw exact commands: 2,752/3,793 valid requests (72.55%), or 2,752/2,925 within the implemented subset (94.09%). The remaining 868 valid requests stay in the denominator as misses. Deployed correct accepted commands: 0/3,793; emissions: 0; accepted precision: undefined; negative false accepts: 0/200. Zero emissions is not successful recognition. These are dataset test results, not a new independent human recording campaign.

| Reference family | Raw exact / requests | Correct accepted |
|---|---:|---:|
| Power on | 461/472 | 0 |
| Power off | 339/353 | 0 |
| Media play | 76/89 | 0 |
| Media resume | 32/32 | 0 |
| Media pause | 58/59 | 0 |
| Media stop | 44/46 | 0 |
| Relative volume | 698/741 | 0 |
| Relative temperature | 1044/1133 | 0 |
| Unsupported valid requests | 0/868 | 0 |

Absolute numbers, connected color commands, new contact/device/music names, timer/alarm, general queries, reminders and messages are not acoustically established. Symbolic numeric/catalog fixtures are separate. [The use-case map](configs/intents.json) retains all ten assignment categories. No all-intent completion claim is made.

On the measured Windows AMD64 host, 12 development WAVs and three warmed repetitions per thread setting gave processing p95 41.34/42.08/43.05 ms for 1/2/4 threads. Maximum observed process peak working set was 128,118,784 bytes (122.18 MiB), including all timing runs. Disk decoding, utterance duration and endpoint waiting are excluded. These are desktop measurements, not Pi acceptance. Nonzero-cache PyTorch/ONNX parity passed for 1/3/4/8/17 frames, maximum absolute difference 1.526e-5 against a predeclared 2e-5 bound.

Missing release evidence: independent human confirmation, realistic noise/far-field and continuous negative audio, physical Pi and microphone measurements, all-ten-use-case coverage, satisfactory acceptance recall, and training-pool speaker-group cross-validation. Exact PCM overlap was audited; near-duplicate audio similarity and manual listening were not performed. Confidence and small-denominator limitations are in [uncertainty.json](reports/semifinal-v1/uncertainty.json).

The original run did not freeze feature-file hashes before training. A later audit verified every source audio/PCM hash, measured feature recomputation differences up to 1.90735e-6, and reproduced train normalization exactly from cached features. Strict feature equality failed for 32,526/32,543 arrays. Future runs enforce feature hashes; the post-run audit cannot retroactively establish historical cache identity. Both the failed strict audit and numerical follow-up are preserved in [DATASET.md](DATASET.md).

Use is restricted to academic research by the FSC source terms. See [DATASET.md](DATASET.md). The package contains no corpus audio, source recordings, PyTorch, trainers, legacy runtime, visualizer, or second acoustic graph. Fresh installation requires the hash-pinned dependencies; inference runs locally without a network connection.

Selected graph SHA-256: `871c4a218684d50d132d8458b178e30c53ebff92c82b194bf3adaaa8652b1599`.

Catalog identity: `d8eabaa0dd73060f18725575e18c4b0c20b8d44dbc109ab110615a7b6f7db3ea`.

Frontend identity: `4410aa8bebc1fd2f9433f94613e17ca26d7d29834199c68494cf440ac37f1bda`.
