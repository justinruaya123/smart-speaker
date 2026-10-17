# Local handoff: compact VCM comparison on Raspberry Pi 4

Scope: local preparation only. Both model archives are frozen; no Pi connection, model installation, new training, threshold fitting or microphone use is performed by this handoff. Actions and strict qualification remain disabled. Run one model at a time in separate extracted folders.

## Files to transfer

Download each ZIP and its matching checksum file into the same directory on the Pi.

| Package | Model | ZIP | Checksum |
| --- | --- | --- | --- |
| Control | Mixed epoch 32, 935,567 parameters | [speaker-pi4-compact-final.zip](../dist/speaker-compact-recall-v1/speaker-pi4-compact-final.zip) | [ZIP SHA-256](../dist/speaker-compact-recall-v1/speaker-pi4-compact-final.zip.sha256) |
| Candidate | Mixed epoch 28, 935,567 parameters | [speaker-pi4-pipeline-repair.zip](../dist/speaker-pipeline-repair-v1/speaker-pi4-pipeline-repair.zip) | [ZIP SHA-256](../dist/speaker-pipeline-repair-v1/speaker-pi4-pipeline-repair.zip.sha256) |

Control SHA-256: `185e81a85e0c637494784c9a40fb3e65eabb77094995bdad500247399dad920d`.

Candidate SHA-256: `6923a1636a4b506c2b75b0c5308a752b8f714dce5f0ce5a732498c282dbfe4dc`.

The full VCM graph is 3,864,875 bytes (3.69 MiB) in each package, with unchanged architecture and no new training. Both archives contain the same 15 original licensed smoke recordings: 13 supported commands and two UNKNOWN examples. They include offline target wheels and wake/VAD assets; training checkpoints, training code and PyTorch are excluded.

## Target and setup

The frozen offline wheels require **64-bit Linux ARM64 (`aarch64`), CPython 3.12 and glibc >=2.28**. A Pi 4 running 32-bit Linux or a different Python version does not match this package. The interpreter/venv support and documented PortAudio/libsndfile system prerequisites must already be available; the bootstrap does not provision the operating system.

On the Pi, inspect the target first:

```sh
uname -m
python3.12 --version
ldd --version | head -n 1
```

From the directory containing both transferred ZIPs and checksum files:

```sh
sha256sum -c speaker-pi4-compact-final.zip.sha256
sha256sum -c speaker-pi4-pipeline-repair.zip.sha256
unzip -n speaker-pi4-compact-final.zip -d control-epoch32
unzip -n speaker-pi4-pipeline-repair.zip -d candidate-epoch28
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Use fresh extraction directories. `unzip -n` preserves any existing files; the following integrity check detects mismatched contents. Each package gets its own local `.venv`. Setup uses bundled wheels with `--no-index` and required hashes; recognition needs no network connection.

## Run the same recorded comparison

Control:

```sh
cd control-epoch32
python3.12 pi_test.py --verify-only
python3.12 pi_test.py --setup --engine onnxruntime --mode demo --threads 1 --output results/pi4-demo-01
.venv/bin/python pi_test.py --engine onnxruntime --mode qualified --threads 1 --output results/pi4-qualified-01
cd ..
```

Candidate:

```sh
cd candidate-epoch28
python3.12 pi_test.py --verify-only
python3.12 pi_test.py --setup --engine onnxruntime --mode demo --threads 1 --output results/pi4-demo-01
.venv/bin/python pi_test.py --engine onnxruntime --mode qualified --threads 1 --output results/pi4-qualified-01
cd ..
```

Keep the same Pi, power/cooling, CPU settings, thread count and background load. Each output folder must be new; use `-02` for a later run. Where available, save `vcgencmd measure_temp` and `vcgencmd get_throttled` before and after the runs.

Inspect each `results/pi4-demo-01/metrics.json`, `trials.jsonl` and `REPORT.md`. Confirm the device model and `hardware.physical_pi_measured`, `decision_mode`, model binding, one CPU thread and sample denominators. Compare complete-command correctness, UNKNOWN accepts, latency p50/p95, session loading and peak RSS. File replay does not test wake, microphone, speaker ownership or overlap. Its latency excludes file reading, spoken duration, wake listening and endpoint silence.

Qualified mode is expected to reject everything because human qualification is disabled. Demo mode uses the graph-bound per-intent policy; the `.95` value in `config.toml` documents the fixed baseline and does not override that policy. Neither result enables device actions.

## Optional microphone check, performed later on the Pi

From the chosen extracted folder, list the actual Pi inputs:

```sh
.venv/bin/python pi_test.py --list-mics
```

Select the index from that list, then run in Bash:

```bash
read -r -p "Pi microphone index from the list: " PI_MIC_INDEX
.venv/bin/python pi_test.py --engine onnxruntime --mode demo --threads 1 --guided --mic "$PI_MIC_INDEX" --seconds 25 --output results/pi4-live-01
```

The guide requests a wake phrase and command; press `q` to stop. Completed live captures, capture failures and skipped/untested cases remain separate from recorded replay in the results. A requested prompt is not independent verification of what was spoken. Keep the selected graph and thresholds unchanged during this comparison. Wake success does not establish speaker ownership or overlap recovery.

## Interpret the tradeoffs

| Previously examined human test | Control | Candidate |
| --- | --- | --- |
| Correct accepts | 229/636 (36.01%) | 238/636 (37.42%) |
| Accepted-command precision | 91.60% | 92.61% |
| UNKNOWN false accepts | 2/229 | 1/229 |
| Raw complete commands | 493/636 | 489/636 |
| BRIGHTNESS correct accepts | 5/23 | 0/23 |
| VOLUME correct accepts | 26/113 | 21/113 |

The candidate's aggregate gain is modest and uneven. Synthetic correct accepts also fall 3,099 -> 3,078, and human OptionB wrong accepts rise 3 -> 4. Both archives' previous desktop smoke replay accepted only 2/13 commands correctly; matching all 15 locked software decisions is parity evidence, not 100% command accuracy. These archives support a device comparison, not a qualified all-intent release.

The [full per-intent report, unchanged augmentations and citations](EXPERIMENT-speaker-pipeline-repair-v1.md), [frozen study verification](../experiments/speaker-pipeline-repair-v1/verification.json), [extracted desktop execution](../experiments/speaker-pipeline-repair-v1/package-verification.json) and [local handoff receipt](../experiments/speaker-pipeline-repair-v1/local-handoff-v2.json) distinguish measured software evidence from the still-unmeasured Pi/live results. The old control, installed bundle, model architecture and frozen selection/calibration evidence remain preserved.
