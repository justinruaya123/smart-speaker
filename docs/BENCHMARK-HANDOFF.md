# Portable recorded benchmark

[Download the GitHub-built benchmark ZIP](https://github.com/justinruaya123/smart-speaker/releases/latest/download/smart-speaker-benchmark-2m.zip). The source workflow builds it alongside inference and publishes a SHA-256 checksum and member-hash receipt. To build locally: `python -m scripts.submission.build_benchmark`. All 196 original WAVs are stored as the hash-bound source fixture `submission/benchmark/holdout-audio.zip`; the builder validates each recording against the frozen manifest and never fetches a prebuilt archive or dataset.

Run `python benchmark.py` from a Git checkout or extracted benchmark ZIP. No
`reproduce.py`, training checkpoint, PyTorch, microphone, audio playback or device
action is needed. Choose Raspberry Pi and enter its SSH address, or choose this
PC. Python bootstraps a project-local, hash-locked uv environment; setup needs
network access. The Pi needs 64-bit Raspberry Pi OS, `python3`, SSH access and
network for its first setup. An installed uv is reused, otherwise a pinned uv
binary is downloaded and verified locally. No sudo or global package install is
used. Windows may use `py benchmark.py`.

```sh
python benchmark.py --target pc
python benchmark.py --target pi
python benchmark.py --target pi --host pi@raspberrypi.local --output results/pi-01
# Optional comparison in a checkout that includes the compact model:
python benchmark.py --target pc --model-choice compact
```

The Pi command asks for the SSH address when `--host` is omitted. SSH uses the
system client and existing keys/agent or its normal password prompt. Optional
`--identity PATH` and `--ssh-port PORT` select a private key and SSH port. The first
connection records the host key using OpenSSH `accept-new`; changed host keys are
rejected. SSH aliases and `user@hostname`/IPv4 are accepted. An existing result
directory is refused so earlier evidence survives. Run again with a new output.

The PC obtains only the hash-pinned published holdout Parquet shard when original
WAVs are absent. The standalone ZIP includes its original held-out WAV bytes, so
its replay needs no dataset download. All 196 published recordings are retained,
with human and synthetic results separate; uncertain typed references retain
their predictions and are excluded from accuracy denominators. These references
were already examined during prior research. This is frozen recorded replay, not
the separate official live/acoustic benchmark or a new independent test set.

The runner transfers the inference runtime, selected model, source notices and
held-out WAVs into a unique `~/.cache/smart-speaker-submission/<run>/` directory on
the Pi. It verifies archive and individual-file hashes before importing code,
runs silent replay, and retrieves `metrics.json`, `REPORT.md`, `trials.jsonl`,
`timing.jsonl` and `telemetry.json`. `ssh-receipt.json` records the transfer hash
and remote path. SSH transport, authentication and actual Pi behavior still need
verification on the user's device.

The shipping default is the 2,066,831-parameter direct VCM, control epoch 32 from
`speaker-final-refinement-v1`. Its INT8 ONNX SHA-256 is
`3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e`.
The 2M ZIP ships only this selected model. `--model-choice compact` optionally
compares the 935,567-parameter repaired compact model when its folder is present
in the Git checkout. Both models use the exact same held-out recording references.
Raw candidates, demo decisions and qualified outputs are scored separately.
For the 2M model, demo acceptance requires .95 intent confidence and .90 for every
required slot/target; the compact comparison retains its bound per-intent recall
policy and .90 slots. Raw recognition accuracy is different from accepted-command
recall: a correct finite candidate can still be rejected by a confidence gate.
The report explicitly counts correct raw commands rejected by each demo gate.
Qualification and actions remain disabled. The benchmark reports the VCM graph
and bundle identity; shipped wake/VAD assets are preserved but not exercised by
this complete-utterance replay.

Inference timing covers the frontend, CPU ONNX, finite catalog decoder, typed
validation and demo policy. It excludes file decoding, speech duration, wake/VAD,
endpoint waiting, setup and SSH. Default settings use one ONNX thread and, where
supported, one available logical CPU; use `--threads 2` and `--cpu N` explicitly
for other local measurements. Three repeated timings per recording follow one
warmup recording. Temperature, CPU frequency and Pi throttling are captured when
the host exposes them. Peak RSS, session loading, CPU/wall time, timing
distributions and real-time factors are recorded. Desktop results remain desktop
evidence; Pi hardware identity requires both ARM and a Raspberry Pi model string.

`--limit 5 --repeats 1` runs a marked smoke subset. Omit `--limit` for the complete
benchmark. `--no-setup` reuses an already prepared environment.

The independent 2M benchmark archive and immutable public download URL are listed
in [the release download receipt](../submission/release/downloads.json); the latest
benchmark filename is `final-submission-benchmark-2m-v3.zip`. Earlier archives are
preserved. Benchmark setup and execution never require `reproduce.py`.

From the source checkout, feed a returned Pi receipt into the README/slide figures:

```sh
uv run --group training python -m scripts.submission.report --benchmark results/pi-01/metrics.json --update-readme
```

The report accepts Pi fields only when ARM hardware, a Raspberry Pi model string
and the selected graph hash agree. PC timing remains separately labeled. Silent
replay leaves live microphone and wake/keyword accuracy unmeasured. The numerical
frontend and CPU benchmark use one BLAS thread, independently of the ONNX thread
option; this fixes small feature-rounding changes near confidence gates.
