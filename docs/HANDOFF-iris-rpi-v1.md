# Iris Raspberry Pi handoff

This package supplies local finite-command recognition, an experimental wake-speaker check, a compiled product UI and executable configured actions. Physical Raspberry Pi performance and original human wake/command quality have **not been measured**. Software replay results belong in the [model card](../MODEL_CARD.md) and [release experiment](EXPERIMENT-iris-release-v1.md); they do not certify a microphone, room or owner gate.

## Obtain and verify

Use Raspberry Pi OS 64-bit on an RPi4/5, Python 3.11–3.13, a USB microphone and speakers. Download the Pi ZIP and SHA-256 receipt from [the model repository](https://huggingface.co/justinruaya123/ai231-smart-speaker). Check `sha256sum iris-pi.zip` against the published receipt. Extract to a new directory; preserve your private configuration/state separately from software upgrades. Recognition graphs, source and UI assets are covered by `MANIFEST.json`.

```bash
unzip iris-pi.zip -d iris
cd iris
python3 start.py --system-packages --list-mics
```

The first run installs pinned Python and local Node tools/dependencies and needs internet. `--system-packages` installs PortAudio/libsndfile on Debian/Pi OS. It can request sudo. Runtime recognition uses local ncnn/ONNX only; it does not contact a model service. The explicit `--engine onnxruntime` alternative is for comparison, with no automatic engine fallback.

Use the actual microphone index in the list. No browser microphone is sent to a cloud service; capture occurs on the machine running the resident worker. A browser on a laptop cannot select that laptop's microphone when the server is on the Pi.

## Configure entities and providers

The launcher creates `config.toml` from `config.example.toml`. Copy `.env.example` to private `.env` for secrets. Restart after changes. Precedence is CLI > process environment > `.env` > TOML > defaults. Keep both private files out of Git and publication.

Supply your Home Assistant URL, token, explicit entity allowlist and bindings. Example IDs below are placeholders that **must be replaced**:

```toml
[providers]
ha_enabled = true
ha_url = "http://YOUR_HA_HOST:8123"
ha_entity_ids = "light.room,light.kitchen,climate.room,weather.home,media_player.room,script.call_home,script.call_mom,script.send_preset"
data_dir = "state"

[bindings.lights]
"light.ceiling" = "light.room"
"light.all" = ["light.room", "light.kitchen"]
[bindings.songs]
"song.enemy" = "media-source://media_source/local/enemy.mp3"
[bindings.climate]
default = "climate.room"
[bindings.weather]
default = "weather.home"
[bindings.media_player]
provider = "home_assistant"
entity_id = "media_player.room"
[bindings.contacts]
"contact.home" = "script.call_home"
"contact.mom" = "script.call_mom"
[bindings.messages]
default = "script.send_preset"
```

```dotenv
HA_TOKEN=YOUR_LONG_LIVED_ACCESS_TOKEN
```

Bind only IDs supported by the selected acoustic catalog. Editing text cannot teach a new spoken name. Unmapped IDs return **unconfigured**; they never substitute a different target. Explicit light lists are checked member-by-member for entity domain, allowlist and capabilities. Calls/messages invoke your supplied scripts and finite preset behavior; Iris does not place arbitrary telephone calls or transcribe a message.

Time is read locally; the command alarm contract uses Asia/Manila. Timers, alarms and finite reminders are stored with execution identifiers in one atomic `state/action-state.json`. Keep this file private and back it up while the service is stopped. Future jobs restore after restart; expired jobs become missed. A pending provider action at a crash becomes outcome unknown and is not retried automatically.

For local music, explicitly choose `provider = "local"`, configure a permitted local media directory and song paths, then enable playback in one browser. Browser acknowledgement establishes playback outcome; missing acknowledgement becomes unknown. Other providers are explicit alternatives, without fallback.

## Start and operate

```bash
python3 start.py --mode serve --mic INDEX --benchmarks skip
```

Open the printed local UI address. For a remote browser, deliberately configure server host/origin and your network access; the default binds localhost. The main screen shows input selection, capture/listening state, tentative intent and slots, final recognition, jobs and execution receipts. Say **Hey Iris**, pause until command-ready, then speak one supported command. One wake permits one command and resets.

Tentative guesses are provisional and never execute. The final command must pass the VCM policy, independent wake-to-command speaker gate and typed validation. Overlap, another speaker, missing evidence, insufficient speech or timeout rejects. There is no persistent enrollment. Ownership is experimental, calibrated with short human-utterance pairs and synthetic wake proxies.

The locked speaker policy requires at least 0.8 seconds of detected voiced speech in its wake reference. Only 1/32 new standalone synthetic wake clips met that requirement; the other 31 returned `insufficient_speech`. The full session policy also rejected 62/80 same-speaker short-human proxy trials. These are substantial usability limitations, not original human wake measurements. Record these rejections during operator testing; command-only accuracy does not imply successful wake-to-action operation. The published thresholds were not lowered after these test failures.

A receipt distinguishes pending, succeeded, failed, unconfigured and unknown. A Home Assistant service acceptance is not confirmation that the physical device changed; inspect the separately observed state. Never automatically retry an unknown outcome. On reconnect, the browser receives the persisted jobs and receipts.

## Recorded checks: no actions

```bash
python3 start.py --profile benchmark-compatible --wav command.wav
python3 start.py --wav wake-and-command.wav --wake-wav
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python benchmark.py --bundle models/default --output results/pi-smoke --threads 1 --engine ncnn
.venv/bin/python benchmark.py --bundle models/default --output results/pi-ort --threads 1 --engine onnxruntime
```

WAV inputs require mono 16 kHz PCM. Command replay is limited to 15 seconds and wake replay to 60 seconds. These paths never execute. The bundled diagnostic set contains 88 original-designed synthetic recordings (36 personal positives and 52 negatives), not the full official 196-row benchmark or fresh human evaluation. Supply `--manifest /path/to/commands.jsonl` for a separately licensed recorded population; each row includes `id`, relative `audio`, `sha256`, and `expected` typed command or `null`. Keep test manifests fixed.

`metrics.json` records accuracy, false accepts/rejects, p50/p95 full recognition time, real-time factor, session load, peak RSS, host identity and available temperature data. Recorded inference excludes disk/decode, utterance duration and endpoint waiting; report those separately in live tests. Run the unchanged 196-row scorer from the source workflow (`python train.py --skip-train`); its restricted original audio is acquired at pinned upstream revisions and is not newly included in the Pi ZIP.

The five environment settings bound numerical-library threads for comparable command-line timings; `--threads` controls the inference engine. The resident UI worker defaults both to its configured thread count and preserves explicit environment overrides. Keep these settings identical when comparing engines, and retain them with the measurements.

## Operator acceptance record

Record OS, Pi model/RAM, CPU governor, cooling, model/ZIP hashes, engine, threads, microphone/index, capture rate, distance, room/noise and speaker identifiers. Keep these trials separate:

1. Replay the bounded diagnostic set under each explicit engine, then the licensed fixed benchmark if available. Save metrics and trials; compare decisions and actual package hashes.
2. Test original humans of varied voices: standalone command-only, Hey Iris alone, and wake → pause → command. Count correct accepts, wrong commands, false rejects, false wakes, duplicate wakes and timeouts with denominators. Do not relabel prompted synthetic accents as demographic evidence.
3. Test one speaker's wake and command, another speaker's command, a speaker switch inside the command, real overlap, very short/quiet speech, no command and immediate second command. Check reset after acceptance, rejection, timeout and mute. Keep actions disabled during initial microphone qualification.
4. After checking supplied entities, test one intended action per category. Confirm actual device state separately from provider acceptance. Test unavailable provider, unmapped target, duplicate final event, reconnect and server restart. Verify future jobs restore, expired jobs are missed and unknown outcomes do not retry.
5. Measure live wake latency, endpoint delay, command processing, complete wake-to-receipt latency, CPU utilization and peak RAM. Run a sustained workload and record temperature/throttling (`vcgencmd measure_temp`, `vcgencmd get_throttled` where available). Report p50/p95, failures and sample counts.
6. Disable network access and replay/recognize local commands. Time/timers remain local; weather/HA actions require their configured service. Restore only the network needed for those providers.

Do not call the product human/Pi qualified until these original-audio, live-device and thermal results exist. Add operator receipts as a new evidence version; do not retune published thresholds on the held-out test.
