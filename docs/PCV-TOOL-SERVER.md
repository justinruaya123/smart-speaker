# PCV streaming tool-server integration handoff

Implementation status: a tested CPU-only streaming adapter and portable diagnostic package path exist. Trained v2 accuracy, dynamic-name acceptance, the final selected archive and physical Pi measurements are pending. Use [the delivery plan](PCV-DELIVERY-PLAN.md) as the remaining-work ledger and [the model documentation](PCV-MODEL.md) for computation details. Do not integrate the artificial test model as a qualified recognizer.

## Ownership and boundaries

The Pi process recognizes audio and returns typed commands. Your frontend/tool server owns microphone permissions, resampling to mono 16 kHz, presentation, authentication, inventory API access, tool definitions and any action authorization. This repository does not execute lights, music, calls, messages or timers. An accepted finite command is an input to your tool layer, not permission to invoke an arbitrary API.

Use the included service as a local inference worker and proxy it from the tool server. Default address is `127.0.0.1:8765`, with newline-delimited JSON over TCP. This avoids adding a web stack to the Pi recognizer. A WebSocket frontend bridge may forward 40 ms frames and events; it must retain sample ordering and cancellation semantics. Cross-host binding requires `PCV_SERVER_TOKEN`; use an authenticated TLS proxy or secure tunnel for transport privacy. The recognizer itself does not provide TLS.

## Install and start

After receiving the final immutable ZIP, extract it on 64-bit ARM64 Linux. Install OS `libsndfile1`, `libportaudio2` and CPython 3.12 first; the archive's wheels target CPython 3.12/glibc >=2.28. Run `sh setup.sh`. This verifies packaged file hashes, creates `.venv` and installs only bundled, hash-pinned wheels. No PyTorch/GPU/TTS install is involved. Run `sh serve.sh` to start the packaged bundle. A diagnostic ZIP passes `--diagnostic` explicitly and keeps command acceptance disabled.

An enabled ZIP also contains `portable-parity.json`, binding complete development waveform comparisons to the model and deployment adapter sources. The publishing agent must run `scripts/pcv_portable_parity.py` after assembling the portable bundle and provide `--portable-parity <results.json>` to `publish.py --pcv-runtime`. Changing the adapter after that measurement requires a new comparison. This is separate from the speech-accuracy and physical Pi benchmark evidence.

Alternate validated bundles live in sibling directories under `models/`. A direct invocation is `.venv/bin/python -m scripts.runtime.pcv_deploy serve --bundle models/default --releases models`. The original `--bundle` remains the restart default; hot reload changes the current in-memory generation, not that command line. Treat artifacts as immutable, store a rollback copy, and preserve their receipts.

## Wire contract

Each message and response is a UTF-8 JSON object followed by `\n`. Maximum message length is 16 KiB. Each connection starts with:

```json
{"type":"hello","protocol":"pcv-stream-v1","authentication_required":false,"provider":"CPUExecutionProvider","threads":1,"actions_executed":false}
```

When authentication is required, first send `{"type":"auth","token":"<configured token>"}`. Never log the token. The server responds with `{"type":"authenticated"}`. Invalid authentication closes the connection.

Start one stream:

```json
{"type":"start","stream_id":"living-room-001"}
```

The `started` response includes the immutable bundle binding, sample format and limits. The entire stream pins that model generation. Supply signed little-endian PCM16, mono, 16,000 samples/second. Use 640 samples/40 ms per message; maximum is 1,600 samples/100 ms. Encode bytes with standard base64:

```json
{"type":"audio","sequence":0,"pcm_s16le":"<base64 bytes>"}
```

`sequence` is a sample index, not a message number or timestamp. The next message after 640 samples uses `sequence:640`. It must exactly equal the preceding index plus preceding sample count. Do not silently drop frames under backpressure or restart numbering within the same stream. `overflow:true`, a bad sequence, malformed base64 or an oversized/odd-byte payload cancels the stream with a capture fault.

Each audio response contains `next_sequence`, `processing_ms`, model binding and zero or more events. These acknowledgments let the bridge keep at most a small bounded number of frames in flight. Wait for acknowledgments when testing; a production bridge may use a bounded queue but must report overflow rather than accepting a discontinuous stream. One active stream is permitted globally, with at most eight control connections. Idle connections time out after thirty seconds; disconnects cancel owned state.

Stream initial context continuously before inviting speech. The wake detector requires 1.28 seconds of real feature startup; the file benchmark explicitly uses two seconds. Display listening/ready status based on incoming audio, not merely successful TCP connection. Then prompt the user to say **Hey Jarvis**, wait for `command_ready`, and speak one verbatim. The wake gate emits:

```json
{"stage":"wake","status":"detected","wake_phrase":"hey jarvis","score":0.81}
{"stage":"wake","status":"command_ready"}
```

After a real command endpoint, the final event is shaped as:

```json
{"schema_version":2,"stage":"command","accepted":true,"command":{"intent":"BRIGHTNESS","entity_id":"light.kitchen","value":37},"reason":"accepted","wake_required":true,"wake_phrase":"hey jarvis"}
```

This is a schema example, not a measured successful output. Rejected events have `accepted:false`, `command:null` and an explicit reason. Forward both successes and failures to logs/UI. Do not tool-call on `wake`, `command_ready`, score-only diagnostics, a tentative hypothesis or a non-null raw candidate in a research report. Only a final accepted command event may reach the tool dispatch validator.

Use `{"type":"end"}` to close the stream logically. EOF/end does not force a command endpoint and may return `incomplete`; it must not trigger a tool. Use `{"type":"cancel"}` to discard state. Each audio/end/cancel message optionally carries the owning `stream_id`; it cannot target another connection's stream. A disconnect frees its slot and discards pending speech. Following each final command, a fresh wake word is required even if the TCP stream continues.

## Bundle reload and rollback

Place a complete validated bundle in `models/<release-id>` without changing the active directory. Release IDs contain lowercase ASCII letters, digits and hyphens, up to 64 characters. Send:

```json
{"type":"reload","release":"release-002"}
```

The server validates files, policy/catalog/model bindings and measured evidence, loads CPU sessions, then changes the current snapshot. Existing speech continues against its pinned snapshot; new streams use the new binding. A failed load returns `reload_failed` and leaves the previous model available. Only one replacement may wait behind an active older stream, keeping retained generations bounded. `{"type":"rollback"}` restores the preceding in-memory bundle; it does not mix caches from two models. `{"type":"status"}` reports binding, busy state and diagnostic status.

The bundle includes the graph, normalization, endpoint, catalog, policy and wake assets. Replacing `model.onnx` alone is invalid because its score distribution, normalization or calibrated policy may differ. Never copy old thresholds onto a new catalog and declare it qualified.

## Tool mapping and CSV inventories

Validate the emitted schema and catalog binding before dispatch. Illustrative mappings include `MEDIA_PLAY` with `track_id`, `MUSIC_PLAYLIST` with `playlist_id`, `POWER_ON/OFF` with `entity_id`, `BRIGHTNESS` with integer `value` 0-100, `COLOR_SET` with named `color`, `TIMER_SET` with `duration_seconds`, `REMINDER_SET` with delay or local time/timezone, and fixed alarm, thermostat, media, query and contact/message operations. The full supported JSON is in [the verbatim guide](PCV-SYNTHETIC-V2-VERBATIMS.md). A query intent does not contain an externally fetched answer, and fixed message IDs are not arbitrary dictated message content.

Start inventory interchange with this CSV schema:

```csv
kind,id,name,artist,aliases,capabilities
song,track.bts_butter,butter,BTS,,
song,track.bts_stay,stay,BTS,,
song,track.blackpink_stay,stay,Blackpink,,
light,light.kitchen,kitchen lamp,,,power|brightness|color
light,light.bedroom,bedroom lamp,,lamp two,power|brightness
```

Use normal CSV quoting for commas inside fields. Name aliases and capability lists use `|` within their own fields. Stable IDs are the integration identity; names are spoken labels. The compiler currently requires bounded English word spellings and a text pronunciation dictionary. Unknown pronunciations and conflicting phonetic names fail; supply a verified dictionary/alias instead of invoking transcription. Duplicate `stay` rejects without an artist. A title paired with an unsupported artist must be tested as a negative, not resolved by title substring matching.

Your Spotify/Home Assistant adapter may export this CSV from authenticated inventory reads in a separately authorized integration task. Keep credentials in that adapter, outside model assets, recordings and Git. Do not assume the current deployment supplies those API clients. Compile via `python scripts/pcv_delivery.py catalog --csv <inventory.csv> --lexicon <lexicon.json> --base-catalog <current-catalog.json> --output <new-catalog.json>` on the development host. The optional base-catalog argument replaces inventory-backed positive commands while retaining other use cases and explicit rejections; a new name conflicting with a prior rejection requires a versioned amendment. Compilation alone creates an uncalibrated candidate. New-name held-out acoustic testing, catalog competition calibration and a bound release remain required before hot activation; that dynamic validation workflow is pending implementation.

## Benchmarks and acceptance checklist

Use `.venv/bin/python -m scripts.runtime.pcv_deploy benchmark --bundle models/default --wav <continuous-wake-command.wav> --repeats 5` on the Pi. Preserve its JSON, exact WAV/model/catalog hashes, Pi model/RAM/OS/temperature/governor and background workload. Report p50/p95 compute, real-time factor, load time and peak RSS separately from speech duration and the 700 ms endpoint wait. Repeated replay is timing evidence, not independent accuracy trials. The CLI reads the device-tree model before labeling physical Pi measurement; ARM64 alone is insufficient.

Integration tests must cover missing/wrong wake words, incomplete speech, repeated/noisy wake attempts, unknown names, duplicate titles with/without artists, brightness endpoints and invalid values, ambiguous clock times, lost/repeated/out-of-order chunks, cancellation/disconnect, concurrent clients, failed reload, reload during speech and rollback. Check that no test invokes a physical device. Then use the frozen demo list and full error report with their real denominators. Physical microphone and human-quality claims require separate measured evidence; neither archive creation nor this handoff proves them.
