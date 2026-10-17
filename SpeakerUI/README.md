# Iris display

The Svelte display shows microphone/listening state, tentative finite intent and slots, the final accepted or rejected command, and an execution receipt. Use-case panels expose the same typed action routes through explicit touch controls. The product has no decorative aura or simulated device state.

Start from the repository root with `python start.py --mode serve`. The root launcher supplies the selected models, private configuration and resident Python CPU worker, and serves the compiled website. Pi packages include the compiled UI. See the [root guide](../README.md), [operator runbook](../docs/HANDOFF-iris-rpi-v1.md), [requirements](../REQUIREMENTS.md), [configuration template](../config.example.toml) and [release evidence](../docs/EXPERIMENT-iris-release-v1.md).

Open **Microphone settings** in Iris to select a host microphone, mute, replay a mono 16 kHz WAV and inspect exact worker JSON. The history view preserves recent final results. Recognition uses direct finite commands with local wake/VAD and an experimental voice-similarity check. Original-human wake/command and physical Pi qualification remain separate operator work.

Only accepted final microphone commands with passed owner evidence may dispatch configured actions. Tentative results, WAV replay, fixtures and `benchmark-compatible` mode are action-free. Explicit touch controls still require configured action capability. `ACTIONS_ENABLED=false` and recognizer/deployment disablement suppress actions. The bare development gateway defaults to explicitly labelled mock transport; production uses the root launcher, which requires native worker readiness and an installation token.

Configure providers and finite bindings in the root `config.toml` and private root `.env`; [.env.example](.env.example) documents direct-gateway settings. Home Assistant handles bound lights/groups, climate, weather, media players and user-supplied contact/preset-message scripts. Local MP3 playback requires an explicitly chosen local provider. Missing bindings remain unconfigured, and provider outcomes are reported without invented observed device state or automatic retry after an unknown outcome. Secrets remain server-side.

Timers, alarms, finite reminder tasks and action identifiers persist in `DATA_DIR/action-state.json`. Future jobs restore after restart; expired jobs are marked missed. `TIMEZONE` selects the IANA zone for the local clock and TIME receipt, defaulting to `Asia/Manila`; ALARM retains the timezone in its validated finite command.

For source changes, use Node 24.13 or newer and the configured shared dependencies:

```sh
cd SpeakerUI
npm test
npm run check
npm run build
```

The build runs Vite, asset compression and TypeScript. `scripts/browser-runtime-check.mjs URL INSTALLATION_ROOT OUTPUT_DIRECTORY --server-pid LOCAL_GATEWAY_PID --real-only` checks the compiled UI with real worker WAV events at 800×480, 1024×600 and 390×844. It records replay-only receipts, unchanged action state, observed Node/worker RSS and CPU, browser heap/DOM/assets and upload-to-readout timings. Supply `--recording HASH_BOUND_REFERENCE_JSON` for a permitted external replay reference. These are host software measurements, not live microphone or Pi measurements. `scripts/iris-browser-check.mjs` separately tests typed transport fixtures and local jobs; it is not acoustic evidence.

The supported complete source check is `python train.py --skip-train` from the repository root after selected exports are available. It checks the selected lineage, actual CPU engine parity, tests, UI build and package. Use the package's action-free `benchmark.py` for operator replay/resource measurements; see the root runbook for its manifest and output contract.

The separate `?diagnostics` page preserves explicitly **mock audio transport**, browser PCM capture and manual media checks. Its mock audio never recognizes speech or dispatches providers; native-worker mode rejects that legacy audio socket. Use the Iris settings panel for native WAV/microphone diagnostics. Manually selected media playback is a separate configured-provider operation.

Earlier Ambient designs, decorative visualization, simulated interaction tests and their screenshots are historical evidence. Recovery uses the immutable [archive commit](https://github.com/justinruaya123/smart-speaker/tree/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI) and [path/hash manifest](../experiments/iris-release-v1/archive-manifest.json). Historical [design](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/docs/design/ambient.md) and [testing summary](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/docs/testing/ambient/README.md) do not describe the current Iris product. The seven retired QA scripts remain available there:

- [Alerts](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/scripts/qa/alerts.cjs)
- [Camera](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/scripts/qa/camera.mjs)
- [Interactions](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/scripts/qa/interactions.cjs)
- [Parity](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/scripts/qa/parity.cjs)
- [Providers](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/scripts/qa/providers.cjs)
- [Resources](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/scripts/qa/resources.cjs)
- [Startup](https://github.com/justinruaya123/smart-speaker/blob/ae1f1d7c67616e4efbae4619ac75620495e6073a/SpeakerUI/scripts/qa/startup.mjs)

Keep [third-party notices](mockups/ambient-v6/licenses); the compression step copies them into `dist/licenses`. Historical source and negative results remain recoverable through the archive.
