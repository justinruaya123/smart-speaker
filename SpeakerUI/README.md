# Ambient / SpeakerUI

A dark, touch-first voice display for an **800 × 480 landscape Raspberry Pi 4 screen**, built with Svelte 5 and Vite. The approved prototype is now implemented as Svelte components: centered live aura, glass controls, gesture-driven reminders, lights, thermostat, timers/alarms, media and Home Assistant camera views.

**Production inference uses the smart-speaker Python CPU ONNX worker.** The integrated launcher supplies its complete model bundle, serves the compiled frontend and forwards real states/typed results. Actions are disabled while validating inference. The explicit development mock mode retains fixtures. Enabled Home Assistant and Spotify integrations supply live state and touch controls; inference fixtures, weather and audio calls remain simulated. Allowlisted Home Assistant cameras can stream live video. Spotify playback occurs on its active/configured Connect device, not in this browser.

For the final 2M submission, run `python inference.py --mode ui` from the repository root or extracted inference release. It reads the private root `.env` and optional `config.toml`, serves the prebuilt interface and uses prerecorded recognition replies. Copy [the configuration template](.env.example) to `.env` in that root; keep credentials out of Git. `HA_ENABLED=true` and `SPOTIFY_ENABLED=true` enable the configured providers. Set `ACTIONS_ENABLED=true` to permit touch controls and accepted microphone commands; its submission default is false. CLI diagnostics and recorded WAV uploads never operate devices.

## Run

Use **`python start.py` in the parent smart-speaker directory**. It bootstraps local tools and exposes setup/publish/serve/benchmark modes. See the [root usage guide](../README.md#smart-speaker--speakerui). In an extracted Pi ZIP, the same command installs the runtime and serves the prebuilt website immediately; it never compiles Svelte on the Pi.

```sh
# From the repository root:
python start.py --skip-train --benchmarks skip
python start.py --mode serve
# Compatibility alias, inside SpeakerUI:
pnpm start
# Explicit frontend development, with mock backend:
pnpm dev
```

The deployed page, API and WebSocket share **http://127.0.0.1:3000**. Open **Inference debug** for host microphone selection, mute, WAV upload, real state, exact intent/slots and diagnostic reasons. There is no mock fallback in real mode. Models stay outside the public website. Do not copy ONNX files into dist or this frontend's models directory.

The lower-left sliders open the simulator. Choose among 15 shortcuts and intent fixtures, or the nine aura states. **Play music** and **Camera call** are marked **Live** when their providers are enabled. The opposite corner opens the last 30 intent/parameter receipts. The top-right controls and reminder count expose editing; the large central clock/aura is the wake target. Simulator intents remain simulated. Touch controls operate enabled Home Assistant lights and Spotify playback when `ACTIONS_ENABLED=true`.

```sh
pnpm dev                    # development frontend :5173 and gateway :3000
pnpm check                  # Svelte and server type checks
pnpm test                   # offline wrapper/transport tests
pnpm build                  # optimized frontend and backend without starting
pnpm kiosk                  # Chromium kiosk at the configured frontend URL
```

For a prebuilt release, run the exported `start.py`; `pnpm start` is an alias. The shared launcher verifies the optional home-directory model ZIP and starts the precompiled gateway/site. It owns shutdown of the Python worker.

## Display, interaction and retained reference

- Branding, component ownership, animation and touch rules: [docs/design/ambient.md](docs/design/ambient.md). Agent rules: [AGENTS.md](AGENTS.md).
- Retained standalone mockup: [mockups/ambient-v6/04-ambient.html](mockups/ambient-v6/04-ambient.html). Thermostat entry and branding sheet sit beside it. This is a comparison reference, not an iframe or dependency of the production app.
- Verification, screenshots and resource measurements: [docs/testing/ambient/README.md](docs/testing/ambient/README.md).

Mouse, touch and pen drag directly move the alarm wheels. Reminder swipes reveal the completion action progressively; full travel or a deliberate short flick completes only that reminder. Reordering belongs to the grip. Light rows swipe for brightness; compact control tiles and expanded lists share state.

Control center closes after 60 idle seconds, with a final five-second rounded perimeter stroke. Device/media/weather views use five idle seconds plus five seconds of stroke. Focused forms and alarm editing remain open. Ringing alerts flash all four rounded edges in a 3.6-second long/short/short rhythm and dismiss after five minutes without completing reminders. Snooze clears older queued notifications for the same reminder. Reduced-motion settings use steady indicators.

When HA is enabled, the display lists only the configured lights and uses allowlisted temperature sensors for the current temperature. Add the desired `sensor.*` IDs to `HA_ENTITY_IDS`; sensor selection is read-only. Without an available temperature sensor the display shows a dash. Climate target control is not connected and remains unavailable in live HA mode. Disabled HA retains the light and thermostat demos. Add a `camera.*` ID to the same allowlist for a live camera view. Disabled HA retains simulated camera calls; weather remains a local fixture.

## Raspberry Pi deployment

Target the original 7-inch display at 800 × 480; 1024 × 600 is also checked. Use 64-bit Raspberry Pi OS and a current Chromium with working hardware graphics. No font, icon, analytics or animation CDN is required. Runtime assets are local.

The app tries fullscreen on launch and retries once on the first touch if browser activation is required. `?fullscreen=0` disables this for embedded previews/testing. Escape is respected. `pnpm kiosk` starts a dedicated Chromium kiosk window; set `CHROMIUM_PATH` if the executable is named `chromium-browser` or installed elsewhere. A browser cannot promise automatic fullscreen without the relevant permission/activation, so kiosk mode is the reliable unattended route.

Build off-device if useful and copy `dist/`, `dist-server/`, `scripts/`, package/lock/workspace files. Install Linux production dependencies with `pnpm install --prod --frozen-lockfile`; do not copy macOS `node_modules`. Serve the prebuilt release with the commands above. The existing [deploy/speaker-ui.service](deploy/speaker-ui.service) remains an alternative single-process Fastify static/API deployment; its comments describe the required origin/port settings. No system service or kiosk autostart has been installed by this change.

Animations are retained. The build bakes fixed CSS keyframes, the static icon subset and camera background, and emits Brotli/gzip assets. The live LiveKit-derived shader keeps its 18 samples, maximum 320 × 320 backing canvas, 24 fps speaking / 20 fps other active states. Settled idle stops shader draws while a CSS breath continues. Hidden rendering pauses, and all native listeners/renderers have explicit teardown. **Desktop software-GPU results are not a Pi benchmark.**

## Configuration and integrations

Copy [.env.example](.env.example) to a private `.env` in the repository root for `inference.py`/`start.py`, or beside the gateway for direct `pnpm dev` development. Never put credentials in `VITE_*` variables. Filling credentials alone does not enable a provider. Defaults keep integrations and real actions disabled.

| Setting | Purpose |
| --- | --- |
| `FRONTEND_HOST`, `FRONTEND_PORT` | Display server, default `127.0.0.1:3000` |
| `HOST`, `BACKEND_PORT` | Backend binding, default `127.0.0.1:3001` for start modes |
| `DISPLAY_ORIGIN` | Exact browser origin in production; otherwise derived from frontend host/port |
| `BACKEND_URL` | Optional proxy/readiness target when backend URL differs; set explicitly for a TLS backend |
| `APP_ORIGIN`, `PORT` | Existing development/direct gateway configuration; start modes supply their own production values |
| `HA_URL` | Nabu Casa remote HTTPS base, e.g. `https://<instance>.ui.nabu.casa`, without `/api` |
| `HA_TOKEN` | Home Assistant long-lived token, not a Nabu Casa password |
| `HA_ENTITY_IDS` | Comma-separated allowlist of `light.*`, `camera.*`, and `sensor.*`; temperature sensors must report a temperature device class and unit |
| `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` | Server-side Spotify application credentials |
| `SPOTIFY_REDIRECT_URI` | Exact registered callback, typically `http://127.0.0.1:3000/auth/spotify/callback` for production loopback |
| `SPOTIFY_DEVICE_ID` | Optional Spotify Connect playback device |
| `INFERENCE_MODE` | `mock` for explicit development; the integrated launcher sets `onnx` and supplies the private worker paths |
| `ONNX_MODEL_DIR` | Reserved private directory for ONNX model files; not read by the current mock |
| `LOCAL_MEDIA_DIR`, `DATA_DIR` | Local music catalog and private token data |

With `HA_ENABLED=true`, Ambient shows the allowlisted lights' names, on/off state, brightness and color. Touch controls send on/off, dimming, color and white-temperature requests through the gateway. With `SPOTIFY_ENABLED=true`, it shows the current track/episode, artist/show, album when provided, progress, play/pause state, volume, shuffle and repeat. Album/episode artwork appears in the compact player and expanded music view, using the image URL from the existing playback response. Missing or failed images retain the original placeholders. The music view offers **Connect Spotify** when authorization is missing, and controls for resume/pause, previous/next, seek, volume, shuffle and repeat. The square button pauses live Spotify playback. `SPOTIFY_DEVICE_ID` selects the control target when set; otherwise Spotify uses the active device.

`ACTIONS_ENABLED=false` keeps live state visible but disables device/playback controls. Disabled integrations retain their local demos. Enabled integrations never fall back to fabricated state: missing entities, unavailable devices, expired authorization and provider errors are shown explicitly. The explicitly tapped **Play music** shortcut sends a resume request to Spotify without search text, a track URI or a replacement playlist. It continues the active/configured Connect session. If there is no playable session, open Spotify on that device first. Missing authorization opens the connection prompt. The **Camera call** shortcut opens the only allowlisted camera, or the camera selector when more than one is configured. These touch shortcuts bypass dummy inference; simulator/ONNX fixture events still cannot operate live providers. Backend credentials and provider tokens stay on the server. The SPA does not decode Spotify audio or RTSP.

For an ONVIF camera, configure it in Home Assistant first, confirm its feed works there, then append its entity ID to `HA_ENTITY_IDS`, preserving your light IDs. For example:

```dotenv
HA_ENTITY_IDS=light.hue_color_lamp_1,light.dining_light,camera.dining_area_main_camera_tc70_mainstream
```

These are also the final submission launcher's default entity IDs when none are configured. The allowlist makes these entities available to touch controls and the camera view; it does not infer which physical lamp corresponds to an acoustic label such as `light.kitchen`. Map those labels explicitly in root `config.toml` under `[bindings.lights]` when using spoken light commands. Generic **play music** uses Spotify resume with no song/artist search or playlist replacement. Open Spotify on the intended Connect device and authorize it in **Connect Spotify** first. The registered callback must match the launch port, normally `http://127.0.0.1:3000/auth/spotify/callback`.

For example, this private root `.env` enables the existing touch light/camera controls and generic music playback after you fill in your own credentials:

```dotenv
HA_ENABLED=true
HA_URL=https://your-instance.ui.nabu.casa
HA_TOKEN=your-long-lived-home-assistant-token
HA_ENTITY_IDS=light.hue_color_lamp_1,light.dining_light,camera.dining_area_main_camera_tc70_mainstream
SPOTIFY_ENABLED=true
SPOTIFY_CLIENT_ID=your-spotify-app-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-app-client-secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:3000/auth/spotify/callback
ACTIONS_ENABLED=true
```

Run `python inference.py --mode ui`; no source edit is needed. Use `ACTIONS_ENABLED=false` to inspect provider state without sending light or playback controls. The model's light labels (`light.all`, `light.kitchen`, `light.bathroom`, `light.bedroom`) are separate from this entity allowlist. There is no default assignment from these labels to either lamp; missing mappings return `not_configured`. An explicit `[bindings.lights]` entry maps one model label to one allowlisted HA light or group entity. Configure any such group in Home Assistant first.

The gateway proxies HA's `/api/camera_proxy_stream/<entity_id>` as a same-origin MJPEG stream. The HA token stays on the server. Home Assistant performs the ONVIF/RTSP conversion; this app needs no camera decoder package. The camera is the centered call view. Minimizing or leaving that view keeps the ongoing call in a floating preview; restoring it reuses the same stream. This is **video only**, with camera off/on, retry and end controls; no microphone, camera audio or two-way talk is connected. Stream reads work with `ACTIONS_ENABLED=false`. Camera-off, hidden pages, hangup, unmount and disconnection release the stream. Setup has a 15-second deadline; open streams are not cut off by the normal five-second API timeout. The gateway permits at most two concurrent streams. See the [camera checks](docs/testing/camera/README.md).

HA state reaches the backend over its existing WebSocket, with the display refreshing every 1.5 seconds. Spotify normally refreshes every five seconds. After a touch command, it refreshes immediately and every half-second for three seconds; the backend briefly reduces its cache lifetime and prevents older reads from replacing newer results. Successful playback commands may return empty bodies; playback state reads still require valid JSON. The existing rate-limit cooldown remains enforced. Scrubbing immediately shows the requested position with **Seeking…** until observed progress confirms it or a four-second settling window expires. Display polling pauses while hidden and stops on unmount. Other playback state remains based on observed values. Restart `pnpm start` after changing `.env` or backend code.

## Audio diagnostics and real inference handoff

Open **http://127.0.0.1:3000/?diagnostics** (development: `:5173/?diagnostics`) for the preserved microphone/PCM workflow, local audio search and optional provider setup. Each nonempty recording yields a fixed mock media search even for silence; cancellation, disconnect and page hiding release capture. Put your own `.mp3` files in `media/` for local playback. None are bundled.

The loopback setup is for a single trusted user. For remote browsers use a trusted HTTPS reverse proxy and matching origin; non-loopback HTTP cannot capture microphone audio. Keep provider tokens and private data server-side. Existing host/origin checks still apply through the frontend proxy.

[docs/architecture.md](docs/architecture.md) documents the preserved audio gateway. [docs/onnx-sanity.md](docs/onnx-sanity.md) describes the real adapter gate. The older inference-sanity script covers the historical mock boundary. Use `scripts/validate-runtime.mjs` for real WAV/CLI/gateway parity and `scripts/browser-runtime-check.mjs` for compiled-page checks. Typed browser fixtures are not acoustic evidence.

Source notices are retained in [mockups/ambient-v6/licenses](mockups/ambient-v6/licenses) and copied to the build's `licenses/` folder. The aura shader carries its own upstream notice; do not assume the shader is covered by the React wrapper's Apache license.
