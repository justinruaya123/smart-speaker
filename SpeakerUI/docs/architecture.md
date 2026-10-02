# Minimal voice gateway

The browser owns microphone capture, a diagnostic Svelte page and local MP3 playback. One Fastify process serves the Vite build, APIs and a WebSocket. The external inference service owns audio preprocessing, model loading and decoding. Feature modules own inference, media and integrations; there is no broker, database, SSR process or local ONNX dependency.

## Voice protocol

`/ws` requires the configured Origin. Text messages are JSON; binary messages are little-endian PCM16, mono, 16 kHz, normally 640 bytes/20 ms. The final frame may be shorter but must contain whole samples. No WAV headers, compressed audio, base64 or raw model tensors travel on this socket.

Client controls: `{ "type": "start" | "end" | "cancel", "utteranceId": "fresh-id" }`. On connect the server emits `ready` with a fresh session ID and `mode: "mock"`. It acknowledges start with `state: "recording"`, then emits `processing` and finally `idle`. The client waits for recording acknowledgement before forwarding samples. Reconnecting starts a new session; it never resends old audio/actions.

Inference events wrap `{ utteranceId, revision, final, intent, parameters, confidence }` and explicitly report `simulated`. Valid, increasing revisions for the active utterance are accepted. Only one final result can dispatch. Missing/ambiguous parameters produce `needs_input`, invalid scores/labels/targets produce `rejected`. A cancelled result cannot act. Partial results never act. The sole mock returns `media.search` / `{query:"demo"}` with a fixed score and always emits `action.status: "simulated"`, regardless of ACTIONS_ENABLED.

Real adapter interface: `start(id)`, `writeAudio(bytes)`, `end()`, `cancel()`, plus an emit callback. Operations are serialized per utterance. A real adapter must document score normalization, bound its own buffers, abort I/O on cancellation, and emit errors/results only for its own utterance. `INFERENCE_URL` and `INFERENCE_TOKEN` are reserved; they cannot activate an invented wire protocol. Unknown INFERENCE_MODE values fail startup.

Supported commands:

| Intent | Parameters |
| --- | --- |
| `light.on`, `light.off` | `entityId`: one allowlisted `light.*` ID |
| `camera.snapshot` | `entityId`: one allowlisted `camera.*` ID |
| `media.search` | `query`: nonempty, at most 200 characters |
| `media.play` | `source`: `local` or `spotify`; `id`: an owned search-result ID |
| `media.pause` | `source`: `local` or `spotify` |

The inference service must produce canonical entity IDs. Friendly-name resolution/model slot extraction are not guessed here. Real actions additionally require a configured model threshold and ACTIONS_ENABLED. A timed-out action is reported `unknown`; it is never automatically retried. Cancellation cannot undo a provider action already submitted.

## Resource limits

One recording/processing lease across at most four browser sockets. Each socket retains at most 1,024 used utterance IDs, then reconnects. Utterances are limited to 15 seconds/480,000 bytes; inference completion gets 10 seconds. Both the worklet's unacknowledged frame count and transport queues are limited to about one second of audio. Overflow cancels the utterance, without dropping samples and continuing. Local playback pauses during capture. Microphone tracks, audio contexts and sockets are released on cancellation/exit. No microphone recording is persisted.

The browser event log holds 50 entries of at most 2,000 characters. Gateway request logging is disabled to avoid retaining audio or credentials; startup/configuration errors still reach stderr. Provider requests have five-second deadlines. Local indexing uses filenames only, skips symlinks, caps at 10,000 tracks/eight directory levels, and reports index limits. Restart to refresh the index. Audio files stream with range support.

## Prepared integrations

Disabled integrations make no connections, including startup and health checks. Home Assistant retains at most 100 selected light/camera states, gets initial REST snapshots, subscribes to `state_changed`, and discards events for unselected entities. Reconnect refreshes snapshots, disconnected entries become stale, and authentication failures stop retries. Light actions are explicit on/off calls. Camera snapshots stream on demand; no video polling/transcoding.

Spotify uses server-side authorization-code OAuth, a one-use browser-bound state cookie, refresh tokens and a private atomic token file. Search keeps at most 100 recently returned track identities for explicit selection, returns at most five results, and suppresses calls during Retry-After cooldown. Device playback requires an existing Connect device; the wrapper is not itself a Spotify player. On failure it offers filename matches without auto-playing them. Local filenames are searched when Spotify is disabled/unavailable/unmatched. No Spotify audio is downloaded or cached.

Public endpoints: `GET /api/health`, `GET /api/media/search?q=...`, `GET|HEAD /api/media/local/:id`, `POST /api/media/play`, `POST /api/media/pause` (Spotify), `GET /api/home-assistant/entities`, `GET /api/home-assistant/camera/:id`, `POST /auth/spotify`, `GET /auth/spotify/callback`, and `/ws`. Changing HTTP operations and WebSocket upgrades require the configured Origin. Spotify OAuth initiation returns its authorization URL as JSON and sets an HttpOnly state cookie.

## Research decisions

- [Svelte with Vite](https://svelte.dev/docs/svelte/getting-started): compiled assets, no SSR needed for this harness.
- [shadcn-svelte](https://github.com/huntabyte/shadcn-svelte/blob/main/docs/content/installation/vite.md): source components added individually; Tailwind/configuration are initialized without importing unused components.
- [LiveKit Agents UI](https://livekit.com/products/agents-ui): React-based reference only. Revisit LiveKit JavaScript transport if remote conversational sessions justify it; do not embed React components into Svelte.
- [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet): separate audio thread and secure-context requirement. The audio graph resamples microphone input into the requested 16 kHz context; unsupported context rates fail clearly.
- [Home Assistant WebSocket](https://developers.home-assistant.io/docs/api/websocket/) and [REST](https://developers.home-assistant.io/docs/api/rest/): selected-state cache, actions and image proxy.
- [Spotify search](https://developer.spotify.com/documentation/web-api/reference/search), [OAuth](https://developer.spotify.com/documentation/web-api/tutorials/code-flow), [redirects](https://developer.spotify.com/documentation/web-api/concepts/redirect_uri), [quotas](https://developer.spotify.com/blog/2026-07-23-web-api-quota-updates): five results, exact callback URI, Premium/allowlist prerequisites, explicit rate/quota failures.

Documentation was researched on 2026-09-29. Live Home Assistant, Spotify and ONNX testing is intentionally deferred. Offline fixtures exercise wrapper policies, not compatibility with deployed services.

## Ambient provider state and controls

`GET /api/home-assistant/entities` now includes integration/action enablement and a selected set of display attributes. `GET /api/spotify/state` returns authorization/playback status with a five-second shared cache. No credentials or arbitrary provider attributes cross these boundaries.

`POST /api/home-assistant/control` accepts validated on/off, brightness, RGB and Kelvin commands for allowlisted lights. `POST /api/spotify/control` accepts validated resume/pause, next/previous, seek, volume, shuffle and repeat commands. Both require the configured origin and `ACTIONS_ENABLED=true`; disabled providers remain offline. They return observed state after the request. Action acceptance is distinct from observed device state. These routes are used by explicit touch controls, never by simulated inference. Temperature sensors are read-only; climate targets and RTSP are not wired to live services.
