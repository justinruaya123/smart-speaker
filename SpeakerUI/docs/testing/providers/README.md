# Enabled provider verification — 2026-10-01

The Ambient screen now uses enabled Home Assistant and Spotify state. Touch controls operate allowlisted HA lights and Spotify playback when actions are enabled. Disabled providers retain their demos. Simulated inference never operates live providers.

## Evidence

- `pnpm check`: no errors or warnings.
- `pnpm test`: 58 offline tests passed, covering state normalization, album metadata, absent/stale state, authorization/action gates, allowlists, polling lifetime, stale response rejection, post-command cache invalidation, empty successful Spotify command responses, camera streaming and existing diagnostics.
- `pnpm build`: frontend and backend completed successfully.
- `scripts/qa/providers.cjs`: 28 checks passed against the compiled UI, using controlled provider responses. Screenshots cover 800 × 480 and 1024 × 600. Includes toggling, brightness, RGB/white presets, failure recovery, playback controls, mute/unmute, album updates, seek stability during a delayed response, prompt post-command refresh, entity removal, authorization prompts, transport failure, demo fallback, the simulator boundary and the direct Play music shortcut. Results: `checks.json`.
- Existing `scripts/qa/interactions.cjs`: 42 checks passed with integrations disabled, including mouse/touch gestures, cancellation, reminder ordering, alarm wheels, timing and reduced motion. Latest results: `../playback-camera-regression/revision-6-checks.json`.
- Read-only check against the configured Home Assistant: connected; `light.hue_color_lamp_1` / ceiling light is off and current, with `color_temp` and `xy` capabilities. No physical light/playback actions were performed.
- These checks do not operate real Spotify playback or physical devices. Controlled responses validate the application's control paths; real provider latency remains unmeasured.

The delayed Spotify fixture returns a stale seek snapshot after 600 ms and updates its observed position at 900 ms. The requested position remains visible throughout, and the UI confirms the new state within 2.5 seconds. A next-track update delayed by 250 ms appears within two seconds. These bounds verify removal of the prior five-second polling delay, not a promise about Spotify's response time.

## Resource sample

`resources.json` records a desktop Chromium/SwiftShader sample of the updated screen before the final mute/history/lifetime refinements. Cold load was 171–183 ms, first contentful paint 180–196 ms, and idle JavaScript heap about 1.62 MB. The 30-cycle reopen sample grew by about 0.29 MB after collection. No browser errors occurred. Software-rendered aura states dominate CPU use. These are desktop results, not a Raspberry Pi benchmark or a final-artifact performance claim.

## Reproduce

Start an isolated app instance with `HA_ENABLED=false`, `SPOTIFY_ENABLED=false`, `ACTIONS_ENABLED=false` and matching frontend/backend ports. Do not point the interaction suite at real enabled devices. Set `PLAYWRIGHT_MODULE` and `CHROMIUM_EXECUTABLE` to an existing installation; no browser dependency was added to the application.

Run `scripts/qa/providers.cjs` with `TEST_BASE_URL` pointing at that app. The provider suite intercepts provider endpoints and uses controlled responses. Run `scripts/qa/interactions.cjs` against disabled integrations, with a separate `TEST_OUTPUT_DIR` to preserve previous evidence.

The adapter payloads follow [Spotify playback state](https://developer.spotify.com/documentation/web-api/reference/get-information-about-the-users-current-playback), [Spotify playback controls](https://developer.spotify.com/documentation/web-api/reference/start-a-users-playback), [Home Assistant REST](https://developers.home-assistant.io/docs/api/rest/), and [Home Assistant light services](https://www.home-assistant.io/integrations/light/).

## Remaining scope

Temperature sensor selection reads only explicit `sensor.*` entries in `HA_ENTITY_IDS`; climate target control is unavailable in live HA mode. Weather, audio calls and inference remain simulated. Live HA camera views and direct Spotify resume are covered in [the camera/shortcut verification](../camera/README.md). Spotify plays through its active/configured Connect device; the browser does not decode Spotify audio.
