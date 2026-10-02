# Ambient design contract — approved 2026-10-01

Reference: `mockups/ambient-v6/04-ambient.html`; companion thermostat and branding files live beside it. The retained mockup is a comparison artifact, not the runtime application.

## Identity and composition

Ink `#090d11`, pearl `#f1f5f5`, mist `#b7dde2`, muted blue-gray `#8d9ba2`. Red is mute/warning/above-target temperature; blue is below-target. System typography, thin aligned icons, tabular times, no selectable display text. Large invisible touch targets have no hover circles. No topbar/sidebar or large enclosing control panel. Glass tiles sit over a single dark backdrop blur.

Always-centered live aura and clock. Date, then selected sensor temperature and a small weather condition underneath. A large wake target yields to notifications and corners. Top left shows current music only: faint mini bars, stronger animated bottom-edge glow, track, artist, playback/progress. Top right shows reminder count, microphone mute and control trigger. Toast bottom right, subtle intent parameters/history at the opposite corner.

For enabled Spotify, real album/episode artwork replaces the mini bars and expanded record illustration when available. Preserve each image's aspect ratio without cropping, dimming or an overlay; keep the existing geometry and bottom-edge motion. Missing or failed images restore the original placeholders. This is an intentional departure from the fixture-only reference.

## Component ownership

Ambient owns screen state, deadlines and transient views. Aura owns WebGL setup/cleanup and LiveKit states. Home owns clock/environment, music presence, quiet pinned reminders and active time rail. ControlCenter owns two columns of glass controls; its second column from the right always contains the reminder accordion. Reminders use per-entry pins and grip-only reordering. Lights retain compact on-only tiles in controls and full list rows in the expanded menu. Thermostat owns target geometry and a display-sensor selector. AlarmWheels owns bounded direct drag/scroll/key manipulation and short audio ticks. Task views own media, timers, typed reminder forms and calls. Attention owns ringing overlay and all-edge flash. Simulator/history remain inspectable.

## Motion and interactions

- Single right swipe progressively exposes “Mark as done” from the original title position. Complete only that reminder after 80% travel or a deliberate right flick (at least 44 px, recent velocity at least 0.65 px/ms). Slow partial/reversed/cancelled gestures revert. The action must never overlap the moving title.
- Count toggles home reminder editing. After 30 idle seconds, only right-aligned pinned text and right-side number labels remain. Accordion/grips/pins belong only to expanded editing/control center.
- Control center: 60 idle seconds total, including the final five-second perimeter stroke. Other transient device/music/weather views: five idle plus five stroke. Stroke grows both ways from bottom center. Alarm editor and forms do not silently submit while focused.
- Ringing timer/alarm/reminder attention: strong white glow on all four edges, long flash / short pause / two short flashes / long pause. Rounded 24 px perimeter and softly blended corners; only opacity animates over a static inset glow. Faster 3.6-second loop; steady reduced-motion alternative. Each displayed alert automatically dismisses after five minutes without marking a reminder complete. Queued alerts get their own five minutes. Explicit snooze remains five minutes.
- Idle breath 5.5 seconds; brighter gentle white listening edge, explicit red mic-mute edge. Music motion is pre-baked CSS; aura remains a live shader, preserving the current sample/backing/frame limits. Do not remove motion to obtain better benchmark numbers.
- Alarm wheels follow held mouse/touch/pen movement, snap on release, remain bounded and support scroll/tap/keyboard. Relative time and analog hands update during selection; sound mute also mutes ticks.
- Light icon tap toggles, text double tap toggles, horizontal row/tile swipe dims. Color/glow/fill react; supported presets follow capabilities. Thermostat current fill and target pointer are independent; display-sensor selection never writes the climate target.

## Runtime and limitations

Svelte 5 + Vite production SPA; no SSR. Dummy ONNX backend loads on startup and emits structured fixture intents, not recognized speech. Existing diagnostic capture/integration code remains available. Secrets stay in server-only `.env`; Nabu Casa uses the HTTPS remote Home Assistant URL and long-lived token. Weather/calls/music/device fixtures remain simulated unless separately wired and enabled.

Attempt browser fullscreen at startup and retry once on a user gesture if required ([browser activation requirement](https://developer.mozilla.org/en-US/docs/Web/API/Element/requestFullscreen)); provide Chromium kiosk startup for the Pi. Target [800 × 480](https://www.raspberrypi.com/products/raspberry-pi-touch-display/) landscape, also test 1024 × 600. Report measured desktop load, heap and CPU honestly, including the live aura, with reproducible browser tooling. Preserve source notices, including the aura shader's separate license notice.

## Svelte port ratification and comparison notes

The native standalone reference and Svelte application use the same approved geometry, palette, icons, gestures and motion values. On 2026-10-01, 28 measured regions across ten principal 800 × 480 screens matched at zero pixel geometry difference, with matching font sizes; screenshots are retained under `docs/testing/ambient/`. This is layout evidence, not a claim of identical pixels throughout a live shader animation.

Intentional implementation differences: Svelte owns reactive DOM instead of rebuilding HTML strings. The production simulator crosses a validated local dummy-backend boundary, falling back to the same local fixtures when unavailable. Fullscreen/kiosk startup exists only in the application. The camera's static gradient/grid is baked to an SVG and drawn from a cached image while the original scan stripe/timestamp remain live. System audio remains simulated. Enabled HA cameras now supply live MJPEG video converted by Home Assistant from ONVIF/RTSP. Enabled HA now supplies allowlisted light and temperature-sensor states; enabled Spotify supplies playback state. Touch controls may operate HA lights and Spotify when actions are enabled. Climate target control remains unavailable in live HA mode. The independent diagnostic page retains its existing audio/provider functionality.

The fixed Lucide subset is the only trusted HTML insertion in Svelte (`Icon.svelte`); names resolve through a closed build-time dictionary. No user text, intent parameters, provider HTML or credentials enter that insertion. The Svelte analyzer's generic HTML warning was reviewed on that boundary. The aura's upstream source/license notices remain alongside the reference and are copied into each production build.

## Enabled provider behavior — 2026-10-01

Enabled HA and Spotify replace their corresponding fixtures with observed provider state. Disabled providers preserve demo behavior. Missing/disconnected data is labeled unavailable; it does not restore demo devices or tracks. Temperature sensors are selected from the explicit HA allowlist and never write a thermostat target. Live controls honor `ACTIONS_ENABLED`, supported light capabilities, provider authorization, and the entity allowlist. Simulator intents remain unable to operate providers. The explicitly tapped Play music and Camera call shortcuts use enabled providers directly and display a Live label. Camera views retain the preview/expanded layout, show video-only capability, and omit unsupported audio controls. The request receipt moves above the call-resume button while a call is active. Provider status and connection prompts use the existing light/music panels; long names are truncated to preserve the approved geometry.
