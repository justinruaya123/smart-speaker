# Retained approved reference

This standalone prototype is retained for side-by-side comparison with the production Svelte app in `src/ambient/`. Open `04-ambient.html` directly, or serve this directory over HTTP. It is not the production runtime. Keep intentional differences documented in `docs/design/ambient.md`; do not silently replace the reference when altering product design.

# Ambient · refined landscape prototype

Open `04-ambient.html` directly in a browser. `05-thermostat.html` starts on the thermostat. Both are self-contained offline files. `00-brand-and-behavior.html` contains the working identity and component suggestions. This revision updates the same files; reload an already-open tab to see it.

## Try the interaction

- Tap anywhere in the large invisible circular wake target to listen; tap again to stop. Clock and display labels are not selectable. Text fields and the debug JSON remain selectable.
- Tap the top-right grid icon, or swipe down from that corner, to open the compact control center. Separate dark glass cards sit over one black-tinted, 10 px backdrop blur. There is no outer panel or Controls title; the backdrop also blurs any active toast.
- The control center uses two fixed columns. The second column from the right always contains the reminder accordion, with two icon-only squares below it: open the complete list and add a reminder. Each entry owns its pin and drag handle. Only the handle can reorder, including via up/down keys.
- The top-right count opens or closes home reminder editing directly. Home has no accordion heading. Editing closes after 30 seconds of inactivity; only pinned reminders remain, as right-aligned text with a small number on the right. Pins and drag handles disappear. Quiet pinned entries retain the completion swipe.
- Start music to reveal the top-left track card, progress bar, thin, low-opacity five-bar activity loop and a more visible traveling pulse confined to the bottom 20 px. The activity animation is pre-baked CSS, not an audio spectrum; it follows playback state, pauses when playback or sound is muted, and becomes static for reduced motion. A future Spotify integration should drive the playback state; this mockup does not stream Spotify audio. Use its play/pause button directly, or tap the track to expand playback controls, including shuffle, repeat off/context/track, seek, volume and stop. Pausing keeps the current track available; stopping removes the summary.
- The home-only top-right microphone toggles microphone mute and its red perimeter glow. It hides in task views and control center. Sound mute is independent: tap the speaker beside a volume slider for a red crossed speaker and gray fill; the volume setting is retained. Tapping the wake area while muted never unmutes it. Listening produces a brighter white perimeter effect with a gentle 2.6-second flash. The edge shadows are static, with a 24 px corner radius; only opacity animates.
- Light icons toggle their bulbs immediately. A single text tap opens editing; a double tap toggles. Swipe a light row horizontally to change brightness. The gesture records one service request on release; vertical scrolling and cancellation do not commit a brightness change. The slider remains available for keyboard or precise control.
- RGB-capable bulbs offer red, green, blue and orange plus supported white presets. The kitchen fixture is temperature-only and offers warm white, white and cool white. Light brightness changes the icon glow, entry illumination and lower fill bar. Only lights currently on appear as compact control-center tiles, with a live glow/fill but no room or explicit percentage. Swipe a tile to dim, tap its bulb to turn it off, or tap its name to edit. Compact square controls are exclusive to control center; the expanded Lights view retains horizontal rows with room and brightness metadata.
- Automation intents with complete timer/alarm/temperature/light slots apply immediately and toast at the lower right. Tap the toast to edit. It fades after 4.2 seconds. The lower-left receipt shows the intent and named parameters; tapping it opens the last 30 requests.
- The lower-left simulator keeps all nine LiveKit states and the automation fixtures. It remains available during a minimized call.

Reminder completion is one continuous right swipe. “Mark as done ›” begins at the original title position and is revealed into the space left by the moving text. Releasing after 80% of the row width completes only that reminder. A deliberate flick also completes it after at least 44 px of travel and a recent rightward speed of 0.65 px/ms. Slow partial, paused, reversed and cancelled gestures slide back without changing it. The control-center swipe starts only on its trigger or the dedicated corner strip, so reminder dragging cannot open it. Keyboard users can focus the completion action and press Enter. No pending two-swipe state is used.

## Time and attention

The thermostat's center and filled arc represent the selected temperature sensor, initially 26°C. Its draggable pointer represents the target. Rotate with touch or mouse; tap left/right for −1°C/+1°C. The arc is red above target, blue below, and neutral within 0.5°C. Guide marks occur every degree, with larger five-degree marks.

The control center closes after 60 seconds of inactivity: 55 quiet seconds followed by a five-second screen-edge stroke. Thermostat, lights, music and weather keep the shorter five-second wait plus five-second stroke. Two paths start at bottom center and travel in opposite directions, meeting at the top. Interaction resets the countdown. Calls, typed forms, the alarm wheel editor and request history do not auto-close.

An empty reminder intent retains the five-second default-action stroke. Touch or focus cancels automatic submission. An untouched reminder saves as “Reminder.” No ASR is present. Optional deadlines create alerts within a minute and when due. Timer, alarm and reminder alerts share a strong white edge glow repeating a 3.6-second rhythm: one long flash, a short pause, two short flashes, then a long pause. Each alert has a dismissal action and five-minute snooze. It dismisses after five minutes without completing its reminder; every queued alert receives its own five-minute window. Snoozing clears stale notifications for that reminder. All four edges flash together with rounded, softly feathered corners. Reduced-motion mode uses a static indicator.

Control-center timers update in place. Alarm tiles show relative time such as “In 3h 5m” with hands representing the set time. Tapping an alarm opens an inline picker in the right control column. Hours 01–12, minutes 00–59 and AM/PM use native scroll snapping, a cylindrical visual transform and fixed ends; none of the wheels wrap. Hold and drag directly with touch, pen or the left mouse button; the dial follows the pointer and snaps to the closest value on release. Cancelled drags restore the prior value. Trackpad/mouse wheel, adjacent-value clicks and keyboard arrows/Home/End also work. The relative deadline and analog hands follow the draft. Confirm saves and toasts; cancelling preserves the previous alarm. A quiet 20 ms synthesized tick follows user movement and obeys speaker mute. Audio is created only after interaction and released when the picker closes.

Below the date, the home display shows the selected temperature beside a small thermometer and a static “Partly cloudy” demo condition. The thermostat’s Display sensor menu offers four simulated Home Assistant temperature entities. Celsius, Fahrenheit and Kelvin states are normalized to Celsius; unavailable or nonnumeric states display a dash and hide the current-temperature arc. Changing the display sensor leaves the target unchanged, records a read-only GET state request, and updates the home reading immediately. Focus on the sensor menu suspends the inactivity close. These are local fixtures, not live sensor discovery or weather.

The home timer uses an hourglass that turns gently and pauses with the timer. Active calls minimize to an avatar, elapsed duration and activity bars. The camera preview is a local test pattern, not an RTSP connection.

## Synthetic contracts

The mockup accepts structured intent events, not spoken text:

```js
document.getElementById('ambient-assistant').dispatchEvent(new CustomEvent('vcm-intent', {
  detail: {
    intent: 'light.brightness',
    parameters: { entity_id: 'light.living_room', brightness_pct: 40 }
  }
}));
```

The 380 ms inference delay and confidence of 1 are fixture values, explicitly marked `simulated`. Touch interactions are recorded separately from inference fixtures. There is no ONNX model, ASR, actual audio playback, device control, remote call or external network request. State and request history are in memory and reset on reload.

`source/device-contracts.js` contains pure request descriptions. Light actions target `POST /api/services/light/turn_on`, `turn_off` or `toggle`; brightness uses `brightness_pct`, colors use `rgb_color`, whites use `color_temp_kelvin`. Temperature uses `POST /api/services/climate/set_temperature`. These describe service calls, not writes to `/api/states`, which would not control a real device. The display sensor picker describes `GET /api/states/<entity_id>` using temperature sensor fixtures with `device_class: temperature` and `unit_of_measurement`; it never sends a climate write when selecting a sensor. A future gateway should obtain real entity capabilities and states, reconcile returned device state and own authentication.

Spotify shuffle and repeat produce `PUT /v1/me/player/shuffle?state=...` and `PUT /v1/me/player/repeat?state=off|context|track` descriptions. Seek uses `position_ms`. These are mock payloads, not an authenticated Spotify integration. Real playback control requires the appropriate OAuth scope, an active supported playback device and applicable Spotify account access.

Primary references:

- [Home Assistant REST API](https://developers.home-assistant.io/docs/api/rest/)
- [Home Assistant sensor entities](https://developers.home-assistant.io/docs/core/entity/sensor/)
- [Home Assistant light turn-on parameters](https://www.home-assistant.io/actions/light.turn_on/)
- [Spotify shuffle](https://developer.spotify.com/documentation/web-api/reference/toggle-shuffle-for-users-playback)
- [Spotify repeat modes](https://developer.spotify.com/documentation/web-api/reference/set-repeat-mode-on-users-playback)

## Vite and Svelte

`vite-export/` is the production static export with relative asset paths. Serve it on the Pi using a static web server; no Node process is needed at runtime. To edit and rebuild this independent prototype with Node 22.12+:

```sh
npm install
npm run dev
npm run build
```

The complete prototype uses native DOM updates. `source/LiveKitAura.svelte` is the reusable Svelte 5 aura port, with attachment-owned setup and cleanup. The full app has not been converted into Svelte yet. The port was separately compiled and checked for prop changes, unmounting and remounting.

The renderer adapts the actual LiveKit shader and state targets. This revision reduces blur, smoothing and bloom to expose a sharper contour. It retains 18 shader samples, at most a 320 × 320 backing canvas, a 24 fps speaking cap and 20 fps for other active states. Settled idle shader rendering stops; a slightly faster 5.5-second compositor-only scale, rotation and opacity breath keeps idle subtly alive. It pauses behind tasks and becomes static under reduced motion. Hidden shader rendering also stops. These are implementation bounds, not measured Raspberry Pi 4 performance.

`revision-6-checks.json` records this revision's browser validation. Earlier reports refer to the previous layouts. Tests used macOS Chromium with software WebGL; audio API calls were observed, but physical audio output was not measured; Raspberry Pi hardware performance remains unmeasured.

## Source notices

The original LiveKit sources are retained beside the adapted renderer. `licenses/AURA-NOTICE.txt` records the shader's separate upstream license notice, including its broken upstream license link. Its terms need verification before product distribution; the repository's Apache license is not assigned to the separately licensed shader. Lucide notices are included. Standalone files also include notices as comments.

All files remain outside the project. No project source or Git history was changed.
