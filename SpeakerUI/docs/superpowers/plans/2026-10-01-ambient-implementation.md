# Ambient implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. User explicitly requested implementation; continue in this chat without another approval gate.

**Goal:** Port the approved Ambient screen into Svelte, preserve the visual reference, add production start modes and dummy inference, and measure the built page.

**Architecture:** Component-owned Svelte state with native gesture and live WebGL lifetimes. Existing diagnostic/integration code stays intact. A small startup coordinator builds the selected side and owns the frontend/backend processes.

**Tech Stack:** Svelte 5, Vite, TypeScript/JavaScript, existing Fastify, Node 24, pnpm.

**Spec:** `docs/design/ambient.md`.

## Global constraints

800 × 480 landscape first. Preserve all approved motion. No new UI/animation libraries. Dummy inference only; credentials server-only. No commits or destructive cleanup of existing dirty work.

## Review focus

- Flick versus slow/reversed/cancelled swipes; no accidental list-wide completion.
- Drag ownership between alarm wheels, reminder grips and corner controls.
- Five-minute alerts, queued alerts, hidden-page resume and non-completion of reminders on timeout.
- Fullscreen denial, backend unavailable and process shutdown.
- Sensor unavailable states, focused form lifetime, and secret-free frontend builds.

### 1. Ratify and retain

- [x] Update reference alert timing/all-edge treatment and verify swipe behavior.
- [x] Preserve standalone reference, source notices and design documentation in the repository.

### 2. Svelte UI

- [x] Keep diagnostics accessible and replace the main entry with Ambient.
- [x] Compile real Svelte components using the reference CSS and gesture rules, with explicit teardown.
- [x] Bake fixed assets/keyframes, preserve live aura, fullscreen attempt and kiosk launcher.
- [x] Check types and compare all principal screens and gestures with the retained mockup.

### 3. Runtime/startup

- [x] Load a dummy inference runtime at backend startup with validated fixture requests.
- [x] Implement `pnpm start`, `pnpm start frontend`, `pnpm start backend` with optimized builds and clean shutdown.
- [x] Extend configuration examples for Nabu Casa, Spotify and kiosk; preserve existing secret values.
- [x] Run offline backend/startup tests, including disabled integrations and malformed input.

### 4. Verification and handoff

- [x] Run existing tests, type checks and production build.
- [x] Test 800 × 480 and 1024 × 600, screenshots and interactive parity.
- [x] Measure load, transfer size, JS heap, browser CPU and frame pacing, including animated states.
- [x] Document results, preview the compiled page, and report remaining hardware/integration limits.

## Execution evidence and rulings

- Preserved the already dirty/untracked scaffold in place as requested; no staging, commits, branch replacement or service installation.
- Retained native reference and current generated standalone output. Svelte uses real templates/reactive state; native code owns only gestures, focus, WebGL and camera lifetimes.
- Mouse/touch alarm-wheel failure reproduced before the pointer-handler change; held drag now follows movement and snaps without an extra selection in both outputs.
- Independent final review identified first brightness-toast DOM access before Svelte flush, and stale queued due-reminder alerts after snooze. Both reproduced with browser checks, fixed, and verified. The first fix removes obsolete imperative styling; the second filters stale notifications by reminder identity before advancing the queue.
- Ruling: fixed icon HTML is a closed, trusted Lucide dictionary, not arbitrary HTML. Retain that tiny insertion instead of adding an icon library; all user text remains escaped by Svelte.
- Ruling: the existing diagnostic/backend safety contracts and integration restrictions remain intact. Ambient uses explicit fixtures; live sensor/Spotify/RTSP integration is a later connection task, not fabricated validation.
- Ruling: browser startup may deny fullscreen. Attempt once at launch and once on the first user gesture; provide a separate kiosk launcher and respect Escape.
- Ruling: browser resource evidence is macOS M2 / Chromium software WebGL. Do not treat its CPU/frame numbers as Raspberry Pi hardware validation.
- Final review set aside physical Pi/touch/audio evidence, process runtime checks, live providers/model execution and private secret-value inspection. Process startup/shutdown and real local API/WS paths were subsequently verified. Hardware/model/provider limits are explicitly documented; no secret values were printed or copied.
- `pnpm check`: zero Svelte errors/warnings and passing server type check. `pnpm test`: 29/29 offline tests. Production builds pass. Browser reports retain 42 interaction checks per output, 16 alert checks per output, 23 parity/integration/fullscreen checks and 19 startup checks.
