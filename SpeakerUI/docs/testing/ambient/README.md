# Ambient verification — 2026-10-01

All checks below use the production Vite build. No Home Assistant, Spotify or real inference service was contacted.

## Evidence

- `pnpm check`: 0 errors, 0 warnings; server TypeScript passes.
- `pnpm test`: 29 offline tests pass, including dummy runtime lifecycle, malformed fixture requests, origin rejection and zero outbound provider calls.
- [Svelte interaction report](revision-6-checks.json) and [prototype interaction report](prototype-interactions.json): 42 checks each at 800 × 480 and 1024 × 600, including real mouse events and Chromium touch events, partial/full/flick/reversed/cancelled swipes, grip reorder ownership, wheel hold/drag/cancel/bounds/scroll, focus lifetime, sensor conversions/unavailable state, timer and control expiry.
- [Alert report](alerts.json) and [prototype alert report](prototype-alerts.json): 16 checks each; rounded all-edge flash, five-minute expiry, fresh queued lifetimes, stale queue filtering on snooze, reminder non-completion, hidden-page resume and reduced motion.
- [Parity and integration](parity.json): 23 checks; all 15 fixtures through the actual local backend, diagnostic WebSocket/scrolling, denied fullscreen attempt/retry and 28 layout regions across ten 800 × 480 screens. Maximum geometry difference: **0 px**, all sampled font sizes match. Held wheel drag works in both implementations.
- [Startup modes](startup.json): 19 checks; frontend/backend/both builds, runtime readiness, proxy/origin policy, Brotli/gzip negotiation including `q=0`, WebSocket, absent-backend fallback, mode/port rejection, normal and interrupted-build shutdown.
- Captured screenshots were visually inspected for home, control center, alarm wheels, thermostat, lights, media, call and rounded alert edges. They are not a claim of identical shader pixels through every animation frame.

## Browser resources

[Raw measurements](resources.json). Host: **Apple M2 / macOS arm64**, Chromium **153.0.8010.12**, headless, **SwiftShader software WebGL**, 800 × 480. Three cold browser contexts with cache disabled; loopback serving; five-second state samples, no animation suppression or reduced-motion override during measurement. `requestAnimationFrame` sampling itself adds a small amount of work. Wall-time busy percentage includes main-thread waits and is distinct from OS CPU time.

- Median first contentful paint: **116 ms** (range 116–124 ms).
- Median DOM ready / load: **28 / 108 ms**.
- Cold transferred bytes: **60.7 kB**, including resource timing's response overhead, health and pre-baked camera asset. Encoded response bodies: **59.2 kB**; decoded: **222.0 kB**. JavaScript and CSS served as Brotli. No external fonts/CDNs.
- Thirty control/alarm open/close cycles after GC: 2.59 → 2.86 MB retained JS heap (**+0.27 MB**). This short repeatability check cannot rule out every long-running leak.

| State | Browser RSS sum | Used JS heap | Main-thread wall-time busy | Browser CPU / one core | rAF p95 interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| idle | 327 MiB | 1.55 MB | 1.0% | 23.2% | 16.8 ms |
| listening | 346 MiB | 2.60 MB | 26.1% | 184.4% | 16.7 ms |
| speaking | 345 MiB | 2.70 MB | 30.9% | 203.6% | 16.8 ms |
| music playback | 356 MiB | 2.59 MB | 9.7% | 75.0% | 16.8 ms |
| control center glass | 374 MiB | 2.61 MB | 0.9% | 48.5% | 16.8 ms |
| alarm flashing edges | 366 MiB | 2.57 MB | 1.6% | 26.8% | 16.7 ms |
| simulated camera call | 370 MiB | 2.70 MB | 3.4% | 8.6% | 16.7 ms |

CPU is the sum of process CPU-time deltas in the isolated test Chromium instance. **100% means one CPU core; values over 100% use multiple cores.** Software WebGL puts shader work in Chromium's GPU process; listening/speaking are by far the most expensive states. The rAF interval measures browser frame opportunity, **not aura draw frequency**, which remains capped at 20/24 fps. JS heap excludes native browser/GPU textures. RSS is an OS snapshot summed across this isolated Chromium instance's browser, renderer, GPU and network processes; shared pages can be counted more than once. It is not a guaranteed per-tab private-memory budget.

An idle process snapshot on this Mac showed approximately 79 MiB resident for the dummy backend, 63 MiB for the frontend server and 57 MiB for the startup coordinator. These are one-time OS RSS readings, not peak budgets or Pi measurements.

**No Pi 4 hardware benchmark was performed.** Use hardware graphics, then rerun these scripts on the target display. Real touch-controller latency, GPU thermals, screen tearing, audio tick sound, kiosk boot, model execution, camera decoding and provider latency remain device/integration validation. The software-GPU numbers are intentionally reported without claiming a Pi frame rate or lowering animation quality.

## Reproduce

Start `pnpm start` with provider integrations disabled. The QA scripts use an existing Playwright installation and Chromium; they add no browser-test dependency to the shipped app. Provide `PLAYWRIGHT_MODULE` as an absolute module path if Playwright is not installed locally; `CHROMIUM_EXECUTABLE` optionally chooses a browser binary.

```sh
node scripts/qa/interactions.cjs
node scripts/qa/alerts.cjs
node scripts/qa/startup.mjs
# Serve mockups/ambient-v6 over HTTP, then set PROTOTYPE_URL for this comparison:
PROTOTYPE_URL=http://127.0.0.1:8768/04-ambient.html node scripts/qa/parity.cjs
# Run without other competing browser checks:
node scripts/qa/resources.cjs
```

`TEST_BASE_URL` overrides the application URL; `alerts.cjs` also supports `TEST_REPORT` for a separate prototype report. Test scripts use clock acceleration only for deadline checks. Resource sampling uses real time. Screenshot names beginning `prototype-` / `svelte-` form comparison pairs; `r6-` captures retain animated interaction states. Source/reference and upstream notices are under `mockups/ambient-v6/`.
