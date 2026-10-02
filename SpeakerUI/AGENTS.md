# Ambient / SpeakerUI

- Do not overengineer. Preserve unrelated working changes and the audio diagnostic workflow.
- The approved product identity and interaction contract are in `docs/design/ambient.md`.
- `mockups/ambient-v6/04-ambient.html` is the retained visual reference. Match its geometry, typography, colors, animation and touch behavior before introducing new design choices. Document intentional deviations and verify the compiled screen against it.
- Target an 800 × 480 landscape, 7-inch Raspberry Pi 4 touch display first; also verify 1024 × 600. No topbar or sidebar. Dark Ambient palette, sparse text, separate glass controls, centered live aura and clock.
- Implement the SPA in Svelte 5, using component-owned reactive state. Keep native gesture/canvas lifetimes explicit and cleaned up. Avoid new animation, icon, state or component libraries when the platform or existing code suffices.
- Keep every approved animation. Bake fixed icons, shapes and CSS motion at build time; the LiveKit-derived aura stays live. Pause hidden rendering and honor reduced motion. Do not call a desktop browser measurement a Pi benchmark.
- All inference remains explicitly dummy/simulated until a real ONNX contract and model are supplied. No ASR is implied. Keep credentials server-side, never in VITE variables, generated assets, fixtures or logs.
- Changes to Home Assistant/Spotify configuration do not authorize live device actions. Preserve disabled-by-default integrations and existing inference safety boundaries.
- Verify pointer drag, flick velocity, gesture cancellation and event ownership. Only the swiped reminder is completed. Reminder reordering belongs exclusively to its handle and must never open controls.
- Preserve the diagnostic page and existing backend tests. Run type checks, offline tests, optimized build, 800 × 480 browser interaction/layout checks and resource measurements for a substantial UI change.
- Do not commit, stage, deploy, or discard existing work unless explicitly requested.
