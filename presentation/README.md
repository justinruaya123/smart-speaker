# Iris animation source

Narrative: sound becomes one finite command, not a transcript. The application then independently checks ownership and typed parameters; a receipt distinguishes a real provider result from a labeled synthetic one. A fast command classifier is not a full interaction-latency claim.

- **IrisPipeline**: sound/wake → capture → command decision → independent guard → configured action → truthful receipt. Read left to right; the companion owner lane never becomes a second recognizer. The last frame is the reduced-motion poster.
- **IrisModel**: log-mel time/frequency lattice → stride-two causal stem → four two-stage residual blocks → valid-length mean+max → one command/UNKNOWN head. The “aha” is that the head chooses one whole command, not independently composed slots and not words.

Palette is taken from `SpeakerUI/src/ambient/Iris.svelte`: ink `#090d11`, foreground `#e2ecef`, mint `#9ce8d3`, muted `#96adb7`, panel `#14212a`, border `#30444e`. Consolas is used in Manim's geometric labels; system/Inter is used in the HTML. No external fonts or TeX are needed. Architecture labels may be baked into video; changing measurements stay in the DOM and `iris.json`.

## Compatibility found during actual rendering

`manim==0.22.0` installed but `manim-slides==5.7.0` failed before construction: its adapter accesses the removed `ManimConfig.write_to_movie`. This source therefore pins **Manim 0.20.1 + manim-slides 5.7.0**, the actually exercised supported path, instead of claiming the proposed pair rendered. `ManimColor` is used explicitly for the background because the slide exporter expects `.to_hex()`.

## Rebuild (presentation-only environment)

From repository root, with uv and ffmpeg/ffprobe on PATH:

```sh
uv venv --python 3.13 .cache/presentation-venv
uv pip install --python .cache/presentation-venv/Scripts/python.exe -r presentation/requirements.txt
.cache/presentation-venv/Scripts/python.exe presentation/render.py --quality draft
.cache/presentation-venv/Scripts/python.exe presentation/render.py --quality production
```

On POSIX replace `Scripts/python.exe` with `bin/python`. The orchestrator chooses its own environment's CLI. It really executes `manim-slides render` using the Cairo CPU renderer, then `manim-slides convert --offline`; source media and slide intermediates go to `.cache/presentation-render/`. Production is **1920×1080, 30fps**. Finished MP4s, generated Reveal exports and offline dependencies, posters and a hash-bound ffprobe receipt are copied only into `docs/site/animations/`. Draft is 854×480, 15fps and does not overwrite shipped assets. No scientific environment is synchronized or modified.

All subprocess temporary paths are explicitly placed in the worktree's `.cache/presentation-render/<quality>/temporary` on the render volume. This avoids Manim Slides' separate temporary concat files filling the Windows system drive. The generated exports retain genuine Reveal sections/assets, with only Windows URL separator normalization, language declaration and removal of a zoom restriction applied afterward.

The browser checks use presentation's own pinned Playwright1.62.1, not the application dependency tree. From the producer root run `pnpm --dir presentation install --frozen-lockfile`, `pnpm --dir presentation exec playwright install chromium`, then `node presentation/qa.mjs`. A preinstalled compatible Chromium can be selected with `IRIS_BROWSER_EXECUTABLE` or a named installed channel with `IRIS_BROWSER_CHANNEL`; no browser or science lock is rewritten. The captured product image is explicitly **base-main c062894**, an action-disabled software fixture, not a live acoustic/provider result. Ten use-case screenshots, all six chapters at 1920×1080/1280×720/390×844, keyboard/count/fullscreen/deep-link persistence, real local Reveal initialization and reduced-motion posters are exercised by `presentation/qa.mjs`. Narrow view is a scaled16:9 deck, not a separate mobile application layout.

## Release gate

The completed native and same-size companion synthetic wake tests both failed; neither replacement nor a working acoustic release is claimed. `python presentation/validate.py --release-gate` intentionally exits1 for that receipt. Pages deployment requires this separate gate as well as static/browser checks. Do not bypass it because the visuals/software tests pass.

Serve the finished deck using `python -m http.server 8765 --directory docs/site`, then open localhost:8765. Python is needed by the authoring server, not by the viewer. Any ordinary static host works; no viewer GPU, microphone or model runtime is used. Run the stdlib validator and optional existing-Playwright browser QA described in `presentation/qa.mjs`.
