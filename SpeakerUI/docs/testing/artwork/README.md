# Spotify artwork verification — 2026-10-01

The playback normalizer previously discarded Spotify's images, and both music views always rendered placeholders. The existing playback response now retains one validated HTTPS artwork URL. Tracks use album artwork; episodes use their own artwork, falling back to the show when needed. No additional Spotify metadata request is made. The browser loads the public image without a referrer or Spotify authorization header.

Both the compact and expanded player render that image at their existing sizes without cropping. Image errors restore the original equalizer/disc placeholders; later valid artwork recovers. Unavailable playback clears old artwork with the track metadata. Disabled-provider demos remain unchanged.

Validation:

- `pnpm check`: zero errors and warnings.
- `pnpm test`: 62 offline tests passed, including album and episode artwork, malformed/missing metadata, URL validation and stale artwork clearing.
- `pnpm build`: optimized frontend and backend passed.
- `scripts/qa/providers.cjs`: 35 browser checks passed with controlled provider responses and locally fulfilled cover images. Covers both player sizes, next-track changes, missing images, failed loads, recovery, unavailable playback and existing control behavior. Evidence: `checks.json`.
- `scripts/qa/interactions.cjs`: 42 existing touch, gesture, demo, timing and layout checks passed with integrations disabled. Evidence: `regression/revision-6-checks.json`.
- Inspected `live-home-800.png`, `live-music-800.png`, and `live-music-1024.png`. Images use test artwork; no real Spotify playback actions were performed.

Run the provider suite against an isolated app with integrations disabled, with `TEST_OUTPUT_DIR=docs/testing/artwork`. The artwork requests are intercepted locally. Set `PLAYWRIGHT_MODULE` and `CHROMIUM_EXECUTABLE` to an existing browser test installation if needed.

Metadata contract: [Spotify playback state](https://developer.spotify.com/documentation/web-api/reference/get-information-about-the-users-current-playback).
