# Iris / SpeakerUI

Read the root [operating contract](../AGENTS.md), [requirements](../REQUIREMENTS.md), [operator runbook](../docs/HANDOFF-iris-rpi-v1.md) and [release evidence](../docs/EXPERIMENT-iris-release-v1.md). Current explicit user authorization governs implementation and actions.

- `src/App.svelte` mounts `ambient/Iris.svelte`. Show microphone state, tentative intent/slots, final acceptance/rejection and truthful execution receipts. Preserve the use-case panels and responsive 800×480, 1024×600 and mobile layouts.
- Recognition uses the resident local CPU worker and direct finite VCM, wake/VAD and experimental owner voice checks. No transcription, LLM, cloud recognition or automatic model/provider fallback.
- Tentative results are previews. Recorded WAVs, fixtures and benchmark profiles never execute actions. Voice dispatch requires an accepted final microphone command and passed owner check; explicit touch requests use the same canonical dispatcher. Recognizer/deployment disablement overrides server action enablement.
- Keep canonical command IDs/types aligned with `scripts/runtime/pcv_schema.py` in the parent repository. Missing bindings return `not_configured`. Keep credentials, action deduplication and the atomic timer/alarm/reminder store on the server. Unknown provider outcomes are never retried automatically.
- Keep the useful Iris `RuntimeDebug.svelte` and history view. The separate `?diagnostics` page is explicitly mock audio transport; its manually selected media controls use configured providers. Never label a mock as a native session.
- Use the root Python environment and shared Node runtime/dependencies. Worker numeric-library thread defaults match configured threads before Python imports; document and record explicit overrides in comparable measurements.
- Verify `npm test`, `npm run check`, `npm run build` and the relevant compiled-browser checks. Separate contract fixtures, recorded-WAV integration, host resource measurements, live microphone tests and physical Pi evidence.
- Preserve third-party notices and frozen evidence. Historical Ambient aura/simulator/design/QA files are superseded; their immutable recovery is indexed by the root [archive manifest](../experiments/iris-release-v1/archive-manifest.json). Do not restore them to the production import/build closure.
- Follow the parent repository's ownership boundaries. Do not stage, commit, publish, archive or delete unrelated files.
