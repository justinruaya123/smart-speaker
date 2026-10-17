# Iris requirements and acceptance

The objective is a tiny standalone voice-command model (VCM) on Raspberry Pi 4/5, with local recognition and no LLM, cloud acoustic model, ASR/STT/transcription reader or model fallback. The release is a finite-command implementation. Arbitrary names, reminder prose and unrestricted spoken numeric values remain outside its measured acoustic coverage; configuration cannot create that coverage.

## Deliverables

| ID | Required evidence | Current release boundary |
|---|---|---|
| ME-01 | Versioned dataset, provenance, licenses and speaker-disjoint splits | Frozen human/synthetic manifests, separate constructed-overlap labels and audits |
| ME-02 | Individually trained VCM and reproducible recipe | Matched continuation fits, project initializers, source/environment hashes and optional separately identified random-init study |
| ME-03 | Fixed benchmark, unknowns and complete-command metrics | Unchanged 196 rows, unchanged 36 personal positives plus 52 negatives, human regression and fresh designed voices |
| ME-04 | Locked held-out validation, including failed gates | Selection before test; raw/accepted and source/intent/slot/speaker evidence |
| ME-05 | Real-world Pi demo | Software package/UI/actions/runbook delivered; physical Pi and fresh live microphone qualification unmeasured |
| ME-06 | Tiny real-time CPU execution | Parameters, deployed bytes and desktop timing measured; Pi latency/RAM/thermal measurements assigned to operator |
| ME-07 | Standalone recognition | Local CPU graphs and offline replay; optional HA integrations need the user's local HA service |
| ME-08 | Pure VCM, no LLM/transcription | Whole-command outputs, finite typed slots; separate wake/VAD/speaker models |

## Ten use-case categories

| ID | Category | Finite implementation and limits |
|---|---|---|
| UC-01 | Music | Generic play and supported song/artist IDs; supplied HA player/media IDs or explicit local playback. Arbitrary unseen song/playlist enrollment is unproven. |
| UC-02 | Questions/search | Local time and configured weather query. General questions/search are unsupported. |
| UC-03 | Lights | On/off for supported finite acoustic IDs; supplied HA light or explicit group bindings. |
| UC-04 | Brightness/color | Supported finite percentages/colors and target IDs. Validator range 0–100 does not establish 101 spoken values. |
| UC-05 | Timers | Finite recognized durations; persistent server scheduling, restart and missed-job behavior. |
| UC-06 | Alarms | Supported finite clock times; declared Asia/Manila command timezone, persistent local scheduling. |
| UC-07 | Temperature | Supported finite Celsius settings to supplied climate target with capability checks. |
| UC-08 | Media controls | Pause, stop, next and finite volume operations; explicit media provider, no fallback. |
| UC-09 | Reminders/lists | Finite task IDs and reminder list; no arbitrary speech transcription. |
| UC-10 | Calls/messages | Finite contacts and preset message; user-supplied HA scripts. Unmapped contacts/scripts are unconfigured. |

The reference set keeps all 31 baseline intent-and-slot outcomes and 93 wordings, plus the requested personal commands. The acoustic catalog is authoritative; unsupported output rows stay blocked by policy. [Baseline wordings](scripts/runtime/pcv_schema.py) and [current protocol](experiments/iris-release-v1/protocol.json) define the finite scope. Configuration binds recognized IDs to user entities; it does not enroll a new acoustic name.

## Acceptance and evidence

Promotion requires at least 30/36 **correctly accepted** personal positives, unchanged populations, no regression in original-human correct accepts or unknown false accepts against the predeclared same-width baseline, and no regression in accepted exact score on the 196-row scorer. Development selects width/objective/seed and fits thresholds before test. A failed candidate remains published evidence; the predeclared baseline supplies the release.

The personal cohort is generated speech from two held-out synthetic voices, with zero original-human personal positives. Fresh designed voices remain a separate holdout; requested characteristics are not verified demographics. Report wrong accepted positives and negative accepts alongside correct counts. Original human regression excludes ambiguous reference labels from accuracy denominators while retaining their counts.

Speaker isolation must use genuine backend evidence, same/impostor pairs, constructed overlap/switch, short/insufficient speech, timeout and reset checks. Short human utterance pairs and synthetic wake pairs are proxies; they do not qualify original human wake-to-command ownership.

Software completion requires export and decision parity, extracted-package replay, offline recognition, UI/worker integration, provisional guesses, final-only actions, duplicate handling, provider failures, entity validation, reconnects, persistent scheduler restart, immutable publication/hash verification and fresh-cache reconstruction. Physical Pi, fresh human wake/command testing and observed device action outcomes remain operator acceptance steps. See [release evidence](docs/EXPERIMENT-iris-release-v1.md), [model card](MODEL_CARD.md) and [Pi handoff](docs/HANDOFF-iris-rpi-v1.md).
