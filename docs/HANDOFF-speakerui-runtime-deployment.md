# Smart speaker build and Raspberry Pi inference handoff

**Superseded design record.** The user subsequently requested direct implementation with SpeakerUI source inside this repository and `start.py` as the single entry point. Follow the current [README](../README.md#smart-speaker--speakerui) and implementation/tests. References below to a separate repository, `release.py` as the user entry point and `setup.sh` describe the earlier proposal, not current usage. The project-trained wake requirement remains unfulfilled; do not treat this design as evidence of trained wake performance.

Status: proposed implementation, revised 2026-10-02. Deliver one final pipeline: **smart-speaker is the inference backend; compiled SpeakerUI is the frontend**. The immediate acceptance milestone is a real microphone inference test on the Pi, with runtime-driven UI states and exact intent/parameter results.

Build and optimize the website on the PC. Package it with the selected runtime/models. The Pi serves the prebuilt website and performs inference. Keep first setup and model updates simple; no installation menus, additional service framework, or frontend redesign is required.

## 1. One PC command builds everything

Extend the existing [release.py](../release.py) and [release workflow](../scripts/pcv_release.py); reuse the canonical publisher.

Proposed commands:

~~~sh
# Bootstrap, acquire missing data, train, test, benchmark, build UI and publish.
python release.py

# Reuse selected models; test, build UI and publish without training.
python release.py --skip-train

# Optional output selection; default is both.
python release.py --skip-train --target standalone
python release.py --skip-train --target pi
~~~

Retain --run-id. Bootstrap the root .venv using standard-library code before importing heavy dependencies, then re-execute in it. Reuse existing environment/configuration/locks; remove host-specific interpreter and Codex Node-path assumptions.

The normal build acquires only missing pinned data and runs the selected VCM and **Hey Jarvis** wake-training workflows. Resume compatible completed stages; changed protocols/data require new run identities. --skip-train reuses bound selected models without fitting, training-data acquisition, threshold tuning or silent fallback. Missing selected artifacts fail clearly. Export/integrity and correctness checks still run.

The trained Hey Jarvis requirement remains. Record actual training, frozen/pretrained components, data provenance and held-out wake evaluation. Copying the current pretrained head is not project training. A missing trained wake model remains an explicit incomplete requirement; direct command-only diagnostics are not a passed wake pipeline.

For a Pi target, the PC workflow also:

1. Obtains a tested exact SpeakerUI integration commit from branch codex/voice-scaffold in a dedicated build checkout; preserve existing user work.
2. Installs its locked build dependencies and runs checks/build.
3. Packages its optimized dist/ website, dist-server/ gateway, startup scripts and deployment dependency metadata.
4. Binds the UI commit/assets, runtime version and model identities in the release manifest.

No Pi-side Git pull, Svelte compilation or Vite build is needed. The PC build may reuse an identical verified UI build, but --skip-train alone does not skip preparing the UI for a Pi publication. Standalone publication needs no UI checkout/build.

Two target ZIPs use the same authoritative model/decoder/policy:

| Output | Purpose |
| --- | --- |
| smart-speaker-standalone.zip | Reviewer installation, WAV/live inference and correctness tests without SpeakerUI or Node. |
| smart-speaker-pi.zip | First setup with the prebuilt website, or a model-only update detected by pnpm start. |

## 2. A compiled webpage can communicate with inference

Keep Vite's optimized HTML/JavaScript/CSS/assets bundle initially. A single HTML file containing its scripts can use the same API, but flattening everything is not required for deployment or fast inference. Preserve worklet files, icons, fonts, licenses and compression outputs needed by the existing app.

Serve the page from the local gateway, for example http://127.0.0.1:3000. Its JavaScript exchanges microphone controls through HTTP and receives pushed state/results through the existing WebSocket. The actual ONNX sessions run in the local backend process.

~~~text
PC: release.py -> train/reuse models + build SpeakerUI -> target ZIPs

Pi microphone -> Hey Jarvis/VAD -> resident ONNX runtime
                                        |
                                 typed results/states
                                        |
                          Node gateway and WebSocket
                                        |
                           prebuilt SpeakerUI webpage
~~~

SpeakerUI already has a Fastify static-serving path for dist/. Use it to serve website, API and WebSocket from one origin in deployed mode. Keep existing integrations server-side. Do not open index.html through file://, embed credentials in the webpage, or use the Vite development/preview server as production deployment.

[ Vite's static deployment guide ](https://vite.dev/guide/static-deploy.html) describes publishing dist/. The browser [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket) works independently of whether the page originated from a Svelte source build or one HTML file. Static files still require a running inference service for backend computation; they do not start Python themselves.

Retain the existing Node gateway to minimize changes. A Python-only web server could serve the same static UI later, but replacing the current gateway/integrations is outside this first milestone.

## 3. Standalone ZIP for the reviewer

Include a small setup.py, a thin run.py over the shared core runtime, complete bound models, target-appropriate locked dependencies, known audio fixtures, exact expected outputs and a test/report entry point.

After extraction, the intended one-command review is:

~~~sh
python setup.py --test
~~~

It creates the runtime .venv, installs appropriate dependencies and runs the real replay checks inside it. Print exact host-specific commands for subsequent WAV inference and live listening, such as run.py --wav command.wav --json and run.py --listen --mic select using that environment's Python.

Report expected/observed intents, every typed slot, rejections and failures. Keep accepted commands separate from diagnostic candidates; missing audio is untested. --test must not open a microphone. The package requires no SpeakerUI, Node, training framework or training corpus. Include dependency manifests for the declared reviewer platforms; Pi ARM64 wheels are not Windows wheels.

## 4. Pi ZIP first setup

After extracting smart-speaker-pi.zip, the user runs the included **setup.sh**. It:

1. Checks the target and installs/checks supported Python, Node/pnpm and audio-system prerequisites.
2. Creates the runtime .venv and installs its pinned dependencies.
3. Installs runtime code, complete models and standalone CLI/test entry points.
4. Installs the already-built SpeakerUI website/gateway and target-compatible production dependencies; preserves an existing private .env.
5. Configures the real runtime connection, verifies model loading, then invokes **pnpm start** and shows the local UI URL.
6. Offers microphone selection in the terminal or SpeakerUI and initializes listening.

The reviewed runtime wheels target Linux ARM64, CPython 3.12, glibc >=2.28; check the release manifest rather than assuming the OS Python matches. Production Node dependencies must be installed/bundled for Linux ARM64. Never copy Windows node_modules to the Pi. Remove the unused Node ONNX dependency from this deployment path if Python owns all inference, after verifying no other consumer needs it.

The Pi never trains or downloads the training corpus. First setup can download missing system/production dependencies; declare network needs clearly. Recognition and compatible local model updates work offline. Setup contains the website already, so it does not need GitHub access.

Suggested layout:

~~~text
~/smart-speaker/
  .venv/
  runtime/
  run.py
  models/releases/<bundle-id>/
  models/current
  state/                         # microphone, bindings, applied ZIP digest
  results/
  SpeakerUI/
    dist/                        # prebuilt website
    dist-server/                 # precompiled Node gateway
    scripts/
    package.json and lockfile
~~~

Models and credentials are private backend files, never public website assets.

## 5. Hotswap is copy the ZIP then pnpm start

The update workflow is:

~~~text
1. Copy a new release to ~/smart-speaker-pi.zip.
2. Run pnpm start in the installed SpeakerUI.
~~~

Before launching services, pnpm start checks that exact path using the OS home-directory API. It reads a flagged release manifest. Do not scan unrelated ZIPs or guess the newest archive.

Proposed manifest example:

~~~json
{
  "package_type": "smart-speaker-pi",
  "package_version": 1,
  "startup_action": "install-model",
  "bundle_id": "example-release",
  "runtime_api_version": 1,
  "output_schema_version": 3
}
~~~

The actual manifest additionally requires complete file SHA-256 inventories, model qualification, trained-wake provenance and runtime/UI compatibility. The Pi ZIP contains both setup assets and a clearly delimited model payload; startup hotswap consumes **only** that compatible model payload.

Startup checks:

- No flagged ZIP, or its digest was already applied: start the installed version.
- New flagged ZIP: stage and check hashes, safe archive paths, graph interfaces and runtime/UI compatibility.
- Stop only any prior runtime/server belonging to this installation, release capture, and atomically replace the complete model bundle.
- Load-check, then continue normal startup once. Record the ZIP digest only after successful readiness. On load/start failure, restore the previous model and restart it.

The bundle includes ONNX graph(s), normalization, catalog, custom slots, recipe, schema, policy and wake metadata. Replacing just model.onnx can invalidate the model contract.

**Hotswap does not train, benchmark, install dependencies, pull/rebuild SpeakerUI, or change its website, gateway, private .env or user data.** If new runtime code/dependencies or an incompatible UI protocol are required, stop with an instruction to use that release's setup.sh. The same ZIP's UI files are ignored during model-only hotswap.

Change scripts/start.mjs: it currently rebuilds frontend/backend and waits for dummy-onnx. Deployed pnpm start must serve valid prebuilt outputs, apply the model-update preflight, and wait for the real runtime. Do not recursively invoke pnpm start from the preflight. Keep explicit development/build commands separately.

Retain downloaded ZIPs and use the applied digest for idempotence. Copy via a temporary filename and rename to the watched name after copying finishes. Preserve a prior bundle for rollback. Never kill unrelated Node/Python processes or a process merely because it occupies a port.

## 6. Real runtime and frontend state contract

Use one persistent Python worker launched by the Node gateway, with control messages on stdin and bounded JSONL events on stdout; logs go to stderr. Reuse the existing frontend/ONNX/finite decoder instead of writing a second recognizer. The standalone CLI calls the same core without Node.

The runtime owns the Pi microphone. CLI and UI enumerate/select the same actual backend inputs. A remote browser's microphone list is not the Pi list. Noninteractive startup stays awaiting_microphone until UI selection. Persist the selected device descriptor, re-enumerate at startup, and do not silently switch to a different input.

Switching device cancels active speech, closes capture and resets wake/VAD. Mute releases capture. Show no-signal, unplug, unsupported sample rate and overflow errors. Maintain one capture owner; browser PCM diagnostics require an exclusive lease while native capture is paused.

Use mono 16 kHz PCM and explicit validated resampling/conversion. The selected VCM is complete-utterance: endpoint first, then features -> ONNX -> finite decoder -> policy. Do not run each transport frame as an independent command.

| Runtime state/result | SpeakerUI |
| --- | --- |
| awaiting_microphone | Microphone selector. |
| wake_listening | Idle/ready aura; “Say Hey Jarvis”. |
| command_ready / recording | Listening aura; “Speak now”. |
| processing | Thinking aura for actual work. |
| accepted final | Exact intent/slots and the corresponding supported screen. |
| rejected final | Actual rejection reason; separate labeled candidate preview if requested. |
| muted / capture_error / runtime_error | Actual condition and recovery control. |

Events carry runtime/utterance identity and increasing sequence. Forward the current schema-v3 accepted/command/reason output from [pcv_schema.py](../scripts/runtime/pcv_schema.py), model identity and measured timing. Diagnostic candidates stay separate. Do not reinterpret the CTC score as a generic confidence or replace the frozen policy with INTENT_MIN_CONFIDENCE.

Only authoritative accepted output may enter normal intent handling. A diagnostic preview can display a candidate but cannot execute it. Ignore cancelled/duplicate/late finals and do not replay actions after reconnect. Simulator fixtures remain explicitly simulated and never serve as fallback for a failed runtime.

Real events bypass the current 380 ms simulated delay and fabricated success handlers. Drive the aura from actual runtime state; “speaking” is reserved for actual playback.

### Exact intent and parameter adaptation

Preserve raw typed output in history and map IDs through persistent installation-specific bindings. Missing bindings request input; unsupported capability stays explicit. Add only the handlers needed to display/test these results; no new external service providers.

| Model intent | Slots | Frontend adaptation |
| --- | --- | --- |
| PLAY_MUSIC | Optional song_id; artist_id requires song_id | media.play with bound media IDs; no IDs means resume, never a fixture track. |
| WEATHER | None | answer.weather; unavailable/demo data remains labeled. |
| TIME | None | answer.time showing configured local time. |
| LIGHT_ON | light_id | light.on with bound allowed entity/group. |
| LIGHT_OFF | light_id | light.off; light.all is only the configured group. |
| PAUSE | None | media.pause for the configured target. |
| STOP | None | media.stop; report unsupported instead of silently substituting pause. |
| NEXT | None | media.next. |
| VOLUME | operation; percent for set only | media.volume preserving up/down/mute/set and exact explicit values. |
| CALL | contact_id | call.start preview with bound contact; no camera inference or real call. |
| MESSAGE | None | message.compose/request input; no recipient/content exists in this schema. |
| LIST_REMINDERS | None | reminder.list. |
| TIMER | duration_seconds | timer.set with exact seconds. |
| ALARM | time, timezone | alarm.set preserving Asia/Manila from the current schema, not browser-local substitution. |
| TEMPERATURE | degrees, unit C | climate.set_temperature preview; do not clamp model -50..60 to UI demo 16..30. |
| BRIGHTNESS | light_id, percent | light.brightness with entity_id and brightness_pct. |
| COLOR | light_id, color | light.color with explicit supported mappings; no guessed nearest color. |
| CREATE_REMINDER | task_id | reminder.create with resolved text; request a missing deadline. |

The current UI lacks some handlers and supports fewer color names than the model. Display unsupported results honestly. Preserve existing touch integrations and disable actions during inference tests.

## 7. Start simple and measure the Pi

First keep sessions/decoder resident, avoid live WAV-file roundtrips, keep capture/control responsive, send events immediately, and reuse the PC-built website. Measure one versus two VCM threads with SpeakerUI active before adding more tuning.

Report actual Pi hardware, graph/runtime identity, p50/p95 inference and speech-end-to-final-UI turnaround, sample counts and audio drops. Separate endpoint waiting, model, decoder and UI delivery. Keep the current 320 ms wake-release and 400 ms command silence endpoint for the first compatible test. Any shorter timing requires acoustic truncation checks.

A C/C++ port or custom execution provider is not required for this first milestone; profile before undertaking it. The reference [threading guide](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) describes the relevant controls, not a proven speedup for this model.

## 8. Implementation and acceptance

Change only these integration areas:

- smart-speaker release.py/workflow: environment bootstrap, --skip-train, target selection and PC SpeakerUI build.
- Existing publisher: two target ZIPs, included setup entry points, prebuilt web/gateway assets, flagged manifest and complete bound model payload.
- Existing core runtime: persistent worker, microphone controls, real state/result events and standalone wrappers.
- SpeakerUI scripts/start.mjs: home-ZIP detection, model update, no deployed rebuild, real readiness and owned-process lifecycle.
- SpeakerUI gateway/protocol/adapter/controller: microphone controls, schema-v3 results, pushed states and the exact mapping above.

Completion checks:

- [ ] One PC command builds the selected outputs; --skip-train performs no fitting or training-data acquisition.
- [ ] Standalone review produces actual expected/observed results without SpeakerUI.
- [ ] Pi setup installs the environment and prebuilt application, then starts the real backend via pnpm start without cloning/building UI.
- [ ] Terminal/UI microphone selection controls the same Pi input.
- [ ] Live speech generates real wake/listening/processing/final events and exact intent/parameter display.
- [ ] Command, silence/noise, rejection, cancellation and capture errors never substitute mock results.
- [ ] All 18 mapping cases are checked independently of acoustic accuracy.
- [ ] A home-directory flagged ZIP changes model identity through pnpm start without a UI rebuild, benchmark, dependency change or training.
- [ ] Repeated ZIPs are idempotent; bad/incompatible updates preserve the prior installation.
- [ ] The same audio gives the same authoritative output through CLI and SpeakerUI.
- [ ] Actual Pi turnaround and failed recognition cases are recorded.

The current model policy is disabled. Showing real diagnostic predictions/rejections proves the inference connection, not successful command qualification. Missing/failed trained wake evidence also remains explicit. Do not enable acceptance to make the UI appear functional.

Preserve dirty work, installed artifacts, frozen studies, credentials and existing UI design. This handoff does not authorize unrelated cleanup, commits, pushes or physical actions. Read both repositories' AGENTS.md before implementation.

Reviewed SpeakerUI baseline: [aa1d829](https://github.com/justinruaya123/SpeakerUI/tree/aa1d8295462d4b6e733da45c081b7802569e80a1), including [startup](https://github.com/justinruaya123/SpeakerUI/blob/aa1d8295462d4b6e733da45c081b7802569e80a1/scripts/start.mjs) and [static-serving gateway](https://github.com/justinruaya123/SpeakerUI/blob/aa1d8295462d4b6e733da45c081b7802569e80a1/server/main.ts). It remains mock-only until adapted. Reuse the current [runtime](../scripts/runtime/pcv_final_runtime.py), [publisher](../scripts/pcv_final_delivery.py) and [release evidence](../experiments/pcv-release-train-v1/workflow/20261001T193735329546Z-76e34e/workflow.json).

The requested result is **build on PC -> ZIP -> setup or model hotswap -> pnpm start -> microphone -> ONNX inference -> live SpeakerUI states/results**, with a short real Pi test receipt.
