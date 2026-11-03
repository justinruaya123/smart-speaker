# ME2 VCM baseline A snapshot and current project state

## Retained model and scope

The user requested committing the best retained model before further issue analysis. This snapshot preserves **baseline A: `shared-demo-slots-v1`, heads_s53, epoch38, 175,653 parameters**. It is the human-retention winner in the current shared-model series, not a global ranking across different command ontologies.

- Bundle: `submission/models/shared-demo-a/` (graph, normalizer, command order, policy and recipe).
- Selected recovery checkpoint: `submission/checkpoints/shared-demo-a-selected.pt`.
- Portable relative-path hashes and provenance: `submission/shared-demo-a/snapshot.json`.
- Catalog, original protocol, compact metrics and every-wording table: `submission/shared-demo-a/`.
- Full original selected-model report: [shared demo slots](EXPERIMENT-shared-demo-slots-v1.md).

All original model bytes are preserved. No fitting, threshold change, source-label change, application default switch, action enablement or remote push is part of this snapshot.

## What this model does

One local direct acoustic CNN emits action/percentage/alarm scores, decoded into a complete typed command or UNKNOWN. Runtime uses mono16kHz PCM and the exact frozen40-bin frontend/normalizer. No ASR, transcript parser, LLM or remote recognition fallback. Torch is unnecessary for ONNX inference; it is needed only to inspect/recover the selected native checkpoint.

The finite scope is **18 intent families,34 complete supported commands,102 canonical wordings**. Percentages are0/50/100; alarm times06:00/12:30/21:45 Asia/Manila. The frozen policy is action-authoritative argmax with rejection for missing required slots; independent-head scores are not calibrated joint confidence.

**Kitchen and Ceiling names are unsupported by A.** They belong to the separate176,681-parameter/50-command/150-wording candidates. Those candidates remain research diagnostics because they lost human/per-intent/old-slot retention. A cannot be described as retaining their expanded device coverage.

## Measured model state

| Evidence | Correct / total | Interpretation |
|---|---:|---|
| Retained human DEV |132/167 (79.04%)|Used for selection; not a fresh test|
| Old percentage/alarm DEV |108/108|Finite values, not arbitrary numeric recognition|
| Existing human regression |71/77 (92.21%)|Exposed supported cohort; not all human requests|
| Exact synthetic TEST |203/204 (99.51%)|102 wordings, two declared TEST voices|
| Context-added synthetic TEST |98/204 (48.04%)|Failure retained as diagnostic; user currently prioritizes exact commands|
| New synthetic percentage/alarm TEST |106/108 (98.15%)|Two supported rejections, no wrong accepted values in this subset|

Training completed48 epochs/4,608 updates. Six additional demo TRAIN voice IDs, two DEV and two TEST identities were reserved separately; existing human/synthetic training also contributed. These counts are not six total training people or broad independent-human voice qualification. Independent TTS listening/accent validation remains unverified.

Qualification remains false. A is retained as a scientific/recovery baseline, not a uniformly reliable production voice controller.

## Application and project state

`smart-speaker` owns the model and scientific evidence. `SpeakerUI` owns capture, wake/VAD/owner orchestration, UI and actuation. The user subsequently authorized publishing A to producer main and making the identical bundle the SpeakerUI default. Consumer integration is a separate commit with an immutable producer lock; qualification and action permission stay false. This producer snapshot does not restore removed application entrypoints or overwrite current training/frontend modules.

SpeakerUI retains the original258,912-parameter epoch44 bundle for explicit rollback and the176,681-parameter epoch39 PH candidate as an optional diagnostic. A is the newly authorized application default; none of these identities can be conflated. An installed/default selection is not evidence that an already-running process has reloaded, nor production qualification.

The user's present priority is exact-command recognition across diverse voices with speaker/music interference controlled. Arbitrary spoken-filler robustness is a recorded limitation, not the next priority. Current PC work is distinct from the broader still-unmeasured Pi/live-demo deliverables.

## Acoustic frontend evidence: promising but unsafe to enable by default

The separate consumer probe `SpeakerUI/spikes/002-reference-aec/` tested actual WebRTC reference AEC with NS/AGC/VAD disabled. It used recorded music through a declared simulated linear three-tap echo path, not physical-room microphone recordings. AEC and WASAPI loopback device enumeration work on this Windows host; live synchronized capture has not been exercised.

The existing exposed cohort contains204 exact synthetic commands from two TTS identities and77 human commands from two publisher human speaker IDs. Three signal-to-echo levels and playback-only controls produced3,127 observations. A separate saved-data audit reconciles25 metric cells and32 unchanged protected source/model/input bindings; eight native tests pass.

| Condition | Raw correct | Reference-AEC correct | Raw wrong accepted | AEC wrong accepted |
|---|---:|---:|---:|---:|
| Clean synthetic |203/204|203/204|0|0|
| Speech5dB above echo, synthetic |114/204|168/204|18|23|
| Equal-level speech/echo, synthetic |75/204|114/204|27|55|
| Clean human |71/77|70/77|3|3|
| Speech5dB above echo, human |23/77|53/77|5|11|
| Equal-level speech/echo, human |8/77|43/77|3|17|

Music-only **direct VCM** false accepts increase2→11/12 after AEC. These are not observed full-system wake/owner activations or executed actions. The prototype improves recognition counts but can replace safe rejection with incorrect commands. One clean human command also regresses. Do not enable it by default or relax VCM thresholds to conceal these failures.

No live audio was captured or played. The retained VCM, app environment and existing capture settings were unchanged. Failed partial run `full-01` remains preserved; `full-02` fixes only embedded-Parquet audio lookup and completes the same declared settings.

Local detailed evidence: `D:/School/AI231/SpeakerUI/.cache/spikes/002-reference-aec/full-02/`; user report: `C:/Users/Justin/AppData/Local/hermes/outputs/me2-speaker-echo-feasibility.md`. The consumer probe remains experimental source in SpeakerUI, not an enabled frontend or part of this producer model snapshot.

## Prioritized issues for the next analysis

1. **Wrong-command safety under playback.** Correct recognition gains alone are inadequate. Measure full wake/VAD/owner admission on music-only and simultaneous speech/playback before actuation; direct-classifier false accepts cannot be equated with real actions, nor ignored.
2. **Live reference quality and acoustic capture.** Select a real room microphone/speaker pair, then verify timestamp alignment, resampling/channel mapping, buffering, delay and clock drift with a raw bypass. Default enumerated devices are a JBL headset, not a validated room pair. External TV/phone audio without a reference is a different problem.
3. **Voice diversity evidence.** Broader unseen human voices, accents, delivery and distances remain unverified. Existing synthetic success and two publisher human IDs do not establish that. No new-human recordings were supplied in this phase.
4. **Coverage versus retention.** A preserves stronger human performance but lacks named Kitchen/Ceiling control. Expanded candidates cover them but failed retention. Keep this tradeoff explicit; do not infer Dining is acoustically better without a matched test.
5. **End-to-end readiness.** Actual room echo, live wake/endpoint/owner behavior, physical Pi resources and provider outcomes remain separate gates. PC replay and version control do not close them.

No further training, vocabulary expansion, architecture search or automatic default change is authorized merely by this issue list.

## Loading the committed snapshot

The exact historical inference/architecture closure is isolated inside `submission/shared-demo-a/source/`, rather than reintroducing removed producer application modules. From that directory with compatible runtime dependencies (`numpy`, `onnxruntime`, `einops`, `soundfile`):

```
python -B -m scripts.runtime.demo_slots_v1 --bundle ../../models/shared-demo-a --wav ABSOLUTE_PATH_TO_MONO_16KHZ_WAV
```

This emits recognition JSON only, with actions disabled and unqualified-model output retained. It does not provide a microphone server or execute a provider action. The exact snapshot command is separate from the existing application launcher.

Frozen artifact hashes are recorded in `snapshot.json`; `.gitattributes` prevents line-ending conversion of companion files. Raw audio, features, environment payloads, private settings, failed candidate weights and unrelated research changes are deliberately excluded.
