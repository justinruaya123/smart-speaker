# Voice debug service

Run from the repository root using the shared workspace:

```powershell
uv sync --locked
uv run --locked python scripts/check_setup.py
uv run --locked python -m src.voice --list-mics
uv run --locked python -m src.voice --mic 2
```

Replace `2` with an input number from `--list-mics`. Microphone numbers can change
when devices reconnect. The app and model package use the root `.venv` and
`uv.lock`; see the [repository setup](../../README.md).

Model artifacts are ignored by Git. On a fresh clone, restore or rebuild
`model/artifacts/tinyvcm/` using the repository instructions before starting.

The command runs in the foreground until **Ctrl+C**. Optional flags are
`--channel 0`, `--capture-rate 48000`, and `--bundle path/to/tinyvcm`.
The default bundle resolves to `model/artifacts/tinyvcm` in this checkout.

## Pipeline

```text
numeric microphone index
    -> audio capture (40 ms blocks)
    -> energy VAD waits for speech
    -> retain pre-roll and stream audio into the local VCM
    -> print provisional intent and parameter requirements
    -> end utterance after silence; print final result
    -> return to waiting for speech
```

The service reuses `tinyvcm.cli.listen` and `tinyvcm.inference.Recognizer` from
`model/`. Capture uses up to 200 ms of pre-roll, 400 ms of trailing silence, and
an eight-second utterance limit. Input is converted to 16 kHz for inference.
An overlong utterance or audio overflow is rejected. The VAD uses audio energy
against an adaptive noise floor; loud non-speech can also trigger it.

JSON debug events are flushed to stdout. The selected microphone and actual
capture rate are printed to stderr by the shared listener. An illustrative
sequence, shortened to the relevant fields, is:

```json
{"stage":"service","status":"starting","microphone":2,"parameter_asr":false}
{"stage":"vad","status":"speech_started"}
{"stage":"intent","intent":"COLOR_SET","confidence":0.9995,"provisional":true,"parameters_needed":true,"required_parameters":["target","color"]}
{"stage":"result","status":"needs_parameters","intent":"COLOR_SET","confidence":0.997,"provisional":false,"parameters_needed":true,"required_parameters":["target","color"]}
```

A final `TIME_QUERY`, for example, has `parameters_needed: false` and
`required_parameters: []`. For rejected model predictions, `intent`, `parameters_needed`, and
`required_parameters` are `null`; `vcm_command` retains the rejected candidate
for debugging. Early events are provisional and can be retracted at the end of
the utterance. The existing model confidence thresholds are retained.

## Parameters for the current trained intents

| Intent | Required input |
|---|---|
| `POWER_ON`, `POWER_OFF` | `target` device name |
| `BRIGHTNESS_UP`, `BRIGHTNESS_DOWN` | `target`; amount is optional, defaulting to ten percentage points |
| `COLOR_SET` | `target`, `color` |
| `MEDIA_PLAY` | `selection_text`: media description or an explicit generic request such as "music" |
| `MEDIA_RESUME`, `TIME_QUERY`, `DATE_QUERY` | None |
| `UNKNOWN` | Rejected |

`parameters_needed` describes the intent's input requirements. This service
prints those requirements; parameter ASR is disabled, so it does not extract
parameter values or conduct a follow-up dialogue. It performs no device actions.
The field definitions also cover additional commands supported by the model's
router, but only labels in the loaded bundle can be recognized. Current time
and date intents represent local queries and do not extract a place name.

## Python entry point

```python
from src.voice import start_service

start_service(2)
```

## Check the service

```powershell
uv run --locked python -m unittest discover -s tests -p "test_voice.py" -v
```

Tests use simulated microphone blocks to exercise the shared VAD and repeated
utterance lifecycle without recording the host microphone. See
[`model/README.md`](../../model/README.md) for the trained model's measured
accuracy and rejection limitations.
