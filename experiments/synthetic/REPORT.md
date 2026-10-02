# Synthetic fixed-command VCM experiment

**Outcome: the all-intent human-speech quality target failed.** The frozen synthetic policy accepted no human test commands because every intent failed development gates. Before confidence gates, exact command recall was 95.66% on held-out generated voices and 23.23% on human speech; human absolute levels were correct on only 2/75 examples. Generating all 0-100 labels did not establish real-speech numeric recognition.

See the [comparison and conclusions](../mixed/COMPARISON.md), including the human-only control and mixed optimizer limitations. The models and policies remain frozen; this outcome paragraph was added after scoring.

Research-only results; installed VCM/STT and previous human-only studies are preserved.

Synthetic acoustic expert uses only generated training audio. Mixed uses separately trained human and synthetic experts with per-intent source preference; it is a calibrated ensemble, not a jointly fitted single model.

Weights were selected on source-specific development. Both systems' acceptance calibration uses human development; no final test labels were used by Optuna.

TTS utterances: 4032; human recordings: 11591. Qwen3-TTS 1.7B VoiceDesign creates original anchor voices; the 1.7B Base model produces whole English utterances. Filipino accents are requested by prompt and remain unverified.

| Test | N | Accepted | Correct accepted | Precision | Exact command recall | Negative false accepts |
|---|---:|---:|---:|---:|---:|---:|
| human_test | 1803 | 0 | 0 | n/a | 0.00% | 0/628 |
| synthetic_test | 672 | 0 | 0 | n/a | 0.00% | 0/96 |

## Human test by intent

| Intent | N | Accepted | Precision | Exact recall | Dev gate |
|---|---:|---:|---:|---:|---|
| UNKNOWN | 478 | 0 | n/a | n/a | None |
| POWER_ON | 251 | 0 | n/a | 0.00% | False |
| POWER_OFF | 287 | 0 | n/a | 0.00% | False |
| BRIGHTNESS | 242 | 0 | n/a | 0.00% | False |
| COLOR_SET | 129 | 0 | n/a | 0.00% | False |
| MEDIA_PLAY | 90 | 0 | n/a | 0.00% | False |
| MEDIA_RESUME | 37 | 0 | n/a | 0.00% | False |
| TIME_QUERY | 50 | 0 | n/a | 0.00% | False |
| DATE_QUERY | 59 | 0 | n/a | 0.00% | False |
| VOLUME | 180 | 0 | n/a | 0.00% | False |

## Confidence and source preference

`real_weight=0` uses the synthetic expert; `1` uses the human expert. Values between them blend acoustic probabilities. `log_odds_bias` changes the family prior. Intent, slot and margin thresholds must pass before emitting a command. Parameter heads cannot change the selected family.

Optuna ran 1,800 development trials per system. Targets were at least 95% accepted precision, at most 1% negative false acceptance and at least ten development accepts per intent. An intent without support is disabled, not promoted by reducing the gate.

```json
{
  "UNKNOWN": {
    "real_weight": 0.0,
    "log_odds_bias": 0.0,
    "intent_threshold": 0.5,
    "slot_threshold": 0.5,
    "margin": 0.0
  },
  "POWER_ON": {
    "real_weight": 0.0,
    "log_odds_bias": 0.9499484049024496,
    "intent_threshold": 1.1,
    "slot_threshold": 0.8910022511200848,
    "margin": 0.17920627606560746
  },
  "POWER_OFF": {
    "real_weight": 0.0,
    "log_odds_bias": -0.903771651126485,
    "intent_threshold": 1.1,
    "slot_threshold": 0.8830179238693056,
    "margin": 0.13726086673022467
  },
  "BRIGHTNESS": {
    "real_weight": 0.0,
    "log_odds_bias": -0.4500243052645161,
    "intent_threshold": 1.1,
    "slot_threshold": 0.6329791479769811,
    "margin": 0.09303704468032531,
    "disabled_operations": [
      "SET",
      "INCREASE",
      "DECREASE"
    ]
  },
  "COLOR_SET": {
    "real_weight": 0.0,
    "log_odds_bias": 1.2789765415452992,
    "intent_threshold": 1.1,
    "slot_threshold": 0.8428589401954906,
    "margin": 7.132732428133046e-05
  },
  "MEDIA_PLAY": {
    "real_weight": 0.0,
    "log_odds_bias": -0.9562238831501637,
    "intent_threshold": 1.1,
    "slot_threshold": 0.27263995982418243,
    "margin": 0.16851267708095585
  },
  "MEDIA_RESUME": {
    "real_weight": 0.0,
    "log_odds_bias": -0.43466871315491035,
    "intent_threshold": 1.1,
    "slot_threshold": 0.8913552125221382,
    "margin": 0.24536346804117626
  },
  "TIME_QUERY": {
    "real_weight": 0.0,
    "log_odds_bias": -1.4690543592481915,
    "intent_threshold": 1.1,
    "slot_threshold": 0.5710057481858144,
    "margin": 0.24802436643484643
  },
  "DATE_QUERY": {
    "real_weight": 0.0,
    "log_odds_bias": 1.0492216256718372,
    "intent_threshold": 1.1,
    "slot_threshold": 0.50269465679962,
    "margin": 0.2202213456237201
  },
  "VOLUME": {
    "real_weight": 0.0,
    "log_odds_bias": 0.5223903555032964,
    "intent_threshold": 1.1,
    "slot_threshold": 0.9437408597843853,
    "margin": 0.02394119344174111,
    "disabled_operations": [
      "SET",
      "INCREASE",
      "DECREASE"
    ]
  }
}
```

## Coverage and limits

All nine supported fixed intent families and UNKNOWN are in the manifest. Amount classes contain every integer 0-100. Bare UP/DOWN means ten percentage points. Per-value test counts and numeric/relative strata are in results.json; unobserved human values are not proven by parser coverage.

Known-corpus original human speech and held-out synthetic voices are separate tests. Names, artists, songs and new-word corpora are excluded. Silver labels are not human adjudication. No Filipino-human accent benchmark, live microphone, endpointed full-pipeline run or Raspberry Pi measurement is claimed.

## New words within fixed intents

A fixed softmax cannot learn acoustic words from a JSON alias alone. The next supported research path is labeled audio enrollment into an embedding prototype bank, with each prototype bound to an existing intent/finite value. New wording changes exemplars, not output classes. Version the enrollment set, require multiple independent examples, and recalibrate the margin/unknown gate on development before activation. A phonetic keyword graph is another option using reviewed pronunciations and a validated phoneme encoder. Neither path is enabled or claimed validated by this fixed-corpus experiment.

## Reproduce

Run from the root environment:
```powershell
.venv/Scripts/python.exe experiments/synthetic/acquire_tts.py
.venv/Scripts/python.exe scripts/synthetic.py prepare
.venv/Scripts/python.exe scripts/synthetic.py generate --batch-size 16
.venv/Scripts/python.exe scripts/synthetic.py train
.venv/Scripts/python.exe scripts/synthetic.py check
.venv/Scripts/python.exe scripts/synthetic.py evaluate
```

Reports and selection receipts are here; generated recordings are in data/audio/tts, manifests in data/tts, and model/feature artifacts in .cache. Test evaluation refuses to overwrite an existing result.

Sources: [Qwen models](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), [official inference API](https://github.com/QwenLM/Qwen3-TTS).

## Desktop CPU timing

Identical preloaded development PCM, two ONNX threads: p50 5.03 ms, p95 6.98 ms. Session loading: 54.40 ms.

These measurements include the frontend and finite decoder; mixed also includes both acoustic experts and probability blending. Disk I/O, speech duration, energy VAD and the 400 ms endpoint wait are excluded. They do not establish Raspberry Pi or microphone performance.

## Before confidence gates

| Test | Head-based precision | Exact command recall | Accepted |
|---|---:|---:|---:|
| human_test | 40.44% | 23.23% | 675 |
| synthetic_test | 95.99% | 95.66% | 574 |

The row above retains the learned readiness decision and finite-value validity but removes confidence, margin and disabled-operation gates. It is a diagnostic result, not an alternative calibrated operating point.

## Numeric operation support gates

```json
{
  "BRIGHTNESS/SET": {
    "passed": false,
    "accepted": 0,
    "precision": 0.0,
    "available_dev_positives": 47,
    "final_accepted": 0,
    "final_precision": null
  },
  "BRIGHTNESS/INCREASE": {
    "passed": false,
    "accepted": 0,
    "precision": 0.0,
    "available_dev_positives": 24,
    "final_accepted": 0,
    "final_precision": null
  },
  "BRIGHTNESS/DECREASE": {
    "passed": false,
    "accepted": 0,
    "precision": 0.0,
    "available_dev_positives": 53,
    "final_accepted": 0,
    "final_precision": null
  },
  "VOLUME/SET": {
    "passed": false,
    "accepted": 0,
    "precision": 0.0,
    "available_dev_positives": 3,
    "final_accepted": 0,
    "final_precision": null
  },
  "VOLUME/INCREASE": {
    "passed": false,
    "accepted": 0,
    "precision": 0.0,
    "available_dev_positives": 76,
    "final_accepted": 0,
    "final_precision": null
  },
  "VOLUME/DECREASE": {
    "passed": false,
    "accepted": 0,
    "precision": 0.0,
    "available_dev_positives": 80,
    "final_accepted": 0,
    "final_precision": null
  }
}
```

## Held-out synthetic voices by requested accent

Accent names below describe generator prompts. They are not verified accent labels or a Filipino-human benchmark.

| Requested accent | N | Accepted | Precision | Exact recall |
|---|---:|---:|---:|---:|
| en-GB | 336 | 0 | n/a | 0.00% |
| en-PH-requested | 336 | 0 | n/a | 0.00% |

## Model size

323,685 trainable parameters across 1 acoustic model(s); 1,348,493 ONNX bytes. TTS checkpoints and PyTorch are not part of this CPU inference path.
