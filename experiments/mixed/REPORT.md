# Mixed fixed-command VCM experiment

**Outcome: the all-intent human-speech quality target failed.** The frozen mixed policy accepted 14 correct DATE_QUERY commands out of 1,175 valid human test commands (1.19% exact recall). Every other family and all numeric operations failed development gates. Before confidence gates, human exact recall was 54.47%, with 67.30% output precision and only 2/75 absolute levels correct.

See the [comparison and conclusions](COMPARISON.md). The human-only control obtained 15.15% test recall but only 89% precision; the mixed coordinate search did not preserve that stronger development endpoint. Models and policies remain frozen; this outcome paragraph was added after scoring.

Research-only results; installed VCM/STT and previous human-only studies are preserved.

Synthetic acoustic expert uses only generated training audio. Mixed uses separately trained human and synthetic experts with per-intent source preference; it is a calibrated ensemble, not a jointly fitted single model.

Weights were selected on source-specific development. Both systems' acceptance calibration uses human development; no final test labels were used by Optuna.

TTS utterances: 4032; human recordings: 11591. Qwen3-TTS 1.7B VoiceDesign creates original anchor voices; the 1.7B Base model produces whole English utterances. Filipino accents are requested by prompt and remain unverified.

| Test | N | Accepted | Correct accepted | Precision | Exact command recall | Negative false accepts |
|---|---:|---:|---:|---:|---:|---:|
| human_test | 1803 | 14 | 14 | 100.00% | 1.19% | 0/628 |
| synthetic_test | 672 | 10 | 10 | 100.00% | 1.74% | 0/96 |

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
| DATE_QUERY | 59 | 14 | 100.00% | 23.73% | True |
| VOLUME | 180 | 0 | n/a | 0.00% | False |

## Confidence and source preference

`real_weight=0` uses the synthetic expert; `1` uses the human expert. Values between them blend acoustic probabilities. `log_odds_bias` changes the family prior. Intent, slot and margin thresholds must pass before emitting a command. Parameter heads cannot change the selected family.

Optuna ran 1,800 development trials per system. Targets were at least 95% accepted precision, at most 1% negative false acceptance and at least ten development accepts per intent. An intent without support is disabled, not promoted by reducing the gate.

```json
{
  "UNKNOWN": {
    "real_weight": 0.5,
    "log_odds_bias": 0.0,
    "intent_threshold": 0.5,
    "slot_threshold": 0.5,
    "margin": 0.0
  },
  "POWER_ON": {
    "real_weight": 0.9005742319282527,
    "log_odds_bias": 1.0714387054133423,
    "intent_threshold": 1.1,
    "slot_threshold": 0.8939937358011273,
    "margin": 0.15832037137528088
  },
  "POWER_OFF": {
    "real_weight": 0.9577050522592653,
    "log_odds_bias": -1.4714788424027965,
    "intent_threshold": 1.1,
    "slot_threshold": 0.8979762625413154,
    "margin": 0.051654039906318845
  },
  "BRIGHTNESS": {
    "real_weight": 0.6543839890694344,
    "log_odds_bias": -1.3382854281348129,
    "intent_threshold": 1.1,
    "slot_threshold": 0.6692236576655604,
    "margin": 0.010913707467489854,
    "disabled_operations": [
      "SET",
      "INCREASE",
      "DECREASE"
    ]
  },
  "COLOR_SET": {
    "real_weight": 0.7457244781302267,
    "log_odds_bias": -0.3641120293589159,
    "intent_threshold": 1.1,
    "slot_threshold": 0.6821629461233072,
    "margin": 0.15111564320321438
  },
  "MEDIA_PLAY": {
    "real_weight": 0.1498348071935201,
    "log_odds_bias": 0.22472201377820478,
    "intent_threshold": 1.1,
    "slot_threshold": 0.755557666394787,
    "margin": 0.15403804734882393
  },
  "MEDIA_RESUME": {
    "real_weight": 0.8816723992813355,
    "log_odds_bias": 0.8599280508001029,
    "intent_threshold": 1.1,
    "slot_threshold": 0.9002775026434348,
    "margin": 0.05996159669122961
  },
  "TIME_QUERY": {
    "real_weight": 0.09997495132786108,
    "log_odds_bias": 0.8385904238952748,
    "intent_threshold": 1.1,
    "slot_threshold": 0.6001219102216866,
    "margin": 0.04932758787182957
  },
  "DATE_QUERY": {
    "real_weight": 0.4660754266421964,
    "log_odds_bias": -1.4065890632306617,
    "intent_threshold": 0.5909150258519738,
    "slot_threshold": 0.6507173947973909,
    "margin": 0.2298586840824511
  },
  "VOLUME": {
    "real_weight": 0.8232177323418595,
    "log_odds_bias": -1.4710655812546258,
    "intent_threshold": 1.1,
    "slot_threshold": 0.2681473131428189,
    "margin": 0.065507839655398,
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

Identical preloaded development PCM, two ONNX threads: p50 7.18 ms, p95 9.76 ms. Session loading: 147.37 ms.

These measurements include the frontend and finite decoder; mixed also includes both acoustic experts and probability blending. Disk I/O, speech duration, energy VAD and the 400 ms endpoint wait are excluded. They do not establish Raspberry Pi or microphone performance.

## Before confidence gates

| Test | Head-based precision | Exact command recall | Accepted |
|---|---:|---:|---:|
| human_test | 67.30% | 54.47% | 951 |
| synthetic_test | 26.76% | 17.19% | 370 |

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
| en-GB | 336 | 4 | 100.00% | 1.39% |
| en-PH-requested | 336 | 6 | 100.00% | 2.08% |

## Model size

647,370 trainable parameters across 2 acoustic model(s); 2,696,986 ONNX bytes. TTS checkpoints and PyTorch are not part of this CPU inference path.

## Human-only control

This control uses the same human expert, development budgets and acceptance targets, with real_weight fixed to one. It supplies a comparison for the generated-speech ensemble. Independently selected policies and the coordinate optimizer limitation mean this is not an isolated causal estimate of synthetic data's value.

| Test | Accepted | Precision | Exact command recall | Negative false accepts |
|---|---:|---:|---:|---:|
| human_test | 200 | 89.00% | 15.15% | 3/628 |
| synthetic_test | 7 | 57.14% | 0.69% | 2/96 |
