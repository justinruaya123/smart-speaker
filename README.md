# Iris

A local finite-command smart speaker for Raspberry Pi 4/5, with Hey Iris activation, command VAD, a direct voice-command model, a separate experimental speaker check, a Svelte interface and configured actions. No transcription, LLM or cloud recognition runs in the product.

```bash
python train.py                       # verified inputs, bounded fitting, checks and package
python train.py --skip-train          # verify selected lineage and rebuild/evaluate
python train.py --skip-train --publish
```

Use the root environment; pinned inputs and permitted project checkpoints are downloaded and hash-verified. The optional `--from-scratch` is a separate random-init study, not identical historical initialization. Training needs authorized access to the original data sources; restricted audio is not republished.

[Model and Pi downloads](https://huggingface.co/justinruaya123/ai231-smart-speaker) · [Model card](MODEL_CARD.md) · [Pi setup and operator tests](docs/HANDOFF-iris-rpi-v1.md) · [Experiment evidence](EXPERIMENTS.md) · [Requirements and gaps](REQUIREMENTS.md) · [Dataset provenance](docs/DATASETS.md)

The UI shows live provisional guesses, final intent/slots and execution receipts. Only accepted final microphone commands with speaker evidence execute. Supply Home Assistant entities/scripts; missing mappings return unconfigured. WAV replay and benchmarks never execute actions. Timers, alarms, finite reminders and execution identifiers persist across restarts.

This is an experimental software release. The model card reports exact held-out scores and failed gates. Original human wake/command qualification, physical Pi measurements and observed device action results remain operator acceptance work.
