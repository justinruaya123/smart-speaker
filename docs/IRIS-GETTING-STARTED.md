# Iris: model and application repositories

- **smart-speaker (this repository):** training, calibration/export verification, immutable evidence and model distribution.
- **[SpeakerUI](https://github.com/justinruaya123/SpeakerUI):** the Ambient application, local inference worker, gateway, configuration/integrations, benchmarks, packaging and device deployment.

For the current application, follow [SpeakerUI README](https://github.com/justinruaya123/SpeakerUI/blob/main/README.md) and [operations](https://github.com/justinruaya123/SpeakerUI/blob/main/docs/OPERATIONS.md). Run application commands from that checkout, never this one. Its trusted asset lock pins exact model bytes; no sibling checkout or shared environment is required. Repository access/privacy has not changed.

## Verify models here

```sh
python -B -m scripts.optionb_epoch44 verify
python -B reproduce.py
```

These verify original artifacts; they never fit or bootstrap the application. [Original-source fitting/evaluation preparation](REPRODUCIBILITY-optionb-epoch44.md) remains explicit and preserves the failed CPU exact-fit comparison. [Repository boundary and migration](REPOSITORY-BOUNDARY.md) explains producer packaging and historical source locks.

## Historical monorepo runbook

The [unchanged runbook at the original source revision](https://github.com/justinruaya123/smart-speaker/blob/94c0021d7e22c9aa719246ac97105d7140a9f2a2/docs/IRIS-GETTING-STARTED.md) documents the old combined checkout. It is historical reconstruction material, not current launch instructions. Existing model data, protocols, failed gates and source snapshots are preserved; no acoustic or physical Pi qualification is added by this split.
