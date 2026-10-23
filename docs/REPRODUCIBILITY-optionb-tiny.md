# OptionB tiny snapshot and GitHub reproduction

## Frozen model

The selected finite OptionB candidate is the original **width160 /258,912-parameter epoch44** atomic-command CNN, with its exact 16kHz/40-bin frontend, human-TRAIN normalizer, 32-class order and separate **.50 posterior /.10 margin** demo policy. No failed fine-tune, larger encoder, ASR, transcript parser or independent intent/slot acceptance gate replaces it.

The snapshot is versioned on **`snapshot/optionb-tiny-epoch44`**, separate from the historical `codex/final-submission` branch's push-triggered 2M release workflow. The snapshot does not change installed runtime defaults, qualification or action permissions.

- Bundle: `submission/models/optionb-tiny/` (all five original files).
- Selected recovery checkpoint: `submission/checkpoints/optionb-tiny-selected.pt`.
- Model/data/source identities: `submission/optionb-tiny/snapshot.json`.
- Metadata-only compressed membership, references, original tiny feature hashes and golden predictions: `submission/optionb-tiny/inputs/`.
- Original training: CPU, four Torch threads, seed53, six human-only epochs then42 mixed epochs,96 batches of32 per epoch. Preserve this recipe when testing training reproducibility; using CUDA is a different numerical run. The measured original fit took2,987.91 seconds.

The input snapshot preserves exact ordered membership and labels. It removes obsolete local audio/feature paths and the older2M feature-cache hashes; those are not tiny-model fitting inputs. The tiny fit freshly extracted40-bin features with BLAS limited to one thread. Its24,566 original TRAIN/DEV tensor hashes and original1,889-human normalization membership are retained separately. Source records identify ten pinned Parquet shards and184 auxiliary synthetic DEV recordings. No raw recordings, feature tensors, private configuration or environment payloads are committed. Source-specific data/model terms remain applicable; see [dataset terms](DATASETS.md) and the original reports.

## Fresh-checkout baseline

Clone the dedicated branch into a **new directory**. Do not copy an old `.venv`, `.cache`, recordings or feature tensors into the initial test. Use the repository's unchanged lockfile and a new root environment:

```sh
uv sync --locked --python 3.13.14 --group training --group benchmark
uv run --no-sync python -m scripts.runtime.optionb_worker --bundle submission/models/optionb-tiny --check
```

The second command verifies the exact tracked candidate and loads CPU ONNX without operating a microphone or providers. The older root `reproduce.py`, `benchmark.py` and submission package builders still target the historical2M/935k workflows; they are not evidence of tiny-model reproduction.

**Full tiny-model clean-clone reproduction is not yet verified at this snapshot.** The old tiny research CLIs refer to ignored study caches and immutable historical source locks. Running them from a clean checkout is an acceptance test, not permission to silently copy those caches, weaken hashes or overwrite their receipts. The requested GitHub experiment must record the original failing command/stage before any portable follow-up fix. Later reconstruction must download exact public source bytes, verify original WAV/PCM identities, regenerate the tiny TRAIN/DEV features and normalizer, and compare actual fresh inference/metrics with the golden expectations. A training reproduction must perform all4,608 updates and compare selected epoch/tensors separately from source/model inference replay. Elapsed time and latency need not be bitwise equal.

## What stays unqualified

[The current release report](RELEASE-optionb-tiny-v1.md) retains exposed-cohort metrics and known errors. Reproducing them does not establish fresh-blind accuracy, independent listener truth, all93 human wording cells, live capture or Pi readiness. There are no new independent-speaker recordings. Pi tests are excluded; actions remain disabled, and the snapshot is not a promoted/installed release.
