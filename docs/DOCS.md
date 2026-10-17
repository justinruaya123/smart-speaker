# Documentation and archive receipt

The final release uses one requirements file ([REQUIREMENTS.md](../REQUIREMENTS.md)), one experiment directory ([EXPERIMENTS.md](../EXPERIMENTS.md)), one current [release report](EXPERIMENT-iris-release-v1.md), one [dataset directory](DATASETS.md), the [model card](../MODEL_CARD.md) and [Pi handoff](HANDOFF-iris-rpi-v1.md).

The authorized research snapshot was pushed **before cleanup** to private GitHub branch `archive/iris-research-20261017`, immutable commit [`ae1f1d7c67616e4efbae4619ac75620495e6073a`](https://github.com/justinruaya123/smart-speaker/tree/ae1f1d7c67616e4efbae4619ac75620495e6073a). Its Git tree `f9a037ec364ca3abe49ce9896fcc8fa7897c510d` matches the local reviewed research snapshot. [publication-archive.json](../experiments/iris-release-v1/publication-archive.json) records the tree comparison and ignore-only predecessor commit.

[archive-manifest.json](../experiments/iris-release-v1/archive-manifest.json) is the exact 1,337-file cleanup plan and execution receipt. Every named file was verified against its recovery hash before mutation and rechecked immediately before removal. These paths cover historical experiments, obsolete model/runtime/trainers, visualizer, historical generated reports and packages. The last 42 paths remove unreachable former UI components, visualization assets, their obsolete helper tests and stale validators after their build/documentation consumers were updated. The active UI retains useful diagnostics and all third-party notices. Useful historical results, including failures, remain on the immutable archive and are summarized/linked in the current experiment directory. No history was rewritten.

Only the isolated release checkout was changed. Original dirty research worktrees, shared source recordings, other users' running jobs and historical ignored caches were not moved, deleted, reset or overwritten. Archived payloads already tracked in Git are recoverable from the archive tree. [archive-payloads.json](../experiments/iris-release-v1/archive-payloads.json) records a read-only audit of 176 retained historical payloads (681,788,861 bytes), all stable during hashing. These ignored candidates retain their original shared-cache recovery paths and are not newly uploaded. Large raw recordings, generated audio, features and candidate weights remain outside Git; selected input/checkpoint recovery is hash-bound by the published reproduction receipt. Upstream restricted audio remains at pinned original sources. The original primary source card remains at its receipt-declared historical path as a provenance exception; third-party source/license notices remain preserved.

To recover a named historical file without changing branches:

```bash
git fetch origin archive/iris-research-20261017
git show ae1f1d7c67616e4efbae4619ac75620495e6073a:PATH > /YOUR/RECOVERY/DESTINATION
```

Check the restored SHA-256 against the manifest. Restoration is explicit; ignored legacy paths must not silently become active runtime dependencies again. A new maintenance task must inventory ownership, changed hashes and inbound consumers before removing any further path. The current cleanup is not permission to alter another worktree or shared dataset.
