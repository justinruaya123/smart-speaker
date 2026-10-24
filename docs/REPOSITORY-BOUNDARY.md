# Repository boundary: smart-speaker and SpeakerUI

## One owner per executable

| Repository | Owns |
|---|---|
| smart-speaker | Training, calibration/model evaluation, exported-model verification, provenance, permitted evaluation-data preparation and model distribution |
| [SpeakerUI](https://github.com/justinruaya123/SpeakerUI) | Ambient display, gateway, local Python inference, wake/VAD/owner orchestration, configuration/integrations, runtime benchmarks, application packaging and device deployment |

There is no submodule, sibling-source lookup, shared virtual environment or synchronous application build in model training. Model forward passes used to calibrate and verify exports remain producer responsibilities; they are not a resident application or a hardware-benchmark CLI.

SpeakerUI's `models.lock.json` identifies exact producer Git revisions, files, sizes and SHA-256 values. Its installer verifies those trusted pins and refuses corrupt existing installations. The initial handoff uses existing immutable tracked assets, not an invented release/tag/archive URL. Application configuration, private state, recordings, environments and downloaded model assets stay out of application Git.

## Preserved model contract

The original default epoch44 model remains 258912 parameters, with 31 complete finite commands plus UNKNOWN, the same graph, normalization, ordered commands, policy and recipe bytes. Graph SHA-256:

```text
c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce
```

Wake/VAD/owner companions remain separately pinned. Qualification remains false; failed wake/owner/original-CPU reconstruction findings are not erased or promoted. No new full fit, acoustic release, provider execution, microphone capture or physical Pi measurement is implied by the repository split.

## Model operations

From the producer root:

```sh
python -B -m scripts.optionb_epoch44 verify
python -B reproduce.py
```

Both verify original artifacts only. They do not fit, bootstrap the application or choose legacy Iris implicitly. Original fitting and permitted evaluation-data preparation require the explicit pinned-source workflow in [REPRODUCIBILITY-optionb-epoch44.md](REPRODUCIBILITY-optionb-epoch44.md). Standalone inference and benchmarks must be run from SpeakerUI, following its [operations guide](https://github.com/justinruaya123/SpeakerUI/blob/main/docs/OPERATIONS.md).

The producer packaging API is:

```python
from pathlib import Path
import subprocess
from scripts.model_package import build_archive

revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
result = build_archive(
    Path('submission/models/optionb-tiny'),
    Path('.cache/model-distribution/new-run/epoch44.zip'),
    source_revision=revision,
)
print(result)
```

Use a new output path; archives are immutable. The frozen archive contains exactly the five model payloads under `model/` and `MANIFEST.json`. Any explicitly selected companion payloads/notices are separately declared and verified. No application source, worker, launcher, UI, private settings, recording corpus or training checkpoint belongs in this artifact. This local build is not a remote publication; do not run `--publish` merely to verify a migration.

The explicit legacy continuation workflow packages its verified exports through `scripts.model_package.package_release`, not the old complete Pi/application packager. Its `models/default/` layout retains selected-bundle comparison semantics while omitting the executable application. Runtime resource benchmarks and application software gates are consumer responsibilities.

## Source/provenance compatibility

The extraction baseline is producer commit `94c0021d7e22c9aa719246ac97105d7140a9f2a2`. SpeakerUI's `SOURCE-MIGRATION.json` records imported paths and reviewed local restoration origins separately from later integration changes. Existing dirty source worktrees/private configuration are protected backups, not publication targets.

Historical source locks, experimental protocols, populations, failed evidence, checkpoints and frozen model bytes are not rewritten to make a relocation appear original. Pure reference/calibration helpers live in `scripts/train/`; new engineering/source-equivalence tests bind the actual old and new implementations. Source inputs for new receipts are mapped by explicit ownership in `scripts/repository_boundary.py`, not filtered by whichever files happen to exist. Missing unknown model sources must still fail closed.

Exact old application-containing releases use their immutable original source. Existing original epoch44 source/environment reconstruction uses `3c05b3beb46f06c3108089e0d8c79474670c6cb7`, as before. The producer retains no deployment-runtime directory or compatibility forwarding modules. Archived prefit scripts remain immutable provenance inputs; active model trainers use the training-owned references. The verification environment now lives in `scripts/train/pyproject.toml` and `scripts/train/uv.lock`, with the original lockfile bytes unchanged. Official model scoring keeps only the original two serialization functions, not the old live benchmark logger or CLI.

## Verification and evidence limits

Use the owning repository's compatible test interpreter:

```sh
python -B -m pytest -p no:cacheprovider scripts/test/test_model_only_boundary.py scripts/test/test_model_training_bootstrap.py scripts/test/test_model_package.py scripts/test/test_model_reference_boundary.py scripts/test/test_model_evaluation_audio.py scripts/test/test_model_publication_boundary.py -q
python -B -m scripts.optionb_epoch44 verify
python -B scripts/iris_publication_data.py --check
python -B presentation/build_readme.py --check
python -B presentation/validate.py --require-integration
```

Static model presentations own their small pinned browser project in `presentation/`, not SpeakerUI's Node dependency tree:

```sh
pnpm --dir presentation install --frozen-lockfile
pnpm --dir presentation exec playwright install chromium
node presentation/qa.mjs
```

The separate acoustic release gate still intentionally fails for the existing failed wake receipt. Do not bypass it because software/site tests pass. Distinguish source-equivalence, frozen-byte checks, model-package inventory, original-audio replay, browser/native-worker integration, full fitting, live providers and physical hardware evidence. Compare legacy Windows test failures by exact node ID with the untouched baseline; a green focused migration suite is not an all-green historical scientific suite.

## Cutover and recovery

Publish and read back the independently tested application before deleting the embedded application. Remove active application entrypoints/build/package dependencies from the producer only after that consumer-first handoff. Preserve repository privacy, original scaffold/history, immutable model tags and existing remote releases. Push normally without force; read back each exact remote commit/tree.

Recovery uses ordinary revert commits and the prior pinned model installation. Do not reset/force-push shared branches or delete original worktrees/private settings.
