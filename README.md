# Smart speaker: VCM-only research branch

The independent character-transcription component has been removed from this
branch's active runtime, trainer and publisher. There is no transcription fallback
for uncertain parameters. Old accuracy reports remain historical evidence; they
are not a new evaluation of this modified pipeline.

## Two distinct checkpoints

| Path | Acoustic model | Current status |
|---|---|---|
| `Architecture.ipynb`, `train.py`, `publish.py` | Original v12 family encoder and finite parameter heads, 398,422 parameters | VCM weights unchanged; original local weights/audio are needed to rerun recognition tests. Unsupported custom requests clarify. |
| `VCM_Final_Review.ipynb`, `final_suite/`, `visualizer/` plus restored research payload | `phone_s53`, 113,064-parameter streaming causal CNN and constrained phoneme decoder | Research candidate for numbers and dynamic catalogs. Not an unrestricted sentence transcriber. Human/Pi/acceptance qualification still incomplete. |

Removing the independent model does not automatically substitute the research
checkpoint for v12 or establish either checkpoint's accuracy. Neither path
executes Home Assistant, Spotify or other external device actions.

## Workstation workflow

```bash
python train.py --tests-only --no-setup
python train.py --no-setup
python train.py --fit --no-setup
python publish.py
```

The root commands load Python implementation cells from `Architecture.ipynb`.
The default workflow preserves weights. An explicit `--fit` continues the finite
heads using train/development only; the original family encoder remains frozen.
`pipeline.toml` controls VAD, VCM, selected checks, training, endpointing and
publication. Obsolete transcription switches are rejected instead of silently
being ignored.

Local `artifacts/vcm.pt`, `artifacts/vcm.fp32.onnx`, `data/vcm.jsonl`, the shared
audio index and VCM-referenced WAV files are not fetched automatically. The
shared index may still contain recordings used only by previous experiments;
they are no longer required by the VCM-only data audit. Existing source/license
ledgers remain with the data. Labels are programmatic silver, not reviewed truth.

`tests/test_vcm_only.py` uses a small **random-weight software fixture**, including
an actual ONNX export and CPU session, to check removal independently of the
missing trained v12 checkpoint. Passing it is not recognition-quality evidence.

## Pi package

The publisher packages exactly one acoustic graph, `vcm.fp32.onnx`, with bundle
metadata, runtime-only code, microphone support and configured dependency wheels.
Old untracked transcription files in a workstation artifact directory cannot be
swept into a new package. They are not deleted from your workstation by this
source change. Previously built ZIPs must be rebuilt; they are not modified in
place.

On 64-bit Raspberry Pi OS, the published executable retains its VCM microphone
and WAV modes:

```bash
python3 pipeline.pyz setup --system-packages
python3 pipeline.pyz --test
python3 pipeline.pyz --list-mics
python3 pipeline.pyz --mic 2
python3 pipeline.pyz --wav recording.wav
```

`setup --system-packages` uses apt and needs network/sudo. With those OS
prerequisites present, setup uses the bundled Python wheels. Recognition is
local. No physical Pi or live microphone qualification is claimed here.

## Research, regressions and visualization

Read [FINAL_REVIEW.md](FINAL_REVIEW.md) and
[VCM_Final_Review.ipynb](VCM_Final_Review.ipynb) for the historical complete-command
ledger and limitations. Restore the matching research payload using
`final_suite/restore_payload.py` before running acoustic/browser tests. The
visualizer is a development tool, not part of the Pi inference payload.

The new [removal and Option B review](research/no_asr_review/RESULTS.md) separates
verified source changes, reference-dataset findings and newly measured tests.
The external reference is read-only and no third-party model weights or WAVs
are committed here. The active VCM's acoustic weights and acceptance thresholds
were not weakened by removal.

## Follow-up: no external pretrained initialization

Both active paths retain this project's own trained weights. External pretrained
models are permitted only in separate benchmark experiments; they must not
initialize the VCM or enter the publisher's single-graph allowlist.

[Follow-up verification and Option B re-audit](research/no_asr_followup/RESULTS.md)
records the new first-convolution-only experiment, strict paired-development
selection, ONNX Runtime replay, and the reference folder's move to `MEX2/Data`.
The frozen baseline remains selected; failed candidates never overwrite it.
