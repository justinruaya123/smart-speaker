# Selected 2M smart-speaker submission

This submission ships the existing **2,066,831-parameter** `speaker-final-refinement-v1` mixed-control epoch-32 model. Its selected INT8 ONNX SHA-256 is `3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e`. The compact model remains comparison research. This publication does not establish full qualification or physical microphone performance. GUI providers and actions default off and honor explicit private local configuration; CLI and recorded replay never execute actions.

Use the dedicated [`codex/final-submission` source branch](https://github.com/justinruaya123/smart-speaker/tree/codex/final-submission). The repository's `main` branch contains an independent Iris implementation with different root entry points. This submission does not merge or replace that work.

```bash
git clone --branch codex/final-submission --single-branch https://github.com/justinruaya123/smart-speaker.git
cd smart-speaker
python reproduce.py --verify
python reproduce.py --seed 53 --yes
```

The reproduction command installs its locked environment and writes to a new workspace. It fits the original recipe rather than using a shipped completed checkpoint as a new fit. A changed seed is a separately identified run; exact cross-machine CUDA equality is not guaranteed. Completion and benchmark evidence belong to each run receipt.

For inference, extract the [GitHub source-built ZIP](https://github.com/justinruaya123/smart-speaker/releases/latest/download/smart-speaker-inference-2m.zip) and run `python inference.py`. Choose CLI diagnostics or SpeakerUI; configure private credentials using `SpeakerUI/.env.example`. General lights default to the Hue lamp and calls open the dining camera. Named acoustic rooms still require explicit bindings. Spotify supports generic play/resume. The UI speaks actual tool results using local OS TTS; fixed rejection clips remain included. Debug > Inference selects experimental Hey Iris with Jarvis fallback, Iris only, or Jarvis. Recognition remains local and finite, without ASR or an LLM. Weather fetches external current conditions after recognition; device actions retain explicit configuration gates.

The [GitHub-built benchmark ZIP](https://github.com/justinruaya123/smart-speaker/releases/latest/download/smart-speaker-benchmark-2m.zip) runs independently of reproduction and includes all 196 original held-out recordings. GitHub builds/verifies it from the repository code, selected model and hash-bound audio fixture alongside inference; `python -m scripts.submission.build_benchmark` performs the same assembly locally. Extract it on the controlling PC, then run:

```bash
python benchmark.py --target pc
python benchmark.py --target pi --host user@raspberrypi
```

The Pi command uses SSH, installs the locked runtime, transfers only its allowlisted runtime and held-out samples, and retrieves recorded-inference results. Use an actual reachable SSH host. PC and Pi receipts remain separate; replay timing does not measure live wake, microphone, or speaker performance. The user operates the physical Pi from their PC.

The [GitHub Actions workflow](https://github.com/justinruaya123/smart-speaker/actions/workflows/final-submission-release.yml) builds the inference ZIP from the checked-out source and repository model/audio assets, runs the locked tests and optimized UI build, then verifies the package. It publishes the ZIP, SHA-256 checksum and per-member receipt as GitHub release assets tied to that source commit. It does not download a prebuilt ZIP, graph or fixed audio from Hugging Face. Historical Hugging Face archive receipts remain preserved as earlier distribution evidence. Dataset access for separate reproduction/benchmark work is unchanged.

To assemble the same inference profile locally after installing the locked inference dependencies, run `python -m scripts.submission.build_inference`. Node 24.13.0 and pnpm 12.5.1 are installed locally if needed. Missing source model/audio files fail the build; they are not silently fetched from another host.

Project-owned code uses the [MIT license](https://github.com/justinruaya123/smart-speaker/blob/codex/final-submission/LICENSE). Dataset, pretrained-model, pronunciation, and dependency terms retain their own scopes and notices; see [submission licenses](https://github.com/justinruaya123/smart-speaker/blob/codex/final-submission/docs/SUBMISSION-LICENSES.md). Training corpora, credentials, personal captures, and mutable research caches are excluded from the release archives. Repository visibility is preserved by this workflow.

Source publication updates the existing final-submission branch using immutable GitHub blobs and a checked branch-head lease, preserving the active checkout, its index, other branches, installed artifacts and experiment outputs. Release creation runs only after the source build and archive verification succeed. A failed workflow is not reported as a completed build or release.
