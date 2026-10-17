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

For inference, extract the inference ZIP and run `python inference.py`. Choose CLI diagnostics or SpeakerUI; the UI defaults are documented in the source README and `SpeakerUI/.env.example`. Its HA entity allowlist defaults to the three user-specified IDs. Finite acoustic room labels still require explicit bindings; no physical room assignment is inferred. Spotify supports generic “play music” resume without named song or playlist configuration. Runtime recognition is local, finite VCM inference without ASR or an LLM. Pretrained Hey Jarvis and Silero VAD are included, together with their notices. Fixed spoken replies are included as recordings rather than synthesized at runtime.

The benchmark ZIP runs independently of reproduction. Extract it on the controlling PC, then run:

```bash
python benchmark.py --target pc
python benchmark.py --target pi --host user@raspberrypi
```

The Pi command uses SSH, installs the locked runtime, transfers only its allowlisted runtime and held-out samples, and retrieves recorded-inference results. Use an actual reachable SSH host. PC and Pi receipts remain separate; replay timing does not measure live wake, microphone, or speaker performance. The user operates the physical Pi from their PC.

The exact archive URLs, byte sizes, hashes, and immutable Hugging Face dataset revision are in [`submission/release/downloads.json`](https://github.com/justinruaya123/smart-speaker/blob/codex/final-submission/submission/release/downloads.json). Anonymous downloads of the final archives are verified independently before release preparation. The release helper verifies the outer SHA-256, complete ZIP member inventory, every member hash, and selected model identity before uploading either archive. It verifies the draft's uploaded asset hashes and target again before publication.

Project-owned code uses the [MIT license](https://github.com/justinruaya123/smart-speaker/blob/codex/final-submission/LICENSE). Dataset, pretrained-model, pronunciation, and dependency terms retain their own scopes and notices; see [submission licenses](https://github.com/justinruaya123/smart-speaker/blob/codex/final-submission/docs/SUBMISSION-LICENSES.md). Training corpora, credentials, personal captures, and mutable research caches are excluded from the release archives. Repository visibility is preserved by this workflow.

The source publication uses immutable GitHub blobs and a new branch, preserving the development checkout, its index, existing branches, installed artifacts, and active experiment outputs. The source commit precedes the release-automation commit so the release can target the complete source tree without introducing the new automation workflow itself. Any actual GitHub permission failure is recorded in `submission/release/github-status.json`; a failed run is not reported as a created release.
