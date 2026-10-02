# Speech embeddings for custom music names

Isolated original-human-speech study. Text-only enrollment is the primary target; real-audio enrollment is an optional comparison. “Spotify names” means artist/song catalog labels in offline fixtures. This experiment does not connect to Spotify or search a real account.

Read the [measured report](../../STT/speechembedding/REPORT.md), [interpretation and failed quality gates](../../STT/speechembedding/EVIDENCE_NOTES.md), [BC-ResNet paper investigation](../../STT/speechembedding/BCRESNET.md), and [research rationale](../../STT/speechembedding/RESEARCH.md).

Run from the repository root using its existing `.venv`:

```powershell
.venv/Scripts/python.exe experiments/speechembedding/run.py prepare
.venv/Scripts/python.exe experiments/speechembedding/run.py train
.venv/Scripts/python.exe experiments/speechembedding/run.py check
.venv/Scripts/python.exe experiments/speechembedding/run.py evaluate
.venv/Scripts/python.exe experiments/speechembedding/run.py benchmark
.venv/Scripts/python.exe experiments/speechembedding/run.py report
```

`train` refuses to overwrite an existing candidate; `evaluate` refuses to overwrite completed test evidence. Create a new version for a changed training/evaluation protocol. `export` can recover ONNX graphs from completed fitting without retraining. `report` only regenerates Markdown from saved evidence.

Implementation lives in the canonical `scripts/models/speech_embedding.py`, `scripts/train/speech_embedding_{data,study}.py` and `scripts/test/speech_embedding_eval.py`; this folder is an entry point, not a duplicate implementation. All modules are research-only in `scripts/modules.json`. The frozen inputs are in `data/speechembedding-v1/`; generated artifacts are in `.cache/speechembedding-v1/`.

The model returns 96-dimensional audio and pronunciation embeddings. Its main contrastive loss pairs whole human utterances with pronunciation strings, and a weak localization loss pairs annotated parameter names with ten feature windows. `dual_kws` adds a fixed keyword-classification loss. No waveform is synthesized, spliced or augmented; no pretrained weights are loaded. Windows do not establish name timestamps.

The preferred text reader is selected on development data. An offline query must receive an already accepted intent:

```powershell
.venv/Scripts/python.exe experiments/speechembedding/run.py query --audio data/audio/my-original-recording.wav --intent MEDIA_PLAY --role artist --catalog expanded
```

Without an accepted music intent, the command refuses with zero acoustic calls. Even with that intent, failed quality gates abstain. Raw top-five candidates are research diagnostics, not accepted results.

Custom text catalogs may use the [schema example](catalog.example.json):

```powershell
.venv/Scripts/python.exe experiments/speechembedding/run.py query --audio data/audio/my-original-recording.wav --intent MEDIA_PLAY --role artist --catalog experiments/speechembedding/catalog.example.json
```

Use a stable ID and name per entry. Supply reviewed pronunciations in the existing CMU phone vocabulary for unfamiliar spellings; this does not add acoustic output classes. Catalog modifications invalidate calibration. The example catalog is not acoustically validated and therefore cannot return an accepted result without new development calibration under a versioned study. Multiple aliases/automatic G2P and automatic artist/title role parsing are not implemented.
