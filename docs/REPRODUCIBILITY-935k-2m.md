# Submission fields and reproducibility: 935k and 2M VCMs

## Current portable workflow: selected 2M submission

The user selected the 2,066,831-parameter model for the final submission. The root entry points below implement the new portable workflow; the later historical audit sections preserve the earlier research-checkout commands and their original limitations. See [final integration evidence](EXPERIMENT-final-submission-v1.md) for the measured new-run status and [download checksums](../submission/release/downloads.json) for released bundles.

```sh
python reproduce.py
# Unattended original-seed run; always creates a new workspace:
python reproduce.py --seed 53 --yes --no-benchmark
python benchmark.py
python inference.py
```

Initial setup uses Python's standard library to install uv when absent, then a locked Python 3.13 environment. Training prefers CUDA when `torch.cuda.is_available()` and automatically falls back to CPU otherwise; CPU runs may be much slower and do not promise identical Linux/CUDA weights. Training device is recorded in protocols, checkpoints and selection receipts. CPU inference/benchmarking is independent and does not train or download fitting data. The original selected lineage uses seed 53, 24 human-foundation epochs and 32 mixed-control epochs, with development-only selections at human 20 and mixed 32. The separate losing completion arm is retained as historical evidence and is not rerun. Use `--model-choice compact` for the 935k comparison; a changed seed is a new experiment, not a claim to recover the original weights.

Windows CPU fallback has a real optimizer/checkpoint smoke test, but full Windows reproduction still stops at 26 frozen feature hashes; see [Windows compatibility findings](EXPERIMENT-windows-reproduction-benchmark-20261019.md). macOS is untested. No MPS backend is selected: a Mac without CUDA uses the CPU path, subject to the same dependency and data-integrity checks.

`reproduce.py` downloads exact pinned public upstream shards and the separately pinned synthetic development clips, verifies recording/PCM/frontend identities and declared split separation, reconstructs normalization, fits from random weights, checks INT8 development fidelity, calibrates on human calibration data and records selection before opening test. Results live under the printed new workspace's `results/`: `comparison.json`, selection receipts, `model/`, and SVG/PNG `figures/`. Raw histories and checkpoints remain within that workspace. `--workspace NEW_DIRECTORY` chooses a destination that must not exist. `--source-cache EXISTING_CHECKOUT` optionally reuses hash-verified upstream files read-only; it does not reuse weights or extracted features. `--verify` checks supplied assets without training and is not a reproduction test.

Numerical frontend settings preserve the original 64-thread BLAS cached fitting/development extraction and use one BLAS thread for fresh calibration/test/holdout extraction. These scopes are explicit because rounding differences of roughly `1e-6` can affect INT8 confidence near a fixed threshold. The training group pins `threadpoolctl`; standalone inference workers fix numerical-library thread environment variables before importing NumPy. Feature hashes still guard the fitting inputs. This changes no labels, weights or acceptance thresholds.

After training, an interactive prompt offers the PC benchmark against that newly generated graph. Successful replay refreshes its slide card with PC latency, RTF and memory, retaining Pi/live/wake fields as unmeasured. The independent benchmark menu also supports SSH Raspberry Pi replay without reproduction; its hardware identity and graph hash control whether [the report generator](../scripts/submission/report.py) may fill Pi fields. Keep typed human/synthetic denominators and raw/demo/qualified policies separate. Recorded replay does not establish live wake accuracy.

The inference-only ZIP already contains the built UI, production JavaScript packages, selected ONNX/wake assets and prerecorded acknowledgment WAVs. It downloads no fitting corpus or synthesis model. GUI Home Assistant/Spotify settings use a private root `.env` as described in [SpeakerUI setup](../SpeakerUI/README.md); only generic “play music” is in the Spotify voice scope. CLI/WAV replay and benchmarks never operate devices.

## Preserved original audit

Prepared 2026-10-17, Asia/Manila. This guide covers the **two exact model bundles supplied for `airimonda/vcm-benchmark`**, not the older semifinal, the compact epoch-32 predecessor, or the broader UI/ncnn package. Dataset membership is authoritative in [DATASETS.md](DATASETS.md); references and publisher terms are in [DATASET-CITATIONS.md](DATASET-CITATIONS.md) and [dataset-citations.bib](dataset-citations.bib). The [machine audit](model-dataset-usage.json) binds the observed manifests, checkpoints, archives and experiment evidence. No publication, training, installation or device operation accompanies this documentation update.

## Values for the model/dataset/training slide

Replace the template's causal/KV-cache encoder with **complete-utterance contextual encoder after VAD endpoint**. The frontend can accumulate frames, but the selected encoder does not expose streaming attention caches. The actual recognition path is:

```text
16 kHz mono PCM -> wake/VAD capture -> 40 x T log-mel
 -> 6 contextual blocks -> finite acoustic + intent/target heads
 -> finite catalog scorer + typed validation -> command JSON / rejection
```

The acoustic head scores finite catalog pronunciations; no general transcript is emitted. The drawn "slot head" should be labeled **finite slot/catalog decoding**. The drawn "actuator" should be labeled **validated command output** for these diagnostic packages. Their actions are disabled. Wake/VAD assets are pretrained components and must be counted separately from VCM parameters.

| Slide field | 935k submission | 2M submission | Evidence |
| --- | --- | --- | --- |
| Model identity | `speaker-pipeline-repair-v1`, `saved-epoch28` | `speaker-final-refinement-v1`, control epoch 32 | Selected-checkpoint receipts below |
| Encoder | `PooledContextVCM`, 6 blocks, width 128, 4 attention heads, feedforward 256, dropout .1 | Same class; width 192, feedforward 384 | Frozen protocols below |
| Input | 16 kHz mono; **40 × T** log-mel; 25 ms window, 10 ms hop, 512-point FFT, no centered padding; output grid 20 ms | Same | [Frontend](../vcm/frontend.py), selected export receipts |
| Outputs | 40 acoustic outputs (39 phones + blank), 19 intent outputs (18 commands + UNKNOWN), 2 target outputs; finite typed decoding | Same | Model definitions and experiment reports |
| Parameters | **935,567 (0.935567 M)** | **2,066,831 (2.066831 M)** | Protocols and archive bindings |
| ONNX graph size | **3,864,875 bytes = 3.864875 MB = 3.685832 MiB; FP32** | **3,173,089 bytes = 3.173089 MB = 3.026093 MiB; INT8** | Graph bytes freshly read and hashed inside supplied ZIPs |
| Dataset location | Pinned AI231 primary + pinned Filipino supplement | Same upstream releases | Source table below |
| Fitting hours / utterances | **14.033144 h / 23,289**, comprising 1,863 human + 21,426 synthetic | **14.038888 h / 23,295**, comprising 1,869 human + 21,426 synthetic | Actual manifests; stored sample durations, not a new waveform decode |
| Fitting speakers / voices | **173 declared human IDs + 70 synthetic voice IDs** | **179 declared human IDs + 70 synthetic voice IDs** | Source-qualified IDs; independent human/voice aliases unverified |
| Shared dev / calibration | 1,251 dev; 210 human calibration; 1,197 synthetic calibration/diagnostic | Same | Preserve human/synthetic separation |
| Published test / holdout | 4,418 / 196; project typed-scoring eligible 4,233 / 194 | Same | 185 / 2 uncertain references retained separately |
| Labels | **18 command intents + UNKNOWN**; 93 official baseline wordings; 415 project phrase definitions | Same | Finite schema/catalog; definitions do not prove acoustic coverage |
| Slot display | Finite typed slots; 13 distinct keys appear across 1,079 catalog entries | Same | Keys listed below; these are not 13 neural slot heads |
| Compute used | **1 shared NVIDIA A100-SXM4-40GB** | **1 shared NVIDIA A100-SXM4-40GB** | GPU receipts; do not report the host's eight GPUs as eight used GPUs |
| Fitting node | Not stored in the selected fitting receipt | Not stored in the selected fitting receipt | Separate synthesis receipt names `ai-n002.hpc.coe.upd.edu.ph`; this does not by itself prove the fitting node |
| Fitting/development wall-clock | **1,442.9723 s / 24.0495 min**, excluding input loading; repair adds no training | **1,990.4675 s / 33.1745 min**, excluding input loading, for foundation + both mixed comparison arms | Training selection receipts |
| Seed / framework | **53; PyTorch 2.10.0+cu128; CUDA 12.8** | Same | Training receipts; CUDA CTC is not bitwise deterministic |
| Objective | **CTC + .30 intent CE + .15 target CE** | Same | Frozen recipe; finite-command learning, no pretrained VCM encoder |
| Optimizer | **AdamW; weight decay .001; gradient clip 5; cosine LR** | Same | Human LR .0008, mixed .00025, floor .00002 |
| Batch / updates | Batch 32, microbatch 8; 96 optimizer updates/epoch; completed 24 human + 32 mixed epochs = **5,376 updates** | Same sizes; completed 24 human + 32 control + 32 completion epochs = **8,448 updates** across study | Selected-weight paths and completed-study totals differ |
| Selected-weight path | Human epoch **8** -> mixed epoch **28**: 3,456 optimizer updates along this lineage | Human epoch **20** -> control mixed epoch **32**: 4,992 updates along this lineage | Later discarded epochs are still part of run cost |
| Selected mixed-epoch training losses | CTC **.6652324**, intent CE **.08536265**, target CE **.006286416** | CTC **.2313055**, intent CE **.02765864**, target CE **.0008108017** | Saved histories; component means, not a held-out score or unweighted total |
| Augmentation | Training-only conservative time/frequency masks; no noise, reverb, speed, splicing or time warp | Same in both original 2M comparison arms | Frozen protocols; isolated masking benefit was not tested |

The catalog keys are `artist_id`, `color`, `contact_id`, `degrees`, `duration_seconds`, `light_id`, `operation`, `percent`, `song_id`, `task_id`, `time`, `timezone`, `unit`. Some are optional/intent-specific. Prefer “finite typed slots” on the visible slide; retain this list in notes. Normalization uses 1,869 original-human training utterances for both models. The six compact exclusions must not be silently used to refit the parent normalization.

Keep **all physical Raspberry Pi validation cells marked “not measured”**: keyword/wake accuracy, intent accuracy, false accepts, latency p95, real-time factor, live memory and thermal behavior. State the package's intended runtime as ONNX Runtime CPU, one thread, ARM64 CPython 3.12. Offline x86 measurements, parser fixtures and packaged wheel versions are different evidence and must not fill those Pi cells. No independent wake-owner/overlapping-speaker test has been established.

## Dataset location, exact versions and slide attribution

| Resource | Version to preserve | Citation / access field |
| --- | --- | --- |
| [AI231 ME2 Voice Commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) | **`25111444af3adff7588d86ab27c304bb080895cf`**; Hub `default` config, called `primary` in local manifests | airimonda (2026); upstream dataset DOI **[10.57967/hf/10723](https://doi.org/10.57967/hf/10723)**; source-dependent research/education and consent terms |
| [Filipino synthetic supplement](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data) | **`230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`**; `default`, `gap_fill`, `numeral_wordings` | martinnavs (2026); no verified dataset DOI; publisher prohibits redistribution while reference rights are unresolved |
| Local Qwen completion | [Protocol](../experiments/speaker-synthetic-completion-v1/protocol.json), [generated-row manifest](../experiments/speaker-synthetic-completion-v1/generated.jsonl) | **0 new commands in either winning fit**; 184 shared dev + 184 synthetic calibration; preserve generating-model/anchor identity |
| CMUdict | **`74790861f652b15e4ac49015a90074ad62a27690`** | Text pronunciations; preserve BSD-style notices; no verified DOI |
| Prepared derivative repository | `justinruaya123/ai231-smart-speaker`; older observed target commit **`3ed018c1bc731124a1b2ac5bd6653b5cd01bf576`** | **Not an exact published reproduction of these two fits. No derivative DOI or new release commit established.** Its extra configs and counts differ |

Visible slide text can be:

> Data: AI231 ME2 Voice Commands, DOI 10.57967/hf/10723 (revision 25111444af3a…), plus Filipino synthetic supplement (revision 230c8b8d58ca…). Per-source research/access terms; derivative release preparation in progress.

Put full revisions and original-source citations in slide notes. The [dataset table](DATASETS.md#submitted-935k-and-2m-models-verified-input-membership) identifies exactly which sources enter each fit. The main original references are FSC — paper DOI `10.21437/Interspeech.2019-2396`; SLURP — paper DOI `10.18653/v1/2020.emnlp-main.588`; SNIPS — official 2019 conference paper, DOI still primary-unverified; Common Voice — LREC 2020 paper, separate from CV19 release identity; Timers and Such — dataset DOI `10.5281/zenodo.4623772`; Multi-Sensor — dataset DOI `10.48804/IEKKVZ`. [The citation ledger](DATASET-CITATIONS.md) supplies titles, authors, venues and direct publisher links. These upstream paper/dataset identifiers do not identify the prepared derivative.

## Values for “To Be Submitted” and the reviewer checklist

| Template item | Evidence now | Status to put on the slide |
| --- | --- | --- |
| GitHub repository — public, MIT | Local origin is `https://github.com/justinruaya123/smart-speaker`; anonymous browser lookup returned 404 in this audit. Inspected checkout has no root project LICENSE. HEAD is `5d872e3a249e5876aaa8130b76f6e3b31622355f` with uncommitted model/research source | **Public access and code-license declaration pending**; 404 alone does not establish the reason |
| Dataset location — DOI + access | Primary DOI and two pinned upstream Hub inputs above; source matrix and card draft prepared | **Upstream citable; derivative DOI, per-source release decisions and portable export pending** |
| A100 cluster — node, GPUs, wall-clock, seeds | One shared A100; seed 53; actual fitting times above; fitting node missing from selection receipt | **GPU/time/seed recorded; fitting-node evidence incomplete** |
| Model weights — release URL + license | Both selected checkpoints, ONNX graphs and ZIP hashes verified locally below; no verified public release URL or trained-weight license | **Local artifacts verified; public weight release/license pending** |
| 1. Public repo, one-command reproduction | Local study scripts require hash-bound cached/prepared inputs; uncommitted dependency closure and portable reconstruction remain | **Incomplete for a clean clone** |
| 2. Dataset licensed and citable (DOI) | Primary/source citations verified; SLURP/FSC/SNIPS/supplement/student/reference conditions documented | **Partially documented; recording inclusion unresolved** |
| 3. Training logs + final checkpoint committed | Local history/log/checkpoint paths and hashes below; data/weights live in ignored caches, not a verified immutable public release | **Retained locally; publication pending** |
| 4. Pi4 latency reproduced by posted script | Each selected ZIP contains `pi_test.py`; upstream benchmark runs externally over Pi logs | **Script packaged; physical result missing** |
| 5. Held-out test, unseen speakers | Published membership retained; fresh stored-ID/PCM comparison has zero overlap, with aliases/near-duplicates unverified and prior test exposure disclosed | **Declared-ID separation verified; independent unseen-speaker validation incomplete** |
| 6. Comparable-size baseline | Both submitted candidates and prior experiments are preserved; this 935k/2M comparison changes width, six labels, export precision and demo policy | **Size comparison available; a matched-budget/data/precision baseline is not established here** |

The assignment template's “MIT” code field must not be copied onto model weights, third-party wake models or recording data. The asset receipt records openWakeWord pretrained files as CC BY-NC-SA 4.0 and Silero VAD as MIT. Keep complete notices and investigate the existing SLURP smoke-sample notice mismatch before distributing the diagnostic ZIPs further. No new license grant is made by this audit.

## Exact selected artifacts and evidence paths

| Artifact | 935k | 2M |
| --- | --- | --- |
| Selected checkpoint | `.cache/speaker-compact-recall-v1/training/mixed-control/epoch-028.pt` | `.cache/speaker-final-refinement-v1/training/mixed-control/epoch-032.pt` |
| Checkpoint SHA-256 | `a727e5070053045d6f276a3f58084ef16dd2bcbf9f931eea71db42a512f7d76a` | `ff6a3af63115a8f92a0106b20a17ee57c3e84f263177b89aaedb9cce326fce9b` |
| Packaged ONNX SHA-256 | `fc87c98c242ccb6f37ed9fcd84bad3869b8c90eaae6267fdab2a1a5af2737f6d` | `3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e` |
| Local diagnostic archive | [speaker-pi4-pipeline-repair.zip](../dist/speaker-pipeline-repair-v1/speaker-pi4-pipeline-repair.zip) | [speaker-pi4-final.zip](../dist/speaker-final-refinement-v1/speaker-pi4-final.zip) |
| Archive SHA-256 | `6923a1636a4b506c2b75b0c5308a752b8f714dce5f0ce5a732498c282dbfe4dc` | `f21292606581ea793b099b5f6a4fbe5a3d3ea221775595bdeb6794f38a21e93a` |
| Protocol | [Pipeline repair](../experiments/speaker-pipeline-repair-v1/protocol.json), [original compact fit](../experiments/speaker-compact-recall-v1/protocol.json) | [Final refinement](../experiments/speaker-final-refinement-v1/protocol.json) |
| Selected checkpoint receipt | [Saved epoch 28](../experiments/speaker-pipeline-repair-v1/saved-epoch28/selection-before-calibration.json) | [Final selection](../experiments/speaker-final-refinement-v1/selection-before-calibration.json) |
| Pre-test lock | [Compact repair](../experiments/speaker-pipeline-repair-v1/selection-before-test.json) | [2M](../experiments/speaker-final-refinement-v1/selection-before-test.json) |
| Metrics | [Compact repair](../experiments/speaker-pipeline-repair-v1/results.json) | [2M](../experiments/speaker-final-refinement-v1/results.json) |
| Training histories to preserve | `.cache/speaker-compact-recall-v1/training/{human-shared,mixed-control}/history.json` | `.cache/speaker-final-refinement-v1/training/{human-shared,mixed-control,mixed-completion}/history.json` |
| Raw fitting log | `.cache/speaker-compact-recall-v1/training.log` | `.cache/speaker-final-refinement-v1/training.log` |

The machine audit freshly verified checkpoints, graph bytes and archives; its history-file hashes identify the retained histories. A successful local integrity check is not a public hosting or licensing result. Preserve optimizer/training/environment evidence and exact selected checkpoints in an immutable reviewed release; an ONNX-only release supports inference, not full recovery of training state. Seed 53 is insufficient for bitwise regeneration of GPU fitting or synthesized waveforms.

For numerical result context, the **project typed test** reports compact raw complete-command correctness **489/636**, relaxed-demo correct accepts **238/636**, and UNKNOWN accepts **1/229**. The 2M fixed-demo view reports raw **516/636**, correct accepts **274/636**, UNKNOWN accepts **3/229**. These are existing recorded-command regression results, not Pi keyword accuracy or independent blind evaluation. Strict qualification remains disabled in both. [Official benchmark results](../experiments/speaker-official-vcm-benchmark-v1/results.json) use a separate all-196-reference holdout denominator and preserve raw/demo/fixed/qualified streams; never tune from those results or pool them with the project test figures.

## Reproduction commands and their actual scope

Run from the `smart-speaker` repository root. This command verifies the locally available evidence and selected weight/package identities using only Python's standard library. It fails on missing files or changed hashes. It is an **integrity check of this research checkout**, not clean-clone training or a new acoustic benchmark:

```bash
python - <<'PY'
import hashlib, json, pathlib, zipfile
r = pathlib.Path('.')
a = json.loads((r / 'docs/model-dataset-usage.json').read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
for p, expected in a['evidence_files'].items():
    if sha(r / p) != expected:
        raise SystemExit('Changed evidence: ' + p)
for label, m in a['models'].items():
    for rec in [m['checkpoint'], m['package'], *m['histories']]:
        if sha(r / rec['path']) != rec['sha256']:
            raise SystemExit('Changed artifact: ' + rec['path'])
    with zipfile.ZipFile(r / m['package']['path']) as z:
        graph = z.read('models/default/model.onnx')
    if hashlib.sha256(graph).hexdigest() != m['export']['sha256']:
        raise SystemExit('Changed ONNX: ' + label)
print('Verified evidence, selected checkpoints, histories, packages and ONNX hashes')
PY
```

In each separately extracted Pi package, the supplied entry points are:

```bash
python3.12 pi_test.py --verify-only
python3.12 pi_test.py --setup --engine onnxruntime --mode demo --threads 1 --output results/demo-01
.venv/bin/python pi_test.py --mode qualified --threads 1 --output results/qualified-01
```

Use new output directories. The packaged setup targets Linux ARM64 CPython 3.12 and includes offline wheels; its presence has not established a physical Pi result. `--mode demo` uses the bundle's separately documented application policy; `qualified` preserves failed qualification. For live capture, follow the package handoff and choose an actual listed microphone. The [external class benchmark](https://github.com/airimonda/vcm-benchmark#what-your-pi-must-print) additionally needs a running assistant that appends decision lines with `intent`, required `slot`, `infer_ms` and `audio_ms` under `~/vcm_benchmark/`; a model file alone is insufficient. The official recorded replay does not supply wake audio or measure a physical microphone path.

To reconstruct the fitted datasets, download the primary **at its historical commit**, plus all three pinned supplement configurations, under their recorded access terms. Apply the existing installer, pronunciation/annotation preparation, exclusion rules and explicit per-model memberships. The shared source-qualified hash partition and speaker exceptions are documented in [DATASETS.md](DATASETS.md). Copy raw/derived labels and evaluation eligibility exactly, retaining 26 pronunciation exclusions, the compact-only six conflicts, 529 failed supplemental QA rows and all ambiguous evaluation references. Exact local Qwen dev/calibration bytes and anchors also have to be recoverable where permitted. Do not substitute newer Hub `main`, other voices or regenerated waveforms while calling it the same experiment.

The existing fitting entry points are `scripts/train/speaker_final_train.py`, `scripts/train/speaker_compact.py` and `scripts/train/speaker_pipeline_repair.py`; they bind existing protocol/cache paths and protect immutable prior output. They are provenance, not a tested portable single-command interface. The local root environment has changed between historical studies; the runtime lock alone does not describe all A100 training dependencies. Preserve the run's Python/framework/CUDA/dependency receipt and complete dirty-source dependency closure before advertising fresh-clone training.

## Remaining publication work

1. Preserve a reviewed immutable code revision/source archive including the currently uncommitted trainer/runtime dependency closure; record the code-license scope and exact tested environment.
2. Export reversible portable row maps for **both** selected fitting sets and shared views, including source repository/revision/config/shard/row and original/derived hashes. Preserve existing frozen manifests byte-for-byte. `model-dataset-usage.json` is the aggregate audit, not this row-level export.
3. Resolve per-source recording inclusion, original clip/version ancestry, student/reference consent and supplementary distribution restrictions. Where audio cannot be redistributed, provide a pinned upstream reconstruction route and accurately state access limits. A public metadata release must also omit private participant details.
4. Stage an allowlisted metadata/citation release and any cleared audio separately from training caches, credentials, historical experiments and weight bundles. Preserve existing target configurations. Validate every proposed config from a clean cache; the current target's default does not reproduce these fits.
5. Retain selected weights, histories, source/environment receipts and checksums in an immutable model release with a verified download URL and explicit weight terms. No public URL is manufactured from a local `dist/` path.
6. After the reviewed release exists, obtain the derivative's **own** version-specific DOI through the [official Hugging Face DOI workflow](https://huggingface.co/docs/hub/doi), or an actual Zenodo deposition. Cite upstream sources in related metadata. Re-download the exact release and run a clean-clone reconstruction before marking one-command reproduction complete.
7. Collect actual Pi4 recorded/live measurements and independent unseen-human/alias/near-duplicate/listening evidence. Keep failed quality gates and missing calibration classes visible in the slide and Hub card.

These are measured release gaps, not claims that completed local experiments must be rerun during this documentation task. Nothing has been uploaded, committed, renamed or relicensed by this preparation.
