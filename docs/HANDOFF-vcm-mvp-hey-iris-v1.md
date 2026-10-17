**AI231 MEX2 finite command and Hey Iris continuation handoff**

Prepared 17 October 2026, Asia/Manila. The latest instruction is **handoff only**: stop implementation and GPU work here; continue full training and evaluation on the next host. The process started for this experiment was interrupted and its exit confirmed. Preserve the partial run. No model was installed, no device action was enabled, no code was pushed, and no new commit was made.

**Objective and current decisions.** Keep the unchanged assignment benchmark, all ten use-case categories, the existing typed command meanings, and both model-size tracks. Prefer static learned slots and complete-command outputs. Dynamic catalogs and phonetic matching are optional historical research, not MVP requirements. Use our existing encoders and code; do not copy others' implementations or mention their project names in the report or slides. Keep implementation small. Improve acceptance through development-only calibration and better supervision, while measuring wrong accepts and unknown-speech false accepts. Never replace a missing or unfamiliar explicit value with a plausible default.

The agreed practical baseline has 19 benchmark intents, 93 exact wordings and **31 distinct intent-plus-slot outcomes**, plus rejection. The six slotted families have these baseline values:

| Family | Fixed baseline values |
| --- | --- |
| TIMER | 10 seconds, 30 seconds, 1 minute |
| ALARM | 6 AM, 8 AM, 9 PM |
| TEMPERATURE | 18, 22, 26 degrees |
| BRIGHTNESS | 20, 60, 100 percent |
| COLOR | Red, Blue, Green |
| CREATE_REMINDER | Drink water, Study, Exercise |

The existing internal schema merges volume up/down into VOLUME operations, so its 18 command intents plus UNKNOWN correctly map to the benchmark's 19 intents. Preserve the larger existing vocabulary as an additive superset. Grammar support, acoustic training coverage and accepted-command accuracy must remain separate claims.

**Use the actual local source.** The original working directory is `/mnt/jfs_hpc/home/justin.jose.ruaya/ME2/smart-speaker`. The isolated working directory is `.worktrees/vcm-mvp-hey-iris-v1`, branch `codex/vcm-mvp-hey-iris-v1`, starting at `5d872e3a249e5876aaa8130b76f6e3b31622355f`. It incorporates the original dirty patch and 133 untracked files from the latest local work. A checkout of that commit alone will omit those updates. The preflight receipt records the dirty patch SHA-256 `bd23a50f1036f58611cce25b3eee586f0e3909a5f9a0632d308ad2201698e175` and copied-file identities.

Read `AGENTS.md`, `REQUIREMENTS.md`, `EXPERIMENTS.md`, the supplied execution handoff, and these current evidence files before restarting:

- `docs/REPRODUCIBILITY-935k-2m.md`
- `docs/EXPERIMENT-speaker-pipeline-repair-v1.md`
- `docs/EXPERIMENT-speaker-final-refinement-v1.md`
- `docs/PLAN-vcm-mvp-hey-iris-v1.md` in the isolated directory
- `experiments/vcm-mvp-hey-iris-v1/{preflight,protocol,catalog,data-audit,handoff-status,prior-candidate-replay}.json` in the isolated directory

Earlier intent/slot work exists in other worktrees, including uncommitted work. Leave it intact. Its old zero-acceptance findings do not replace the newer results below. There are no staged changes from this task. The isolated directory's `.cache`, `.venv` and `data/audio` are absolute symlinks into the original working directory; they are shared inputs, not portable payloads. They currently appear as untracked symlinks. **Do not stage them, use broad staging, recreate environments per experiment, or overwrite another run.**

**Verified original candidates.** Fresh inference was executed on every one of the fixed 196 recorded holdout clips for each exported model, using its matching frontend, catalog and decoder. These are previously examined regression recordings, not a fresh blind sample, a live wake test or a Pi measurement.

| Measurement | Compact existing model | Full existing model |
| --- | --- | --- |
| Identity | saved epoch 28, pipeline repair | control epoch 32, final refinement |
| Learned parameters | 935,567 | 2,066,831 |
| Graph | FP32, 3,864,875 bytes | INT8, 3,173,089 bytes |
| Prior project human test raw exact commands | 489/636 | 516/636 |
| Fresh recorded human holdout raw exact positives | 80/86 | 81/86 |
| Fresh human correct accepts, identical .95 intent/.90 slot rule | 38/86 | 62/86 |
| Unknown accepts under that identical rule | 0/10 | 0/10 |
| Fresh official complete-command accuracy under that rule, all 196 rows | 143/196 | 171/196 |
| Strict qualification | disabled | disabled |

The compact model's own relaxed demo policy instead accepted 42/86 human positives correctly and 1/10 unknowns. Its corresponding all-row official score was 146/196. Those policies differ; do not describe that comparison as a controlled capacity effect. Raw candidates from **both** models accepted 9/10 unknowns: removing abstention entirely is unsuitable. The full model is the stronger existing accuracy baseline; retain compact as the preferred small-model track. Neither is a qualified household recognizer.

Original selected checkpoint and graph identities:

- Compact checkpoint: `.cache/speaker-compact-recall-v1/training/mixed-control/epoch-028.pt`; SHA-256 `a727e5070053045d6f276a3f58084ef16dd2bcbf9f931eea71db42a512f7d76a`.
- Compact ONNX: SHA-256 `fc87c98c242ccb6f37ed9fcd84bad3869b8c90eaae6267fdab2a1a5af2737f6d`; preserved in `dist/speaker-pipeline-repair-v1/speaker-pi4-pipeline-repair.zip`.
- Full checkpoint: `.cache/speaker-final-refinement-v1/training/mixed-control/epoch-032.pt`; SHA-256 `ff6a3af63115a8f92a0106b20a17ee57c3e84f263177b89aaedb9cce326fce9b`.
- Full ONNX: SHA-256 `3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e`; preserved in `dist/speaker-final-refinement-v1/speaker-pi4-final.zip`.

**What was implemented before the stop request.** New files are isolated under the experiment worktree:

| File | Actual state |
| --- | --- |
| `scripts/models/mvp_static.py` | Existing six-block contextual encoder with masked mean pooling and one atomic command classifier. No phone head or phonetic runtime scorer. Supports width 128 and 192. |
| `scripts/train/mvp_static_train.py` | Executable `prepare` and `train` stages; full typed human labels do not require phone or role-span annotations. Continues our own trained encoder with a random classifier. |
| `scripts/runtime/mvp_static.py` | Prototype finite catalog, probability/margin abstention and ORT/ncnn adapter scaffolding. Export, runtime construction, engine parity and resident integration have **not** been executed. |
| `scripts/test/test_mvp_static.py` | Six contract tests cover atomic command identity, malformed slots, UNKNOWN, unsupported classes, policy split exclusion, padding and model budgets. |

The existing 1,079 catalog entries deduplicate to 310 typed outcomes including UNKNOWN. The additive user-requested entries bring the prototype inventory to 328. Its actual learned parameter counts are **969,928** for compact and **2,118,280** for full. These are new direct-classifier counts, not the old exact 935,567/2,066,831 counts. Compact remains below the 1.2M ceiling.

The additional names are the ceiling light, the seven specified TWICE tracks (Talk that Talk, Queen of Hearts, Basics, Trouble, Brave, Gone, When We Were Kids), and Enemy with Imagine Dragons. Existing kitchen/bedroom/bathroom lights, songs, contacts and aliases remain in the old inventory. New song IDs and `light.ceiling` were added provisionally; their parser, media/action mapping and alias precedence still need review. They have **zero training recordings**, and the prototype rejects classes absent from `trained_classes`. Their presence in JSON does not establish voice support. Home/House and outdoor-camera wording must follow the last validated local mapping; an outdoor-camera mapping was not established in this run. Do not invent one.

**Partial training, not a selected result.** A 24-example human wiring probe reached 100% raw training accuracy in three optimizer steps, 6.24 seconds. It was discarded before the real fit. The compact fit then completed seven epochs and was interrupted during the next epoch at the user's request. The last saved best checkpoint is:

```text
.cache/vcm-mvp-hey-iris-v1/compact/best.pt
SHA-256 f1fb788569618351d63db80c316ba035fbd8221578f4b94827d22fa2dda32ef4
epoch 7; development raw human positives correct 97/193
macro human-speaker raw recall 0.4569158774
```

The history's third selection value, 735, counts synthetic exact classes including UNKNOWN; do not present it as positive-only accepted-command recall. There is no final selection receipt, fitted acceptance policy, export, test/holdout evaluation or candidate qualification. The checkpoint lacks optimizer and RNG state, so it is not an exact interrupted-training resume. **Do not replace the original models with it.** Full-model prototype training never started.

Six new contract tests passed before training; 60 focused existing tests had also passed. The loader initially reproduced an einops lazy backend-initialization race during parallel feature extraction, before fitting. Initializing the unchanged frontend once on the main thread fixed the reproduced case; 100 concurrent frontend calls then passed. The failed preparation directory and receipt are preserved. Original code and installed artifacts were checked across 286 files and remained byte-identical.

**Frozen data and benchmark.** The fixed benchmark code is pinned at `ab39857cf73dd067d068fed388c321ab160dc191`, under `.cache/vcm-benchmark/` with that revision directory. Its actual pinned loader used **196 clips: 86 human positives, 10 human unknowns, 100 synthetic positives**. Keep that cohort and scorer unchanged; do not silently substitute an expanded 202-row cohort. Physical full runs additionally need the prescribed no-wake trials, actual microphone path and telemetry; none were run here.

Primary corpus revision: `25111444af3adff7588d86ab27c304bb080895cf`. Filipino supplement revision: `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`. Source cards, access restrictions and citations are retained under `data/sources/` and `docs/DATASET-CITATIONS.md`. The primary dataset DOI is [10.57967/hf/10723](https://doi.org/10.57967/hf/10723); it is not a DOI for a newly prepared derivative.

The direct-training audit contains 23,499 training rows: **1,889 human and 21,610 synthetic**; dev 1,251; calibration 1,407; test 4,418; holdout 196. All ten split pairs have zero stored ID, declared-speaker, PCM-hash and pair-ID intersections. This does not prove voice-alias, near-duplicate or independent listening clearance. Full evaluation views retain ambiguous references separately. Do not convert those to UNKNOWN or remove capture failures.

Train/dev feature inputs and original data are under `.cache/speaker-synthetic-completion-v1/prepared-v1/` and `data/audio/ai231-pool-v1/`. The prototype uses that parent training-only normalization, based on 1,869 human utterances. It now supervises additional valid human complete commands that formerly lacked usable phone labels, while excluding six unresolved slot-metadata conflicts. Synthetic recordings remain synthetic. The fitting sampler is nominally 50% human, balances commands/speakers, and adds explicit UNKNOWN draws; synthetic row count must not be mistaken for its batch share.

On another filesystem, fix relocation deliberately: cached parent manifests contain absolute feature paths, and the worktree has absolute symlinks. Preserve frozen receipts, then create a relocation map or separately versioned manifest view; never rewrite historical hash-bound evidence in place. A remote clone or source-only transfer will omit the datasets, feature cache, original checkpoints, partial run and dirty-source dependency closure.

**Next host execution.** Inspect available GPUs and processes first; select one permitted GPU without interrupting other users. The previous host was `ai-n002.hpc.coe.upd.edu.ph`; CUDA was hidden inside its command sandbox but available to approved host execution. One shared A100 was used with a 12% per-process memory cap. Slurm querying failed with `slurm_load_jobs error: Unable to contact slurm controller (connect failure)`; do not assume a scheduler allocation. The root environment had PyTorch 2.10.0+cu128, CUDA 12.8; ncnn, pnnx, python-pptx and reportlab were importable.

Use the shared root environment and isolated source, not a fresh stale checkout. Existing commands actually implemented are:

```bash
cd /mnt/jfs_hpc/home/justin.jose.ruaya/ME2/smart-speaker/.worktrees/vcm-mvp-hey-iris-v1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider scripts/test/test_mvp_static.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.train.mvp_static_train prepare
```

Before restarting compact, preserve the interrupted directory under a new immutable name. The trainer intentionally refuses to overwrite it:

```bash
.venv/bin/python - <<'PY'
from scripts.train.mvp_static_train import CACHE
source = CACHE / 'compact'
archive = CACHE / 'compact-interrupted-20261017'
if archive.exists():
    raise SystemExit('Archive already exists; inspect it before proceeding')
if source.exists():
    source.rename(archive)
PY
```

After verifying all recorded input hashes and resolving paths, run each full fit sequentially; replace GPU index 0 with the actual authorized device:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 .venv/bin/python -m scripts.train.mvp_static_train train --model compact --device cuda
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 .venv/bin/python -m scripts.train.mvp_static_train train --model full --device cuda
```

The current recipe is one seed (53), up to 16 epochs, 64 updates/epoch, batch 32, microbatch 8, AdamW LR .0003 with cosine decay, weight decay .001, gradient clip 5, label smoothing .02. Selection uses development raw human macro-speaker correctness, then raw human and synthetic counts; early stopping begins after epoch 8 with patience 4. These settings are an initial bounded trial, not an established optimum.

**Remaining implementation in priority order.** Complete and inspect the direct classifier comparison first. Review source/input hash verification on protocol reuse, missing-label handling, sampling, complete catalog mapping and the untested runtime adapter before treating the new commands as a production path. The current CLI has no evaluate/export/package stages; add and test them instead of invoking nonexistent commands. Register new modules in `scripts/modules.json` without losing inherited dirty edits. Freeze chosen weights, catalog, normalization and policy before the unchanged historical test and official scoring.

Calibrate the direct joint command score and margin on dev/calibration only. The current helper starts with zero observed fitting wrong/unknown accepts; it has not been run on the real candidate. Relaxing the old multi-slot criteria is authorized, but threshold selection must retain empirical negative constraints and report accepted recall, wrong accepted positives, false rejects and unknown false accepts. Do not report softmax as a guaranteed correctness probability. Missing human coverage must remain visible without silently forcing every otherwise supported class off. Human, synthetic, accent/source and per-intent/value metrics stay separate. Test errors cannot select weights or thresholds.

Only then add the missing fixed-name utterances as a separately versioned training augmentation. Prefer permitted original human commands; use diverse licensed whole-utterance synthetic voices where human coverage is unavailable. Cover accents and gender presentations with actual metadata and separate held-out voices; intended voice prompts are not verified demographic labels. Keep all waveform derivatives with their parent speaker/split, use hard negatives and near-confusable numbers/names, and never alter the fixed benchmark to fit the model.

Train an independent **Hey Iris** detector from random initialization. No such positive corpus, wake training, export or streaming integration was created in this session. Local Qwen synthesis assets and existing wholly synthetic voice-design anchors are available under `.cache/tts/models/` and `.cache/speaker-synthetic-completion-v1/`; use their pinned source receipts and permissions rather than cloning unapproved human voices. This model needs bounded rolling-window inference, hysteresis/cooldown and held-out streaming false-wake measurements. An exported untrained graph or synthetic-only recall is not real wake qualification.

Reuse the existing resident service and session policy. The prototype Runtime is not yet wired into `speaker_worker.py`; its initializer/event interface differs, and the service currently reapplies the old phonetic confidence rule. Add an explicit adapter so one direct-model decision remains authoritative. Keep safe simulated actions. Preserve ncnn as the final demo engine where required, and verify actual WAV decisions against ORT and training, including padding and threshold boundaries. No ncnn conversion/parity was run for this new model.

Speaker isolation remains separate from wake detection. Reuse one wake/one command/reset behavior; train and calibrate a genuine speaker-consistency backend only with suitable permitted speaker data. Existing command embeddings are not speaker identity. Benchmark-compatible mode must declare owner gating disabled for its entire run, because the fixed benchmark can combine one person's wake with another person's command. Owner-protected mode must declare gating enabled, reject inadequate or overlapping evidence, and have separate same-person/impostor testing. No ownership backend was trained here; do not advertise isolation as complete or promise overlapping-speaker separation.

**Citations and submission slides.** Document each data source, augmentation implementation, distribution and intended scope. The executed command trial uses training-only frequency masking (0–4 of 40 bins, probability .5) and time masking (0–min(10, 5% of valid frames), probability .5), with zero in normalized space. Cite [Park et al., SpecAugment, Interspeech 2019, DOI 10.21437/Interspeech.2019-2680](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html). It is an adaptation; no isolated augmentation benefit was measured. No speed, noise, gain or RIR augmentation was added to this trial. If speed perturbation is subsequently used, [Ko et al., Audio augmentation for speech recognition, Interspeech 2015](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html) is a checked primary reference; record actual settings and keep evaluation audio untouched. Add citations for any other augmentation before using it.

The user supplied a two-slide submission template. No new PPTX/PDF was created before the stop request. Fill it after the next host's measured results. Correct its causal/KV-cache diagram: the existing and prototype command encoders process a complete utterance after endpointing. Input is 16 kHz mono, 40×T log-mel, 25 ms window and 10 ms hop; six blocks, width 128 or 192. For the prototype show an atomic finite-command head and typed validator, not a phone-to-text or dynamic slot reader. Recompute dataset hours, actual speakers/voices, parameter counts, trained steps/loss, node/GPU/time/seed and artifact size from the selected completed run. The submission should name only our work and legitimate dataset/method citations. Public hosting/license/DOI fields remain pending until actually established. Keep every physical Pi wake/intent accuracy, FAR, p95, RTF, memory and thermal cell **not measured** until tested; desktop replay cannot fill those cells.

The next completion report must distinguish software checks, raw commands, accepted commands, wake accuracy, ownership accuracy and physical Pi evidence. Preserve failed trials and all old models. Build a candidate package separately; install only after its actual runtime decisions and declared profile have passed the applicable checks. Do not restart any training on this host as part of delivering this handoff.
