# Direct VCM acceptance refinement and Hey Iris continuation

17 October 2026. Status: **the command-only 30+/36 personal target passes for both widths; broader quality gates fail; isolated candidate packages only.** Wake is evaluated separately. No installed model, physical device, microphone, external publication, commit or push was changed/performed.

The compact model improves from 0/36 to **31/36** correctly accepted personal commands; full improves from 23/36 to **30/36**. Each makes one wrong positive-command acceptance; compact also accepts one of 52 negative clips. These are unchanged synthetic recordings from two held-out designed voices, not human participants. The model does not become qualified by passing this narrow target.

## Scope, baseline and decision

Continue the preserved `codex/vcm-mvp-hey-iris-v1` worktree based on `5d872e3a249e5876aaa8130b76f6e3b31622355f`, including its inherited dirty patch and untracked dependency files. A checkout of that commit alone is insufficient. The user's subsequent directions make command-only VCM the primary benchmark and permit wake to be decoupled. All ten UC definitions and ME-01 through ME-08 remain in scope; physical demonstration, owner protection and original-human personal-name coverage remain unsupported.

V1 replaces the old phonetic scorer with our own atomic finite-command head. Both existing encoder widths remain: six bidirectional contextual blocks, width 128/192, four attention heads, feed-forward width 256/384, local depthwise convolution, masked mean and one complete-command classifier. Input is complete-utterance 16 kHz mono -> 40×T log-mel, 25 ms window / 10 ms hop -> stride-two stem. There is no causal/KV cache, transcript, phonetic acoustic decoder, dynamic reader, LLM or fallback. A deterministic table retrieves the typed command associated with the acoustic class and validates it. A rejected class cannot be rescued by the application.

The catalog contains 328 outputs including UNKNOWN: 31 baseline typed outcomes from 93 required wordings, plus preserved extensions and 18 requested personal outcomes. V2 has training evidence for 117 positive outcomes; 210 outputs have no training evidence and cannot be accepted. Catalog membership does not prove acoustic coverage. Personal additions are eight requested songs with/without explicit artist, plus ceiling-light on/off. No outdoor mapping, arbitrary-name enrollment or action integration was invented. Compact is 969,928 parameters, below the 1.2M ceiling; full is a 2,118,280-parameter comparison.

## Data and reproducibility

The original 4,418-row project test and 196-row official benchmark remain unchanged. Official source is pinned to `ab39857cf73dd067d068fed388c321ab160dc191`; primary source revision is `25111444af3adff7588d86ab27c304bb080895cf`, supplemental revision `230c8b8d58caf78e4bc90886d9c6fa77fe6024b8`. Original source/license receipts remain under `data/sources/`; restricted supplemental audio is not redistributed in these packages.

V2 training has 23,739 recordings: 1,889 original human (1.542 h, 176 declared IDs), 21,850 synthetic (12.729 h, 76 declared voices). Development has 1,339 rows; calibration 1,495; test 4,418; typed holdout 196; separate personal test 88. Test has 636 eligible human positives, 229 human negatives, 185 ambiguous human references and 3,368 synthetic positives. Typed holdout has 83 eligible human positives, 11 negatives, two ambiguous references and 100 synthetic positives. Ambiguities and capture failures remain present; they are not silently relabeled or dropped. The official benchmark has its own original 186 positive / 10 UNKNOWN references and trim/normalize preprocessing, so it is not interchangeable with the typed holdout.

V2 adds only wholly generated recordings: 240 train, 88 dev, 88 calibration, 88 separate personal test. All 15 split pairs have zero stored ID, declared-speaker, exact PCM and pair-ID intersections. This is not proof against voice aliases or near duplicates. Human demographic labels and intended synthetic accent/gender prompts are not independently verified; synthetic voice IDs are never counted as human speakers. The inherited normalization uses 1,869 eligible original human training utterances. No test statistics fit preprocessing.

[Corpus counts/durations](../experiments/vcm-mvp-hey-iris-v2/corpus-summary.json), [split audit](../experiments/vcm-mvp-hey-iris-v2/data-audit.json), [V2 protocol](../experiments/vcm-mvp-hey-iris-v2/protocol.json), [V3 protocol](../experiments/vcm-mvp-hey-iris-v3/protocol.json).

## Completed training and controlled improvement

Command fitting uses seed 53, our own encoder continuation with a random atomic head, AdamW, batch 32 / microbatch 8, weight decay .001, cosine decay, clip 5 and CE label smoothing .02. Sampling is nominally 50% human / 50% synthetic with speaker/command balancing and .20 UNKNOWN draws. Training-only feature masks are 0–4 frequency bins and 0–min(10,5% frames) time frames, each probability .5, inspired by [SpecAugment](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html). They are not an isolated proven benefit in this comparison. No new waveform augmentation enters these command fits.

Initial 16×64-update V1 fits selected raw human dev 110/193 compact and 121/193 full. A separately frozen optimization-budget-only extension (48×96, min24, patience12) raises those to 130/193 and 144/193. Its selected checkpoints remain preserved. V2 restarts from those own static checkpoints, adds the declared generated recordings and .8 priority sampling for baseline/personal outcomes, uses LR .0001, up to 16×96 updates, min8/patience6. This is a combined practical revision, not an isolated causal estimate of synthetic augmentation. Both V2 small-slice probes learn 24/24 examples before full fitting; the probes are discarded and do not initialize the selected runs.

On `ai-n003.hpc.coe.upd.edu.ph`, A100-SXM4-40GB GPU 5, root Python 3.13.9 / Torch 2.10.0+cu128 / CUDA 12.8, V2 compact completes 16 epochs/1,536 updates in 129.30 s and selects epoch11; full stops at 11 epochs/1,056 updates in 90.70 s and selects epoch5. Selected mean CE losses are .34521/.31565. Feature preparation, small probes and inherited V1 training are outside those fitting durations. Seeded TF32 execution does not promise bitwise repeatability.

V3 makes **zero neural weight updates**. The observed failure was concrete: full PAUSE fitting errors reach .99611 while correct scores reach .99789, but the coarse grid jumps .995 -> 1.000001. A regression test reproduces rejection of every command under that grid. The first paired change uses midpoint thresholds between observed probabilities, excluding gaps below 1e-5. The second separates thresholds for each finite output instead of transferring one song/value's errors to every command in its family; acceptance requires at least two correct fitting positives. All weights, 328-way competition, inputs, error limits and frontend remain fixed.

Policies fit development/calibration only, maximize correct human then synthetic accepts, retain the .03 human/.01 synthetic empirical known-error limits and require zero fitting UNKNOWN accepts in every group. Original wins a complete tie. These are fitting constraints, not population confidence guarantees; the same folds cannot independently validate their own fitted risk. The selected policies and model/native/file hashes were frozen **before** V3 test inference. Historical tests were already examined in previous runs; the new scores are regression evidence, not fresh blind estimates. No threshold was tuned after V3 scoring.

| Fitting correct accepts, dev + calibration | Compact human /315 | Full human /315 | Compact synthetic /2208 | Full synthetic /2208 |
| --- | ---: | ---: | ---: | ---: |
| V2 coarse per intent | 87 | 70 | 1382 | 1416 |
| Observed intervals per intent | 89 | 81 | 1468 | 1624 |
| Observed intervals per finite command | 92 | 89 | 1728 | 1823 |

The final class policies have zero observed fitting wrong/UNKNOWN accepts. Missing, partial or noncanonical class policies fail closed. An independent review verified graph identity, split scope, error limits, complete class policies and dispatch action-disable behavior. The action bug identified in the initial review was reproduced RED/GREEN: worker `actions_enabled:false` now overrides server settings for providers and local timers, and deployment configuration locks actions off even with `ACTIONS_ENABLED=true`.

## Locked command-only results

| Outcome | Compact V2 -> V3 | Full V2 -> V3 |
| --- | ---: | ---: |
| Personal correctly accepted /36 synthetic positives | 0 -> **31** | 23 -> **30** |
| Personal wrong accepted positive commands | 0 -> 1 | 0 -> 1 |
| Personal accepted negatives /52 | 0 -> 1 | 0 -> 0 |
| Official exact score /196, includes correct UNKNOWN rejections | 115 -> **138** | 116 -> **147** |
| Official UNKNOWN accepts /10 | 0 -> 1 | 0 -> 0 |
| Original project correct human accepts /636 | 143 -> 165 | 136 -> 192 |
| Original project wrong positive-command accepts | 15 -> 25 | 16 -> 24 |
| Original project UNKNOWN accepts /229 | 11 -> 12 | 6 -> 15 |
| Typed holdout correct human accepts /83 | 12 -> 21 | 13 -> 28 |
| Typed holdout UNKNOWN accepts /11 | 0 -> 0 | 0 -> 0 |
| Original project synthetic correct accepts /3368 | 2371 -> 2762 | 2350 -> 2889 |

Personal V3 recall is 86.11% / 83.33%; precision across all accepted personal/negative inputs is 31/33 (93.94%) compact and 30/31 (96.77%) full. Compact misreads “Gone by Twice” as “Brave by Twice” and accepts unknown “Mystery Planet” as “Basics”; full omits the explicit artist on one “Queen of Hearts by Twice” recording. These are retained acoustic/complete-command errors, not default substitutions added by a parser. They were not used for another tuning pass.

Raw diagnostic personal correctness remains 33/36 / 31/36; raw official exact scores remain 172/196 / 180/196. Those raw outputs accept 8/10 / 4/10 official UNKNOWNs and are not the application policy. Official V3 correct-positive counts are 129/186 / 137/186; the all-row score also counts 9/10 / 10/10 correct UNKNOWN rejections. The original project human precisions are 165/202 (81.68%) and 192/231 (83.12%). Thus the practical personal target passes while broader precision/UNKNOWN and human coverage gates fail. No release qualification or installation follows from this target.

| Human project family | Positive n | Compact raw / accepted correct | Full raw / accepted correct |
| --- | ---: | ---: | ---: |
| ALARM | 32 | 20 / 11 | 28 / 19 |
| BRIGHTNESS | 23 | 13 / 9 | 13 / 7 |
| CALL | 6 | 6 / 6 | 3 / 3 |
| COLOR | 67 | 47 / 17 | 44 / 12 |
| CREATE_REMINDER | 18 | 4 / 4 | 14 / 10 |
| LIGHT_OFF | 45 | 20 / 2 | 24 / 10 |
| LIGHT_ON | 30 | 10 / 2 | 8 / 1 |
| LIST_REMINDERS | 18 | 8 / 5 | 12 / 6 |
| MESSAGE | 6 | 0 / 0 | 6 / 6 |
| NEXT | 36 | 28 / 27 | 33 / 30 |
| PAUSE | 50 | 43 / 38 | 43 / 3 |
| PLAY_MUSIC | 8 | 2 / 0 | 4 / 4 |
| STOP | 72 | 65 / 5 | 63 / 15 |
| TEMPERATURE | 28 | 17 / 5 | 16 / 15 |
| TIME | 49 | 31 / 21 | 32 / 24 |
| TIMER | 19 | 13 / 5 | 12 / 5 |
| VOLUME | 113 | 53 / 8 | 69 / 19 |
| WEATHER | 16 | 7 / 0 | 6 / 3 |

Per-value complete-command results and strata/source summaries are in [results.json](../experiments/vcm-mvp-hey-iris-v3/results.json). [Per-role results](../experiments/vcm-mvp-hey-iris-v3/per-role-results.json) use every eligible positive with that expected role; exact value and matching intent are required. They retain intent-matched denominators rather than claiming parser coverage as spoken accuracy. Raw records remain under `.cache/vcm-mvp-hey-iris-v3/runs/policy-20261017/{compact,full}/`; earlier V1/V2 failures remain unchanged.

## Wake remains a separate experiment

Hey Iris is a random-initialized 82,114-parameter width64 depthwise network, 1.6 s rolling window / 80 ms scoring, two-hit hysteresis, .98 threshold, .3 release and 1 s cooldown. It uses no pretrained acoustic embedding. Local pinned Qwen VoiceDesign/Base assets generate wholly designed voices, not unapproved human cloning: 504 clips from ten synthetic voices (4 train / 2 dev / 2 calibration / 2 test), including 144 Hey Iris, 120 confusable non-wakes and 240 commands. Generated utterance and saved-PCM hashes are verified. The separate wake data path does not alter any historical human-only study.

Seed61, AdamW .001, 20×96 updates, batch64, selected epoch3 on development. Wake fitting takes 12.18 s, excluding generation (424.7 s) and preprocessing. The 96 train positives produce 1,152 shifted windows; whole phrases are preserved, no positives excluded. Training uses random position, gain ±6 dB and white-noise SNR20–35 with .5 probability, adapting the official [TensorFlow Speech Commands augmentation example](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/examples/speech_commands/input_data.py). This is white noise, not a room-response or realistic-noise study.

Both the original and separately declared balanced-hard-negative wake arm yield 8/16 calibration detections and zero false/duplicate wakes on 200 negatives; original wins the locked pre-test tie. Held-out test detects **12/16 synthetic wakes**, with zero duplicate/false wakes over 200 negative clips and 421.53 seconds of negative audio. Both actual exported engines reproduce every event count. Zero observed false wakes in about seven minutes does not establish the .5/hour household target (one-sided 95% Poisson upper bound approximately 25.6/hour). There are zero original human positive wakes, no owner backend and no real overlap-recovery evidence.

The existing session gate enforces wake -> release -> one endpointed command -> reset. Constructed same-synthetic-voice wake+command smoke pairs yield 23/36 compact and 28/36 full correctly accepted; these are separate from the primary 31/36 / 30/36 command-only results and are not original paired ownership data. Owner-protected startup fails closed. [Wake evidence](../experiments/vcm-mvp-hey-iris-v1/wake-results.json), [native/ORT streams](../experiments/vcm-mvp-hey-iris-v1/wake-engine-replay.json).

## Export, resident runtime and packages

ORT variable-length parity covers frames1,3,37,128,231; native ncnn uses the identical weights with an explicit positional input and also checks511 frames. Final extracted-package tests compare all 88 personal/negative clips plus 36 representative dev recordings: **124/124** source/native/ORT/resident decisions per track; Torch/native/ORT real-audio logits remain within 2e-5. Raw diagnostics never replace rejected authoritative outputs. All packages use the existing shared ZIP writer and launcher, native ncnn by default, a local ORT reference, the selected Iris export and hash-pinned Silero VAD (2,327,524 bytes). Both reference and native model bytes count.

The Linux resident workers run under libseccomp rules denying socket/connect/sendto/sendmsg/sendmmsg; a socket probe raises EPERM before inference. Both process all 124 requests with network blocked, including environment `ACTIONS_ENABLED=true`, while every event remains action-disabled. Fresh-process import audits show no Torch, old phonetic decoder or STT module. This proves desktop offline recognition, not offline dependency installation. Initial Python/Node setup needs network; dependencies/tools are not bundled wheels. No training corpus, TTS model, trainers or old wake embeddings ship.

| Final x86 CPU measurement, one pinned logical CPU, one thread | Compact | Full |
| --- | ---: | ---: |
| VCM frontend + inference + validation p50/p95 (ms), 372 observations | 6.04 / 12.33 | 9.98 / 20.12 |
| Resident WAV request round-trip p50/p95 (ms), 124 requests | 7.72 / 15.37 | 11.61 / 23.25 |
| Fresh worker startup / reported model load (ms) | 606.10 / 126.26 | 665.75 / 135.69 |
| Worker high-water RSS incl. wake/VAD smoke (MiB) | 121.99 | 126.29 |
| Transfer ZIP bytes | 10258660 | 18760448 |
| Extracted manifest payload bytes | 12473519 | 21661100 |

These timings exclude file I/O, utterance duration and the 400 ms VAD endpoint wait. Worker round-trip includes JSON/base64 control transport. CPU scheduling was shared; the core was pinned, not exclusively reserved. Memory includes the bounded wake/VAD replay and excludes Node/browser. **Every physical Pi accuracy, wake FAR, p95, RTF, RAM, CPU and thermal field remains not measured.**

Final reviewed archives are [compact R2](../dist/vcm-mvp-hey-iris-v3/iris-compact-candidate-r2.zip) and [full R2](../dist/vcm-mvp-hey-iris-v3/iris-full-candidate-r2.zip). Manifest hashes bind all 113 payload files. Their startup action is `candidate-only`; they cannot be hot-installed through the production model-update path. Extract into a new directory and retain the installed bundle. Earlier V2/V3 package attempts and receipts remain preserved. The two-slide [PPTX](../dist/vcm-mvp-hey-iris-v3/iris-command-only-results.pptx) / [PDF](../dist/vcm-mvp-hey-iris-v3/iris-command-only-results.pdf) is generated beside these ZIPs; the original supplied template file was not mounted, so these are replacement editable slides, not a claim to have filled that binary template.

```bash
# In a fresh candidate extraction (first setup needs network):
python3 start.py --wav command.wav
# Explicit wake+command replay, separately:
python3 start.py --wav wake-and-command.wav --wake-wav
```

Software verification: **63 focused Python tests**, **89 UI/server tests**, zero type-check errors/warnings and production UI build pass. New policy/action regressions were observed failing before fixes and passing afterward. Repository-wide bare pytest still stops during collection on two unrelated historical problems: duplicate `test_catalog` module names across old experiment folders and `scripts/test_prototype.py` importing missing `train.register_sources`; they were not concealed or changed. [Verification and source hashes](../experiments/vcm-mvp-hey-iris-v3/verification.json). Existing scoped ignore rules pass; no new ignore commit is needed. All 29 recorded original input hashes, 12 recorded installed files and the interrupted compact checkpoint remain unchanged. No branch switch, staging, cleanup, deletion, installation or unrelated commit occurred.

## Reproduction and next evidence

Run from the preserved worktree with the root `.venv`. Exact completed run IDs are `complete-20261017`, `longer-v2-20261017`, `augmented-20261017`, `policy-20261017`; commands refuse to overwrite evidence. A new run must use a new ID/version instead of rerunning into completed output directories.

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m scripts.train.mvp_static_train train --model compact --device cuda --run-id longer-v2-20261017 --recipe experiments/vcm-mvp-hey-iris-v1/longer-fit-v2.json
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m scripts.train.mvp_static_train train --study vcm-mvp-hey-iris-v2 --model compact --device cuda --run-id augmented-20261017
.venv/bin/python -m scripts.train.mvp_policy_refine fit --model compact
.venv/bin/python -m scripts.train.mvp_policy_refine personal --model compact
.venv/bin/python -m scripts.train.mvp_policy_refine official --model compact
.venv/bin/python -m scripts.train.mvp_policy_refine evaluate --model compact
.venv/bin/python -m scripts.mvp_delivery --bundle .cache/vcm-mvp-hey-iris-v3/runs/policy-20261017/compact/export --wake .cache/vcm-mvp-hey-iris-v1/wake-v1/export --vad-folder .cache/speaker-final-refinement-v1/bundle/wake --output dist/vcm-mvp-hey-iris-v3/iris-compact-candidate-r2.zip
```

Use `full` for the parallel comparison track. V3 calibration uses CPU only; no second GPU is needed. Historical source/machine paths and cache dependencies remain documented, so portable fresh-clone training is not claimed.

The measured ablation supports a calibration granularity problem in the personal target. It does not prove the cause of all remaining acoustic errors: raw complete-command errors, capture/endpoint differences and source generalization remain separately measured limitations. The next quality study needs additional independently verified human positive/unknown personal recordings and a new frozen evaluation, plus explicit Pi/live testing when hardware is available. Do not tune against the named V3 failures or describe five unseen people, ownership, room noise/distance, overlap separation or a physical Pi demo as completed.

Skills were found through the system skill discovery workflow. [Superpowers systematic debugging](https://github.com/obra/superpowers/blob/main/skills/systematic-debugging/SKILL.md) supplied reproduce/ablate/verify discipline; [K-Dense experimental design](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/experimental-design/SKILL.md) supplied the paired, held-constant comparison. The [PyTorch training skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/pytorch-lightning/SKILL.md) and [MLOps routing skill](https://github.com/vaquarkhan/ml-ops-agent-skills/blob/main/skills/using-mlops-agent-skills/SKILL.md) were inspected; this small local experiment retains existing PyTorch/JSON tracking rather than introducing Lightning, a cloud tracker or Kubernetes. No skill was installed. Software/workflow reference: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065); bibliographic record checked 17 October 2026. That paper is workflow provenance, not evidence for these model scores. [Discovery receipt](../experiments/vcm-mvp-hey-iris-v3/skill-discovery.json).
