# OptionB tiny joint-command VCM — speaker-optionb-tiny-v1

Status: **completed bounded fit/export/evaluation. Parameter budget passes; usable recognition/actuation quality fails. Do not promote this candidate.** Existing installed/submission/UI models are unchanged.
Date: 2026-10-22. Requirements: ME-01/02/03/04/06/07/08; all ten use-case families only within the user-approved finite OptionB milestone.

## Decision and scope

The user requested a practical model around <=300k parameters, not a larger architecture. This candidate reuses the existing separable temporal CNN, predicting one **complete typed command** or UNKNOWN. It retains all **93 OptionB definitions /31 complete commands**. A timer value, light target, color, AM/PM or reminder task is part of the class; a correct intent with the wrong value is not a correct command.

The model has **258,912 learned parameters**, including stem, four residual blocks, per-frame normalization and the full-command output head. The input is 40-bin causal log-mel at 16kHz; fixed in-model temporal stride 2 yields a 20ms frame step. The four blocks use width160, kernel7 and dilations1/2/4/8, with masked mean-plus-max pooling and one 32-way linear head. No transformer, CTC, transcription, teacher, second recognizer, keyword rescue, separate intent head or separate slot head. Depthwise-separable CNN literature motivates small-footprint acoustic modeling; its results are not this project's accuracy evidence.[1]

Sources of truth:
- [Protocol](../experiments/speaker-optionb-tiny-v1/protocol.json).
- [Model](../scripts/models/speaker_optionb_tiny.py), reusing [VCM](../scripts/models/vcm.py).
- [Shared class order](../scripts/runtime/optionb_contract.py).
- [Training/data code](../scripts/train/speaker_optionb_tiny.py).
- [CPU runtime](../scripts/runtime/speaker_optionb_tiny.py).
- [Tests, export/selection and final-replay entry point](../scripts/test/test_speaker_optionb_tiny.py).

## Data and segregation

[Data freeze](../experiments/speaker-optionb-tiny-v1/data-freeze.json) binds the new views and implementations, plus all289protected inputs from the prior accepted-command investigation. No original source/audio, historical trainer, shipped graph, policy, publisher input or module-map entry is changed.

| View /origin | Original rows | OptionB positives | True UNKNOWN | Outside-OptionB negatives | Ambiguous, unscored | Covered complete commands |
|---|---:|---:|---:|---:|---:|---:|
| Train human |1889|1284|473|132|0|25/31|
| Train provided synthetic |21426|21426|0|0|0|31/31|
| Dev human |256|171|63|22|0|22/31|
| Dev provided synthetic |995|873|28|94|0|31/31|
| Calibration human |210|100|88|22|0|15/31|
| Test human |1050|609|229|27|185|31/31|
| Test provided synthetic |3368|3368|0|0|0|31/31|
| Holdout human |96|80|11|3|2|30/31|
| Holdout provided synthetic |100|100|0|0|0|31/31|

The treated human train view retains the26previously recovered semantic-only originals; PAUSE requires no additional slots and the recovered negatives remain rejection references. There are no fabricated phoneme targets because this model has no phoneme loss. The521unresolved references remain excluded. References are existing frozen metadata labels, not newly independent listening truth; prior ASR-derived/unlistened-reference limitations are not erased.

Six complete commands have **zero original-human training recordings**: brightness60/100%, reminder tasks`task.drink_water`/`task.exercise`, MESSAGE and a10-second timer. Their coverage is synthetic-only in training. Synthetic training also contains **zero task-negative recordings**, whereas synthetic dev has122negatives. These are measured coverage/distribution gaps, not proof that either gap alone causes the observed recognition failures.

Train/reserved recording ID, file/PCM hash and declared human speaker/parent/session/alias separation pass. Source references and original partition assignments remain intact. Valid broader commands outside the31allowed commands are explicit task negatives, not silently removed positives. Ambiguity is never fitted or scored as UNKNOWN.

Normalization is refitted from **exactly1889unique admitted human TRAIN recordings**, with equal utterance weighting. Synthetic/dev/calibration/test/holdout do not fit it. Fresh raw features are extracted with BLAS threads1, held in RAM and tensor-hashed before optimization; no duplicate corpus or feature payload is written. This is a new matched training/inference numerical version, not a hash bypass for historical cached features.

Integrity checks do read protected held-out payload bytes to verify hashes. They do **not decode held-out PCM, extract held-out training features or run held-out inference before selection**. The historical-style `heldout_audio_read=false` fit-receipt field means no acoustic decode/inference; the selection receipt records this distinction explicitly.

## Training and selection

The predeclared run is random initialization, seed53, six human-only epochs followed by42mixed epochs,96steps/epoch,32examples/step. AdamW starts at.002, with.0001weight decay,.02label smoothing, gradient clipping5 and mixed-phase cosine decay. The existing root environment is CPU-only PyTorch, so the run uses four CPU threads; no CUDA/dependency/environment replacement is performed.

The real run completed all48epochs /4608optimizer updates /147456sample draws in2987.91seconds (49.8minutes of fitting plus development evaluation, excluding initial extraction). **Epoch44** was selected, never a holdout-selected checkpoint. [Training summary](../experiments/speaker-optionb-tiny-v1/training-summary.json) and [all48epoch metrics](../experiments/speaker-optionb-tiny-v1/training-curve.json) preserve the measured trajectory. Replacement sampling actually drew1872distinct human and13865distinct synthetic rows across the full run; the selected checkpoint had seen1868human/13336synthetic rows. These are not additional recordings. Normalization still uses all1889admitted unique human-train recordings.

Human-only batches contain24human positives/8human negatives. Mixed batches contain12human positives/8human negatives/12provided-synthetic positives:62.5%human. Positives are balanced by complete-command class and declared speaker. Negatives balance true UNKNOWN versus outside-OptionB, then speaker. Sampling/repetition and augmentation are not additional independent recordings.

Augmentation retains every original frame: random0–1.2s floor-valued context on either side, smooth log-energy channel/gain perturbation on every synthetic sample and usually on humans, small normalized feature jitter and at most four frequency bins masked. No time cropping, word splicing or transcript-based inference. Feature context is a training transform; final robustness probes use actual zero PCM and fresh extraction, not a batch-padding test.

Only mixed epochs can be selected. Rank is human-dev complete-command raw micro-F1, then precision, then synthetic-dev recall. One global posterior/margin policy is selected from the predeclared grid on human dev/calibration only. Each cohort requires >=.99precision and <=.01negative false-accept rate with at least one true accept; otherwise the application policy is disabled. Recall target is>=.90, not merely zero false accepts. No per-command threshold fitting or threshold/weight adjustment after final test/holdout.

## Software/runtime evidence already exercised

- Real random-initialized optimizer fixture executes both phases and changes weights; this is software evidence, not speech accuracy.
- FP32 ONNX export executes on CPU with dynamic batch/frame lengths, matching PyTorch on multiple odd/even shapes. Normalizer, model, command order and policy hashes fail closed. Runtime import is Torch-free.
- [Action boundary fixture](../experiments/speaker-optionb-tiny-v1/action-boundary-test.mjs) runs the **existing** TypeScript dispatcher with fake providers, isolated transpilation and network forbidden. [Receipt](../experiments/speaker-optionb-tiny-v1/action-boundary-results.json):31complete-command routes,31WAV previews and31duplicate rejections pass. Exact finite values survive dispatch. Timers are closed after the fixture. No weather is fetched, notification sent, camera opened, music/light changed or live microphone operated.
- The boundary fixture uses the pinned TypeScript5.9.3 compiler in Hermes scratch; no UI/node_modules install, global PATH change, project dependency change or UI rebuild. It does not establish a full GUI or Pi run.
- **132focused tests pass.** The selected graph additionally matches PyTorch on all466recorded human dev/calibration inputs, with maximum absolute logit difference4.7684e-6. The ONNX checker and variable-batch/odd-even-frame tests pass. Export warnings about the legacy exporter and non-folded strided Slice nodes are preserved in [export verification](../experiments/speaker-optionb-tiny-v1/export-verification.json), not hidden.
- The real WAV CLI was exercised on an already-correct development recording and emitted PLAY_MUSIC with `source=wav`, `actions_enabled=false`. `.cache/speaker-optionb-tiny-v1/run-01/cli-smoke.json` is a selected software smoke, not an accuracy estimate.

## Final acoustic results

The [selection lock](../experiments/speaker-optionb-tiny-v1/selection-before-test.json) predates final inference. [Policy selection](../experiments/speaker-optionb-tiny-v1/policy-selection.json) chooses joint posterior>=.99 /margin>=.30, using human dev/calibration only. This yields20/171correct dev accepts and5/100calibration accepts, with zero observed wrong accepts in both. It meets the feasibility safety constraints but **fails the>=90%recall target before final evaluation**. Selection is retained honestly rather than adjusted on test/holdout.

[Final results](../experiments/speaker-optionb-tiny-v1/results.json) reconcile all4896inference cases: the4418test originals,196holdout originals and282dependent PCM-silence trials. Human/synthetic and original/stress cases are not pooled. Human holdout has80positives/14task negatives/two ambiguous unscored; human test has609positives/256task negatives/185ambiguous unscored. The positive holdout recordings come from only two declared speakers; these data are already exposed, not blind.

| Cohort /decision | Correct complete commands | All wrong accepts (FP) | Accepted precision | Positive recall | Complete-command micro-F1 | Task-negative false accepts |
|---|---:|---:|---:|---:|---:|---:|
| Human holdout, raw argmax |76/80|12|86.36%|95.00%|90.48%|8/14|
| Human holdout, fixed joint.95 |26/80|0|100.00%|32.50%|49.06%|0/14|
| Human holdout, selected policy |3/80|0|100.00%|3.75%|7.23%|0/14|
| Human test, raw argmax |469/609|187|71.49%|77.01%|74.15%|87/256|
| Human test, fixed joint.95 |133/609|3|97.79%|21.84%|35.70%|2/256|
| Human test, selected policy |41/609|0|100.00%|6.73%|12.62%|0/256|
| Synthetic holdout, raw argmax |99/100|1|99.00%|99.00%|99.00%|N/A, no negatives|
| Synthetic holdout, selected policy |42/100|0|100.00%|42.00%|59.15%|N/A, no negatives|
| Synthetic test, raw argmax |3326/3368|39|98.84%|98.75%|98.80%|N/A, no negatives|
| Synthetic test, selected policy |1317/3368|0|100.00%|39.10%|56.22%|N/A, no negatives|

A wrong accepted value contributes both FP and FN. Raw holdout has four wrong positive commands plus eight negative proposals, not merely four total errors. Its95%positive correctness is **not**95%overall safe actuation. Raw false accepts are6/11true UNKNOWN plus2/3outside-OptionB commands. The broader human test has75/229true-UNKNOWN and12/27outside-task raw accepts. Zero observed selected-policy errors does not prove zero risk, especially when almost everything is rejected. The fixed.95 diagnostic is a new full-command posterior, not the same probabilistic quantity as the historical.95intent/.90slot gates.

The predeclared comparison is the prior full model's scoped original human holdout:79/80raw correct,39/80correct accepted,0/14task-negative accepts,48.75%recall and65.55%F1. The tiny model is **worse as an accepted-command replacement**, despite meeting the size target. No threshold was lowered after seeing this regression.

### Silence context and full finite coverage

| Human holdout condition | Raw correct | Fixed.95 correct accepts | Selected correct accepts | Raw negative proposals |
|---|---:|---:|---:|---:|
| Original |76/80|26/80|3/80|8/14|
| Prefix1s |76/80|24/80|2/80|9/14|
| Suffix1s |76/80|22/80|3/80|6/14|
| Both ends1s |76/80|22/80|2/80|8/14|

These are paired transformations of94eligible human parents, not282new human recordings. Raw positive correctness stays stable, while negative decisions and acceptance still change. No causal augmentation benefit or resilience to unrelated spoken words is established.

[Per-command CSV](../experiments/speaker-optionb-tiny-v1/per-command.csv) has217rows (31commands across seven strata), [per-intent CSV](../experiments/speaker-optionb-tiny-v1/per-intent.csv) has126rows, and [per-verbatim CSV](../experiments/speaker-optionb-tiny-v1/per-verbatim.csv) retains all93definitions. Verbatim counts are exact metadata-reference matches, not independent listening; paraphrases remain positive at the typed-command level. LIGHT_OFF has no positive human holdout reference and is N/A, not a failed scored positive.

| Human holdout intent | Positives | Raw correct | Selected correct accepts |
|---|---:|---:|---:|
| PLAY_MUSIC |3|1|0|
| WEATHER |3|3|0|
| TIME |3|3|0|
| LIGHT_ON |2|2|0|
| LIGHT_OFF |0|N/A|N/A|
| PAUSE |3|3|1|
| STOP |3|3|0|
| NEXT |3|3|1|
| VOLUME |4|3|0|
| CALL |3|3|0|
| MESSAGE |3|3|0|
| LIST_REMINDERS |3|3|0|
| TIMER |9|9|1|
| ALARM |8|8|0|
| TEMPERATURE |9|8|0|
| BRIGHTNESS |6|6|0|
| COLOR |6|6|0|
| CREATE_REMINDER |9|9|0|

### Diagnosis and next decision

The independent auxiliary-head disagreement is structurally absent, but **a single head is not sufficient to separate valid commands from unsupported speech at useful recall**. At the selected epoch, human-dev raw metrics are141/171correct positives with27/85negative false accepts; synthetic-dev raw negative proposals are103/122despite865/873correct positives. Tight calibration therefore rejects most correct commands. This is observed score/negative-separation weakness, not an index-order or arithmetic repair, and not proof that<=300k is impossible.

Keep this as a small, reproducible baseline, **not an installation candidate**. Prioritize a development-only comparison of training-negative/source coverage and confidence separation, including the six missing original-human command cells, before growing the architecture. A compact version of the existing finite acoustic matcher is also a distinct follow-up hypothesis, not an implemented or measured result here. New data, losses or rejection features require a new study version; these final results must not drive fitting or thresholds in this version.

Weight selection in this version maximizes raw development micro-F1, not accepted recall at the desired safety point. Operationally aligned checkpoint selection is another follow-up hypothesis; this run does not show that epoch44is the best possible calibrated operating point among all small models. No additional checkpoint or threshold is selected from final test/holdout outcomes.

## Measured resource envelope

[Export verification](../experiments/speaker-optionb-tiny-v1/export-verification.json) counts258912ONNXinitializer elements, matching all learned parameters. FP32 graph size is1082940bytes (**1.03MiB**); the five-file model/commands/normalizer/policy/recipe bundle is1088530bytes (**1.04MiB**), excluding Python/native dependencies and wake/VAD/TTS/UI assets.

The [fresh-process desktop probe](../experiments/speaker-optionb-tiny-v1/desktop-resource-results.json) measures256original human-dev utterances on Windows11, Python3.13.14, ORT1.23.2, one inference and BLAS thread: **p50 4.91ms /p95 7.70ms**, session creation21.28ms. Peak process working set is79.75MiB; peak private commit574.11MiB (not the same as physical RSS). Torch is not imported. Timings exclude WAV loading, speech duration, wake/VAD/endpoint wait, UI/TTS and providers. These are **not Pi, microphone, end-to-end or full-application measurements**.

## Verification and preserved state

[Independent verifier](../experiments/speaker-optionb-tiny-v1/verify-results.py) and [receipt](../experiments/speaker-optionb-tiny-v1/verification.json) pass: all4896prediction rows, seven metric strata, command/intent tables, original/stress identities, exact freshly recomputed1889-row train normalization, sample membership and all289historical bindings. The full fit, source/selection locks and no-overlap assertions are verified;132focused tests, compile checks and `git diff --check` pass. No source records, historical models, source-map entries, installed artifacts, provider settings or prior policies were replaced. No commit/push, microphone operation, Pi deployment or real device action occurred.

All final datasets remain historically exposed regression evidence, **not fresh blind qualification**.

## Reproduction /artifacts

Run from the repository root using its existing `.venv/Scripts/python.exe`:

1. `-m scripts.train.speaker_optionb_tiny prepare` (exclusive creation; already completed).
2. `-m scripts.train.speaker_optionb_tiny fit` (preserve the run; no implicit overwrite/resume).
3. `-m scripts.test.test_speaker_optionb_tiny select` (requires completed fit receipt; exports/calibrates/locks before evaluation).
4. `-m scripts.test.test_speaker_optionb_tiny evaluate` (no fitting or threshold search).
5. `experiments/speaker-optionb-tiny-v1/verify-results.py --write-receipt` (independent metrics, exact train-only normalization and identities).

Mutable weights, ONNX, raw predictions and logs: `.cache/speaker-optionb-tiny-v1/run-01/`. Compact protocol, freeze, selection, metrics and verification: `experiments/speaker-optionb-tiny-v1/`. Inference command: `-m scripts.runtime.speaker_optionb_tiny --bundle .cache/speaker-optionb-tiny-v1/run-01/bundle --wav <mono-16kHz-command.wav>`; WAV inference is non-actuating.

The <=300k count is the **entire new VCM**, not the entire existing wake/VAD/TTS application. Runtime dependency bytes and physical-Pi resource measurements remain separate. This candidate is **not installed**, does not replace the existing submission/UI model, and has no live speech, wake/VAD endpoint, other-speaker ownership, overlap or physical actuation qualification. A working recorded-command graph and a provider fixture are not proof of those outcomes.

## Sources

[1] https://arxiv.org/abs/1711.07128 — Hello Edge: Keyword Spotting on Microcontrollers
