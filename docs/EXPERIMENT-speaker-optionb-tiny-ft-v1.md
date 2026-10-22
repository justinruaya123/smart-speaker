# OptionB tiny-model fine-tune and false-accept diagnosis

Status: **completed; recognition regressed and quality gate failed. Do not promote.** Study started 2026-10-22 and final verification completed 2026-10-23 (UTC+08). Architecture remains **258,912 parameters**, with the same 93 OptionB definitions /31 complete typed commands plus UNKNOWN. No runtime ASR, transformer, second recognizer or auxiliary rescue head is added.

## Evidence and scope

- [Predeclared protocol](../experiments/speaker-optionb-tiny-ft-v1/protocol.json), [data/source freeze](../experiments/speaker-optionb-tiny-ft-v1/data-freeze.json), [executed baseline audit](../experiments/speaker-optionb-tiny-ft-v1/audit-results.json).
- New [trainer](../scripts/train/speaker_optionb_tiny_ft.py), [audit helpers](../scripts/test/speaker_optionb_tiny_ft_audit.py), [export/replay driver](../scripts/test/speaker_optionb_tiny_ft_eval.py), [regression fixtures](../scripts/test/test_speaker_optionb_tiny_ft.py).
- The [previous tiny study](EXPERIMENT-speaker-optionb-tiny-v1.md) remains untouched. The new research graph is not the installed UI/submission model. Nothing here operates a microphone, Pi or provider, or performs real actuation.

## Why the earlier model proposes wrong commands

### 1. A measured rejection-generalization gap

The exact epoch44 parent, without a confidence gate, gets **1,249/1,284 human training positives** correct and proposes a command on only **1/605 training negatives**. On human development it gets **141/171 positives** correct but proposes commands on **27/85 negatives**. Synthetic development has **103/122 negative proposals**. These are raw proposals, not accepted device actions.

A calibration recording whose frozen reference is a **3,660-second timer** is proposed as **60 seconds**, posterior **0.9819359477**. Development also shows unsupported temperatures being mapped to a supported temperature and named-light requests mapped to generic lighting. The schema does not round these references: they project to class0. The learned classifier emits the nearest supported command, and typed validation cannot independently recover what the speaker actually requested.

The original **.99 posterior/.30 margin** gate rejects the compound-timer proposal. Saved human dev/calibration decisions contain **zero wrong accepts** at that gate, but only20/171 and5/100correct accepts. Earlier human-test `.95` diagnostic decisions do contain three wrong accepts; raw results are still less safe. Do not conflate ungated argmax accuracy, a diagnostic threshold, the selected research policy and what an installed application actually runs.

These are errors against frozen metadata references. The compound timer has conflicting publisher slot metadata; no independent listening was added, so its true spoken content is not newly adjudicated here.

### 2. Coverage is not just a recording-count problem

All **21,426 original synthetic training recordings** are positive, whereas development has122synthetic negatives. The independent local-source inventory found no safe omitted training-negative pool: the apparent122extra negatives are development recordings,529omitted gap-fill recordings failed transcription QA, and1,013other omitted synthetics are calibration-assigned. None is moved into training or converted into a negative.

Original-human training covers only **25/31 complete commands and31/93 exact normalized wordings**. Missing human command cells are brightness60%, brightness100%, reminder `task.drink_water`, reminder `task.exercise`, MESSAGE and a10-second timer. Synthetic coverage does not close these human-speaker gaps. The605existing human negatives include132valid broader commands outside exact OptionB, retained as task negatives rather than deleted.

This is measured dataset imbalance and domain/coverage shift. It is not a matched causal proof that any one source alone caused the failures.

### 3. Confidence and the actual operating objective

The parent was selected for **raw development F1**, not accepted recall at the required precision. Its `.02` smoothed32-class objective has idealized correct-class target **0.980625**, below its `.99` decision gate: PyTorch mixes the hard target with a uniform distribution.[1] This is **not a hard confidence ceiling**, nor proof smoothing is a bug. Confidence calibration and correctness are distinct; a high neural posterior is not an independently verified probability of safe actuation.[2]

A no-training finer-grid control on the unchanged epoch44 weights selects `.985/.30`, improving development correct accepts20→35 but calibration only5→6. Threshold resolution alone therefore does not repair the observed separation gap. This control is retained so any threshold-only improvement is not presented as acoustic learning.

## Label and data recheck

All **23,315 original training rows** were rechecked from pinned publisher metadata using the **archived approved annotation policy**, whose identity matches its release freeze. Result: **zero rule/reference disagreements, zero stale class projections and zero conflicting typed references among identical stored training texts**. No live proposal checksum guard is bypassed. This is annotation-rule consistency, **not independent ground truth**.

**288 human references remain ASR-derived and unlistened.** The521unresolved references, including six disputed training compound timers, remain excluded. The428blank listening templates are not completed reviews. Publisher `variation` is a category/prompt, not necessarily the full spoken reference: substituting it can erase a room, artist or complete duration. No label is changed on the basis of a model prediction.

The new optimizer view conservatively excludes **656 QA-approved supplemental gap-fill recordings** carrying historical publisher `val`105/`test`388/blank163assignments. It retains only664gap-fill rows whose historical field is exactly`train`. The publisher explicitly calls the supplemental corpus train-only for the current benchmark; the historical field is **not proof of current test overlap**. This is a new-version precaution, not a retroactive relabeling of the prior study.

New fitting membership: **1,889 human +20,770 provided synthetic =22,659 recordings**. All31commands remain represented. Current dev/calibration/test/holdout membership and eligibility are byte-identical to the parent views. Present declared recording/speaker/parent/pair/alias fields have no cross-split overlap, but missing synthetic alias fields and unverified cloned-reference overlap prevent a stronger voice-independence claim. Supplemental licensing/consent caveats remain unresolved; this stays research-only.

## Architecture assessment

The compact CNN is a valid baseline, not a demonstrated safe recognizer:

- One32-way atomic command/UNKNOWN head, **258,912 learned parameters**, with exact bundle/shared class order and all93definitions round-tripping.
- Four width160separable residual blocks, kernel7/dilations1,2,4,8; temporal stride2 and masked mean/max pooling.
- Each encoded frame has **3.705s of causal PCM support**. Pooling consumes the **whole valid utterance**, so this is not a3.705s utterance cutoff.421human training recordings exceed that duration, but they are not truncated.
- Independent random fixtures found correct causal/padding/valid-length behavior. Nine frontend cases matched training/runtime bitwise; actual parent parity on16training recordings differed by at most4.77e-6. Exact1889-row train-only normalization was reconstructed successfully. No near-silent, runtime-overlength or heavily clipped human training recordings were found by the stated numeric screens; these screens are not listening or VAD truth.
- Global mean/max pooling loses sufficiently distant order; a separated-block permutation fixture demonstrates that limitation, not the acoustic cause of the observed timer error. The shared frontend/stride can leave up to19.94ms of final PCM outside the retained windows. No causal real-speech tail-loss experiment has been performed.

The immediate study keeps architecture and preprocessing unchanged, testing training/selection before adding capacity or decoders. Reliable rejection of unsupported slots remains learned from data; a supported output JSON does not validate the utterance.

## Controlled continuation

All three arms start from the exact **epoch6 human-only tiny foundation**:1,642sampled original-human IDs/18,432draws, **zero synthetic draws**. This avoids inheriting the old epoch44 continuation's excluded historical gap-fill exposures. The resulting candidates are **mixed-source fine-tunes**, not human-only models or new random initializations.

Each arm executes8epochs×96updates×32examples, seed73, fresh AdamW, cosine LR.0004→.0001, weight decay.0001, clip5. Same original augmentation and inherited1889-human normalizer; no inference-only preprocessing swap. No new TTS, splices, phone targets, downloads or independent human recordings are created.

| Arm | Label smoothing | Human positive /human negative /synthetic positive draws | Extra hard negatives |
|---|---:|---|---|
| smoothed_control |.02|12/8/12|0|
| unsmoothed |0|12/8/12|0|
| negative_focus |0|8/16/8|8of16negative draws|

The first two arms share exact sampling/augmentation RNG. The third tests the **combined** higher-negative-exposure/hard-mining intervention, not separately identified causal effects. Hards are the fixed top quartile within negative scope, scored from human TRAIN by the same human-only foundation. ASR-derived references are excluded from extra hard-example amplification; their existing labels are not silently rewritten.

Weights and one arm are chosen on **human development feasible accepted recall first**, then raw human F1/precision and synthetic recall; exact ties keep the earlier epoch/arm. Calibration does not choose weights/arm. After that model lock, one global probability/margin policy is selected on dev/calibration with>=.99precision,<=.01negative false-accept rate and>0TP in **each** cohort; otherwise disable. No per-command exceptions. Target recall remains>=.90, and zero observed errors in small negative sets does not certify population risk.

## Final results

All three arms completed8epochs/768updates each: **2,304 real optimizer updates /73,728 draws**, not additional recordings. Development selected **smoothed_control, continuation epoch1**. The no-smoothing arm selected epoch5and negative-focus epoch1, but neither beat the predeclared development safety/recall rank. Matched smoothing-control sampling counts were independently verified equal.

| Development-selected arm | Human raw correct /171 | Synthetic raw correct /873 | Dev-only correct accepts /171 | Wrong accepts |
|---|---:|---:|---:|---:|
| Smoothed control, epoch1 |103|387|21|0|
| No smoothing, epoch5 |120|641|20|0|
| Negative focus, epoch1 |99|335|19|0|

The selected model was locked **before calibration**. Export matches Torch on all466recorded human development/calibration inputs plus three dynamic-batch/odd-even-frame fixtures, max recorded logit error4.7684e-6. The final global **.98posterior/.30margin** policy gives **4/171dev and3/100calibration** correct accepts, zero observed wrong accepts in those cohorts. It already fails the>=90%recall target there.

[Final results](../experiments/speaker-optionb-tiny-ft-v1/results.json) and the [unchanged-weight control](../experiments/speaker-optionb-tiny-ft-v1/baseline-controls.json):

| Human cohort /model and policy | Raw complete commands correct | Correct accepts | Wrong accepted commands | Accepted recall |
|---|---:|---:|---:|---:|
| Holdout /original epoch44, original.99 |76/80|3/80|0|3.75%|
| Holdout /same epoch44, finer-grid.985 |76/80|5/80|0|6.25%|
| Holdout /new selected continuation.98 |35/80|0/80|0|0%|
| Test /original epoch44, original.99 |469/609|41/609|0|6.73%|
| Test /same epoch44, finer-grid.985 |469/609|59/609|1|9.69%|
| Test /new selected continuation.98 |277/609|8/609|2|1.31%|

The new test acceptance precision is **80%**, not100%: both errors are positive commands with the wrong slot, not true-negative activations. Against frozen references, two22°C requests are accepted as18°C, with posteriors0.9845and0.9945. Thus **zero task-negative accepts does not mean zero wrong actuation decisions**. These are metadata-referenced acoustic results, not newly listened adjudications or physical device actions.

New synthetic raw correctness is51/100holdout and1455/3368test, with selected correct accepts1/100and23/3368. These do not establish human accuracy. The untouched-parent finer-grid control also introduces one human-test false accept: its apparent recall gain is not an acoustic improvement or a qualified safety improvement.

### Interpretation and next bounded correction

This continuation setup is a failed replacement. Starting from the earlier human-only foundation and selecting an early checkpoint did **not retain the later mixed checkpoint's learned command coverage**. The development-only objective preferred21safe hits over20despite much worse raw/synthetic coverage; the final calibration gate then removed most of those21hits. The synthetic recall tie-break was not a coverage floor. These weaknesses were visible on development before final replay; the frozen first-study choices are not edited after seeing final results.

This is **not a clean fine-tune of epoch44**, nor evidence that the small architecture cannot improve. A separate [direct-continuation protocol](../experiments/speaker-optionb-tiny-direct-ft-v1/protocol.json) corrects the warm start to the actual trained epoch44checkpoint, keeps unchanged parent0as an eligible fallback and adds a development recognition-coverage floor. Its inherited mixed-source exposures are disclosed; no current reserved data may enter fitting. It cannot erase this failed run or claim fresh-blind qualification.

### Verified artifact and resource envelope

The [independent verifier/receipt](../experiments/speaker-optionb-tiny-ft-v1/verification.json) reconciles **4,896 predictions**,32fresh actual-graph original/stress replays, exact1889-human normalization reconstruction, sample membership and all**338protected artifacts**, including the original289inputs. Per-command/intent/verbatim exports contain217/126/93rows. Test/holdout memberships remain unchanged, and no labels are rewritten.

The runnable research bundle is `.cache/speaker-optionb-tiny-ft-v1/run-01/bundle`: **258,912 parameters /1,082,940-byte FP32 graph**. A fresh Torch-free one-thread CPU probe on256original human-dev recordings measures **p50 4.87ms /p95 7.46ms** and peak working set83,214,336bytes; process private commit is not physical RSS. This is not Pi/wake/VAD/UI/TTS/device-provider evidence. The actual WAV CLI returns PAUSE on a selected correct development smoke with `source=wav`, `actions_enabled=false`, `model_qualified=false`; that selected smoke is not an accuracy estimate.

The [36-case training-reference review queue](../experiments/speaker-optionb-tiny-ft-v1/human-train-reference-review.csv) includes nine ASR-derived cases, retains exact audio identities and leaves reviewer/decision blank. A model/reference disagreement is not proof of a bad label. No review case is automatically relabeled or admitted as new data.

The [independent export/evaluation review](../experiments/speaker-optionb-tiny-ft-v1/independent-review.json) passes with no reported security or logic findings and four bounded reviewer tests. Its non-payload inspection is separate from the parent's executed full verifier and real-audio replay. This verdict covers the first study only; it does not supersede the later direct-verifier findings. A final unchanged-source rerun again verifies all 4,896 records and 338 protected bindings. [Official reference evidence](../experiments/speaker-optionb-tiny-ft-v1/reference-evidence-receipt.json) is also retained outside the automatically pruned scratch directory, without changing frozen citations.

## Reproduction and verification

From project root with `.venv/Scripts/python.exe`:

1. `-m scripts.test.speaker_optionb_tiny_ft_audit` — exclusive executed baseline diagnostic.
2. `-m scripts.train.speaker_optionb_tiny_ft prepare` — exclusive source/label/split freeze; completed.
3. `-m scripts.train.speaker_optionb_tiny_ft fit` — three real continuations; preserve run.
4. `-m scripts.test.speaker_optionb_tiny_ft_eval select` — completed-fit guard, dev-only model lock, ONNX parity, global policy and final lock.
5. `-m scripts.test.speaker_optionb_tiny_ft_eval evaluate` — unchanged final reference membership, original/silence replay and no-training parent finer-grid control.
6. `experiments/speaker-optionb-tiny-ft-v1/verify-results.py --write-receipt` — independent memberships, lineage, updates, normalization, logits/metrics and sampled actual-graph replay.
7. `experiments/speaker-optionb-tiny-ft-v1/benchmark-runtime.py` — fresh Torch-free one-thread CPU/memory probe, not Pi evidence.

Runtime/weights/logs remain under `.cache/speaker-optionb-tiny-ft-v1/`. Prior source/weights/policies and all289original protected inputs remain frozen. No installation, commit/push, external provider call, microphone recording or physical actuation is authorized by this study.

## Sources

[1] https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html — PyTorch 2.10 CrossEntropyLoss documentation
[2] https://proceedings.mlr.press/v70/guo17a.html — On Calibration of Modern Neural Networks
