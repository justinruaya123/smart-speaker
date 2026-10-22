# Independent review handoff: final OptionB tiny candidate

Date: 2026-10-23. Start here for the user's request to read the final small model and criticize its flow. This is a review handoff, not a new fit, release approval or installation.

## 1. Exactly which model is under review?

**Primary target: `speaker-optionb-tiny-demo-v1`, original tiny epoch44 weights, 258,912 parameters, FP32 ONNX, global acceptance posterior0.50 and posterior margin0.10.** There are93 OptionB wording definitions,31 distinct complete typed commands and32 classifier classes including UNKNOWN. The current goal is a useful finite-phrase model around<=300k parameters, not unrestricted speech understanding.

Do not confuse this with the historical935k compact model, the2,066,831-parameter submission model, or the unsuccessful tiny fine-tunes. Their sources/evidence are retained for comparison. This tiny candidate is **not installed in the GUI or root runtime**. Its CLI emits typed preview commands but never performs physical actions. `model_qualified=false`, `actions_enabled=false`, and `qualified_output` always rejects.

The original .99/.30 tiny policy and later .985/.30 control are unchanged. The user explicitly authorized a separate less strict demo policy. This improved acceptance, **not the acoustic weights**.

## 2. Read these files first

Paths below are relative to the repository root; `experiments/` contains compact committed evidence, while `.cache/` contains local payloads and generated recordings/features/predictions.

| Priority | Files | What to inspect |
| --- | --- | --- |
|1|[Latest experiment report](EXPERIMENT-speaker-optionb-tiny-demo-v1.md), [requirements](../REQUIREMENTS.md), [experiment index](../EXPERIMENTS.md)|Actual scope, results, evidence levels and historical-versus-current decisions|
|2|[Tiny model](../scripts/models/speaker_optionb_tiny.py), [shared CNN](../scripts/models/vcm.py)|Actual forward path and parameter count; optional keyword/evidence branches in the shared class are NOT enabled by the tiny config|
|3|[Frontend](../vcm/frontend.py), [tiny trainer](../scripts/train/speaker_optionb_tiny.py)|Audio/features, masks, equal-utterance normalization, sampling, loss, augmentation and checkpoint selection|
|4|[Finite command contract](../scripts/runtime/optionb_contract.py), [typed schema and93 definitions](../scripts/runtime/pcv_schema.py)|Class order, allowed values and complete-command mapping; schema validity does not prove audio correctness|
|5|[Tiny runtime](../scripts/runtime/speaker_optionb_tiny.py)|Bundle integrity, capture rejection, ONNX interface, softmax/margin, UNKNOWN and preview/qualified output separation|
|6|[Original training protocol](../experiments/speaker-optionb-tiny-v1/protocol.json), [training summary](../experiments/speaker-optionb-tiny-v1/training-summary.json), [curve](../experiments/speaker-optionb-tiny-v1/training-curve.json)|What actually fitted and why epoch44 was selected|
|7|[Demo protocol](../experiments/speaker-optionb-tiny-demo-v1/protocol.json), [selector](../scripts/test/speaker_optionb_tiny_demo.py), [policy grid](../experiments/speaker-optionb-tiny-demo-v1/policy-selection.json), [pre-test lock](../experiments/speaker-optionb-tiny-demo-v1/selection-before-test.json)|One global development/calibration operating point, exact cached-score identity alignment, no held-out policy selection|
|8|[Final results](../experiments/speaker-optionb-tiny-demo-v1/results.json), [per-command](../experiments/speaker-optionb-tiny-demo-v1/per-command.csv), [per-intent](../experiments/speaker-optionb-tiny-demo-v1/per-intent.csv), [wording-v2 table](../experiments/speaker-optionb-tiny-demo-v1/per-verbatim-cohorts-v2.csv)|Complete-command TP/FP/FN, origin/split separation, negatives, ambiguous cases and missing evidence|
|9|[Core verifier](../scripts/test/speaker_optionb_tiny_demo_report.py), [metadata addendum](../scripts/test/speaker_optionb_tiny_demo_metadata.py), [core receipt](../experiments/speaker-optionb-tiny-demo-v1/verification.json), [wording receipt](../experiments/speaker-optionb-tiny-demo-v1/verification-wording-v2.json)|Exact canonical coverage, independent metrics/ranks, graph spot checks, file bindings and corrected reference availability|
|10|[Initial review](../experiments/speaker-optionb-tiny-demo-v1/independent-review-v1-full.json), [closure review](../experiments/speaker-optionb-tiny-demo-v1/independent-review-v2.json)|Original reporting failure and additive fix, not a silently rewritten success story|
|11|[Fine-tune report](EXPERIMENT-speaker-optionb-tiny-ft-v1.md), [direct-continuation report](EXPERIMENT-speaker-optionb-tiny-direct-ft-v1.md), their protocols/trainers/verifiers|Why neither attempted acoustic update became this candidate|
|12|[Review input fingerprints](../experiments/speaker-optionb-tiny-demo-v1/review-inputs.json)|Exact model/bundle, checkpoint and cached-evidence identities supplied separately from Git|

`AGENTS.md` governs repository preservation, not model quality. Existing historical reports may state “no commit” because that was true at experiment completion. The user subsequently requested committing the accumulated work on2026-10-23; their frozen protocols/receipts have not been rewritten to pretend the later commit existed during selection.

## 3. Model and recognition flow

1. **Input:** complete mono16kHz PCM utterance. The WAV CLI rejects other sample rates/channel counts rather than silently resampling. `Runtime.recognize` rejects nonfinite/out-of-range, near-silent, too-short or over15-second capture input.
2. **Frontend:** causal25ms Hann windows,10ms hop,512-point FFT,40 HTK mel bins,20–7600Hz, natural-log energy floor1e-8. No transcript is produced. Frozen per-bin mean/std are applied identically in training/inference.
3. **Temporal model:** take every second input feature frame; causal width160 stem, four residual blocks with kernel7 and dilations1/2/4/8. Each block has two depthwise/pointwise stages, per-frame layer norms, SiLU and dropout0.1 during training. Valid-length masked mean and max are concatenated into a320-value embedding and one32-logit linear head.
4. **Decision:** stable softmax over all32 classes; class0 always rejects. A nonzero winner must have posterior>=.50 and top-minus-second posterior margin>=.10. There are no per-phrase thresholds, transcript/cue parsers, nearest-word rescues, separate intent/slot gates or coercion to an allowed number.
5. **Output:** the selected class maps to the entire fixed typed command, then schema validation. Wrong room/value/duration/task/AM-PM is a wrong complete command, even if the JSON schema is valid. Schema validation cannot establish which number was spoken.
6. **Boundary:** the small-model runtime returns non-actuating research events. It does not implement wake-word capture, neural session ownership, overlapping-speaker recovery, VAD, TTS or physical provider actions. Existing larger-model GUI/dispatcher code and fake-provider tests are separate context, not a demonstrated tiny live integration.

The encoder is causal, but the delivered tiny recognition call pools the supplied utterance; that does not by itself establish streaming endpoints or a complete real-time microphone pipeline. A per-frame receptive-field calculation is not a claim that the utterance is cropped to that duration.

## 4. Data, labels, training and lineage

- Original tiny TRAIN:23,315 rows =1,889 human originals +21,426 supplied synthetic. Human TRAIN has1,284 supported positives and605 task negatives (473 original UNKNOWN,132 outside OptionB). Only25/31 typed commands have human TRAIN evidence; synthetic TRAIN has no task negatives. Human/Synthetic DEV contain negatives. Do not conflate source proportions with optimizer batch exposure.
- Normalization is the original frozen equal-utterance mean/second-moment calculation over exactly1,889 human TRAIN recordings. It is not refitted on DEV/calibration/test/holdout and not replaced only at inference.
- Initialization: random seed53; no pretrained encoder/teacher/CTC/ASR. AdamW, learning rate.002, weight decay.0001, label-smoothed cross-entropy.02, gradient norm clipping5. Six human-only epochs then42 mixed epochs,96 batches of32 per epoch. Human-phase batch:24 human positives +8 human negatives. Mixed:12 human positives +8 human negatives +12 synthetic positives. Sampling balances command/scope, then declared speaker, then recording with replacement.
- Augmentation is training-only feature channel/gain perturbation, small normalized jitter, up to four frequency-bin masks and optional floor-valued prefix/suffix context retaining all original frames. It is NOT measured acoustic reverberation, unrelated-word robustness or live VAD qualification.
- Completed48 epochs/4,608 updates/147,456 draws. The eligible mixed checkpoint with best human DEV raw complete-command F1, then precision, then synthetic recall is epoch44. Development checkpoint selection and later acceptance-policy selection are different steps.
- Subsequent three8-epoch foundation continuations regressed; the separate four-epoch direct mixed-parent continuation completed384 updates but retained unchanged parent epoch0 under its predeclared recognition-retention guard. The latest demo therefore contains no fine-tuned acoustic improvement.
- The accepted demo policy maximizes complete-command F1 over138 predeclared gates using466 unique HUMAN DEV/calibration references (271 supported positives,195 negatives). Its tie-breaks are minimum split F1, pooled precision/recall, then stricter probability/margin. Cached logits require exact unique IDs, reference classes/splits and32 finite values; test/holdout scores do not choose the gate.
- Current reserved membership stays outside fitting and normalization. Historical mixed-parent publisher val/test/blank gap-fill exposures and unresolved synthetic voice aliases remain disclosed.351 historical-gap records were actually sampled by the selected parent; filtering later replay does not erase ancestry. This is not proof that current held-out recordings entered fitting, nor a clean-source-lineage claim.
- All23,315 parent labels agree with frozen metadata rules, not independent listeners.288 human TRAIN references remain ASR-derived/unlistened;521 unresolved references remain excluded and review templates are not adjudications. No one has independently verified that every publisher prompt/transcript matches the audio.

## 5. Results the reviewer should challenge

| Human evaluation | Correct accepted / supported positives | Wrong accepted |
| --- | ---: | ---: |
|Exact93-definition wordings, holdout|58/62 (93.55%)|1|
|Exact93-definition wordings, test|173/232 (74.57%)|4|
|Broader supported complete commands, holdout|72/80 (90.00%)|3|
|Broader supported complete commands, test|393/609 (64.53%)|59|

Broader wrong accepts include negative clips: holdout1/14; test24/256 (plus35 wrong positive commands). Original .99/.30 gave3/80 and41/609 correct accepts with zero wrong accepts. Lowering the gate changes this tradeoff; it does not make softmax scores calibrated probabilities of real-world correctness.

Synthetic holdout99/100 and test3289/3368 are separate evidence; neither cohort supplies task negatives. All93 wording definitions have exact metadata evidence somewhere across origins, but only72 have exact human TEST/HOLDOUT matches;21 gaps remain. Exact metadata matching is not independent listening or a new blind test.

Known human holdout mistakes: “Play a song” -> NEXT;18°C ->22°C; true UNKNOWN “More heat” -> PAUSE. Three other exact-wording cases reject. Prefix/suffix/both one-second silence conditions each retain72/80 correct accepts with2 wrong accepts, not independent extra recordings. No noise/distance/live/wake/owner/Pi qualification is established.

## 6. What verification proves—and does not

The frozen replay has4,896 actual original/stress cases. Independent guards reconstruct exact canonical identities, raw/selected/fixed decisions, capture-fault rejection, complete-command metrics, declared/final-output hashes and all138 policy ranks;32 graph spot replays and399 historical bindings pass.161 focused software tests pass across12 modules. Software correctness and integrity do not establish adequate acoustic accuracy or device safety.

A real reporting failure is preserved: projected TEST/HOLDOUT manifests omitted full text, making the original372-cell wording table all-zero. That meant metadata unavailable in that projection, NOT absent corpus recordings. The separately versioned source-locator/SHA256-verified transcript-only addendum produces `per-verbatim-cohorts-v2.csv`; original source, table and receipt remain byte-preserved. Full-reference text is used only for offline scoring, never inference or fitting. The follow-up independent review passes; additional boundary/tamper fixtures remain nonblocking hardening, not completed claims.

The graph is1,082,940 bytes. Prior one-thread desktop p50/p95 4.91/7.70ms and peak79.75MiB are historical tiny-runtime measurements, not a fresh Pi/full-pipeline benchmark. The trained graph is the same; that does not authorize copying desktop timing into a Pi claim.

TEST/HOLDOUT were historically exposed through prior studies. The new policy is selected on DEV/calibration, but this evaluation is still regression evidence, not fresh-blind qualification. Repeating code tests or hashes does not undo historical exposure.

## 7. Git and review-archive boundaries

The user-authorized commit captures accumulated source, tests, compact study evidence, failed studies/reviews, configuration metadata and documentation. No raw recordings, feature tensors, caches, environment/credential files, new model installation or physical actions are added to Git. Older targeted-study metadata is retained as historical context and provenance, not represented as a new fit.

The separate ignored archive is planned at `dist/optionb-tiny-review-v1/optionb-tiny-review.zip`. It contains a focused committed source snapshot, this guide, the exact five-file candidate bundle, original selected checkpoint and explicitly listed cached split/normalization/score evidence. Its generated `REVIEW-MANIFEST.json` records the actual commit, entry hashes and exclusions. It is a review handoff, not a full raw-data distribution or portable source-only reproduction.

Cached manifests retain their original absolute source audio/feature paths and exact IDs. They are not repaired or rewritten to pretend missing payloads are present. Historical full verifiers need original external files/source shards at those recorded locations. The archive can support source inspection, fixture tests, model loading, new user-supplied WAV inference and cached-decision reconciliation; it does not provide fresh corpus audio inference, retraining or independently listened truth. The existing project environment is reused for extraction checks, not a claimed clean-machine install.

From the extracted `repository/` folder, with the existing dependencies available:

```
python -m scripts.runtime.speaker_optionb_tiny --bundle .cache/speaker-optionb-tiny-demo-v1/run-01/bundle --wav YOUR_MONO_16000HZ.wav --threads 1
```

Do not run immutable preparation/fitting/selection into existing run IDs. Do not lower thresholds from these test failures, enable actions, silently install the tiny model, treat ambiguous labels as true UNKNOWN or remove negatives to improve the score.

## 8. Independent critic prompt

> Review this as an independent ML and systems engineer, not as an endorsement exercise. The primary target is the258,912-parameter OptionB tiny epoch44 model with the separate.50/.10 demo policy, not the older2M submission. Treat prose as claims to verify against source, locked protocols, receipts, tables and actual archive contents. State clearly what you inspected or executed and which missing files prevent stronger conclusions. Do not invent measurements or retrain/tune on exposed evaluation data.
>
> Critique the entire chain: source labels and splits -> feature extraction/normalization -> batch sampling/augmentation/loss -> checkpoint selection -> posterior/margin selection -> ONNX/runtime -> capture/wake/VAD/session boundaries -> typed outputs/possible future actions -> evaluation/reporting. Distinguish implementation defects from weak data, acoustic errors, calibration limitations and integration gaps. Audit complete-command FP/FN semantics, per-wording denominators, human/synthetic separation, negative coverage, exposed-test history and inherited training lineage. Challenge any inference that a valid JSON command is the right spoken command.
>
> Assess whether the predeclared pooled-F1 demo objective matches the user's actual error cost, whether unseen-speaker reliability is supported, and whether research bookkeeping makes a simple model unnecessarily hard to run or audit. Account for disabled qualification/actions and the absence of a demonstrated tiny live/Pi pipeline. Do not assume optional shared-CNN keyword/CTC/ASR branches are active without tracing the actual config/forward call.
>
> Return: (1) severity-ranked concrete findings with file:line evidence, consequence and minimal correction; (2) a corrected architecture/data/runtime-flow explanation; (3) independently reconciled metrics or an explicit inability to recompute them; (4) which claims are supported/unsupported; (5) a small prioritized improvement/experiment plan within the roughly<=300k budget, preserving split integrity and failed evidence. Separate necessary fixes from speculative redesigns. Explicitly say whether this is a useful recorded-phrase candidate, an actually integrated live demo, and/or a qualified release—these are different conclusions.
