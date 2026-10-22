# OptionB tiny balanced demo v1

Date: 2026-10-23. Experiment: `speaker-optionb-tiny-demo-v1`.

**Decision:** use a separate, less conservative **0.50 posterior / 0.10 posterior-margin** demo gate with the unchanged **258,912-parameter epoch44** model. This materially improves recorded OptionB acceptance; it is not new acoustic learning, proof that all speakers/wordings work, or authorization to install or actuate devices. The versioned metadata-report correction and its independent review/closure are complete.

## Scope and selection

The user explicitly requested proceeding because `.99` was too strict and OptionB verbatims mattered. The [predeclared protocol](../experiments/speaker-optionb-tiny-demo-v1/protocol.json) keeps all **93 definitions /31 complete typed commands**, the original 32-way command/UNKNOWN head, frontend, ontology and exact train-only normalizer. No architecture change, optimizer update, new recording, relabeling or reserved-row admission occurs. ME-02/03/04/06/07/08 are addressed as a finite recorded-command research milestone; ME-05 physical Pi/live readiness is not established.

One global gate maximizes complete-command micro-F1 on **466 unique nonfitting human DEV/calibration references**: DEV256 (171 positives/85 negatives), calibration210 (100/110). All138 predeclared probability/margin combinations are considered. Ties maximize minimum split F1, pooled precision, pooled recall, then the higher probability/margin thresholds. No99%-precision feasibility requirement remains in this demo objective. Wrong accepted complete commands are both FP and missed true commands; negatives remain in selection. UNKNOWN winners always reject; there is no per-command threshold, inference-time text lookup, keyword rescue or unsupported-slot coercion.

The [policy selection](../experiments/speaker-optionb-tiny-demo-v1/policy-selection.json) has pooled194/271 correct accepts,40 wrong accepts,82.91% precision,71.59% recall and76.83% F1. These are development selection results, not test results. The [immutable pre-test lock](../experiments/speaker-optionb-tiny-demo-v1/selection-before-test.json) binds the unchanged graph and separate policy before this study's final inference. Cached source ONNX logits are used only after exact unique reference-ID, class, split and32-finite-logit validation. No test/holdout score selects the gate.

## Original-recording results

[Final results](../experiments/speaker-optionb-tiny-demo-v1/results.json) contain the fresh actual-graph replay. Denominators below are supported complete commands; wrong accepts also include negative clips in the scored cohort.

| Original cohort | Original .99/.30 correct accepts | New .50/.10 correct accepts | New wrong accepts | New negative accepts |
| --- | ---: | ---: | ---: | ---: |
| Human holdout |3/80|**72/80 (90.00%)**|3|1/14|
| Human test |41/609|**393/609 (64.53%)**|59|24/256|
| Synthetic holdout |Not the human baseline|99/100|1|No negative references supplied|
| Synthetic test |Not the human baseline|3289/3368 (97.65%)|14|No negative references supplied|

Human holdout has94 scored clips plus2 ambiguous unscored; precision96.00%, complete-command F192.90%. Its3 wrong accepts are2 wrong positive commands and1 original-UNKNOWN false accept. The broader human test has865 scored plus185 ambiguous unscored; precision86.95%, F174.08%, with35 wrong positive commands and24 false accepts (18/229 original UNKNOWN,6/27 outside OptionB). Original .99/.30 had zero wrong accepts in these two cohorts. The much higher acceptance comes with a measurable error tradeoff, not a free model improvement. Synthetic cohorts cannot establish human generalization or negative rejection.

All three dependent one-second silence conditions—prefix, suffix, both—retain72/80 correct human holdout accepts, with2 wrong accepts and1/14 negative accepts each. These282 perturbations are not independent human recordings. They do not prove wake/VAD, room-noise, owner-speaker or microphone robustness.

### Remaining wrong human holdout accepts

Publisher full-reference metadata and frozen typed labels, not independent listening:

| Full reference | Frozen reference | Accepted prediction | Source row |
| --- | --- | --- | --- |
|Play a song|PLAY_MUSIC|NEXT|primary/holdout-00000-of-00001/000000|
|Set the temperature to 18 degrees|TEMPERATURE18°C|TEMPERATURE22°C|primary/holdout-00000-of-00001/000126|
|More heat|UNKNOWN|PAUSE|primary/holdout-00000-of-00001/000195|

The first and third are outside the exact93-wording evidence subset, not deleted from the full task score. No case-specific rescue or post-test threshold adjustment follows these observations.

## All93 exact-wording evidence

The authoritative [metadata-enriched table](../experiments/speaker-optionb-tiny-demo-v1/per-verbatim-cohorts-v2.csv) retains **372 rows:93 definitions × human/synthetic × test/holdout**. Exact matching uses the whole publisher full transcript and definition text/spoken form, with the existing text normalization; no courtesy-word stripping, keyword matching, bucket/variation fallback or inference-time text access.

| Cohort | Definitions with exact full-reference evidence | Reference matches | Correct accepts | Wrong accepts | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
|Human holdout|62/93|62|**58/62 (93.55%)**|1|3|
|Human test|71/93|232|**173/232 (74.57%)**|4|55|
|Synthetic holdout|84/93|90|89/90|1|0|
|Synthetic test|84/93|2996|2920/2996|12|64|

All93 definitions have at least one exact metadata match across the four cohorts, but only72 have human matches across TEST/HOLDOUT: **21 lack exact human evaluation-reference evidence**. This is not absence of human training data, and synthetic references do not close the gap. Reference-match totals are wording-table counts, not a claim of independent listeners/speakers or newly verified transcripts.

The four exact human holdout cases not accepted correctly are `B1-058` (Temperature18 degrees), `B1-061` (Change the temperature to18 degrees), `B1-064` (Set the temperature to18 degrees; wrong22°C accepted) and `B1-086` (Reminder Study). The other three reject. Do not claim all93 spoken wordings are solved from the58/62 result.

### Preserved reporting failure and additive correction

The original tiny TEST/HOLDOUT projected manifests retain source locators and typed labels but **omit all full-reference text fields**. The first [372-row table](../experiments/speaker-optionb-tiny-demo-v1/per-verbatim-cohorts.csv) and [verification](../experiments/speaker-optionb-tiny-demo-v1/verification.json) therefore contain zero wording matches. This establishes **unavailable projected metadata, not absence of corpus recordings**. That table is superseded for wording claims, not overwritten; its canonical prediction/decision/digest verification remains applicable. The independent first review identified this medium reporting issue.

The additive `scripts/test/speaker_optionb_tiny_demo_metadata.py` joins all4614 original references using the full repository/revision/shard/SHA256/row locator. It reads only `transcript`, `transcript_source`, `is_synthetic` from the two exact pinned primary Parquet shards, verifies source hashes/provenance and copies references in memory. It does not decode audio, read variation/bucket/alternative-ASR fields, change labels or touch policy selection. Prompt-derived/ASR-derived publisher references remain metadata, not independently listened truth. The [wording-v2 receipt](../experiments/speaker-optionb-tiny-demo-v1/verification-wording-v2.json) binds the new table, driver, tests, both source shards and preserved v1 receipt; source/frozen bindings are checked before and after reporting. All old outputs and policies remain unchanged.

## Verification and artifacts

- [Data/input freeze](../experiments/speaker-optionb-tiny-demo-v1/data-freeze.json):399 protected historical bindings, plus new source/score/model identities; old studies, original .99/.30 and research .985/.30 policies, installed/submission models, source map and private settings remain unchanged.
- [Core verification](../experiments/speaker-optionb-tiny-demo-v1/verification.json): exact4896 canonical original/stress cases; independent reconstruction of raw, selected and fixed.95 decisions including capture faults; metrics/CSV/source hashes; all138 policy ranks independently reconstructed;32 fresh actual-graph spot replays. Normalizer, graph and ontology are byte-identical to the already-verified parent. This study does not refit normalization.
- **161 tests pass across12 focused modules**, including24 new selector/cache/wording/metadata tests. RED/GREEN failures were witnessed for missing selector, invalid scope/cache alignments, missing wording/report CLI and missing source-metadata enrichment. Fixtures are software evidence, not additional recognition samples. Log: `.cache/speaker-optionb-tiny-demo-v1/run-01/logs/optionb_demo_final_tests.log`.
- Actual WAV CLI repeats stored outputs with actions/model qualification false. Checks include both the wrong NEXT on row000000 and correct PLAY_MUSIC on row000004; a successful CLI is not automatically a correct recognition.
- Graph remains **1,082,940 bytes**. No new controlled hardware benchmark is claimed. Prior tiny desktop timings are reference only; this replay's per-cohort timings in `results.json` are not Pi/live/end-to-end measurements.
- Independent core review found the missing-wording-metadata issue. The [full original failed review](../experiments/speaker-optionb-tiny-demo-v1/independent-review-v1-full.json), earlier truncated finding and source transcript remain evidence. The full original verdict arrived after closure and confirms that metadata reporting was its only failure; it independently validates all138 operating points/414 metric sets and399 protected bindings. The [independent additive-fix/closure review](../experiments/speaker-optionb-tiny-demo-v1/independent-review-v2.json) passes: seven metadata fixtures independently pass,372 unique cells/denominators and source/provenance bindings agree, original v1 source/CSV/receipt remain unchanged, and no security/logic concern remains in the reviewed correction. Additional missing/null-transcript and loader-boundary tamper fixtures are nonblocking future hardening, not claimed completed. No commit or push.

Run from the project root using the existing environment:

```
.venv/Scripts/python.exe -m scripts.runtime.speaker_optionb_tiny --bundle .cache/speaker-optionb-tiny-demo-v1/run-01/bundle --wav YOUR_MONO_16000HZ.wav --threads 1
.venv/Scripts/python.exe -m scripts.test.speaker_optionb_tiny_demo_report verify
.venv/Scripts/python.exe -m scripts.test.speaker_optionb_tiny_demo_metadata verify
```

Do not rerun immutable selection or overwrite receipts. Reproduce selection in a separately declared run/version. The bundle is `D:/School/AI231/.worktrees/me2/smart-speaker-fresh/.cache/speaker-optionb-tiny-demo-v1/run-01/bundle` and is **not installed**. CLI/WAV/events remain non-actuating, model qualification false, qualified output rejected. No microphone, physical Pi, provider action, installation, dependency change or new training occurred.

## Limits and next evidence

These TEST/HOLDOUT recordings were historically exposed: this is regression evidence, not fresh-blind qualification. Original mixed-parent historical gap-fill exposures and unresolved synthetic voice aliases remain disclosed; later policy-only work does not erase ancestry. The earlier288 ASR-derived human TRAIN references and six missing human command cells are not independently resolved here.

The result supports a more useful finite recorded-phrase demo candidate, especially exact controlled wordings. It does not establish uniform unseen-speaker performance: human exact TEST recall74.57% and broader64.53% remain material limits. Next evidence should be separately collected original human recordings covering the21 exact evaluation gaps and independently listened wrong/near-miss cases, without reusing evaluation speakers/recordings for fitting or adapting this frozen policy to its failures. Live integration/actuation requires separate user authorization.
