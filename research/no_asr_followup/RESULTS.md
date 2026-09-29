# VCM-only follow-up: verified removal, Option B audit, guarded training

## Decision

Retain the existing project-trained `phone_s53` research checkpoint. No external
pretrained weights were downloaded or used, and no acoustic layer was added.
The new six-epoch first-convolution adaptation failed the development command
regression guard. It did not replace either the research model or the root v12
VCM. Automatic acceptance remains disabled in the research path.

The interrupted run had already completed the independent ASR removal in
`8607d1519228ae6d1d6f307956acbea3da121c48`. This follow-up starts from
`0268974d7e8f73bb1a6303dd1899e02164b5ec20`, on the same
`research/vcm-final-suite-visualizer` branch. Main is not modified.

## What is removed and what remains

The active notebook no longer registers the independent character-ASR model,
its runtime, its trainer, or its transcription validation. Obsolete ASR switches
are rejected. Unresolved custom requests cannot fall back to transcription.
The publisher allowlist contains exactly `bundle.json` and `vcm.fp32.onnx`.
Tests place a stale local ASR graph beside a VCM fixture and verify exclusion.
Historical reports are retained as historical, not rewritten as new measurements.

The previous bundle declared the removed graph as 5,319,980 bytes. This is a
packaging reduction, not a measured RAM or latency saving: the old fallback
could be lazy or disabled. Existing workstation ZIPs must be rebuilt.

| Path | Parameters | FP32 graph bytes | Status |
|---|---:|---:|---|
| Original v12 finite-head VCM | 398,422 | 1,647,732, from its metadata | Weights unchanged; original trained weights/audio absent here |
| Research causal phoneme CNN | 113,064 | 458,230, file verified | Retained, 32,064 bytes of neural caches |

These are separate alternatives, not two simultaneously loaded acoustic models.
The research model performs constrained phoneme recognition, not unrestricted
sentence transcription. Root runtime/publisher tests execute an explicitly
random-weight ONNX fixture; they do not establish trained v12 accuracy.

External pretrained systems may be isolated benchmarks, never VCM initialization
or packaged inference dependencies. New static regression checks reject known
external model-loading APIs in active sources. They are source-contract checks,
not a general security proof about arbitrary Python code.

## Option B: current location and pinned evidence

The supplied URL's directory moved from `MEX2/OptionB` to `MEX2/Data`.
The inspected new head is `79d7d524aee9b46a110ca87a36ddcc6bcbd37599`.
Reconstructing its Git tree from the old pinned tree while replacing only the
README blob exactly reproduces the new tree digest. The dataset descendants,
manifest, labels, slots and dataset-design file are unchanged.

Reference revisions:

- `markandrian30/AI231@ed245708c2573188eb769096aba76588f79b8de1`, old OptionB path.
- `markandrian30/AI231@79d7d524aee9b46a110ca87a36ddcc6bcbd37599`, new Data path.

The README's 150 reference speakers and 27,956 files describe a larger local
inventory. The committed manifest/tree/design instead contain **18,375 WAVs,
100 reference-speaker IDs, 19 intents, 32 intent/slot folders, and 96 texts**.

| Split | Files | Speaker IDs | Clean | Noisy | Unique texts |
|---|---:|---:|---:|---:|---:|
| Train | 14,677 | 80 | 7,344 | 7,333 | 96 |
| Validation | 1,856 | 10 | 926 | 930 | 96 |
| Test | 1,842 | 10 | 922 | 920 | 96 |

The independent re-audit found no duplicate manifest paths, missing/extra tree
WAVs, cross-split speaker IDs, cross-split clean/noisy parent groups, or
cross-split identical Git blobs. All 96 texts recur in every split: this does
not establish unseen-phrase, unseen-number or dynamic-name generalization.
Reference IDs do not by themselves establish independent human-command recording
or synthesis provenance. Exact hashes do not detect every near-duplicate.

I decoded and verified Git-blob and SHA-256 identity for 1,845 WAVs: all 1,787
brightness examples and 58 controls. The subset is mono, with 603 files at 16 kHz
and 1,242 at 12 kHz. No nonfinite PCM, exact PCM duplicates, audited clipping,
or duration-metadata errors above 1.1 ms were found. Durations span 0.76-4.52 s.
These rate counts describe this subset, not all 18,375 files. Recordings were
resampled as complete signals, not by relabeling their native sample rate.
This is not manual transcript adjudication.

No model architecture, Python trainer, training notebook, optimizer settings,
epoch schedule, or recognition result is present in the inspected MEX2 subtree.
The documented offline transcription-similarity filter is data preparation; it
does not imply the VCM needs an ASR runtime. The absent training recipe cannot
be inferred from the directory layout.

Brightness covers only 20, 60 and 100. Volume has up/down, not arbitrary numeric
settings. There is no UNKNOWN/background label, custom light-name list, or
song-title/BY/artist supervision. Treat the source as a finite-command control
and a possible training supplement, not the sole VCM dataset. No third-party
WAVs or failed transfer weights are committed or redistributed here.

## New bounded training experiment

Initialization used this project's own checkpoint, not an external pretrained
model. The architecture, normalization, phoneme vocabulary and 1,878-form V6
catalog stayed fixed. Only the first convolution's 7,744 existing parameters
were trainable; every other tensor was checked against the parent after each
epoch. The trial adds zero inference parameters.

Training used 1,427 external brightness recordings and 850 own synthetic training
recordings. Balanced batches paired up to 12 external examples with the same
number of replay examples; replay cycling is training repetition, not new data.
Development used 180 external and 149 own examples (128 valid own requests).
Own eSpeak audio was re-rendered and all 1,753 manifest PCM hashes matched.

Settings: seed 811 plus epoch for deterministic shuffles; six epochs; batch 24;
AdamW learning rate 0.0002, weight decay 0.0001; cosine minimum 0.00002; gradient
clip 3.0; CTC blank 0 with nonfinite/infeasible losses rejected. There was no
aggressive time/frequency masking of short numbers and no teacher network.

| Epoch | Own dev exact / 128 | External dev exact / 180 | Previously correct own commands lost | Eligible |
|---|---:|---:|---:|---|
| Parent | 114 | 1 | 0 | Retained reference |
| 1 | 83 | 0 | 31 | No |
| 2 | 97 | 1 | 18 | No |
| 3 | 99 | 2 | 18 | No |
| 4 | 99 | 1 | 18 | No |
| 5 | 98 | 2 | 18 | No |
| 6 | 98 | 2 | 18 | No |

External phoneme error improved from 87.88% to 76.14%, but own phoneme error
worsened from 25.80% to 46.23%. Those aggregate phoneme changes are not reliable
command improvements. Every new checkpoint was rejected before test evaluation.

The new paired-development guard requires identical recording IDs, PCM hashes,
reference commands, categories, and development partitions. It recomputes exact
commands and ambiguity rather than trusting saved accuracy flags. It rejects a
candidate that loses a previously correct replay command, even if net accuracy
or phoneme error improves. This is a deliberately conservative experiment rule,
not a statistical guarantee or a replacement for acceptance calibration.

A separate evidence bug was also fixed: Python regards `True == 1` as true.
The consolidated ledger now compares typed JSON commands, so a Boolean cannot
masquerade as a numeric parameter; nonfinite values cannot enter exact-match
scoring. All 36,972 existing evidence rows retain their prior correct flags.

## Fresh frozen-model ONNX Runtime replay

After selection remained at epoch zero, I replayed 934 recordings through both
40 ms and 80 ms ONNX graphs using ONNX Runtime 1.23.2 CPU. The V6 test recordings
are previously inspected regressions, not a newly sealed test. Neither test set
was used for this trial's selection, and rejected checkpoints were not tested.

| Dataset | Files | Valid | Complete raw matches |
|---|---:|---:|---:|
| Own V6 regression | 754 | 706 | 645 |
| External brightness test | 180 | 180 | 0 |

V6 category matches remain: numbers 375/404; relative 24/24; title plus artist
100/108; title-only 85/90; light names 34/40; named values 27/40. Its six ambiguous
and 42 negative cases are kept separate. Zero changed V6 commands/ties versus
saved predictions; zero changed frame paths or commands/ties between full
PyTorch and either cached ONNX Runtime graph. Maximum absolute log-posterior
difference was 0.00006866455078125. Nonzero caches were carried across chunks.

The retained model's 0/180 external result is a serious transfer failure, not a
judgment that the dataset is unusable. The earlier interrupted-run adaptation
improved that domain but regressed existing commands; its report remains in
`research/no_asr_review/RESULTS.md`. No candidate has earned global promotion.

Desktop neural-call-plus-cache timings: 40 ms graph median 0.0881 ms, p95
0.3072 ms, 50,527 calls; 80 ms graph median 0.0946 ms, p95 0.3294 ms, 25,497 calls.
These exclude frontend, decoder, endpoint waiting, and actions. They do not
measure Raspberry Pi speed, total pipeline latency, or runtime RAM.

## Verification and reproduction

450 Python tests, 18 JavaScript/WASM tests and eight Chromium application
scenarios passed. There were five nonfatal ONNX export warnings. No physical Pi,
physical microphone or pretrained model benchmark was used. The release gate
remains unqualified. Source-only GitHub CI does not replace full payload tests.

From this branch, with the earlier full research payload restored:

```bash
python -m pytest -q
node --test visualizer/tests/core.test.mjs
python final_suite/browser_check.py
```

For the trial, first re-render the existing V6 manifest's audio/features with
`experiments/vcm_services_v6/src/prepare_probe.py`. Supply the pinned Option B
subset folder, containing its hashed `manifest.json` and referenced WAVs. The
supplement contains the subset manifest and source metadata, but not the WAVs.
An offline materializer is included in the supplement for a local pinned source
checkout. Do not substitute a new dataset revision while keeping the old hash.

```bash
python research/no_asr_followup/experiment.py --audio /path/to/optionb-subset
for epoch in 1 2 3 4 5 6; do
  python research/no_asr_followup/experiment.py \
    --audio /path/to/optionb-subset --epoch "$epoch"
done
python research/no_asr_followup/verify_replay.py --audio /path/to/optionb-subset
```

`verify_replay.py` refuses to call a newly selected checkpoint the old model;
an eligible changed checkpoint must first be separately exported and bound to
its own graph metadata. All complete prediction rows and trial decisions are
in the downloadable supplement; their hashes are recorded in `reports/summary.json`.
No unsupported command becomes an executable action in this experiment.

## Training priority

Keep one small encoder and ONNX. Do not add a second recognizer to compensate
for the transfer failure. The next matched study needs broader, reviewed acoustic
training with connected 0-100 values, actual target microphones, numeric and
name contrasts, negation/extra-word negatives, and title/BY/artist tuples.
Compare from-scratch joint training and explicit own-checkpoint continuation on
the same development protocol. Keep source-speaker, clean/noisy parent, session,
and name/value holdouts distinct; publish complete-command acceptance and false
acceptance, not only phoneme error. Only advance a candidate after the retained
command regressions and independently calibrated human/Pi qualification pass.
