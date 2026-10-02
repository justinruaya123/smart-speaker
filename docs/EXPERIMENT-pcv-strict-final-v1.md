# PCV strict-final v1: architecture audit and executable preflight

**Status: 37 targeted tests passed; four architectures received untrained CPU cost probes. No new acoustic model was trained or qualified.**

Base: `codex/semifinal-handoff` at `7d8296c96e19e1e65fc3f0cd1b82d2391df33dbc`. Child: `codex/pcv-strict-final-v1`. This experiment preserves all installed artifacts and prior research. The newest user instruction supersedes the old root handoff's broader dataset acquisition and branch instructions for this study.

## Charter

Goal: choose an evidence-based path to a scratch-trained, offline, finite-command ONNX model on Raspberry Pi 4, with the class benchmark kept separate from custom-slot improvements. Scope: tiny spoken-language understanding. Object: the branch's data, encoder, finite decoder, calibration, benchmark and deployment. Subjects: the fixed 93 phrases, 31 semantic outcomes, the current context/CTC model, compact temporal and recurrent challengers, and the permitted sources.

Knowledge depth: working knowledge from the supplied run and source inspection; no access to the user's local checkpoint, recorded corpus or Pi. Questions: Q1 Are schema and training/evaluation contracts consistent? Q2 Is the existing CTC implementation mathematically wrong? Q3 What cost reduction can be demonstrated without weights? Q4 What evidence is needed to enable acceptance? Q5 What must remain for Codex?

The historical-experimental phase audits existing behavior; the descriptive phase specifies a bounded next experiment. This is not an exhaustive architecture search.

## Canonical sources

1. User's October 2 meeting summary and uploaded Option B CSV: intended scope and labeling. The CSV is retained at `experiments/pcv-strict-final-v1/option_b.csv`.
2. Exact smart-speaker commit above: `scripts/models/pcv_contextual.py`, `pcv_release_pooling.py`, `scripts/runtime/pcv_final_runtime.py`, `pcv_schema.py`, `experiments/phrase-contract-v1/pcvcm/decoder.py`, `scripts/train/pcv_release_training.py`, `pcv_release_prepare.py`, `pcv_context_training.py`, and release/source-card receipts.
3. Class benchmark at `ab39857cf73dd067d068fed388c321ab160dc191`: `vcmbench/schema.py`, `metrics.py`, README. It is one commit beyond the prior `17d2dfe...` pin. Comparison shows only `benchmark.py`, `tests/test_core.py`, and `vcmbench/report.py` changed; core schema, slot normalization and metrics did not.
4. Primary HF source: `airimonda/ai231-me2-voice-commands`. The branch preserves the known `6947f13073e57eb6ae67e7e2fc3680700b82aa13` benchmark snapshot. Live HF access failed in this session; **this is not a claim that the pin is the latest HF main**. Optional `martinnavs/ai231-fil-supplemental-data` was not downloaded or audited and remains disabled.
5. Original research and official documentation: MatchboxNet (Majumdar and Ginsburg, Interspeech 2020, arXiv:2004.08531); TC-ResNet (Choi et al., arXiv:1904.03814); BC-ResNet (Kim et al., arXiv:2106.04140); sequential semantic CTC/CTL (Cao et al., arXiv:2106.04660); speech-to-entities (Kuo et al., arXiv:2009.14386); SpeechRAG (Min et al., arXiv:2412.16500); ONNX Runtime threading and quantization documentation. Only source landing pages/abstracts and documentation were consulted, not new full-paper reproductions.
6. Class notes repository `roatienza/Deep-Learning-Experiments`: its 2026 index identifies CNN, RNN, Transformer, optimization, regularization, efficiency and ONNX-serving material. The linked lecture PDFs were not inspected. Its MNIST notebooks are pedagogical references, not evidence of speech-command accuracy.

## Findings ledger

| ID | Finding | Evidence | Status |
|---|---|---|---|
| F1 | The supplied schema expands to 93 phrases, 19 external intents and 31 distinct intent/value outcomes. | CSV expansion test and preflight JSON. | Newly computed |
| F2 | The current graph uses a stride-two convolution, six bidirectional attention/local-convolution/feedforward blocks, and pooled semantic heads. It is endpoint-only, not the old causal streaming CNN. | Pinned context/pooling/runtime source. | Inspected and shape-tested |
| F3 | Current human raw complete-command accuracy is 498/613; frozen checks retain 251/613. | User-supplied last-run report, primary test. | Historical, not replayed |
| F4 | Prefix CTC matches independent PyTorch CTCLoss on eight seeded cases and an exhaustive four-frame path enumeration, including repeated phones. | New targeted tests. | Newly executed |
| F5 | Adding blank-dominated frames can increase the existing frame-normalized CTC score without adding command evidence. | Score changes from -2.5765803 at 10 frames to -0.2342346 at 110 frames. | Newly executed mechanism probe |
| F6 | Removing rival catalog paths leaves retained exact CTC likelihoods unchanged but can increase the command margin. | Subset/alias tests and 1,031/415/93-path probe. | Newly executed |
| F7 | A 97,319-parameter TCN and 119,783-parameter CRNN pass shape, padding and gradient checks. | Candidate source and tests. | Implemented, untrained |
| F8 | At 200 input feature frames, model-only median PyTorch CPU time is 9.712 ms for the current large encoder, 4.265 ms for the small context encoder, 1.450 ms for TCN and 4.264 ms for CRNN. | One-core preflight, 20 repetitions after five warmups. | Newly measured; not ONNX/Pi |
| F9 | Even zero errors among 250 negatives gives a one-sided 95% upper FAR bound of 1.191%; 0/299 gives 0.997%. | Exact binomial calculation and unit tests. | Newly computed |
| F10 | Supplemental filename `train` does not override reserved `voice_split`; derived source parents need split grouping too. | Preserved source card; new fail-closed routing/component tests. | Inspected and contract-tested, not a fresh dataset audit |
| F11 | The release trainer calls WavLM feature preparation before its hard-label arm. The selected release is nevertheless identified as the hard-label model. | Release training source and release summary. | Inspected; orchestration dependency, not proof of pretrained student weights |
| F12 | The benchmark's command label function credits correct intent/slot against the truth variation; exact wording is a separate metric. Its slot table conditions on correct intent. | Pinned `vcmbench/metrics.py` and README. | Inspected |

## Scope filter

Discard treating KWS benchmark accuracy as evidence for numeric slots, arbitrary names, Filipino-accent generalization or strict OOS rejection. Discard pretrained SpeechRAG/SSL initialization as a deployment solution under this source-only contract. Keep architecture ideas, not their weights or outside acoustic datasets. The optional publisher-provided synthetic data is distinct from new custom generation.

## Object filter

Restrict F5 to a demonstrated score property: it does **not** establish why the eight reported negative clips fired. Restrict F8 to untrained PyTorch CPU cost and different output objectives; it is neither an accuracy ranking nor a forecast of ARM/ONNX timing. F4 validates CTC algebra on adversarial numerical fixtures, not acoustic recognition. The previous 6,768-clip result remains historical repeated-test evidence, not a fresh run here.

## Conclusions

### 1. Simplify the strict task before enlarging the acoustic model [F1, F2, F3, F12]

The strict catalog has 13 fixed intent outcomes and 6 slotted intents with 3 values each: 31 semantic outcomes. A 32-way command head (31 + OOS) can predict exactly the payload the official benchmark requires. Retain all 93 phrases for training coverage, phrase-level reporting and strict wording/OOS checks. Do not claim that 31 semantic classes prove rejection of every non-schema paraphrase.

The current 1,031-entry finite catalog includes extensions and training-derived phrases. Letting those compete in the strict profile spends time and creates routes to unapproved outputs. For the current model, a restricted catalog is a useful controlled diagnostic; it is **not** a licensed drop-in deployment because margins and OOS behavior must be recalibrated.

Use two explicitly selected profiles: `strict_class` and `extended`. Never infer the profile from a filename, benchmark process or expected label. Official scoring stays unchanged. Store strict-contract labels and any disagreements with publisher metadata as a separate view; do not rewrite evaluation truth to improve the result. The October 2 agreement and the stored card's allowance for “close” wording require an explicit discrepancy report.

### 2. Preferred compact candidate, not a claimed trained winner [F7, F8]

`StrictTinyVCM(kind='tcn', width=96)` is the first compact candidate. It has a 40-to-96 stride-two stem, six depthwise temporal residual blocks (kernel 7; dilations 1,2,4,8,16,1), masked layer normalization, attentive pooling directly from hidden features, a 32-way command head, a 94-way auxiliary phrase head and a 40-output auxiliary CTC head. The strict export exposes only command logits. Auxiliary heads are not required runtime decoders.

The CRNN challenger uses the same frontend and heads, with two bidirectional GRU layers, 48 units per direction. Padding is handled with packed lengths during training. It is also endpoint-only. This is a convolutional-recurrent model, not object-detection R-CNN. It might recognize ordered/long patterns better; the present cost measurements do not establish that advantage.

At 200 input frames, TCN has about 21.2 times fewer parameters and about 6.7 times lower measured model-only PyTorch median cost than the existing large encoder. At 800 frames the measured gap widens. ONNX fusion, quantization and ARM kernels can change these ratios.

| Model | Parameters | Median at 200 frames | Median at 800 frames |
|---|---:|---:|---:|
| Existing pooled context, width 192 | 2,066,831 | 9.712 ms | 50.477 ms |
| Existing pooled context, width 80 | 377,423 | 4.265 ms | 28.711 ms |
| New DS-TCN, width 96 | 97,319 | 1.450 ms | 2.707 ms |
| New CRNN, output width 96 | 119,783 | 4.264 ms | 16.426 ms |

These runs used random weights, batch one, PyTorch 2.10.0+cpu and Linux x86-64, with process affinity `[0]` and one PyTorch thread. They exclude frontend, decoder, endpoint and transport. They are not directly comparable to the user's Windows ONNX measurements.

### 3. Keep CTC as a bounded alignment tool; do not treat it as confidence [F4, F5, F6]

There is no evidence from these probes that the trie recurrence is wrong. Repeated-phone transitions, chunk equivalence, alias grouping and subset exactness passed. The larger concern is objective and decision design.

The stored score is `log P(phone_path | audio) / frames`. Appending blank-dominated frames divides a largely unchanged log likelihood by a larger denominator. Our numerical fixture changes from -2.5766 to -0.2342 without new command evidence. A fixed threshold can therefore respond to silence occupancy. The separate intent gate may still prevent acceptance; this probe did not execute trained runtime decisions.

For a strict direct classifier, fit a temperature and rejection rule on held-out calibration and validate on a separate confirmation partition. For the CTC reference, diagnose score versus speech occupancy, duration, explicit OOS/blank competitors and margin; do not lower a global threshold using the exposed negative test. Cropping/endpoint changes require their own replay and calibration.

The artificial 1,031-path decoder fixture measured 118.34 ms versus 11.02 ms for its 93-path subset, with zero change in retained scores. These are seeded random paths, not the actual catalog; this demonstrates scaling, not a measured 10.7-times production speedup.

### 4. Human complete-command accuracy remains the quality blocker [F3]

The user's previous primary-test result is 4,054/4,203 raw exact positives (96.45%). Human positives are only 613/4,203 (14.59%); human raw exact is 498/613 (81.24%) and correct human commands surviving frozen checks are 251/613 (40.95%). A rejection-only policy cannot raise a fixed candidate set's human recall above its raw 81.24% ceiling.

Priority failures from that report: CREATE_REMINDER 9/18 human raw exact; BRIGHTNESS 14/23; LIST_REMINDERS 11/18; PLAY_MUSIC 5/8; TEMPERATURE 19/28; VOLUME 79/107; TIMER 14/19. Small denominators are not reliable population rankings. VOLUME had mostly intent/operation errors, while TEMPERATURE and BRIGHTNESS had numeric-value errors. Training must distinguish those failure types rather than add one universal parameter model.

The reported 8/250 synthetic-negative fires remain a failure, not cured by the new unit tests. Silence-only acceptance being disabled is safe behavior but does not satisfy recall.

### 5. Dataset handling is part of the architecture [F10, F11]

Only the two user-approved HF repositories may provide acoustic data. Do not use old bootstrap profiles that separately pull SLURP, FSC, Speech Commands or other corpora. Those sources may already appear inside the primary publisher's approved snapshot; preserve that provenance rather than reacquire them independently.

Use the primary publisher's train recordings, supplied train-only synthetic negatives and optional `supplemental_synth` rows whose `voice_split` is train. Reserve entire speaker/voice and derived-audio components before deriving development/calibration/confirmation. Keep synthetic-negative source parents, clean/noisy pairs and cross-repository duplicates together. Exact PCM checks alone are insufficient for re-encoded or mixed derivatives.

Keep numeral clips auxiliary; do not label an isolated numeral as a full brightness/volume command. Missing phone pronunciations should mask the phone loss, not automatically discard otherwise valid semantic examples. Apply CTC length feasibility after stride two, including required blanks between repeated phones. Keep label uncertainty distinct from true negatives. Never use a model's own prediction as evaluation truth.

The hard-label training route must not instantiate/download WavLM. The inspected selected model is recorded as hard-label; the objection is the current orchestration dependency, not a claim that its weights are pretrained. The new strict protocol prohibits pretrained command teachers. Self-distillation from a source-only teacher can be a later measured ablation, not a prerequisite.

### 6. Slots and custom names need a separate evidence track

For the strict six slotted intents, a joint finite command head already resolves the allowed value; numeric regression is unnecessary and can produce invalid in-between values. For expanded brightness/volume 0–100, compare a conditional 101-way value head against a compact compositional numeric/phone CTC head on held-out values and speakers. A JSON range alone proves nothing acoustically.

For named lights, contacts and songs, use a finite pronunciation/alias catalog returning stable IDs, with explicit unknown-name and duplicate-alias rejection. Phone edit distance is a diagnostic/reranker, not a calibrated acceptance probability. Semantic embeddings do not solve acoustic name identity. Preserve numbers inside names, `to` versus `by`, on/off, up/down, missing units, negation and multi-command rejection. A known unavailable device can be recognized while action remains blocked.

Pure KWS is suitable for wake/fixed classes but loses relationships when used as an unordered bag of keywords. “Do not turn on...” cannot be accepted merely because “turn on” is present. Semantic CTC can preserve ordered typed tokens without producing a general transcript, but it still requires labeled value coverage. SpeechRAG's pretrained speech encoder/LLM retrieval and speech generator are outside this tiny source-only deployment contract.

### 7. Acceptance must have honest denominators [F9, F12]

Always report raw candidates, accepted exact commands, wrong accepted commands and rejections separately. Slot exactness conditional on correct intent is supplementary; end-to-end exactness counts wrong intent, wrong/missing value, rejection and timeout as failures. Report per-intent, per-phrase, human/synthetic, accent/source and value/target slices.

Use the untouched class acoustic pipeline as the official track. It includes laptop playback, room/microphone, wake word, endpoint, inference and authoritative output logging. Direct WAV replay and HTTP/WebSocket parity are different tracks. Log one authoritative event, never diagnostic candidates, with profile/model/policy hashes and timing definitions.

The stronger proposed assurance gate uses 95% one-sided bounds: precision lower bound at least 99% and FAR upper bound at most 1%, with human accepted exact recall at least 90%. It is a project proposal, not an asserted class requirement. Zero of 250 negatives is insufficient for the 1% upper-bound claim. Multiple TTS renditions of the same source are correlated and cannot simply be counted as independent confirmations. Report per-clip intervals plus speaker/parent-cluster intervals. Where the permitted data cannot support a confidence claim, report `insufficient_evidence`, not a pass.

## Next steps / Risks

The executable next steps and explicit implementation gaps are in `HANDOFF-pcv-strict-final-v1.md`. The main remaining work is **fresh acoustic training, calibration, waveform qualification and physical Pi testing**, not just packaging. No honest result from this chat makes those tasks completed.

Reproduce the completed tests:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q scripts/test/test_pcv_strict_final.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.test.pcv_strict_preflight --output .cache/pcv-strict-final-v1/preflight-new.json
```

The reference checks intentionally bind the parent source hashes. A source change requires a new audit version, not bypassing the check. The local fallback copies used in this chat were byte-verified and are not production dependencies; on the GitHub child branch the tests use the already-tracked parent files.

## Source links

- https://github.com/airimonda/vcm-benchmark/tree/ab39857cf73dd067d068fed388c321ab160dc191
- https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands
- https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data
- https://arxiv.org/abs/2004.08531
- https://arxiv.org/abs/1904.03814
- https://arxiv.org/abs/2106.04140
- https://arxiv.org/abs/2106.04660
- https://arxiv.org/abs/2009.14386
- https://arxiv.org/abs/2412.16500
- https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- https://github.com/roatienza/Deep-Learning-Experiments
