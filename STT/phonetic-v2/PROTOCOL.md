# Phonetic catalog and grammar study, version 2

Declared before fitting or scoring the new candidates, 19 September 2026.

The approved follow-up is an isolated implementation and measurement of the prior report's phonetic-keyword-plus-grammar recommendation. The installed VCM/STT graphs, shipping default, existing releases and historical study results remain unchanged. No synthetic or concatenated speech, pretrained acoustic weights, Spotify/playlist/filepath integration or Home Assistant integration is used.

## Data and split

Expand the original LibriSpeech train-clean-100 extraction from the publisher-listed archive, checking its published MD5 and recording SHA-256. Select at most 30 hours for acoustic training and one hour each for general-speech development/test by identifier hash. Dictionary coverage and 0.1–20 second duration are input eligibility rules. Hold out speakers; these internal splits are not the official LibriSpeech benchmark. Existing study speakers retain their prior split.

Preserve all previous study test recordings and their partition for a matched historical comparison. Retain the previously reserved names, and exclude occurrences from acoustic training, including the additional speech. Use the prior declared fixed/expanded catalogs.

Improve the connected-command experiment with a supplemental SmartLights speaker holdout. Among prior-training speakers with at least 25 numeric recordings, order by a declared hash and select one development and one test speaker, retaining at least one for training. Selection uses transcripts/counts, never predictions. Remove these speakers' recordings from all gradient training and enrollment. This supplemental evaluation includes speech previously seen by historical models: only new v2 models may claim generalization on it. The legacy test remains separately reported. All original close-microphone SmartLights recordings are considered; do not synthesize missing numbers. Other source splits are preserved.

## Fitting and selection

Fit two CNN/BiGRU phoneme CTC models from random initialization: width 64/seed 41 and width 96/seed 42, both two GRU layers. Maximum 36 epochs, two isolated-word warmup epochs, AdamW 0.0015 with cosine schedule to 0.00015, weight decay 0.001, clip norm 5, 64-example batches and patience 10 after epoch 15. Sample natural parameter commands with weight four and other training examples with weight one. Select the lowest unweighted mean of development phoneme error on word, parameter-command and general-speech groups. No waveform augmentation.

Evaluate two decoders per fitted model: the unchanged prior greedy phoneme/edit matcher, and a new CTC Viterbi phrase graph. The latter preserves frame probabilities, shares phrase prefixes, constrains natural numbers to brightness sentence templates, and uses keyword/filler matching for names. Numeric templates are declared text examples plus templates from training transcripts only. No template is mined from development or test.

Choose each decoder/catalog/group confidence threshold exclusively on development: at least 95% precision and 10 emissions, otherwise disable it. Connected numbers have an independent gate from isolated words. Lock graph, manifest, catalog, grammar and decoder hashes plus every threshold before test inference. Do not modify the decoder or training recipe after examining test results. Failures remain failures.

When the selected cutoff has a lower rejected development score, place the threshold at their midpoint. This preserves exactly the selected development decisions and avoids setting a floating-point cutoff directly on an observation. It is not test-based threshold adjustment.

CPU single/batch logit comparison uses absolute tolerance 1e-4 plus relative tolerance 1e-5; decoded parameter identity and gated decisions must additionally match. Before test scoring, a development-only diagnostic found a 1.26e-4 absolute difference at logit magnitude 17.3 with zero argmax changes. The relative tolerance accounts for FP32 scale; it does not relax the recognition-quality gate.

## Required outputs

- Split/source counts, duration, per-value connected-number coverage, duplicate/speaker/reserved-name audit.
- Raw and gated exact parameter accuracy separately for isolated words, connected commands, artists, songs and room-name proxies; negative false accepts and reserved-name expansion.
- Within the broad conditional numeric task, report text-valid absolute BRIGHTNESS requests separately. Volume, relative or unsupported utterances do not establish the absolute-brightness adapter's command support. Slot value accuracy is distinct from operation/intent correctness.
- The same new v2 models with fixed and expanded catalogs, without acoustic retraining. Also test retaining the fixed catalog's thresholds after expansion to show calibration drift; a shipping adapter requires fresh calibration.
- Legacy-test comparison against the prior phonetic reader. Supplemental holdout cannot fairly evaluate the old reader because it saw some of those speakers.
- Full CPU ONNX inference path p50/p95 on identical preloaded original command waveforms, two intra-op and one inter-op threads, one BLAS thread; graph/session/catalog construction reported separately. The isolated CLI pins BLAS/OpenMP before importing NumPy. No claim of Pi or live-microphone timing.
- The main timing sample is a fixed-seed selection of 48 natural parameter recordings plus every text-valid absolute brightness test recording. Report brightness-only timing separately, so name-query speed cannot obscure the cost of the numeric grammar.
- A separate 100/1,000/5,000-name timing experiment on eight original artist-query recordings, using generated dictionary-word text distractors. It creates no speech and supplies no large-catalog accuracy claim.
- Tiny-graph CTC checks against exhaustive path enumeration, repeated phones, ambiguous pronunciations, 0–100 typed grammar, catalog-version invalidation, VCM ownership and variable-length Torch/ONNX parity.

No reader is added to the shipping menu merely for being implemented. Promotion requires natural-command quality and coverage, unknown rejection, catalog-expansion checks and runtime integration checks; the prior requirement for an approved reader remains in force. This study still receives a slot role and an accepted intent; it does not establish automatic multi-slot music parsing or complete VCM accuracy.
