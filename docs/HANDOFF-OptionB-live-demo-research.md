# OptionB and unfamiliar-speaker live demo: research handoff

Prepared 2026-10-14 for GPT web/Pro review. This is an evidence-based diagnosis and bounded experimental proposal; metadata/parser controls were executed, but no trained-model improvement is claimed. Repository findings below come from the current `codex/speaker-demo-v1` source and hash-bound receipts. Existing dirty dataset-preparation work was read only. The precise checkout, ignored-data inventory, historical metrics, exact OptionB table and executed compatibility control are consolidated below.

## Review objective and final user decisions

Determine why the existing direct VCM misses intents or required slots across unfamiliar human voices, and what evidence would justify improving it. The supplied OptionB table defines the finite benchmark. Existing modifications and extensions define a **superset for personal-demo acceptance** and must remain preserved. Use one shared command model across speakers and across benchmark and extension commands. A 93-phrase-only replacement or separate male/female command models would not satisfy that decision. Report benchmark and superset coverage/results separately, preserving their respective denominators and versioned mappings.

The user confirmed **one command per accepted wake, then release ownership**. A previously unfamiliar person must be able to start a new session by saying the wake phrase. During that session, commands or slot fragments spoken by other people must not produce an accepted owner command. Distinguish rejecting a contaminated command from successfully recovering the owner's command during simultaneous speech; the latter is a stronger capability.

Governing constraints are in [REQUIREMENTS.md](../REQUIREMENTS.md) and [AGENTS.md](../AGENTS.md): recognition must run entirely on the RPi4/5, with finite command/slot decisions, no runtime ASR/STT/transcript parser, LLM or cloud model calls. Keep the 0.95 intent and 0.90 required-slot application gates unless a separately approved protocol changes them. An acoustic CTC head is assessed by what it computes and exposes, not its name alone. Count all shipped frontends, heads, decoders, catalogs, wake/VAD, speaker and possible filtering components. The existing pretrained wake/VAD exception and historical offline-teacher exception do not silently authorize new pretrained speaker weights. No physical device action is authorized by this document.

## Recorded outcome for the Pro reviewer

**The live demo is not qualified.** The strongest supported diagnosis is a combination of missing owner integration, incomplete human complete-command coverage, and benchmark-log/protocol selection hazards. Voice memorization, demographic imbalance, dominant label noise and insufficient capacity remain competing hypotheses. Adding sentences or assigning male/female IDs is not a demonstrated remedy.

The improvement investigation included a bounded **benchmark-log comparison**: unadapted diagnostic JSON correctly parses only **78/93** stipulated-correct fixtures, while the repository's **existing canonical benchmark adapter passes 93/93** against the current pinned parser and preserves rejection. This changes no model decision and measures no acoustic accuracy; reuse that adapter rather than implement a duplicate. Fresh metadata, owner-helper and adapter regressions pass **53/53**. Actual high holdout/testing accuracy, unfamiliar-person robustness and owner recovery during overlap still require acoustic evidence.

Prioritize resolving the benchmark decision/ownership contract and preserving the full demo catalog, then verify labels/crops/coverage before choosing a matched training ablation. The new pinned pool is available, but the trained shipping bundle is absent here; a fair baseline-versus-improvement acoustic run cannot currently be made from the retained metadata alone. This task did not train a substitute model or use historical test errors to select new weights/thresholds.

## Exact OptionB target and preserved superset

The following is the user's authoritative benchmark table. Fixed rows have three utterances; slotted rows expand each of three templates with each of three values. This is **39 fixed + 54 slotted = 93 phrases, 19 benchmark labels and 31 distinct typed outputs** under the existing runtime mapping. The one shared model has 18 semantic intents because VOLUME_UP/DOWN share VOLUME with different operations; OOV makes its intent head 19 outputs. None of these counts proves acoustic coverage.

| Label | Type | Variation 1 | Variation 2 | Variation 3 | Value 1 | Value 2 | Value 3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PLAY_MUSIC | Fixed | Play music | Start music | Play some music | — | — | — |
| WEATHER | Fixed | Weather | What's the weather? | Tell me the weather | — | — | — |
| TIME | Fixed | Time | What time is it? | Tell me the time | — | — | — |
| LIGHT_ON | Fixed | Lights on | Power on the lights | Turn on the lights | — | — | — |
| LIGHT_OFF | Fixed | Lights out | Kill the lights | Shut off the lights | — | — | — |
| PAUSE | Fixed | Pause | Pause audio | Pause song | — | — | — |
| STOP | Fixed | Stop | Stop playing | End playback | — | — | — |
| NEXT | Fixed | Next song | Skip song | Play next song | — | — | — |
| VOLUME_UP | Fixed | Volume up | Increase the volume | Turn the volume up | — | — | — |
| VOLUME_DOWN | Fixed | Volume down | Lower the volume | Turn the volume down | — | — | — |
| CALL | Fixed | Call | Make a call | Make a phone call | — | — | — |
| MESSAGE | Fixed | Message | Send a message | Send my message | — | — | — |
| LIST_REMINDERS | Fixed | Reminders | Show my reminders | List my reminders | — | — | — |
| TIMER | Slotted | Timer {duration} | Countdown for {duration} | Start a timer for {duration} | 10 seconds | 30 seconds | 1 minute |
| ALARM | Slotted | Alarm {time} | Wake me up at {time} | Set an alarm for {time} | 6:00 AM | 8:00 AM | 9:00 PM |
| TEMPERATURE | Slotted | Temperature {degrees} | Change the temperature to {degrees} | Set the temperature to {degrees} | 18 degrees | 22 degrees | 26 degrees |
| BRIGHTNESS | Slotted | Brightness {percent} | Adjust brightness to {percent} | Brightness level {percent} | 20 percent | 60 percent | 100 percent |
| COLOR | Slotted | Change color to {color} | Switch color to {color} | Set color to {color} | Red | Blue | Green |
| CREATE_REMINDER | Slotted | Reminder {task} | Remind me to {task} | Create a reminder to {task} | Drink water | Study | Exercise |

Preserve existing typed conventions: timer values become 10/30/60 seconds; alarm times become 06:00/08:00/21:00 with Asia/Manila; temperature remains integer degrees with the declared C unit; brightness remains 20/60/100 with `light.all`; color remains red/blue/green with `light.all`; reminders bind `task.drink_water`, `task.study`, `task.exercise`. Bare CALL retains the existing `contact.home` convention, and bare light commands retain `light.all`. These defaults are conventions, not spoken contact/target recognition. MESSAGE remains an intent-only compose/request-input decision, not dictated content. [Authoritative typed schema](../scripts/runtime/pcv_schema.py).

Personal acceptance retains all existing extensions: named lights, declared song/artist/contact/task IDs, wider brightness/volume values, supported operations/colors and the full versioned catalog and rejection cases. A benchmark adapter projects the actual accepted output into the benchmark's label/value names; it must not insert the expected answer, rescue a reject, restrict candidate competition to the correct phrase, or change the model when switching evaluation tracks. Both tracks bind the same weights, frontend, normalization, catalog and decision policy. Report OptionB results and extension results with separate denominators; do not blend them into one apparent benchmark score.

Fresh static comparison found `baseline_cases()` exactly matches all 93 pinned `variations.csv` rows. Current `baseline_cases()+extended_cases()` generates **414 definitions / 254 distinct positive typed commands**. The historical publication reports **415 definitions**; the preserved final-qualification catalog contains **968 acoustic entries** (including broader supervision aliases). These are different versioned objects, not interchangeable class counts. Resolve/bind the actual selected catalog before a new run; no definition or existing extension was removed by this audit.

## What is confirmed, and what remains a hypothesis

| Finding | Evidence and implication |
|---|---|
| The actual full-command runtime is an endpoint model with finite phone-CTC catalog scoring, an intent head and a named-target head. | [pcv_final_runtime.py](../scripts/runtime/pcv_final_runtime.py), `Runtime.recognize`, requires outputs `ctc`, `intent`, `target`, widths 40/19/2. `PrefixCTC` scores catalog phone paths and `rank_commands` returns typed candidates. The 19 intent-head outputs mean OOV plus 18 runtime intents; they are not 19 independently validated user use cases. Inspect the finite graph/catalog and output interface; do not describe this path as a transcript-then-parser system merely because it uses phones or CTC. |
| The source benchmark labels and runtime schema have explicit mappings. | [pcv_schema.py](../scripts/runtime/pcv_schema.py), `baseline_cases`, maps VOLUME_UP/DOWN to VOLUME plus operation. Bare CALL phrases have an existing `contact.home` mapping while typed CALL requires `contact_id`. Preserve that previously declared mapping; report its default semantics separately from an acoustically spoken contact. Do not invent a contact, silently remove the default, or count it as measured name recognition. |
| Application scores are not empirical accuracy probabilities. | [speaker_confidence.py](../scripts/runtime/speaker_confidence.py), `slot_confidences`, normalizes the best CTC likelihood per competing slot value within the selected intent. Its docstring explicitly says these are finite-catalog conditional scores, not calibrated empirical probabilities. Catalog coverage/competition and wrong-intent conditioning can change their interpretation. A score above 0.90 does not independently establish 90% slot accuracy. |
| A wake-owner policy helper exists, but the shipping path has no integrated trained speaker/overlap backend. | [speaker_session.py](../scripts/runtime/speaker_session.py) validates caller-supplied identity, voiced duration, overlap and continuity evidence. [speaker_worker.py](../scripts/runtime/speaker_worker.py) and [pcv_final_runtime.py](../scripts/runtime/pcv_final_runtime.py), `make_gate`, connect wake/VAD to command recognition without that policy. [The v1 handoff](HANDOFF-speaker-demo-v1.md) explicitly records the missing backend, calibration and capture integration. Passing numerical policy tests does not establish acoustic owner isolation. |
| Short speech and unknown continuity are explicit policy failure cases. | `SpeakerPolicy` in [speaker_session.py](../scripts/runtime/speaker_session.py) defaults to 0.8 voiced seconds for the reference, 0.4 for the command and 0.25 per continuity window; overlap `None` and missing required windows reject. Measure actual voiced wake/short-command durations before claiming these settings cover Hey Jarvis, Stop or Next. Silence padding cannot provide identity evidence. |
| Existing command-invariance and ownership objectives differ. | [speaker_demo.py](../scripts/models/speaker_demo.py) uses same-complete-command/different-speaker contrastive positives and explicitly forbids reusing its keyword embedding as speaker identity evidence. [speaker_demo_train.py](../scripts/train/speaker_demo_train.py) labels canonical complete typed commands, not broad intents with unresolved slots. The 22,027-parameter pilot is unqualified, has a different interface and cannot replace the required shared full-command model. |
| Eligible human coverage is incomplete before any proposed v2 fitting. | [PLAN-speaker-demo-v2.md](PLAN-speaker-demo-v2.md) and [data-audit.json](../experiments/speaker-demo-v2-data/data-audit.json) record missing human MESSAGE training, CALL/MESSAGE development and CALL/MESSAGE/LIST_REMINDERS/TEMPERATURE/CREATE_REMINDER calibration. Synthetic rows do not supply the missing human evidence. Exact-phrase/slot coverage must also be audited separately from intent presence. |
| Declared IDs/hash disjointness is necessary but does not establish independent voices or clean labels. | [speaker_dataset_install.py](../scripts/train/speaker_dataset_install.py) explicitly records unresolved speaker aliases and synthetic-reference identities; its audits detect exact file/PCM/declared-group overlap. Near-duplicates, independent listening and cross-source voice identity remain unaudited. |

**Not established:** gender imbalance as the root cause; accent as the sole root cause; voice memorization; adequate or inadequate model capacity; label noise as the dominant cause; clipping/endpoint truncation as the dominant cause; benefit from additional sentences, synthetic voices, contrastive loss or source separation. The inspected evidence supports investigating these hypotheses, not selecting one as a proven explanation. Historical aggregate accuracy cannot attribute a particular unfamiliar-person failure without the corresponding PCM, reference, capture stages and prediction receipt.

## Pinned dataset evidence and limits

The primary source is [airimonda/ai231-me2-voice-commands at revision 25111444af3adff7588d86ab27c304bb080895cf](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/blob/25111444af3adff7588d86ab27c304bb080895cf/README.md). Its card describes 10,682 train, 4,418 test, 196 holdout and 66,390 separate isolated-numeral clips; 93 OptionB variations; and declared speaker/synthetic-voice separation. The holdout includes real takes from one declared human speaker and synthetic fills. `speaker_id` can denote a human or synthetic voice; `accent_group` is source metadata. No gender column is defined. Some human references rely on automated transcription checks, with unreviewed train rows and explicit accent-mishearing exceptions; primary synthetic clips lack that check. These facts do not establish independent listening, demographic balance, broad unseen-human holdout coverage or connected-slot accuracy.

The supplement is [martinnavs/ai231-fil-supplemental-data at revision 230c8b8d58caf78e4bc90886d9c6fa77fe6024b8](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data/blob/230c8b8d58caf78e4bc90886d9c6fa77fe6024b8/README.md). It contains cloned speech from 17 reference voices, intended for training only. Gap-fill includes failed QA; eligibility is `qa_passed` or explicitly rescued QA. Reference-speaker overlap with the primary source, output accent authenticity and reference consent/license are unverified. Its reported improvement on one real Filipino holdout speaker also changed other training data, so the supplement's independent contribution was not measured. Treat it as a versioned augmentation arm, not 17 newly evaluated humans or wake-owner evidence. Its default, gap-fill and numeral-wording configurations must be distinguished.

Access verification: direct pinned HF blob/raw/resolve reads failed in the web reader; a read-only shell HTTP request also failed DNS. The existing local primary card [source-card.txt](../experiments/pcv-source-refresh-v1/source-card.txt) hashes to `fc5c2ca3b2a09c6f69f7edf8fcd178b7733d08a64509a0338375f0113d245877`, matching [ai231-primary.source.json](../data/sources/ai231-primary.source.json). The preserved supplemental [notice](../data/sources/fil-supplemental-NOTICE.txt) hashes to `46801041963a08d608293cfdc7bdee4be4c7c4ed4a7fe7017834c4f46af05af0`, matching [fil-supplemental.source.json](../data/sources/fil-supplemental.source.json). The current supplemental root card was also readable through the web tool. This verifies the locally retained pinned notices against receipts, not a new independent download of their revisions or a new corpus audit.

## Fresh read-only dataset findings, 2026-10-14

The full ignored inventory and every installed/verified view were hashed. The two inventory copies are byte-identical, and each authoritative verified view matches its saved audit SHA-256. All **97,659** referenced pool paths exist; their stat sizes total **4,682,018,344 bytes**. This task did not decode/listen to or rehash those WAV bytes. The earlier two installation runs decoded and hashed every recording and validated all 16 pinned Parquet shards (**4,137,836,220 bytes**); those prior tests are distinguished from this metadata pass. [Preparation audit](../experiments/speaker-demo-v2-data/data-audit.json), [prior verification](../experiments/speaker-demo-v2-data/verification.json).

The pool comprises 81,686 primary and 15,973 supplemental rows: **70,412 human + 27,247 synthetic**. Critically, **66,390 human rows are isolated numerals**, leaving 4,022 primary human non-numeral records before eligibility filtering. The raw inventory's `role="command"` also appears on numerals; use `reference_status` and the explicit auxiliary view, not that role or the pooled human count, to claim complete-command training. Do not feed this auxiliary pool into command fitting by accident or let its row count dominate command sampling.

Existing fitting exclusions are 504 ambiguous references, 529 supplemental failed-QA rows and 11 missing-speaker rows. Ambiguous evaluation records are retained and counted separately, not relabelled as true UNKNOWN. Eleven primary human training records carry `slot_metadata_conflict`; their cause/value validity still needs independent review. Metadata `transcript_source` counts are: label 59,276; dataset 37,471; whisper 496; prompt 324; folder_phrase 81; silence 11. These origins are not listener verification. No gender/sex field exists in the inspected metadata. Source-qualified individual IDs, pair IDs and declared PCM hashes have **zero cross-split groups** in the saved inventory; acoustic aliases, clone ancestry, near-duplicates, label correctness and true demographic balance remain unverified.

| Eligible view | Rows | OptionB typed outputs present / 31 | Exact metadata wordings present / 93 | Publisher variation labels present / 93 |
| --- | ---: | ---: | ---: | ---: |
| human/train | 1,895 | 25/31 | 27/93 | 61/93 |
| human/dev | 256 | 22/31 | 15/93 | 51/93 |
| human/calibration | 210 | 15/31 | 10/93 | 44/93 |
| synthetic/train | 21,426 | 31/31 | 93/93 | 93/93 |
| evaluation/test | 4,418 | 31/31 | 92/93 | 93/93 |
| evaluation/holdout | 196 | 31/31 | 92/93 | 93/93 |
| synthetic/diagnostic | 1,824 | 31/31 | 84/93 | 93/93 |
| auxiliary | 66,390 | 0/31 | 0/93 | 0/93 |

“Exact metadata wording” matches normalized literal/spelled-out baseline text, not an independently heard sentence; alternate legitimate pronunciations/wordings may be missed. A publisher's variation bucket can be populated by a paraphrase rather than that verbatim. Presence establishes neither sufficient distinct speakers nor accuracy. The mixed evaluation rows above do not constitute all-human evidence.

Eligible human train is missing MESSAGE, TIMER 10 seconds, BRIGHTNESS 60/100 at `light.all`, and CREATE_REMINDER Drink water/Exercise: **six of 31 outputs**. Human dev is missing CALL, MESSAGE, TIMER 10/30 seconds, ALARM 8 AM, BRIGHTNESS 60/100 and CREATE_REMINDER Drink water/Exercise: **nine of 31**. Human calibration is missing CALL, MESSAGE, LIST_REMINDERS, all three OptionB timer values, ALARM 9 PM, all three temperature values, BRIGHTNESS 60/100, COLOR Blue and all three reminder tasks: **sixteen of 31**. The presence of two calibration TIMER examples does not cover the benchmark durations; broader extension values can populate an intent without covering OptionB. These are measured metadata coverage gaps, not proof of why a particular spoken trial failed.

The eligible human fitting views contain 182/36/24 declared speaker groups for train/dev/calibration. These are declared IDs, not acoustically resolved people. Synthetic train covers all 31 outputs and 93 metadata wordings, but cannot substitute for the missing human development/calibration evidence or owner pairs.

| Preserved evaluation | Human rows | Human metric-eligible / supported / negative | Human OptionB outputs | Synthetic rows | Limitation |
| --- | ---: | --- | ---: | ---: | --- |
| test | 1050 | 865 / 636 / 229 | 31/31 | 3368 | Prior test exposure and programmatic references; no fresh model inference |
| holdout | 96 | 94 / 83 / 11 | 30/31 | 100 | Prior test exposure and programmatic references; no fresh model inference |

These denominators come from the **current prepared annotation/view version** and differ from prior acoustic reports' 609-positive/187-negative human test. Preserve both receipts. The primary source card describes one real holdout talker; additional declared holdout IDs are not independent-person verification. A high score on this household-sized holdout cannot establish “regardless of voice.”

A separate evaluation-only cache, `.cache/speaker-final-benchmark/`, binds primary revision `bfca0bbebfe17e00d6a685d1da573615d29ecdce`, not the preparation's `25111444...`. It has 4,895 rows (4,443 test + 202 holdout + 250 synthetic negatives), 920 human / 3,975 synthetic rows, and 4,894 referenced WAV paths, all present (323,369,340 row-summed stat bytes). One same-file OOV recording appears twice within holdout (`data_holdout/187` and `/188`); this is a repeated trial, not two independent negatives or evidence of cross-split leakage. Retain official denominators and separately report dependency-aware uncertainty. This cache excludes the historical publication's additional 1,873 supplemental evaluation rows; no 6,768-recording replay was newly run here. The source/rows/reserved receipts are in the ignored inventory below.

## Would more sentences or male/female voice IDs fix this?

Additional **original connected command recordings** can address measured coverage gaps if they add distinct people, supported values, phrasing and deployment conditions. More text alone adds no acoustic supervision. Repeating the same sentence/voice or expanding isolated-numeral counts can increase total rows without resolving unfamiliar-speaker connected-slot coverage. A controlled learning curve should vary unique training people separately from utterances per person; retain an untouched human evaluation denominator. This is an experimental recommendation, not a promised improvement.

Retain globally resolved **individual speaker IDs** for split integrity, sampling and per-speaker metrics. Male/female categories are not individual identities and cannot enforce wake ownership: two different people can share a category. Preserve supplied metadata and use verified voluntary demographic metadata only for assessment; unknown remains unknown. Do not infer sex, gender or accent from pitch, filenames or TTS voice names. Binary labels also omit variation within groups and voices outside those categories.

Separate male/female recognizers are currently unjustified, would fragment sparse data, require an additional routing decision and conflict with the user's one-shared-command-model decision. The bounded alternative is to measure shared-model errors by individual person, source, verified accent metadata and verified demographic groups, with phrase/value/microphone/duration coverage shown. A gender difference observed in an unmatched dataset could reflect those confounds. Source/voice balancing can be tested without claiming a demographic cause.

## Command invariance and owner identity require different evidence

For commands, the desired representation maps the same complete command spoken by different people toward the same decision while retaining distinctions such as 20 versus 60 percent, on versus off, and Drink water versus Study. For ownership, a representation must preserve the voice difference between two people saying the same phrase, while recognizing one person across a wake phrase and a different command.

Supervised contrastive learning pulls same-labelled samples together and separates different-labelled samples; its original published results are image benchmarks. Using same-command/different-speaker positives is a project-specific speech adaptation whose gain must be measured. Do not cite ImageNet gains as VCM evidence. [Khosla et al., Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362). Preserve the current complete-command labels: aligning all BRIGHTNESS utterances as identical positives while ignoring numeric values could discard information the slot task needs; that is a risk to test in a proposed full-model adaptation, not a defect established in the existing pilot.

TDNN/x-vector-style speaker verification uses statistics pooling to produce fixed-length embeddings characterizing the speaker; ECAPA-TDNN improves that speaker task through channel attention, aggregation and attentive statistics. This is an architectural reference, not evidence that its weights, VoxCeleb error rates, footprint or utterance durations transfer to this short-wake Pi demo. [Desplanques et al., ECAPA-TDNN](https://arxiv.org/abs/2005.07143). A separate small speaker encoder/head needs same-person/different-phrase positives and different-person/same-phrase negatives. Any shared frontend or multi-task trunk would need an explicit controlled comparison; one embedding should not be assumed to satisfy both objectives.

## Wake ownership is different from recovering speech during overlap

| Mechanism | Capability and boundary |
|---|---|
| Wake detector plus ordinary VAD | Detects a phrase and speech activity. It does not establish who spoke each later command fragment. |
| Wake-conditioned verification | Extract a trustworthy reference from the actual accepted wake, compare subsequent command-bearing speech to it, reject mismatches and release after one command. This can block a handover only if temporal coverage and identity evidence are valid. It does not remove an interfering voice from mixed PCM. |
| Personal VAD | Classifies frames as non-speech, target speech or non-target speech using a target embedding/verification score. Useful for temporal gating; those labels alone do not output a separated target waveform or prove correct slots during overlap. |
| Target-speaker filtering/extraction | Conditions enhancement on the wake owner's voice representation and attempts to suppress the other talker in waveform or acoustic features. It requires separate mixture/clean-target supervision, calibration, distortion checks and hardware cost evidence. |

openWakeWord's custom verifier is a speaker-specific logistic second-stage wake filter trained for known users/environments. Its documentation explicitly cautions that it is not intended as general speaker verification. Restricting wake acceptance to registered voices conflicts with the requirement that an unfamiliar person can open a fresh session, and it does not verify subsequent different-phrase commands. [Official custom-verifier documentation](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md).

Personal VAD motivates conditioned frame gating. Its paper evaluates constructed sequential speaker turns, uses a pretrained speaker encoder and does not establish this repository's overlapping-command recovery. The often-cited 130 KB conditioned VAD is not the whole wake-owner system: obtaining the reference embedding still has a cost. Do not import its forced-alignment/transcription tooling into the runtime or treat its training data construction as authorization to alter a human-only study. [Personal VAD paper](https://storage.googleapis.com/gweb-research2023-media/pubtools/5193.pdf).

VoiceFilter-Lite is a streaming speaker-conditioned feature filter designed for on-device ASR; its separation stage does not itself emit transcripts. It illustrates an architecture that could be researched upstream of a finite VCM without importing its ASR. The paper assumes the d-vector is already available without additional runtime cost, uses a pretrained speaker encoder and evaluates recognizer error rather than this project's typed commands. Its clean/noisy tradeoffs require measuring whether filtering corrupts correct VCM slots. Neither its real-time report nor model bytes prove Pi feasibility for the combined system. [VoiceFilter-Lite paper](https://storage.googleapis.com/gweb-research2023-media/pubtools/5748.pdf).

**Inference for this project:** verification/continuity plus overlap rejection is a defensible first diagnostic track, but it deliberately loses supported commands when others overlap. If the required demo must recover an owner command during simultaneous speech, a measured target-conditioned filtering/extraction track or another explicitly justified acoustic solution is still required. A mixed utterance can have an owner-dominated embedding while its recognized number comes from a bystander. Whole-utterance cosine similarity therefore cannot certify that every accepted slot came from the owner. Do not call the rejection-only track complete overlap support.

The session design should retain a bounded pre-wake PCM buffer, isolate the actual wake span, reject a contaminated/too-short reference, bind an immutable reference to a session ID, verify exactly the PCM consumed by the VCM and the entire command-bearing interval, enforce the conjunction server-side, and consume the session on completion/rejection/expiry or capture faults. Do not relearn the owner from an unverified command, let a second wake replace an active owner, or keep ownership after the one command. Existing [SpeakerSession](../scripts/runtime/speaker_session.py) implements portions of the state policy; producing valid evidence and integrating capture remain separate tasks.

## Bounded diagnostic experiments for the next authorized run

These proposals should be frozen as new/versioned protocols before fitting or viewing a new locked test. They have not been executed by this research task.

| ID | One question / controlled comparison | Evidence that would resolve it |
|---|---|---|
| D1 — stage attribution | Replay identical saved original command PCM through frozen runtime/engines; compare full command crop to the actual wake/VAD crop. Keep model, catalog, normalization and gates identical. | Per-recording reason: capture fault, wake miss, truncation, raw wrong intent, raw wrong slot, intent disagreement, confidence rejection, schema/catalog gap or owner rejection. Numeric features, logits and typed decisions locate the boundary. Restore the verified selected payload/environment first; metadata alone cannot run this experiment. |
| D2 — reference and catalog audit | Verify the OptionB benchmark mappings and the preserved extension contract, independently reviewing a bounded stratified development sample of label conflicts/errors. | Exact phrase/value/typed-command coverage, aliases, genuine OOV versus ambiguous supervision, defaults and competing catalog values. Preserve original test labels/results; any adjudicated scoring track gets its own version. Diagnostic annotation-supplied intent may localize slot failure offline, but must never rescue runtime intent. |
| D3 — voice/source assessment | Score the same frozen model on disjoint people, with per-source, verified-accent and verified-demographic summaries; keep phrase/slot/duration/capture cells visible. | Raw and accepted exact commands plus per-role confusions, macro-person recall, worst-person counts, false accepts and missing cells. Distinguish human/synthetic. Match or reweight common cells before interpreting a group gap as a voice effect. Unknown metadata is a separate stratum. |
| D4 — data quantity versus diversity | At fixed capacity/update budget, compare increasing utterances from existing training people with increasing distinct training people. Hold development/test people fixed. | Learning curves by unique people and recordings, source/phrase coverage and three-seed uncertainty. Unique-human benefit supports a diversity hypothesis; phrase-specific benefit supports wording coverage. Neither establishes overlap capability. |
| D5 — balancing versus invariance | Use the v1 matched arms: uniform CE, class/person-balanced CE, balanced CE plus exact-command cross-person contrastive loss; seeds 53/71/89, identical frontend/architecture/updates. | Development macro-person recall, per-slot separation, negative errors and usable cross-person-positive fraction. Run on the required full typed-command architecture before claiming full-model benefit. Keep the existing small classifier a pilot ablation. v1 selects raw recall; v2 proposes accepted recall, so changing selection is a declared protocol difference. |
| D6 — supplemental contribution | In a new v2 version, compare the same eligible human data with and without QA-eligible supplied synthetic augmentation; identical seeds, updates and evaluation. Preserve the plan's synthetic sampling cap and publish actual sampling mass. | Human results by person/intent/slot/source, synthetic results separately, worst-person regression, label-review exclusions and unresolved cloned-reference aliases. Gains on supplied synthetic clips or the already-examined single holdout person do not establish unseen-human generalization. Do not inject TTS into the frozen human-only v1. |
| D7 — ownership versus extraction | Separate command-only baseline, cropped genuine/impostor wake pairs, continuous handovers, overlap-reject track, and any later extraction track. Freeze speaker thresholds on different calibration people. | A-wake/A-command; unseen B-wake/B-command; A-wake/B-command; A-start/B-finish; A silent/B command; both speakers overlapping; noisy/clipped/short wake; timeout/rearm/overflow/stale dispatch. Include similar voices and same-command impostors, not just easily different gender pairs. For mixtures bind constituent PCM hashes, offsets/SIR/SNR and provenance, and verify both speakers actually speak during the labelled overlap. Real continuous sessions remain separate from constructed fixtures. |
| D8 — full Pi cost and retention | After parity/quality selection, measure the exact combined deployment on a physical RPi4/5 with the declared mic and threads, including clean, bystander and overlap scenes. | Load time, full endpoint-to-decision p50/p95, per-stage processing, real-time factor, peak RSS, CPU, temperature/throttling, bytes and genuine-command distortion/retention. Test network-disabled recognition. Report model compute, endpoint wait and utterance duration separately. Desktop/package tests remain desktop/package evidence. |

Do not change capacity, frontend, synthetic mixture, weighting, catalog and thresholds together and then attribute the result to one factor. Training loss versus validation loss alone cannot distinguish speaker leakage, supervision noise, missing command cells and deployment shift. Retain failures, inspect one boundary at a time, and predeclare what observation falsifies each hypothesis.

## Official benchmark compatibility and an ownership conflict

The official benchmark is additionally pinned at [`airimonda/vcm-benchmark`, revision ab39857cf73dd067d068fed388c321ab160dc191](https://github.com/airimonda/vcm-benchmark/tree/ab39857cf73dd067d068fed388c321ab160dc191). The existing local snapshot is `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/`. Direct GitHub revision reads failed with a web cache miss; the following findings come from the locally preserved source, not a fresh remote verification.

Its command score compares intent plus semantic slot, without requiring the predicted wording to match the spoken template. It reports human and synthetic tracks separately, Wilson intervals and timing from `infer_ms` / `audio_ms`; its ten holdout OOV clips are real, so they cannot establish a synthetic false-accept rate or a precise 1% OOV claim. Match this scoring contract while retaining the project's stricter typed-command/superset diagnostics. [Pinned benchmark README](https://github.com/airimonda/vcm-benchmark/blob/ab39857cf73dd067d068fed388c321ab160dc191/README.md).

The parser recursively searches nested dictionaries and joins slot-dictionary values. It does not require `accepted=true`; a rejected event containing `diagnostic_candidate.candidate` can therefore be interpreted as a command. Generic runtime diagnostic logs are unsafe as the benchmark decision stream. The repository already provides [pcv_benchmark_adapter.py](../scripts/runtime/pcv_benchmark_adapter.py), enabled in the worker by `PCV_BENCHMARK_LOG`, to project authoritative outputs separately from diagnostics. Fresh checks below confirm its 93-case compatibility; using the wrong log is a scoring risk, not a demonstrated defect in that existing adapter. [Pinned `vcmbench/pi.py`, `LineParser` and `_pick`](https://github.com/airimonda/vcm-benchmark/blob/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/pi.py).

The benchmark's aliases omit generic VOLUME, and its schema separates VOLUME_UP/DOWN. Reminder task IDs and multi-field alarm/target objects also need semantic projection to the benchmark's single slot. Map recognized typed values deterministically to benchmark labels/units/display text, preserving all existing richer slots in the personal-demo output. Never use the expected trial, phrase ID or ground-truth slot to fill a missing value, turn a rejected candidate into an accept, or hide an unsupported extension. Test all 93 definitions and abstention events against the pinned parser; fixtures establish serializer compatibility only. [Pinned `vcmbench/schema.py`](https://github.com/airimonda/vcm-benchmark/blob/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/schema.py), [slot normalization](https://github.com/airimonda/vcm-benchmark/blob/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/slots.py), [runtime schema](../scripts/runtime/pcv_schema.py).

**Protocol conflict:** the benchmark records the tester's wake takes and prepends them cyclically to prerecorded holdout commands. Its trial construction is `A.patch(takes[wi], cmd, gap)`, not a same-speaker wake/command pairing. Strict one-wake-owner verification should reject a positive benchmark command whenever that source voice differs from the tester. A command-only benchmark result and an owner-aware result therefore need explicitly separate, labelled tracks; owner-matched genuine wake trials require the corresponding source speaker's wake, or a new original-human paired protocol. Preserve and report the official mismatched-pair outcome rather than silently bypassing ownership to inflate acceptance. Repeated operator wakes also do not provide independent owner references. [Pinned `benchmark.py`, wake collection and trial construction](https://github.com/airimonda/vcm-benchmark/blob/ab39857cf73dd067d068fed388c321ab160dc191/benchmark.py), [audio patch function](https://github.com/airimonda/vcm-benchmark/blob/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/audio.py).

## Executed benchmark compatibility experiment

Source under test: official `vcm-benchmark` revision `ab39857cf73dd067d068fed388c321ab160dc191`, locally cached with exact file hashes below. Ninety-three fixture commands come from the existing `pcv_schema.baseline_cases`, matched to the official variations CSV. These fixtures are stipulated correct outputs, not audio predictions. The actual official `LineParser`, `resolve_prediction` and `slot_distance` were used without modifying the benchmark.

**Before:** a representative complete runtime JSON event, even with a correct candidate, parses correctly for **78/93** commands. Six volume phrases become `OTHER:VOLUME`; all nine reminder phrases expose `task.*` IDs instead of the reference task text. A deliberately rejected `PLAY_MUSIC` diagnostic candidate is still parsed as a command. In fixtures with total processing 6 ms and model-only 2 ms, the parser picks 2 ms; `audio_seconds` never populates `audio_ms` (**93/93 missing**). This reproduces a concrete integration risk if full diagnostic events are fed to the benchmark log; it does not establish that every prior scorer used that log format or invalidate their internal typed-command replays.

**Existing adapter:** `pcv_benchmark_adapter.log_record` matches **93/93** stipulated-correct fixtures with the actual current official parser/scorer, preserves all input typed decisions, and outputs measured-total and audio-duration fields correctly. A separate documentation-only serializer control also matched 93/93, but added no demonstrated capability over the canonical adapter and is not proposed as a second implementation. No source fix, training or installation was performed.

The canonical implementation's decision layer is important: `log_record` reads **`event.output`**, not `event.application`. A fixture with frozen `output.accepted=false` and demo `application.accepted=true` still logs OUT_OF_SCOPE. Therefore a benchmark watching that authoritative log currently measures the frozen disabled policy, even when the UI's explicitly selected demo rule accepts a command. Bind and name the intended decision mode before scoring; do not switch streams or promote diagnostic candidates after seeing results. Model weights remain shared. This is a confirmed interface distinction, not proof that the raw VCM is unable to classify the audio.

Another confirmed fixture limitation: the current adapter maps an `output.reason=capture_fault` event with timing into OUT_OF_SCOPE. Those faults must be identified separately by the benchmark/trial ledger rather than counted as successful negative rejections. A future bounded improvement should extend the existing canonical adapter/ledger with explicit fault and declared-decision-layer handling, versioned and regression-tested; it should not change the acoustic command, synthesize a missing slot, hide extension misfires or create another formatter. No claim is made that a real faulted negative trial was miscounted in past reports.

The existing adapter correctly translates VOLUME up/down and declared task IDs, emits UNMAPPED_VOLUME for unrepresentable operations rather than pretending they are correct rejections, and is wired to the opt-in worker log. Its task-name lookup currently uses `DEFAULT_CUSTOM`; dynamic task-catalog binding requires a compatibility check before claiming arbitrary editable task support. Preserve the entire richer command in the project's personal acceptance output. [Canonical adapter](../scripts/runtime/pcv_benchmark_adapter.py), [worker log wiring](../scripts/runtime/speaker_worker.py), [existing regression tests](../scripts/test/test_pcv_full_benchmark.py).

Fresh commands executed (root interpreter; no trained speech checkpoint loaded):

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_optionb_audit.py
# 4 passed in 0.08 s: metadata provenance/leakage fixtures.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider scripts/test/test_speaker_demo_v1.py
# 41 passed in 10.39 s: policy, sampling, model-padding and helper fixtures.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider scripts/test/test_pcv_full_benchmark.py -k 'adapter or live_log'
# 8 passed, 9 deselected in 0.20 s: existing authoritative adapter/log regressions.
.venv/bin/python /tmp/optionb-live-demo-audit-20261014.py
# Hash/count/path audit plus official-parser fixtures; completed successfully.
```

The last script/result and assembly helpers are temporary `/tmp` products, not repository implementations or sole durable evidence. Their verified counts, hashes and canonical-adapter findings are consolidated in this document. Existing tests create/decode temporary numerical WAV fixtures and exercise random numerical model gradients/updates; these are software checks, not corpus training or acoustic accuracy. No corpus fitting/feature extraction, ASR, held-out model inference or physical measurement occurred. Prior repository-wide collection/preparation failures (dependency/import collisions and frozen CRLF/LF identity checks) remain in the existing verification receipt; 53 focused passes are not a full-suite pass.

## Historical acoustic findings that must not be hidden

These are saved measurements, not fresh inference by this handoff. Test sets have already been examined; follow-up selection needs a new/versioned protocol and genuinely reserved people where appropriate. Raw ranking, a hypothetical locked policy point, frozen qualified acceptance and user-authorized demo acceptance are different outcomes.

| Model / track | Correct complete commands / positive trials | Accepted precision / OOV false accepts | Interpretation |
| --- | --- | --- | --- |
| updated-voice-v3 continued, human test raw | 398/609 = 65.35% | 398/742 = 53.64%; 145/187 OOV accepts | Saved raw candidates, unqualified |
| Same, human hypothetical policy point | 134/609 = 22.00% | 134/148 = 90.54%; 5/187 OOV accepts | Policy never enabled |
| Same, synthetic test raw | 3,227/3,368 = 95.81% | 3,227/3,363 = 95.96%; no supplied synthetic negatives | Synthetic result does not establish human robustness |
| final-qualification context128 INT8, human test raw | 473/609 = 77.67% | 473/724 = 65.33%; 120/187 OOV accepts | Development-selected candidate |
| Same, human hypothetical policy point | 184/609 = 30.21% | 184/187 = 98.40%; 0/187 OOV accepts | Frozen qualification remains disabled |
| Same, human holdout raw / hypothetical policy | 80/84 = 95.24% / 29/84 = 34.52% | Raw 80/84; policy 29/29 | Narrow, previously examined human holdout; no OOV denominator |
| Same, synthetic test raw / hypothetical policy | 3,286/3,368 = 97.57% / 1,788/3,368 = 53.09% | No supplied synthetic negatives | Do not pool with human claims |
| Refreshed release-v1 INT8, human published-test raw | 502/636 = 78.93% | Saved release receipt; qualification disabled | Different checkpoint/export from older context128 |
| Full benchmark INT8, expanded primary human raw | 498/613 = 81.24% | Saved human stratum; qualification disabled | Pooled mixed-source score must not hide this |
| Historical shipping ncnn, primary test raw | 4,051/4,203 = 96.38% | 4,051/4,290 = 94.43%; 102/139 OOV accepts | Primary source mixes human and synthetic |
| Same, demo .95 intent / .90 each slot | 3,652/4,203 = 86.89% | 3,652/3,687 = 99.05%; 10/139 = 7.19% OOV accepts | Actual demo application rule in saved replay |
| Same, primary holdout demo | 142/186 = 76.34% | 142/143 = 99.30%; 0/16 OOV accepts | Mixed-source holdout, no live owner gate |
| Same, supplemental test / holdout demo | 1,513/1,612 = 93.86% / 219/261 = 83.91% | 99.47% / 99.55%; no OOV rows | Supplemental synthetic evaluation is separate |
| Same, synthetic negative demo | No positives | 13/250 = 5.20% OOV accepts | Rejection gate still fails |
| Frozen qualified mode in these unqualified runs | 0 accepted | Precision unavailable | Zero false accepts from rejecting everything is not qualification |

Sources: [updated-voice-v3 comparison](../experiments/pcv-updated-voice-v3/comparison.json), [final qualification](../experiments/pcv-final-qualification-v1/results.json), [shipping publication verification](../experiments/speaker-publication-v2/verification.json), and [refreshed release/full-benchmark entries](../EXPERIMENTS.md). Older [semifinal-v1 replay](../reports/semifinal-v1/summary.json) has 2,752 raw correct over 3,793 valid requests but zero qualified accepts; it is a different retired diagnostic bundle, not the current full-command baseline. The earlier [human-slots qualification](../experiments/pcv-human-slots-v1-qualification1/results-human.json) explicitly found only 21/93 exact phrases with original-human evidence and remained unqualified. Do not replace these results with software fixtures or remove failed runs.

The saved ncnn shipping replay used 6,768 recordings on Windows/i9-14900, logical CPU 0, one thread: full frontend/model/finite-decoder p50/p95 **44.48/115.60 ms**; RTF p95 **0.02710**; whole-process peak RSS **130.34 MiB**. Median model compute was 5.59 ms versus 37.91 ms finite decoding, so decoder/catalog cost matters in addition to encoder size. It excludes disk, live capture, wake, endpoint wait and UI. There is **no physical Pi/live-microphone measurement** in that receipt, and provider actions were mocked. Graph parity, repeated inference stability and packaging do not certify live quality.

Historical final-qualification selected context128 has 935,485 parameters; FP32/int8 exports 3,839,438/1,558,203 bytes, INT8 SHA-256 `0b2701bacaeff770d0098433e018123b4f6d607e9d8a893e251dbbafca6b3aef`. The shipping publication is a **different selected checkpoint**: checkpoint `c471f68f389bb01fa6da1c664d1878a25f0892397200a35a0d664b76b2cb468b`, FP32 ONNX `588f5f44f2b8bca9f68be605bc3a0be2be36159477c6002f21708824503c8421`, ncnn BIN `a0623cc31f9bd9d6197cdffd28895b132705356d276f32db7e455108d9c0a20c`, PARAM `8e132ac2c91746c12915e06edfdf4b421de37adb225a3298d7ec891a8e4a13ee`. Bind the correct selected bundle; do not mix checkpoint, export or policy identities across these reports.

## Checkout, improvement preflight and preservation

Audit ID: `optionb-live-demo-research-handoff-v1`, an observational ME-01/03/04/05/06/07/08 review covering the supplied benchmark and preserved project-demo use cases. Baseline: `codex/speaker-demo-v1`, HEAD `5d872e3a249e5876aaa8130b76f6e3b31622355f`. No paths were staged at task start. Pre-existing work: modified `EXPERIMENTS.md`, `scripts/modules.json`; untracked source receipts, `docs/PLAN-speaker-demo-v2.md`, `experiments/speaker-demo-v2-data/`, installer and installer test. This task edits only this handoff, an additive navigation entry in `EXPERIMENTS.md`, and a dated scope/evidence entry in `REQUIREMENTS.md`; it preserves those other owners' work.

The root `.venv/bin/python` actually imports Python 3.13.9, PyTorch **2.10.0+cpu**, ONNX Runtime **1.23.2** and NumPy **2.3.5**. CUDA is unavailable and ncnn is absent. The earlier dataset verification recorded missing torch/ORT at that earlier time; an initial system-Python discovery probe in this task was also not the root environment. The final root-interpreter imports above are the authoritative current preflight; no dependency installation/environment replacement was performed.

No ONNX/checkpoint/ncnn graph payload exists under current `artifacts/` or `models/`, `models/default` and `dist/` are absent, and `.cache/speaker-demo-v1/` is absent. `models/selected/` contains older **semifinal diagnostic metadata only**, with recurrent state shapes; it is incompatible with `pcv_final_runtime.verify_bundle`'s complete-utterance interface. Root artifact metadata is retained but cannot substitute for missing graphs. Do not rename/copy this older bundle into the current default or claim a baseline replay. Thus no honest full-model acoustic improvement comparison was possible here without first restoring the source-bound shipping checkpoint/bundle and resolving fitting/qualification coverage. The optional improvement investigation was limited to metadata/coverage and benchmark-log controls; no candidate was trained/exported/installed.

Ignore-first checks passed for intended cache/audio/weight outputs and the new handoff/source/compact-audit paths remain visible. Existing `.gitignore` also globally ignores model files and `data/**/*.jsonl`: a tracked checkout alone omits essential graphs, row manifests and predictions. Report this reproducibility gap; no ignore rule, staging, branch, commit or cleanup operation was changed to hide it. Exact payloads below remain local. No shared recording was moved/rewritten, no experiment process was interrupted, and no microphone/device was operated.

## Evaluation quantities that must stay separate

Report correct accepted complete commands divided by **all supported trials**, accepted-command precision, raw exact commands, per-intent and per-role confusion, true OOV false accepts, capture/wake failures and schema/metadata-ambiguous rows. Measure both the OptionB benchmark and personal superset. Do not count errors as valid negative rejections, missing predictions as successes, or a parser definition as a spoken trial.

For ownership, report wrong-speaker accepted commands divided by impostor-session trials; genuine-owner accepted exact commands divided by genuine-session trials; and retention relative to otherwise-correct command-only accepts. Show the full wake-to-command session denominator in addition to conditional crop-pair metrics. Group results by reference duration, command duration and interference condition. Record unique people, unique recordings and repeated pairs; many combinations of the same people are dependent observations. Use uncertainty that resamples people/session clusters, rather than presenting every repeated pair as a new independent person.

The v1 protocol's 95% human accepted recall, 99% accepted precision, 90% worst-person recall where adequately sampled, at most 1% observed wrong-speaker acceptance and 95% genuine retention are **proposed pilot targets**, not measured performance or official assignment limits. Its 20-new-person holdout and 1,000-positive/10,000-impostor example are planning targets, not acquired data. [Frozen v1 protocol](../experiments/speaker-demo-v1/protocol.json). Final resource/quality acceptance also needs the complete benchmark and physical Pi evidence required by [REQUIREMENTS.md](../REQUIREMENTS.md).

## Ignored-data inventory and transfer boundary

This is the complete bounded inventory for data used/referenced in this handoff's in-scope preparation, historical comparisons and metadata/parser tests. “Content/hash read” means metadata/source bytes were read here; “prior verified” means a preserved acquisition/audit receipt supplies the payload digest. WAV payloads were only stat-checked here unless a small historical fixture is explicitly marked. Incidental interpreter/bytecode/pytest temporary caches are tool products, not research input; no credentials/private environment file was read.

**Recording groups:** `data/audio/ai231-pool-v1/` contains the 97,659 hash-addressed original WAVs indexed one-by-one by `verified/inventory.jsonl` (each row records audio path, original-file/decoded-PCM SHA-256, source/shard row, identity, split, typed reference, QA and flags). Its hash is `0c71cdc8347b8c9d01075f7cce6349c4eb4bfe2666653c4f149847ca8b2ded9a`. The 4,894 evaluation-only WAV paths under `.cache/speaker-final-benchmark/audio/` are indexed in its `rows.jsonl`; 4,895 rows intentionally retain the repeated holdout file. No new copies were created. A single useful handoff should not expand 97,659 filenames into narrative prose; these hash-bound row inventories identify the full sets, and all diagnostic counts needed for this review are embedded above. The web reviewer does not automatically receive ignored payloads or local link targets from Git.

### Present source shards referenced by preparation

All 16 exist at the listed sizes. Digests below are from the previously completed source verification; no new 4.14 GB shard hash pass was performed by this handoff.

| Ignored path | Bytes now | Prior verified SHA-256 |
| --- | ---: | --- |
| `.cache/pcv-source-refresh-v1/source/data/holdout-00000-of-00001.parquet` | 14,553,313 | `9917e336ec596eaaddb476e266f36451326e3f8e5a2cb3dafbb675c33be133aa` |
| `.cache/pcv-updated-voice-v1/source/data/numerals-00000-of-00006.parquet` | 321,316,611 | `cefd4884acf3aec7c33625dc46a970d7b5b921d0507e7b6e41bd430a3eb87504` |
| `.cache/pcv-updated-voice-v1/source/data/numerals-00001-of-00006.parquet` | 326,077,666 | `dcd8edade416bba1553d7d50db3bdfcacd8ac4aaad54b775316990741168a8a8` |
| `.cache/pcv-updated-voice-v1/source/data/numerals-00002-of-00006.parquet` | 321,668,264 | `768f4d9c426668831ddbece9d496c161ac3149ba1ab16bec722d1d0fd35592c8` |
| `.cache/pcv-updated-voice-v1/source/data/numerals-00003-of-00006.parquet` | 347,705,349 | `d9a58e4d10f625f170743f1623b7750f2933d8e6a2902ab99b634488a6041ec7` |
| `.cache/pcv-updated-voice-v1/source/data/numerals-00004-of-00006.parquet` | 348,820,081 | `b057c8930fa00f52481007be50e15382c4b68b9044a46cda1e923b295235f281` |
| `.cache/pcv-updated-voice-v1/source/data/numerals-00005-of-00006.parquet` | 349,447,467 | `aced4a5f0588d285df4155cb2da66ba327c1d35336079228a31f05a142bf315f` |
| `.cache/pcv-source-refresh-v1/source/data/test-00000-of-00001.parquet` | 284,419,574 | `81794982a6b7fd8c75758ad5f4c0257abedea6c1c0575bbaaabcba8e0e0a43ea` |
| `.cache/pcv-source-refresh-v1/source/data/train-00000-of-00002.parquet` | 315,293,385 | `977e6c6be208818a35496ac6e1a2b2da62f3387acba377664e0f48dbdc2cec9e` |
| `.cache/pcv-source-refresh-v1/source/data/train-00001-of-00002.parquet` | 397,331,137 | `4c29353d0ab14bb27fd3af8b513c2b9dd41f7707e86fbaa55104bfe71a328674` |
| `.cache/fil-supplemental-v1/source/data/train-00000-of-00004.parquet` | 244,957,155 | `01d32c3ed7b2e9a0ee131a1dad128c8e63e7b20a198ce1170625bd0940db0778` |
| `.cache/fil-supplemental-v1/source/data/train-00001-of-00004.parquet` | 276,168,578 | `c66da6df2579243d27fb956ddbc413f2e2ed0a4c5acaa77c333e905f866d885a` |
| `.cache/fil-supplemental-v1/source/data/train-00002-of-00004.parquet` | 235,971,285 | `8a7e0460893a97cb4e3ad4b3729ae781a277dcbccced7484898e2f5985122566` |
| `.cache/fil-supplemental-v1/source/data/train-00003-of-00004.parquet` | 235,454,295 | `0d31aa80f447423413fa097567c8df9cbdaa70439774d43df209c8ffc47df257` |
| `.cache/fil-supplemental-v1/source/gap_fill/train-00000-of-00001.parquet` | 118,240,931 | `1e229f67df1b539711886e01062dbc0309c58f38e739dad787b11c669d314512` |
| `.cache/fil-supplemental-v1/source/numeral_wordings/train-00000-of-00001.parquet` | 411,129 | `9808c05811bc0e3bdd152c74c4473673b1355e2dc355873ceb8bb48936a5ac5a` |

### Other present evaluation-source shards

These are the separate `bfca0b...` evaluation-only source and training-coverage receipts; they are not additional v2 fitting inputs. Stat sizes are current; expected digests are supplied by their preserved source receipts.

| Ignored path | Bytes now | Receipt SHA-256 |
| --- | ---: | --- |
| `.cache/speaker-final-benchmark/source/data/holdout-00000-of-00001.parquet` | 15,363,388 | `c36227c81eee4eac63de62cee94631ee9e73952ccf8e16c27912eb049508eb6e` |
| `.cache/speaker-final-benchmark/source/data/test-00000-of-00001.parquet` | 272,526,000 | `40e9b66b4e12bc5b012bfa2dc7367602740ac394467e48d4924364c46f912f32` |
| `.cache/speaker-final-benchmark/source/synthetic_negatives/test-00000-of-00001.parquet` | 17,411,607 | `b861abd530bbf5e15e6a9f61caca8a029a2867e33c9397317ac5e8f01b74b9f4` |
| `.cache/speaker-final-benchmark/source/data/train-00000-of-00002.parquet` | 295,319,324 | `f4b0717e95dc1a7fa5566be6a20212c04b554c7f69de8815646b36606a30bd83` |
| `.cache/speaker-final-benchmark/source/data/train-00001-of-00002.parquet` | 391,176,249 | `5011a4ff41c99b32acf07b3e5ea23ba36b5c03bf286b0731a0f85b5b4d9b5669` |

### Present ignored metadata, views, source and test evidence read or hashed

The paired `installed` and `verified` inventories/views are retained separately as two-run evidence; equal hashes do not authorize deletion. Small source-card/benchmark copies are preservation/inspection inputs, not competing newly authored handoffs.

| Ignored path | Bytes | SHA-256 now | Access |
| --- | ---: | --- | --- |
| `.cache/dataset-pull-2026-10-13/README.md` | 7,802 | `fc5c2ca3b2a09c6f69f7edf8fcd178b7733d08a64509a0338375f0113d245877` | content_read |
| `.cache/dataset-pull-2026-10-13/pinned-acquisition.json` | 2,616 | `994798b431f9d376f70fc74f2e4758a2d5fa630d384200131f5481578955cb0a` | content_read |
| `.cache/dataset-pull-2026-10-13/pinned-protocol.json` | 3,996 | `283544d5a23685e9a5dcb9ced24181cd06c8c145a6188b10f7d9994eb581323e` | content_read |
| `.cache/dataset-pull-2026-10-13/receipt.json` | 4,404 | `fa4e6f1f067a47424ab2bd29ba4a5b8e42b482d64ae0c27326c4c3195d07c7a8` | receipt_or_source_hash_read |
| `.cache/dataset-pull-2026-10-13/remote-tree.json` | 4,420 | `94a7d576b620cc73d851dc96997528282dd313b76c009a42700159a21e5de781` | receipt_or_source_hash_read |
| `.cache/dataset-pull-2026-10-13/variations.csv` | 3,433 | `6004b3b2b58137058c705b5769b0d094e05a5cff6ddcbed6de548b4a0105fcda` | content_read |
| `.cache/fil-supplemental-v1/acquisition.json` | 3,762 | `c791ae4d2627245c758ec96183736aaf454d41d2354ab59aedd07501d5911200` | content_read |
| `.cache/fil-supplemental-v1/publisher-metadata.json` | 4,486 | `42b45f12d4de9eeb6b97858935a258c8bf9e7e2aa579296075af46c9ad9d82e0` | receipt_or_source_hash_read |
| `.cache/fil-supplemental-v1/source-card.md` | 6,052 | `46801041963a08d608293cfdc7bdee4be4c7c4ed4a7fe7017834c4f46af05af0` | content_read |
| `.cache/speaker-dataset-v1/atomic-test.wav` | 48,342 | `000028cfd91f1c72ac11dcebcc38412d538f0b4f35f3bbab39d21b19b4dc6c94` | historical_fixture_hash_only |
| `.cache/speaker-dataset-v1/full-pytest.log` | 11,134 | `7ac598ed56bec58d14e823004f54da2cb391a0d59920aab7d1902937d37aeb4e` | receipt_or_source_hash_read |
| `.cache/speaker-dataset-v1/installed/audit.json` | 8,825 | `a71fa882ba5c40c4f0361ce45939f8c3a294a36f42261f9c580416b3a8fe9298` | receipt_hash_only |
| `.cache/speaker-dataset-v1/installed/environment.json` | 433 | `068852ee02860c88856293bfc6018068046c137af60dd693ca4852f8cd27247e` | receipt_hash_only |
| `.cache/speaker-dataset-v1/installed/inventory.jsonl` | 183,551,765 | `0c71cdc8347b8c9d01075f7cce6349c4eb4bfe2666653c4f149847ca8b2ded9a` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/auxiliary.jsonl` | 120,116,827 | `0dca5439d8ac063b207709ebfc2ff933ffbfa6e70feb18a6d81eaae2d52d4521` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/evaluation/holdout.jsonl` | 379,782 | `a51e720b97f7b31e3e2f025c21bf679006ad444fd2d363916d9a11aa0f488b5b` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/evaluation/test.jsonl` | 8,374,083 | `a4fdcedfc86efebb36414114887abdc4ee51eba102b0247740aef18d1ad15abc` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/excluded.jsonl` | 2,176,829 | `d19b5d809ecc18e9372ae5a351e0daa4fa7af70a7ef907ce02f6934ec0fdd5af` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/human/calibration.jsonl` | 407,464 | `af7858186f401f0c016cd930dd057c08c1e4d8552d7dc4179e489ca3e0c48253` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/human/dev.jsonl` | 491,329 | `9ce783bfc24086a12ed0e66bae348f903b438eea87f8788b1ec0c3529e3e485a` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/human/train.jsonl` | 3,625,046 | `77ef84aa57260e4af9942fd3ee495b377bba4df295ba434d41ffa2cad495774e` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/synthetic/diagnostic.jsonl` | 3,477,485 | `93b4c9111fd7ce5006b18b6f20d9b1c6764a50cc5e322eaf2dc39c5b4190f322` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/installed/views/synthetic/train.jsonl` | 44,549,355 | `b711d4a8f7d9b29b7e1c19687e9d43f821e033ae01105ad52eae023498234dec` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/procurement-plan.json` | 10,623 | `df3a31ff1cf6906c380570b21589733d8be7437dad2dcf78464dea5851bea803` | receipt_or_source_hash_read |
| `.cache/speaker-dataset-v1/verified/audit.json` | 8,815 | `7e5164df17c24aaecea8d58991cd9c1da03d46010f584c519223dd0684457c8f` | receipt_hash_only |
| `.cache/speaker-dataset-v1/verified/environment.json` | 433 | `068852ee02860c88856293bfc6018068046c137af60dd693ca4852f8cd27247e` | receipt_hash_only |
| `.cache/speaker-dataset-v1/verified/inventory.jsonl` | 183,551,765 | `0c71cdc8347b8c9d01075f7cce6349c4eb4bfe2666653c4f149847ca8b2ded9a` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/auxiliary.jsonl` | 120,116,827 | `0dca5439d8ac063b207709ebfc2ff933ffbfa6e70feb18a6d81eaae2d52d4521` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/evaluation/holdout.jsonl` | 379,782 | `a51e720b97f7b31e3e2f025c21bf679006ad444fd2d363916d9a11aa0f488b5b` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/evaluation/test.jsonl` | 8,374,083 | `a4fdcedfc86efebb36414114887abdc4ee51eba102b0247740aef18d1ad15abc` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/excluded.jsonl` | 2,176,829 | `d19b5d809ecc18e9372ae5a351e0daa4fa7af70a7ef907ce02f6934ec0fdd5af` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/human/calibration.jsonl` | 407,464 | `af7858186f401f0c016cd930dd057c08c1e4d8552d7dc4179e489ca3e0c48253` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/human/dev.jsonl` | 491,329 | `9ce783bfc24086a12ed0e66bae348f903b438eea87f8788b1ec0c3529e3e485a` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/human/train.jsonl` | 3,625,046 | `77ef84aa57260e4af9942fd3ee495b377bba4df295ba434d41ffa2cad495774e` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/synthetic/diagnostic.jsonl` | 3,477,485 | `93b4c9111fd7ce5006b18b6f20d9b1c6764a50cc5e322eaf2dc39c5b4190f322` | hash_checked_against_prior_receipt |
| `.cache/speaker-dataset-v1/verified/views/synthetic/train.jsonl` | 44,549,355 | `b711d4a8f7d9b29b7e1c19687e9d43f821e033ae01105ad52eae023498234dec` | hash_checked_against_prior_receipt |
| `.cache/speaker-final-benchmark/current-train-human-coverage.json` | 1,659 | `6988d80879af0f2d3b68a1cd9f74afc05df2adbf9827f71b5ca9270b62673e6b` | receipt_or_source_hash_read |
| `.cache/speaker-final-benchmark/current-train-source.json` | 734 | `3b9ce60753aee18891a1897a6eaf5907da87442060b36be0817f0987fcaf709d` | receipt_or_source_hash_read |
| `.cache/speaker-final-benchmark/metadata.json` | 18,789 | `eed88a37ed8fdfb950b90f80b25ac7089a61cc1c7264b02cd5379242d35efaca` | content_read |
| `.cache/speaker-final-benchmark/reserved.json` | 362,931 | `0f8d867bac696bae372d63c68d3c029e33a28c99b67b6f6883adeb2115db2b76` | receipt_or_source_hash_read |
| `.cache/speaker-final-benchmark/rows.jsonl` | 4,531,006 | `feb850354cc49561055afeecc45ef3daa4e8d82ec6484176fb7efd47daf06633` | content_read |
| `.cache/speaker-final-benchmark/source.json` | 1,197 | `3e5689373e0395c7ebe2b7020ccedfe8233732df82d2bb6fb8eafd487c8ea7d4` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/README.md` | 20,692 | `5242baa27a11aa30b18c44b3ecd7c87c4157417f625477169af91212e4ffa0fa` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/benchmark.py` | 64,742 | `0b42ddda6a15f6f169417d878ea21ede198df5a18cda37b0dd63ec0453bd520c` | receipt_or_source_hash_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/audio.py` | 8,609 | `13eae82c351b3c553b35f18ec8bf4d463ef1f50672e3e860e5f21650f1aa4200` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/dataset.py` | 3,764 | `17d74702d4b9230965c96fbf7dd9a5b717fea47cc3374adfc48e9c2225dd6ba1` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/metrics.py` | 8,085 | `518ef4bc0646f93ad936d84ff2982bdd56691783e6892799e0728eff4e9410f6` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/pi.py` | 28,799 | `772a73dffa5810645182d2a6a9c16a828bb25d4a84f2d9cb3bbf60993d76545f` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/schema.py` | 10,197 | `cb7cefcd893a2443b9a9b9e9b2756efa27d273f4f0086b94491ca3b304e7a006` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/slots.py` | 17,062 | `7fc483ad851699d09e948ed942a36141012f474333e2ce521e705acce7fa30ef` | content_read |
| `.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191/vcmbench/variations.csv` | 3,433 | `6004b3b2b58137058c705b5769b0d094e05a5cff6ddcbed6de548b4a0105fcda` | content_read |
| `.cache/vcm-benchmark/current.json` | 325 | `248a3c7b52671d400ff41f0dc724109c29dd7040123f80102d96e153f2380602` | content_read |

### Historical ignored paths absent from this checkout

Recorded comparisons depend on payloads outside Git that are currently missing. Their saved metrics remain evidence; they were not rerun or silently recreated. Resolve the correct original source-bound payload before fitting/replaying/publishing.

| Missing path/group | Role and retained recovery identity |
| --- | --- |
| `models/default/`, including ONNX, normalization, policy, catalog and `ncnn/` | Actual complete-utterance shipping bundle required by runtime; use the shipping identities above |
| `models/selected/model.onnx` | Older semifinal diagnostic graph; expected `871c4a218684d50d132d8458b178e30c53ebff92c82b194bf3adaaa8652b1599`; not a compatible default replacement |
| `artifacts/*.onnx`, pretrained activation payloads | Only root metadata/notices remain; a manifest is not a resident inference session |
| `.cache/pcv-final-qualification-v1/exports/context128/model.int8.onnx` | Historical qualification graph `0b2701...`; differs from shipping selected checkpoint |
| `.cache/pcv-updated-context-int8-v1/mixed/epoch-012.pt` | Qualification checkpoint `aaa638fb225976d3dea33bd69dc103e87314c8f49651e1dbebf0bb3ff1e7ea92` |
| `.cache/pcv-updated-voice-v3/` raw predictions/checkpoint/exports | Historical continued-arm report retains metrics/export/selection evidence; raw run payloads absent |
| `.cache/speaker-demo-v1/` features, grid and paired acoustic results | Software-only v1 handoff does not establish completed acoustic runs |
| `.cache/pcv-human-slots-v1-qualification1/` predictions and model candidates | Historical unqualified human-slot study; detailed compact receipts remain tracked |
| `.cache/pcv-release-train-v1/`, `.cache/pcv-full-benchmark-v1/`, `.cache/speaker-publication-v2/ncnn-stable-full-replay/` selected outputs/raw predictions | Historical publication baseline and extra supplemental replays; no new replay here |
| `dist/speaker/publication-v2-shipping/` ZIPs | Historical inference `5f47c7fdc89dc15fc9f0fc1a70e75f2656e20dc93500f2dc7e977839ce10a56d`, benchmark `3136b65d498788e45f3b36d3620794f1b7162fab616916d3a91322cdee282808`, full `ad014d219687d90ed0ed8501b093c42739c99c1f0a36d1e615e3cd9a679035c6` |
| `data/vcm.jsonl`, `data/references.jsonl`, earlier research audio/caches | Older E-series regression inputs unavailable here; historical results preserved, no ASR baseline promoted |

Use the pinned source receipts/bootstrap only within an authorized acquisition task, retaining source-use restrictions. Restoring a missing recorded bundle is distinct from training a new candidate. For a web/Pro text diagnosis, this document supplies the essential schema, data gaps, measured results, provenance limitations, interfaces and questions. Reproducing an acoustic failure additionally needs the exact declared bundle, reviewed original recordings and per-trial output/stage receipts; request specific missing evidence rather than infer it from a hash or filename. No private recording/model archive was uploaded by this task.

## Consolidation and maintenance receipt

This is the **single new final research handoff**. The user's permitted consolidation alternative was used; no Markdown file, dataset, frozen protocol, source/license notice, archive or run output was deleted/moved. The active-work hold remains for broader maintenance, and no proposed maintenance skill/registry was created. Legacy reports remain linked because they retain unique failed results, frozen hashes, implementation instructions, recovery receipts or active-owner work. Consolidating findings here does not rewrite those histories or remove benchmark evidence.

| Existing document/group | Content consolidated here | Disposition / reason |
| --- | --- | --- |
| `docs/HANDOFF-speaker-demo-v1.md`, `docs/EXPERIMENT-speaker-demo-v1.md`, `docs/PLAN-speaker-demo-v1.md` | Owner helper, invariance pilot, software-only status, acoustic protocol and pending integration | Keep unchanged; unique protocol/execution references and active experiment |
| `docs/PLAN-speaker-demo-v2.md` | Pinned preparation, missing human coverage, mixed-training ablation and future hardware gates | Keep unchanged; another agent's current plan/source-bound work |
| `docs/HANDOFF-speakerui-runtime-deployment.md` | Superseded integration design status and remaining physical evidence | Keep unchanged; its superseded marker, implementation references and historical context remain useful |
| `docs/SPEC-speaker-publication-v2.md` and publication/qualification experiment documents | Current single-model application policy, engine/bundle distinction, actual shipping/quality limitations | Keep unchanged; source/packaging/replay evidence and writer references remain authoritative |
| Root `AGENTS.md`, `REQUIREMENTS.md`, `EXPERIMENTS.md` | Governing requirements and navigation | Preserve authoritative files; add this handoff's directory and dated requirement entry |
| Historical STT/phonetic/synthetic protocols and archive/recovery Markdown | Historical evidence boundaries | Keep unchanged; frozen/provenance/unique negative evidence, not duplicate disposal candidates |

No commit or push was made. The installed artifact metadata hashes below were captured and checked for preservation; graph payloads were already missing. Detailed receipts/source cards remain in place. There is no training, installation, physical Pi or live-demo success claim.

### Preserved artifact metadata hashes

These are the bytes present in this checkout, not proof that corresponding ignored model graphs are available or that every historical graph binding uses this text representation.

| Present artifact metadata | SHA-256 |
| --- | --- |
| `artifacts/brightness-validation.json` | `615f6b266124d89ee58de054874a4c752fd736bf0cb12449abdfca95e4f0c34c` |
| `artifacts/bundle.json` | `e1f9519400058b9fc5ebadcbb68f520a61faed93ff50eb528e530748e369ec85` |
| `artifacts/export-verification.json` | `a83385ef8d234a61b7997909911d72d86e533bf2b87d5b80c3013dcabe39169e` |
| `artifacts/family-metrics.json` | `a1bc4c392ece85d0afdf1ddae24e0e4461537865941f6f385cd8e9103b0f48e7` |
| `artifacts/manifest.json` | `8c1745c07dbb5a47299c09e0eac460fdfacc174d052c06cdacbcaa8e5446c128` |
| `artifacts/openwakeword-LICENSE.txt` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `artifacts/pilot-results.json` | `a5c8b6653fa7fceb8d621bf5774052b3f1fa5a194ec88f7ae4459462865938c2` |
| `artifacts/pretrained-NOTICE.txt` | `16f146f382cd504803fb425b1bcfe1680b23a58eabfa03ab310c37e0353de8a5` |
| `artifacts/silero-LICENSE.txt` | `2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b` |
| `artifacts/training-config.json` | `c087d451e3a58638978b6ab9550311a2c59e414ffab664db9799bbbbedd60a81` |
| `artifacts/training-history.json` | `9d30fd81efc7b0cd5408d68038d255fa9f50b4880182d361b14deaec1b19868c` |
| `artifacts/wake-models.json` | `9bc153473704dd425961ac0d4cc2f3b5607a8a35601a0bb5d687082651788c9e` |
| `models/selected/catalog.json` | `cbda06a063597950ac9467ced5b525d6e727d51915cb75df21689c17bfe28305` |
| `models/selected/policy.json` | `6c81885d649b06d30d4e8dbb9b0f30559b32e8e29f739ff319edde8cec652247` |
| `models/selected/recipe.json` | `88f0733d6a36bdd0aad16bff5282a72e8193f06e46fd7e4dbd540ca7aee439a5` |

## Reproduce the core compatibility and coverage checks

Run this from the repository root with `.venv/bin/python` after restoring the listed ignored metadata inputs. It exercises the existing formatter against the pinned parser and verifies the three human fitting views. It does not load weights, decode recordings, fit, change labels or modify the benchmark. Fixture timing is deliberately stipulated, not a measured inference duration.

```python
import hashlib, json, sys
from pathlib import Path
from scripts.runtime.pcv_schema import baseline_cases, canonical
from scripts.runtime.pcv_benchmark_adapter import log_record

bench = Path('.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191')
sys.path.insert(0, str(bench.resolve()))
from vcmbench.pi import LineParser
from vcmbench.schema import VARIATIONS, build_alias_table, resolve_prediction
from vcmbench.slots import slot_distance

cases = baseline_cases()
assert len(cases) == len(VARIATIONS) == 93
parser, aliases, correct = LineParser(), build_alias_table(), 0
for case, truth in zip(cases, VARIATIONS):
    assert case['text'] == truth.phrase
    event = {'output': {'accepted': True, 'command': case['command']},
             'timing': {'total_ms': 6.0}, 'audio_seconds': 1.5}
    parsed = parser.parse(canonical(log_record(event)), 0.0)
    intent, slot, known, _ = resolve_prediction(
        parsed.intent, parsed.slot, parsed.variation, aliases)
    exact = intent == truth.intent and (
        not truth.value or slot_distance(intent, truth.value, slot)['exact'])
    correct += int(exact and known)
assert correct == 93

receipts = json.loads(Path('experiments/speaker-demo-v2-data/data-audit.json').read_text())
required = {canonical(c['command']) for c in cases}
for view in ('human/train', 'human/dev', 'human/calibration'):
    receipt = receipts['views'][view]
    path = Path(receipt['path'])
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256']
    present, rows = set(), 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows += 1
            if row['command'] is not None:
                present.add(canonical(row['command']))
    assert rows == receipt['rows']
    print(view, rows, len(required & present), 'of', len(required),
          'missing', sorted(required - present))
```

Final handoff validation checked **36 local links with zero broken targets**, Markdown table/heading structure, preservation of **15 present artifact metadata hashes**, unchanged HEAD/branch and empty staging. The new handoff remains visible to Git. No missing acoustic graph was claimed verified. The document and the additive requirements/directory entries are the durable outputs; no old Markdown was deleted.

## Questions GPT web/Pro must answer before recommending implementation

1. Which observed errors belong to capture/wake/VAD, finite intent/slot decoding, catalog/schema, confidence calibration or speaker ownership? Which explanations are directly evidenced, and which require a falsifiable ablation?
2. How should the exact OptionB benchmark and preserved personal superset map into one shared typed-command model, including VOLUME operation mappings, the existing bare-CALL default and a final-decision-only benchmark serializer, without hiding missing acoustic slot evidence?
3. What is the smallest data addition justified by the measured human phrase/value/person gaps? Specify original recordings, distinct people, partitions and review; do not answer with a generic sentence-count target.
4. How would you test speaker memorization and source/accent shift with the pinned data when speaker aliases, cloned references and demographic metadata are unresolved? What fresh holdout is still needed after historical test exposure?
5. Which single matched ablation should run first: label/crop audit, class/person balancing, exact-command contrastive supervision or capacity? Give a decision rule and explain why the current evidence supports that order.
6. How can speaker variation be reduced for intent/slots while retaining every parameter distinction? Would a shared trunk with separate content/identity heads add benefit, or does a separately trained small identity encoder give a clearer controlled baseline?
7. Can a single short Hey Jarvis reference support identity checks on Stop/Next in the observed duration/noise conditions? Propose a duration-bucketed genuine/impostor calibration, not an arbitrary cosine threshold or silence-padded workaround.
8. What minimum evidence-bearing backend is needed for handovers and simultaneous speech? Explain why whole-utterance verification, Personal VAD, overlap rejection and target-speaker extraction have different guarantees. State whether the chosen first track rejects overlap or recovers commands through it, and resolve the official tester-wake/source-command pairing conflict with explicitly labelled protocols.
9. If filtering/extraction is proposed, how will it avoid taking a bystander's numeric/other slot or suppressing genuine owner speech? Define clean-owner retention and downstream exact-command tests, all model costs, training provenance and Pi feasibility.
10. Are the supplied synthetic QA flags and pinned source terms sufficient for the proposed use? What source/listening/voice-alias uncertainties remain? Separate supplementary training benefit from claims of new human speakers or authentic accent coverage.
11. Which pretrained research controls, if any, require a new user exception, and which parts would ship? Preserve no-ASR/no-LLM/offline inference and count the reference embedding cost; an offline teacher or published ASR experiment does not authorize a runtime transcriber.
12. Provide a bounded next experiment with frozen source/version/hash bindings, human-first selection/calibration, one-variable comparisons, accepted typed-command/rejection/resource gates and a stop rule. Label unavailable graphs, unaudited metadata, absent wake pairs and unmeasured Pi/live evidence as missing; do not fabricate accuracy or declare the demo finished from software fixtures.

Requested research answer: a ranked diagnosis with confidence and supporting sources, a decision on the first bounded experiment, and a concrete shared-VCM plus one-wake/one-command owner design whose unproven overlap and hardware capabilities remain visible. Keep benchmark results, personal-superset acceptance and physical demonstration qualification distinct.
