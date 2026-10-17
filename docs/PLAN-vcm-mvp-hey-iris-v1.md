# Finite command and custom wake implementation plan

Goal: retain the complete typed interface and fixed benchmark, while testing simpler direct command outputs in the existing approximately 935k and 2M model families. The supplied execution handoff is the specification; current local results override its historical accuracy assumptions.

Architecture: reuse our 40-bin frontend and contextual encoders, replace phonetic catalog scoring with one atomic command classifier, and reuse existing service/session interfaces. Existing weights, thresholds, benchmark data and installed models remain immutable. A separately trained small wake network must have its own evidence. Speaker ownership remains a separate, explicit profile.

Execution is inline under the user's instruction to implement without another planning approval cycle. All payloads use the existing ignored cache; durable evidence uses this experiment and the flat docs folder.

- [x] Freeze the current source/data/benchmark identities and deduplicate typed commands; preserve all 93 baseline wordings and personal demo definitions.
- [x] RED/GREEN tests: command identity, absent/malformed arguments, padding, direct outputs, negative-constrained calibration, profile transparency and streaming reset.
- [x] Fit the existing encoder families with direct complete-command supervision, first proving a small balanced slice can be learned; select only on development.
- [x] Freeze policy and weights before historical regression; retain human, synthetic and negative denominators separately. Compare both original candidates unchanged.
- [x] Train/export a small random-init Hey Iris network if permitted positive/negative audio can be constructed; document synthetic provenance and missing real-speaker evidence.
- [x] Export and check ORT/ncnn and actual resident adapter decisions; package an isolated candidate without replacing installed artifacts unless its declared gates pass.
- [x] Record failures and physical/live/ownership limits, exact commands, augmentation citations, and measured values in a two-slide submission artifact.

Review focus: unknown explicit names must not map to defaults; a rejected output must not be rescued; padding must not change a command; benchmark references must never enter fitting; a synthetic voice must not be presented as a human participant.

Rulings: preserve the existing encoder sizes instead of starting another architecture grid; prioritize the direct recognizer before custom wake/ownership work; no physical Pi is available; original prerecorded evaluations are regression evidence, not a new blind test.


Completed 17 October 2026: all bounded implementation/evaluation/package tasks have evidence in [the final report](EXPERIMENT-vcm-mvp-hey-iris-v3.md). The added V3 calibration comparison satisfies the user's command-only30+/36 target for both tracks; qualification, real-person ownership, five-person live tests and physical Pi evidence remain incomplete. No installation, merge, commit or push.
