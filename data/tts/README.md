# tts: fixed-command synthetic pilot

`planned.jsonl` is the frozen text/voice plan. `manifest.jsonl` contains completed whole-utterance recordings, original 16 kHz mono audio hashes and generation provenance. `human.jsonl` is a separate reference manifest pointing to original human recordings; it contains no TTS. Neither file alters the earlier human-only studies.

Every synthetic row records the fixed family, readiness, operation, integer amount and color labels; the original English text; highlighted keyword spans using Python character offsets; speaker/voice identity; split; requested accent; random batch seed; batch members; anchor hash; source model; duration and file hash. Keyword spans are text positions. They are not verified audio boundaries.

The classes are fixed to POWER_ON, POWER_OFF, BRIGHTNESS, COLOR_SET, MEDIA_PLAY, MEDIA_RESUME, TIME_QUERY, DATE_QUERY and VOLUME, with UNKNOWN for rejection. Generic light commands need no named entity. Nine colors and integers 0-100 are explicit finite classes. UP/DOWN normalize to +/-10 percentage points; explicit relative amounts use `by`. Malformed, conflicting and out-of-range numbers have REJECT readiness.

Eight designed voices are used for training and two each for development and test. Every split contains requested Filipino-English and other English voices. Accent quality and the acoustic fidelity of the intended TTS transcript remain unverified unless a separate review receipt says otherwise. Dataset metadata never treats a prompt as acoustic proof.

Reports, acquisition receipts and training/evaluation evidence live under `experiments/synthetic/` and `experiments/mixed/`. Large generated recordings live under `data/audio/tts/`, while model and feature artifacts stay under `.cache/`.

Acoustic templates cover every absolute percentage. Explicit relative-by examples use 1, 10, 25, 50 and 100; bare relative commands use 10. Parser support for other relative amounts is not evidence that all such spoken combinations were tested. The auxiliary keyword head has the declared 65-word inventory, distinct from the complete intent labels.
