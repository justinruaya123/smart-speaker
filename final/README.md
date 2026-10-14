# Final accuracy-first closeout

Repository: `justinruaya123/smart-speaker`  
Branch: `codex/vcm-intent-slots-v1`  
Parent examined: `5be3a7f5f5ac38ba6c4fcc14d7fd99de478b0bbd`  
Scope: staged service integration, settled A100 training verification, an additional
six-voice synthetic command benchmark, and the submission card. **No Pi validation
is required in this accuracy-first stage. No active job or installed default is changed.**

Start from `MODEL_CARD.md`, `verification.json`, `tts-protocol.json`, and the staged
package's `VCM_FINALIZATION.md`. Current card status is blocked, not a claim about
the running A100 model: this session has no final checkpoint or current job logs.
The previous 0/636 and 0/83 human results belong to historical unqualified models.

## 1. Integrate the existing package without overwriting another agent

`staging/finalization.tar.xz` contains the complete previous finalization package,
with all original file bytes preserved, including the four source/test files,
patch, audit and receipts. It is a transport archive, not another implementation.
Archive SHA-256:

`7c7a1d40d94579298d570a6d6c59a1e742e2748cedc3e5a89f75d281206a1b80`

Original supplied ZIP SHA-256:

`1f11a6188cbaa27bb4199e7aca728a3a502a7d387c1306ac5c47283f1187e491`

Use an isolated integration worktree and the existing root interpreter, denoted
`VCM_PY`. Do not pull/reset over dirty work or try to align local/remote commit IDs
by discarding changes. First compare trees and the paths involved.

```bash
printf '%s  %s\n' \
  7c7a1d40d94579298d570a6d6c59a1e742e2748cedc3e5a89f75d281206a1b80 \
  final/staging/finalization.tar.xz | sha256sum --check
mkdir -p .cache
mkdir .cache/finalization-import-01
tar -tJf final/staging/finalization.tar.xz
tar -xJf final/staging/finalization.tar.xz -C .cache/finalization-import-01
(cd .cache/finalization-import-01 && sha256sum --check SHA256SUMS.txt)
git hash-object scripts/runtime/speaker_worker.py
# Expected original: 944ce510fbe14254ebd72362d11c1987ecd5eead
# If already integrated or changed, reconcile by hash/diff; do not overwrite.
git apply --check .cache/finalization-import-01/service-readiness.patch
git apply .cache/finalization-import-01/service-readiness.patch
```

Verify that the three newly added test/runner paths are absent before applying.
After applying, compare all four source hashes with the staged `verification.json`.
Use the actual project's complete test commands, including the SpeakerUI suite,
not only this package's unit tests. Preserve and name historical failures rather
than calling the broad suite green. The local evidence here is 76 passing tests:
49 re-executed package fixtures plus 27 new numerical/accounting tests, not speech.

## 2. Confirm the actual A100 training before final evaluation

Inspect, but do not interrupt, the running process. Record its source/tree, command,
node/job allocation, GPU model/count, CUDA visibility, Torch/driver versions, seed,
arm, actual optimizer updates, wall clock, logs and checkpoint hashes. The existing
CLI defaults to CPU: confirm the actual job uses `--device cuda`, not merely that
an A100 was allocated. Do not start duplicate runs.

Re-audit the actual updated data rather than assuming the remote manifests include
another agent's work. Require WAV/decoded-PCM integrity, person/alias and recording
separation, rights, complete intent/role/value/validity coverage, and connected
entity supervision/enrollment. Report effective loss denominators and train/dev
curves; missing or zero-supervision heads are not fixed by more GPU time. Preserve
the original human-only uniform/balanced seeds 53/71/89, 30 x 96 updates, batch 64,
AdamW 0.001, decay 0.0001 and clipping 1. No use of the new TTS set in fitting,
checkpoint choice, score calibration, entity enrollment or hyperparameter selection.

Select on human development only. Re-encode final entity enrollment with the actual
selected checkpoint-file and serialized-normalization hashes; do not relabel a
development memory. Independently calibrate, export the actual candidate, verify
PyTorch/ORT/ncnn tensor and final-decision parity, and create the original complete
selection-before-test lock. Preserve 0.95 intent and 0.90 for every supplied slot.
Unknown names must not become generic defaults. Do not substitute the numeral
checkpoint, an untrained fixture, or the unavailable old shipping payload.

## 3. Additional six-voice TTS sanity benchmark

The selected source model is **hexgrad/Kokoro-82M v1.0**, using the FP32 ONNX
artifacts mirrored by **fastrtc/kokoro-onnx**. The exact revision and both full file
SHA-256 values are in `tts-protocol.json`. Downloads remain in ignored `.cache`.
Upstream weights are Apache-2.0; the MIT label on the mirror/wrapper does not
relicense the weights or this project's dataset. Preserve upstream notices.

| Source-described preset | Sex label | English variety |
|---|---|---|
| af_heart | Female | American |
| af_bella | Female | American |
| bf_emma | Female | British |
| am_fenrir | Male | American |
| am_michael | Male | American |
| bm_george | Male | British |

These are six distinct preset packs, not pitch-shifted copies and not six human
participants. They do not establish Philippine-accent or unseen-human performance.
The upstream author reports short-utterance weaknesses, important for commands
such as “Stop.” **Do not prepend a carrier phrase, concatenate different commands,
change verbatims, or retry synthesis because the VCM rejected a sample.** Every
waveform needs independent listening before VCM outcomes: exact words, all values
and names, no clipping/truncation/artifacts, and quality at least 4/5. A failed QA
trial blocks this version; preserve it and version any approved regeneration.

`kokoro-onnx==0.5.0` supports Python 3.10–3.13. The original `kokoro==0.9.4` package
requires Python below 3.13; do not downgrade the training environment or bypass
package requirements. Use the existing root dependency/lock workflow to add the
ONNX generator as evaluation-only tooling after active training finishes, unless
an already prepared approved tooling host is available. Preview resolution and
preserve installed Torch/NumPy/ORT pins. The wrapper uses an eSpeak text frontend
only to synthesize audio; no phoneme, text, or reference label enters the VCM.

The runtime network fetch failed in this chat environment and the generator was
not installed. **No TTS waveform was generated here.** The commands below are
implemented but need the actual network-enabled cluster environment and selected
bundle. The generator runs on CPU by default and must not consume an active
training GPU. Its dependencies are not part of the shipped Pi recognizer.

```bash
# These variables must point to the actual selected and calibrated candidate.
# BUNDLE: final candidate directory; SELECTION: existing human-test selection lock.
# VCM_PY: the existing root interpreter. Use a unique study directory each time.
RUN=.cache/final-six-voice-v1
"$VCM_PY" final/tts_sanity.py plan \
  --contract "$BUNDLE/contract.json" --output "$RUN/plan.json"
"$VCM_PY" final/tts_sanity.py fetch --assets "$RUN/assets"
"$VCM_PY" final/tts_sanity.py generate \
  --plan "$RUN/plan.json" --assets "$RUN/assets" --output "$RUN/generated"
# Copy review-template.json to reviewed.json and genuinely listen to every WAV.
# Fill reviewer, reviewed_at, heard_exact and quality; no fabricated QA.
"$VCM_PY" final/tts_sanity.py seal \
  --manifest "$RUN/generated/manifest.json" --review "$RUN/reviewed.json" \
  --bundle "$BUNDLE" --parent-selection "$SELECTION" --output "$RUN/lock.json"
for ENGINE in ncnn onnxruntime pytorch; do
  "$VCM_PY" final/tts_sanity.py replay \
    --lock "$RUN/lock.json" --bundle "$BUNDLE" --engine "$ENGINE" \
    --output "$RUN/replay-$ENGINE"
done
```

The plan uses every exact case and the complete selected catalog from the bound
contract, not a rewritten phrase list. It checks the original 93 OptionB verbatims
and outputs against the existing schema. There are at least **558 OptionB trials**
(93 x 6). At the last documented counts, 414 definitions plus 1,031 catalog entries
produce 8,670 case references; identical text/voice pairs may reuse one WAV without
losing any reference. Actual counts are generated, never forced to those numbers.
Native 24 kHz audio is saved, anti-aliased to 16 kHz mono, and passed as WAV bytes
through the actual `AudioService` / `IntentSlotsRuntime` application path. This is
not a loudspeaker-to-microphone or wake/ownership test.

The new synthetic sanity gate is deliberately strict: every positive must be an
accepted exact typed command, every specified negative must reject correctly,
and there must be no missing outputs or faults, across every voice and track.
This is a local additional criterion, not a claim that the class benchmark requires
100%. Reports include per-voice, per-intent, per-case, supplied-slot and track
counts, accepted precision/recall, false accepts, faults and missing results.
Cross-engine final decisions must agree. Do not interpret an all-reject model's
negative rejections as success or its undefined accepted precision as 100%.

## 4. Keep the official benchmark separate

Keep `airimonda/vcm-benchmark` pinned at
`ab39857cf73dd067d068fed388c321ab160dc191`, with the exact parser hash recorded in
the protocol. Use the existing canonical adapter of actual `final_application`.
Run the real held-out command audio through the final candidate and report the
strict typed-command score alongside official intent/slot scoring; do not submit
expected labels or diagnostic candidates as predictions.

The official README describes a live laptop-speaker -> Pi-microphone test. Since
the user defers the Pi now, report **offline command-audio replay / official-parser
compatibility**, not a completed official live benchmark. Do not use `--mode sim`
to produce accuracy claims. Keep command-only, real paired-owner, and the official
spliced-wake tracks distinct. Fresh unseen-human accuracy remains necessary:
synthetic success is an extra check, not a replacement for human generalization.

## 5. Fill the card and clean up only after successful finalization

Update `MODEL_CARD.md` from this exact final candidate's immutable receipts:
actual manifest hours/utterances/speakers, A100 node/GPU/seed/wall-clock/update/loss
history, selected epoch, model/export sizes, separate official-offline, human and
six-voice TTS results, per-intent/slot/voice counts, precision/FAR and all failures.
Unknown values remain “not measured”; old untrained results must not fill final
A100 fields. Raspberry Pi latency/RTF/RSS/CPU/wake/thermal fields remain **deferred**.
Use the complete-utterance temporal CNN description, not causal/KV-cache template
text. Keep the public-repo/license, dataset DOI/access, baseline-comparison and
checkpoint-release submission gaps visible. Do not publish private audio, change
repository visibility, or invent a DOI/license/release link.

Before deleting the package, integrate canonical source/tests, preserve its audit
and verification in the existing experiment records, record their hashes and the
integration commit, finish the actual accuracy gates, and update the card. Then
execute only this user-authorized package deletion in the integration worktree:

```bash
git rm -- final/staging/finalization.tar.xz
# Commit the removal and a durable cleanup receipt with the integration commit,
# preserved evidence paths/hashes, tests and accuracy-result identities.
```

**Do not delete `final/` wholesale, this runner/protocol/card, source data,
checkpoints, generated benchmark recordings, raw logs, previous failed experiments,
or another agent's files.** If finalization is blocked or failed, retain the
package. No `rm -rf`, `git clean`, history rewriting or blanket cache cleanup.

### Primary sources for the generator and benchmark

- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- https://huggingface.co/fastrtc/kokoro-onnx/tree/main
- https://huggingface.co/fastrtc/kokoro-onnx/blob/main/kokoro-v1.0.onnx
- https://huggingface.co/fastrtc/kokoro-onnx/blob/main/voices-v1.0.bin
- https://pypi.org/project/kokoro-onnx/0.5.0/
- https://pypi.org/project/kokoro/0.9.4/
- https://github.com/airimonda/vcm-benchmark/blob/ab39857cf73dd067d068fed388c321ab160dc191/README.md

Inspected 2026-10-15. Source quality descriptions are not a local listening result.
