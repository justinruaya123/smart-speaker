# Fixed execution handoff: PC-v1 speech → intent + parameters

## 0. Restore the complete source delivery

GitHub commit `1a4caaf3263a0106653b8bdd33b8e3f4a4ccac27` contains the six-file decoder milestone, not all files in this delivery. Apply the supplied `VCM_PCv1_remaining_implementation.patch` from a clean checkout of that commit (or a verified unchanged descendant):

```bash
git -c core.whitespace=cr-at-eol apply --check /path/to/VCM_PCv1_remaining_implementation.patch
git -c core.whitespace=cr-at-eol apply /path/to/VCM_PCv1_remaining_implementation.patch
python -m pytest experiments/phrase-contract-v1/tests -q
git add experiments/phrase-contract-v1
git commit -m "Add PC-v1 training, runtime, evidence pipeline and fixed execution handoff"
git push origin codex/semifinal-handoff
```

The patch contains only additional experiment files; it does not overwrite the committed decoder or unrelated research. The full ZIP is an alternative source delivery, not a model bundle. Generated spec files can be recreated from the committed generator using `python experiments/phrase-contract-v1/run.py init-spec`; they are intentionally ignored in ordinary subsequent staging. The immutable original proposal is preserved in the ZIP.

## 1. Outcome and current state

Use the implementation in this directory, not a new architecture search. The final candidate family is the existing **113,064-parameter causal CNN**, 40 causal log-mel inputs, 39 phonemes plus CTC blank, six explicit caches, and the **exact shared-prefix full-command decoder** now implemented in `pcvcm/decoder.py`. The public output is the strict v2 typed command; no free sentence transcript, cloud service, pretrained acoustic initialization, LLM, external action, second acoustic model, or arbitrary runtime name catalog.

The working PC-v1 catalog has 570 literal positive wordings, 402 complete commands, and 200 reviewed rejection prompts. Every 0–100 brightness/volume value is a separate phrase row. Keep all ten use cases and finite named/content catalogs. `spec/ACCEPTANCE_CONTRACT.md` controls scope and gates. The old implementation handoff in `spec/` is historical; this file identifies which parts are now executable.

**This is not a trained final model.** The present measurements are decoder and software experiments only. No new original speech, ONNX graph, or physical Pi/microphone evidence was available. The human-recording requirement is substantial, not a token administrative task. Do not claim a likely passing model solely because this implementation is complete enough to train.

## 2. Preserve and install the implementation

Work only on `codex/semifinal-handoff`. Start from `a788bb38e3345a3a7d9dbd0316b9c9719ee20639` or a verified descendant containing these changes. Do not reset to the old hash when the remote advances. Preserve unrelated tracked/untracked work; do not clean, stash, overwrite existing outputs, force-push, or modify main.

This delivery adds `experiments/phrase-contract-v1/**` only. Runtime training imports the existing root `vcm` package; it does not copy a competing model. Use the repository's Python environment and pinned dependency practices. Training requires PyTorch, numpy, einops, soundfile; testing requires pytest. Actual export/runtime additionally requires ONNX and ONNX Runtime. The offline host did not contain the latter two. Proposed versions 1.19.1/1.23.2 were not installed/tested; resolve and hash-lock target CPython/ARM64 wheels before claiming package compatibility.

From repository root:

```bash
E=experiments/phrase-contract-v1
python -m pytest "$E/tests" -q
python "$E/run.py" check
python "$E/run.py" --help
python "$E/run.py" recording-plan --output .cache/phrase-contract-v1/recording-plan.jsonl
```

The supplied patch/ZIP contains all generated catalogs. `spec/build_contract.py` can regenerate them in a **new directory**. Never overwrite a frozen approved spec during a model run. Verify the six authoritative file hashes against `reports/contract-binding.json`.

`reports/verification.json` reports the checks actually executed here. The entire original repository test suite still needs a fresh checkout; the downloaded source artifact was incomplete. A source-contract pass is not waveform validation.

## 3. Data: exact procedure, no substitutes

Use new consented first-party human recordings as the default source policy. Do not inherit FSC-derived weights or introduce corpus rights by assumption. Separately licensed original data may be added only under a versioned pre-training source receipt; changing the recipe then invalidates earlier selection/calibration. TTS, splicing, resampling, replay, isolated words, pronunciation entries, and simultaneous microphones never count as independent human utterances.

For each of the 570 positive wordings collect 24 train recordings from 12 speakers, 8 development from 4, 8 calibration from 4, and 20 new sealed confirmation recordings from 10. The four speaker pools are disjoint. A speaker may record multiple phrases. C1–C4 are quiet 0.5 m, quiet 2 m, about 10 dB real background noise 0.5 m, and about 10 dB 2 m/reverberant; final has five per condition. Keep actual English population, rooms and microphone/gain/SNR metadata. Include 300 ms preroll and at least 1 s postroll; do not crop endings to make endpointing easy.

The executable plan contains **48,800 unrecorded slots**: 34,200 positive utterances, 12,000 prompted rejection utterances, and 2,600 natural unknown clips across splits. It records slots, not actual people or data. Replace `speaker_slot` with a stable real `speaker_id`; do not invent identities. Final alone has 11,400 positives, 4,000 prompted rejections, and 1,000 natural unknown clips.

Each original manifest row must contain recording_id, utterance_parent_id, speaker_id, session_id, source_id, source_license, consent_reference, actual audio-relative path, original SHA256, decoded float32 PCM SHA256, split, condition_id, phrase_id (or null for natural unknown), use_case, exact text, expected_command, reference_class, synthetic=false, human_reviewed=true. Add distance, microphone, actual SNR, actual rate/channel metadata. Unknown speech has its actual audited transcript; only actual nonspeech gets an empty phone target.

The supplied lexicon is a frozen CMUdict subset plus explicit unmute and AM handling. Natural speech outside those 212 words requires a training-only reviewed pronunciation extension using the same 39 phones. Freeze the extension before preparation; never map ordinary unknown speech to blank or force nearest dictionary words. Use the `--lexicon` option for preparation and retain its hash. The deployment catalog remains the exact original command lexicon unless a separately documented pronunciation change is made.

A custodian—not the fitting process—holds final audio. The final commitment must include contract SHA256, original final-manifest SHA256, 11,400+ positive-recording count, and the complete set of hashes of actual final speaker identities. Freeze it before fitting. The implementation verifies this commitment on final replay. Also audit raw/PCM/parent overlap and manually inspect acoustic near duplicates before fitting. A truth flag alone is not the audit: retain the reviewed record pairs, source/consent evidence, reviewer and date.

Prepare and verify:

```bash
python "$E/run.py" prepare \
  --manifest .cache/phrase-contract-v1/original-train-dev-cal.jsonl \
  --audio-root .cache/phrase-contract-v1/original-audio \
  --confirmation-commitment .cache/phrase-contract-v1/confirmation.commitment.json \
  --output .cache/phrase-contract-v1/prepared
```

Do not include final audio in this preparation. The output uses portable relative feature paths, hashed features, train-only normalization, and immutable train/dev/calibration manifests. It refuses incomplete positive quotas, synthetic/unreviewed inputs, missing identity, invalid rate, repeated event/PCM, cross-split speakers, incorrect references and CTC-infeasible audio. Add new tests before any necessary importer correction.

Create `rights-and-duplicates.json` with `frozen_data_sha256` equal to the actual prepared freeze.json SHA256 and `passed:true` **only after the actual review**, plus detailed evidence references. This is an external review artifact, not something Codex should fabricate to pass a check.

## 4. Bounded final candidate training

No architecture sweep. Use `configs/train-primary.json`: random initialization; AdamW .001, weight decay .0001, clip5, batch64, seeds53/71, at most40 epochs each. No artificial audio. Hierarchically balance use case, phrase and speaker within 80% positives/20% negatives. Train-only mean/std stay fixed. Keep every epoch checkpoint, RNG/optimizer state, predictions, hashes and all rejected candidates.

```bash
python "$E/run.py" train --config "$E/configs/train-primary.json" \
  --output .cache/phrase-contract-v1/run-primary
```

Checkpoint choice is deterministic: number of positive phrases at ≥90% raw exactness, then worst-use-case raw exactness, then macro exactness, total exactness, lower seed/earlier epoch. Selection must see every development phrase and full target/value JSON, never just intent or phoneme loss. With eight development recordings, reaching a per-phrase 90% raw gate requires all eight. No missing phrase may be hidden by averages. Paired full per-recording predictions are saved; add a generated gained/lost CSV before final selection if the implementation does not already emit it.

Training-only cross-validation is mandatory before final training: implement a fixed **three-fold speaker-group CV** over the training pool. Each fold recomputes normalization on its fit speakers and trains the primary architecture for at most 12 epochs with seed53. Keep fold speaker IDs and no-overlap audit. Gate: raw complete-command macro accuracy ≥80% in each fold and worst use case ≥70%. CV informs viability only; do not average it with final confirmation or tune on fold test predictions.

If the primary fails the development selection gates, run exactly one prescribed ablation: `configs/train-hard-pairs.json`, seeds53/71, same architecture/hyperparameters/epochs, changing only sampling to double fixed numeric/contact confusion targets. Do not run the ablation because calibration fails after primary development success. Stop after these four maximum full candidates. Do not remove phrases, widen thresholds, or search architectures to make the report pass.

The whole effort—including three 12-epoch CV folds, the two-seed primary and optional two-seed ablation—must stay below **200 epoch-equivalents and 24 GPU-hours**, whichever ends first. The worst-case prescribed schedule is 196 epoch-equivalents. If compute stops earlier, retain the partial histories and report the blocker.

## 5. Endpoint, export and calibration

The implemented causal endpoint is deterministic and sample-count based: at least 300 ms preroll/valid speech, 700 ms trailing below-threshold silence, 12 s max utterance, bounded capture. **One selection task remains before calibration:** on development only, evaluate the following four frozen activity thresholds with the final runtime and choose by lowest endpoint miss count, then median absolute endpoint delay, then the threshold closer to -45 dBFS:

```text
-50 dBFS, -45 dBFS, -40 dBFS, -35 dBFS
```

Internal pauses up to 600 ms in curated numbers/negation/names must not terminate. Overflows, capture gaps, invalid audio and >12 s fail closed. Freeze endpoint config/hash before calibration. Never use transcript/WAV EOF as the final endpoint.

Export only the development-selected real checkpoint:

```bash
python "$E/run.py" export \
  --run .cache/phrase-contract-v1/run-primary \
  --output .cache/phrase-contract-v1/candidate-bundle
```

The command must fail when no real selected checkpoint exists. Verify parameters=113064, graph<1MiB, one graph, same six cache shapes, PyTorch↔ONNX parity on nonzero caches for1/3/4/8/17 frames, and identical full-catalog candidate commands on an actual human-waveform development subset. Keep FP32 first. Do not quantize before the FP32 confirmation pass. Any later static INT8 test is optional and is a new bound model needing full regression/calibration plus Pi speed improvement; do not assume it helps.

Calibrate once on the held calibration split:

```bash
python "$E/run.py" calibrate \
  --bundle .cache/phrase-contract-v1/candidate-bundle \
  --manifest .cache/phrase-contract-v1/prepared/calibration.jsonl \
  --output .cache/phrase-contract-v1/calibration
```

Current implementation selects global score/margin only, exhaustively over 31×21 quantile candidates (≤651 combinations), and computes every gate using the **same deployed predicate** as runtime. This is intentionally smaller than the previous proposal's unsupported per-UC threshold complexity. Gate per each of ten UCs: accepted precision≥.99 and complete accepted recall≥.90. Each negative class—prompted rejection and natural unknown—needs≥50 calibration samples and FAR≤.01; combined FAR≤.01. Calibration that emits nothing fails recall. Store all candidate thresholds and the immutable selected policy bound to model/catalog/frontend/endpoint/data.

If no global threshold passes but raw development/calibration exactness is already strong, you may add **one** predeclared per-UC threshold calibration pass without retraining, at the same 31×21 candidates per UC, and bind the selected vector to the policy. Do not add learned calibrators or arbitrary transforms. If that fails, stop; model/data is the blocker.

## 6. Confirmation, Pi and release

After policy lock, mount the custodian-held final recordings and verify the final commitment **before inference**. Run replay exactly once for qualification; if code crashes before generating predictions, fix only the implementation defect with unit/development data and rerun the same sealed set, recording the incident. If an accuracy result is exposed, final is no longer untouched for another candidate.

Required final gates: every phrase has20 independent recordings from10 speakers;≥18/20 accepted exact; zero wrong accepted;≥4/5 in each C1–C4; each UC precision≥99% and complete accepted recall≥90%; each negative class≥100 and FAR≤1%; prompted and natural FAR≤1% separately; critical negated/ambiguous/incomplete/contradictory controls have zero false accepts. Retain all rows, including failures. Report exact correct JSON, not intent-only accuracy. `pcvcm/evidence.py` also verifies that expected references match the frozen catalog, duplicate recording/event identities do not inflate denominators, and required metadata is present.

Run per-condition raw complete-command gates clean≥95% and C3/C4 realistic ~10 dB≥90%. These are secondary to accepted command gates; they cannot rescue a failed phrase. Report speaker-clustered confidence intervals. Do not equate observed 99% with a population guarantee.

**Live microphone adapter remains to implement.** Use callback capture→bounded PCM queue with monotonically increasing sample sequence numbers; decoder/UI on consumer thread. Any overflow/dropout invalidates the utterance. Same `PCVRuntime.feed` must handle WAV and live PCM; no separate recognizer. Start with push-to-talk plus validated endpoint, no wake-word claim.

Physical qualification is separately mandatory for Pi4 and Pi5 if both are advertised. Use64-bit Pi OS/CPython target, intended microphone, and selected catalog; record OS/kernel/Python/ORT/CPU serial profile, microphone/gain, power/cooling/ambient. Bench 1/2/4 ORT intra-op threads, spinning off, sequential execution; select once. For every profile: graph<1MiB, RSS<256MiB, processingp95≤100ms, RTF<1, responsep95≤1s after speech,30 min sustained capture, zero unhandled queue overruns, and **three armed hours of diverse continuous non-command audio with zero accepted commands**. Record CPU/temp/throttling. Synthetic silence is not the three-hour control. Pi5 does not qualify Pi4. Replay actual C1–C4 captures first, then real microphone.

The current allowlist publisher supports diagnostics only and requires a disabled policy. This is intentional: the physical verifier/qualified publisher must not be replaced by manual `qualified=true` flags. The remaining finalization task is to verify every actual recording/physical report hash, rerun all gates for the exact shipped binding, require real enabled calibration, and permit one qualified package only when all agree.

Ship one graph, exact frontend, caches/schema/catalog/phones/lexicon provenance, endpoint, policy, runtime source, THIRD_PARTY_NOTICES, hash-locked target-compatible dependencies and license notices. No torch, trainers, extra graph, corpora, fixtures, visualizer, credentials or external executor. The diagnostic publisher currently emits an asset/source ZIP; offline ARM64 installation wheels/lock and executable launch entry point are remaining packaging work. Test fresh extraction with old research directories unavailable, inference network-disabled, and all documented commands. Rebuild ZIP from identical assets twice and compare hashes.

## 7. Commit and close out

Run new tests, the full real repository suite, real ONNX checks, replay reports, preservation checks and clean-install tests. Preserve failed/skipped outputs distinctly. Commit only this experiment and deliberately reviewed opt-in integration changes; never .github workflows or research cleanup by implication. Use a normal fast-forward push to the requested branch.

Return actual branch/commit, artifact hashes, per-UC and worst-phrase gates, complete report paths, failed/skipped checks, physical profile evidence, and reproducible installation commands. Final state is QUALIFIED_PC_V1 only if all required gates truly passed; otherwise name the exact blocker and retain the unqualified bundle. Never label artificial-logit results, the artificial-feature smoke network, a successful export, or an all-reject policy as the final speech model.
