# VCM intent and slots v1 execution — 2026-10-14

The approved architecture is implemented in an isolated experimental workspace.
It is **not acoustically trained or qualified**. Installed artifacts, runtime
defaults, other agents' dataset preparation and provider actions are preserved.
The [design protocol](../experiments/vcm-intent-slots-v1/protocol.json) is unchanged.
[Execution bindings](../experiments/vcm-intent-slots-v1/execution-protocol.json)
record the implemented state.

The workspace starts at exact design commit
`ede9d3caa036876cb112907b8d85e239b799c618`. HTTPS Git had no credentials;
the connected GitHub app supplied the four design blobs. Their blob hashes,
tree `ffd28157ffb26ea5a2f9833177f74cd5b8183e71` and reconstructed commit
hash were verified before worktree creation. The dirty main checkout was never
switched, reset, cleaned, stashed or staged.

The selected catalog retains all 1,031 entries. The current schema generates
414 cases, including 93 OptionB phrases and 31 positive typed outputs.
Historical 414/415/968 counts were not forced equal. No playlist field,
new default model, transcription pipeline or provider integration was added.

## Implementation and verification

The model uses the existing mono 16 kHz, 40-bin frontend, width-128 stem and
eight masked temporal blocks with per-frame LayerNorm. Its 280,818 parameters
include authoritative intent, role-state, categorical value, request-validity
and role-specific acoustic entity heads. Padding invariance and masked
supervision are tested. Enrollment uses multiple original examples, absolute
and margin rejection, duplicate handling and encoder/catalog identities.

Typed decisions preserve 0.95 intent and 0.90 for every supplied slot.
Default fields retain request/rule provenance. Raw selection scores are
explicitly uncalibrated; composed scores require separately frozen calibration.
The pipeline supplies matched samplers, human development selection, separate
calibration, complete selection locks, guarded evaluation, exact-baseline
attribution, FP32 exports and an explicit action-free worker.

The owner subsystem remains separate. Its software interface binds exact PCM,
sample spans, immutable calibration/policy, session and sequence identities.
It rejects unavailable overlap/coverage, impostors, stale results and overflow,
and consumes the one-command session on completion/reject/fault/expiry.
No command embedding becomes an identity or extraction backend.

Final combined verification: **154 tests passed**, comprising 94 new and 60
existing owner/helper, adapter and worker tests; 24 exporter deprecation warnings.
Compilation and whitespace checks passed. Full-suite collection failed with
three historical errors: unavailable safetensors, the test_catalog import-name
collision, and incompatible train.register_sources in scripts/test_prototype.py.
No full-suite or full PEP8 pass is claimed; pycodestyle/ruff are unavailable.

An independent read-only review found five Important and zero Critical issues.
One fix pass added semantic calibration/enrollment checks, explicit calibration
attachment, digest-bound owner policy, fault-preserving owner replay and
full-scope training prerequisites with per-checkpoint development enrollment.
Failing regressions and the final green suite verify the fixes.
There are no deferred Minor findings.

The audio integrity audit verified original file and decoded PCM hashes for
**6,975 recordings**: 1,895 train, 256 dev, 210 calibration, 4,418 test, 196 holdout.
Test was integrity-audited, not scored. Feature extraction covered 2,151 human
train/dev recordings; normalization used **553,467 training frames only**.
Declared identity separation is checked; acoustic aliases, near duplicates and
independent listening remain unverified.

The pinned official parser ab39857cf73dd067d068fed388c321ab160dc191 maps
**93/93 stipulated outputs** through the canonical final-application adapter.
All 414 generated cases have software projection checks. These are fixtures.

PyTorch/ONNX Runtime/ncnn FP32 checks passed **48 tensor/decision probes**
covering nine numerical lengths and 15 original human development recordings.
Final maximum head error: **1.431e-6**. Thirty PCM outcomes agree, including two
overlong-input faults; repeat inference is stable. Four actual controlled exports
agree across engines immediately below/above the 0.95 and 0.90 thresholds.
ONNX is 1,141,847 bytes. Exact ncnn bytes/hashes are in the export receipt.
These checkpoints are **untrained random/controlled software fixtures**.
The final random export uses real train-only normalization. Quantization is
separate. One actual AudioService ncnn WAV replay produced one command event
with identical application/final decisions and actions disabled throughout.
This rejected fixture command is pipeline evidence, not acoustic recall.

## Failed gates and missing inputs

All six prescribed uniform/balanced runs at seeds 53/71/89 stopped **before fitting**.
Human train lacks MESSAGE; dev lacks CALL/MESSAGE; calibration lacks CALL,
MESSAGE, LIST_REMINDERS, TEMPERATURE and CREATE_REMINDER. OptionB coverage is
25/31, 22/31 and 15/31 respectively. Reviewed role states, connected timer/alarm
constituents, localization spans, original entity enrollment and a predeclared
development retrieval policy are also missing. Unknown supervision stays masked.

There is no real selected checkpoint, fitted calibration or fresh selection lock.
Human/synthetic, OptionB/extensions and command-only/owner-aware acoustic metrics
are separately **not measured**. Numerical fixture optimization is not an
acoustic training run. Synthetic, contrastive, capacity, quantization and
extraction changes were not combined with this study.

Exact publication-v2-shipping payloads are absent. Required checkpoint:
c471f68f389bb01fa6da1c664d1878a25f0892397200a35a0d664b76b2cb468b.
The recovery receipt lists archive, ONNX, ncnn and policy hashes. Reviewed
original-PCM wake/VAD crop spans and capture/wake/endpoint fault labels are missing.
Reference/crop comparison remains blocked; no older graph was substituted.

Genuine wake/command, impostor, handover and overlap pairs, a separately trained
identity/overlap/coverage backend, and fresh independent people remain missing.
Official operator-wake/source-command mismatches stay a separate track.
Owner fixtures do not establish discrimination; overlap rejection does not
establish extraction. Pi hardware and approved numerical resource limits are
absent, and no physical device was operated.

## Commands and handoff

Use the existing root interpreter and unique run IDs. Real inputs must pass
the acquisition and binding gates before later stages.

```sh
PY=/home/justin.jose.ruaya/ME2/smart-speaker/.venv/bin/python
$PY scripts/vcm_intent_slots.py preflight --run-id preflight-new
$PY scripts/vcm_intent_slots.py audit --check-audio --run-id audit-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  $PY -m pytest -q -p no:cacheprovider scripts/test/test_vcm_intent_slots_*.py \
  scripts/test/test_speaker_demo_v1.py scripts/test/test_pcv_full_benchmark.py \
  scripts/test/test_speaker_worker.py
$PY scripts/vcm_intent_slots.py train --arm uniform --seeds 53 71 89 --run-id uniform-new
$PY scripts/vcm_intent_slots.py train --arm balanced --seeds 53 71 89 --run-id balanced-new
# After actual selected inputs exist:
$PY scripts/vcm_intent_slots.py export --checkpoint selected.pt --memory entities.json \
  --entity-policy retrieval.json --run-id raw-export-new
$PY scripts/vcm_intent_slots.py calibrate --bundle raw-bundle --run-id calibration-new
$PY scripts/vcm_intent_slots.py export --checkpoint selected.pt --memory entities.json \
  --calibration calibration.json --run-id calibrated-export-new
$PY scripts/vcm_intent_slots.py parity --bundle calibrated-bundle --run-id parity-new
$PY scripts/vcm_intent_slots.py lock --bundle calibrated-bundle --split test --run-id lock-new
$PY scripts/vcm_intent_slots.py evaluate --bundle calibrated-bundle --split test \
  --selection selection-before-test.json --run-id evaluation-new
```

Already examined tests remain regression evidence. Fresh scoring requires the
complete lock and matching runtime/data/calibration identities. On an explicitly
authorized prepared Pi, collect actual hardware/resource measurements:

```sh
/usr/bin/time -v python -m scripts.runtime.speaker_worker \
  --experimental-intent-slots --engine ncnn \
  --bundle /path/to/verified/candidate --wav /path/to/immutable/command.wav
vcgencmd measure_temp
vcgencmd get_throttled
```

Record the named Pi, threads, endpoint wait, duration/RTF, repeated p50/p95,
RSS, CPU and thermal state. Verify network-disabled inference and genuine live
owner trials after their prerequisites and physical-operation authorization.
No resource gate or hardware result is inferred from desktop timing.

Raw outputs stay in .cache/vcm-intent-slots-v1. Durable compact receipts:
[verification](../experiments/vcm-intent-slots-v1/verification.json),
[coverage](../experiments/vcm-intent-slots-v1/coverage.json),
[acquisition](../experiments/vcm-intent-slots-v1/acquisition-list.json),
[recovery](../experiments/vcm-intent-slots-v1/baseline-recovery.json),
[training](../experiments/vcm-intent-slots-v1/training-results.json),
[frontend](../experiments/vcm-intent-slots-v1/frontend-verification.json),
[audio](../experiments/vcm-intent-slots-v1/audio-verification.json),
[export](../experiments/vcm-intent-slots-v1/export-verification.json),
[application](../experiments/vcm-intent-slots-v1/application-replay.json),
and [preservation](../experiments/vcm-intent-slots-v1/preservation-verification.json).
Fourteen installed/configuration/lock hashes and the main checkout's dirty/staged
state are unchanged. After initial automatic review rejection, the user explicitly
approved the pinned pnnx/ncnn tools in the existing environment; no model was installed.
