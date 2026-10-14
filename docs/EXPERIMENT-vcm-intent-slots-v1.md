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

## Tests, benchmark and speaker diversity follow-up — 2026-10-15

The user requested completion of tests, holdout/benchmark execution and speaker
diversity documentation, then directed an upstream repository check. The
[follow-up protocol](../experiments/vcm-intent-slots-v1/evaluation-protocol.json)
was written before historical scoring. No fitting, calibration, selection or
threshold change followed exposure. The original design, prepared views and
installed artifacts remain unchanged.

### Executed evaluation and remaining qualification

All **4,614 previously examined recordings** ran through the actual experimental
server-side `final_application` decision on PyTorch, ONNX Runtime and ncnn:
13,842 engine outcomes. Each replay checks the original WAV and decoded PCM
hashes. Actions are disabled and ownership is disabled for the whole named
command-only track. `software-export-03` is explicitly random/untrained, bound
to checkpoint `58369c6a…`; these outcomes establish execution and accounting,
not the quality of a trained candidate.

| Historical partition | Kind | OptionB supported | Extension supported | Negative | Ambiguous | Accepted exact | Faults |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | Human | 609 | 27 | 229 | 185 | 0 | 2 |
| Test | Synthetic | 3,368 | 0 | 0 | 0 | 0 | 0 |
| Holdout | Human | 80 | 3 | 11 | 2 | 0 | 0 |
| Holdout | Synthetic | 100 | 0 | 0 | 0 | 0 | 0 |

Metrics are identical across engines. Every supported rejection is a failure:
test human accepted recall is 0/636 and synthetic 0/3,368; holdout human 0/83
and synthetic 0/100. Accepted precision is undefined because no command was
accepted. The 240 unambiguous negative trials reject correctly; the
187 ambiguous references remain separate. The two faults are inside the 185
ambiguous test rows: original durations 8.640 and 9.375 seconds exceed the
unchanged eight-second capture limit. They are never successful OOV rejections.
OptionB membership here means typed-output membership, not coverage of all
93 spoken phrase definitions. Per-intent and supplied-slot denominators are in
the [evaluation receipt](../experiments/vcm-intent-slots-v1/historical-evaluation.json).

The canonical adapter's actual records pass the pinned official `LineParser`:
4,416 test and 196 holdout command records per engine, plus two separately kept
fault records. There are zero serializer/parser mismatches and zero semantic
final-decision or fault disagreements across all 4,614 inputs. Floating-point
diagnostic scores differ by at most 1.390e-7; they are not asserted byte-identical.
The earlier threshold, short/tail/padding and head-tensor parity evidence remains
in [export verification](../experiments/vcm-intent-slots-v1/export-verification.json).

Observed one-thread host test p50/p95 application latency is 17.14/20.11 ms for
PyTorch, 15.01/17.52 ms for ONNX Runtime and 16.05/20.39 ms for ncnn. The ncnn
test p95 RTF is 0.01545. This measures frontend, model and decision on the current
Linux x86_64 host; it excludes WAV loading and does not measure microphone
capture, endpoint waiting, Pi RSS/CPU, thermal state or physical ownership.

Fresh test and holdout commands without a selection lock still exit 2. Creating
a lock from this fixture exits 2 because frozen real calibration is absent.
Owner replay exits 2 because genuine pairs and an independently validated
identity/overlap/coverage backend are absent. All six prescribed human-only
uniform/balanced 53/71/89 arms remain stopped before fitting. The 95% accepted
human recall target fails for the untrained export; precision, genuine retention,
wrong-speaker acceptance and Pi qualification remain unmeasured. No synthetic,
capacity, quantization or extraction intervention was merged into this arm.

### Speaker composition and supplied demographic metadata

The frozen views contain **3,507 human recordings and 337 declared source-qualified
IDs**. Declared speaker, pair and PCM cross-split groups are zero. Cross-source
aliases, near duplicates and physical identity remain unverified. The 31 synthetic
voice IDs in evaluation do not count as human people; clone ancestry is unresolved.

The prepared views carry no gender/sex field. The registered Fluent Speech
Commands archive was recovered at its pinned revision; its entire
1,552,390,845-byte payload matches SHA-256 `c9fd67f2…`. The extracted supplied
demographic CSV matches `a7d6df25…`. It has 101 records, of which **87 IDs and
459 recordings** join the frozen views by exact `FluentSpeechCommands/speakerId`;
14 records do not join. No archive audio was extracted into a fitting/evaluation
view. Only aggregate metadata is committed.

| Split | Human recordings | Declared IDs | Supplied female IDs | Supplied male IDs | Gender unknown IDs | Synthetic recordings / voice IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,895 | 182 | 20 | 22 | 140 | 0 / 0 |
| Development | 256 | 36 | 3 | 7 | 26 | 0 / 0 |
| Calibration | 210 | 24 | 6 | 1 | 17 | 0 / 0 |
| Historical test | 1,050 | 92 | 14 | 13 | 65 | 3,368 / 29 |
| Historical holdout | 96 | 3 | 1 | 0 | 2 | 100 / 2 |
| Total | 3,507 | 337 | 44 | 43 | 250 | 3,468 / 31 |

Supplied gender covers 25.8% of declared human IDs and 13.1% of recordings.
The known subset has 273 female-labeled recordings and 186 male-labeled
recordings; nearly equal person counts do not establish equal recording or
command coverage. Age ranges are supplied for the same 87 IDs: 66 aged 22–40,
19 aged 41–65 and two aged 65+. First language, work/school language and
self-reported fluency are kept as distinct source fields. No native language,
utterance language, sex or recording device is inferred. Missing metadata stays
unknown, and demographic fields never route recognition or establish identity.
The original FSC paper describes participant-provided demographics and
person-disjoint splits. [Lugosch et al., sections 3.2–3.3](https://arxiv.org/pdf/1904.03670).

Publisher accent labels also reveal sparse cells: 818 recordings come from ten
`Filipino (group)` IDs, while the 210-recording calibration view has no such
group-labelled recording. Forty-one IDs have no known accent label. These are
publisher categories, not listening-verified accents or independent identities.
Holdout human evidence consists of only three declared IDs, with gender known
for one. No population-wide or gender-specific accuracy claim is justified.
Exact distributions and provenance are in
[speaker-diversity.json](../experiments/vcm-intent-slots-v1/speaker-diversity.json).

Keep **each person in one split first**. For a later newly versioned collection,
use voluntarily supplied demographics to balance representation within splits
alongside command/value, accent, duration and recording conditions; retain an
unknown stratum. Report errors and sample counts by person and supplied group.
Splitting only into male/female categories cannot prevent person leakage or
verify wake ownership. The current evidence does not establish gender imbalance
as the cause of failures. Documenting subgroup definitions and how labels were
obtained follows [Datasheets for Datasets, composition and collection questions](https://arxiv.org/html/1803.09010v8).

### Upstream availability audit

The latest primary revision, `bfca0bbebfe17e00d6a685d1da573615d29ecdce`, changes
all four command shards. Each was downloaded and checked against its published
LFS size/SHA separately from the frozen inputs. Metadata counts are 10,733 train
(2,359 human), 4,443 test (824 human) and 202 holdout (96 human). All seven
publisher-labelled human training MESSAGE rows remain unsupported under the
unchanged exact reference rules: they request email/Facebook operations or
dictated content beyond the compose-only schema. The new revision supplies no
reviewed role-state, connected grammar or acoustic-span columns, enrollment or
wake-owner pairs. Its three holdout human IDs all occur in the previously examined
views; no fresh people were found. Updated test/holdout MESSAGE examples cannot
be moved into training. This is an availability preview, not new independent
label verification or scoring. [Pinned checked source](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/tree/bfca0bbebfe17e00d6a685d1da573615d29ecdce).

The supplemental repository remains at `230c8b8…` and remains synthetic cloned
speech. The linked `airimonda/vcm-me2` models declare a 64-mel frontend, 20 command
outputs, six three-value slot heads and threshold 0.4. Their interface and policy
do not implement the approved role architecture or complete personal scope.
No upstream model was installed or substituted. Authenticated GitHub inspection
finds no release assets in this smart-speaker repository; the exact shipping
baseline and crop labels remain missing.
[Checked upstream model interface](https://github.com/airimonda/vcm-me2/blob/ccf56db497c362a6b2334faa114a6e3c8c3719a9/models/vcm_conformer_M.json).
See [upstream-availability.json](../experiments/vcm-intent-slots-v1/upstream-availability.json)
for revisions, hashes, sources and limits.

### Tests and reproducible commands

Final focused/relevant verification is **172 passed, 24 exporter deprecation
warnings**: 112 experiment tests and 60 existing related tests. Reporting changes
were tested red-to-green, including synthetic/human person separation, duplicate
IDs, cluster resampling, faults/missing outcomes, timing, supplied metadata joins
and prohibited overwrites. Independent reviews reproduced 26 then 28 reporting/
training tests and verified the source CSV/sidecar values; neither found a blocking
finding. The broad historical suite remains failing: 626 passed, 20 failed,
13 skipped, 13 errors. A separate `scripts/test tests` run reports 387 passed,
12 failed and six skipped. Missing old artifacts/dependencies, sibling import
assumptions, a frozen parent CRLF identity mismatch and absent UI/dictionary
outputs are recorded; no historical integrity gate was weakened to obtain a pass.

```sh
PY=/mnt/jfs_hpc/home/justin.jose.ruaya/ME2/smart-speaker/.venv/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  $PY -m pytest -q -p no:cacheprovider scripts/test/test_vcm_intent_slots_*.py \
  scripts/test/test_speaker_demo_v1.py scripts/test/test_pcv_full_benchmark.py \
  scripts/test/test_speaker_worker.py
$PY scripts/vcm_intent_slots.py diversity --run-id diversity-new
# The optional audit-inputs.json binds the recovered CSV and exact-ID sidecar.
$PY scripts/vcm_intent_slots.py diversity \
  --inputs .cache/vcm-intent-slots-v1/upstream-fluent-demographics/audit-inputs.json \
  --run-id diversity-with-supplied-metadata-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  $PY scripts/vcm_intent_slots.py evaluate \
  --bundle .cache/vcm-intent-slots-v1/software-export-03/bundle \
  --engine ncnn --split holdout --regression --run-id historical-holdout-new
```

Repeat the last command explicitly for `test`/`holdout` and
`pytorch`/`onnxruntime`/`ncnn` using unique run IDs. Exact executed argv and real
receipts are in [historical-evaluation.json](../experiments/vcm-intent-slots-v1/historical-evaluation.json).
Test logs, red/green evidence and gate exit codes are bound in
[followup-verification.json](../experiments/vcm-intent-slots-v1/followup-verification.json).
Fourteen protected installed/configuration/lock hashes, all five view hashes,
and the main dirty/staged state remain unchanged;
[followup-preservation.json](../experiments/vcm-intent-slots-v1/followup-preservation.json)
records no environment installation or physical device operation in this follow-up.
