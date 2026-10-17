# Shipping and Hugging Face dataset implementation plan

> **For agentic workers:** Use the repository's execution and verification contracts to implement future tasks separately. This task prepares documentation and read-only access evidence only. No publishing, pushing, remote branches/PRs/tags, training, installation, dataset relocation or cleanup is authorized.

**Goal:** Deliver a reviewable final project structure and a versioned, attributable dataset release that reproduces the exact final experiment.

**Architecture:** Preserve the current research checkout and frozen experiment identities. Build a dedicated, allowlisted Hugging Face staging directory containing portable selection/provenance evidence; include recording bytes only after a source-specific distribution decision. Keep dataset release, experiment source/evidence and Pi runtime packages independently versioned and cross-linked by hashes.

**Tech stack:** Existing Python environment and locks, JSON/JSONL provenance, Hugging Face `hf` CLI / `huggingface_hub`, embedded-audio Parquet if audio is cleared.

**Spec:** User request dated 2026-10-16; [REQUIREMENTS.md](../REQUIREMENTS.md), [dataset directory](DATASETS.md), final [experiment protocol](../experiments/speaker-final-refinement-v1/protocol.json).

## 2026-10-17 scope extension: both submitted model identities

The current dataset/reproducibility preparation covers **935k pipeline repair saved epoch 28** as well as **2M final-refinement control epoch 32**. The original final-refinement-only layout below remains a plan, not an exported/published state. Preserve separate per-model fitting-ID manifests and selection locks: compact has **23,289** eligible rows (1,863 human/21,426 synthetic); 2M control has **23,295** (1,869/21,426). Six unresolved Timers and Such slot-conflict rows are excluded only from compact. Shared normalization still uses 1,869 human utterances. The shared 23,479-row completion superset and all 66,390 auxiliary numerals must not be mislabeled as either selected fitting membership.

Prepared now: [original publisher citations and DOI distinctions](DATASET-CITATIONS.md), [reusable BibTeX](dataset-citations.bib), [source-by-split counts](DATASETS.md#submitted-935k-and-2m-models-verified-input-membership), [two-model card additions](HF-DATASET-CARD-DRAFT.md), [slide fields and review gaps](REPRODUCIBILITY-935k-2m.md), and [fresh local input/selected-artifact audit](model-dataset-usage.json). The audit binds the repaired compact graph/archive rather than its older epoch-32 predecessor. There is no new derivative DOI, tested row-level portable export or public weight/source release from this work.

For a future staging layout, add `releases/speaker-pipeline-repair-v1/` alongside `releases/speaker-final-refinement-v1/`, retaining shared source material once and referencing it by hashes. The selected compact checkpoint belongs to the existing `speaker-compact-recall-v1` fit; preserve that lineage and history, not a fictitious extra repair fitting run. Source-specific recording inclusion, complete dirty-source closure, immutable selected checkpoints/histories, clean-cache reconstruction and Pi measurements remain necessary. The [official Hugging Face DOI workflow](https://huggingface.co/docs/hub/doi) can mint a DOI for an actual released revision; the upstream primary DOI is not a substitute for a derivative identifier. Existing target configurations and original frozen evidence remain preserved.

## Global constraints

- Do not publish yet. `hf upload`, `--create-pr`, remote branch creation and tags all mutate the Hub and remain unexecuted.
- Preserve active changes, existing source paths, frozen files, installed artifacts and all prior negative evidence. This preparation is not a repository-wide maintenance window.
- Final shipping inference remains pure VCM: no ASR/STT pipeline, LLM, cloud inference or TTS generator.
- Human-only historical protocols retain original-human-only rules; synthetic research stays explicitly labeled.
- Exact source revisions, split membership, labels, hashes and selection locks define the experiment; `main` is not an experiment version.

## Expected final project structure

This is the delivery boundary, not a proposal to move today's files. Existing legacy studies stay accessible through `EXPERIMENTS.md`. Trees abbreviate unrelated existing files. The current baseline is branch `codex/speaker-demo-v1`, HEAD `5d872e3a249e5876aaa8130b76f6e3b31622355f`, with uncommitted research: HEAD alone cannot reproduce it.

```text
smart-speaker/                          # source and research evidence
  README.md                            # setup, reproduction, dataset revision, limits
  AGENTS.md
  REQUIREMENTS.md
  EXPERIMENTS.md                        # experiment directory
  requirements.lock
  pipeline.toml
  start.py / release.py / publish.py    # retain their existing distinct interfaces
  scripts/{models,train,test,runtime}/
  vcm/
  SpeakerUI/
  data/
    sources/                           # acquisition hashes and upstream notices
    audio/ai231-pool-v1/                # local immutable bytes; ignored in Git
    <historical-study>/                # existing frozen protocols/audits
  experiments/
    speaker-demo-v2-data/               # installed source/view audit
    speaker-synthetic-completion-v1/    # prompts, voices, generation receipts
    speaker-final-refinement-v1/        # protocol, selection locks, results, package receipt
    <other-existing-study>/
  docs/                                # flat durable documentation
    DATASETS.md
    PLAN-shipping-and-huggingface.md
    HF-DATASET-CARD-DRAFT.md
    dataset-provenance.snapshot.json
    hf-access-check.json
    <existing-reports>.md
  .cache/                              # ignored features/checkpoints; recover exact evidence
  dist/                                # ignored generated release products
    speaker-final-refinement-v1/speaker-pi4-final.zip
    hf-dataset/<release-id>/           # FUTURE dedicated upload staging, absent today
```

The proposed Hub payload below supplements the already populated target; it does not overwrite its default data. Evidence is versioned below a release directory. The target root card will need a reviewed integration of the local card draft. The selected model is **control epoch 32**, with a 23,295-row fitting pool; preserve the 23,479-row completion superset separately and export explicit per-arm membership. Newly generated development examples participate in the shared comparison even though new completion training examples are excluded from the winner.

```text
justinruaya123/ai231-smart-speaker/
  README.md                            # merged dataset card, explicit configs, limitations
  LICENSE.md                           # scoped project terms plus per-source matrix
  data/...                             # existing upstream-derived configs: preserve
  supplemental_fil/...                 # existing; rights unresolved
  supplemental_synth/...
  synthetic_negatives/...
  releases/speaker-final-refinement-v1/
    provenance/
      sources.json                     # original source identities and checksums
      transformations.jsonl            # versioned label/audio/feature mutation records
      selection.json                   # inclusion/exclusion rules, seeds, exact row IDs
      source-map.json                  # portable ID -> pinned shard/row -> local path map
      checksums.sha256                 # generated payload hashes, excludes itself
      environment.json                 # exact versions/platforms/determinism limits
      code-revision.json               # immutable source commit; dirty-source closure if needed
    manifests/
      human/{train,dev,calibration}.jsonl
      synthetic/{train,dev,calibration}.jsonl
      evaluation/{test,holdout}.jsonl
      auxiliary.jsonl
      excluded.jsonl
      excluded-training.jsonl
      arms/{control,completion}-train-ids.jsonl
    protocols/
      data-installation.json
      synthesis.json
      training.json
      selection-before-calibration.json
      selection-before-test.json
    catalogs/                          # finite commands, schema, lexicon, normalization
    evidence/                          # compact metrics and failed gates
    reconstruction/                    # pinned scripts + dependency locks + commands
    audio/<cleared-source>/...         # OPTIONAL; never unresolved or private payloads
```

The Pi deliverable is separate. The existing final-refinement ZIP's actual central directory contains `pi_test.py`, `wake_gate.py`, `models/default/` (model/catalog/policy/normalization/schema and wake graphs), `scripts/runtime/`, `vcm/`, `pcvcm/`, `wheels/`, `samples/`, `licenses/`, `evidence/`, config, dependency lock, manifest and handoff. It is a diagnostic ARM64 CPython 3.12 package; [package receipt](../experiments/speaker-final-refinement-v1/package.json) says acceptance disabled and physical Pi/live microphone unmeasured. Review the SLURP smoke-audio notice discrepancy before distribution. The broader UI/ncnn publication in README is a different package profile and must not be represented as this exact final-refinement artifact.

## Current Hugging Face sanity check

Checked 2026-10-16. [Sanitized receipt](hf-access-check.json) records the actual requests; no credential values were written.

- Root `.venv/bin/hf` exists but is not on the ambient PATH; `huggingface_hub` is **0.36.2**. No package or skill was installed/upgraded.
- `get_token()` found no environment/cached token in this session. Authenticated write access is **unverified**.
- A network-enabled read-only GET of the dataset API returned **200**, public/ungated, commit **`3ed018c1bc731124a1b2ac5bd6653b5cd01bf576`**. The unauthenticated write-permission GET returned **401**. That proves neither authenticated denial nor push capability. Initial sandbox DNS failures are retained in the receipt.
- Existing payloads include primary, supplemental and negative configs. The existing target card has count/provenance inconsistencies relative to local final inputs; see [DATASETS.md](DATASETS.md).
- Nothing was uploaded, changed, deleted or published. No sentinel-file push was attempted.

The user's [agents CLI guide](https://huggingface.co/docs/hub/en/agents-cli) uses the ordinary `hf` CLI and an optional version-generated agent skill. Local 0.36.2 lacks the guide's newer `hf skills` command. Prepare a reviewed CLI/dependency upgrade separately if desired; the current environment remains unchanged. [Official upload reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli#hf-upload) and local `hf upload --help` provide **no upload `--dry-run`**. `--create-pr` also uploads. Offline validation and an upload-file allowlist are the preparation check.

For a later authenticated check, configure a repository-scoped token securely outside chat and tracked files. With a current compatible SDK, the [documented permission probe](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.auth_check) is:

```python
from huggingface_hub import HfApi
HfApi().auth_check("justinruaya123/ai231-smart-speaker", repo_type="dataset", write=True)
```

The inspected 0.36.2 method lacks `write`. Its compatible non-mutating fallback is an authenticated GET through library-built headers; do not run until a token is configured and network access is available:

```python
from huggingface_hub import get_token
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

token = get_token()
if not token:
    raise SystemExit("No Hugging Face credential configured; write access unverified")
response = get_session().get(
    "https://huggingface.co/api/datasets/justinruaya123/ai231-smart-speaker/auth-check/write",
    headers=build_hf_headers(token=token), timeout=20,
)
hf_raise_for_status(response)
print("Dataset content-write permission verified; no upload performed")
```

This endpoint is a GET in [Hugging Face's implementation](https://github.com/huggingface/huggingface_hub/blob/v2.2.0/src/huggingface_hub/hf_api.py). `whoami`, public read access, ownership inferred from a name and broad token scope alone do not establish permission for this exact dataset. A successful check would establish permission at that time, not guarantee later storage/quota or upload success.

## Tasks remaining before any upload

### 1. Resolve source identity and inclusion decisions

**Inputs:** `data/sources/`, composite cards, `DATASETS.md`, source manifests. **Outputs:** reviewed source/license matrix and portable `sources.json`.

- [x] Document known source citations, versions, selections and unresolved rights.
- [ ] Resolve SLURP audio/text and MS-SNSD code/audio mismatches; recover FSDD/MLEnd and composite clip-level pins.
- [ ] Record consent scope for student/reference voices; exclude unresolved audio and private metadata from staging. A public metadata release still needs a privacy/rights review of its text and IDs.
- [ ] Preserve the target's existing state; compare proposed additions by path/hash without deleting or silently overwriting existing data. Publication cannot cure rights issues already present there.

### 2. Freeze portable experiment evidence

**Inputs:** installer inventory/views; completion `prepared-v1`; final selection locks. **Outputs:** versioned manifests, path map, transformation ledger, environment/code receipt and checksum inventory.

- [ ] Copy frozen manifests byte-for-byte into a preservation archive; separately export relative-path representations with new hashes and a reversible path map. Never silently replace absolute paths inside old hash-bound files.
- [ ] Preserve each source row's repo/revision/config/shard/row index, container/PCM hash, speaker/pair identity, upstream and experimental split, human/synthetic kind, raw/derived label, reference status, metric eligibility and exclusion reason.
- [ ] Resolve the complete dirty-source dependency closure, then freeze a reviewed code commit or source archive with file hashes. Current HEAD alone is insufficient.
- [ ] Retain 26 pronunciation exclusions, all uncertain test references, 66,390 auxiliary numerals and source movement disclosures. Distinguish inventory, fitting eligibility and metric eligibility.
- [ ] Preserve exact generated command and anchor bytes if distributable, plus prompts, model revisions, per-attempt seeds and native/derived hashes; otherwise disclose the resulting reconstruction limit.

### 3. Build and verify the local staging directory

**Files to implement later:** a focused dataset export entry point under `scripts/`, its offline integrity checks, and `dist/hf-dataset/<release-id>/`. Keep `publish.py`'s runtime contract intact.

- [ ] Use explicit file allowlists; reject credentials, private captures, unrestricted copies of `.cache/`, model weights and unrelated experiments. Existing `/dist/` ignore already protects generated staging; provenance documents remain trackable.
- [ ] For evidence-only release, keep manifests outside data globs. Declare them reference tables, not a waveform dataset. Reconstruct recordings through pinned upstream access with hash verification.
- [ ] If audio is cleared, use explicit [Hub configs](https://huggingface.co/docs/hub/en/datasets-manual-configuration). Embedded-audio [Parquet](https://huggingface.co/docs/hub/en/datasets-audio#parquet-format) must contain bytes; [AudioFolder](https://huggingface.co/docs/datasets/en/audio_dataset#audiofolder) must use `file_name` relative to its metadata. Workstation paths alone cannot supply recordings.
- [ ] Reconcile all rows against original receipts, verify file/PCM/speaker/pair overlap and QA/exclusion denominators, inspect accidental evaluation leakage, then load from a clean temporary directory. Test missing/upstream-changed files fail clearly. Export must not train or alter experiment splits.
- [ ] Review README metadata and its rendered configurations: `default` remains the existing dataset; use a new version-specific name for any cleared experiment data. Never reinterpret all JSONL sidecars as samples.

### 4. Publication and post-publication verification — expressly deferred

**Preconditions:** authenticated write probe passes; source disposition and exact staging diff reviewed; user explicitly authorizes publishing.

The following is a future command example, **not executed**. It would upload the reviewed directory and can create a remote branch; approval must cover that mutation. Choose the release branch only after reviewing the existing target contents:

```bash
.venv/bin/hf upload justinruaya123/ai231-smart-speaker \
  dist/hf-dataset/speaker-final-refinement-v1 . \
  --repo-type dataset --revision release-candidate-speaker-final-v1 \
  --commit-message "Add versioned final experiment reproducibility evidence"
```

Never invoke this from the whole project directory, add `--delete`, or use a background upload schedule. `--private` does not make an existing public repository private. A new branch/PR is visible remote publication; it is not a permission test.

After an authorized upload, capture its immutable commit SHA, verify downloaded checksums in a clean cache, and rerun source reconstruction/count/overlap checks. Bind the model recipe to that dataset SHA, exact config/view IDs and source code revision. A [pinned `load_dataset` revision](https://huggingface.co/docs/datasets/en/loading#hugging-face-hub) must be the final commit hash; substitute real values only after publication. For evidence-only releases, use `hf download --repo-type dataset --revision <SHA>` and reconstruction rather than claiming an audio configuration exists.

## Review focus and completion status

Review specifically: current target versus fitted-source mismatch; missing/private/restricted sources; absolute paths and hashes changed by portability export; alias/paired-channel leakage beyond exact file hashes; online/synthetic nondeterminism and missing byte preservation. Each belongs to tasks 1–3's validation. Acceptance-disabled status, previously exposed tests, missing human intents and physical Pi/live evidence must survive every release card/report.

Completed now: source/path inspection, expected structures, source/selection ledger, read-only network probe, local CLI compatibility inspection, card draft and hash-bound documentation snapshot. Remaining: credentials/write verification, source decisions, executable portable export/reconstruction, staging validation and physical Pi evidence. Publishing remains deferred by explicit user instruction.
