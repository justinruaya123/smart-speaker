# Agent operating contract

## 0. Active-work hold and future maintenance skill

**Current documentation authorization: edit root `AGENTS.md` and `.gitignore`, and create/update root `REQUIREMENTS.md` and `EXPERIMENTS.md` as explicitly requested by the user. Other agents are actively working. Do not create the proposed docs/registries, move datasets, change report writers, archive/delete existing files, change branches or alter running experiments during this task.** This hold remains until the user explicitly opens a maintenance window. The architecture below is a future operating contract, not a claim that migration has happened.

This hold applies to repository organization and cleanup. It does not revoke other agents' separately authorized experiment work. Those agents should preserve current paths and follow their own task scope; no maintenance agent may interrupt or relocate their files.

Organization and archiving should be performed through a dedicated `research-repository-maintenance` skill when it is created/installed and explicitly invoked. This is a proposed skill name, not an available capability. A general file-organizer skill must be adapted to this research contract before use; its defaults never authorize destructive cleanup. Do not create/install a skill during the current hold.

An existing starting point is ComposioHQ's [file-organizer skill](https://github.com/ComposioHQ/awesome-claude-skills/blob/master/file-organizer/SKILL.md), inspected on 2026-09-20. It covers organization, duplicates, archiving and move logs, but its general cleanup rules do not establish research provenance or active-agent coordination. It is not installed by this task. Review and pin the chosen source revision when implementing the project-specific skill later.

Required skill behavior (use [EXPERIMENTS.md](EXPERIMENTS.md) to locate study summaries and detailed test evidence):

1. **Audit by default:** inventory only in-scope files and report generators; identify active agent/run ownership, frozen hashes, references, source/license notices, shared datasets and installed artifacts. Read status without interrupting other agents. Unclear ownership means leave the file alone.
2. **Prepare a concrete plan:** classify each named path as keep, consolidate, migrate, archive or delete; record its hash, reason, replacement, inbound dependencies and recovery location. A dry run prints the plan without writing files. Existing failing or historical results are evidence, not clutter.
3. **Coordinate a maintenance window:** execute only on explicitly released paths when owning agents/runs have finished or agreed to handoff. Do not kill jobs, reset worktrees or infer permission from elapsed time. Recheck status/hashes immediately before each mutation; stop that mutation if they changed.
4. **Consolidate first:** retain unique findings, citations, exact protocol/selection receipts and negative results. Fix inbound links, code consumers and generators. Preserve frozen files byte-for-byte unless a new study version is authorized.
5. **Archive reversibly:** prefer a named archive with a manifest of original path, hash, reason, experiment and restore destination. Keep evidence in version control when appropriate; ignored cache alone is not a durable archive. `docs/archive/` is not a new Markdown tree: historical Markdown stays in the flat docs folder with status recorded centrally. Large payload archives remain outside Git with a verified receipt.
6. **Delete selectively:** only named, now-redundant files meeting section 8's criteria; no glob cleanup, age-based deletion, history rewriting or removal of another agent's work. Missing recovery/provenance blocks deletion. Verify restore paths and write a maintenance receipt in `docs/DOCS.md` once the maintenance window is open.
7. **Validate and hand off:** check links, registry/path consistency, immutable hashes and relevant publisher references. Report exact moves/deletions, preserved evidence and unresolved dependencies. Do not train, install a model, publish, push or commit unrelated work as part of cleanup.

## Current runtime override: user-authorized E12, 2026-09-20

The user subsequently authorized installing the best existing direct VCM and connecting pretrained wake/VAD microphone capture. [E12](EXPERIMENTS.md#e12--wake-pipeline-v1) now uses the E02 frame-attention VCM, pretrained Hey Jarvis and Silero VAD, and `parameters.enabled=false` / `backend="none"`. The Pi publisher excludes transcription graphs and modules. Earlier descriptions below of an installed STT default and energy-VAD-only capture are historical snapshots, not the current runtime contract. Historical study files and source paths remain preserved. This implementation authorization does not open a repository maintenance window or authorize physical device actions. Pretrained activation is an explicit user exception for wake/VAD; it does not change random-initialization or human-only contracts for other studies. Current quality gates still fail. The user selected Windows microphone 26, then deferred spoken testing after capture returned all zeros.

## 1. Start here

This repository develops local smart-speaker models through isolated experiments and consolidates approved results into one installed artifact bundle. Follow the user's task boundaries, preserve unrelated changes, and read the current files before trusting an older report.

**Read [REQUIREMENTS.md](REQUIREMENTS.md) first.** It defines the core ME problem, the eight deliverables, all ten target use cases, the installed implementation gaps and completion evidence. The governing target is a tiny standalone VCM on an RPi4/5, with no LLM, cloud model calls or ASR/STT pipeline, including local, narrow, optional and downstream transcription readers. The existing STT brightness reader is a noncompliant historical baseline. Preserve evidence but do not extend, recommend or promote it. Renaming transcription as phonetic/keyword/parameter reading does not bypass the strict no-ASR rule. Do not silently change the installed configuration to resolve this gap during another agent's work.

**Maintain [EXPERIMENTS.md](EXPERIMENTS.md) as the single experiment summary and directory.** Read it before starting a duplicate study. At completion or a meaningful state change, update the relevant entry with data/splits, intents, methodology, architecture, training/selection status, test/validation results, failed gates, key insights and links to detailed evidence. Long reports, per-intent tables and raw predictions may remain in their existing files; summarize rather than copy them. Record incomplete runs as incomplete and historical ASR results as excluded from the current design. The maintenance skill defined in section 0 must use this directory before consolidation or archiving; it remains proposed, not installed.

Every experiment must identify the ME/UC requirement IDs it addresses, the measurable hypothesis and what remains unsupported. The installed nine command families plus `UNKNOWN` are not the ten user-facing use cases. Parser mappings, keyword accuracy and standalone family inference do not prove complete-command or Pi-demo coverage. Research the gaps listed in REQUIREMENTS.md without narrowing the assignment or adding action integrations unless the user authorizes that scope. Record new verified evidence in its gap map; keep detailed run findings in the experiment report.

Read the existing study contracts in section 10 and the relevant current dataset, intent and experiment files before changing their contracts. Once the planned documentation hub exists, start from `docs/INDEX.md`. Missing proposed registries are an initialization task for the future maintenance window, not permission to create them now. Repository paths in this file are relative to the root. `scripts/modules.json` is the implementation source map; `Architecture.ipynb` explains it and performs light verification.

At task start:

1. Inspect `git status --short`, staged paths and the current branch. Identify user-owned changes and the installed artifact hashes. Do not reset, clean, overwrite, stash or stage unrelated work.
2. State the experiment ID, purpose, allowed edits, baseline and intended evidence. Search the central registries before adding a dataset, vocabulary, test or model.
3. Complete the ignore-first gate below **before downloads, feature extraction, training or staging experiment files**.
4. Read the frozen protocol and selection lock. A change in data, splits, frontend, acoustic labels, catalog competition or test-driven design requires a new experiment version.
5. Use the root `.venv`, `pipeline.toml` and `requirements.lock`. Do not create a separate environment or duplicate implementation in each experiment.

## 2. Ignore first; commit only .gitignore first

Classify every planned output before creating it:

| Keep in Git | Ignore or keep outside Git |
| --- | --- |
| Source, tests, config, lockfile, source map | `.venv/`, `.runtime/`, Python/tool caches |
| Dataset manifests, source/license receipts, split and hash audits | Audio payloads, downloads, private captures, quarantined payloads |
| Versioned protocols, vocabulary/catalog registries, test definitions | Features, intermediate checkpoints, predictions and raw logs in `.cache/` |
| Compact metrics, selection receipts and durable findings | `dist/`, wheel caches, local worktrees, temporary render/export products |
| Deliberately installed `artifacts/` files and integrity metadata | Experiment weights outside `artifacts/`; credentials and `.env` |

Use scoped ignore rules. Never blanket-ignore `data/`, `experiments/`, `artifacts/`, Markdown, JSON, JSONL or CSV: these contain reproducibility evidence. `.gitignore` does not untrack a file already in Git; report tracked payloads and migrate them deliberately, without rewriting history or mass `git rm --cached`.

If new rules are needed, edit **only `.gitignore` first**, check positive and negative examples with `git check-ignore --no-index`, inspect `git diff -- .gitignore`, then stage that exact path. Verify `git diff --cached --name-only` contains only `.gitignore` before committing. If unrelated changes are already staged, preserve them and use a path-only commit (`git commit --only -- .gitignore`) after reviewing the exact `.gitignore` diff; never unstage the user's work to satisfy this rule. Check the resulting commit's file list.

The first commit for this setup must contain only `.gitignore`. Once the necessary rules are already committed, later tasks record that the gate passed; do not create empty ignore commits. Explicit user instructions not to commit take precedence: prepare/review the ignore change and report that the commit gate remains pending. Do not commit other files unless the task authorizes it. Never use `git add .` or `git add -A` in this dirty research checkout.

## 3. One repository architecture

| Location | Purpose and ownership |
| --- | --- |
| `AGENTS.md` | Root agent entry point; link to focused contracts instead of growing a diary |
| `REQUIREMENTS.md` | Authoritative ME objective, use cases, implementation gaps and acceptance evidence; read before experiments |
| `EXPERIMENTS.md` | Central experiment summaries and evidence links; update when experiments change state or finish |
| `docs/` (planned) | All new durable Markdown, in one flat folder; see lifecycle rules below |
| `registries/` (planned) | Central machine-readable intent, vocabulary and test indexes |
| `data/audio/<source>/<release>/...` | Shared immutable recording pool; reuse existing source paths |
| `data/functions/{vad,vcm,stt,tts}/index.json` (planned) | Functional views referencing shared manifests, never copied recordings |
| `data/<study-version>/` | Existing frozen manifests, catalogs, protocols and machine-readable audits |
| `data/sources/` | Dataset source/license/provenance receipts |
| `experiments/<experiment-id>/` | Versioned experiment configuration, protocol JSON, compact evidence and thin entry points |
| `scripts/models/`, `scripts/train/`, `scripts/test/` | Canonical models, fitting/data code and validation/evaluation |
| `scripts/runtime/`, `scripts/modules.json` | Runtime implementation and explicit publish allowlist |
| `.cache/<experiment-id>/<run-id>/` | Mutable run output: features, weights, ONNX, predictions, logs, timing |
| `artifacts/` | One explicitly installed working bundle; never a candidate dumping ground |
| `dist/` | Generated executable/transfer ZIP; never the source of truth |
| `STT/` | Existing research scripts and evidence; record future Markdown migration in `docs/DOCS.md` |

Function names are lowercase and stable. VAD means endpoint/capture data; VCM means intent/finite-command data; STT means parameter transcription/phonetic data; TTS means separately labeled synthesis research. A recording used by VAD and VCM has one audio payload and two references. Do not duplicate it to achieve folder segregation.

## 4. Shared datasets and continuous audits

`docs/DATASETS.md` will be the **single human-readable dataset directory and audit history** after maintenance is authorized. Do not create a dataset README/audit Markdown in every data folder. Keep detailed audits machine-readable beside their frozen manifests; link them from this directory.

For each source/version, register purpose/functions, physical payload location, manifest and source receipt, license/access constraints, natural/constructed/synthetic provenance, sample format, speaker identity policy, splits, label/transcript quality, counts, audit date/status and hash. Unknown fields are `not audited` or `unknown`, never inferred from a filename. TTS voice IDs are not independent human speakers. MLEnd labels are category annotations, not verified transcripts.

Before fitting, verify paths, file SHA-256, recording readability, source metadata, duplicate and conflicting labels, cross-split speaker/hash overlap, reserved evaluation/enrollment speakers, excluded root calibration recordings and per-intent/role coverage. Log exclusions and denominators. Keep an immutable audit for the manifest version; append a dated summary to `docs/DATASETS.md`. Do not silently overwrite a previous test audit.

Read shared recordings without modifying them. Write derived mono 16 kHz PCM/features into a cache keyed by audio hash, frontend version and settings; record the derived hash. Human-only studies use original human utterances and their declared frontend (including per-utterance 80-bin CMVN for the parameter study), not the installed VCM frontend by assumption. No concatenation, splicing, TTS or synthetic audio augmentation in human-only studies.

Never move current payloads/manifests just to match the target layout. Existing path consumers include `publish.py`, which restricts packaged samples to `data/audio/`. A migration needs a path map, all reader/writer updates, hash/split equivalence checks and preserved frozen evidence. New functional indexes can reference the current locations immediately.

## 5. Isolate experiments; consolidate proven results

Use a stable ID such as `vcm-keyword-v5` or `stt-phonetic-v3`; use distinct run IDs for seeds/retries. Register the owner, hypothesis, parent/baseline, source revision plus dirty diff receipt, allowed paths, data/protocol hashes, vocabulary/catalog versions, metrics, gates, commands and output paths in `EXPERIMENTS.md` and the experiment's protocol JSON before fitting.

Use `codex/<experiment-id>` branches when branch isolation is needed. Never switch branches over dirty work or create a worktree from HEAD while pretending it includes uncommitted research. Record the actual baseline; establish the needed changes explicitly. Worktrees must use the one configured root interpreter and shared read-only dataset pool, with unique output directories. Do not create duplicate environments or write into another run's outputs. Do not spawn agents unless the user or applicable instructions explicitly request delegation. When parallel work is authorized, assign disjoint writable paths and one integration owner for shared registries/config/artifacts.

Keep branch-specific evaluations under `scripts/test/` with experiment-specific names/configuration, registered through the source map where appropriate. Keep fixtures/configuration under the experiment. Use thin entry points in `experiments/`; reuse shared preprocessing and runtime code. Research/training modules have `runtime: false`. A regression test that protects shared behavior belongs in shared validation after review; it should not require downloading the whole research corpus or training a model.

Freeze train/dev/test before fitting. Parameter-study and human brightness candidates start from random weights; VCM continuation is a distinct declared workflow. Development selects weights and thresholds. Write `selection-before-test.json` with protocol, manifest, graph, vocabulary/catalog hashes and chosen thresholds **before** evaluating test. Never tune on test failures. Preserve failed runs and prior test receipts before changing a protocol; make a new version for follow-up research.

Consolidation sequence: candidate -> development selection -> locked test -> runtime/export compatibility checks -> explicit installation -> packaging -> separate Pi verification. A promising score does not authorize installation. Keep `training.update_artifacts=false` by default. Preserve the installed VCM and bundle. Compare per-intent complete commands, per-role results, UNKNOWN false accepts, cost and regressions. Install only within user scope after declared gates pass, then update bundle/manifest hashes and the selected-model registry evidence together. Retain the prior bundle recovery receipt before replacement. Merge reusable code, tests and compact findings; leave raw run output in the cache. Do not merge or commit unrelated research automatically.

## 6. Central intent, vocabulary and test registries

`registries/intents.json` indexes every current family, required parameter roles, contract sources and acceptance/rejection test IDs. `registries/vocabularies.json` indexes acoustic output inventories, grammars and dynamic catalogs by stable ID, version/hash, role, provenance and installed/research status. `registries/test-cases.json` indexes test definitions, expected outcomes, level of evidence, implementation pointers and result receipts. These are indexes of authoritative source contracts, not alternative runtime implementations. When initialized, definitions with no run receipt must be explicitly unexecuted; no registry is being seeded during the active-work hold.

Before adding or changing an intent, update all three indexes, identify datasets/splits and add positive, negative, malformed, ambiguous and unknown cases applicable to that intent. Keep text-only grammar cases separate from original-audio cases. Record sample IDs/hash, expected typed command or abstention, test harness, last run revision and receipt for measured cases. Never label a proposed case as passing. Report missing acoustic coverage per value/name; 0-100 parser support is not 101 tested spoken values.

The installed label order comes from `artifacts/bundle.json`; the finite contract is `data/contract.json` and `scripts/runtime/family_contract.py`. Verify consistency before modifying either. Do not append acoustic classes to an existing graph. Vocabulary/frontend changes require fresh fitting/export/calibration. Dynamic catalog changes need versioning, pronunciation or enrollment provenance, competition recalibration and unknown-name tests; a text edit cannot create acoustic evidence. See the full reader contracts in section 10.

## 7. Runtime, training and Raspberry Pi publishing

Preserve microphone/file -> mono 16 kHz PCM -> energy VAD -> accepted VCM family -> approved parameter reader -> typed validated JSON. No device execution. VCM owns intent; no reader changes/rescues a rejected family or automatically falls back to another model. `Recognizer(..., parameter_backend=None)` is standalone VCM. The retired width-256 character-convolutional ASR stays retired. The installed experimental STT brightness reader is distinct from the human-only study candidates and retains its constructed-phrase provenance.

Use section 10 for study-specific commands and release gates. From the root on Windows:

```powershell
.venv/Scripts/python.exe train.py --tests-only --no-setup
.venv/Scripts/python.exe train.py --no-setup
# Only when fitting is in scope; inspect the enabled stages first:
.venv/Scripts/python.exe train.py --fit --no-setup
# Only when packaging is in scope:
.venv/Scripts/python.exe publish.py
```

`train.py` handles setup, optional fitting, checks and held-out validation. `publish.py` uses `scripts/modules.json` runtime entries, installed VCM/STT ONNX graphs, `pipeline.toml`, integrity metadata and pinned ARM64 wheels to build `dist/smart-speaker-rpi4.zip`. It also includes bounded human smoke samples and their source receipts; it does **not** ship the shared training corpus, research candidates, trainers or PyTorch. Keep one packaging implementation. Never edit a generated executable as source.

On an explicitly targeted Pi, the existing flow is `python3 pipeline.pyz setup`, `python3 pipeline.pyz --test`, `python3 pipeline.pyz --list-mics`, then `python3 pipeline.pyz --mic <index> --vad-debug`. Use the actual listed microphone index and stay quiet during calibration. Uploading a ZIP, running setup, and operating a microphone are distinct steps; only perform the steps in task scope.

Label evidence accurately: source/static checks, parser tests, original-audio replay, CPU ONNX parity/timing, archive integrity, physical Pi tests and live microphone tests. Desktop timings exclude disk, utterance duration and the default 400 ms endpoint wait. Report identical PCM/thread settings, frontend/decoder/catalog work, p50/p95, session load, size and catalog scaling. Packaging is not deployment or quality approval. Preserve the recorded cuDNN shutdown fault and CPU export recovery distinction in the research constraints.

## 8. One Markdown folder and useful findings

All **new durable Markdown** belongs directly in `docs/`, using names such as `EXPERIMENT-vcm-keyword-v5.md`; avoid new nested report trees. Root `AGENTS.md`, the explicitly requested `REQUIREMENTS.md` and `EXPERIMENTS.md`, and the GitHub entry `README.md` are discoverability exceptions. Keep one authoritative requirements file and experiment directory; do not create duplicate copies under `docs/`. Detailed legacy test documents may remain linked from EXPERIMENTS.md while agents work. Third-party license/source documents and frozen/hash-referenced protocols are preservation exceptions. Cache-generated Markdown is disposable output; consolidate its useful findings into `docs/` when closing a run. During the future maintenance audit, list existing exceptions and migration dependencies in `docs/DOCS.md`. In particular, `STT/phonetic-v2/PROTOCOL.md` is hashed by `scripts/test/phonetic_eval.py`; preserve it. Existing `STT/` and experiment report generators still write to their current paths.

Use one report per experiment version, with a dated findings log inside it. Keep the dataset directory only in `docs/DATASETS.md`. Keep architecture/experiment navigation in `EXPERIMENTS.md` and documentation lifecycle records in `docs/DOCS.md`; do not create competing summaries.

At completion record: hypothesis and decision; baseline/candidate IDs and hashes; exact command/seed/environment; datasets/splits/protocol/selection receipt; per-intent and per-role metrics with denominators; failed gates and uncertainty; failure attribution with evidence; latency/size; reusable insights; next experiment; installed-versus-candidate state. Link compact machine evidence and raw output locations. A cause is a hypothesis until an ablation or audit supports it. Distinguish family, parameter, policy, data coverage, preprocessing and capture failures.

A Markdown file is deletable only after its unique useful content has been consolidated, inbound links and code writers have been checked, it is not required provenance/frozen evidence/user-owned work, and an explicit replacement or evidence-backed 'no remaining value' reason is recorded in `docs/DOCS.md`. Age, failure, lack of recent reads or disagreement with a current result is insufficient. Fix references/writers and validate the replacement before deleting a named file. Preserve unique negative results and original receipts. Do not bulk-delete Markdown by glob or remove third-party notices. A frozen protocol needs a new study version, not a cosmetic rewrite.

## 9. Finish checklist

- Review only the intended diff and preserve the user's staged/dirty work.
- Verify ignore rules protect payloads but expose reproducibility metadata; verify any ignore-first commit contains only `.gitignore`.
- Update dataset audit history if data changed, and experiment/intent/vocabulary/test indexes if their contracts changed.
- Run checks appropriate to the actual change. Documentation work needs link/schema/path review, not a model training run. Model changes need the declared acoustic/export/regression gates.
- Record results, failures, reusable insights and unfinished work; update the docs lifecycle rather than adding redundant status files.
- Report exactly what changed, what was committed, what was measured, and whether the installed artifact or Pi was touched. Never call a model finished because only software checks passed.

## 10. Existing study contracts (preserve during reorganization)

**Historical transcription-study directions in this section are superseded by REQUIREMENTS.md and the strict no-ASR rule.** Commands to train/install/publish an STT reader document the previous implementation; they do not authorize new transcription pipelines or their promotion. Preserve reports and active-agent files during this documentation task. Direct finite VCM/embedding work still needs architecture and quality checks.

These study-specific rules remain in this file while other agents are working. Existing paths and commands remain valid; the proposed layout does not move them. Future consolidation may extract this section only after the maintenance hold is lifted and all references are updated.

### Intent, parameters and retirement

The local flow is microphone/file → mono 16 kHz PCM → energy VAD → accepted VCM family → approved parameter reader → typed, validated command JSON. It does not execute device actions. VCM remains authoritative for intent. A parameter reader cannot change intent, rescue a rejected intent or invoke another model as an automatic fallback. `Recognizer(..., parameter_backend=None)` is the standalone VCM path.

The legacy width-256 character-convolutional **ASR model is retired**. Do not restore its runtime, trainer, model files, switches, custom-media fallback or shipping option. Its historical `data/asr.jsonl` is a retained transcription dataset, not an active model. Historical reports and backup/cache evidence may mention it. STT and ASR describe the same general transcription task; calling a new candidate STT is not proof of a cheaper architecture.

The installed experimental default is `[parameters] enabled = true` and `backend = "stt"`. This is the earlier finite-word CNN/BiGRU brightness checkpoint. It predates the human-only study and included constructed human-word phrases. Preserve that provenance. Research candidates are evaluated in isolation; do not silently install them or claim that their results validate the installed checkpoint.

Absolute brightness uses the entire utterance with timing marked unverified. The shared grammar supports integers 0–100 with optional percent, rejects malformed/conflicting/out-of-range values and preserves accepted finite relative commands. Text parser coverage is separate from acoustic coverage. Other finite VCM families retain their contracts. Custom music/device names abstain when no approved reader resolves them. No Spotify, playlist, filepath or Home Assistant integration belongs in this study.

### Sources and commands

| Source | Responsibility |
| --- | --- |
| `train.py`, `publish.py` | Environment, checks, fitting, artifact integrity, executable and Pi transfer ZIP |
| `scripts/modules.json` | Canonical implementation map; research/training modules use `runtime: false` |
| `scripts/runtime/{config,pipeline,inference,parameter_backend}.py` | Configuration, VAD, intent acceptance and reader dispatch |
| `scripts/runtime/stt_runtime.py` | Installed CPU ONNX frontend/decoder; no PyTorch |
| `scripts/models/{vcm,slot_model,stt,parameter_models}.py` | Model definitions; use einops for tensor shape operations |
| `scripts/train/brightness_{data,training,workflow}.py` | Human-only overlays, random initialization, export and guarded installation |
| `scripts/test/brightness_benchmark.py` | Development calibration and human-only brightness evaluation |
| `scripts/train/parameter_study{,_data}.py` | Frozen multi-source corpus and four random-initialization experiments |
| `scripts/test/parameter_study_eval.py` | Fixed/expanded catalogs, real-audio enrollment, held-out scoring and CPU timing |
| `scripts/phonetic.py`, `scripts/train/phonetic_{data,training}.py` | Versioned human-only phonetic follow-up; expanded LibriSpeech, frozen speaker splits and random-initialization fitting |
| `scripts/models/phonetic_reader.py`, `scripts/test/phonetic_eval.py` | Research-only posterior phrase graphs, numeric grammar, versioned catalog adapter and development-locked evaluation |
| `scripts/test/validation.py` | Runtime/parser/routing/VAD/capture checks and full dataset validation |
| `Architecture.ipynb` | Explanation and light verification; no duplicate source or training logs |
| `STT/alternatives/` | Current report, dataset/model survey, retirement receipt and pronunciation source |

Run from the root using `.venv/Scripts/python.exe` on Windows (shown as `python`):

```bash
python train.py --tests-only --no-setup
python train.py --no-setup
python scripts/parameter_study.py prepare
python scripts/parameter_study.py train --models fixed prototype phoneme character --epochs 28
python scripts/parameter_study.py evaluate
python scripts/parameter_study.py scale
python STT/alternatives/write_report.py
python STT/phonetic-v2/acquire.py
python scripts/phonetic.py prepare
python scripts/phonetic.py train --models small wide --epochs 36
python scripts/phonetic.py export --models small wide
python scripts/phonetic.py check
python scripts/phonetic.py evaluate
python scripts/phonetic.py scale
python STT/phonetic-v2/write_report.py
python scripts/brightness.py prepare --refresh-data
python scripts/brightness.py train
python scripts/brightness.py benchmark
python scripts/brightness.py install
python publish.py
```

Study training always starts from random weights, with no pretrained acoustic checkpoint. CMUdict supplies textual pronunciations only. Fixed classification and metric embeddings use a temporal depthwise residual encoder; phoneme and character baselines use a compact CNN/BiGRU CTC encoder. These are implementations inspired by published approaches, not reproductions of paper benchmark scores. Development selects weights; test never selects weights. Explicit reruns replace study candidates; preserve evidence before changing a protocol.

Default `training.update_artifacts=false` keeps newly fitted brightness candidates separate. Installation checks the exact human-only manifest, graph hashes, development precision and runtime tests. VCM continuation is separate from the from-scratch parameter study. Publishing includes only installed VCM/STT graphs and runtime modules; research models, training code, PyTorch and datasets are not shipped. A transfer ZIP is not deployment or physical Pi validation.

### Data and extension rules

**The brightness, parameter-study and phonetic-v2 human-only studies use original human recordings only. No TTS, word concatenation, spliced commands or synthetic audio augmentation in those studies.** Existing separately labeled synthetic/mixed experiments and `data/tts/` are independent research, not evidence for the human-only studies or the installed model. A TTS folder is not authorization to synthesize more audio, change a frozen protocol or mix it into a human-only manifest. New synthesis requires explicit experiment scope and an untouched real-human test. Model regularization does not change this rule. Keep historical constructed manifests under `data/stt/` unchanged; new brightness fitting uses `data/stt-human/`.

The study manifest, audit and declared catalogs are in `data/parameter-study/`. Reuse exact publisher recordings, convert stereo to mono, resample to 16 kHz and use per-utterance 80-bin CMVN features. Retain transcript/label provenance, license, speaker and file SHA-256. MLEnd word labels are canonical categories, not independently verified verbatim transcripts. Keep one microphone recording per paired utterance. Enforce speaker separation, deduplicate hashes, quarantine conflicting duplicate labels and exclude root calibration recordings.

Keep all parameter-labeled examples and a deterministic bounded background sample. Report subset counts rather than complete-dataset benchmark scores. The study is a feasibility pilot, not general-English pretraining. Freeze train/dev/test before fitting. Development chooses checkpoints and per-role thresholds; score test after all choices are written to `selection-before-test.json`. Do not rebalance test or adapt a decoder based on test errors.

Add original brightness recordings through an overlay:

```json
{"id":"my-set/speaker01/001","audio":"data/audio/my-set/001.wav","transcript":"brightness fifty percent","speaker_id":"my-set/speaker01","source":"my-set","split":"train","kind":"natural"}
```

```bash
python scripts/brightness.py prepare --extra-manifest data/my-brightness.jsonl --refresh-data
```

Supply source/license metadata and independent dev/test speakers. Data, vocabulary or frontend changes require fresh training/export/calibration. Do not append acoustic output classes to an existing checkpoint. Extend the parameter study through a new versioned manifest/protocol and `parameter_study_data.py`; preserve previous test evidence.

### Dynamic catalogs and future shipping

Fixed softmax/finite VCM heads emit only trained classes. Adding catalog text does not add acoustic classes. Audio-enrollment embeddings require original recordings of each new name, with enrollment speakers disjoint from test speakers. The study tests one, three and five available speakers per name; report missing enrollment coverage.

Phoneme matching can add text plus pronunciations without changing acoustic weights; unknown spellings need reviewed pronunciations or a separately evaluated G2P model. Character CTC accepts new text catalogs but still needs evidence of acoustic generalization. Ambiguous/homophonic names must abstain. Catalog additions change competition and require recalibration and unknown-name tests. Dictionary coverage alone is insufficient.

Before adding a study reader to runtime: implement an adapter receiving only the accepted intent/slot role and a versioned catalog. Return a typed value or entity ID with calibrated confidence/abstention; never execute actions. Register the implementation, frontend, decoder, graph/vocabulary/catalog compatibility and publisher selection. Require per-role natural-audio precision/recall, unknown false-accept rates, corpus-expansion regression, variable-length/batch ONNX parity and VCM ownership checks. Only then make it selectable; no fallback or default change without evidence and the user's scope.

The implemented phonetic follow-up stays research-only under `STT/phonetic-v2/`, `data/phonetic-v2/` and `.cache/phonetic-v2/`. Its CTC phrase graph uses training-derived brightness templates and a separate keyword/filler graph for names. The adapter supports explicit pronunciations, stable entity IDs, calibrated abstention and catalog-version invalidation; a JSON change never modifies acoustic output classes. Its connected-number confidence gate is separate from isolated words. Only absolute brightness has a connected numeric grammar; do not claim connected volume or automatic artist/title role parsing.

The recorded Windows/PyTorch build has a reproduced cuDNN GRU shutdown fault: fitting completed all epochs and wrote checkpoints, but the CUDA processes returned native exit code 3221226505. Fresh CPU export/validation succeeds. `export` is an independent recovery step, not retraining; verify the selected checkpoint/history and graph parity. `train --no-cudnn` avoids the fault in the focused GRU reproducer but is slower; a complete fitting run with this option has not been measured. Keep that distinction in reports instead of hiding the native failure or changing the reported recipe.

Preserve both evaluation views: the exact prior test for historical comparisons, and the supplemental SmartLights speaker holdout for newly fitted v2 models. Historical models saw some supplemental speakers and cannot be fairly scored there. Freeze the v2 protocol, manifest, templates, catalog versions and thresholds before test. Changing a frozen protocol requires a new study version and preserved evidence. The LibriSpeech internal speaker holdout is not its official benchmark. No candidate or catalog is approved for shipping merely because its software checks pass.

### Evidence boundaries

Report isolated words, connected numeric commands, artists, songs, room-name proxies and unknown speech separately. A role supplied from annotations is conditional parameter evaluation, not full-pipeline accuracy. Room/place labels are proxies for Home Assistant names; catalog fixtures are not Spotify integrations.

Time identical preloaded PCM on CPU ONNX with the same thread count, including frontend, decoder and catalog matching. Report p50/p95, session loading, parameters/bytes and catalog scaling separately. Explicitly exclude disk, speech duration and endpoint waiting; the default endpoint wait is 400 ms. Desktop results do not establish Pi or live-microphone performance.

Keep failed quality gates visible. Do not promote a candidate using pooled word accuracy when natural names/numbers fail, relax confidence to conceal failures, or present 404 parser checks as 404 successful spoken examples.
