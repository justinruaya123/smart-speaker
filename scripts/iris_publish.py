"""Publish only reviewed, hash-verified Iris release assets to Hugging Face.

Authentication uses huggingface_hub's local token store. Tokens and private
settings are never copied into a staging tree or publication receipt. The source
recipe lives on GitHub main; this publisher never changes GitHub visibility.
Model publication requires a private repository and preserves dataset visibility.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import uuid

from scripts.iris_release import (ROOT, HERE, CACHE, read, save, sha, relative_path,
                                  name_under, validate_release, verify_bindings, verify_package)

MODEL_REPO = 'justinruaya123/ai231-smart-speaker'
DATASET_REPO = MODEL_REPO
SOURCE_REPO = 'https://github.com/justinruaya123/smart-speaker'
ROOT_SOURCE_FILES = {'AGENTS.md', 'REQUIREMENTS.md', 'EXPERIMENTS.md', 'README.md', 'MODEL_CARD.md',
                     'train.py', 'reproduce.py', 'publish.py', 'start.py', 'pipeline.toml', 'requirements.lock'}
AUDIO_SUFFIXES = {'.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a'}


def validate_publication(root, plan):
    """Finish every local identity/access check before creating a remote repo."""
    if plan.get('model_visibility', 'private') != 'private':
        raise ValueError('Model publication requires model_visibility="private"')
    if plan.get('schema_version') != 1 or plan.get('verification', {}).get('passed') is not True:
        raise ValueError('Publication requires a passed software/package verification receipt')
    files = plan.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('Publication must declare explicit files')
    seen = set()
    for item in files:
        path = relative_path(root, item['local_path'])
        destination = item['path_in_repo']
        relative_path(root, destination)
        pure = PurePosixPath(destination)
        local = PurePosixPath(item['local_path'])
        if ('reproduction', 'source') in zip(pure.parts, pure.parts[1:]):
            raise ValueError('New source mirrors are prohibited; publish the recipe on GitHub main: ' + destination)
        if (local.name == '.env' or (local.name.startswith('.env.') and local.name != '.env.example')
                or '.git' in local.parts or local.name in ('token', 'stored_tokens', 'credentials')):
            raise ValueError('Private settings cannot be published under another name: ' + item['local_path'])
        if (pure.name == '.env' or (pure.name.startswith('.env.') and pure.name != '.env.example')
                or '.git' in pure.parts or 'node_modules' in pure.parts or '__pycache__' in pure.parts
                or pure.name in ('token', 'stored_tokens', 'credentials')):
            raise ValueError('Private settings cannot be published: ' + destination)
        key = (item.get('repo_type'), item.get('repo_id'), destination)
        if key in seen:
            raise ValueError('Duplicate publication destination: ' + destination)
        seen.add(key)
        if item.get('repo_type') not in ('model', 'dataset') or not re.fullmatch(r'[^/\s]+/[^/\s]+', item.get('repo_id', '')):
            raise ValueError('Invalid publication repository')
        expected = item.get('sha256', '')
        if not re.fullmatch(r'[0-9a-f]{64}', expected) or not path.is_file() or sha(path) != expected:
            raise ValueError('Publication file identity mismatch: ' + item['local_path'])
        if path.suffix.lower() in AUDIO_SUFFIXES:
            rights = item.get('redistribution', {})
            if rights.get('allowed') is not True or not rights.get('license') or not rights.get('source_receipt'):
                raise ValueError('Audio requires explicit redistribution and source/license evidence: ' + destination)
            notice = relative_path(root, rights['source_receipt'])
            if not notice.is_file() or sha(notice) != rights.get('source_receipt_sha256'):
                raise ValueError('Audio redistribution source receipt identity mismatch: ' + destination)
    verification = plan['verification']
    if 'bindings' in verification:
        verify_bindings(root, verification['bindings'])
    return plan


def file_item(root, path, destination, repo_type='model', repo_id=None, **extra):
    path = Path(path)
    return {'local_path': name_under(root, path), 'path_in_repo': destination,
            'sha256': sha(path), 'bytes': path.stat().st_size,
            'repo_type': repo_type, 'repo_id': repo_id or (MODEL_REPO if repo_type == 'model' else DATASET_REPO), **extra}


def card_linked_files(root, cards):
    """Declare exact local evidence links without uploading a source mirror."""
    prefix = 'https://huggingface.co/' + MODEL_REPO + '/blob/main/'
    files = []
    for card in cards:
        for link in re.findall(r'\]\(([^\s)]+)\)', card.read_text(encoding='utf-8')):
            if not link.startswith(prefix):
                continue
            destination = link[len(prefix):].split('#', 1)[0].split('?', 1)[0]
            if destination.startswith('evidence/'):
                filename = destination.removeprefix('evidence/')
                if PurePosixPath(filename).name != filename:
                    raise ValueError('Card-linked evidence must name a declared flat receipt: ' + destination)
                name = HERE + '/' + filename
            else:
                continue
            path = relative_path(root, name)
            if not path.is_file():
                raise ValueError('Card-linked publication evidence is missing: ' + name)
            files.append(file_item(root, path, destination))
    return files


def published_links(markdown):
    """Keep model evidence on Hugging Face and source/documentation on GitHub main."""
    base = 'https://huggingface.co/' + MODEL_REPO + '/blob/main/'
    source = SOURCE_REPO + '/blob/main/'
    old_tree = 'https://huggingface.co/' + MODEL_REPO + '/tree/main/reproduction/source'
    def target(match):
        original = match.group(1)
        path = original.removeprefix('../').removeprefix('./')
        if path.startswith(base + 'reproduction/source/'):
            return '](' + source + path.removeprefix(base + 'reproduction/source/') + ')'
        if path == old_tree or path.startswith(old_tree + '/'):
            return '](' + SOURCE_REPO + '/tree/main' + path.removeprefix(old_tree) + ')'
        if path.startswith('experiments/iris-release-v1/'):
            path = 'evidence/' + path.split('/')[-1]
        elif (path.startswith(('docs/', 'data/sources/', 'scripts/', 'SpeakerUI/'))
              or path.split('#', 1)[0].split('?', 1)[0] in ROOT_SOURCE_FILES):
            return '](' + source + path + ')'
        else:
            return match.group(0)
        return '](' + base + path + ')'
    return re.sub(r'\]\(([^)\s]+)\)', target, markdown)


def write_cards(root, release, workflow, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / 'model-README.md'
    dataset = directory / 'dataset-README.md'
    evidence = '\n'.join('- [ ' + Path(p).name + ' ](evidence/' + Path(p).name + ')' for p in release.get('evaluation_receipts', []))
    model.write_text(f'''---
language:
- en
tags:
- audio
- audio-classification
- finite-commands
- onnx
- ncnn
license: other
license_name: component-specific-research-terms
license_link: https://huggingface.co/{MODEL_REPO}/blob/main/NOTICES.md
---

# Iris finite voice commands

Iris maps whole spoken commands directly to a finite typed command inventory.
Runtime inference uses the local command network, our Hey Iris wake network,
Silero VAD and the local owner embedding model. It has no ASR/STT transcription,
LLM, cloud inference or TTS model in its recognition path.

The Raspberry Pi archive contains the compiled local interface, ncnn and ONNX
assets, integrity manifest and software evidence. Extract it into a new folder,
run `python3 start.py --list-mics`, then use an actual listed microphone index.
Initial dependency setup needs network access; recognition operates locally.

Live actions require explicit trusted configuration and provider credentials.
WAV uploads, benchmark replay and recorded fixtures never execute actions.
The wake speaker creates a session-only reference for one command; the
reference resets after the command. No persistent owner enrollment is used.
Missing or uncertain speaker/continuity/overlap evidence fails closed. Speaker
embeddings do not prove liveness or establish household security.

## Evidence and limitations

Strict model qualification is unresolved. Physical Raspberry Pi timing, live
microphone quality, household ownership/overlap trials and natural human Hey
Iris recordings are not established by this software package. Synthetic voices
are not independent human speakers. Detailed denominators and failed gates are
preserved in the bound evidence instead of being replaced by a pooled score.

''' + evidence + '\n\n'
        + 'Selected VCM parameter count: `' + str(release.get('parameters', 'not measured')) + '`.\n\n'
        + 'Release binding: `reproduction/reproduction-release.json`. Training command: '
        + '`python train.py`; verify/rebuild: `python train.py --skip-train`; '
        + 'publish verified outputs: `python train.py --skip-train --publish`.\n\n'
        + '[Source recipe on GitHub main](' + SOURCE_REPO + '/tree/main). '
        + 'Pinned initializers, source identity, input manifests and exact access requirements '
        + 'are listed in `reproduction/reproduction-inputs.json`. Missing restricted inputs '
        + 'must be obtained from their lawful source; no replacement corpus is silently substituted.\n', encoding='utf-8')
    authoritative_card = Path(root) / 'MODEL_CARD.md'
    if authoritative_card.is_file() and release.get('study_kind') != 'random':
        header = re.match(r'\A---\r?\n.*?\r?\n---(?:\r?\n|$)', model.read_text(encoding='utf-8'), re.S).group(0)
        body = re.sub(r'\A---\r?\n.*?\r?\n---(?:\r?\n|$)', '', authoritative_card.read_text(encoding='utf-8'), count=1, flags=re.S)
        model.write_text(header + '\n' + published_links(body).lstrip(), encoding='utf-8')
    if release.get('study_kind') == 'random':
        quantitative = []
        for name in release.get('evaluation_receipts', []):
            report = read(relative_path(root, name))
            if 'reproduction-from-scratch-' not in name:
                continue
            quantitative.append('\n## Independent random-initialization study\n\n')
            quantitative.append('This is a separate study; it does not reproduce or replace the historical V2 weights.\n\n')
            quantitative.append('| Track | Correct accepts / positives | UNKNOWN accepts / negatives |\n| --- | --- | --- |\n')
            for track, result in report.get('results', {}).items():
                metric = result.get('overall', {})
                quantitative.append('| ' + track + ' | ' + str(metric.get('correct_accepts')) + '/' + str(metric.get('positive_n'))
                    + ' | ' + str(metric.get('unknown_accepts')) + '/' + str(metric.get('negative_n')) + ' |\n')
        model.write_text(model.read_text(encoding='utf-8') + ''.join(quantitative), encoding='utf-8')
    dataset.write_text(f'''---
language:
- en
task_categories:
- audio-classification
tags:
- finite-voice-commands
- synthetic-speech
license: other
license_name: source-specific-research-terms
license_link: https://huggingface.co/datasets/{DATASET_REPO}/blob/main/iris-release-v1/NOTICES.md
---

# Iris command research data

This repository preserves its existing source datasets and adds versioned Iris
generation/provenance and reproduction receipts. Original, supplied synthetic
and newly generated recordings remain distinguishable. Speaker/voice identity
and frozen split membership are preserved; generation voices are not counted
as independent human speakers.

New audio is published only when its file record explicitly permits
redistribution and binds the relevant source/license receipt. Restricted human
recordings, private household captures, secrets and unlicensed caches are excluded.
Consult `iris-release-v1/NOTICES.md` and each source receipt before reuse.

The experiment evaluates complete typed commands and UNKNOWN false accepts.
Recorded benchmarks, synthetic wake recall, owner proxy trials and physical
household/Pi performance are separate forms of evidence. Synthetic evaluation
does not establish natural human or physical-device acceptance.

Pinned source revisions, hashes and access requirements are recorded in
`iris-release-v1/reproduction-inputs.json`. The paired command study uses only
development/calibration for selection; reserved tests are reported separately.
''', encoding='utf-8')
    previous_card = Path(root) / CACHE / 'inputs/provenance/dataset-card-before-release.md'
    authoritative_dataset = Path(root) / 'docs/DATASETS.md'
    if previous_card.is_file():
        # Keep the original YAML/license and full historical body. Current
        # evidence leads the page so old architecture is not a release guide.
        original = previous_card.read_text(encoding='utf-8')
        parts = re.match(r'\A(---\r?\n.*?\r?\n---(?:\r?\n|$))(.*)\Z', original, re.S)
        header, history = (parts.group(1), parts.group(2)) if parts else ('', original)
        current = ('# Current Iris release\n\n'
                   'Iris uses direct finite command recognition, our Hey Iris wake model, Silero VAD '
                   'and a session-only speaker reference for one command. Recognition runs locally '
                   'without ASR/STT transcription or a language model. Designed voices are synthetic '
                   'and do not establish independent human, household or Raspberry Pi acceptance.\n\n'
                   'Pinned data/source/license receipts, quantitative results and limitations follow. '
                   'The preserved historical card records earlier experiments and source terms.\n\n')
        evidence = (published_links(authoritative_dataset.read_text(encoding='utf-8'))
                    if authoritative_dataset.is_file() else dataset.read_text(encoding='utf-8').split('# Iris command research data', 1)[1])
        dataset.write_text(header + '\n' + current + evidence.rstrip()
                           + '\n\n# Preserved dataset history and source terms\n\n' + history.lstrip(), encoding='utf-8')
    elif authoritative_dataset.is_file():
        dataset.write_text(dataset.read_text(encoding='utf-8') + '\n\n## Current dataset evidence\n\n'
                           + published_links(authoritative_dataset.read_text(encoding='utf-8')), encoding='utf-8')
    notices = directory / 'NOTICES.md'
    notices.write_text('''# Source and asset terms

No blanket license for third-party recordings or assets is granted by these
cards. Existing source notices and access restrictions continue to apply.
The reproduction input receipt identifies upstream sources, immutable revisions
and any inaccessible inputs. Explicit audio redistribution records bind their
source notices. Silero VAD and ncnn notices are included in the runtime archive.
Generated speech is labeled synthetic and retains its generation-model, voice,
prompt, split and content-hash provenance. Private household captures are excluded.
''', encoding='utf-8')
    return model, dataset, notices


def build_publication_plan(*, root=ROOT, workflow, extra_files=None):
    root = Path(root)
    if workflow.get('package_verification', {}).get('passed') is not True:
        raise ValueError('Publication requires verified package inventory and hashes')
    release_path = relative_path(root, workflow['release'])
    if sha(release_path) != workflow['release_sha256']:
        raise ValueError('Release identity changed before publication')
    release = validate_release(root, release_path)
    package = relative_path(root, workflow['package']['path'])
    if sha(package) != workflow['package']['sha256']:
        raise ValueError('Package identity changed before publication')
    verify_package(package)
    token = workflow['run_id']
    directory = root / CACHE / 'publication' / token
    model_card, dataset_card, notices = write_cards(root, release, workflow, directory)
    files = [file_item(root, package, 'releases/iris-pi.zip'),
             file_item(root, model_card, 'README.md'),
             file_item(root, dataset_card, 'README.md', 'dataset'),
             file_item(root, notices, 'NOTICES.md'),
             file_item(root, notices, 'iris-release-v1/NOTICES.md', 'dataset'),
             file_item(root, release_path, 'reproduction/reproduction-release.json')]
    input_path = relative_path(root, release['input_receipt'])
    files.extend([file_item(root, input_path, 'reproduction/reproduction-inputs.json'),
                  file_item(root, input_path, 'iris-release-v1/reproduction-inputs.json', 'dataset')])
    for role, prefix in (('bundle', 'vcm'), ('wake_folder', 'wake'), ('owner_folder', 'owner'), ('vad_folder', 'vad')):
        folder = PurePosixPath(release[role])
        for name in sorted(release['bindings']):
            pure = PurePosixPath(name)
            if pure.is_relative_to(folder):
                files.append(file_item(root, relative_path(root, name), prefix + '/' + pure.relative_to(folder).as_posix()))
    for name in release.get('evaluation_receipts', []):
        files.append(file_item(root, relative_path(root, name), 'evidence/' + Path(name).name))
    # Source bindings remain in the release receipt; GitHub main owns the recipe.
    # Late native/package/fresh checks bind the finished release, so they stay
    # outside its frozen component hashes and enter this verified upload plan.
    files.extend(card_linked_files(root, (model_card, dataset_card)))
    inputs = read(input_path)
    published_inputs = inputs.get('input_publication_receipt')
    verified_inputs = {}
    if published_inputs:
        receipt = relative_path(root, published_inputs['path'])
        if sha(receipt) != published_inputs.get('sha256'):
            raise ValueError('Input publication receipt identity changed')
        proof = read(receipt)
        if proof.get('status') != 'published_and_verified':
            raise ValueError('Input publication receipt does not prove verified remote files')
        for repository in proof['repositories']:
            if repository.get('verified') is not True:
                raise ValueError('Input publication repository was not verified')
            for item in repository['files']:
                key = (repository['repo_type'], repository['repo_id'], repository['revision'], item['path'])
                verified_inputs[key] = item['sha256']
        files.append(file_item(root, receipt, 'reproduction/publication-inputs.json'))
    for item in inputs.get('files', []):
        publication = item.get('publication')
        if publication:
            remote = item.get('remote', {})
            key = (remote.get('repo_type'), remote.get('repo_id'), remote.get('revision'), remote.get('filename'))
            if published_inputs:
                if verified_inputs.get(key) != item['sha256']:
                    raise ValueError('Pinned input lacks verified publication evidence: ' + item['path'])
                continue
            files.append(file_item(root, relative_path(root, item['path']), publication['path_in_repo'],
                                   publication.get('repo_type', 'model'), publication.get('repo_id'),
                                   **({'redistribution': publication['redistribution']} if 'redistribution' in publication else {})))
    for item in extra_files or []:
        files.append(dict(item))
    unique = {}
    for item in files:
        key = (item['repo_type'], item['repo_id'], item['path_in_repo'])
        if key in unique and unique[key] != item:
            raise ValueError('Conflicting publication destination: ' + item['path_in_repo'])
        unique[key] = item
    files = list(unique.values())
    if release.get('study_kind') == 'random':
        prefix = 'studies/iris-random-' + token + '/'
        files = [{**item, 'path_in_repo': prefix + item['path_in_repo']} for item in files]
    plan = {'schema_version': 1, 'experiment': 'iris-release-v1', 'run_id': token,
            'verification': {'passed': True, 'bindings': {workflow['release']: workflow['release_sha256'],
                                                        workflow['package']['path']: workflow['package']['sha256']}},
            'source_repository': SOURCE_REPO, 'source_ref': 'main',
            'source_visibility': 'preserve_existing', 'model_visibility': 'private',
            'dataset_visibility': 'preserve_existing', 'files': files}
    validate_publication(root, plan)
    plan_path = root / HERE / ('publication-plan-' + token + '.json')
    save(plan_path, plan)
    return plan


def publish_plan(*, root=ROOT, plan, api=None, downloader=None, max_files_per_commit=200):
    root = Path(root)
    plan = read(plan) if isinstance(plan, (str, Path)) else plan
    validate_publication(root, plan)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    from huggingface_hub import CommitOperationAdd
    token = plan.get('run_id') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt_path = root / HERE / ('publication-hf-' + token + '.json')
    if receipt_path.exists():
        raise FileExistsError('Preserve existing publication receipt: ' + str(receipt_path))
    record = {'schema_version': 1, 'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_repository': SOURCE_REPO, 'source_ref': 'main',
              'source_visibility': 'preserve_existing', 'model_visibility': 'private',
              'repositories': [], 'receipt': name_under(root, receipt_path)}
    save(receipt_path, record)
    grouped = defaultdict(list)
    for item in plan['files']:
        grouped[(item['repo_type'], item['repo_id'])].append(item)
    try:
        # Authentication is checked without printing or serializing credentials.
        identity = api.whoami()
        record['account'] = identity.get('name')
        for (repo_type, repo_id), items in sorted(grouped.items()):
            if repo_type == 'model':
                api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True)
            info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
            if repo_type == 'model' and info.private is not True:
                api.update_repo_settings(repo_id=repo_id, repo_type='model', private=True)
                info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
                if info.private is not True:
                    raise ValueError('Model repository must be private before publication')
            api.auth_check(repo_id=repo_id, repo_type=repo_type)
            staged = root / CACHE / 'publication' / token / (repo_type + '-immutable-' + uuid.uuid4().hex[:8])
            operations = []
            for item in items:
                source = relative_path(root, item['local_path'])
                if sha(source) != item['sha256']:
                    raise ValueError('Publication identity changed before staging: ' + item['local_path'])
                destination = relative_path(staged, item['path_in_repo'])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if sha(destination) != item['sha256']:
                    raise ValueError('Publication staging identity mismatch: ' + item['local_path'])
                operations.append(CommitOperationAdd(path_in_repo=item['path_in_repo'], path_or_fileobj=destination))
            revision = info.sha
            for start in range(0, len(operations), max_files_per_commit):
                if repo_type == 'model' and api.repo_info(repo_id=repo_id, repo_type=repo_type).private is not True:
                    raise ValueError('Model repository must be private before committing files')
                commit = api.create_commit(repo_id=repo_id, repo_type=repo_type,
                                           operations=operations[start:start + max_files_per_commit],
                                           commit_message='Publish verified Iris release and reproduction evidence',
                                           parent_commit=revision)
                revision = commit.oid
            published = {'repo_type': repo_type, 'repo_id': repo_id, 'revision': revision,
                         'private': bool(info.private), 'files': [], 'verified': False}
            record['repositories'].append(published)
            save(receipt_path, record)
            def verify_remote(item):
                path = Path(downloader(repo_id=repo_id, repo_type=repo_type, revision=revision,
                                       filename=item['path_in_repo']))
                if sha(path) != item['sha256']:
                    raise ValueError('Published remote file identity mismatch: ' + item['path_in_repo'])
                return {'path': item['path_in_repo'], 'sha256': item['sha256'], 'bytes': path.stat().st_size}
            with ThreadPoolExecutor(max_workers=6) as pool:
                published['files'] = list(pool.map(verify_remote, items))
            remote = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
            published['private'] = bool(remote.private)
            if repo_type == 'model' and remote.private is not True:
                raise ValueError('Published model repository is not private')
            published['verified'] = True
            save(receipt_path, record)
        record['status'] = 'published_and_verified'
        save(receipt_path, record)
        return record
    except Exception as error:
        record['status'] = 'failed'
        # Error type is enough for audit; SDK exception bodies can include request details.
        record['error_type'] = type(error).__name__
        save(receipt_path, record)
        raise


def publish_workflow(*, root=ROOT, workflow):
    plan = build_publication_plan(root=root, workflow=workflow)
    return publish_plan(root=root, plan=plan)


def publish_reproduction_inputs(*, root=ROOT, input_receipt=None, verified_receipts=()):
    """Publish only the declared immutable input capsule, then pin its revisions.

    This audit establishes input identity and permitted distribution. It does
    not claim selected-model or physical-device acceptance.
    """
    root = Path(root)
    path = Path(input_receipt or root / HERE / 'reproduction-inputs.json')
    inputs = read(path)
    files, reusable, reused, pointers, inventory = [], {}, {}, [], []
    for pointer in verified_receipts:
        proof_path = relative_path(root, pointer['path'])
        if sha(proof_path) != pointer.get('sha256'):
            raise ValueError('Earlier input publication receipt identity changed')
        proof = read(proof_path)
        if proof.get('status') != 'published_and_verified':
            raise ValueError('Earlier input publication is not complete and verified')
        for repository in proof['repositories']:
            if repository.get('verified') is not True or not re.fullmatch(r'[0-9a-f]{40}', repository.get('revision', '')):
                raise ValueError('Earlier input publication lacks a verified immutable revision')
            for item in repository['files']:
                key = (repository['repo_type'], repository['repo_id'], item['path'], item['sha256'], item['bytes'])
                reusable[key] = (repository, item)
        pointers.append(dict(pointer))
    for item in inputs['files']:
        publication = item.get('publication')
        if not publication:
            continue
        candidate = file_item(root, relative_path(root, item['path']), publication['path_in_repo'],
                              publication.get('repo_type', 'model'), publication.get('repo_id'),
                              **({'redistribution': publication['redistribution']} if 'redistribution' in publication else {}))
        if candidate['sha256'] != item['sha256'] or ('bytes' in item and candidate['bytes'] != item['bytes']):
            raise ValueError('Reproduction input identity changed: ' + item['path'])
        inventory.append(candidate)
        key = (candidate['repo_type'], candidate['repo_id'], candidate['path_in_repo'], candidate['sha256'], candidate['bytes'])
        if key in reusable:
            repository, verified = reusable[key]
            revision_key = (repository['repo_type'], repository['repo_id'], repository['revision'])
            group = reused.setdefault(revision_key, {**repository, 'files': []})
            group['files'].append(dict(verified))
        else:
            files.append(candidate)
    validate_publication(root, {'schema_version': 1, 'verification': {'passed': True}, 'files': inventory})
    token = 'inputs-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    staged = root / CACHE / 'publication' / token
    staged.mkdir(parents=True, exist_ok=True)
    copy = staged / 'reproduction-inputs-before-pinning.json'
    shutil.copyfile(path, copy)
    files.append(file_item(root, copy, 'reproduction/reproduction-inputs-before-pinning.json'))
    plan = {'schema_version': 1, 'run_id': token,
            'verification': {'passed': True, 'scope': 'immutable reproduction input identity and source/provenance audit; no model qualification'},
            'files': files, 'source_repository': SOURCE_REPO, 'source_ref': 'main',
            'source_visibility': 'preserve_existing', 'model_visibility': 'private',
            'dataset_visibility': 'preserve_existing'}
    validate_publication(root, plan)
    plan_path = root / HERE / ('publication-inputs-plan-' + token + '.json')
    save(plan_path, plan)
    record = publish_plan(root=root, plan=plan)
    record['repositories'].extend(reused.values())
    record['reused_publication_receipts'] = pointers
    record['reused_files'] = sum(len(repository['files']) for repository in reused.values())
    save(root / record['receipt'], record)
    revisions = {(r['repo_type'], r['repo_id'], item['path'], item['sha256']): r['revision']
                 for r in record['repositories'] for item in r['files']}
    for item in inputs['files']:
        publication = item.get('publication')
        if publication:
            kind = publication.get('repo_type', 'model')
            repo_id = publication.get('repo_id') or (MODEL_REPO if kind == 'model' else DATASET_REPO)
            item['remote'] = {'repo_id': repo_id, 'repo_type': kind,
                              'revision': revisions[(kind, repo_id, publication['path_in_repo'], item['sha256'])],
                              'filename': publication['path_in_repo']}
    inputs['status'] = 'published_input_capsule_verified'
    inputs['input_publication_receipt'] = {'path': record['receipt'], 'sha256': sha(root / record['receipt'])}
    save(path, inputs)
    return {'receipt': record['receipt'], 'repositories': [{k: r[k] for k in ('repo_id', 'repo_type', 'revision', 'verified')}
             for r in record['repositories']], 'input_receipt_sha256': sha(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--workflow', type=Path)
    source.add_argument('--plan', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    plan = read(args.plan) if args.plan else build_publication_plan(workflow=read(args.workflow))
    if args.prepare_only:
        validate_publication(ROOT, plan)
        print(json.dumps({'status': 'validated', 'files': len(plan['files'])}), flush=True)
    else:
        result = publish_plan(plan=plan)
        print(json.dumps({'status': result['status'], 'receipt': result['receipt']}), flush=True)


if __name__ == '__main__':
    main()
