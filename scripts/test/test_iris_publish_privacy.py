"""Publication privacy and source boundaries, without contacting a remote service."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.iris_publish import publish_plan, published_links


class Hub:
    """The remote boundary preserves existing privacy and immutable file snapshots."""

    def __init__(self, root, *, model_private=True, dataset_private=False,
                 apply_privacy=True, public_after_auth=False, public_after_download=False):
        self.root = root
        self.privacy = {'model': model_private, 'dataset': dataset_private}
        self.revisions = {'model': 'a' * 40, 'dataset': 'a' * 40}
        self.apply_privacy = apply_privacy
        self.public_after_auth = public_after_auth
        self.public_after_download = public_after_download
        self.calls, self.commits, self.snapshots = [], [], {}
        self.historical_source = b'preserved historical source evidence'
        self.files = {'model': {'reproduction/source/historical.py': self.historical_source}, 'dataset': {}}

    def whoami(self):
        self.calls.append(('whoami', {}))
        return {'name': 'fixture-user'}

    def create_repo(self, **kwargs):
        self.calls.append(('create_repo', kwargs))
        kind = kwargs['repo_type']
        if self.privacy[kind] is None:
            self.privacy[kind] = kwargs['private']

    def update_repo_settings(self, **kwargs):
        self.calls.append(('update_repo_settings', kwargs))
        if self.apply_privacy:
            self.privacy[kwargs['repo_type']] = kwargs['private']

    def repo_info(self, **kwargs):
        self.calls.append(('repo_info', kwargs))
        return SimpleNamespace(sha=self.revisions[kwargs['repo_type']], private=self.privacy[kwargs['repo_type']])

    def auth_check(self, **kwargs):
        self.calls.append(('auth_check', kwargs))
        if self.public_after_auth and kwargs['repo_type'] == 'model':
            self.privacy['model'] = False

    def create_commit(self, **kwargs):
        self.calls.append(('create_commit', kwargs))
        kind = kwargs['repo_type']
        assert kwargs['parent_commit'] == self.revisions[kind]
        self.commits.append((kind, self.privacy[kind]))
        for operation in kwargs['operations']:
            self.files[kind][operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        revision = format(len(self.commits), '040x')
        self.revisions[kind] = revision
        self.snapshots[(kind, revision)] = dict(self.files[kind])
        return SimpleNamespace(oid=revision)

    def download(self, **kwargs):
        self.calls.append(('download', kwargs))
        kind, revision, name = kwargs['repo_type'], kwargs['revision'], kwargs['filename']
        path = self.root / 'readback' / kind / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.snapshots[(kind, revision)][name])
        if self.public_after_download and kind == 'model':
            self.privacy['model'] = False
        return path


def plan_for(root, *, kinds=('model',), destination='README.md'):
    path = root / 'reviewed.md'
    path.write_text('hash-bound publication fixture\n')
    return {'schema_version': 1, 'run_id': 'privacy', 'verification': {'passed': True}, 'files': [
        {'local_path': path.name, 'path_in_repo': destination,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
         'repo_type': kind, 'repo_id': 'example/iris'} for kind in kinds]}


@pytest.mark.parametrize('initial_private', [False, True, None], ids=['existing-public', 'existing-private', 'new-model'])
def test_model_publication_is_private_by_default_and_preserves_historical_source(tmp_path, initial_private):
    plan = plan_for(tmp_path)
    hub = Hub(tmp_path, model_private=initial_private)

    result = publish_plan(root=tmp_path, plan=plan, api=hub, downloader=hub.download)

    assert result['status'] == 'published_and_verified'
    assert result['repositories'][0]['private'] is True
    assert hub.commits == [('model', True)]
    assert hub.files['model']['reproduction/source/historical.py'] == b'preserved historical source evidence'
    creates = [arguments for method, arguments in hub.calls if method == 'create_repo']
    assert creates == [{'repo_id': 'example/iris', 'repo_type': 'model', 'private': True, 'exist_ok': True}]
    updates = [arguments for method, arguments in hub.calls if method == 'update_repo_settings']
    assert updates == ([{'repo_id': 'example/iris', 'repo_type': 'model', 'private': True}]
                       if initial_private is False else [])
    assert hub.calls[-1] == ('repo_info', {'repo_id': 'example/iris', 'repo_type': 'model', 'revision': '0' * 39 + '1'})


@pytest.mark.parametrize('visibility', ['public', 'preserve_existing', 'PRIVATE', None, False])
def test_explicit_nonprivate_model_plan_is_rejected_before_any_api_call(tmp_path, visibility):
    plan = plan_for(tmp_path)
    plan['model_visibility'] = visibility
    hub = Hub(tmp_path)

    with pytest.raises(ValueError, match='private'):
        publish_plan(root=tmp_path, plan=plan, api=hub, downloader=hub.download)

    assert hub.calls == []
    assert not (tmp_path / 'experiments/iris-release-v1/publication-hf-privacy.json').exists()


@pytest.mark.parametrize('dataset_private', [False, True], ids=['public-dataset', 'private-dataset'])
def test_model_privacy_conversion_does_not_change_dataset_visibility(tmp_path, dataset_private):
    hub = Hub(tmp_path, model_private=False, dataset_private=dataset_private)
    result = publish_plan(root=tmp_path, plan=plan_for(tmp_path, kinds=('dataset', 'model')),
                          api=hub, downloader=hub.download)

    assert [(r['repo_type'], r['private'], r['verified']) for r in result['repositories']] == [
        ('dataset', dataset_private, True), ('model', True, True)]
    mutations = [arguments for method, arguments in hub.calls
                 if method in ('create_repo', 'update_repo_settings')]
    assert all(arguments['repo_type'] == 'model' and arguments['private'] is True for arguments in mutations)


def test_failed_privacy_conversion_blocks_commit_and_records_failure(tmp_path):
    hub = Hub(tmp_path, model_private=False, apply_privacy=False)

    with pytest.raises(ValueError, match='private'):
        publish_plan(root=tmp_path, plan=plan_for(tmp_path), api=hub, downloader=hub.download)

    assert hub.commits == []
    receipt = json.loads((tmp_path / 'experiments/iris-release-v1/publication-hf-privacy.json').read_text())
    assert receipt['status'] == 'failed'
    assert not any(repository['verified'] for repository in receipt['repositories'])


def test_privacy_is_rechecked_immediately_before_commit(tmp_path):
    hub = Hub(tmp_path, public_after_auth=True)

    with pytest.raises(ValueError, match='private'):
        publish_plan(root=tmp_path, plan=plan_for(tmp_path), api=hub, downloader=hub.download)

    assert hub.commits == []


def test_public_model_after_readback_never_receives_verified_receipt(tmp_path):
    hub = Hub(tmp_path, public_after_download=True)

    with pytest.raises(ValueError, match='private'):
        publish_plan(root=tmp_path, plan=plan_for(tmp_path), api=hub, downloader=hub.download)

    assert hub.commits == [('model', True)]
    receipt = json.loads((tmp_path / 'experiments/iris-release-v1/publication-hf-privacy.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['repositories'][0]['verified'] is False
    assert receipt['repositories'][0]['private'] is False


@pytest.mark.parametrize('destination', ['reproduction/source/scripts/train.py',
                                       'studies/iris-random-fixture/reproduction/source/train.py'])
@pytest.mark.parametrize('kind', ['model', 'dataset'])
def test_explicit_source_mirrors_are_rejected_before_any_api_call(tmp_path, destination, kind):
    hub = Hub(tmp_path)

    with pytest.raises(ValueError, match='GitHub|source'):
        publish_plan(root=tmp_path, plan=plan_for(tmp_path, kinds=(kind,), destination=destination),
                     api=hub, downloader=hub.download)

    assert hub.calls == []


@pytest.mark.parametrize('destination', ['reproduction/inputs/provenance/diversity-prefit-trainer.py',
                                       'reproduction/inputs/official-benchmark/scorer.py'])
def test_pinned_source_input_evidence_can_be_published_privately(tmp_path, destination):
    hub = Hub(tmp_path)
    result = publish_plan(root=tmp_path, plan=plan_for(tmp_path, destination=destination),
                          api=hub, downloader=hub.download)

    assert result['repositories'][0]['files'][0]['path'] == destination
    assert hub.files['model'][destination] == b'hash-bound publication fixture\n'
    assert result['repositories'][0]['private'] is True


@pytest.mark.parametrize(('original', 'expected'), [
    ('docs/HANDOFF-iris-rpi-v1.md#setup',
     'https://github.com/justinruaya123/smart-speaker/blob/main/docs/HANDOFF-iris-rpi-v1.md#setup'),
    ('../scripts/iris_release.py',
     'https://github.com/justinruaya123/smart-speaker/blob/main/scripts/iris_release.py'),
    ('reproduce.py',
     'https://github.com/justinruaya123/smart-speaker/blob/main/reproduce.py'),
    ('REQUIREMENTS.md#deliverables',
     'https://github.com/justinruaya123/smart-speaker/blob/main/REQUIREMENTS.md#deliverables'),
    ('../SpeakerUI/server/README.md',
     'https://github.com/justinruaya123/smart-speaker/blob/main/SpeakerUI/server/README.md'),
    ('data/sources/ai231-primary.source.json',
     'https://github.com/justinruaya123/smart-speaker/blob/main/data/sources/ai231-primary.source.json'),
    ('https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/reproduction/source/train.py',
     'https://github.com/justinruaya123/smart-speaker/blob/main/train.py'),
    ('https://huggingface.co/justinruaya123/ai231-smart-speaker/tree/main/reproduction/source',
     'https://github.com/justinruaya123/smart-speaker/tree/main'),
])
def test_source_and_document_links_resolve_to_github_main(original, expected):
    assert published_links('[Recipe](' + original + ')') == '[Recipe](' + expected + ')'
