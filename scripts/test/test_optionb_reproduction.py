"""Portable OptionB reproduction guards; generated audio is software-only evidence."""
import hashlib
import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest


def software_snapshot(tmp_path, *, pinned=False, auxiliary=False):
    """Fresh tones and Parquet only; no old audio, caches or speech gold."""
    import gzip
    import io
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest

    wave = io.BytesIO()
    sf.write(wave, np.sin(np.arange(1600)) * .2, 16000, format='WAV', subtype='PCM_16')
    payload = wave.getvalue()
    pcm, rate = sf.read(io.BytesIO(payload), dtype='float32')
    shard = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'audio': [{'bytes': payload}]}), shard)
    source: dict[str, Any] = dict(repository='software/fixture', revision='1' * 40, file='data/train.parquet',
                  sha256=hashlib.sha256(shard.read_bytes()).hexdigest(), bytes=shard.stat().st_size,
                  url='https://huggingface.co/datasets/software/fixture/resolve/' + '1' * 40
                      + '/data/train.parquet?download=true')
    location: dict[str, Any] = ({k: source[k] for k in ('repository', 'revision', 'file', 'sha256')} if pinned
                else dict(sha256=source['sha256'], url=shard.as_uri()))
    location['row'] = 0
    if auxiliary:
        location = dict(auxiliary_file='nested/tone.wav')
    row: dict[str, Any] = dict(id='software-fixture', split='train', kind='human', expected=None,
               sha256=hashlib.sha256(payload).hexdigest(), pcm_sha256=pcm_digest(pcm, rate),
               location=location)
    archive = tmp_path / 'train.jsonl.gz'
    archive.write_bytes(gzip.compress((json.dumps(row) + '\n').encode(), mtime=0))
    record = dict(path='train.jsonl.gz', sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), rows=1)
    snapshot: dict[str, Any] = dict(files={'train.jsonl.gz': dict(bytes=archive.stat().st_size, sha256=record['sha256'])},
                    source_bindings={}, views={'train': record},
                    auxiliary_source=dict(repository='software/fixture', revision='1' * 40,
                                          prefix='reproduction/auxiliary'))
    if pinned:
        snapshot['source_shards'] = [source]
    path = tmp_path / 'submission/optionb-tiny/snapshot.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot), encoding='utf8')
    return snapshot, row, payload, shard


def test_prepare_resolves_url_less_pinned_locator_without_changing_frozen_rows(tmp_path, monkeypatch):
    from scripts.submission.common import fetch
    from scripts.test.speaker_demo_suite import read_jsonl

    module = importlib.import_module('scripts.submission.optionb_reproduce')
    snapshot, row, payload, shard = software_snapshot(tmp_path, pinned=True)
    frozen = deepcopy(row)
    metadata_bytes = (tmp_path / 'train.jsonl.gz').read_bytes()
    calls = []

    def local_source(url, path, expected):
        assert url == snapshot['source_shards'][0]['url']
        calls.append((url, expected))
        return fetch(shard.as_uri(), path, expected)

    monkeypatch.setattr(module, 'fetch', local_source)
    receipt = module.prepare_source(tmp_path, '.cache/run-01', ('train',))
    expected = [{**frozen, 'audio': f".cache/run-01/audio/{row['sha256']}.wav"}]
    assert read_jsonl(tmp_path / '.cache/run-01/prepared/train.jsonl') == expected
    assert read_jsonl(tmp_path / '.cache/run-01/canonical/train.jsonl') == expected
    assert (tmp_path / expected[0]['audio']).read_bytes() == payload
    assert row == frozen and 'url' not in expected[0]['location']
    assert (tmp_path / 'train.jsonl.gz').read_bytes() == metadata_bytes
    assert calls == [(snapshot['source_shards'][0]['url'], row['location']['sha256'])]
    assert receipt['original_reference_order_and_labels_preserved'] is True


def test_actual_26_url_less_training_locators_resolve_before_any_source_fetch(tmp_path, monkeypatch):
    """Actual frozen metadata, but stop before downloading or decoding corpus PCM."""
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    root = Path(__file__).resolve().parents[2]
    snapshot = module.verify_snapshot(root)
    rows = [row for row in module.read_snapshot_rows(root, snapshot['views']['train'])
            if 'url' not in row['location'] and 'auxiliary_file' not in row['location']]
    assert len(rows) == 26
    frozen = deepcopy(rows)
    resolved = []
    resolve = module._source_location

    def record_source(location, source_shards=None):
        source = resolve(location, source_shards)
        if source_shards is not None:
            assert source == next(s for s in source_shards if s['sha256'] == location['sha256'])
            resolved.append(source)
        return source

    class SourceFetchReached(RuntimeError):
        pass

    def stop_fetch(url, path, expected):
        assert url == next(s['url'] for s in snapshot['source_shards'] if s['sha256'] == expected)
        raise SourceFetchReached('Metadata-only test: never fetch real corpus audio')

    monkeypatch.setattr(module, '_source_location', record_source)
    monkeypatch.setattr(module, 'fetch', stop_fetch)
    with pytest.raises(SourceFetchReached):
        module.hydrate_rows(tmp_path, rows, '.cache/downloads', '.cache/run/audio',
                            snapshot['auxiliary_source'], source_shards=snapshot['source_shards'])
    assert len(resolved) == 26 and rows == frozen
    assert not (tmp_path / '.cache/downloads').exists()
    assert list((tmp_path / '.cache/run/audio').iterdir()) == []


@pytest.mark.parametrize('change', [
    'repository', 'revision', 'file', 'sha256', 'url', 'bytes', 'duplicate_inventory',
    'inventory_url', 'inventory_digest_collision', 'inventory_locator_collision',
    'malformed_repository', 'malformed_revision', 'malformed_file', 'malformed_bytes',
    'partial_locator',
])
def test_pinned_source_identity_rejects_conflicts_or_malformed_inventory_before_fetch(tmp_path, monkeypatch, change):
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    snapshot, row, _, _ = software_snapshot(tmp_path, pinned=True)
    inventory = snapshot['source_shards']
    source = inventory[0]
    location = row['location']
    if change in {'repository', 'revision', 'file', 'sha256', 'url', 'bytes'}:
        location[change] = dict(repository='other/fixture', revision='2' * 40,
                               file='data/other.parquet', sha256='0' * 64,
                               url=source['url'] + '&other=true', bytes=source['bytes'] + 1)[change]
    elif change == 'duplicate_inventory':
        inventory.append(deepcopy(source))
    elif change == 'inventory_url':
        source['url'] = source['url'].replace('/software/fixture/', '/other/fixture/')
    elif change == 'inventory_digest_collision':
        inventory.append({**source, 'file': 'data/other.parquet',
                          'url': source['url'].replace('train.parquet', 'other.parquet')})
    elif change == 'inventory_locator_collision':
        inventory.append({**source, 'sha256': '0' * 64})
    elif change.startswith('malformed_'):
        key = change.removeprefix('malformed_')
        value = dict(repository='software//fixture', revision='A' * 40,
                     file='data//train.parquet', bytes=True)[key]
        location[key] = source[key] = value
    elif change == 'partial_locator':
        del location['repository']
    frozen = deepcopy(row)
    calls = []

    def forbidden_fetch(*args):
        calls.append(args)
        pytest.fail('Invalid source identity must be rejected before any fetch')

    monkeypatch.setattr(module, 'fetch', forbidden_fetch)
    with pytest.raises(ValueError, match='[Ss]ource'):
        module.hydrate_rows(tmp_path, [row], '.cache/downloads', '.cache/run/audio', {},
                            source_shards=inventory)
    assert calls == [] and row == frozen
    assert list((tmp_path / '.cache/run/audio').iterdir()) == []


def directory_link(link, target, kind):
    link.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'symlink':
        link.symlink_to(target, target_is_directory=True)
    else:
        import os
        import subprocess
        if os.name != 'nt':
            pytest.skip('Windows junctions are unavailable on this platform')
        subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(link), str(target)],
                       check=True, capture_output=True, text=True)
    assert link.resolve() == target.resolve()


def outside_files(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('kind', ['symlink', 'junction'])
@pytest.mark.parametrize('outside_checkout', [True, False], ids=['outside-checkout', 'other-cache-run'])
@pytest.mark.parametrize('target', ['run', 'prepared', 'canonical', 'audio', 'downloads',
                                  'auxiliary', 'auxiliary/nested'])
def test_derived_output_directory_links_cannot_escape_the_selected_boundary(
        tmp_path, monkeypatch, kind, outside_checkout, target):
    from scripts.submission.common import fetch

    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    auxiliary = target.startswith('auxiliary')
    _, _, payload, _ = software_snapshot(checkout, auxiliary=auxiliary)
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    run = checkout / '.cache/study/run-01'
    downloads = run.parent / 'downloads'
    outside = tmp_path / 'outside' if outside_checkout else checkout / '.cache/other-run'
    outside.mkdir(parents=True)
    (outside / 'preserve.txt').write_bytes(b'earlier evidence\n')
    before = outside_files(outside)
    if target == 'run':
        link = run
    elif target == 'downloads':
        link = downloads
    elif auxiliary:
        link = downloads / target
    else:
        link = run / target
    directory_link(link, outside, kind)
    if auxiliary:
        source = checkout / 'auxiliary-source.wav'
        source.write_bytes(payload)
        monkeypatch.setattr(module, 'fetch', lambda url, path, expected: fetch(source.as_uri(), path, expected))
    with pytest.raises(ValueError, match='[Pp]ath|[Cc]ache|[Oo]utput'):
        module.prepare_source(checkout, '.cache/study/run-01', ('train',))
    assert outside_files(outside) == before
    assert not (run / 'prepare-train.json').exists()


@pytest.mark.parametrize('outside_checkout', [True, False], ids=['outside-checkout', 'other-cache-run'])
@pytest.mark.parametrize('target', ['prepared', 'canonical', 'receipt', 'audio', 'downloads', 'auxiliary'])
def test_existing_output_filename_links_cannot_escape_the_selected_boundary(
        tmp_path, monkeypatch, outside_checkout, target):
    from scripts.submission.common import fetch

    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    _, row, payload, shard = software_snapshot(checkout, auxiliary=target == 'auxiliary')
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    run = checkout / '.cache/study/run-01'
    downloads = run.parent / 'downloads'
    outside = tmp_path / 'outside' if outside_checkout else checkout / '.cache/other-run'
    outside.mkdir(parents=True)
    contents = shard.read_bytes() if target == 'downloads' else payload
    external_file = outside / 'existing-evidence'
    external_file.write_bytes(contents)
    paths = dict(prepared=run / 'prepared/train.jsonl', canonical=run / 'canonical/train.jsonl',
                 receipt=run / 'prepare-train.json', audio=run / 'audio' / (row['sha256'] + '.wav'),
                 downloads=downloads / (row['location'].get('sha256', '') + '.parquet'),
                 auxiliary=downloads / 'auxiliary/nested/tone.wav')
    link = paths[target]
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(external_file)
    assert link.is_symlink() and link.resolve() == external_file
    if target == 'auxiliary':
        source = checkout / 'auxiliary-source.wav'
        source.write_bytes(payload)
        monkeypatch.setattr(module, 'fetch', lambda url, path, expected: fetch(source.as_uri(), path, expected))
    before = outside_files(outside)
    with pytest.raises(ValueError, match='[Pp]ath|[Cc]ache|[Oo]utput'):
        module.prepare_source(checkout, '.cache/study/run-01', ('train',))
    assert outside_files(outside) == before
    assert link.is_symlink()


@pytest.mark.parametrize('role', ['prepared', 'canonical'])
def test_manifest_path_is_rechecked_after_hydration_before_writing(tmp_path, monkeypatch, role):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    software_snapshot(checkout)
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    outside = tmp_path / 'outside'
    outside.mkdir()
    hydrate = module.hydrate_rows

    def link_after_preflight(*args, **kwargs):
        rows = hydrate(*args, **kwargs)
        directory_link(checkout / '.cache/run-01' / role, outside, 'symlink')
        return rows

    monkeypatch.setattr(module, 'hydrate_rows', link_after_preflight)
    with pytest.raises(ValueError, match='[Pp]ath|[Cc]ache|[Oo]utput'):
        module.prepare_source(checkout, '.cache/run-01', ('train',))
    assert outside_files(outside) == {}
    assert not (checkout / '.cache/run-01/prepare-train.json').exists()


def test_cache_root_is_not_a_run_and_cannot_put_downloads_in_checkout(tmp_path):
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    software_snapshot(tmp_path)
    before = outside_files(tmp_path)
    with pytest.raises(ValueError, match=r'\.cache/'):
        module.prepare_source(tmp_path, '.cache', ('train',))
    assert outside_files(tmp_path) == before
    assert not (tmp_path / 'downloads').exists()
    assert not (tmp_path / '.cache').exists()


@pytest.mark.parametrize(('change', 'message'), [
    ('source_sha', 'Downloaded checksum mismatch'), ('wav_sha', 'Source WAV identity changed'),
    ('pcm_sha', 'Source PCM identity changed'), ('duplicate', 'Duplicate source recording identity'),
    ('missing_row', 'Missing frozen source recordings'),
])
def test_pinned_resolution_does_not_weaken_existing_integrity_or_membership_guards(
        tmp_path, monkeypatch, change, message):
    from scripts.submission.common import fetch

    module = importlib.import_module('scripts.submission.optionb_reproduce')
    snapshot, row, _, shard = software_snapshot(tmp_path, pinned=True)
    rows = [row]
    if change == 'source_sha':
        with shard.open('ab') as stream:
            stream.write(b'deliberately corrupted software source')
    elif change == 'wav_sha':
        row['sha256'] = '0' * 64
    elif change == 'pcm_sha':
        row['pcm_sha256'] = '0' * 64
    elif change == 'duplicate':
        rows.append(deepcopy(row))
    elif change == 'missing_row':
        row['location']['row'] = 1
    frozen = deepcopy(rows)
    monkeypatch.setattr(module, 'fetch', lambda url, path, expected: fetch(shard.as_uri(), path, expected))
    with pytest.raises(ValueError, match=message):
        module.hydrate_rows(tmp_path, rows, '.cache/downloads', '.cache/run/audio', {},
                            source_shards=snapshot['source_shards'])
    assert rows == frozen
    assert list((tmp_path / '.cache/run/audio').iterdir()) == []


def test_multiple_pinned_shards_preserve_interleaved_row_order_and_labels(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.submission.common import fetch

    module = importlib.import_module('scripts.submission.optionb_reproduce')
    snapshot, row, payload, second = software_snapshot(tmp_path, pinned=True)
    first = tmp_path / 'two-rows.parquet'
    pq.write_table(pa.table({'audio': [{'bytes': payload}, {'bytes': payload}]}), first)
    second_source = snapshot['source_shards'][0]
    first_source = {**second_source, 'file': 'data/two-rows.parquet',
                    'url': second_source['url'].replace('train.parquet', 'two-rows.parquet'),
                    'sha256': hashlib.sha256(first.read_bytes()).hexdigest(), 'bytes': first.stat().st_size}
    inventory = [first_source, second_source]
    rows = []
    for ident, source, index, label in [('first-row-1', first_source, 1, 3),
                                       ('second-row-0', second_source, 0, 1),
                                       ('first-row-0', first_source, 0, 2)]:
        rows.append({**row, 'id': ident, 'class_index': label,
                     'location': {**{k: source[k] for k in ('repository', 'revision', 'file', 'sha256')},
                                  'row': index}})
    frozen = deepcopy(rows)
    calls = []

    def local_source(url, path, expected):
        calls.append((url, expected))
        local = first if expected == first_source['sha256'] else second
        return fetch(local.as_uri(), path, expected)

    monkeypatch.setattr(module, 'fetch', local_source)
    hydrated = module.hydrate_rows(tmp_path, rows, '.cache/downloads', '.cache/run/audio', {},
                                   source_shards=inventory)
    assert hydrated == [{**r, 'audio': f".cache/run/audio/{r['sha256']}.wav"} for r in frozen]
    assert rows == frozen
    assert calls == [(s['url'], s['sha256']) for s in inventory]


def test_nested_auxiliary_download_still_verifies_wav_pcm_and_frozen_metadata(tmp_path, monkeypatch):
    from scripts.submission.common import fetch
    from scripts.test.speaker_demo_suite import read_jsonl

    module = importlib.import_module('scripts.submission.optionb_reproduce')
    _, row, payload, _ = software_snapshot(tmp_path, auxiliary=True)
    source = tmp_path / 'auxiliary-source.wav'
    source.write_bytes(payload)
    monkeypatch.setattr(module, 'fetch', lambda url, path, expected: fetch(source.as_uri(), path, expected))
    receipt = module.prepare_source(tmp_path, '.cache/study/run-01', ('train',))
    assert (tmp_path / '.cache/study/downloads/auxiliary/nested/tone.wav').read_bytes() == payload
    assert read_jsonl(tmp_path / '.cache/study/run-01/prepared/train.jsonl') == [
        {**row, 'audio': f".cache/study/run-01/audio/{row['sha256']}.wav"}]
    assert receipt['wav_and_pcm_verified'] is True
    assert receipt['source_download_cache'] == '.cache/study/downloads'


def test_receipt_filename_is_rechecked_immediately_before_save(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    software_snapshot(checkout)
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    outside = tmp_path / 'outside.json'
    outside.write_bytes(b'preserved earlier receipt\n')
    verify = module.verify_snapshot
    calls = []

    def link_after_manifests(*args, **kwargs):
        snapshot = verify(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            (checkout / '.cache/run-01/prepare-train.json').symlink_to(outside)
        return snapshot

    monkeypatch.setattr(module, 'verify_snapshot', link_after_manifests)
    with pytest.raises(ValueError, match='Output path leaves selected cache directory'):
        module.prepare_source(checkout, '.cache/run-01', ('train',))
    assert len(calls) == 2
    assert outside.read_bytes() == b'preserved earlier receipt\n'


def test_snapshot_verification_needs_only_tracked_files(tmp_path):
    payload = tmp_path / 'bundle.json'
    payload.write_bytes(b'{"fixture": true}\n')
    source = tmp_path / 'source.py'
    source.write_bytes(b'pass\n')
    snapshot = {
        'files': {'bundle.json': {'bytes': payload.stat().st_size,
                                  'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}},
        'source_bindings': {'source.py': hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    (tmp_path / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf8')
    assert importlib.util.find_spec('scripts.submission.optionb_reproduce') is not None, (
        'Missing portable final-model reproduction module')
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    assert module.verify_snapshot(tmp_path, 'snapshot.json') == snapshot
    assert not (tmp_path / '.cache').exists()


def test_hydration_reconstructs_parquet_wav_and_pcm_without_old_paths(tmp_path):
    import io
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest

    wave = io.BytesIO()
    sf.write(wave, np.sin(np.arange(1600)) * .2, 16000, format='WAV', subtype='PCM_16')
    payload = wave.getvalue()
    pcm, rate = sf.read(io.BytesIO(payload), dtype='float32')
    shard = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'audio': [{'bytes': payload}]}), shard)
    row = dict(id='software-fixture', sha256=hashlib.sha256(payload).hexdigest(),
               pcm_sha256=pcm_digest(pcm, rate), split='train',
               location=dict(row=0, sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                             url=shard.as_uri()))
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    assert hasattr(module, 'hydrate_rows'), 'Missing source-only WAV/PCM reconstruction'
    hydrated = module.hydrate_rows(tmp_path, [row], '.cache/downloads', '.cache/audio', {})
    assert hydrated == [{**row, 'audio': f".cache/audio/{row['sha256']}.wav"}]
    assert (tmp_path / hydrated[0]['audio']).read_bytes() == payload
    assert not (tmp_path / 'features').exists()


def test_prepare_preserves_frozen_membership_in_new_manifests(tmp_path):
    import gzip
    import io
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest, read_jsonl

    wave = io.BytesIO()
    sf.write(wave, np.sin(np.arange(1600)) * .2, 16000, format='WAV', subtype='PCM_16')
    payload = wave.getvalue()
    pcm, rate = sf.read(io.BytesIO(payload), dtype='float32')
    shard = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'audio': [{'bytes': payload}]}), shard)
    row = dict(id='software-fixture', sha256=hashlib.sha256(payload).hexdigest(),
               pcm_sha256=pcm_digest(pcm, rate), split='train', kind='human',
               location=dict(row=0, sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                             url=shard.as_uri()))
    archive = tmp_path / 'train.jsonl.gz'
    archive.write_bytes(gzip.compress((json.dumps(row) + '\n').encode(), mtime=0))
    record = dict(path='train.jsonl.gz', sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), rows=1)
    snapshot = dict(files={'train.jsonl.gz': dict(bytes=archive.stat().st_size, sha256=record['sha256'])},
                    source_bindings={}, views={'train': record}, auxiliary_source={})
    path = tmp_path / 'submission/optionb-tiny/snapshot.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot), encoding='utf8')
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    assert hasattr(module, 'prepare_source'), 'Missing portable manifest reconstruction'
    receipt = module.prepare_source(tmp_path, '.cache/run-01', ('train',))
    actual = read_jsonl(tmp_path / '.cache/run-01/prepared/train.jsonl')
    assert actual == [{**row, 'audio': f".cache/run-01/audio/{row['sha256']}.wav"}]
    assert read_jsonl(tmp_path / '.cache/run-01/canonical/train.jsonl') == actual
    assert receipt['counts'] == {'train': 1}
    assert receipt['wav_and_pcm_verified'] is True
    assert receipt['old_audio_or_features_copied'] is False


def test_verify_cli_checks_actual_tracked_bytes(tmp_path, capsys):
    payload = tmp_path / 'fixture.json'
    payload.write_bytes(b'{}\n')
    snapshot = dict(files={'fixture.json': dict(bytes=payload.stat().st_size,
                        sha256=hashlib.sha256(payload.read_bytes()).hexdigest())}, source_bindings={})
    path = tmp_path / 'submission/optionb-tiny/snapshot.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot), encoding='utf8')
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    assert hasattr(module, 'main'), 'Missing executable portable reproduction CLI'
    module.main(['verify'], root=tmp_path)
    assert json.loads(capsys.readouterr().out)['tracked_snapshot_files_verified'] == 1


@pytest.mark.parametrize('relative', ('../outside', '/outside', 'C:/outside', 'a\\outside', '.'))
def test_snapshot_paths_cannot_escape_checkout(tmp_path, relative):
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    with pytest.raises(ValueError, match='snapshot path'):
        module.pinned_path(tmp_path, relative)


def test_corrupted_tracked_payload_does_not_verify(tmp_path):
    payload = tmp_path / 'fixture.json'
    payload.write_bytes(b'{}\n')
    snapshot = dict(files={'fixture.json': dict(bytes=payload.stat().st_size,
                        sha256=hashlib.sha256(payload.read_bytes()).hexdigest())}, source_bindings={})
    (tmp_path / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf8')
    payload.write_bytes(b'[]\n')
    module = importlib.import_module('scripts.submission.optionb_reproduce')
    with pytest.raises(ValueError, match='Snapshot identity changed'):
        module.verify_snapshot(tmp_path, 'snapshot.json')
