"""Data gates for a fair human-first comparison, using original WAV fixtures."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


COMMANDS = [
    {'intent': 'PLAY_MUSIC', 'slots': {}},
    {'intent': 'WEATHER', 'slots': {}},
    {'intent': 'TIME', 'slots': {}},
    {'intent': 'LIGHT_ON', 'slots': {'light_id': 'light.all'}},
    {'intent': 'LIGHT_OFF', 'slots': {'light_id': 'light.all'}},
    {'intent': 'PAUSE', 'slots': {}},
    {'intent': 'STOP', 'slots': {}},
    {'intent': 'NEXT', 'slots': {}},
    {'intent': 'VOLUME', 'slots': {'operation': 'up'}},
    {'intent': 'CALL', 'slots': {'contact_id': 'contact.home'}},
    {'intent': 'MESSAGE', 'slots': {}},
    {'intent': 'LIST_REMINDERS', 'slots': {}},
    {'intent': 'TIMER', 'slots': {'duration_seconds': 60}},
    {'intent': 'ALARM', 'slots': {'time': '08:00', 'timezone': 'Asia/Manila'}},
    {'intent': 'TEMPERATURE', 'slots': {'degrees': 20, 'unit': 'C'}},
    {'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': 20}},
    {'intent': 'COLOR', 'slots': {'light_id': 'light.all', 'color': 'red'}},
    {'intent': 'CREATE_REMINDER', 'slots': {'task_id': 'task.study'}},
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data) + '\n')


def adapter():
    name = 'scripts.train.speaker_demo_v2_data'
    assert importlib.util.find_spec(name) is not None, 'v2 preparation adapter is missing'
    return importlib.import_module(name)


@pytest.fixture
def corpus(tmp_path):
    from scripts.test.speaker_demo_suite import pcm_digest
    installed = tmp_path / 'installed'
    installed.mkdir()
    views = {}
    counter = 0

    def row(split, command, kind='human'):
        nonlocal counter
        counter += 1
        audio = tmp_path / f'{counter}.wav'
        t = np.arange(1600 + counter) / 16000
        sf.write(audio, .2 * np.sin(2 * np.pi * (200 + 17 * counter) * t), 16000,
                 subtype='PCM_16')
        pcm, rate = sf.read(audio, dtype='float32')
        return dict(id=f'row-{counter}', audio=str(audio), sha256=digest(audio),
                    pcm_sha256=pcm_digest(pcm, rate), speaker_id=f'person-{counter}',
                    pair_id=f'pair-{counter}', kind=kind, split=split, role='command',
                    command=command, qa_eligible=True, audio_flags=[],
                    metric_eligible=True, supplemental=False,
                    reference_status='supported' if command else 'unsupported')

    for split in ('train', 'dev', 'calibration'):
        views['human/' + split] = [row(split, c) for c in [*COMMANDS, None]]
    views['synthetic/train'] = [row('train', COMMANDS[6], 'synthetic')]
    views['evaluation/test'] = [row('test', COMMANDS[6]),
                                row('test', COMMANDS[6], 'synthetic')]
    views['evaluation/holdout'] = [row('holdout', None)]
    views['evaluation/holdout'][0].update(reference_status='ambiguous',
                                         metric_eligible=False)
    source = tmp_path / 'source.json'
    write_json(source, {'revision': 'pinned-fixture'})
    catalog = tmp_path / 'catalog.json'
    write_json(catalog, {'entries': [{'command': c} for c in COMMANDS]})
    definitions = tmp_path / 'definitions.json'
    write_json(definitions, [dict(id='stop', text='Stop', tier='baseline',
                                  command=COMMANDS[6]),
                             dict(id='color', text='Red light', tier='extension',
                                  command=COMMANDS[16])])
    protocol = tmp_path / 'protocol.json'
    config = dict(experiment_id='fixture-v2', required_intents=[c['intent'] for c in COMMANDS],
                  catalog=dict(path=str(catalog), sha256=digest(catalog)),
                  definitions=dict(path=str(definitions), sha256=digest(definitions)),
                  bindings={})
    audit = dict(views={}, inventory=str(installed / 'inventory.jsonl'),
                 source_receipts={str(source): digest(source)})

    def rebind():
        for name, rows in views.items():
            path = installed / (name.replace('/', '-') + '.jsonl')
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            audit['views'][name] = dict(path=str(path), rows=len(rows), sha256=digest(path))
        inventory = installed / 'inventory.jsonl'
        inventory.write_text(''.join(json.dumps(r) + '\n'
                                     for rows in views.values() for r in rows))
        audit['inventory_sha256'] = digest(inventory)
        write_json(installed / 'audit.json', audit)
        config['installation_audit_sha256'] = digest(installed / 'audit.json')
        write_json(protocol, config)

    rebind()
    return SimpleNamespace(installed=installed, views=views, protocol=protocol,
                           output=tmp_path / 'run', config=config, audit=audit,
                           rebind=rebind, source=source, catalog=catalog)


@pytest.mark.parametrize('view,intent', [('human/train', 'MESSAGE'),
                                        ('human/dev', 'CALL'),
                                        ('human/calibration', 'TEMPERATURE')])
def test_missing_human_intent_blocks_features_and_records_collection_gap(corpus, view, intent):
    m = adapter()
    corpus.views[view] = [r for r in corpus.views[view]
                          if r['command'] is None or r['command']['intent'] != intent]
    corpus.rebind()
    with pytest.raises(m.PreparationBlocked):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)
    report = json.loads((corpus.output / 'readiness.json').read_text())
    assert report['missing_human_intents'][view] == [intent]
    assert report['ready_for_fitting'] is False
    assert not (corpus.output / 'features').exists()
    assert not (corpus.output / 'normalization.json').exists()


@pytest.mark.parametrize('view', ['human/train', 'human/dev', 'human/calibration'])
def test_synthetic_cannot_masquerade_as_human_coverage(corpus, view):
    m = adapter()
    corpus.views[view][0]['kind'] = 'synthetic'
    corpus.rebind()
    with pytest.raises(ValueError, match='human view'):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)


@pytest.mark.parametrize('key', ['sha256', 'pcm_sha256', 'speaker_id', 'pair_id'])
def test_cross_split_identity_leakage_is_rejected(corpus, key):
    m = adapter()
    corpus.views['human/train'][0][key] = corpus.views['evaluation/test'][0][key]
    corpus.rebind()
    with pytest.raises(ValueError, match='overlap|duplicate'):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)


@pytest.mark.parametrize('change', [dict(qa_eligible=False), dict(audio_flags=['clipped']),
                                   dict(reference_status='ambiguous'),
                                   dict(metric_eligible=False), dict(split='test')])
def test_ineligible_fitting_input_is_not_silently_relabelled(corpus, change):
    m = adapter()
    corpus.views['synthetic/train'][0].update(change)
    corpus.rebind()
    with pytest.raises(ValueError):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)


def test_missing_required_slot_is_rejected_without_fabricated_default(corpus):
    m = adapter()
    corpus.views['human/train'][9]['command'] = {'intent': 'CALL', 'slots': {}}
    corpus.rebind()
    with pytest.raises(ValueError, match='slots'):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)


def test_changed_manifest_is_rejected_even_with_same_row_count(corpus):
    m = adapter()
    path = Path(corpus.audit['views']['human/train']['path'])
    path.write_text(path.read_text().replace('person-1', 'other-person'))
    with pytest.raises(ValueError, match='identity'):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)


def test_changed_source_receipt_is_rejected(corpus):
    m = adapter()
    corpus.source.write_text('{}')
    with pytest.raises(ValueError, match='identity'):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)


def test_changed_inventory_is_rejected(corpus):
    m = adapter()
    Path(corpus.audit['inventory']).write_text('{}\n')
    with pytest.raises(ValueError, match='identity'):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)


def test_preparation_never_opens_calibration_or_evaluation_audio(corpus):
    m = adapter()
    for view in ('human/calibration', 'evaluation/test', 'evaluation/holdout'):
        for row in corpus.views[view]:
            Path(row['audio']).unlink()
    receipt = m.prepare_run(corpus.protocol, corpus.installed, corpus.output)
    assert receipt['calibration_test_holdout_audio_read'] is False
    assert receipt['ready_for_fitting'] is True


def test_missing_unknown_examples_block_the_human_comparison(corpus):
    m = adapter()
    corpus.views['human/dev'] = [r for r in corpus.views['human/dev'] if r['command']]
    corpus.rebind()
    with pytest.raises(m.PreparationBlocked):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)
    report = json.loads((corpus.output / 'readiness.json').read_text())
    assert report['missing_human_unknown'] == ['human/dev']


def test_preparation_preserves_published_denominators_and_uses_only_human_train_normalization(corpus):
    m = adapter()
    receipt = m.prepare_run(corpus.protocol, corpus.installed, corpus.output)
    from vcm.frontend import features
    arrays = [features(sf.read(r['audio'], dtype='float32')[0]).astype('float64')
              for r in corpus.views['human/train']]
    expected_mean = np.mean([a.mean(axis=1) for a in arrays], axis=0)
    norm = json.loads(Path(receipt['normalization']['path']).read_text())
    np.testing.assert_allclose(norm['mean'], expected_mean, atol=1e-10)
    assert norm['source_ids'] == [r['id'] for r in corpus.views['human/train']]
    for view in ('evaluation/test', 'evaluation/holdout'):
        assert Path(receipt['manifests'][view]['path']).read_bytes() == Path(
            corpus.audit['views'][view]['path']).read_bytes()
    feature_rows = [json.loads(line) for line in Path(
        receipt['feature_receipts']['path']).read_text().splitlines()]
    assert {r['split'] for r in feature_rows} == {'train', 'dev'}
    assert len(feature_rows) == 39
    assert all(digest(r['path']) == r['sha256'] for r in feature_rows)
    assert receipt['ready_for_fitting'] is True
    assert receipt['model_qualified'] is False


def test_catalog_gap_does_not_turn_a_supported_command_into_unknown(corpus):
    m = adapter()
    corpus.views['human/train'][6]['command'] = {
        'intent': 'PLAY_MUSIC', 'slots': {'song_id': 'song.new'}}
    corpus.rebind()
    with pytest.raises(m.PreparationBlocked):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)
    report = json.loads((corpus.output / 'readiness.json').read_text())
    assert report['outside_catalog']['human/train'] == [
        {'intent': 'PLAY_MUSIC', 'slots': {'song_id': 'song.new'}}]


def test_existing_output_is_never_overwritten(corpus):
    m = adapter()
    corpus.output.mkdir()
    marker = corpus.output / 'previous.json'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)
    assert marker.read_text() == 'preserve'


def test_changed_original_audio_cannot_produce_features(corpus):
    m = adapter()
    Path(corpus.views['human/train'][0]['audio']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        m.prepare_run(corpus.protocol, corpus.installed, corpus.output)
    assert not (corpus.output / 'preparation.json').exists()
