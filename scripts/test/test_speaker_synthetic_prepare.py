"""Synthetic research preparation preserves human/evaluation evidence."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.test.test_speaker_demo_v2_data import corpus, digest, write_json
from scripts.test.test_speaker_synthetic_data import voices, completed
from scripts.test.speaker_demo_suite import pcm_digest


def adapter():
    name = 'scripts.train.speaker_synthetic_prepare'
    assert importlib.util.find_spec(name) is not None, 'synthetic preparation adapter is missing'
    return importlib.import_module(name)


@pytest.fixture
def synthetic(corpus, tmp_path):
    from scripts.train.speaker_synthetic_data import build_plan
    texts = ['play music', 'weather', 'time', 'lights on', 'lights off', 'pause',
             'stop', 'next song', 'volume up', 'call home', 'message', 'reminders',
             'timer one minute', 'alarm eight a m', 'temperature twenty degrees',
             'brightness twenty percent', 'change color to red', 'remind me to study']
    for name, rows in corpus.views.items():
        if name.startswith(('human/', 'synthetic/')):
            for i, row in enumerate(rows):
                row['publisher_metadata'] = {'transcript': texts[i] if i < len(texts) else 'tell me a joke'}
                # Connected training labels require enough output frames for CTC.
                t = np.arange(32000) / 16000
                frequency = 400 + int(row['id'].split('-')[-1]) * 17
                sf.write(row['audio'], .2 * np.sin(2 * np.pi * frequency * t),
                         16000, subtype='PCM_16')
                values, _ = sf.read(row['audio'], dtype='float32')
                row.update(sha256=digest(row['audio']), pcm_sha256=pcm_digest(values))
    for name in ('human/train', 'human/dev', 'human/calibration'):
        corpus.views[name] = [r for r in corpus.views[name]
                              if not r['command'] or r['command']['intent'] != 'MESSAGE']
    corpus.views['synthetic/diagnostic'] = []
    corpus.rebind()
    generated = tmp_path / 'generated'
    generated.mkdir()
    conditions = voices()
    prompts = [dict(id='m', text='Send a message', spoken_text='Send a message',
                    command={'intent': 'MESSAGE', 'slots': {}}, case_ids=['m']),
               dict(id='n', text='Tell me a joke', spoken_text='Tell me a joke',
                    command=None, case_ids=['n'])]
    plan = build_plan(prompts, conditions)
    rows = completed(plan)
    anchors = []

    def wave(path, frequency):
        t = np.arange(32000) / 16000
        sf.write(path, .2 * np.sin(2 * np.pi * frequency * t), 16000, subtype='PCM_16')
        values, _ = sf.read(path, dtype='float32')
        return dict(audio=str(path), sha256=digest(path), pcm_sha256=pcm_digest(values))

    for i, condition in enumerate(conditions):
        anchors.append(dict(id=condition['id'], split=condition['split'],
                            **wave(generated / f'anchor-{i}.wav', 2300 + i * 31)))
    for i, row in enumerate(rows):
        row.update(wave(generated / f'row-{i}.wav', 1500 + i * 29), role='command',
                   reference_status='supported' if row['command'] else 'unsupported',
                   metric_eligible=True, source='synthetic-fixture',
                   anchor_pcm_sha256=next(a['pcm_sha256'] for a in anchors if a['id'] == row['voice_id']))
    plan_path, voices_path = generated / 'plan.json', generated / 'voices.json'
    write_json(plan_path, plan); write_json(voices_path, conditions)
    anchors_path = generated / 'anchors.json'
    write_json(anchors_path, anchors)
    protocol = generated / 'protocol.json'
    write_json(protocol, dict(experiment_id='synthetic-fixture', parent_protocol=str(corpus.protocol),
                              generation_plan=str(plan_path), voices_file=str(voices_path),
                              catalog_file=str(corpus.catalog), bindings={}))

    def rebind():
        path = generated / 'generated.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        write_json(generated / 'generation-complete.json', dict(complete=True,
            manifest=str(path), manifest_sha256=digest(path), anchors=str(anchors_path),
            anchors_sha256=digest(anchors_path), protocol_sha256=digest(protocol)))

    rebind()
    corpus.generated, corpus.generated_rows = generated, rows
    corpus.synthetic_protocol, corpus.rebind_generation = protocol, rebind
    return corpus


def test_synthetic_data_allows_research_and_keeps_missing_human_classes_visible(synthetic):
    m = adapter()
    r = m.prepare_run(synthetic.synthetic_protocol, synthetic.installed,
                      synthetic.generated, synthetic.output)
    assert r['research_ready_for_fitting'] is True
    assert r['human_qualified'] is False
    assert r['human_missing_intents']['human/train'] == ['MESSAGE']
    for view in ('human/train', 'human/dev', 'human/calibration',
                 'evaluation/test', 'evaluation/holdout'):
        assert Path(r['manifests'][view]['path']).read_bytes() == Path(
            synthetic.audit['views'][view]['path']).read_bytes()
    assert r['coverage']['synthetic/dev']['rows'] == 2
    assert r['coverage']['synthetic/calibration']['rows'] == 2


def test_normalization_uses_only_human_training_waveforms(synthetic):
    m = adapter()
    r = m.prepare_run(synthetic.synthetic_protocol, synthetic.installed,
                      synthetic.generated, synthetic.output)
    from vcm.frontend import features
    arrays = [features(sf.read(row['audio'], dtype='float32')[0]).astype('float64')
              for row in synthetic.views['human/train']]
    expected = np.mean([a.mean(axis=1) for a in arrays], axis=0)
    norm = json.loads(Path(r['normalization']['path']).read_text())
    np.testing.assert_allclose(norm['mean'], expected, atol=1e-10)
    assert norm['human_training_utterances'] == 18


def test_preparation_does_not_open_published_or_human_calibration_audio(synthetic):
    m = adapter()
    for view in ('human/calibration', 'evaluation/test', 'evaluation/holdout'):
        for row in synthetic.views[view]:
            Path(row['audio']).unlink()
    r = m.prepare_run(synthetic.synthetic_protocol, synthetic.installed,
                      synthetic.generated, synthetic.output)
    assert r['published_evaluation_audio_read'] is False
    assert r['calibration_features_extracted'] is False


def test_failed_generation_is_rejected_even_if_manifest_receipt_is_rebound(synthetic):
    m = adapter()
    synthetic.generated_rows[0]['qa_eligible'] = False
    synthetic.rebind_generation()
    with pytest.raises(ValueError):
        m.prepare_run(synthetic.synthetic_protocol, synthetic.installed,
                      synthetic.generated, synthetic.output)


def test_corrupt_generated_audio_is_not_accepted_from_metadata_alone(synthetic):
    m = adapter()
    Path(synthetic.generated_rows[-1]['audio']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        m.prepare_run(synthetic.synthetic_protocol, synthetic.installed,
                      synthetic.generated, synthetic.output)
    assert not (synthetic.output / 'preparation.json').exists()


def test_previous_preparation_is_not_overwritten(synthetic):
    m = adapter()
    synthetic.output.mkdir()
    marker = synthetic.output / 'keep.txt'
    marker.write_text('previous evidence')
    with pytest.raises(FileExistsError):
        m.prepare_run(synthetic.synthetic_protocol, synthetic.installed,
                      synthetic.generated, synthetic.output)
    assert marker.read_text() == 'previous evidence'


def test_insufficient_ctc_frames_do_not_become_training_coverage(synthetic):
    m = adapter()
    row = next(r for r in synthetic.generated_rows if r['split'] == 'train' and r['command'])
    values = .1 * np.sin(2 * np.pi * 750 * np.arange(1600) / 16000)
    sf.write(row['audio'], values, 16000, subtype='PCM_16')
    actual, _ = sf.read(row['audio'], dtype='float32')
    row.update(sha256=digest(row['audio']), pcm_sha256=pcm_digest(actual))
    synthetic.rebind_generation()
    with pytest.raises(ValueError, match='Post-label-filter'):
        m.prepare_run(synthetic.synthetic_protocol, synthetic.installed,
                      synthetic.generated, synthetic.output)
    assert not (synthetic.output / 'preparation.json').exists()
