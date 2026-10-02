"""Benchmark contracts: no diagnostic promotion, disappearing trials or fake cores."""
import importlib
import json

import pytest


def module(name):
    try:
        return importlib.import_module('scripts.runtime.' + name)
    except ModuleNotFoundError:
        pytest.fail('Benchmark component is not implemented: ' + name)


def test_adapter_never_promotes_disabled_policy_candidate():
    adapter = module('pcv_benchmark_adapter')
    event = {'output': {'accepted': False, 'command': None},
             'diagnostic_candidate': {'candidate': {'intent': 'TIME', 'slots': {}}},
             'timing': {'total_ms': 12.5}, 'audio_seconds': 2.0}
    assert adapter.log_record(event) == {'intent': 'OUT_OF_SCOPE', 'slot': '',
                                       'infer_ms': 12.5, 'audio_ms': 2000.0}


@pytest.mark.parametrize('command,expected', [
    ({'intent': 'VOLUME', 'slots': {'operation': 'up'}}, ('VOLUME_UP', '')),
    ({'intent': 'VOLUME', 'slots': {'operation': 'down'}}, ('VOLUME_DOWN', '')),
    ({'intent': 'TIMER', 'slots': {'duration_seconds': 60}}, ('TIMER', '60 seconds')),
    ({'intent': 'ALARM', 'slots': {'time': '21:00', 'timezone': 'Asia/Manila'}}, ('ALARM', '21:00')),
    ({'intent': 'CREATE_REMINDER', 'slots': {'task_id': 'task.drink_water'}}, ('CREATE_REMINDER', 'Drink water')),
])
def test_adapter_preserves_direction_and_typed_slots(command, expected):
    adapter = module('pcv_benchmark_adapter')
    assert adapter.class_prediction(command) == expected


def test_adapter_keeps_unrepresentable_command_wrong_not_correct_rejection():
    adapter = module('pcv_benchmark_adapter')
    assert adapter.class_prediction({'intent': 'VOLUME', 'slots': {'operation': 'mute'}})[0] == 'UNMAPPED_VOLUME'


def test_full_metrics_preserve_ambiguous_and_separate_acceptance():
    suite = module('pcv_full_benchmark')
    good = {'intent': 'TIME', 'slots': {}}
    rows = [
        {'id': 'p', 'reference_status': 'supported', 'expected': good, 'accepted': None, 'candidate': good},
        {'id': 'n', 'reference_status': 'unsupported', 'expected': None, 'accepted': None, 'candidate': good},
        {'id': 'a', 'reference_status': 'ambiguous', 'expected': None, 'accepted': None, 'candidate': good},
    ]
    result = suite.typed_metrics(rows, 'candidate')
    assert result['source_n'] == 3 and result['ambiguous_n'] == 1
    assert result['positive_n'] == result['negative_n'] == result['exact'] == 1
    assert result['precision'] == .5 and result['false_accept_rate'] == 1
    assert suite.typed_metrics(rows, 'accepted')['recall'] == 0


def test_manifest_refuses_missing_audio_and_escape(tmp_path):
    suite = module('pcv_full_benchmark')
    manifest = tmp_path/'manifest.json'
    for audio in ('missing.wav', '../outside.wav'):
        manifest.write_text(json.dumps([{'id': 'x', 'audio': audio, 'sha256': 'a'*64}]))
        with pytest.raises((ValueError, FileNotFoundError)):
            suite.load_trials(manifest)


def test_single_core_affinity_is_observed():
    suite = module('pcv_full_benchmark')
    before = suite.get_affinity()
    try:
        receipt = suite.pin_cpu(min(before))
        assert receipt['observed'] == [min(before)] and receipt['verified'] is True
    finally:
        suite.set_affinity(before)


def test_output_directory_is_immutable(tmp_path):
    suite = module('pcv_full_benchmark')
    folder = tmp_path/'run'
    folder.mkdir()
    with pytest.raises(FileExistsError):
        suite.create_output(folder)


def test_live_log_writes_only_measured_authoritative_decisions(tmp_path):
    adapter = module('pcv_benchmark_adapter')
    path = tmp_path/'benchmark.jsonl'
    log = adapter.BenchmarkLog(path)
    log({'event':'state','state':'processing'})
    log({'event':'wake','status':'detected'})
    log({'event':'command','output':{'accepted':False,'command':None},
         'audio_seconds':1,'timing':{'total_ms':7},
         'diagnostic_candidate':{'candidate':{'intent':'TIME','slots':{}}}})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{'event':'wake'}, {'intent':'OUT_OF_SCOPE','slot':'','infer_ms':7,'audio_ms':1000}]


def test_acquisition_keeps_config_name_for_multiple_test_shards():
    from scripts.test.pcv_benchmark_source import evaluation_payloads
    files = [{'rfilename':n} for n in ('data/test-00000.parquet', 'synthetic_negatives/test-00000.parquet',
                                      'supplemental_synth/train-00000.parquet','data/holdout-00000.parquet')]
    assert len(evaluation_payloads({'siblings':files})) == 3


def test_application_packages_full_benchmark_entrypoint():
    from scripts.speaker_publish import application_files
    entries = application_files('standalone', 'only')
    assert 'benchmark.py' in entries
    assert 'scripts/runtime/pcv_full_benchmark.py' in entries
    assert 'scripts/runtime/pcv_benchmark_adapter.py' in entries


def test_launcher_selects_full_corpus_when_packaged(tmp_path, monkeypatch):
    import argparse
    from scripts.runtime import speaker_start
    (tmp_path/'benchmark').mkdir()
    (tmp_path/'benchmark/manifest.json').write_text('[]')
    monkeypatch.setattr(speaker_start, 'installed_bundle', lambda root: root/'models/default')
    seen = []
    def run(command, **kwargs):
        seen.append(command)
        output = command[command.index('--output')+1]
        output.mkdir(parents=True)
        (output/'metrics.json').write_text(json.dumps({'tracks':{}, 'policy_enabled':False}))
    monkeypatch.setattr(speaker_start, 'run', run)
    args = argparse.Namespace(output=tmp_path/'results', threads=1, cpu=0, settings={'inference':{'engine':'ncnn'}})
    result = speaker_start.benchmark(tmp_path, 'python', args)
    assert result['policy_enabled'] is False
    assert seen[0][0] == 'python'
    assert seen[0][seen[0].index('--manifest')+1] == tmp_path/'benchmark/manifest.json'


def test_failed_benchmark_restores_caller_affinity(tmp_path):
    import argparse
    suite = module('pcv_full_benchmark')
    before = suite.get_affinity()
    args = argparse.Namespace(cpu=min(before),output=tmp_path/'out',manifest=tmp_path/'absent.json')
    with pytest.raises(FileNotFoundError):suite.run(args)
    assert suite.get_affinity() == before


def test_supplemental_train_filename_uses_reserved_voice_split():
    try:
        from scripts.test.pcv_benchmark_supplemental import evaluation_split
    except ImportError:
        pytest.fail('Supplemental split audit is not implemented')
    assert evaluation_split({'voice_split':'test'}) == 'test'
    assert evaluation_split({'voice_split':'holdout'}) == 'holdout'
    assert evaluation_split({'voice_split':'train'}) is None
    with pytest.raises(ValueError):evaluation_split({'voice_split':''})
