"""Published-artifact replay of the real selected graph, not a source-fit lock."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'submission/models/optionb-tiny'


def fixture_manifest(tmp_path):
    sf.write(tmp_path/'tone.wav', np.sin(np.arange(16000)*.13)*.05, 16000, subtype='PCM_16')
    sf.write(tmp_path/'silence.wav', np.zeros(16000), 16000, subtype='PCM_16')
    rows = [dict(id='tone', audio='tone.wav', expected={'intent':'TIME','slots':{}}, provenance='software_tone'),
            dict(id='silence', audio='silence.wav', expected=None, provenance='software_silence'),
            dict(id='ambiguous', audio='tone.wav', expected=None, metric_eligible=False,
                 reference_status='ambiguous', provenance='software_ambiguous')]
    for row in rows:
        row['sha256'] = hashlib.sha256((tmp_path/row['audio']).read_bytes()).hexdigest()
    path = tmp_path/'manifest.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')
    return path


def test_published_artifact_uses_real_recipe_identity_and_original_denominators(tmp_path):
    from scripts.iris_epoch44_benchmark import benchmark
    path = fixture_manifest(tmp_path)
    result = benchmark(BUNDLE, path, tmp_path/'results', repeats=3, root=tmp_path)
    assert result['parameters'] == 258912 and result['selected_epoch'] == 44
    assert result['bundle_identity_file'] == 'recipe.json'
    assert result['bundle_sha256'] == hashlib.sha256((BUNDLE/'recipe.json').read_bytes()).hexdigest()
    assert result['model_sha256'] == 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce'
    assert result['artifact_provenance'] == 'frozen_published_epoch44'
    assert result['runtime_source_sha256'] == hashlib.sha256((ROOT/'scripts/runtime/iris_epoch44.py').read_bytes()).hexdigest()
    assert result['source_n'] == 3 and result['observation_n'] == 9
    assert result['excluded_reference_n'] == 1 and result['accepted']['scored_n'] == 2
    assert result['accepted']['negative_n'] == 1 and result['accepted']['positive_n'] == 1
    assert result['capture_fault_source_n'] == 1
    assert result['raw']['unavailable_n'] == 1
    assert result['physical_pi_measured'] is False
    assert result['actions_enabled'] is False and result['qualified'] is False
    assert result['latency_ms']['end_to_end']['n'] == 9
    assert result['policy_sha256'] == 'b9242e46fb093642a41023e8ace3c0cadc2fe51ffede512aa09e399622f355f0'
    saved = json.loads((tmp_path/'results/metrics.json').read_text())
    assert saved == result
    trials = [json.loads(line) for line in (tmp_path/'results/trials.jsonl').read_text().splitlines()]
    assert len(trials) == 9 and all(row['actions_enabled'] is False for row in trials)
    with pytest.raises(FileExistsError):
        benchmark(BUNDLE, path, tmp_path/'results', repeats=3, root=tmp_path)


def test_public_benchmark_defaults_to_epoch44_and_requires_explicit_audio(tmp_path):
    result = subprocess.run([sys.executable, '-B', str(ROOT/'benchmark.py'), '--manifest', str(tmp_path/'absent')],
                            capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert result.returncode == 2
    assert 'manifest' in result.stderr.lower() and 'Traceback' not in result.stderr
    help_result = subprocess.run([sys.executable, '-B', str(ROOT/'benchmark.py'), '--help'],
                                capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert '--model-choice' in help_result.stdout and 'epoch44' in help_result.stdout
    assert '--data-root' in help_result.stdout


def test_explicit_legacy_benchmark_remains_accessible(monkeypatch):
    import benchmark as launcher
    import importlib
    from scripts.runtime import iris_pi_benchmark
    seen = []
    monkeypatch.setattr(iris_pi_benchmark, 'main', lambda argv: seen.append(argv) or 17)
    # Historical packages do not contain the source-only epoch44 facade.
    monkeypatch.setitem(sys.modules,'scripts.iris_epoch44_benchmark',None)
    importlib.reload(launcher)
    assert launcher.main(['--model-choice','legacy','--threads','2']) == 17
    assert seen == [['--threads','2']]


def test_scorer_explicitly_separates_wrong_positive_accepts_and_negative_accepts():
    from scripts.runtime.iris_pi_benchmark import accuracy
    command={'intent':'TIME','slots':{}}
    wrong={'intent':'WEATHER','slots':{}}
    rows=[dict(scored=True,raw_available=True,expected=expected,accepted=actual)
          for expected,actual in [(command,command),(command,wrong),(command,None),(None,None),(None,wrong)]]
    rows.append(dict(scored=False,raw_available=True,expected=None,accepted=wrong))
    result=accuracy(rows,'accepted')
    assert result['wrong_positive_n']==1 and result['rejected_positive_n']==1
    assert result['false_accept_n']==1 and result['exact_positive_n']==1
    assert result['exact_n']==2 and result['scored_n']==5
    assert result['complete_command_accuracy']==.4


def test_benchmark_first_use_bootstraps_before_scientific_imports(tmp_path,monkeypatch):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.runtime import epoch44_setup, speaker_tools
    calls=[]
    def sync(root,*,extras=()):
        calls.append(('setup',root,extras))
        return tmp_path/'managed-python'
    def execute(argv,*,cwd,env):
        calls.append(('child',argv,cwd,env))
        return 17
    monkeypatch.setattr(epoch44_setup,'sync_epoch44',sync)
    monkeypatch.setattr(speaker_tools,'sync_python',lambda *a,**k:pytest.fail('Legacy full setup must not be used'))
    monkeypatch.setattr(subprocess,'call',execute)
    monkeypatch.setattr(entry,'benchmark',lambda *a,**k:pytest.fail('Scientific runtime imported before first-use setup'))
    monkeypatch.setenv('PYTHONPATH','unrelated-source')
    manifest=fixture_manifest(tmp_path)
    result=entry.main(['--manifest',str(manifest),'--output',str(tmp_path/'results')],root=ROOT)
    assert result==17
    assert calls[0]==('setup',ROOT,())
    _,argv,cwd,env=calls[1]
    assert argv[:3]==[str(tmp_path/'managed-python'),'-B',str(ROOT/'benchmark.py')]
    assert '--no-setup' in argv and '--manifest' in argv
    assert argv[argv.index('--manifest')+1]==str(manifest)
    assert cwd==ROOT and 'PYTHONPATH' not in env
    assert env['ACTIONS_ENABLED']=='false' and env['IRIS_FEEDBACK_ENABLED']=='false'


def test_benchmark_no_setup_uses_current_interpreter_without_installing(tmp_path,monkeypatch):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.runtime import epoch44_setup, speaker_tools
    monkeypatch.setattr(epoch44_setup,'sync_epoch44',lambda *a,**k:pytest.fail('Explicit no-setup must not install'))
    monkeypatch.setattr(speaker_tools,'sync_python',lambda *a,**k:pytest.fail('Explicit no-setup must not install'))
    expected=dict(artifact_provenance='frozen_published_epoch44',model_sha256='bound',parameters=258912,
        selected_epoch=44,hardware={'measurement_host':'fixture'},physical_pi_measured=False,
        source_n=1,observation_n=1,excluded_reference_n=0,raw={},accepted={},latency_ms={'end_to_end':{}})
    monkeypatch.setattr(entry,'benchmark',lambda *a,**k:expected)
    manifest=fixture_manifest(tmp_path)
    assert entry.main(['--manifest',str(manifest),'--no-setup'],root=ROOT)==expected


def test_public_benchmark_propagates_managed_child_failure(monkeypatch):
    import runpy
    from scripts import iris_epoch44_benchmark as entry
    monkeypatch.setattr(entry,'main',lambda argv:17)
    monkeypatch.setattr(sys,'argv',['benchmark.py','--manifest','originals.jsonl'])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT/'benchmark.py'),run_name='__main__')
    assert stopped.value.code==17


def test_default_bootstrap_requests_only_benchmark_extra_before_assets(tmp_path, monkeypatch):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.runtime import epoch44_setup, iris_benchmark_assets, speaker_tools
    calls = []
    monkeypatch.setattr(epoch44_setup, 'sync_epoch44', lambda root, *, extras=(): calls.append(('setup', root, extras)) or tmp_path/'python')
    monkeypatch.setattr(speaker_tools, 'sync_python', lambda *a, **k: pytest.fail('Full legacy setup is forbidden'))
    monkeypatch.setattr(iris_benchmark_assets, 'default_manifest', lambda *a, **k: pytest.fail('Assets require the managed interpreter'))
    monkeypatch.setattr(entry, 'benchmark', lambda *a, **k: pytest.fail('Replay must follow managed setup'))
    monkeypatch.setattr(subprocess, 'call', lambda argv, **kw: calls.append(('child', argv, kw)) or 17)
    assert entry.main(['--output', str(tmp_path/'results')], root=ROOT) == 17
    assert calls[0] == ('setup', ROOT, ('benchmark',))
    child = calls[1][1]
    assert child[:3] == [str(tmp_path/'python'), '-B', str(ROOT/'benchmark.py')]
    assert '--manifest' not in child and '--_managed' in child
    assert '--no-setup' not in child  # An explicitly offline request must never be repurposed for downloads.


@pytest.mark.parametrize('case', ['existing_output', 'output_under_bundle', 'output_parent_file', 'bad_json', 'missing_reference', 'empty_manifest', 'bad_typed_reference', 'repeats'])
def test_invalid_inputs_fail_before_setup_downloads_or_model_imports(tmp_path, monkeypatch, capsys, case):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.runtime import epoch44_setup, iris_benchmark_assets
    monkeypatch.setattr(epoch44_setup, 'sync_epoch44', lambda *a, **k: pytest.fail('Invalid inputs must precede setup'))
    monkeypatch.setattr(iris_benchmark_assets, 'default_manifest', lambda *a, **k: pytest.fail('Invalid inputs must precede downloads'))
    monkeypatch.setattr(entry, 'benchmark', lambda *a, **k: pytest.fail('Invalid inputs must precede replay'))
    argv = ['--output', str(tmp_path/'result')]
    if case == 'existing_output':
        (tmp_path/'result').mkdir()
    elif case == 'output_under_bundle':
        argv += ['--bundle', str(tmp_path)]
    elif case == 'output_parent_file':
        (tmp_path/'result').write_text('not a directory')
        argv[1] = str(tmp_path/'result'/'nested')
    elif case == 'repeats':
        argv += ['--repeats', '0']
    else:
        payload = {'bad_json': 'not-json\n', 'missing_reference': '{}\n', 'empty_manifest': '\n',
                   'bad_typed_reference': json.dumps(dict(id='x', sha256='0'*64, expected={'intent': 'NOT_AN_INTENT', 'slots': {}}, audio='missing.wav'))}[case]
        path = tmp_path/'bad.jsonl'
        path.write_text(payload)
        argv += ['--manifest', str(path)]
    with pytest.raises(SystemExit) as stopped:
        entry.main(argv, root=ROOT)
    assert stopped.value.code == 2
    assert 'error:' in capsys.readouterr().err


def test_managed_default_downloads_only_holdout_then_no_setup_replays_cached_input(tmp_path, monkeypatch):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.runtime import epoch44_setup
    from scripts.test.test_iris_benchmark_assets import tiny_holdout
    monkeypatch.setattr(epoch44_setup, 'sync_epoch44', lambda *a, **k: pytest.fail('Managed/offline replay must not bootstrap again'))
    with tiny_holdout(tmp_path, monkeypatch) as (assets, gold, _wavs, calls, _original):
        first = entry.main(['--_managed', '--bundle', str(BUNDLE), '--repeats', '2', '--output', str(tmp_path/'first')], root=tmp_path)
        assert first['source_n'] == 3 and first['observation_n'] == 6
        assert first['excluded_reference_n'] == 1 and first['accepted']['scored_n'] == 2
        assert first['accepted']['positive_n'] == 1 and first['accepted']['negative_n'] == 1
        assert first['selected_epoch'] == 44 and first['artifact_provenance'] == 'frozen_published_epoch44'
        assert len(calls) == 1
        manifest = Path(first['manifest'])
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert [row['expected'] for row in rows] == [row['expected'] for row in gold]
        monkeypatch.setattr(assets, 'urlopen', lambda *a, **k: pytest.fail('--no-setup must not download'))
        second = entry.main(['--no-setup', '--bundle', str(BUNDLE), '--repeats', '1', '--output', str(tmp_path/'second')], root=tmp_path)
        assert second['manifest_sha256'] == first['manifest_sha256']
        assert second['source_n'] == 3 and second['observation_n'] == 3
        assert second['accepted'] == first['accepted']


def test_no_setup_missing_default_is_clear_offline_error(tmp_path, monkeypatch, capsys):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.runtime import epoch44_setup
    from scripts.test.test_iris_benchmark_assets import tiny_holdout
    monkeypatch.setattr(epoch44_setup, 'sync_epoch44', lambda *a, **k: pytest.fail('No setup is explicit'))
    monkeypatch.setattr(entry, 'benchmark', lambda *a, **k: pytest.fail('Missing input must precede inference'))
    with tiny_holdout(tmp_path, monkeypatch) as (assets, _gold, _wavs, calls, _original):
        monkeypatch.setattr(assets, 'urlopen', lambda *a, **k: pytest.fail('No setup must never access network'))
        with pytest.raises(SystemExit) as stopped:
            entry.main(['--no-setup', '--output', str(tmp_path/'missing')], root=tmp_path)
        assert stopped.value.code == 2
        assert 'without --no-setup' in capsys.readouterr().err
        assert calls == []


def test_default_replay_saves_and_prints_exposed_holdout_scope(tmp_path, monkeypatch, capsys):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.test.test_iris_benchmark_assets import tiny_holdout
    with tiny_holdout(tmp_path, monkeypatch) as (assets, _gold, _wavs, _calls, _original):
        output = tmp_path/'scoped-result'
        result = entry.main(['--_managed', '--bundle', str(BUNDLE), '--repeats', '1', '--output', str(output)], root=tmp_path)
        scope = result['benchmark_scope']
        assert scope['population'] == 'existing_exposed_fixed_holdout'
        assert scope['fresh_blind'] is False and scope['full_corpus'] is False
        assert scope['measurement_scope'] == 'actual_host_only'
        assert 'exposed' in scope['description'] and 'not fresh blind' in scope['description']
        assert scope['source']['sha256'] == assets.PARQUET_SHA256
        assert scope['source']['bytes'] == assets.PARQUET_BYTES
        assert scope['gold_reference_sha256'] == assets.GOLD_SHA256
        saved = json.loads((output/'metrics.json').read_text())
        sidecar = json.loads((output/'benchmark-inputs.json').read_text())
        printed = json.loads(capsys.readouterr().out)
        assert saved == result
        assert sidecar['benchmark_scope'] == printed['benchmark_scope'] == scope
        assert sidecar['manifest_sha256'] == result['manifest_sha256']
        assert result['physical_pi_measured'] is False and result['actions_enabled'] is False


def test_explicit_manifest_scope_is_not_labelled_as_official_holdout(tmp_path, monkeypatch):
    from scripts import iris_epoch44_benchmark as entry
    from scripts.runtime import iris_benchmark_assets
    monkeypatch.setattr(iris_benchmark_assets, 'default_manifest', lambda *a, **k: pytest.fail('Explicit manifests must not prepare defaults'))
    monkeypatch.setattr(iris_benchmark_assets, 'urlopen', lambda *a, **k: pytest.fail('Explicit manifests must not download'))
    manifest = fixture_manifest(tmp_path)
    result = entry.main(['--manifest', str(manifest), '--data-root', str(tmp_path), '--no-setup',
                         '--bundle', str(BUNDLE), '--repeats', '1', '--output', str(tmp_path/'custom')], root=ROOT)
    assert result['benchmark_scope']['population'] == 'explicit_manifest'
    assert 'source' not in result['benchmark_scope']


def test_cold_launcher_and_assets_use_only_stdlib_before_minimal_setup(tmp_path):
    # A separate interpreter avoids this test module's NumPy/soundfile imports hiding eager imports.
    code = f'''
import importlib.abc
from pathlib import Path
import sys
from types import ModuleType
class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'numpy', 'onnxruntime', 'pyarrow', 'soundfile', 'torch'}}:
            raise AssertionError('Scientific import before setup: ' + fullname)
sys.meta_path.insert(0, NoScience())
from scripts import iris_epoch44_benchmark as entry
from scripts.runtime import iris_benchmark_assets
setup = ModuleType('scripts.runtime.epoch44_setup')
calls = []
setup.sync_epoch44 = lambda root, extras=(): calls.append(extras) or Path('managed-python')
sys.modules[setup.__name__] = setup
entry.subprocess.call = lambda *a, **k: 17
entry.benchmark = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Replay before setup'))
folder = Path({str(tmp_path)!r})
manifest = folder/'input.jsonl'
manifest.write_text('{{"id":"fixture","audio":"original.wav","sha256":"'+'0'*64+'","expected":null}}\\n')
assert entry.main(['--manifest', str(manifest), '--output', str(folder/'custom')]) == 17
assert entry.main(['--output', str(folder/'default')]) == 17
assert calls == [(), ('benchmark',)]
print('stdlib-only bootstrap verified')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'stdlib-only bootstrap verified'
