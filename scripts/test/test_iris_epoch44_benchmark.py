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
    from scripts.runtime import iris_pi_benchmark
    seen = []
    monkeypatch.setattr(iris_pi_benchmark, 'main', lambda argv: seen.append(argv) or 17)
    assert launcher.main(['--model-choice','legacy','--threads','2']) == 17
    assert seen == [['--threads','2']]
