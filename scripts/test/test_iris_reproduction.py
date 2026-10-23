"""Safe public reproduction selection and lossless evaluation references."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_default_reproduction_only_verifies_frozen_artifacts(capsys, monkeypatch):
    import reproduce
    from scripts import optionb_epoch44
    seen = []
    from scripts import iris_release
    def forbidden(*a, **k):
        pytest.fail('Default verification must never bootstrap or fit')
    monkeypatch.setattr(iris_release, 'ensure_source_environment', forbidden)
    monkeypatch.setattr(optionb_epoch44, 'main', lambda argv, **kw: seen.append(argv) or 0)
    assert reproduce.main([]) == 0
    assert seen == [['verify']]


def test_fit_requires_explicit_intent_and_cannot_fall_into_legacy(monkeypatch):
    import reproduce
    def forbidden(*a, **k):
        pytest.fail('No fitting/bootstrap is authorized by an implicit device choice')
    from scripts import iris_release
    monkeypatch.setattr(iris_release, 'ensure_source_environment', forbidden)
    monkeypatch.setattr(subprocess, 'call', forbidden)
    with pytest.raises(SystemExit):
        reproduce.main(['--device','cuda'])
    with pytest.raises(SystemExit):
        reproduce.main(['fit','--source-checkout','source','--source-python','python'])


def test_prepare_uses_unchanged_pinned_preparer_test_holdout_only_and_propagates_failure(tmp_path, monkeypatch):
    from scripts import iris_reproduction as module
    seen = []
    monkeypatch.setattr(module, 'verify_source_checkout', lambda source: None)
    monkeypatch.setattr(module.subprocess, 'call', lambda command, **kwargs: seen.append((command, kwargs)) or 23)
    source = tmp_path/'source'; source.mkdir()
    python = tmp_path/'python.exe'; python.touch()
    code = module.main(['prepare-evaluation','--source-checkout',str(source),
                        '--source-python',str(python),'--output','.cache/evaluation/run-01'])
    assert code == 23
    command, kwargs = seen[0]
    assert command == [str(python.resolve()), '-B', '-m', 'scripts.submission.optionb_reproduce',
                       'prepare','--splits','test','holdout','--output','.cache/evaluation/run-01']
    assert kwargs['cwd'] == source.resolve()
    assert not (source/'.cache/evaluation/run-01/portable').exists()


def test_explicit_fit_preserves_original_comparison_exit_code(tmp_path, monkeypatch):
    from scripts import iris_reproduction as module
    monkeypatch.setattr(module, 'verify_source_checkout', lambda source: None)
    seen = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command, **kwargs: seen.append(command) or 1)
    python = tmp_path/'python.exe'; python.touch()
    assert module.main(['fit','--execute-fit','--source-checkout',str(tmp_path),
        '--source-python',str(python),'--output','.cache/fit/run-01','--device','cuda']) == 1
    assert seen[0][3:] == ['scripts.submission.optionb_fit','--output','.cache/fit/run-01','--device','cuda']


def test_portable_ambiguous_reference_is_not_relabelled_unknown():
    from scripts.iris_reproduction import portable_row
    row = dict(id='ambiguous', audio='.cache/run/audio.wav', sha256='a'*64, pcm_sha256='b'*64,
               expected=None, command=None, metric_eligible=False, kind='human', split='holdout',
               reference_reason='unresolved_reference', source='original')
    result = portable_row(row)
    assert result['expected'] is None and result['metric_eligible'] is False
    assert result['reference_status'] == 'ambiguous'
    assert result['pcm_sha256'] == 'b'*64
    assert result['provenance']['kind'] == 'human'
    assert result['provenance']['reference_reason'] == 'unresolved_reference'


def test_real_compressed_membership_counts_without_audio_or_fitting():
    from scripts.iris_reproduction import snapshot_rows
    rows = snapshot_rows(ROOT)
    assert len(rows['test']) == 4418 and len(rows['holdout']) == 196
    assert sum(row['kind']=='human' for split in rows.values() for row in split) == 1146
    assert sum(row['metric_eligible'] is False for split in rows.values() for row in split if row['kind']=='human') == 187


def test_legacy_workflow_is_explicit_and_retains_arguments(monkeypatch):
    import reproduce
    seen = []
    monkeypatch.setattr(reproduce, 'legacy_main', lambda argv: seen.append(argv) or 19)
    assert reproduce.main(['--model-choice','legacy','--device','cuda']) == 19
    assert seen == [['--device','cuda']]
