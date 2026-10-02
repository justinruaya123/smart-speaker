import importlib.util
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision
from version_context import bind


def test_version_context_restores_shared_modules_and_preserves_source_bytes():
    import study,synthbench
    before={p:revision.file_hash(p) for p in revision.V1.glob('*.py')}
    cache,here,contract=study.CACHE,study.HERE,synthbench.Contract
    with bind():
        assert study.HERE==revision.HERE and study.CACHE==revision.CACHE
        assert synthbench.Contract is revision.RevisionContract
    assert (study.CACHE,study.HERE,synthbench.Contract)==(cache,here,contract)
    assert all(revision.file_hash(p)==digest for p,digest in before.items())


def test_v2_executes_shared_export_to_report_with_artificial_waveforms(tmp_path,monkeypatch):
    path=revision.V1/'tests/test_stage_integration.py'
    spec=importlib.util.spec_from_file_location('shared_stage_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    with bind(),monkeypatch.context() as scoped:
        fixture.test_export_replay_parity_calibration_and_report_preserve_failed_acceptance(tmp_path,scoped)
