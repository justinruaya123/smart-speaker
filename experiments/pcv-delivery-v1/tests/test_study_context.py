import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
import revision  # Initialize shared study paths even when this file runs alone.


@pytest.mark.parametrize('version',['pcv-synthetic-v2','pcv-synthetic-v2-repair1'])
def test_host_tools_bind_the_candidate_study_and_restore_it(version):
    from scripts.pcv_study_context import bind_study
    import study,synthbench
    before=(study.CACHE,synthbench.HERE,synthbench.load_split)
    with bind_study(ROOT/'experiments'/version) as selected:
        assert selected.CACHE==ROOT/'.cache'/version
        assert synthbench.HERE==ROOT/'experiments'/version
    assert (study.CACHE,synthbench.HERE,synthbench.load_split)==before


def test_unknown_study_version_cannot_select_an_arbitrary_import(tmp_path):
    from scripts.pcv_study_context import bind_study
    (tmp_path/'protocol.json').write_text(json.dumps(dict(version='../../arbitrary')))
    with pytest.raises(ValueError,match='Unsupported'):
        with bind_study(tmp_path):raise AssertionError('must not enter')


def test_v3_delivery_context_selects_its_adapter_and_restores_globals(tmp_path,monkeypatch):
    import importlib.util
    from scripts.pcv_study_context import bind_study
    import study,synthbench
    spec=importlib.util.spec_from_file_location('_pcv_synthetic_v3',ROOT/'experiments/pcv-synthetic-v3/workflow.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    validated=[];monkeypatch.setattr(module,'validate',lambda:validated.append(True))
    monkeypatch.setitem(sys.modules,'_pcv_synthetic_v3',module)
    (tmp_path/'protocol.json').write_text(json.dumps({'version':'pcv-synthetic-v3'}))
    before=(study.CACHE,synthbench.HERE,synthbench.load_split)
    with bind_study(tmp_path) as selected:
        assert validated and selected.CACHE==ROOT/'.cache/pcv-synthetic-v3'
        assert synthbench.load_split is module.load_split
    assert (study.CACHE,synthbench.HERE,synthbench.load_split)==before
