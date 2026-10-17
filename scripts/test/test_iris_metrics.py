"""Regression coverage for interrupted or stale held-out replay caches."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.test import iris_metrics as metrics


@pytest.fixture
def replay(tmp_path,monkeypatch):
    folder=tmp_path/'bundle';folder.mkdir()
    for name in ('bundle.json','model.onnx','policy.json','catalog.json','normalization.json'):
        (folder/name).write_text('{}')
    runtime=SimpleNamespace(root=folder,engine='fixture',meta={'files':{
        name:metrics.digest(folder/name) for name in ('model.onnx','policy.json','catalog.json','normalization.json')}})
    rows=[dict(id=str(i),class_index=1,speaker='speaker',split='test',synthetic=False,source='fixture') for i in range(2)]
    calls=[]
    def infer(runtime,rows,stream):
        calls.append(True)
        result=[{**{k:r.get(k) for k in ('id','speaker','split','synthetic','source','accent_group')},
                 'expected_index':r['class_index'],'accepted_index':1,'raw_index':1} for r in rows]
        for r in result:stream.write(json.dumps(r)+'\n')
        return result
    monkeypatch.setattr(metrics,'infer_rows',infer)
    return runtime,rows,tmp_path/'predictions.jsonl',calls


def test_complete_bound_replay_reuses_without_inference(replay):
    runtime,rows,path,calls=replay
    first=metrics.replay_rows(runtime,rows,path)
    assert metrics.replay_rows(runtime,rows,path)==first
    assert len(calls)==1 and len(first)==2


def test_unbound_truncated_cache_fails_closed(replay):
    runtime,rows,path,calls=replay
    path.write_text(json.dumps({'id':'0','expected_index':1})+'\n')
    with pytest.raises(ValueError,match='Incomplete/unbound'):metrics.replay_rows(runtime,rows,path)
    assert not calls and path.exists()


@pytest.mark.parametrize('change',['predictions','policy','manifest','ids'])
def test_changed_replay_dependencies_reject_reuse(replay,change):
    runtime,rows,path,calls=replay
    metrics.replay_rows(runtime,rows,path)
    if change=='predictions':path.write_text(path.read_text().splitlines()[0]+'\n')
    elif change=='policy':(runtime.root/'policy.json').write_text('{"threshold":0.5}')
    elif change=='manifest':rows[0]={**rows[0],'class_index':2}
    else:rows.reverse()
    with pytest.raises(ValueError,match='changed'):metrics.replay_rows(runtime,rows,path)
    assert len(calls)==1


def test_interrupted_new_replay_never_publishes_completed_filename(replay,monkeypatch):
    runtime,rows,path,calls=replay
    def interrupted(runtime,rows,stream):
        stream.write(json.dumps({'id':'0'})+'\n');raise RuntimeError('interrupted')
    monkeypatch.setattr(metrics,'infer_rows',interrupted)
    with pytest.raises(RuntimeError,match='interrupted'):metrics.replay_rows(runtime,rows,path)
    assert not path.exists() and not path.with_suffix('.jsonl.complete.json').exists()
    assert list(path.parent.glob('predictions.jsonl.*.tmp'))


def test_parity_refuses_to_claim_shadow_source_as_executing_code(tmp_path):
    from scripts.test.iris_decision_parity import verify
    with pytest.raises(ValueError,match='executing source checkout'):verify(tmp_path)


def test_official_cache_requires_complete_bound_population(tmp_path):
    from scripts.test.iris_benchmark import _cached,_complete
    binding={'model':'selected','policy':'locked'}
    score={k:{'command_level':{'n':196}} for k in ('accepted','raw_diagnostic')}
    _complete(tmp_path,binding,score)
    assert _cached(tmp_path,binding)==score
    with pytest.raises(ValueError,match='changed'):_cached(tmp_path,{**binding,'model':'other'})
    (tmp_path/'results.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):_cached(tmp_path,binding)


def test_unbound_official_cache_and_changed_population_fail_closed(tmp_path):
    from scripts.test.iris_benchmark import _cached,_complete
    with pytest.raises(ValueError,match='population'):
        _complete(tmp_path,{}, {k:{'command_level':{'n':195}} for k in ('accepted','raw_diagnostic')})
    (tmp_path/'results.json').write_text('{}')
    with pytest.raises(ValueError,match='Incomplete/unbound'):_cached(tmp_path,{})


def test_official_binding_includes_label_and_slot_inventory(replay,tmp_path):
    from scripts.test.iris_benchmark import _binding
    runtime,rows,path,calls=replay
    scorer=tmp_path/'scorer';(scorer/'vcmbench').mkdir(parents=True)
    inventory=scorer/'vcmbench/variations.csv';inventory.write_text('intent,slot\nLIGHT_ON,ceiling\n')
    data=tmp_path/'frozen.parquet';data.write_bytes(b'fixed fixture')
    before=_binding(runtime,scorer,data)
    inventory.write_text('intent,slot\nLIGHT_OFF,ceiling\n')
    assert _binding(runtime,scorer,data)!=before
