import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study import file_hash, write_json, identity
from study import HERE,read_rows
from synthbench import load_runtime
from synthbench import validate_parity_receipt


def test_runtime_rejects_current_source_drift_before_loading_graph(tmp_path):
    write_json(tmp_path/'source-identity.json',{'vcm/frontend.py':'0'*64})
    files={'source-identity.json':file_hash(tmp_path/'source-identity.json')}
    write_json(tmp_path/'synthetic-bundle.json',{'files':files,'binding':identity(files)})
    with pytest.raises(ValueError,match='source identity'):
        load_runtime(tmp_path)


def test_unchanged_hash_binding_rejects_modified_payload(tmp_path):
    (tmp_path/'model.onnx').write_bytes(b'initial')
    write_json(tmp_path/'synthetic-bundle.json',{'files':{'model.onnx':file_hash(tmp_path/'model.onnx')}})
    (tmp_path/'model.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='file changed'):
        load_runtime(tmp_path)


def test_partial_or_wrong_bundle_parity_cannot_unlock_final_test(tmp_path):
    write_json(tmp_path/'results.json',dict(binding='other',complete=True,recordings=1,max_error=0))
    with pytest.raises(ValueError,match='Complete waveform parity'):
        validate_parity_receipt(tmp_path/'results.json','selected')


def test_summary_cannot_claim_complete_parity_over_an_empty_rows_file(tmp_path):
    (tmp_path/'rows.jsonl').write_text('')
    n=sum(r['split']=='dev' for r in read_rows(HERE/'generation-plan.jsonl'))
    write_json(tmp_path/'results.json',dict(binding='selected',complete=True,recordings=n,
        max_error=0.,rows_sha256=file_hash(tmp_path/'rows.jsonl')))
    with pytest.raises(ValueError,match='parity'):
        validate_parity_receipt(tmp_path/'results.json','selected')
