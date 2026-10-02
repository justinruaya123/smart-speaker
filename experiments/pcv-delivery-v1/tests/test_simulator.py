import json
import sys
from pathlib import Path
import pytest
from test_bundle import fixture_bundle,qualify_fixture
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from visualizer.export_assets import export


def test_portable_simulator_export_binds_model_and_stays_diagnostic(fixture_bundle,tmp_path):
    result=export(fixture_bundle,tmp_path/'fixture.pt',tmp_path/'inspector',pcv_runtime=True)
    assert result['acceptance_enabled'] is False
    assert result['source_profile']=='DIAGNOSTIC_UNQUALIFIED'
    assert sum(layer['parameters'] for layer in result['layers'])==113064
    assert len(result['vocabulary'])==40
    assert json.loads((tmp_path/'inspector/examples.json').read_text())==[]


def test_simulator_rejects_other_models_measured_results(fixture_bundle,tmp_path):
    evidence=tmp_path/'evidence.json';evidence.write_text(json.dumps({'binding':'wrong'}))
    with pytest.raises(ValueError,match='another model'):
        export(fixture_bundle,tmp_path/'fixture.pt',tmp_path/'inspector',pcv_runtime=True,evidence=evidence)
    assert not (tmp_path/'inspector').exists()


def test_simulator_exports_model_bound_recorded_outputs(fixture_bundle,tmp_path):
    qualify_fixture(fixture_bundle)
    source=json.loads((fixture_bundle/'source-manifest.json').read_text())
    expected={'intent':'POWER_ON','entity_id':'light.kitchen'}
    evidence=tmp_path/'recorded.json';evidence.write_text(json.dumps(dict(binding=source['binding'],commands_with_results=[
        dict(reference=dict(text='turn on kitchen lamp',expected_command=expected),outputs=[dict(
            recording_id='fixture',candidate=expected,engine='fixture',voice_group='fixture',output=dict(accepted=True))])])),encoding='utf-8')
    recipe=export(fixture_bundle,tmp_path/'fixture.pt',tmp_path/'inspector',pcv_runtime=True,evidence=evidence)
    rows=json.loads((tmp_path/'inspector/evidence.json').read_text())['rows']
    assert recipe['evidence_sha256'] and recipe['acceptance_enabled'] is False
    assert rows[0]['outcome']=='correct_accepted' and rows[0]['id']=='fixture'
