import json
import torch
import pytest
from safetensors.torch import save_file


def fixture(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_encoder=torch.nn.Module()
            self.audio_encoder.model=torch.nn.Module()
            self.audio_encoder.model.decoder=torch.nn.Module()
            self.audio_encoder.model.decoder.model=torch.nn.ModuleList([torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(2,3,3))])
            self.other=torch.nn.Linear(2,2)
    m=Model();weights={k.replace('.parametrizations.weight.original0','.weight_g').replace('.parametrizations.weight.original1','.weight_v'):v.detach().clone() for k,v in m.state_dict().items()}
    save_file(weights,str(tmp_path/'model.safetensors'))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in weights}}))
    return m,weights


def test_repairs_known_codec_load_drift_and_verifies_bfloat16(tmp_path):
    from scripts.pcv_parler_weights import repair_and_verify
    m,weights=fixture(tmp_path);m=m.bfloat16()
    layer=m.audio_encoder.model.decoder.model[0]
    with torch.no_grad():layer.parametrizations.weight.original0.fill_(9)
    report=repair_and_verify(m,tmp_path)
    assert report['changed_before']==['audio_encoder.model.decoder.model.0.parametrizations.weight.original0']
    assert report['verified_after'] and report['tensors_checked']==len(weights)
    assert torch.equal(layer.parametrizations.weight.original0,weights['audio_encoder.model.decoder.model.0.weight_g'].bfloat16())


def test_rejects_unrelated_bad_weights_without_repair(tmp_path):
    from scripts.pcv_parler_weights import repair_and_verify
    m,_=fixture(tmp_path)
    with torch.no_grad():m.other.weight.fill_(999)
    with pytest.raises(ValueError,match='Unexplained'):repair_and_verify(m,tmp_path)


def test_reverse_coverage_allows_only_actual_tensor_aliases(tmp_path):
    from scripts.pcv_parler_weights import repair_and_verify
    m,_=fixture(tmp_path);m.alias=m.other.weight
    report=repair_and_verify(m,tmp_path)
    assert report['verified_aliases']['alias']=='other.weight'
    m.register_buffer('unexplained',torch.ones(1))
    with pytest.raises(ValueError,match='Runtime tensor absent'):repair_and_verify(m,tmp_path)


def test_rejects_index_path_escape_and_nonfinite_source(tmp_path):
    from scripts.pcv_parler_weights import repair_and_verify
    m,weights=fixture(tmp_path);index=tmp_path/'model.safetensors.index.json'
    original=index.read_text();index.write_text(original.replace('model.safetensors','../model.safetensors'))
    with pytest.raises(ValueError,match='shard'):repair_and_verify(m,tmp_path)
    index.write_text(original);weights['other.bias'][0]=float('nan');save_file(weights,str(tmp_path/'model.safetensors'))
    with pytest.raises(ValueError,match='Nonfinite'):repair_and_verify(m,tmp_path)


def test_rejects_remapping_collision(tmp_path):
    from scripts.pcv_parler_weights import repair_and_verify
    m,weights=fixture(tmp_path)
    weights['audio_encoder.model.decoder.model.0.parametrizations.weight.original0']=weights['audio_encoder.model.decoder.model.0.weight_g'].clone()
    save_file(weights,str(tmp_path/'model.safetensors'))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in weights}}))
    with pytest.raises(ValueError,match='collision'):repair_and_verify(m,tmp_path)
