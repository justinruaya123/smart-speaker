"""Published identity characterization; no fitting, capture or actions."""
import hashlib
import json
from pathlib import Path

from scripts.optionb_epoch44 import verify_frozen

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'submission/models/optionb-tiny'
HASHES = {
    'model.onnx': 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce',
    'commands.json': '71017cf22366d15557e8878636135f5ff2d0ac81fc4dfd6f058fc88f0f2d4b5d',
    'normalization.json': '28f7ef1b2c926c587d611d17eef91184c2c7e2de66abe352d1fc2365e45d69ed',
    'policy.json': 'b9242e46fb093642a41023e8ace3c0cadc2fe51ffede512aa09e399622f355f0',
    'recipe.json': '020c3deccb0f8cece0f23b8c016b6a63bdd0359321bb06cbd00e091966d46621',
}


def test_fifteen_original_payloads_and_five_independent_pins():
    result = verify_frozen(ROOT)
    assert result['verified_payloads'] == 15
    assert result['parameters'] == 258912 and result['selected_epoch'] == 44
    assert result['original_CPU_model_retrained'] is False
    assert result['actions_enabled'] is False and result['qualified'] is False
    assert {name: hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() for name in HASHES} == HASHES
    assert (BUNDLE / 'model.onnx').stat().st_size == 1082940
    commands = json.loads((BUNDLE / 'commands.json').read_bytes())
    assert len(commands) == 32 and commands[0] is None
    assert len({json.dumps(c, sort_keys=True) for c in commands[1:]}) == 31
    assert json.loads((BUNDLE / 'policy.json').read_bytes()) == {
        'enabled': True, 'min_probability': .5, 'min_margin': .1}


def test_actual_cpu_onnx_load_is_not_an_acoustic_or_retraining_claim():
    result = verify_frozen(ROOT, load=True)
    assert result['output_shape'] == [1, 32]
    assert result['providers'] == ['CPUExecutionProvider']
    assert result['inference_checked'] is True
    assert result['acoustic_quality_claimed'] is False
    assert result['optimizer_updates'] == 0 and result['live_capture_opened'] is False
