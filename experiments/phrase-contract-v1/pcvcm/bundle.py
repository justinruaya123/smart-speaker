"""Immutable graph export and identity-bound runtime assets; not qualification."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .contract import Contract, compile_catalog, file_hash, identity, write_json
from .data import below
from .endpoint import EndpointConfig

ASSETS = ('model.onnx', 'catalog.json', 'normalization.json', 'endpoint.json')
RUNTIME_CODE = ('contract.py', 'data.py', 'decoder.py', 'endpoint.py',
                'policy.py', 'runtime.py', 'bundle.py', 'evidence.py')


def code_identity() -> dict:
    import vcm.frontend

    root = Path(__file__).parent
    output = {f'pcvcm/{name}': file_hash(root / name) for name in RUNTIME_CODE}
    output['vcm/frontend.py'] = file_hash(Path(vcm.frontend.__file__))
    return output


def seal_bundle(path: Path, contract: Contract, state_shapes: list,
                status: str) -> dict:
    hashes = {name: file_hash(path / name) for name in ASSETS}
    binding = identity({'assets': hashes, 'code': code_identity(),
                        'contract': contract.sha256})
    policy_path = path / 'policy.json'
    if not policy_path.exists():
        write_json(policy_path, {'enabled': False, 'binding': binding,
                                'thresholds': {'global': [0., 1.]},
                                'status': 'UNCALIBRATED'})
    if json.loads(policy_path.read_text())['binding'] != binding:
        raise ValueError('Policy belongs to different runtime assets')
    recipe = {'format_version': 2, 'status': status, 'qualified': False,
              'contract_sha256': contract.sha256, 'binding': binding,
              'state_shapes': state_shapes, 'code': code_identity(),
              'files': {**hashes, 'policy.json': file_hash(policy_path)},
              'endpoint_source': 'causal_PCM_not_WAV_EOF',
              'raw_logits': True, 'action_execution': False}
    write_json(path / 'recipe.json', recipe)
    return recipe


def verify_bundle(path: Path, contract: Contract) -> dict:
    path = Path(path)
    recipe = json.loads((path / 'recipe.json').read_text())
    if recipe.get('format_version') != 2:
        raise ValueError('Requires a PC-v1 bundle')
    if set(recipe['files']) != set(ASSETS) | {'policy.json'}:
        raise ValueError('Bundle has a missing or unexpected bound asset')
    if recipe['contract_sha256'] != contract.sha256:
        raise ValueError('Command contract changed')
    if recipe['code'] != code_identity():
        raise ValueError('Runtime/frontend source identity changed')
    for name, expected in recipe['files'].items():
        if file_hash(below(path, name)) != expected:
            raise ValueError(f'Bundle file changed: {name}')
    hashes = {name: recipe['files'][name] for name in ASSETS}
    binding = identity({'assets': hashes, 'code': recipe['code'],
                        'contract': contract.sha256})
    policy = json.loads((path / 'policy.json').read_text())
    if recipe['binding'] != binding or policy.get('binding') != binding:
        raise ValueError('Policy/runtime identity mismatch')
    from vcm.frontend import SPEC
    shapes = [[1, 40, 2], [1, 64, 4], [1, 64, 8], [1, 64, 16],
              [1, 64, 32], [1, 64, 64]]
    if recipe['state_shapes'] != shapes:
        raise ValueError('Baseline cache shapes changed')
    norm = json.loads((path / 'normalization.json').read_text())
    if norm.get('frontend_spec') != SPEC or norm.get('fit_split') != 'train':
        raise ValueError('Frontend or normalization provenance mismatch')
    catalog = json.loads((path / 'catalog.json').read_text())
    expected = {r.get('phrase_id', r.get('rejection_id')): r
                for r in contract.positive + contract.negative}
    if (len(catalog.get('entries', [])) != 770
            or {r['id'] for r in catalog['entries']} != set(expected)):
        raise ValueError('Incomplete runtime phrase catalog')
    for row in catalog['entries']:
        reference = expected[row['id']]
        if (row['text'] != reference['exact_text']
                or identity(row['command']) !=
                identity(reference.get('expected_command'))):
            raise ValueError('Runtime catalog changed a required phrase')
    return {'recipe': recipe, 'binding': binding, 'policy': policy,
            'model_path': path / 'model.onnx', 'state_shapes': shapes,
            'normalization': norm,
            'catalog': catalog,
            'endpoint': json.loads((path / 'endpoint.json').read_text())}


def export_selected(run: Path, output: Path, contract: Contract,
                    lexicon: Path, endpoint: EndpointConfig) -> dict:
    run, output = Path(run), Path(output)
    selection = json.loads((run / 'selection.json').read_text())
    if selection.get('status') != 'DEVELOPMENT_SELECTED':
        raise ValueError('No development-selected candidate; keep failed evidence')
    checkpoint = Path(selection['winner']['checkpoint'])
    if file_hash(checkpoint) != selection['winner']['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    if output.exists():
        raise FileExistsError('Export output already exists')
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import torch
    from vcm.model import CausalVCM, STATE_SHAPES, export_model
    from .training import native_cpu_convolution

    catalog = compile_catalog(contract, lexicon)
    saved = json.loads((run / 'catalog.json').read_text())
    if identity(catalog) != identity(saved):
        raise ValueError('Training/export catalogs differ')
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CausalVCM()
    model.load_state_dict(loaded['weights'])
    output.mkdir(parents=True)
    with native_cpu_convolution():
        report = export_model(model, output / 'model.onnx')
    write_json(output / 'export.json', report)
    write_json(output / 'catalog.json', catalog)
    shutil.copyfile(run / 'normalization.json', output / 'normalization.json')
    write_json(output / 'endpoint.json', asdict(endpoint))
    recipe = seal_bundle(output, contract, [list(s) for s in STATE_SHAPES],
                         'DEVELOPMENT_CANDIDATE_UNCALIBRATED')
    verify_bundle(output, contract)
    return recipe
