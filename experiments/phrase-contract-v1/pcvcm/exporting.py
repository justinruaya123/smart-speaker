"""Development-selected export tooling; excluded from runtime packages."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from .contract import Contract, compile_catalog, file_hash, identity, write_json
from .endpoint import EndpointConfig
from .bundle import seal_bundle, verify_bundle


def export_selected(run: Path, output: Path, contract: Contract,
                    lexicon: Path, endpoint: EndpointConfig, endpoint_selection: Path | None = None) -> dict:
    run, output = Path(run), Path(output)
    selection = json.loads((run / 'selection.json').read_text())
    if selection.get('status') != 'DEVELOPMENT_SELECTED':
        raise ValueError('No development-selected candidate; keep failed evidence')
    checkpoint = Path(selection['winner']['checkpoint'])
    if file_hash(checkpoint) != selection['winner']['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    if endpoint_selection is None:
        raise ValueError('Development endpoint selection receipt required')
    endpoint_receipt = json.loads(Path(endpoint_selection).read_text())
    if (endpoint_receipt.get('status') != 'ENDPOINT_SELECTED'
            or endpoint_receipt.get('checkpoint_sha256') != file_hash(checkpoint)
            or endpoint_receipt.get('selection_sha256') != file_hash(run / 'selection.json')
            or endpoint_receipt.get('endpoint_sha256') != file_hash(Path(endpoint_selection).parent / 'endpoint.json')
            or json.loads((Path(endpoint_selection).parent / 'endpoint.json').read_text()) != asdict(endpoint)):
        raise ValueError('Selected endpoint binding mismatch')
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
    shutil.copyfile(endpoint_selection, output / 'endpoint-selection.json')
    recipe = seal_bundle(output, contract, [list(s) for s in STATE_SHAPES],
                         'DEVELOPMENT_CANDIDATE_UNCALIBRATED')
    verify_bundle(output, contract)
    return recipe
