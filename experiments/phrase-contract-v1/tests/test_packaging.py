import json
from pathlib import Path
import zipfile
import numpy as np
import pytest


def test_runtime_zip_is_deterministic_complete_and_detects_tampering(tmp_path):
    import torch
    from vcm.model import CausalVCM, STATE_SHAPES, export_model
    from vcm.frontend import SPEC
    from pcvcm.bundle import seal_bundle
    from pcvcm.contract import Contract, compile_catalog, write_json, file_hash
    from pcvcm.dependencies import lock_wheels, REQUIRED
    from pcvcm.packaging import publish
    from pcvcm.launcher import verify_archive_files
    from test_dependencies import wheel
    from dataclasses import asdict
    from pcvcm.endpoint import EndpointConfig
    root = Path(__file__).resolve().parents[1]
    contract = Contract(root / 'spec')
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    torch.set_num_threads(2); torch.manual_seed(53)
    export_model(CausalVCM(), bundle / 'model.onnx')
    write_json(bundle / 'catalog.json', compile_catalog(contract, root / 'configs/lexicon.json'))
    write_json(bundle / 'normalization.json', {'mean': [0.]*40, 'std': [1.]*40, 'fit_split': 'train', 'frontend_spec': SPEC})
    write_json(bundle / 'endpoint.json', asdict(EndpointConfig()))
    write_json(bundle / 'endpoint-selection.json', {'status': 'SOFTWARE_FIXTURE_ONLY'})
    seal_bundle(bundle, contract, [list(s) for s in STATE_SHAPES], 'SOFTWARE_FIXTURE_ONLY')
    wheels = tmp_path / 'wheels'; wheels.mkdir()
    for name in REQUIRED:
        wheel(wheels, name, '1.0', 'py3-none-any')
    lock_wheels(wheels, tmp_path / 'deps')
    a, b = tmp_path / 'a.zip', tmp_path / 'b.zip'
    for path in (a, b):
        report = publish(bundle, path, contract, tmp_path / 'deps', wheels, diagnostic=True)
        assert report['qualified'] is False
    assert file_hash(a) == file_hash(b)
    with zipfile.ZipFile(a) as z:
        assert sum(n.endswith('.onnx') for n in z.namelist()) == 1
        assert not any('training' in n or 'torch' in n for n in z.namelist())
        assert {'pcv1.py', 'requirements.txt', 'THIRD_PARTY_NOTICES.txt'} <= set(z.namelist())
        import ast
        for name in z.namelist():
            if not name.endswith('.py'):
                continue
            source = z.read(name).decode('utf-8')
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert not any(n.name.startswith('torch') for n in node.names), name
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith(('torch', 'vcm.model')), name
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert node.name not in {'export_selected', 'prepare', 'fit', 'calibrate', 'training_smoke'}, name
        z.extractall(tmp_path / 'extracted')
    verify_archive_files(tmp_path / 'extracted')
    import subprocess
    import sys
    import soundfile as sf
    pcm = np.r_[np.zeros(4800), np.sin(np.arange(8000) * .13) * .1, np.zeros(16000)].astype('float32')
    sf.write(tmp_path / 'software-fixture.wav', pcm, 16000, subtype='FLOAT')
    # Fresh interpreter outside the checkout. This verifies software packaging,
    # using installed host dependencies, not a physical ARM64 installation.
    guard = (
        "import sys,socket,runpy; "
        "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network forbidden')); "
        "entry=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(entry,run_name='__main__'); "
        "assert 'torch' not in sys.modules and 'vcm.model' not in sys.modules"
    )
    for args in [['--check'], ['--wav', str(tmp_path / 'software-fixture.wav')]]:
        completed = subprocess.run([sys.executable, '-I', '-c', guard,
            str(tmp_path / 'extracted/pcv1.py'), *args], cwd=tmp_path,
            capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        if args == ['--check']:
            assert result['integrity'] == 'verified'
        else:
            assert result['accepted'] is False
            assert result['reason'] == 'policy_disabled'
    (tmp_path / 'extracted/bundle/policy.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        verify_archive_files(tmp_path / 'extracted')
