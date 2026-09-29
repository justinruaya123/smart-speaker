"""The historical baseline remains inspectable without executing notebook cells."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_source_extracted_byte_for_byte():
    with zipfile.ZipFile(ROOT / 'docs/finalization/baseline-cd98092.zip') as archive:
        notebook=json.loads(archive.read('Architecture.ipynb'))
    for cell in notebook['cells']:
        metadata = cell.get('metadata', {}).get('checkpoint')
        if not metadata:
            continue
        source = ''.join(cell['source']).split('\n', 1)[1]
        path = ROOT / 'vcm/legacy' / (metadata['module'].replace('.', '/') + '.py')
        assert path.read_text(encoding='utf-8') == source


def test_registration_uses_normal_imports():
    import sys
    from vcm import legacy_workflow as train
    before = list(sys.meta_path)
    train.register_sources(ROOT)
    assert sys.meta_path == before
    import tinyvcm.config
    assert Path(tinyvcm.config.__file__).is_file()
