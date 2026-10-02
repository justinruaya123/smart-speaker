"""Host-only selection of a known frozen study; never shipped in the Pi runtime."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def bind_study(protocol_directory):
    version = json.loads((Path(protocol_directory)/'protocol.json').read_text(encoding='utf-8')).get('version')
    sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
    if version == 'pcv-synthetic-v2':
        import revision
        from version_context import bind
        validate = revision.validate
    elif version in ('pcv-synthetic-v2-repair1','pcv-synthetic-v3'):
        name = '_pcv_training_repair1' if version=='pcv-synthetic-v2-repair1' else '_pcv_synthetic_v3'
        if name not in sys.modules:
            path = ROOT/'experiments'/version/'workflow.py'
            spec = importlib.util.spec_from_file_location(name,path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module); sys.modules[name] = module
        module = sys.modules[name]; bind,validate = module.bind,module.validate
    else:
        raise ValueError(f'Unsupported frozen study version: {version!r}')
    validate()
    with bind():
        import study
        yield study
