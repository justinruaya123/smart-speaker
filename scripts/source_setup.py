"""Workstation-only dependencies; never imported or shipped by the Pi runtime."""
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys

from scripts.runtime.speaker_tools import find_uv,run

BUILD = {'pyarrow':'25.0.1','torch':'2.13.0','pnnx':'20260526'}
TRAIN = {'torch':'2.13.0','onnx':'1.22.0','onnxscript':'0.7.2','transformers':'4.57.3',
         'huggingface-hub':'0.36.2','pytest':'9.1.1','requests':'2.34.2','tqdm':'4.70.1'}


def ensure_dependencies(root, *, training=False):
    wanted={**BUILD,**(TRAIN if training else {})}
    missing=[]
    for package,expected in wanted.items():
        try:current=version(package).split('+')[0]
        except PackageNotFoundError:current=None
        if current!=expected:missing.append(package+'=='+expected)
    if missing:run([find_uv(root),'pip','install','--python',sys.executable,*missing],cwd=root)
    if training:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('The frozen release training workflow requires a CUDA workstation. Publish/serve/benchmark work on CPU with --skip-train. No data or fitting was started.')
