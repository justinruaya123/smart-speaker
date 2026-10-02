"""Run isolated phonetic keyword/grammar experiments from the root environment."""
import os
import sys
from pathlib import Path
# Study timings/calibration use one BLAS thread plus two ONNX intra-op threads.
# Pin before NumPy is imported; this affects only this isolated CLI process.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
train.register_sources()
from tinyvcm.phonetic_training import main
if __name__ == '__main__':
    main()
