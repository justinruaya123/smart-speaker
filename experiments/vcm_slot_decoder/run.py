"""Isolated E19 entry point. No registration in the shipping source map."""
import os
import sys
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train
train.register_sources()
sys.path.insert(0, str(ROOT / 'scripts' / 'models'))
sys.path.insert(0, str(ROOT / 'scripts' / 'test'))

if __name__ == '__main__':
    from vcm_slot_decoder_eval import main
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
