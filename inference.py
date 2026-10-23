"""Iris: frozen 258k epoch44 local inference (never trains)."""
import os

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'1')

from scripts.iris_inference import main


if __name__ == '__main__':
    raise SystemExit(main())
