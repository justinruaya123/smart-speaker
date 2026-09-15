import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyvcm.data import prepare_slurp, prepare_libri, prepare_task_manifests, prepare_asr_manifests

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=["vcm", "asr"])
args = parser.parse_args()
if args.task == "vcm":
    if not (Path(__file__).resolve().parents[1] / "data/manifests/slurp.all.jsonl").exists():
        prepare_slurp()
    prepare_task_manifests()
else:
    prepare_libri()
    prepare_asr_manifests()
