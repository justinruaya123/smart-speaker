"""Replay a frozen manifest through the deployed ONNX path and write all outcomes."""
import argparse
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from vcm.evaluation import evaluate
    print(evaluate(a.bundle,a.manifest,a.output))
