"""Prepare the pinned human command corpus without modifying previous studies."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vcm.data import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); prepare(Path(__file__).resolve().parents[1],a.output)
