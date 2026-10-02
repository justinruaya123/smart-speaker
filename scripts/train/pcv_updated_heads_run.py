"""Serialization-only retry; preserve the v2 prefit failure and source hash."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.train import pcv_updated_heads as workflow
from scripts.train.pcv_three_engine import immutable
from scripts.train.pcv_public_data import sha


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value


def write(path,value):
    value=native(value)
    if path.name=='protocol.json':
        value['version']='pcv-updated-voice-v2-run1'
        value['retry']='Only NumPy scalar JSON conversion; prior run stopped before feature extraction or fitting'
        value['bindings'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
        value['bindings']['experiments/pcv-updated-voice-v2/protocol.json']=sha(ROOT/'experiments/pcv-updated-voice-v2/protocol.json')
    immutable(path,value)


if __name__=='__main__':
    workflow.HERE=ROOT/'experiments/pcv-updated-voice-v2-run1'
    workflow.CACHE=ROOT/'.cache/pcv-updated-voice-v2-run1'
    workflow.immutable=write
    workflow.run()
