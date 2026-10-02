"""Research fixed-keyword command pipeline, using the root environment."""
import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train
train.register_sources()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=['prepare','train','train-intent','export','calibrate','evaluate','check','report','predict','listen','list-mics'])
    p.add_argument('--mode',choices=['mixed','synthetic','both'],default='mixed')
    p.add_argument('--audio',type=Path);p.add_argument('--wake-end-ms',type=float,default=0.)
    p.add_argument('--vad',action='store_true');p.add_argument('--mic',type=int);p.add_argument('--channel',type=int,default=0)
    p.add_argument('--capture-rate',type=int);p.add_argument('--vad-debug',action='store_true')
    args=p.parse_args()
    if args.command=='prepare':
        from tinyvcm.keyword_data import prepare
        prepare()
    elif args.command=='train-intent':
        from tinyvcm.keyword_training import fit_intent
        fit_intent()
    elif args.command in ('train','export'):
        from tinyvcm.keyword_training import fit,export
        for mode in (['mixed','synthetic'] if args.mode=='both' else [args.mode]):
            (fit if args.command=='train' else export)(mode)
    else:
        from tinyvcm.keyword_eval import main
        main(args)
