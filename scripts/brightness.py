"""Brightness experiment facade; use the root environment and implementation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('fetch','prepare','train','benchmark','install','all'))
    parser.add_argument('--config',type=Path,default=ROOT/'pipeline.toml')
    parser.add_argument('--extra-manifest',type=Path,help='workspace JSONL with audio, transcript, speaker_id, source, split and id')
    parser.add_argument('--refresh-data',action='store_true',help='explicitly rebuild frozen STT manifests')
    args=parser.parse_args()
    if args.extra_manifest and not args.refresh_data:parser.error('--extra-manifest requires --refresh-data')
    config=train.load_config(args.config)
    from tinyvcm.brightness_workflow import run
    print(json.dumps(run(ROOT,config,args.action,args.extra_manifest,args.refresh_data),indent=2))


if __name__=='__main__':main()
