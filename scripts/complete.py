"""Versioned complete-command research; does not change installed artifacts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import register_sources
register_sources()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train','finalize','diagnose','lock','evaluate'])
    parser.add_argument('--arm',choices=['human_full','human_filtered','human_keywords','mixed_keywords','v2_background','v2_masked','v5_cues','v5_cues_masked'])
    args=parser.parse_args()
    if args.stage=='prepare':
        from tinyvcm.complete_data import prepare
        prepare()
    elif args.stage=='train':
        if not args.arm:parser.error('train requires --arm')
        from tinyvcm.complete_training import train
        train(args.arm)
    elif args.stage=='finalize':
        if not args.arm:parser.error('finalize requires --arm')
        from tinyvcm.complete_training import finalize
        finalize(args.arm)
    elif args.stage=='diagnose':
        if not args.arm:parser.error('diagnose requires --arm')
        from tinyvcm.complete_diagnosis import diagnose
        diagnose(args.arm)
    elif args.stage=='lock':
        from tinyvcm.complete_eval import lock
        lock()
    else:
        from tinyvcm.complete_eval import evaluate
        evaluate()


if __name__=='__main__':main()
