"""Prepare/freeze E18 data or train; never opens held-out test predictions."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import register_sources
register_sources()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--arm', choices=['uniform', 'source_balanced'])
    args = parser.parse_args()
    if args.prepare:
        if args.arm: parser.error('--prepare and --arm are separate steps')
        from tinyvcm.prototype_data import prepare
        prepare()
    elif args.arm:
        from tinyvcm.prototype_training import train
        train(args.arm)
    else:
        parser.error('Choose --prepare or --arm')
