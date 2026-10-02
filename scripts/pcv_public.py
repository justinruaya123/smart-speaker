"""Entry point for the isolated PCV-B1 public-speech study."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['definitions', 'prepare', 'train', 'evaluate'])
    args = parser.parse_args()
    if args.stage == 'definitions':
        from scripts.train.pcv_public_data import make_definitions
        make_definitions()
    elif args.stage == 'prepare':
        from scripts.train.pcv_public_data import prepare
        prepare()
    elif args.stage == 'train':
        from scripts.train.pcv_public_training import train
        train()
    else:
        from scripts.train.pcv_public_training import evaluate
        evaluate()


if __name__ == '__main__':
    main()
