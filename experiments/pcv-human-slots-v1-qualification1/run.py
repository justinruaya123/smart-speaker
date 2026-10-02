"""Thin entry point; this child never modifies parent training or test data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze','generate','evaluate'])
    args = p.parse_args()
    if args.stage == 'evaluate':
        from scripts.test.pcv_public_qualification import evaluate
        evaluate()
    else:
        from scripts.train.pcv_calibration import freeze_child, generate
        print(freeze_child() if args.stage=='freeze' else generate())


if __name__=='__main__':
    main()
