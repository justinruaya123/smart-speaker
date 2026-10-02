"""Export and calibrate on development, then lock selection before test."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import register_sources
register_sources()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--arm', choices=['uniform', 'source_balanced'])
    mode.add_argument('--lock', action='store_true')
    args = parser.parse_args()
    from tinyvcm.prototype_validation import validate, lock
    validate(args.arm) if args.arm else lock()
