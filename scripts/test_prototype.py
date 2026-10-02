"""Evaluate locked E18 on historical/revised human tests; never fits a policy."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import register_sources
register_sources()

if __name__ == '__main__':
    from tinyvcm.prototype_eval import evaluate
    evaluate()
