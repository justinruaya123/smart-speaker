"""Replay the frozen keyword study through the real file/endpoint entry points."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train
train.register_sources()
from tinyvcm.keyword_replay_eval import run
if __name__=='__main__':run()
