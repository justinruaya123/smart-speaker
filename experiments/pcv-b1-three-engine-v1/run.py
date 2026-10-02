"""Thin entry point using the configured root environment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.train.pcv_three_engine import main
if __name__=='__main__':
    main()
