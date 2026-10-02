"""Run the isolated parameter study using the root environment and source map."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
train.register_sources()
from tinyvcm.parameter_study import main
if __name__ == '__main__':
    main()
