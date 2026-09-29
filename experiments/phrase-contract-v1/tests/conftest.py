"""Make the isolated experiment importable from the repository test suite."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcvcm.initialize import initialize

initialize(Path(__file__).resolve().parents[1])
