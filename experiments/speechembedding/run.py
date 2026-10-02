"""Research entry point; all implementation is registered in scripts/modules.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train

train.register_sources()
from tinyvcm.speech_embedding_study import main
from threadpoolctl import threadpool_limits

if __name__ == '__main__':
    # The historical feature cache was built with one BLAS thread. This also
    # makes fresh-PCM queries numerically match development calibration.
    with threadpool_limits(limits=1, user_api='blas'):
        main()
