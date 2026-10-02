"""One entry point for building, publishing and running smart-speaker."""
from pathlib import Path
from scripts.runtime.speaker_start import main

if __name__ == '__main__':
    try:
        raise SystemExit(main(Path(__file__).resolve().parent))
    except (OSError, ValueError, RuntimeError) as error:
        import sys
        print('Smart speaker:', error, file=sys.stderr)
        raise SystemExit(1)
