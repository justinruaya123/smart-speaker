"""Reconstruct pinned data and rerun the selected seed-53 compact experiment."""
from scripts.submission.reproduce import main

if __name__ == '__main__':
    import subprocess
    import sys
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('Reproduction:', error, file=sys.stderr)
        raise SystemExit(1)
