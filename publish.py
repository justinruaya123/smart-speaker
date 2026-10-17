"""Verify/rebuild the selected Iris release, then publish its pinned artifacts."""
from scripts.iris_release import main
if __name__ == '__main__':
    import sys
    raise SystemExit(main(['--skip-train','--publish',*sys.argv[1:]]))
