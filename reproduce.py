"""Install the pinned environment, train and benchmark Iris from a Git clone."""
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg in ('-h', '--help') for arg in args):
        from scripts.iris_release import ensure_source_environment
        device = args[args.index('--device') + 1] if '--device' in args else 'cpu'
        python = ensure_source_environment(device=device)
        if Path(sys.prefix).resolve() != (Path(__file__).resolve().parent / '.venv').resolve():
            return subprocess.call([str(python), str(Path(__file__).resolve()), *args])
    from scripts.iris_reproduce import main as reproduce
    return reproduce(args)


if __name__ == '__main__':
    raise SystemExit(main())
