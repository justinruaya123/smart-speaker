"""Verify frozen epoch44 by default; source fits and legacy Iris are explicit."""
import argparse
from pathlib import Path
import subprocess
import sys


def legacy_main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg in ('-h', '--help') for arg in args):
        from scripts.iris_release import ensure_source_environment
        device = args[args.index('--device') + 1] if '--device' in args else 'cpu'
        python = ensure_source_environment(device=device)
        if Path(sys.prefix).resolve() != (Path(__file__).resolve().parent / '.venv').resolve():
            return subprocess.call([str(python), str(Path(__file__).resolve()), '--model-choice', 'legacy', *args])
    from scripts.iris_reproduce import main as reproduce
    return reproduce(args)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-choice',choices=['epoch44','legacy'],default='epoch44')
    selected, remaining = parser.parse_known_args(args)
    if selected.model_choice == 'legacy':
        return legacy_main(remaining)
    from scripts.iris_reproduction import main as reproduce
    return reproduce(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
