"""Reproduce, evaluate and package Iris; optionally publish verified assets."""


def main(argv=None):
    from pathlib import Path
    import subprocess
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(flag in argv for flag in ('-h', '--help')):
        from scripts.iris_release import ensure_source_environment
        index = argv.index('--device') if '--device' in argv else -1
        device = argv[index + 1] if 0 <= index < len(argv) - 1 else 'cpu'
        python = ensure_source_environment(device=device)
        if Path(sys.prefix).resolve() != python.parent.parent.resolve():
            return subprocess.call([str(python), str(Path(__file__).resolve()), *argv])
    from scripts.iris_release import main as release_main
    return release_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
