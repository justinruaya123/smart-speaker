"""Run the same microphone/file pipeline as the Pi executable from the checkout."""
from train import register_sources, verify_artifacts
import sys

if __name__ == '__main__':
    register_sources()
    verify_artifacts()
    from tinyvcm.launcher import main
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
