"""Run with python -m src.voice --mic NUMBER from the repository root."""
from __future__ import annotations

import argparse
import json
import sys

from .service import DEFAULT_BUNDLE, start_service


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.voice", description="VAD-triggered microphone intent debug service")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", type=int, help="numeric input device index from --list-mics")
    mode.add_argument("--list-mics", action="store_true", help="list input device numbers and exit")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, help="TinyVCM model bundle directory")
    parser.add_argument("--channel", type=int, default=0, help="input channel (default: 0)")
    parser.add_argument("--capture-rate", type=int, help="capture rate in Hz; default: 16 kHz or native rate")
    args = parser.parse_args(argv)
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        print(f"Unable to load microphone runtime: {exc}. Run uv sync --locked "
              "from the repository root and install the system PortAudio library.", file=sys.stderr)
        return 1
    try:
        if args.list_mics:
            hosts = sd.query_hostapis()
            devices = sd.query_devices()
            for index, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    print(f"{index}: {device['name']} [{hosts[device['hostapi']]['name']}] "
                          f"({device['max_input_channels']} input channels)")
            if not any(device["max_input_channels"] > 0 for device in devices):
                print("No microphone inputs found.")
        else:
            start_service(args.mic, bundle=args.bundle, channel=args.channel, capture_rate=args.capture_rate)
    except ImportError as exc:
        print(f"Missing voice dependency: {exc}. Run uv sync --locked from the repository root.",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, sd.PortAudioError) as exc:
        print(json.dumps({"stage": "service", "status": "error", "reason": str(exc)}),
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
