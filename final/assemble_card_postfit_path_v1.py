"""Versioned assembly-only receipt-path compatibility; frozen checks stay intact.

The original waveform evidence consumer supplies a postfit receipt file to a
directory-only verifier. Scope this path conversion to the original assembly
entrypoint. No candidate, receipt, source file, reference or result is rewritten.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final import assemble_card
from scripts.test import vcm_mixed_benchmark as benchmark


@contextmanager
def postfit_path_compatibility():
    """Only coerce an existing exact-named receipt; call every original check."""
    original = benchmark.verify_postfit

    def verify(context, postfit):
        path = Path(postfit)
        if path.is_file():
            resolved = path.resolve()
            if path.name != "postfit-receipt.json" or resolved.name != "postfit-receipt.json":
                raise ValueError("File input must be an existing postfit-receipt.json")
            postfit = resolved.parent
        return original(context, postfit)

    benchmark.verify_postfit = verify
    try:
        yield
    finally:
        benchmark.verify_postfit = original


def main():
    with postfit_path_compatibility():
        return assemble_card.main()


if __name__ == "__main__":
    raise SystemExit(main())
