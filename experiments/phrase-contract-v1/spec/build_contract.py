#!/usr/bin/env python3
"""Reconstruct and execute the frozen PC-v1 contract generator."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "93a4e3e73b61899d08b5dd5bb9690349a028910f094d5a1c3360e1ca09a64ec8"


def frozen_source() -> bytes:
    encoded = "".join(
        path.read_text(encoding="ascii")
        for path in sorted((ROOT / "generator_parts").glob("part*.b64"))
    )
    if not encoded:
        raise RuntimeError("Missing frozen generator parts")
    source = base64.b64decode(encoded, validate=True)
    actual = hashlib.sha256(source).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Frozen generator identity mismatch: {actual}")
    return source


def main() -> None:
    source = frozen_source()
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__"}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)


if __name__ == "__main__":
    main()
