"""Verify local model hashes and load both VCM variants without a mic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/artifacts/tinyvcm"


def verify_bundle(bundle=BUNDLE):
    bundle = Path(bundle)
    metadata = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    expected = {f"{model}.{variant}.onnx" for model in ("vcm", "vcm_stream", "asr")
                for variant in ("fp32", "int8")}
    if set(metadata["files"]) != expected:
        raise ValueError("bundle.json must describe all six runtime models")
    for name in sorted(expected):
        facts = metadata["files"][name]
        path = bundle / name
        if path.stat().st_size != facts["bytes"]:
            raise ValueError(f"Model size mismatch: {name}")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != facts["sha256"]:
            raise ValueError(f"Model SHA-256 mismatch: {name}")
    return sorted(expected)


def main():
    try:
        files = verify_bundle()
        from tinyvcm.inference import Recognizer

        for variant in ("fp32", "int8"):
            Recognizer(BUNDLE, asr=False, variant=variant)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Setup check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "python": sys.version.split()[0],
                      "executable": sys.executable, "verified_models": files,
                      "vcm_variants_loaded": ["fp32", "int8"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
