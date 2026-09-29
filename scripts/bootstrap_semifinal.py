"""Bootstrap a fresh VCM checkout without relying on historical payloads.

The default action is a dry plan. Use --apply to download version-pinned source
files. Existing files are reused only when their size/hash match; mismatches are
never overwritten. Untracked files are allowed and preserved. Tracked or staged
edits stop the bootstrap.

Examples
--------
python scripts/bootstrap_semifinal.py
python scripts/bootstrap_semifinal.py --profile core --apply
python scripts/bootstrap_semifinal.py --profile semifinal --apply --allow-restricted
python scripts/bootstrap_semifinal.py --profile core --apply --pipeline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Callable
import urllib.request


ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "core": (
        "slurp",
        "speech_commands",
    ),
    "semifinal": (
        "slurp",
        "speech_commands",
        "librispeech",
        "fluent",
        "slurp_audio_mirror",
        "smartlights",
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"Unsafe repository-relative path: {value}")
    return str(path)


def _license_is_restricted(value: str) -> bool:
    text = value.lower().replace("_", "-")
    return any(
        token in text
        for token in (
            "noncommercial",
            "non-commercial",
            "cc-by-nc",
            "academic research only",
            "research only",
            "no commercial",
        )
    )


def _entry_from(
    source: str,
    descriptor: dict,
    item: dict,
) -> dict | None:
    url = item.get("url", descriptor.get("url"))
    if not url:
        return None
    if not str(url).startswith("https://"):
        raise ValueError(f"{source}: only HTTPS dataset URLs are allowed")

    raw_path = item.get("path", descriptor.get("path"))
    if not raw_path:
        filename = item.get("filename", descriptor.get("filename"))
        if not filename:
            raise ValueError(f"{source}: downloadable file has no path/filename")
        raw_path = f"data/raw/{source}/{filename}"
    relative_path = _safe_relative_path(str(raw_path))

    digest = item.get("sha256", descriptor.get("sha256"))
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ):
        raise ValueError(f"{source}: a 64-character SHA-256 is required")

    expected_bytes = item.get("bytes", descriptor.get("bytes"))
    if expected_bytes is not None:
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"{source}: invalid byte count")

    license_text = str(item.get("license", descriptor.get("license", "unknown")))
    return {
        "source": source,
        "url": str(url),
        "relative_path": relative_path,
        "sha256": digest.lower(),
        "bytes": expected_bytes,
        "license": license_text,
        "restricted": _license_is_restricted(license_text),
        "revision": descriptor.get("revision"),
    }


def build_download_plan(
    root: Path = ROOT,
    source_names: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Read checked-in source ledgers and return direct downloadable files."""
    root = Path(root)
    source_folder = root / "data/sources"
    if source_names is None:
        source_names = tuple(
            path.name.removesuffix(".source.json")
            for path in sorted(source_folder.glob("*.source.json"))
        )

    plan: list[dict] = []
    for name in source_names:
        path = source_folder / f"{name}.source.json"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset source ledger is absent: {path}")
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        items = descriptor.get("files")
        if items is None:
            items = [descriptor]
        if not isinstance(items, list):
            raise ValueError(f"{name}: files must be a list")

        for raw_item in items:
            if not isinstance(raw_item, dict):
                raise ValueError(f"{name}: invalid file entry")
            entry = _entry_from(name, descriptor, raw_item)
            if entry is not None:
                plan.append(entry)

    identities = [
        (item["source"], item["relative_path"])
        for item in plan
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate source/path entries in procurement plan")
    return plan


def _validate_existing(destination: Path, entry: dict) -> bool:
    if not destination.is_file():
        return False
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and destination.stat().st_size != expected_bytes:
        raise FileExistsError(
            f"Refusing to overwrite mismatched existing file: {destination}"
        )
    if sha256_file(destination) != entry["sha256"]:
        raise FileExistsError(
            f"Refusing to overwrite mismatched existing file: {destination}"
        )
    return True


def _stream_download(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smart-speaker-vcm-bootstrap/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write_bytes(target.read_bytes() + chunk)


def _stream_download_file(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smart-speaker-vcm-bootstrap/1"},
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        target.open("wb") as stream,
    ):
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)


def procure(
    entry: dict,
    root: Path = ROOT,
    *,
    fetcher: Callable[[str], bytes] | None = None,
) -> dict:
    """Fetch one file atomically, or reuse an exact existing copy."""
    root = Path(root).resolve()
    relative = _safe_relative_path(entry["relative_path"])
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Destination escapes repository: {relative}")

    if _validate_existing(destination, entry):
        return {
            "source": entry["source"],
            "path": relative,
            "status": "reused",
            "sha256": entry["sha256"],
            "bytes": destination.stat().st_size,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".part",
        delete=False,
    ) as temporary_stream:
        temporary = Path(temporary_stream.name)

    try:
        if fetcher is None:
            _stream_download_file(entry["url"], temporary)
        else:
            payload = fetcher(entry["url"])
            if not isinstance(payload, (bytes, bytearray)):
                raise TypeError("fetcher must return bytes")
            temporary.write_bytes(bytes(payload))

        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
            raise ValueError(f"Byte-count mismatch for {relative}")
        if sha256_file(temporary) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative}")

        if destination.exists():
            if _validate_existing(destination, entry):
                return {
                    "source": entry["source"],
                    "path": relative,
                    "status": "reused_after_race",
                    "sha256": entry["sha256"],
                    "bytes": destination.stat().st_size,
                }
            raise FileExistsError(
                f"Refusing to overwrite file created during download: {destination}"
            )
        os.replace(temporary, destination)
        return {
            "source": entry["source"],
            "path": relative,
            "status": "downloaded",
            "sha256": entry["sha256"],
            "bytes": destination.stat().st_size,
        }
    finally:
        temporary.unlink(missing_ok=True)


def classify_status(lines: list[str]) -> dict:
    """Separate harmless untracked work from tracked/staged modifications."""
    untracked = []
    tracked_or_staged = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith("?? "):
            untracked.append(line[3:])
        else:
            tracked_or_staged.append(line)
    return {
        "untracked": untracked,
        "tracked_or_staged": tracked_or_staged,
    }


def repository_status(root: Path = ROOT) -> dict:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return classify_status(result.stdout.splitlines())


def require_tracked_clean(root: Path = ROOT) -> dict:
    status = repository_status(root)
    if status["tracked_or_staged"]:
        formatted = "\n".join(status["tracked_or_staged"])
        raise RuntimeError(
            "Tracked/staged changes must be reconciled before bootstrap:\n"
            + formatted
        )
    return status


def pipeline_stages(run_id: str) -> list[dict]:
    """Stable stage order; qualification must pass before publication."""
    del run_id
    python = sys.executable
    return [
        {
            "name": "train",
            "command": [python, "train.py", "--fit"],
        },
        {
            "name": "validate",
            "command": [python, "final_suite/run_all.py", "--qualify"],
        },
        {
            "name": "publish_diagnostic",
            "command": [python, "publish.py"],
        },
    ]


def run_pipeline(root: Path, run_id: str) -> list[dict]:
    results = []
    for stage in pipeline_stages(run_id):
        result = subprocess.run(
            stage["command"],
            cwd=root,
            check=False,
        )
        record = {
            "name": stage["name"],
            "command": stage["command"],
            "returncode": result.returncode,
        }
        results.append(record)
        if result.returncode:
            break
    return results


def _selected_sources(profile: str, explicit: list[str]) -> tuple[str, ...]:
    if explicit:
        return tuple(dict.fromkeys(explicit))
    return PROFILES[profile]


def _print_plan(plan: list[dict]) -> None:
    total_known = sum(item["bytes"] or 0 for item in plan)
    print(
        json.dumps(
            {
                "files": len(plan),
                "known_bytes": total_known,
                "restricted_files": sum(item["restricted"] for item in plan),
                "sources": sorted({item["source"] for item in plan}),
                "plan": plan,
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="core")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-restricted", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--run-id", default="semifinal")
    args = parser.parse_args()

    root = args.root.resolve()
    status = require_tracked_clean(root)
    sources = _selected_sources(args.profile, args.source)
    plan = build_download_plan(root, list(sources))
    _print_plan(plan)
    if status["untracked"]:
        print(
            f"Preserving {len(status['untracked'])} untracked path(s); "
            "bootstrap never deletes them."
        )

    if not args.apply:
        print("Dry plan only. Re-run with --apply to procure data.")
        return 0

    restricted = sorted(
        {item["source"] for item in plan if item["restricted"]}
    )
    if restricted and not args.allow_restricted:
        raise RuntimeError(
            "Restricted/noncommercial source(s) require --allow-restricted "
            "after license review: " + ", ".join(restricted)
        )

    results = [procure(entry, root) for entry in plan]
    receipt = {
        "profile": args.profile,
        "sources": list(sources),
        "untracked_preserved": status["untracked"],
        "downloads": results,
    }
    receipt_path = root / ".cache/bootstrap/receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote procurement receipt: {receipt_path.relative_to(root)}")

    if args.pipeline:
        pipeline = run_pipeline(root, args.run_id)
        receipt["pipeline"] = pipeline
        receipt_path.write_text(
            json.dumps(receipt, indent=2),
            encoding="utf-8",
        )
        if not pipeline or pipeline[-1]["returncode"]:
            return pipeline[-1]["returncode"] if pipeline else 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
