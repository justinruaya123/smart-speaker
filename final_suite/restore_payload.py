"""Restore explicit research payload paths without touching production files.

The Git branch versions source and the compact notebook. Large generated models,
waveforms, complete evidence and historical implementations travel in the ZIP.
Default is a dry run; existing nonidentical files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import stat
import zipfile

ALLOWED = ("experiments/vcm_streaming_v3/", "experiments/vcm_acceptance_v4/",
           "experiments/vcm_acceptance_v5/", "experiments/vcm_services_v6/",
           "final_suite/", "visualizer/")
MAX_BYTES = 300 * 1024 * 1024


def restore(archive: Path, root: Path, apply: bool = False,
            sha256: str | None = None) -> dict:
    raw = archive.read_bytes()
    if sha256 and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Archive SHA-256 mismatch")
    root = root.resolve()
    plans, unchanged, skipped = [], 0, []
    with zipfile.ZipFile(archive) as package:
        if sum(info.file_size for info in package.infolist()) > MAX_BYTES:
            raise ValueError("Archive exceeds the bounded payload size")
        names = [PurePosixPath(name) for name in package.namelist()]
        if not names or len({p.parts[0] for p in names if p.parts}) != 1:
            raise ValueError("Payload must have one top-level directory")
        seen = set()
        for info in package.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts
                    or "\\" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("Unsafe archive path or symbolic link")
            if info.is_dir() or len(path.parts) < 2:
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in seen:
                raise ValueError("Duplicate archive member")
            seen.add(relative)
            if not relative.startswith(ALLOWED):
                skipped.append(relative)
                continue
            if any(part in {".git", "__pycache__", "node_modules"}
                   for part in PurePosixPath(relative).parts):
                skipped.append(relative)
                continue
            destination = root / relative
            if not destination.resolve().is_relative_to(root):
                raise ValueError("Destination escapes repository")
            content = package.read(info)
            if destination.exists():
                if not destination.is_file() or destination.read_bytes() != content:
                    # Source files already committed may differ only because the
                    # new branch deliberately omits generated output bodies.
                    if relative.endswith((".py", ".mjs", ".c", ".html", ".css", ".md")):
                        skipped.append(relative)
                        continue
                    raise FileExistsError(f"Conflicting payload: {relative}")
                unchanged += 1
            else:
                plans.append((destination, content))
        if apply:
            for destination, content in plans:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(content)
    return {"apply": apply, "new_files": len(plans), "unchanged": unchanged,
            "skipped": skipped, "archive_sha256": hashlib.sha256(raw).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    print(restore(args.archive, args.repo, args.apply, args.sha256))
