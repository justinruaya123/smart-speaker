"""Public dataset acquisition. All writes are confined to this STT directory."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
REPO = ROOT.parent

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def fetch(url, relative, expected=None):
    dest = ROOT / relative
    assert dest.resolve().is_relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = dest.with_name(dest.name + ".source.json")
    if dest.exists() and receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))
    start = time.monotonic()
    part = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        try:
            resume = ["-C", "-"] if part.exists() and part.stat().st_size > 0 else []
            subprocess.run(["curl.exe", "-L", "-sS", "--fail", "--connect-timeout", "20",
                            "--max-time", "300", *resume, "-o", str(part), url], check=True)
            actual = sha(part)
            if expected and actual != expected:
                raise ValueError(f"Checksum mismatch: {relative}, {part.stat().st_size} bytes, sha256={actual}")
            part.replace(dest)
            result = {"url": url, "path": relative, "sha256": actual,
                      "bytes": dest.stat().st_size,
                      "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                      "seconds": round(time.monotonic() - start, 2)}
            dump(receipt, result)
            print(f"Downloaded {relative}: {result['bytes']:,} bytes in {result['seconds']}s", flush=True)
            return result
        except Exception as exc:
            print(f"Download attempt {attempt + 1} {relative}: {exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

def snapshot():
    """Capture existing pipeline files without importing or executing pipeline code."""
    output = ROOT / "evidence" / "pipeline_before.json"
    if output.exists():
        return
    records = {}
    for base, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in {"STT", ".git", ".venv", ".cache", "__pycache__"}]
        for name in files:
            path = Path(base) / name
            rel = path.relative_to(REPO).as_posix()
            st = path.stat()
            rec = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
            if not rel.startswith("data/audio/"):
                rec["sha256"] = sha(path)
            records[rel] = rec
    dump(output, records)
    print(f"Protected pipeline snapshot: {len(records)} files", flush=True)

def basic():
    snapshot()
    tasks = [
        ("https://api.github.com/repos/Jakobovski/free-spoken-digit-dataset/commits/master", "data/raw/fsdd/revision.json", None),
        ("https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218/data/splits/en/splits.tar.gz?download=true", "data/raw/mswc/splits.tar.gz", "11c773dc47fece8993693798414f04475e19d596671e1742a3dd040e66d6cbce"),
        ("https://raw.githubusercontent.com/ankilab/SpokeN-100/main/README.md", "data/raw/spoken100/SOURCE.md", None),
    ]
    rev = "8eb16545762be97ace75334109d73824217311f1"
    for name in ("train.jsonl", "devel.jsonl", "test.jsonl", "metadata.json"):
        tasks.append((f"https://raw.githubusercontent.com/pswietojanski/slurp/{rev}/dataset/slurp/{name}", f"data/raw/slurp/{name}", None))
    tasks.append((f"https://raw.githubusercontent.com/pswietojanski/slurp/{rev}/README.md", "data/raw/slurp/SOURCE.md", None))
    source = json.loads((REPO / "data/sources/slurp_audio_mirror.source.json").read_text())
    for item in source["files"]:
        tasks.append((item["url"], "data/raw/slurp/mirror/" + Path(item["path"]).name, item["sha256"]))
    tasks += [
        ("https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz", "data/raw/speech_commands_v0.02.tar.gz", "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"),
    ]
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"task": futures[future], "error": str(exc)})
    revision_path = RAW / "fsdd/revision.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text())["sha"]
        results.append(fetch(f"https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/{revision}", "data/raw/fsdd/fsdd.zip"))
    dump(ROOT / "evidence/downloads_basic.json", {"downloads": results, "failures": failures})
    print(f"Basic downloads finished: {len(results)} successful, {len(failures)} failed", flush=True)

if __name__ == "__main__":
    basic()
