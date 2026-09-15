"""Download original public data with checksums and an auditable source ledger."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def request(url, **kwargs):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "TinyVCM-research/0.1", **kwargs}), timeout=90)


def get_json(url):
    with request(url) as response:
        return json.load(response)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_ledger(name):
    """Versioned source ledgers pin acquisition; never silently follow latest."""
    path = ROOT / "data/manifests" / f"{name}.source.json"
    if not path.is_file():
        raise RuntimeError(f"Missing pinned source ledger: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checked_sha256(path, expected):
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"Pinned source checksum mismatch: {path}")
    return actual


def download(url, target, expected_md5=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_md5 and digest(target, "md5") != expected_md5:
            raise RuntimeError(f"Existing download checksum mismatch: {target}")
        print(f"EXISTS {target.name} {target.stat().st_size}", flush=True)
        return
    with request(url, **{"Range": "bytes=0-0"}) as probe:
        content_range = probe.headers.get("Content-Range", "")
        size = int(content_range.rsplit("/", 1)[1]) if probe.status == 206 and "/" in content_range else 0
    if size > 64 * 1024 * 1024:
        return download_ranges(url, target, size, expected_md5)
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with request(url, **({"Range": f"bytes={offset}-"} if offset else {})) as response:
                if offset and response.status != 206:
                    offset = 0
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned the wrong byte range")
                started = last = time.monotonic()
                size = offset
                with partial.open("ab" if offset else "wb") as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                        size += len(chunk)
                        if time.monotonic() - last > 20:
                            print(f"DOWNLOAD {target.name}: {size / 2**20:.1f} MiB ({(size-offset)/2**20/(time.monotonic()-started):.1f} MiB/s)", flush=True)
                            last = time.monotonic()
                length = response.headers.get("Content-Length")
                if length is not None and size - offset != int(length):
                    raise RuntimeError("Truncated download")
            if expected_md5 and digest(partial, "md5") != expected_md5:
                raise RuntimeError(f"Publisher checksum mismatch: {target.name}")
            partial.replace(target)
            print(f"DOWNLOADED {target.name} {size}", flush=True)
            return
        except Exception as exc:
            print(f"RETRY {target.name} {attempt+1}: {exc}", flush=True)
            if "checksum mismatch" in str(exc):
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Download failed: {url}")


def download_ranges(url, target, size, expected_md5):
    """Bounded parallel ranges; validate every returned range before accepting it."""
    chunk_size = 8 * 1024 * 1024
    parts = target.parent / (target.name + ".parts")
    parts.mkdir(exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        with partial.open("rb") as source:
            for index in range(partial.stat().st_size // chunk_size):
                data = source.read(chunk_size)
                part = parts / f"{index:05d}"
                if not part.exists():
                    part.write_bytes(data)
    def part_download(index):
        begin = index * chunk_size
        end = min(size, begin + chunk_size) - 1
        part = parts / f"{index:05d}"
        if part.exists() and part.stat().st_size == end - begin + 1:
            return part.stat().st_size
        for attempt in range(6):
            try:
                with request(url, **{"Range": f"bytes={begin}-{end}"}) as response:
                    expected = f"bytes {begin}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Wrong Content-Range: {response.headers.get('Content-Range')}, expected {expected}")
                    data = response.read(end - begin + 2)
                if len(data) != end - begin + 1:
                    raise RuntimeError("Incorrect range byte count")
                temporary = part.with_suffix(".partial")
                temporary.write_bytes(data)
                temporary.replace(part)
                return len(data)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 15))
    total = (size + chunk_size - 1) // chunk_size
    done = 0
    start = last = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for future in concurrent.futures.as_completed([pool.submit(part_download, i) for i in range(total)]):
            done += future.result()
            if time.monotonic() - last > 20:
                print(f"RANGES {target.name}: {done/2**20:.0f}/{size/2**20:.0f} MiB; elapsed {time.monotonic()-start:.0f}s", flush=True)
                last = time.monotonic()
    with partial.open("wb") as result:
        for index in range(total):
            result.write((parts / f"{index:05d}").read_bytes())
    if partial.stat().st_size != size:
        raise RuntimeError("Combined file size mismatch")
    if expected_md5 and digest(partial, "md5") != expected_md5:
        raise RuntimeError(f"Publisher checksum mismatch: {target.name}")
    partial.replace(target)
    # Only remove the individually verified task-owned range files after combining.
    for index in range(total):
        (parts / f"{index:05d}").unlink()
    parts.rmdir()
    print(f"DOWNLOADED {target.name} {size}", flush=True)


def extract(archive, destination):
    marker = destination / (archive.name + ".extracted.json")
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(destination, filter="data")
    marker.write_text(json.dumps({"archive": archive.name, "sha256": digest(archive)}), encoding="utf-8")
    print(f"EXTRACTED {archive.name}", flush=True)


def acquire_slurp(audio=True):
    pinned = source_ledger("slurp")
    revision = pinned["revision"]
    hashes = {Path(f["path"].replace("\\", "/")).name: f["sha256"] for f in pinned["files"]}
    folder = RAW / "slurp"
    files = []
    for name in ["train.jsonl", "devel.jsonl", "test.jsonl", "metadata.json"]:
        url = f"https://raw.githubusercontent.com/pswietojanski/slurp/{revision}/dataset/slurp/{name}"
        target = folder / name
        download(url, target)
        sha = checked_sha256(target, hashes[name])
        files.append({"path": target.relative_to(ROOT).as_posix(), "url": url, "sha256": sha, "license": "CC-BY-4.0"})
    if audio:
        api = get_json("https://zenodo.org/api/records/4274930")
        info = next(f for f in api["files"] if f["key"] == "slurp_real.tar.gz")
        url = "https://zenodo.org/records/4274930/files/slurp_real.tar.gz?download=1"
        target = folder / "slurp_real.tar.gz"
        md5 = info["checksum"].split(":", 1)[1]
        download(url, target, md5)
        extract(target, folder)
        files.append({"path": str(target.relative_to(ROOT)), "url": url, "sha256": digest(target), "publisher_md5": md5, "bytes": target.stat().st_size, "license": "CC-BY-NC-4.0"})
    write_ledger("slurp", {"revision": revision, "files": files})


def acquire_libri():
    folder = RAW / "librispeech"
    files = []
    for pinned in source_ledger("librispeech")["files"]:
        name = Path(pinned["path"].replace("\\", "/")).name
        url = pinned["url"]
        target = folder / name
        download(url, target, pinned["publisher_md5"])
        sha = checked_sha256(target, pinned["sha256"])
        extract(target, folder)
        files.append({"path": target.relative_to(ROOT).as_posix(), "url": url, "sha256": sha, "publisher_md5": pinned["publisher_md5"], "bytes": target.stat().st_size, "license": "CC-BY-4.0"})
    write_ledger("librispeech", {"files": files})


def acquire_slurp_mirror():
    """Use a pinned audio mirror, retaining upstream recording identities."""
    import pyarrow.parquet as pq
    acquire_slurp(audio=False)
    pinned = source_ledger("slurp_audio_mirror")
    revision = pinned["revision"]
    sources = pinned["files"]
    folder = RAW / "slurp"
    audio_folder = folder / "slurp_real"
    audio_folder.mkdir(exist_ok=True)
    def fetch_extract(info):
        filename = Path(info["path"].replace("\\", "/")).name
        url = info["url"]
        target = folder / "mirror" / filename
        download(url, target)
        sha = checked_sha256(target, info["sha256"])
        expected = info.get("lfs_sha256")
        count = 0
        parquet = pq.ParquetFile(target)
        print(f"SCHEMA {target.name}: {parquet.schema_arrow}", flush=True)
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                audio = row["audio"]
                name = Path(audio["path"]).name
                if not name.startswith("audio-") or not name.endswith(".flac"):
                    raise RuntimeError(f"Cannot reconcile mirrored recording ID: {name}; keys={list(row)}")
                output = audio_folder / name
                if not output.exists():
                    output.write_bytes(audio["bytes"])
                elif hashlib.sha256(output.read_bytes()).digest() != hashlib.sha256(audio["bytes"]).digest():
                    raise RuntimeError(f"Conflicting mirrored audio: {name}")
                count += 1
        print(f"EXTRACTED MIRROR {target.name}: {count} real recordings", flush=True)
        return {"url": url, "path": target.relative_to(ROOT).as_posix(), "sha256": sha, "lfs_sha256": expected, "bytes": target.stat().st_size, "recordings": count, "license": "CC-BY-NC-4.0"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch_extract, sources))
    write_ledger("slurp_audio_mirror", {"revision": revision, "authority": "Upstream SLURP metadata", "files": files})


def write_ledger(source, data):
    folder = ROOT / "data" / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{source}.source.json").write_text(json.dumps({"source": source, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **data}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["slurp", "libri", "all", "metadata", "mirror"])
    args = parser.parse_args()
    if args.source == "all":
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(acquire_slurp), pool.submit(acquire_libri)]
            for job in jobs:
                job.result()
    elif args.source == "metadata":
        acquire_slurp(audio=False)
    elif args.source == "mirror":
        acquire_slurp_mirror()
    else:
        {"slurp": acquire_slurp, "libri": acquire_libri}[args.source]()
