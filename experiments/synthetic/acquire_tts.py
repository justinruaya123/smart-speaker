"""Download immutable public Qwen TTS snapshots; never execute repository code."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/synthetic"
CACHE = ROOT / ".cache/tts/models"


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "synthetic-vcm-research/1"}), timeout=120)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    receipt = {"source": "Hugging Face", "models": {}, "download_only": True}
    for variant in ("VoiceDesign", "Base"):
        repo = "Qwen/Qwen3-TTS-12Hz-1.7B-" + variant
        with request("https://huggingface.co/api/models/" + repo + "?blobs=true") as stream:
            info = json.load(stream)
        target = CACHE / repo.split("/")[1]
        target.mkdir(parents=True, exist_ok=True)
        revision = info["sha"]
        def fetch(item):
            name = item["rfilename"]
            path = target / name
            expected = item.get("lfs", {}).get("sha256")
            if path.exists() and (not expected or sha(path) == expected):
                return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".download")
            for attempt in range(4):
                try:
                    print("DOWNLOAD", variant, name, flush=True)
                    with request(f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true") as source, temp.open("wb") as dest:
                        shutil.copyfileobj(source, dest, 4 * 1024 * 1024)
                    actual = sha(temp)
                    if expected and actual != expected:
                        raise ValueError("Downloaded hash mismatch: " + name)
                    temp.replace(path)
                    print("VERIFIED", variant, name, path.stat().st_size, flush=True)
                    return {"path": str(path.relative_to(ROOT)), "sha256": actual, "bytes": path.stat().st_size}
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2)
        files = [x for x in info["siblings"] if x["rfilename"] != ".gitattributes"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            downloaded = list(pool.map(fetch, files))
        receipt["models"][variant] = {"repository": repo, "revision": revision,
            "license": info.get("cardData", {}).get("license"), "safetensors": info.get("safetensors"), "files": downloaded}
        (OUT / "tts-acquisition.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
