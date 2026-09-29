from pathlib import Path
import os
import json
from .text import normalize_text
ROOT = Path(os.environ.get("SMART_SPEAKER_ROOT", str(Path.cwd()))).resolve()

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
