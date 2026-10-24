"""On-demand inputs for the existing exposed epoch44 holdout, never training data."""
from __future__ import annotations

import gzip
import hashlib
import json
from http.client import IncompleteRead
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


def _verify(path, size, sha256):
    if path.stat().st_size != size:
        raise ValueError('Benchmark input size mismatch: ' + str(path))
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
            raise ValueError('Benchmark input SHA-256 mismatch: ' + str(path))


def fetch(path, *, url, size, sha256, allow_download):
    path = Path(path)
    if path.exists():
        _verify(path, size, sha256)
        return path
    if not allow_download:
        raise FileNotFoundError('Missing cached benchmark input; run python benchmark.py without --no-setup: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(Request(url, headers={'User-Agent': 'Iris-epoch44-benchmark'}), timeout=30) as response:
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != size:
                raise ValueError('Benchmark download size mismatch')
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while block := response.read(min(65536, size-total+1)):
                    total += len(block)
                    if total > size:
                        raise ValueError('Benchmark download exceeds pinned size')
                    stream.write(block)
        _verify(temporary, size, sha256)
        os.link(temporary, path)  # Exclusive publication; never replace a cache entry.
    except IncompleteRead as error:
        raise ValueError('Benchmark download truncated') from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)  # Windows: all file handles are closed first.
    return path


REPOSITORY = 'airimonda/ai231-me2-voice-commands'
REVISION = '25111444af3adff7588d86ab27c304bb080895cf'
PARQUET_FILE = 'data/holdout-00000-of-00001.parquet'
URL = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{PARQUET_FILE}'
PARQUET_BYTES = 14553313
PARQUET_SHA256 = '9917e336ec596eaaddb476e266f36451326e3f8e5a2cb3dafbb675c33be133aa'
GOLD_SHA256 = '5eda0c581eae1fba7460ac0275cc0a206babc591507239d62d3c8c797391682a'
RECORDINGS = 196
ID_PREFIX = f'{REPOSITORY}/primary/holdout-00000-of-00001/'


def default_manifest(root, *, allow_download=False):
    root = Path(root)
    gold = root/'submission/optionb-tiny/inputs/holdout.jsonl.gz'
    _verify(gold, gold.stat().st_size, GOLD_SHA256)
    with gzip.open(gold, 'rt', encoding='utf-8') as stream:
        references = [json.loads(line) for line in stream if line.strip()]
    if len(references) != RECORDINGS or [row.get('id') for row in references] != [f'{ID_PREFIX}{i:06d}' for i in range(RECORDINGS)]:
        raise ValueError('Frozen holdout membership mismatch')
    for index, row in enumerate(references):
        expected = dict(repository=REPOSITORY, revision=REVISION, file=PARQUET_FILE,
                        bytes=PARQUET_BYTES, sha256=PARQUET_SHA256, row=index)
        if any(row.get('location', {}).get(key) != value for key, value in expected.items()):
            raise ValueError('Frozen holdout source identity mismatch')
    cache = root/'.cache/iris-epoch44-benchmark'/PARQUET_SHA256
    rows = [{**row, 'audio': f'audio/{index:06d}.wav'} for index, row in enumerate(references)]
    manifest = cache/'manifest.jsonl'
    payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n' for row in rows).encode('utf-8')
    if manifest.exists():
        _verify(manifest, len(payload), hashlib.sha256(payload).hexdigest())
    missing = []
    for index, row in enumerate(rows):
        path = cache/row['audio']
        if path.exists():
            _verify(path, path.stat().st_size, row['sha256'])
        else:
            missing.append((index, row))
    parquet = fetch(cache/'official-holdout.parquet', url=URL, size=PARQUET_BYTES,
                    sha256=PARQUET_SHA256, allow_download=allow_download)
    if missing:
        import pyarrow.parquet as pq  # Only after managed benchmark setup.
        audio = pq.read_table(parquet, columns=['audio']).column('audio')
        if len(audio) != RECORDINGS:
            raise ValueError('Frozen holdout Parquet row count mismatch')
        (cache/'audio').mkdir(exist_ok=True)
        for index, row in missing:
            wav = audio[index].as_py()['bytes']
            if not isinstance(wav, bytes) or hashlib.sha256(wav).hexdigest() != row['sha256']:
                raise ValueError('Original holdout WAV SHA-256 mismatch')
            with (cache/row['audio']).open('xb') as stream:
                stream.write(wav)  # Preserve embedded original WAV bytes, not a re-encoding.
    if not manifest.exists():
        with manifest.open('xb') as stream:
            stream.write(payload)
    return manifest
