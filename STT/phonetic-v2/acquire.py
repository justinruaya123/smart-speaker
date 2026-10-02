"""Fetch checksum-verified public LibriSpeech; extract original files safely."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / '.cache/phonetic-v2/downloads'
URL = 'https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz'
SIZE = 6387309499
MD5 = '2a93770f6d5c6c964bc36631d331a522'


def acquire():
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / 'train-clean-100.tar.gz'
    start = time.monotonic()
    if not target.exists() or target.stat().st_size != SIZE:
        chunk = 32 * 1024 * 1024
        jobs = list(range((SIZE + chunk - 1) // chunk))

        def fetch(index):
            begin = index * chunk
            end = min(SIZE, begin + chunk) - 1
            path = RAW / f'part-{index:04d}'
            if path.exists() and path.stat().st_size == end - begin + 1:
                return
            for attempt in range(4):
                try:
                    request = urllib.request.Request(URL, headers={'Range': f'bytes={begin}-{end}', 'User-Agent': 'speech-research/2'})
                    with urllib.request.urlopen(request, timeout=60) as stream:
                        if stream.status != 206 or stream.headers.get('Content-Range', '').split('/')[0] != f'bytes {begin}-{end}':
                            raise ValueError('Mirror did not honor the requested byte range')
                        with path.open('wb') as out:
                            remaining = end - begin + 1
                            while remaining:
                                block = stream.read(min(1024 * 1024, remaining))
                                if not block:
                                    raise IOError('Truncated download')
                                out.write(block)
                                remaining -= len(block)
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)

        with ThreadPoolExecutor(max_workers=6) as pool:
            for count, _ in enumerate(pool.map(fetch, jobs), 1):
                if count % 10 == 0:
                    print('Verified-size download chunks', count, '/', len(jobs), flush=True)
        with target.open('wb') as out:
            for index in jobs:
                path = RAW / f'part-{index:04d}'
                with path.open('rb') as stream:
                    while block := stream.read(8 * 1024 * 1024):
                        out.write(block)
    with target.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'md5').hexdigest()
    if actual != MD5 or target.stat().st_size != SIZE:
        raise ValueError('LibriSpeech publisher checksum/size mismatch')
    with target.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    directory = ROOT / 'data/audio/librispeech'
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(target) as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != 'LibriSpeech' or '..' in parts or any(':' in p for p in parts):
                raise ValueError('Unsafe archive member')
            path = directory.joinpath(*parts)
            payload = archive.extractfile(member).read()
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError('Existing original recording differs: ' + str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            count += 1
    receipt = dict(url=URL, publisher='https://www.openslr.org/12/', checksum_source='https://www.openslr.org/resources/12/md5sum.txt',
                   bytes=SIZE, md5=actual, sha256=sha, files=count, license='CC-BY-4.0', seconds=time.monotonic()-start,
                   pretrained_weights=False, synthetic_audio=False)
    (ROOT / 'STT/phonetic-v2/acquisition.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    acquire()
