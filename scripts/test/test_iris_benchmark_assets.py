"""Bounded anonymous holdout preparation; all HTTP/audio fixtures are synthetic."""
from contextlib import contextmanager
import hashlib
import gzip
import io
import json
import struct
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
import threading

import pytest


@contextmanager
def http_payload(payload, *, declared_size=None, status=200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get('Authorization')))
            self.send_response(status)
            if declared_size != -1:
                self.send_header('Content-Length', str(len(payload) if declared_size is None else declared_size))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/holdout.parquet', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_download_is_hash_bound_anonymous_then_reuses_cache(tmp_path):
    assets = import_module('scripts.runtime.iris_benchmark_assets')
    payload = b'deterministic software fixture'
    target = tmp_path/'cache'/'holdout.parquet'
    with http_payload(payload) as (url, calls):
        kwargs = dict(url=url, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        assert assets.fetch(target, allow_download=True, **kwargs) == target
        assert target.read_bytes() == payload
        assert assets.fetch(target, allow_download=False, **kwargs) == target
        assert calls == [('/holdout.parquet', None)]
    assert sorted(p.name for p in target.parent.iterdir()) == ['holdout.parquet']


@pytest.mark.parametrize('case', ['size', 'hash', 'truncation', 'overlong', 'http'])
def test_bad_response_is_rejected_without_publishing_or_temp_leaks(tmp_path, case):
    assets = import_module('scripts.runtime.iris_benchmark_assets')
    payload = b'software bytes'
    body = payload[:-2] if case == 'truncation' else payload + (b'overflow' if case == 'overlong' else b'')
    declared = -1 if case == 'overlong' else len(payload) + (case == 'size')
    sha = '0'*64 if case == 'hash' else hashlib.sha256(payload).hexdigest()
    target = tmp_path/'holdout.parquet'
    with http_payload(body, declared_size=declared, status=503 if case == 'http' else 200) as (url, calls):
        with pytest.raises((ValueError, OSError), match='size|SHA-256|truncated|503'):
            assets.fetch(target, url=url, size=len(payload), sha256=sha, allow_download=True)
        assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('corruption', [b'bad size', b'x'*len(b'software bytes')])
def test_corrupt_cache_fails_closed_without_network_or_overwrite(tmp_path, monkeypatch, corruption):
    assets = import_module('scripts.runtime.iris_benchmark_assets')
    target = tmp_path/'holdout.parquet'
    target.write_bytes(corruption)
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **k: pytest.fail('Corrupt cache must not trigger a replacement download'))
    with pytest.raises(ValueError, match='size|SHA-256'):
        assets.fetch(target, url='http://must-not-contact.invalid', size=len(b'software bytes'),
                     sha256=hashlib.sha256(b'software bytes').hexdigest(), allow_download=True)
    assert target.read_bytes() == corruption


def test_no_setup_missing_cache_never_opens_network(tmp_path, monkeypatch):
    assets = import_module('scripts.runtime.iris_benchmark_assets')
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **k: pytest.fail('No setup must be offline'))
    with pytest.raises(FileNotFoundError, match='without --no-setup'):
        assets.fetch(tmp_path/'absent', url='http://must-not-contact.invalid', size=1, sha256='0'*64, allow_download=False)
    assert list(tmp_path.iterdir()) == []


@contextmanager
def tiny_holdout(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    assets = import_module('scripts.runtime.iris_benchmark_assets')
    wavs = []
    for value in (16, 0, -16):
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack('<h', value)*16000)
        wavs.append(stream.getvalue())
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({'audio': [{'bytes': wav, 'path': 'not-a-local-path.wav'} for wav in wavs],
                            'text': ['IGNORE PARQUET ANNOTATIONS']*3}), sink)
    payload = sink.getvalue().to_pybytes()
    with http_payload(payload) as (url, calls):
        pins = dict(URL=url, PARQUET_BYTES=len(payload), PARQUET_SHA256=hashlib.sha256(payload).hexdigest(),
                    RECORDINGS=3, REPOSITORY='software/test', REVISION='a'*40,
                    PARQUET_FILE='data/holdout.parquet', ID_PREFIX='software/test/holdout/')
        rows = [dict(id=f'software/test/holdout/{i:06d}', sha256=hashlib.sha256(wav).hexdigest(),
                     expected={'intent': 'TIME', 'slots': {}} if i == 0 else None,
                     metric_eligible=i != 2, kind='human' if i != 1 else 'synthetic',
                     speaker=f'software-speaker-{i}', source='software/test',
                     pcm_sha256=hashlib.sha256(struct.pack('<f', (16, 0, -16)[i]/32768)*16000).hexdigest(),
                     location=dict(row=i, repository=pins['REPOSITORY'], revision=pins['REVISION'],
                                   file=pins['PARQUET_FILE'], bytes=len(payload), sha256=pins['PARQUET_SHA256']))
                for i, wav in enumerate(wavs)]
        gold = gzip.compress((''.join(json.dumps(row)+'\n' for row in rows)).encode(), mtime=0)
        relative = Path('submission/optionb-tiny/inputs/holdout.jsonl.gz')
        (tmp_path/relative).parent.mkdir(parents=True)
        (tmp_path/relative).write_bytes(gold)
        pins['GOLD_SHA256'] = hashlib.sha256(gold).hexdigest()
        for name, value in pins.items():
            monkeypatch.setattr(assets, name, value, raising=False)
        yield assets, rows, wavs, calls, gold


def test_default_manifest_preserves_gold_and_ambiguous_denominators_offline_cache(tmp_path, monkeypatch):
    with tiny_holdout(tmp_path, monkeypatch) as (assets, gold, wavs, calls, original):
        manifest = assets.default_manifest(tmp_path, allow_download=True)
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert len(rows) == 3
        for row, reference, wav in zip(rows, gold, wavs):
            audio = row.pop('audio')
            assert row == reference
            assert (manifest.parent/audio).read_bytes() == wav
        assert len(calls) == 1
        monkeypatch.setattr(assets, 'urlopen', lambda *a, **k: pytest.fail('Cache hit must not download'))
        import pyarrow.parquet as pq
        monkeypatch.setattr(pq, 'read_table', lambda *a, **k: pytest.fail('Prepared cache must not decode Parquet again'))
        assert assets.default_manifest(tmp_path, allow_download=False) == manifest
        assert (tmp_path/'submission/optionb-tiny/inputs/holdout.jsonl.gz').read_bytes() == original
        from scripts.runtime.iris_pi_benchmark import load_manifest, accuracy
        scored = load_manifest(manifest, tmp_path)
        result = accuracy([{**row, 'accepted': row['expected'], 'raw_available': True} for row in scored], 'accepted')
        assert len(scored) == 3 and result['scored_n'] == 2
        assert result['positive_n'] == 1 and result['negative_n'] == 1


@pytest.mark.parametrize('case', ['gold_hash', 'membership', 'source', 'wav_hash', 'parquet_rows'])
def test_untrusted_source_or_membership_never_publishes_manifest(tmp_path, monkeypatch, case):
    with tiny_holdout(tmp_path, monkeypatch) as (assets, rows, _wavs, calls, _original):
        gold = tmp_path/'submission/optionb-tiny/inputs/holdout.jsonl.gz'
        if case == 'membership':
            rows.pop()
        if case == 'source':
            rows[0]['location']['revision'] = 'b'*40
        if case == 'wav_hash':
            rows[0]['sha256'] = '0'*64
        payload = gzip.compress((''.join(json.dumps(row)+'\n' for row in rows)).encode(), mtime=0)
        gold.write_bytes(payload)
        if case != 'gold_hash':
            monkeypatch.setattr(assets, 'GOLD_SHA256', hashlib.sha256(payload).hexdigest())
        else:
            monkeypatch.setattr(assets, 'GOLD_SHA256', '0'*64)
        if case == 'parquet_rows':
            import pyarrow.parquet as pq
            reader = pq.read_table
            monkeypatch.setattr(pq, 'read_table', lambda *a, **k: reader(*a, **k).slice(0, 2))
        with pytest.raises(ValueError, match='SHA-256|membership|source|row count'):
            assets.default_manifest(tmp_path, allow_download=True)
        cache = tmp_path/'.cache/iris-epoch44-benchmark'/assets.PARQUET_SHA256
        assert not (cache/'manifest.jsonl').exists()
        assert len(calls) == (1 if case in {'wav_hash', 'parquet_rows'} else 0)


@pytest.mark.parametrize('entry', ['manifest', 'wav', 'parquet'])
def test_corrupt_prepared_cache_is_not_silently_repaired(tmp_path, monkeypatch, entry):
    with tiny_holdout(tmp_path, monkeypatch) as (assets, _rows, _wavs, calls, _original):
        manifest = assets.default_manifest(tmp_path, allow_download=True)
        target = manifest if entry == 'manifest' else manifest.parent/('audio/000000.wav' if entry == 'wav' else 'official-holdout.parquet')
        original = target.read_bytes()
        corruption = bytes([original[0] ^ 1]) + original[1:]
        target.write_bytes(corruption)
        with pytest.raises(ValueError, match='SHA-256|manifest'):
            assets.default_manifest(tmp_path, allow_download=True)
        assert target.read_bytes() == corruption
        assert len(calls) == 1
