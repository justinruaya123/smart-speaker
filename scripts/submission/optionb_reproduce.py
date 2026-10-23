"""Source-only reproduction of the frozen epoch44 OptionB candidate, never installation."""
from pathlib import Path
from collections import defaultdict
import hashlib
import io
import gzip
import json
import re

from scripts.submission.common import ROOT, fetch, read, safe_name, save, sha

SNAPSHOT = 'submission/optionb-tiny/snapshot.json'


def pinned_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not safe_name(relative):
        raise ValueError('Unsafe snapshot path: ' + str(relative))
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Snapshot path leaves checkout: ' + relative)
    return path


def verify_snapshot(root=ROOT, snapshot_path=SNAPSHOT):
    snapshot = read(pinned_path(root, snapshot_path))
    for relative, record in snapshot['files'].items():
        path = pinned_path(root, relative)
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Snapshot identity changed: ' + relative)
    for relative, expected in snapshot['source_bindings'].items():
        path = pinned_path(root, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Snapshot source identity changed: ' + relative)
    return snapshot


def _source_location(location, source_shards=None):
    """Resolve and check the complete source identity, never normalize a locator."""
    keys = ('repository', 'revision', 'file', 'sha256', 'url', 'bytes')
    pinned = ('repository', 'revision', 'file')
    if 'bytes' in location and (type(location['bytes']) is not int or location['bytes'] <= 0):
        raise ValueError('Invalid source byte count')
    if source_shards is not None:
        inventory = [_source_location(source) for source in source_shards]
        if (len({source['sha256'] for source in inventory}) != len(inventory)
                or len({tuple(source.get(key) for key in pinned) for source in inventory}) != len(inventory)):
            raise ValueError('Conflicting source inventory identities')
        matches = [source for source in inventory
                   if all(source.get(key) == location.get(key) for key in (*pinned, 'sha256'))]
        if len(matches) != 1:
            raise ValueError('Conflicting source locator/inventory identity')
        source = matches[0]
        if any(location[key] != source.get(key) for key in keys if key in location):
            raise ValueError('Conflicting source locator/inventory identity')
        return source

    source = {key: location[key] for key in keys if key in location}
    if (not isinstance(source.get('sha256'), str)
            or re.fullmatch('[a-f0-9]{64}', source['sha256']) is None):
        raise ValueError('Invalid source digest')
    if not isinstance(source.get('url'), str) or not source['url']:
        raise ValueError('Invalid source URL')
    if any(key in source for key in pinned):
        repository, revision, file = (source.get(key) for key in pinned)
        if (not isinstance(repository, str)
                or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repository) is None
                or not isinstance(revision, str) or re.fullmatch('[a-f0-9]{40}', revision) is None
                or not isinstance(file, str) or not safe_name(file)):
            raise ValueError('Invalid pinned source locator')
        url = f'https://huggingface.co/datasets/{repository}/resolve/{revision}/{file}'
        if source['url'] not in (url, url + '?download=true'):
            raise ValueError('Conflicting source URL/locator identity')
    return source


def _writable_path(path, directory):
    """Check the full target against its lexical boundary, including existing links."""
    path, directory = Path(path), Path(directory)
    if not path.resolve().is_relative_to(directory):
        raise ValueError('Output path leaves selected cache directory: ' + str(path))
    return path


def hydrate_rows(root, rows, downloads, audio_directory, auxiliary_source, *, source_shards=None):
    """Read pinned source bytes, verify WAV/PCM, retain exact ordered references."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest

    root = Path(root).resolve()
    pinned_path(root, downloads)
    pinned_path(root, audio_directory)
    downloads, audio_directory = root / downloads, root / audio_directory
    _writable_path(downloads, downloads)
    _writable_path(audio_directory, audio_directory).mkdir(parents=True, exist_ok=True)
    if len(rows) != len({row['id'] for row in rows}):
        raise ValueError('Duplicate source recording identity')
    groups, sources, auxiliary, prepared = defaultdict(list), {}, [], {}

    def admit(row, payload):
        if hashlib.sha256(payload).hexdigest() != row['sha256']:
            raise ValueError('Source WAV identity changed: ' + row['id'])
        pcm, rate = sf.read(io.BytesIO(payload), dtype='float32', always_2d=False)
        if pcm_digest(pcm, rate) != row['pcm_sha256']:
            raise ValueError('Source PCM identity changed: ' + row['id'])
        path = _writable_path(audio_directory / (row['sha256'] + '.wav'), audio_directory)
        if path.exists():
            if sha(path) != row['sha256']:
                raise ValueError('Existing source WAV identity changed: ' + row['id'])
        else:
            with path.open('xb') as stream:
                stream.write(payload)
        prepared[row['id']] = {**row, 'audio': path.relative_to(root).as_posix()}

    for row in rows:
        if any(not isinstance(row[key], str) or re.fullmatch('[a-f0-9]{64}', row[key]) is None
               for key in ('sha256', 'pcm_sha256')):
            raise ValueError('Invalid frozen recording digest')
        location = row['location']
        if 'auxiliary_file' in location:
            auxiliary.append(row)
        else:
            digest = location['sha256']
            if not isinstance(digest, str) or re.fullmatch('[a-f0-9]{64}', digest) is None:
                raise ValueError('Invalid frozen source digest')
            if type(location['row']) is not int or location['row'] < 0:
                raise ValueError('Invalid source row index')
            source = _source_location(location, source_shards)
            if digest in sources and sources[digest] != source:
                raise ValueError('Conflicting source locations')
            sources[digest] = source
            groups[digest].append(row)
    for digest, members in groups.items():
        target = _writable_path(downloads / (digest + '.parquet'), downloads)
        path = fetch(sources[digest]['url'], target, digest)
        wanted = defaultdict(list)
        for row in members:
            wanted[row['location']['row']].append(row)
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['audio']):
            for index, cell in enumerate(batch.column('audio').to_pylist()):
                for row in wanted.get(offset + index, []):
                    admit(row, cell['bytes'])
            offset += len(batch)
        print('Reconstructed source recordings:', len(prepared), '/', len(rows), flush=True)
    for row in auxiliary:
        name = row['location']['auxiliary_file']
        if not safe_name(name):
            raise ValueError('Unsafe auxiliary source path')
        url = (f"https://huggingface.co/datasets/{auxiliary_source['repository']}/resolve/"
               f"{auxiliary_source['revision']}/{auxiliary_source['prefix']}/{name}")
        target = _writable_path(downloads / 'auxiliary' / name, downloads)
        path = fetch(url, target, row['sha256'])
        admit(row, path.read_bytes())
    if set(prepared) != {row['id'] for row in rows}:
        raise ValueError('Missing frozen source recordings')
    return [prepared[row['id']] for row in rows]


def read_snapshot_rows(root, record):
    path = pinned_path(root, record['path'])
    if sha(path) != record['sha256']:
        raise ValueError('Frozen metadata identity changed: ' + record['path'])
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != record['rows'] or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Frozen metadata row count/schema changed')
    return rows


def prepare_source(root, output, splits=('train', 'dev', 'calibration')):
    from scripts.test.speaker_demo_suite import write_new

    root = Path(root).resolve()
    snapshot = verify_snapshot(root)
    pinned_path(root, output)
    destination = root / output
    _writable_path(destination, destination)
    parts = destination.relative_to(root).parts
    if len(parts) < 2 or parts[0] != '.cache':
        raise ValueError('Reproduction outputs must select a run under .cache/')
    if not splits or len(set(splits)) != len(splits) or set(splits) - snapshot['views'].keys():
        raise ValueError('Invalid reproduction split set')
    receipt_path = _writable_path(destination / ('prepare-' + '-'.join(splits) + '.json'), destination)
    targets = [_writable_path(destination / role / (split + '.jsonl'), destination)
               for role in ('prepared', 'canonical') for split in splits]
    if receipt_path.exists() or any(path.exists() for path in targets):
        raise FileExistsError('Preserve earlier prepared views; choose a new run or new splits')
    frozen = {split: read_snapshot_rows(root, snapshot['views'][split]) for split in splits}
    for split, rows in frozen.items():
        if any(row['split'] != split for row in rows):
            raise ValueError('Frozen manifest split role changed')
    audio = _writable_path(destination / 'audio', destination).relative_to(root).as_posix()
    downloads = (destination.parent / 'downloads').relative_to(root).as_posix()
    hydrated = hydrate_rows(root, [row for split in splits for row in frozen[split]],
                            downloads, audio, snapshot['auxiliary_source'],
                            source_shards=snapshot.get('source_shards'))
    by_id = {row['id']: row for row in hydrated}
    bindings = {}
    for split in splits:
        # Re-read the independently frozen compressed view, not a mutable prepared
        # output, when emitting the canonical comparison template.
        expected = [{**row, 'audio': f"{audio}/{row['sha256']}.wav"}
                    for row in read_snapshot_rows(root, snapshot['views'][split])]
        actual = [by_id[row['id']] for row in frozen[split]]
        if actual != expected:
            raise ValueError('Prepared view changes frozen membership/reference order')
        for role, values in (('canonical', expected), ('prepared', actual)):
            path = _writable_path(destination / role / (split + '.jsonl'), destination)
            write_new(path, values, jsonl=True)
            bindings[path.relative_to(root).as_posix()] = sha(path)
    verify_snapshot(root)
    receipt = dict(format='optionb-source-preparation-v1', counts={s: len(frozen[s]) for s in splits},
                   snapshot_sha256=sha(pinned_path(root, SNAPSHOT)), bindings=bindings,
                   wav_and_pcm_verified=True, old_audio_or_features_copied=False,
                   source_download_cache=downloads, derived_feature_cache_reused=False,
                   heldout_pcm_decoded=bool(set(splits) & {'test', 'holdout'}),
                   original_reference_order_and_labels_preserved=True, actions_enabled=False)
    save(_writable_path(receipt_path, destination), receipt)
    return receipt


def main(argv=None, *, root=ROOT):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('verify', 'prepare'))
    parser.add_argument('--output', default='.cache/optionb-source-repro-v1/run-01')
    parser.add_argument('--splits', nargs='+', choices=('train', 'dev', 'calibration', 'test', 'holdout'),
                        default=('train', 'dev', 'calibration'))
    args = parser.parse_args(argv)
    if args.operation == 'verify':
        snapshot = verify_snapshot(root)
        result = dict(tracked_snapshot_files_verified=len(snapshot['files']),
                      frozen_source_files_verified=len(snapshot['source_bindings']),
                      optimizer_updates=0, installed=False, actions_enabled=False)
    else:
        result = prepare_source(root, args.output, tuple(args.splits))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
