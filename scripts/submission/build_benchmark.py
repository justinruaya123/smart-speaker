"""Build the silent PC/SSH benchmark from repository code, model and WAV fixtures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.submission.common import ROOT,read,sha,write_archive,safe_name
from scripts.submission.benchmark import runtime_entries
from scripts.submission.package import GRAPH_SHA256

START_HERE='''2M recorded benchmark — PC or Raspberry Pi over SSH

python benchmark.py --target pc
python benchmark.py --target pi --host pi@raspberrypi.local

Use Python 3.12/3.13. First setup installs the pinned local runtime; the Pi needs
64-bit Raspberry Pi OS and SSH. No training/reproduction is required. This ZIP
includes all 196 original frozen held-out WAVs and needs no dataset download.
Results go to a new output directory. Add --output results/run-01 if desired.
See docs/BENCHMARK-HANDOFF.md for SSH, metrics, uncertainties and limitations.

Recognition and replay are silent: no microphone, TTS, browser or device actions.
Desktop results do not establish physical Pi or live microphone performance.
Keep source/model/recording notices. Two uncertain typed references retain their
predictions but stay outside accuracy denominators. No test tuning is performed.
'''


def audio_entries(path,spec):
    expected={row['audio']:row['sha256'] for row in spec['rows']}
    if len(expected)!=len(spec['rows']) or any(not safe_name(name) or Path(name).name!=name or not name.endswith('.wav') for name in expected):
        raise ValueError('Invalid or duplicate benchmark audio names')
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(set(names))!=len(names) or set(names)!=set(expected):raise ValueError('Benchmark audio source inventory mismatch')
        result={}
        for name,digest in expected.items():
            data=archive.read(name)
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Benchmark audio identity mismatch: '+name)
            result['.cache/submission/holdout/'+name]=data
    return result


def validate_entries(entries):
    graph='submission/models/full/model.onnx'
    if hashlib.sha256(entries[graph]).hexdigest()!=GRAPH_SHA256:raise ValueError('Selected benchmark model identity mismatch')
    spec=json.loads(entries['submission/benchmark/holdout.json'])
    expected={'.cache/submission/holdout/'+r['audio']:r['sha256'] for r in spec['rows']}
    audio={name for name in entries if name.endswith('.wav')}
    if audio!=set(expected):raise ValueError('Benchmark sample inventory mismatch')
    for name,digest in expected.items():
        if hashlib.sha256(entries[name]).hexdigest()!=digest:raise ValueError('Benchmark sample hash mismatch: '+name)
    for name in entries:
        if not safe_name(name) or name.endswith(('.pt','.pth','.ckpt','.safetensors','.pyc')) or name.startswith(('scripts/train/','data/')) or Path(name).name=='.env':
            raise ValueError('Forbidden benchmark payload: '+name)
    return {name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();manifest=json.loads(archive.read('SUBMISSION-MANIFEST.json'))
        if len(set(names))!=len(names) or set(names)!={*manifest['files'],'SUBMISSION-MANIFEST.json'}:raise ValueError('Benchmark archive inventory mismatch')
        entries={name:archive.read(name) for name in manifest['files']}
        if validate_entries(entries)!=manifest['files']:raise ValueError('Benchmark archive checksum mismatch')
        if manifest.get('purpose')!='silent-benchmark-2m' or manifest.get('model_choice')!='full' or manifest.get('actions_enabled') is not False:
            raise ValueError('Benchmark runtime contract mismatch')
        if archive.testzip() is not None:raise ValueError('Benchmark archive CRC failed')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/final-submission/smart-speaker-benchmark-2m.zip')
    parser.add_argument('--verify-only',type=Path)
    args=parser.parse_args(argv)
    if args.verify_only:
        result=verify_archive(args.verify_only);print(json.dumps({'verified_files':len(result['files']),'actions_enabled':False}));return
    source=ROOT/'submission/benchmark/holdout-audio.zip';receipt=read(source.with_suffix('.source.json'))
    if sha(source)!=receipt['sha256']:raise ValueError('Benchmark fixture source ZIP changed')
    manifest=ROOT/'submission/benchmark/holdout.json'
    if sha(manifest)!=receipt['manifest_sha256']:raise ValueError('Benchmark fixture manifest changed')
    entries=runtime_entries('full');entries.update(audio_entries(source,read(manifest)))
    entries['START-HERE.txt']=START_HERE.encode()
    entries['submission/licenses/HELDOUT-SOURCE.json']=(ROOT/'data/sources/ai231-primary.source.json').read_bytes()
    entries['submission/licenses/PROJECT-SCOPE-NOTICE.txt']=(ROOT/'docs/SUBMISSION-LICENSES.md').read_bytes()
    validate_entries(entries)
    result=write_archive(args.output,entries,purpose='silent-benchmark-2m',model_choice='full',model_sha256=GRAPH_SHA256,
                         actions_enabled=False,qualified=False,recordings=len(read(manifest)['rows']),physical_pi_measured=False)
    verify_archive(args.output)
    result['source_audio_sha256']=receipt['sha256'];result['recordings']=len(read(manifest)['rows'])
    args.output.with_suffix('.receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
