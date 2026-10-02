"""Separate mutable continuation outputs from frozen predecessor receipts."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train import pcv_dia_cpu_codec as adapter
from scripts.train.pcv_three_engine import HERE,OUTPUT,immutable
from scripts.train.pcv_public_data import sha
from scripts.test.pcv_public_qualification import verify_bindings


def main():
    work=OUTPUT/'dia-cpu-codec-v1'
    protocol=HERE/'dia-cpu-output-protocol.json'
    sources=[Path(__file__),adapter.PROTOCOL,OUTPUT/'dia-generated.jsonl',OUTPUT/'dia-single-generated.jsonl']
    if not protocol.exists():
        work.mkdir(parents=True,exist_ok=True)
        for name in ('dia-generated.jsonl','dia-single-generated.jsonl'):
            # Metadata references only; shared recordings are never copied.
            with (work/name).open('xb') as stream:stream.write((OUTPUT/name).read_bytes())
        immutable(protocol,{'purpose':'Continue in a separate cache so bound predecessor attempt logs/manifests stay immutable',
            'cache':str(work.relative_to(ROOT)),'changes_generation_math':False,
            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in sources}})
    frozen=json.loads(protocol.read_text());verify_bindings(ROOT,frozen['bindings'])
    previous=adapter.parent.OUTPUT
    adapter.parent.OUTPUT=work
    try:adapter.generate()
    finally:adapter.parent.OUTPUT=previous
    verify_bindings(ROOT,frozen['bindings'])
    immutable(HERE/'dia-cpu-output-completion.json',{'execution_protocol_sha256':sha(protocol),
        'generation_complete_sha256':sha(HERE/'dia-generation-complete.json')})


if __name__=='__main__':main()
