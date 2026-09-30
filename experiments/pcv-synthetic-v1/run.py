"""Explicit synthetic study stages; no physical actions or hidden qualification."""
import argparse
import json
from pathlib import Path
from study import CACHE, HERE, freeze, read_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('freeze');sub.add_parser('status');sub.add_parser('prepare')
    t=sub.add_parser('train');t.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('export');e.add_argument('--run',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('parity');e.add_argument('--run',type=Path,required=True);e.add_argument('--bundle',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('calibrate');c.add_argument('--bundle',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--parity',type=Path,required=True)
    r=sub.add_parser('report');r.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--bundle',type=Path,required=True);e.add_argument('--output',type=Path,required=True);e.add_argument('--split',choices=['dev','test'],required=True)
    a=p.parse_args()
    if a.command=='freeze':result=freeze()
    elif a.command=='status':
        result={}
        for engine in ('qwen','parler','dia'):
            path=CACHE/f'{engine}-manifest.jsonl'
            rows=read_rows(path) if path.exists() else []
            result[engine]={'generated':len(rows),'planned':13860,'remaining':13860-len(rows)}
        result['trained_candidate']=(CACHE/'training/selection.json').exists()
        result['final_test_complete']=(CACHE/'final-test/results.json').exists()
    elif a.command=='prepare':
        from fitting import prepare
        result=prepare()
    elif a.command=='train':
        from fitting import train
        result=train(a.output)
    elif a.command=='export':
        from finishing import export
        result=export(a.run,a.output)
    elif a.command=='calibrate':
        from finishing import calibrate
        result=calibrate(a.bundle,a.output,a.parity)
    elif a.command=='report':
        from reporting import report
        result=report(a.output)
    elif a.command=='parity':
        from finishing import waveform_parity
        result=waveform_parity(a.run,a.bundle,a.output)
    else:
        from synthbench import evaluate
        result=evaluate(a.bundle,a.split,a.output)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
