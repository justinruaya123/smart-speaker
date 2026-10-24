"""Unchanged pinned 196-row command scorer, with explicit no-Pi evidence."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
COMMIT='ab39857cf73dd067d068fed388c321ab160dc191'
DATA_SHA256='9917e336ec596eaaddb476e266f36451326e3f8e5a2cb3dafbb675c33be133aa'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _binding(runtime,bench,data):
    files={'bundle.json':_sha(runtime.root/'bundle.json')}
    files.update({name:_sha(runtime.root/name) for name in runtime.meta['files']})
    if runtime.engine=='ncnn':
        native=json.loads((runtime.root/'ncnn/engine.json').read_text())
        files['ncnn/engine.json']=_sha(runtime.root/'ncnn/engine.json')
        files.update({'ncnn/'+name:_sha(runtime.root/'ncnn'/name) for name in native['files']})
    sources=('scripts/test/iris_benchmark.py','scripts/train/iris_validation.py',
             'scripts/train/iris_benchmark_serializer.py','scripts/train/iris_schema.py','vcm/frontend.py')
    return dict(schema_version=1,population=196,data_sha256=_sha(data),engine=runtime.engine,
        model_files=files,source_files={name:_sha(ROOT/name) for name in sources},
        scorer_files={p.relative_to(bench).as_posix():_sha(p) for p in
                      sorted([*bench.rglob('*.py'),bench/'vcmbench/variations.csv'])})


def _cached(folder,binding):
    result,receipt=folder/'results.json',folder/'completion.json'
    if not result.exists() and not receipt.exists():return None
    if not result.is_file() or not receipt.is_file():raise ValueError('Incomplete/unbound official scorer cache')
    completed=json.loads(receipt.read_text())
    if completed.get('binding')!=binding or completed.get('results_sha256')!=_sha(result):
        raise ValueError('Official scorer cached inputs/model/policy/source or results changed')
    value=json.loads(result.read_text())
    if any(value[k]['command_level']['n']!=196 for k in ('accepted','raw_diagnostic')):
        raise ValueError('Cached official scorer population changed')
    return value


def _complete(folder,binding,result):
    if any(result[k]['command_level']['n']!=196 for k in ('accepted','raw_diagnostic')):
        raise ValueError('Official scorer population changed')
    temporary=folder/('results.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    receipt=dict(binding=binding,results_sha256=_sha(temporary),completed=True)
    os.link(temporary,folder/'results.json');temporary.unlink()
    with (folder/'completion.json').open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def parsed_trial(clip,event):
    from scripts.train.iris_benchmark_serializer import log_record
    from vcmbench import schema
    from vcmbench.pi import LineParser
    record=log_record(event);parsed=LineParser().parse(json.dumps(record),0.)
    if parsed is None or parsed.kind!='command':raise ValueError('Invalid command serializer')
    intent,slot,known,variation=schema.resolve_prediction(parsed.intent,parsed.slot,parsed.variation,
        schema.build_alias_table(),schema.id_orders()['manifest'])
    return dict(order=clip.idx,clip_idx=clip.idx,kind='wake',input_kind='recorded_command_only_no_wake',
        wake_present=False,transcript=clip.transcript,true_intent=clip.intent,true_variation=clip.variation,
        true_slot=clip.slot_value,speaker_id=clip.speaker_id,is_synthetic=clip.is_synthetic,
        accent_group=clip.accent_group,pred_intent=intent,pred_slot=slot,pred_variation=variation,
        pred_variation_id=None,pred_raw=json.dumps(record),prediction_known=known,
        infer_ms=parsed.infer_ms,audio_ms=parsed.audio_ms,latency_s=None,play_t0=None,
        n_command_events=1,wake_logged=False)


def official_score(runtime,folder):
    from scripts.train.iris_schema import output
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    portable=ROOT/'.cache/iris-release-v1/inputs/official-benchmark'
    bench=portable if portable.exists() else ROOT/'.cache/vcm-benchmark'/COMMIT
    if not bench.exists():raise FileNotFoundError('Fetch pinned benchmark '+COMMIT)
    sys.path.insert(0,str(bench))
    from vcmbench import audio,report
    from vcmbench.dataset import load_holdout
    preferred=ROOT/'.cache/iris-release-v1/inputs/official-holdout.parquet'
    data=preferred if preferred.exists() else ROOT/'.cache/pcv-source-refresh-v1/source/data/holdout-00000-of-00001.parquet'
    if hashlib.sha256(data.read_bytes()).hexdigest()!=DATA_SHA256:raise ValueError('Official audio shard changed')
    binding=_binding(runtime,bench,data)
    cached=_cached(folder,binding)
    if cached is not None:return cached
    clips=load_holdout(str(data),folder/'unused-download-cache')
    if len(clips)!=196:raise ValueError('Official 196-row population changed')
    streams={'accepted':[],'raw_diagnostic':[]}
    for clip in clips:
        event=runtime.recognize(audio.normalize(audio.trim(clip.audio)))
        raw=(event.get('diagnostic_candidate') or {}).get('candidate')
        streams['accepted'].append(parsed_trial(clip,event))
        streams['raw_diagnostic'].append(parsed_trial(clip,{**event,'output':output(raw,None if raw else 'OOV')}))
    result={}
    for mode,trials in streams.items():
        score=report.score(trials,[],{},None,0,1,{'mode':'recorded-command-only; no wake or physical Pi'})
        result[mode]=clean(score)
        (folder/f'{mode}-trials.jsonl').write_text(''.join(json.dumps(t,allow_nan=False)+'\n' for t in trials))
    if _binding(runtime,bench,data)!=binding:raise ValueError('Official scorer inputs changed during evaluation')
    _complete(folder,binding,result)
    return result
