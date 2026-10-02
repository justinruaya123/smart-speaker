"""Human-only brightness evaluation for the installed reader architecture."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import time
import numpy as np
from .brightness_data import prepare, load_pcm, write_json, sha
from .parameter_backend import ParameterBackend, parse_brightness
from .pipeline import Pipeline


def value(result):
    if result.get('status')!='accepted':return None
    command=result.get('command',{})
    return command.get('arguments',{}).get('percent') if command.get('action')=='SET_BRIGHTNESS' else None


def summary(records,field='predicted'):
    positives=[r for r in records if r['reference'] is not None]
    accepted=[r for r in records if r[field] is not None]
    correct=sum(r[field]==r['reference'] for r in accepted)
    tp=sum(r[field]==r['reference'] for r in positives)
    return {'n':len(records),'positive_n':len(positives),'correct_values':tp,'emitted':len(accepted),
        'accepted_correct':correct,'exact_value_recall':tp/len(positives) if positives else None,
        'accepted_precision':correct/len(accepted) if accepted else None,
        'false_accepts_on_nonabsolute':sum(r[field] is not None and r['reference'] is None for r in records),
        'abstentions':len(records)-len(accepted)}


def calibrate(records):
    choices=[0.,1.1]+sorted({r['confidence'] for r in records if r['raw_value'] is not None})
    selected=1.1;best=-1;details=None
    for threshold in choices:
        accepted=[r for r in records if r['raw_value'] is not None and r['confidence']>=threshold]
        correct=sum(r['raw_value']==r['reference'] for r in accepted)
        if len(accepted)>=20 and correct/len(accepted)>=.95 and correct>best:
            selected=threshold;best=correct;details={'accepted':len(accepted),'correct':correct,'precision':correct/len(accepted)}
    return {'threshold':selected,'calibrated':True,'selection_split':'dev','target_precision':.95,
            'minimum_emissions':20,'development_gate_passed':details is not None,'development':details,
            'no_test_labels_used':True,'scope':'pooled brightness parameter precision; natural strata reported separately'}


def benchmark(root,config,candidate=None):
    root=Path(root);out=root/'.cache/stt-human';out.mkdir(parents=True,exist_ok=True)
    bundle=Path(candidate or out/'candidate');rows=prepare(root)
    metadata=json.loads((bundle/'bundle.json').read_text())
    fingerprint=sha(root/'data/stt-human/brightness.jsonl')
    if metadata['stt']['training']['manifest_sha256']!=fingerprint:
        raise ValueError('Candidate was not trained on this human-only manifest')
    reader=ParameterBackend(bundle,'stt',config['validation']['threads']);reader.load()
    audio_cache={}
    def decode(row):
        pcm=load_pcm(row,root,audio_cache);started=time.perf_counter()
        recognized=reader.recognize(pcm);parsed=parse_brightness(recognized['text'],metadata['catalog'])
        return {'id':row['id'],'kind':row['kind'],'reference':row['value'],'source':row['source'],
            'raw_value':value(parsed),'text':recognized['text'],'confidence':recognized['confidence'],
            'milliseconds':(time.perf_counter()-started)*1000,'duration_seconds':len(pcm)/16000}
    development=[decode(r) for r in rows if r['split']=='dev' and r['kind']!='word']
    policy=calibrate(development)
    metadata['brightness_parameters']['stt']=policy;write_json(bundle/'bundle.json',metadata)
    write_json(out/'selection_locked_before_test.json',{'policies':{'stt':policy},
        'model_sha256':{'stt':sha(bundle/'stt.fp32.onnx')},'manifest_sha256':fingerprint})
    cfg=copy.deepcopy(config);cfg['runtime']={'vad':False,'vcm':True};cfg['parameters']={'enabled':True,'backend':'stt'}
    pipeline=Pipeline(bundle,cfg);records=[]
    for row in rows:
        if row['split']!='test' or row['kind']=='word':continue
        r=decode(row);r['predicted']=r['raw_value'] if r['confidence']>=policy['threshold'] else None
        r['pipeline_value']=value(pipeline.utterance(load_pcm(row,root,audio_cache)));records.append(r)
    model={'sha256':sha(bundle/'stt.fp32.onnx'),'graph_bytes':(bundle/'stt.fp32.onnx').stat().st_size,
           'raw':summary(records,'raw_value'),'calibrated':summary(records),'pipeline':summary(records,'pipeline_value')}
    report={'manifest_sha256':fingerprint,'policies':{'stt':policy},'models':{'stt':model},
            'original_human_audio_only':True,'test_used_for_selection':False,'live_microphone_measured':False,'physical_pi_measured':False}
    write_json(out/'benchmark.json',report);write_json(out/'development_predictions.json',development);write_json(out/'test_predictions.json',records)
    print(json.dumps(report),flush=True);return report
