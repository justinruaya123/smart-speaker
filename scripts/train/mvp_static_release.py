"""Export, development calibration and locked replay for the finite MVP."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from scripts.models.mvp_static import StaticCommandNet, ExportStatic
from scripts.runtime.mvp_static import Runtime, decide, digest, fit_policy
from scripts.runtime.pcv_schema import output
from scripts.train.mvp_static_train import ROOT, HERE, CACHE, PREP, prepare, read_rows, write


def export_onnx(model, path):
    import onnx
    import onnxruntime as ort
    wrapper=ExportStatic(model.cpu().eval()).eval()
    generator=torch.Generator().manual_seed(53)
    sample=torch.randn(1,40,37,generator=generator)
    torch.onnx.export(wrapper,(sample,),str(path),input_names=['features'],output_names=['commands'],
                      dynamic_axes={'features':{2:'frames'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    errors=[];frames=[1,3,37,128,231]
    for n in frames:
        x=torch.randn(1,40,n,generator=generator)
        with torch.no_grad(): expected=wrapper(x).numpy()
        actual=session.run(None,{'features':x.numpy()})[0]
        if actual.shape != expected.shape: raise ValueError('Export output shape changed')
        errors.append(float(np.abs(actual-expected).max()))
    receipt=dict(passed=max(errors)<2e-5,max_absolute_error=max(errors),frames=frames,
                 tolerance=2e-5,bytes=Path(path).stat().st_size,sha256=digest(path))
    if not receipt['passed']: raise ValueError('ONNX numerical parity failed: '+str(receipt))
    return receipt


class NCNNStatic(torch.nn.Module):
    """The same batch-one endpoint encoder; ncnn supplies explicit positions."""
    def __init__(self,model):
        super().__init__();self.model=model;self.attention=torch.nn.ModuleList()
        for block in model.blocks:
            width=block.qkv.in_features
            attention=torch.nn.MultiheadAttention(width,block.heads,batch_first=True,dropout=0.)
            attention.in_proj_weight.data.copy_(block.qkv.weight)
            attention.in_proj_bias.data.copy_(block.qkv.bias)
            attention.out_proj.load_state_dict(block.attention_output.state_dict())
            self.attention.append(attention)

    def forward(self,features,position):
        x=self.model.stem(features).transpose(1,2)+position
        for block,attention in zip(self.model.blocks,self.attention):
            normalized=block.attention_norm(x)
            attended,_=attention(normalized,normalized,normalized,need_weights=False)
            x=x+attended
            x=x+block.local(block.local_norm(x).transpose(1,2)).transpose(1,2)
            x=x+block.ff(block.ff_norm(x))
        return self.model.classifier(self.model.norm(x).mean(1))


def export_ncnn(model, onnx, folder):
    import ncnn
    import pnnx
    folder=Path(folder);folder.mkdir(exist_ok=False)
    model=model.cpu().eval();wrapper=NCNNStatic(model).eval()
    generator=torch.Generator().manual_seed(53)
    def inputs(n):
        f=torch.randn(1,40,n,generator=generator)
        angle=torch.arange((n+1)//2,dtype=torch.float32)[:,None]*model.position_frequency[None]
        position=torch.stack((angle.sin(),angle.cos()),dim=-1).flatten(-2)[None]
        return f,position
    rewrite=[]
    for n in (1,37,128,231):
        f,pos=inputs(n)
        with torch.no_grad():rewrite.append(float((wrapper(f,pos)-model(f,torch.tensor([n]))).abs().max()))
    if max(rewrite)>2e-5:raise ValueError('ncnn endpoint rewrite changed logits')
    torch.backends.mha.set_fastpath_enabled(False)
    pnnx.export(wrapper,str(folder/'model.pt'),inputs=inputs(128),inputs2=inputs(231),fp16=False)
    write(folder/'position-frequency.json',model.position_frequency.tolist())
    net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=False
    net.opt.use_fp16_packed=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
    net.opt.use_bf16_packed=False;net.opt.use_bf16_storage=False;net.opt.use_packing_layout=False
    if net.load_param(str(folder/'model.ncnn.param')) or net.load_model(str(folder/'model.ncnn.bin')):
        raise ValueError('ncnn export did not load')
    errors=[]
    for n in (1,3,37,128,231,511):
        f,pos=inputs(n)
        with torch.no_grad():expected=model(f,torch.tensor([n])).numpy().reshape(-1)
        buffers=[np.ascontiguousarray(f[0].numpy()),np.ascontiguousarray(pos[0].numpy())]
        mats=[ncnn.Mat(x).clone() for x in buffers]
        with net.create_extractor() as ex:
            if ex.input('in0',mats[0]) or ex.input('in1',mats[1]):raise ValueError('ncnn input failed')
            code,value=ex.extract('out0')
            if code:raise ValueError('ncnn extraction failed')
            actual=np.array(value,copy=True).reshape(-1)
        errors.append(float(np.abs(actual-expected).max()))
    receipt=dict(onnx_sha256=digest(onnx),rewrite_errors=rewrite,
        files={f:digest(folder/f) for f in ('model.ncnn.param','model.ncnn.bin','position-frequency.json')},
        parity=dict(passed=max(errors)<2e-5,max_absolute_error=max(errors),errors=errors,tolerance=2e-5))
    write(folder/'engine.json',receipt)
    if not receipt['parity']['passed']:raise ValueError('ncnn numerical parity failed: '+str(receipt))
    return receipt


def count_metrics(records):
    positive=[r for r in records if r['expected_index'] not in (None,0)]
    negative=[r for r in records if r['expected_index']==0]
    eligible=positive+negative
    accepted=[r for r in eligible if r['accepted_index'] not in (None,0)]
    correct=sum(r['accepted_index']==r['expected_index'] for r in positive)
    return dict(n=len(records),positive_n=len(positive),negative_n=len(negative),
        ambiguous_n=len(records)-len(eligible),raw_correct=sum(r['raw_index']==r['expected_index'] for r in positive),
        correct_accepts=correct,accepted=len(accepted),wrong_accepted_positives=sum(
            r['accepted_index'] not in (None,0,r['expected_index']) for r in positive),
        unknown_accepts=sum(r['accepted_index'] not in (None,0) for r in negative),
        false_rejects=sum(r['accepted_index'] in (None,0) for r in positive),
        recall=correct/len(positive) if positive else None,
        precision=correct/len(accepted) if accepted else None)


def summarize(records, catalog):
    result={'overall':count_metrics(records),'strata':{},'per_intent':{},'per_value':{}}
    for kind in ('human','synthetic'):
        subset=[r for r in records if r['synthetic']==(kind=='synthetic')]
        result['strata'][kind]=count_metrics(subset)
        for intent in sorted({c['command']['intent'] for c in catalog if c['command']}):
            rows=[r for r in subset if r['expected_index'] not in (None,0) and catalog[r['expected_index']]['command']['intent']==intent]
            result['per_intent'].setdefault(intent,{})[kind]=count_metrics(rows)
    groups=defaultdict(list)
    for r in records:
        if r['expected_index'] in (None,0):continue
        cmd=catalog[r['expected_index']]['command']
        for role,value in cmd['slots'].items():groups[(cmd['intent'],role,str(value),r['synthetic'])].append(r)
    result['per_value']={'/'.join(map(str,key)):count_metrics(rows) for key,rows in sorted(groups.items())}
    for field in ('source','accent_group','speaker'):
        groups=defaultdict(list)
        for r in records:groups[str(r.get(field,'unknown'))].append(r)
        result[field]={key:count_metrics(rows) for key,rows in groups.items()}
    return result


def run_folder(name, run_id):return CACHE/'runs'/run_id/name


def infer_rows(runtime, rows, stream=None):
    import soundfile as sf
    lookup={json.dumps(c['command'],sort_keys=True):c['command_id'] for c in runtime.catalog}
    results=[]
    for i,row in enumerate(rows,1):
        audio=ROOT/row['audio']
        if digest(audio)!=row['sha256']:raise ValueError('Audio identity mismatch: '+row['id'])
        pcm,rate=sf.read(audio,dtype='float32')
        event=runtime.recognize(pcm,rate)
        diag=event.get('diagnostic_candidate') or {}
        record={key:row.get(key) for key in ('id','speaker','split','synthetic','source','accent_group')}
        record.update(expected_index=row['class_index'],raw_index=diag.get('class_index'),
            accepted_index=lookup.get(json.dumps(event['output']['command'],sort_keys=True)) if event['output']['accepted'] else None,
            logits=diag.get('logits'),capture_fault=event['output'].get('reason')=='capture_fault',event=event)
        results.append(record)
        if stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        if i%500==0:print('REPLAY',row['split'],i,len(rows),flush=True)
    return results


def calibrate(name,run_id,export_id='export'):
    if not export_id.replace('-','').replace('_','').isalnum():raise ValueError('Invalid export ID')
    cfg=prepare();folder=run_folder(name,run_id);target=folder/export_id;target.mkdir(exist_ok=False)
    selection=json.loads((folder/'selection.json').read_text())
    checkpoint=folder/'best.pt'
    if digest(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('Selected weights changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=StaticCommandNet(**state['config']);model.load_state_dict(state['weights']);model.eval()
    export=export_onnx(model,target/'model.onnx')
    catalog=json.loads((HERE/'catalog.json').read_text())
    trained=[c['command_id'] for c in catalog if c['training_recordings']]
    norm=json.loads((PREP/'normalization.json').read_text())
    write(target/'catalog.json',catalog);write(target/'normalization.json',norm)
    # Provisional reject policy is confined to this calibration directory.
    policy=dict(threshold=1.000001,margin=0.,trained_classes=trained,enabled=True,qualified=False)
    write(target/'policy.json',policy)
    def metadata():return dict(format='mvp-static-v1',parameters=selection['parameters'],
        files={f:digest(target/f) for f in ('model.onnx','catalog.json','normalization.json','policy.json')},
        inference_mode='complete_utterance',profile='benchmark-compatible',qualified=False,
        checkpoint_sha256=selection['checkpoint_sha256'])
    write(target/'bundle.json',metadata())
    runtime=Runtime(target)
    rows=[r for split in ('dev','calibration') for r in read_rows(CACHE/'views'/f'{split}.jsonl')]
    with (target/'development-predictions.jsonl').open('x') as stream:records=infer_rows(runtime,rows,stream)
    fitting=[r for r in records if r['expected_index'] is not None and r['logits'] is not None]
    policy=fit_policy(fitting,trained,catalog=catalog,risk=cfg.get('demo_risk'))
    policy.update(enabled=True,qualified=False,qualification_reason='Original-human coverage and physical/live gates remain incomplete')
    # This is the only mutable construction stage; no test has been read.
    (target/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    (target/'bundle.json').write_text(json.dumps(metadata(),indent=2)+'\n')
    for record in records:
        if record['logits'] is None:continue
        decision=decide(record['logits'],catalog,policy)
        record['accepted_index']=decision['class_index'] if decision['command'] else None
    write(folder/'development-results.json',dict(export=export,policy=policy,
        splits={s:summarize([r for r in records if r['split']==s],catalog) for s in ('dev','calibration')}))
    write(folder/'selection-before-test.json',dict(bundle_path=export_id,protocol_sha256=digest(HERE/'protocol.json'),
        catalog_sha256=digest(target/'catalog.json'),bundle_sha256=digest(target/'bundle.json'),
        policy_sha256=digest(target/'policy.json'),checkpoint_sha256=digest(checkpoint),
        model_sha256=digest(target/'model.onnx'),test_used_for_selection=False,physical_pi_measured=False))
    print('CALIBRATED',name,json.dumps({s:count_metrics([r for r in records if r['split']==s and not r['synthetic']]) for s in ('dev','calibration')}),flush=True)


def verify_selection(folder):
    lock=json.loads((folder/'selection-before-test.json').read_text());target=folder/lock['bundle_path']
    if not target.resolve().is_relative_to(folder.resolve()):raise ValueError('Invalid selected bundle path')
    for key,path in (('protocol_sha256',HERE/'protocol.json'),('bundle_sha256',target/'bundle.json'),
                     ('policy_sha256',target/'policy.json'),('model_sha256',target/'model.onnx'),
                     ('catalog_sha256',target/'catalog.json'),('checkpoint_sha256',folder/'best.pt')):
        if digest(path)!=lock[key]:raise ValueError('Pre-test selection identity mismatch: '+key)
    engine_lock=folder/'engine-selection-before-test.json'
    if engine_lock.exists():
        bound=json.loads(engine_lock.read_text())
        if bound['source_lock_sha256']!=digest(folder/'selection-before-test.json'):raise ValueError('Changed engine source selection')
        for name,expected in bound['files'].items():
            path=(target/name).resolve()
            if not path.is_relative_to(target.resolve()) or digest(path)!=expected:raise ValueError('Pre-test engine identity mismatch: '+name)
    return lock


def engine_export(name,run_id):
    prepare();folder=run_folder(name,run_id);lock=verify_selection(folder);target=folder/lock['bundle_path']
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model=StaticCommandNet(**state['config']);model.load_state_dict(state['weights']);model.eval()
    receipt=export_ncnn(model,target/'model.onnx',target/'ncnn')
    write(folder/'engine-selection-before-test.json',dict(source_lock_sha256=digest(folder/'selection-before-test.json'),
        files={str(p.relative_to(target)):digest(p) for p in (target/'ncnn').iterdir() if p.name in
               ('model.ncnn.param','model.ncnn.bin','position-frequency.json','engine.json')},parity=receipt['parity']))
    print('NCNN_VERIFIED',name,json.dumps(receipt['parity']),flush=True)


def evaluate(name,run_id,engine='onnxruntime',cfg=None):
    if cfg is None:prepare()
    folder=run_folder(name,run_id);lock=verify_selection(folder);runtime=Runtime(folder/lock['bundle_path'],engine=engine)
    result={}
    for split in ('test','holdout'):
        with (folder/f'{split}-predictions.jsonl').open('x') as stream:
            records=infer_rows(runtime,read_rows(CACHE/'views'/f'{split}.jsonl'),stream)
        result[split]=summarize(records,runtime.catalog)
        print(name,split,json.dumps(result[split]['strata']),flush=True)
    result.update(lock=lock,qualification_passed=False,installed=False,physical_pi_measured=False)
    write(folder/'results.json',result)


def official(name,run_id,engine='onnxruntime',cfg=None):
    if cfg is None:cfg=prepare()
    folder=run_folder(name,run_id);lock=verify_selection(folder)
    bench=ROOT/'.cache/vcm-benchmark'/cfg['benchmark_commit'];sys.path.insert(0,str(bench))
    spec=importlib.util.spec_from_file_location('official_replay',ROOT/'experiments/speaker-official-vcm-benchmark-v1/run.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    if digest(previous.DATA)!=previous.DATA_SHA256:raise ValueError('Fixed official audio shard changed')
    from scripts.runtime import pcv_benchmark_adapter as adapter
    from vcmbench import audio,report
    runtime=Runtime(folder/lock['bundle_path'],engine=engine);clips=previous.read_clips()
    if len(clips)!=196:raise ValueError('Fixed benchmark population changed')
    streams={'accepted':[],'raw_diagnostic':[]}
    for clip in clips:
        event=runtime.recognize(audio.normalize(audio.trim(clip.audio)))
        for mode in streams:
            raw=(event.get('diagnostic_candidate') or {}).get('candidate')
            selected=event if mode=='accepted' else {**event,'output':output(raw,None if raw else 'OOV')}
            streams[mode].append(previous.parsed_trial(clip,selected,adapter))
    result={}
    for mode,trials in streams.items():
        scored=report.score(trials,[],{},None,0,1,{'mode':'recorded-command-only; no wake or physical Pi'})
        with (folder/f'official-{mode}-trials.jsonl').open('x') as stream:
            stream.write(''.join(json.dumps(t,allow_nan=False)+'\n' for t in trials))
        result[mode]=previous.clean(scored)
        print(name,mode,json.dumps({k:result[mode]['command_level'][k] for k in ('n','accuracy','false_accepts','n_out_of_scope')}),flush=True)
    write(folder/'official-results.json',result)


def personal(name,run_id,engine='ncnn',cfg=None):
    if cfg is None:prepare()
    folder=run_folder(name,run_id);lock=verify_selection(folder)
    runtime=Runtime(folder/lock['bundle_path'],engine=engine)
    with (folder/'personal-test-predictions.jsonl').open('x') as stream:
        records=infer_rows(runtime,read_rows(CACHE/'views/personal-test.jsonl'),stream)
    result=summarize(records,runtime.catalog);write(folder/'personal-test-results.json',result)
    print('PERSONAL',name,json.dumps(result['overall']),flush=True)


def main():
    global HERE,CACHE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['calibrate','engine','evaluate','official','personal'])
    parser.add_argument('--model',choices=['compact','full'],required=True)
    parser.add_argument('--run-id',default='complete-20261017')
    parser.add_argument('--export-id',default='export',help='Unique construction attempt; failed exports remain intact')
    parser.add_argument('--engine',choices=['ncnn','onnxruntime'],default='onnxruntime')
    parser.add_argument('--study',choices=['vcm-mvp-hey-iris-v1','vcm-mvp-hey-iris-v2'],default='vcm-mvp-hey-iris-v1')
    args=parser.parse_args();torch.set_num_threads(1)
    from scripts.train import mvp_static_train as training
    training.select_study(args.study);HERE,CACHE=training.HERE,training.CACHE
    if args.stage=='calibrate':calibrate(args.model,args.run_id,args.export_id)
    elif args.stage=='engine':engine_export(args.model,args.run_id)
    else:{'evaluate':evaluate,'official':official,'personal':personal}[args.stage](args.model,args.run_id,args.engine)


if __name__=='__main__':main()
