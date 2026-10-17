"""Direct finite commands; no transcription, phonetic scorer or dynamic matching."""
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from scripts.runtime.pcv_schema import canonical, validate_command, output


def build_catalog(entries):
    grouped = {}
    for entry in entries:
        cmd = entry['command']
        if cmd is not None: validate_command(cmd)
        key = canonical(cmd)
        grouped.setdefault(key, {'command': cmd, 'wordings': []})['wordings'].append(entry['text'])
    grouped.setdefault('null', {'command': None, 'wordings': []})
    result = [grouped['null']] + [grouped[k] for k in sorted(grouped) if k != 'null']
    for i, row in enumerate(result):
        row['command_id'] = i
        row['wordings'] = sorted(set(row['wordings']))
    return result


def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Invalid command logits')
    p = np.exp(x-x.max()); return p/p.sum()


def decide(logits, catalog, policy):
    p = probabilities(logits)
    if len(p) != len(catalog): raise ValueError('Command inventory mismatch')
    i = int(p.argmax()); top = float(p[i]); margin = top-float(np.partition(p, -2)[-2])
    if 'by_class' in policy:
        limits=policy['by_class'].get(str(i),{'threshold':1.000001,'margin':0.})
    else:limits=policy.get('by_intent',{}).get((catalog[i]['command'] or {}).get('intent'),policy)
    accepted = (policy.get('enabled',True) and i != 0 and i in policy['trained_classes'] and top >= limits['threshold'] and margin >= limits['margin'])
    return {'command': catalog[i]['command'] if accepted else None,
            'reason': None if accepted else 'OOV',
            'class_index': i, 'class_probability': top, 'margin': margin,
            'confidence_interpretation': 'model softmax; not a correctness guarantee'}


def fit_policy(rows, trained_classes, catalog=None, risk=None, *, threshold_strategy='coarse',grouping='intent',min_correct=2):
    if threshold_strategy not in ('coarse','observed') or grouping not in ('intent','class'):
        raise ValueError('Unknown policy fitting strategy')
    if any(r['split'] not in ('dev', 'calibration') for r in rows):
        raise ValueError('Policy fitting excludes test/holdout and training')
    if any(r['expected_index'] is None for r in rows):
        raise ValueError('Ambiguous references cannot fit a rejection policy')
    if catalog is not None:
        policy=fit_policy(rows,trained_classes,risk=risk,threshold_strategy=threshold_strategy)
        field='by_intent' if grouping=='intent' else 'by_class';policy[field]={}
        groups=(sorted({c['command']['intent'] for c in catalog if c['command']}) if grouping=='intent' else
                [str(i) for i in trained_classes if i])
        for group in groups:
            subset=[r for r in rows if ((catalog[int(np.argmax(r['logits']))]['command'] or {}).get('intent')
                    if grouping=='intent' else str(int(np.argmax(r['logits']))))==group]
            selected=fit_policy(subset,trained_classes,risk=risk,threshold_strategy=threshold_strategy)
            if grouping=='class' and selected['fitting']['correct']+selected['fitting']['synthetic_correct']<min_correct:
                selected=fit_policy([],trained_classes,risk=risk,threshold_strategy=threshold_strategy)
            policy[field][group]={key:selected[key] for key in ('threshold','margin','fitting')}
        policy['method']='Predicted-'+grouping+' joint command probability/margin; explicit empirical error limits per group'
        policy['global_reference_fitting']=policy['fitting']
        policy['fitting']={k:sum(v['fitting'][k] for v in policy[field].values())
                           for k in ('correct','synthetic_correct','wrong','unknown_accepts')}
        policy['fitting'].update(human_n=sum(not r['synthetic'] for r in rows),synthetic_n=sum(r['synthetic'] for r in rows))
        return policy
    prepared = []
    for r in rows:
        p = probabilities(r['logits']); idx = int(p.argmax())
        prepared.append((r, idx, float(p[idx]), float(p[idx]-np.partition(p,-2)[-2])))
    best = None
    thresholds=(.2,.3,.4,.5,.6,.7,.8,.85,.9,.95,.98,.995,1.000001) if rows else (1.000001,)
    if rows and threshold_strategy=='observed':
        observed=np.unique([p for r,i,p,m in prepared])
        thresholds=sorted({.2,1.000001,*[float((a+b)/2) for a,b in zip(observed[:-1],observed[1:]) if b-a>=1e-5 and (a+b)/2>=.2]})
    policy_arrays={
        'index':np.array([i for r,i,p,m in prepared]),'probability':np.array([p for r,i,p,m in prepared]),
        'margin':np.array([m for r,i,p,m in prepared]),'expected':np.array([r['expected_index'] for r,i,p,m in prepared]),
        'synthetic':np.array([r['synthetic'] for r,i,p,m in prepared],dtype=bool)}
    supported=np.isin(policy_arrays['index'],trained_classes)&(policy_arrays['index']!=0)
    wrong_mask=policy_arrays['index']!=policy_arrays['expected']
    for threshold in thresholds:
        for margin in (0.,.1,.2,.35):
            accepted=supported&(policy_arrays['probability']>=threshold)&(policy_arrays['margin']>=margin)
            wrong=int(np.count_nonzero(accepted&wrong_mask))
            unknown=int(np.count_nonzero(accepted&(policy_arrays['expected']==0)))
            correct=int(np.count_nonzero(accepted&~wrong_mask&~policy_arrays['synthetic']))
            synthetic=int(np.count_nonzero(accepted&~wrong_mask&policy_arrays['synthetic']))
            if unknown:continue
            limits=risk or {'human':0.,'synthetic':0.}
            feasible=True
            for kind in ('human','synthetic'):
                group=accepted&(policy_arrays['synthetic']==(kind=='synthetic'))
                errors=int(np.count_nonzero(group&wrong_mask))
                if errors>limits[kind]*np.count_nonzero(group)+1e-10:feasible=False
            if not feasible:continue
            key = (correct, synthetic, -threshold, -margin)
            candidate = dict(threshold=threshold, margin=margin, trained_classes=sorted(trained_classes),
                             fitting=dict(correct=correct, synthetic_correct=synthetic, wrong=wrong,
                                          unknown_accepts=unknown, human_n=sum(not r['synthetic'] for r in rows),
                                          synthetic_n=sum(r['synthetic'] for r in rows)),
                             method='maximize correct human accepts, then synthetic; zero observed fitting wrong/unknown accepts')
            candidate['empirical_error_limits']=limits
            if risk:candidate['method']='maximize fitting human correct accepts; bounded known errors and zero observed unknown accepts; unqualified demo only'
            if best is None or key > best[0]: best = (key,candidate)
    return best[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Runtime:
    decision_authority='atomic_command'

    def __init__(self, folder, threads=1, engine='onnxruntime', mode='demo', profile='benchmark-compatible'):
        started=time.perf_counter()
        if mode not in ('demo','qualified'):raise ValueError('Unknown acceptance mode')
        if profile!='benchmark-compatible':raise ValueError('Owner-protected backend is not trained or qualified')
        self.mode=mode;self.profile=profile
        self.root=Path(folder); self.engine=engine
        self.meta=json.loads((self.root/'bundle.json').read_text())
        if self.meta.get('format')!='mvp-static-v1':raise ValueError('Unknown direct command bundle')
        if not {'model.onnx','catalog.json','normalization.json','policy.json'} <= self.meta['files'].keys():
            raise ValueError('Incomplete direct command bundle')
        for path, expected in self.meta['files'].items():
            resolved=(self.root/path).resolve()
            if not resolved.is_relative_to(self.root.resolve()) or digest(resolved) != expected:
                raise ValueError('Bundle identity mismatch')
        self.catalog=json.loads((self.root/'catalog.json').read_text())
        self.policy=json.loads((self.root/'policy.json').read_text())
        self.norm=json.loads((self.root/'normalization.json').read_text())
        if self.catalog[0]['command'] is not None:raise ValueError('Missing UNKNOWN class')
        for i,row in enumerate(self.catalog):
            if row['command_id']!=i:raise ValueError('Invalid class order')
            if row['command'] is not None:validate_command(row['command'])
        for limits in [self.policy,*self.policy.get('by_intent',{}).values(),*self.policy.get('by_class',{}).values()]:
            if any(type(limits[k]) not in (float,int) or not np.isfinite(limits[k]) for k in ('threshold','margin')):
                raise ValueError('Invalid acceptance threshold')
            if not 0<=limits['threshold']<=1.000001 or not 0<=limits['margin']<=1:raise ValueError('Invalid threshold range')
        if 'by_class' in self.policy and set(self.policy['by_class'])!={str(i) for i in self.policy['trained_classes'] if i}:
            raise ValueError('Incomplete or noncanonical class policy')
        if any(type(i) is not int or not 0<=i<len(self.catalog) for i in self.policy['trained_classes']):
            raise ValueError('Invalid supported class')
        if self.policy.get('qualified') is not False:raise ValueError('This MVP has not passed qualification')
        from vcm.frontend import StreamingFrontend
        StreamingFrontend(self.norm['mean'],self.norm['std'])
        self.recipe={'binding':{'model_sha256':self.meta['files']['model.onnx']}}
        if engine == 'onnxruntime':
            import onnxruntime as ort
            cfg=ort.SessionOptions();cfg.intra_op_num_threads=threads;cfg.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(self.root/'model.onnx'),cfg,providers=['CPUExecutionProvider'])
        elif engine == 'ncnn':
            import ncnn
            self.ncnn=ncnn; self.net=ncnn.Net(); self.net.opt.num_threads=threads
            self.net.opt.use_vulkan_compute=False
            receipt=json.loads((self.root/'ncnn/engine.json').read_text())
            if receipt['onnx_sha256']!=self.meta['files']['model.onnx'] or not receipt['parity']['passed']:
                raise ValueError('Unverified ncnn conversion')
            for name in ('model.ncnn.param','model.ncnn.bin','position-frequency.json'):
                if digest(self.root/'ncnn'/name)!=receipt['files'][name]:raise ValueError('ncnn identity mismatch: '+name)
            self.frequency=np.asarray(json.loads((self.root/'ncnn/position-frequency.json').read_text()),dtype='float32')
            self.net.opt.use_fp16_packed=False;self.net.opt.use_fp16_storage=False;self.net.opt.use_fp16_arithmetic=False
            self.net.opt.use_bf16_packed=False;self.net.opt.use_bf16_storage=False;self.net.opt.use_packing_layout=False
            if self.net.load_param(str(self.root/'ncnn/model.ncnn.param')) or self.net.load_model(str(self.root/'ncnn/model.ncnn.bin')):
                raise ValueError('ncnn model load failed')
        else: raise ValueError('Unknown inference engine')
        self.engine_model_sha256=digest(self.root/'ncnn/model.ncnn.bin') if engine=='ncnn' else self.meta['files']['model.onnx']
        self.load_ms=1000*(time.perf_counter()-started)

    def recognize(self, pcm, sample_rate=16000):
        from vcm.frontend import features
        started=time.perf_counter(); x=np.asarray(pcm,dtype=np.float32)
        if sample_rate != 16000 or x.ndim != 1 or not np.isfinite(x).all() or np.max(np.abs(x),initial=0)>1:
            raise ValueError('Expected finite mono 16kHz float PCM')
        event={'schema_version':1,'event':'command','engine':self.engine,'audio_seconds':len(x)/16000,
               'model_sha256':self.meta['files']['model.onnx'],'profile':self.profile,'owner_gating_enabled':False,
               'engine_model_sha256':self.engine_model_sha256,
               'policy_enabled':self.mode=='demo','qualified':False,'diagnostic_candidate':None,
               'decision_authority':self.decision_authority}
        if len(x)<400 or len(x)>240000 or np.max(np.abs(x),initial=0)<1e-6:
            decision={'command':None,'reason':'capture_fault'}
        else:
            f=features(x,self.norm['mean'],self.norm['std']); front=time.perf_counter()
            if self.engine=='onnxruntime': logits=self.session.run(None,{'features':f[None]})[0][0]
            else:
                frames=(f.shape[1]+1)//2
                angle=np.arange(frames,dtype='float32')[:,None]*self.frequency[None]
                position=np.stack((np.sin(angle),np.cos(angle)),axis=-1).reshape(frames,-1)
                buffers=[np.ascontiguousarray(f),np.ascontiguousarray(position)]
                mats=[self.ncnn.Mat(value).clone() for value in buffers]
                with self.net.create_extractor() as ex:
                    if ex.input('in0',mats[0]) or ex.input('in1',mats[1]):raise ValueError('ncnn input failed')
                    ret,result=ex.extract('out0')
                    if ret: raise ValueError('ncnn extraction failed')
                    logits=np.array(result,copy=True).reshape(-1)
            decision=decide(logits,self.catalog,self.policy)
            event['diagnostic_candidate']={'candidate':self.catalog[decision['class_index']]['command'],
                                           **decision,'logits':np.asarray(logits).reshape(-1).tolist()}
            event['frontend_ms']=1000*(front-started)
        if self.mode=='qualified':decision={'command':None,'reason':'unqualified_model'}
        event['output']=output(decision['command'],decision.get('reason'))
        event['application']={'mode':self.mode,**event['output']}
        event['qualified_output']=output(None,'unqualified_model')
        event['timing']={'total_ms':1000*(time.perf_counter()-started)}
        return event

    def make_gate(self,observed=None):
        from scripts.runtime.mvp_wake import make_gate
        return make_gate(self,observed)
