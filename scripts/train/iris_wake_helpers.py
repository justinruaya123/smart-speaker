"""Shared wake windowing, streaming scoring and verified CPU exports."""
from pathlib import Path
import numpy as np
from scripts.train.iris_diversity import write,digest

def read_pcm(row):
    from scripts.train.iris_wake import read_pcm as reader
    return reader(row)

def window(pcm,positive,rng):
    pcm=np.asarray(pcm,dtype='float32')
    if positive and len(pcm)>25600:raise ValueError('Wake phrase too long; refusing to truncate')
    if len(pcm)>25600:
        offset=int(rng.integers(len(pcm)-25600+1));return pcm[offset:offset+25600].copy()
    result=np.zeros(25600,dtype='float32');offset=int(rng.integers(25600-len(pcm)+1))
    result[offset:offset+len(pcm)]=pcm;return result


def training_arrays(rows,norm,*,training,hard=False,reader=None):
    from vcm.frontend import features
    rng=np.random.default_rng(61);arrays=[];labels=[];excluded=[]
    for row in rows:
        pcm=(reader or read_pcm)(row);positive=row['wake_positive']
        if positive:
            active=np.flatnonzero(np.abs(pcm)>max(.002,float(np.max(np.abs(pcm)))*.02))
            if not len(active):excluded.append({'id':row['id'],'reason':'silent_positive'});continue
            pcm=pcm[max(0,active[0]-1600):min(len(pcm),active[-1]+1601)]
            if len(pcm)>25600:excluded.append({'id':row['id'],'reason':'phrase_exceeds_window'});continue
        repeats=(12 if positive else 2) if training else (3 if positive else 1)
        if hard and training and row.get('role')=='wake_negative':repeats=24
        for _ in range(repeats):
            wav=window(pcm,positive,rng)
            if training:
                wav=wav*10**(rng.uniform(-6,6)/20)
                if rng.random()<.5:
                    rms=max(float(np.sqrt(np.mean(wav**2))),.0001)
                    wav=wav+rng.normal(0,rms/10**(rng.uniform(20,35)/20),len(wav)).astype('float32')
                wav=np.clip(wav,-1,1)
            arrays.append(features(wav,norm['mean'],norm['std']));labels.append(int(positive))
    return np.stack(arrays),np.asarray(labels,dtype='int64'),excluded


def streaming_scores(rows,model,norm,*,reader=None):
    import torch
    from vcm.frontend import features
    results=[]
    for index,row in enumerate(rows,1):
        pcm=(reader or read_pcm)(row);padded=np.concatenate((np.zeros(32000,dtype='float32'),pcm,np.zeros(48000,dtype='float32')))
        windows=[features(padded[end-25600:end],norm['mean'],norm['std']) for end in range(25600,len(padded)+1,1280)]
        values=[]
        with torch.no_grad():
            for offset in range(0,len(windows),128):
                logits=model(torch.from_numpy(np.stack(windows[offset:offset+128]))).cpu().numpy().astype('float64')
                p=np.exp(logits-logits.max(1,keepdims=True));values.extend((p[:,1]/p.sum(1)).tolist())
        results.append(dict(id=row['id'],positive=row['wake_positive'],synthetic=row['synthetic'],
            speaker=row.get('speaker',row.get('speaker_id')),seconds=len(pcm)/16000,scores=values))
        if index%50==0:print('WAKE_STREAM',index,len(rows),flush=True)
    return results


def stream_metrics(records,threshold):
    from scripts.runtime.mvp_wake import RollingWake
    positives=sum(r['positive'] for r in records);negative_seconds=sum(r['seconds'] for r in records if not r['positive'])
    correct=false=duplicates=0
    for row in records:
        values=iter(row['scores']);stream=RollingWake(lambda pcm:next(values),threshold=threshold)
        stream.push(np.zeros(24320,dtype='float32'))
        hits=sum(stream.push(np.zeros(1280,dtype='float32'))==1. for _ in row['scores'])
        if row['positive']:correct+=int(hits>0);duplicates+=max(0,hits-1)
        else:false+=hits
    return dict(positive_n=positives,detected=correct,missed=positives-correct,duplicate_detections=duplicates,
        negative_n=len(records)-positives,negative_seconds=negative_seconds,false_wakes=false,
        recall=correct/positives if positives else None,false_wakes_per_hour=false*3600/negative_seconds if negative_seconds else None,
        negative_exposure='separate clips; resets and synthetic silence padding excluded from exposure; not household continuous audio')


def export_engines(model,out):
    import torch
    import onnxruntime as ort
    import ncnn
    import pnnx
    out=Path(out);model=model.cpu().eval();torch.set_num_threads(1);torch.manual_seed(61)
    torch.onnx.export(model,(torch.randn(1,40,158),),str(out/'iris.onnx'),input_names=['features'],output_names=['wake'],opset_version=17,dynamo=False)
    pnnx.export(model,str(out/'iris.pt'),inputs=torch.randn(1,40,158),fp16=False)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(out/'iris.onnx'),options,providers=['CPUExecutionProvider'])
    net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=False
    net.opt.use_fp16_packed=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
    net.opt.use_bf16_packed=False;net.opt.use_bf16_storage=False;net.opt.use_packing_layout=False
    if net.load_param(str(out/'iris.ncnn.param')) or net.load_model(str(out/'iris.ncnn.bin')):raise ValueError('Wake ncnn load failed')
    errors=[]
    for _ in range(5):
        x=torch.randn(1,40,158)
        with torch.no_grad():expected=model(x).numpy().reshape(-1)
        actual=session.run(None,{'features':x.numpy()})[0].reshape(-1)
        data=np.ascontiguousarray(x[0].numpy());mat=ncnn.Mat(data).clone()
        with net.create_extractor() as ex:
            if ex.input('in0',mat):raise ValueError('Wake ncnn input failed')
            code,value=ex.extract('out0')
            if code:raise ValueError('Wake ncnn extraction failed')
            nc=np.array(value,copy=True).reshape(-1)
        errors.append(max(float(np.abs(actual-expected).max()),float(np.abs(nc-expected).max())))
    if max(errors)>2e-5:raise ValueError('Wake engine numerical parity failed: '+str(errors))
    return dict(max_absolute_error=max(errors),files={f:digest(out/f) for f in ('iris.onnx','iris.ncnn.param','iris.ncnn.bin')})
