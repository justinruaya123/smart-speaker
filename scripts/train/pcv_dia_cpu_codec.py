"""Bound execution adapter: keep Dia FP32/CUDA, move its DAC codec to CPU."""
from contextlib import contextmanager
import gc
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train import pcv_dia_single as parent
from scripts.train.pcv_three_engine import HERE,OUTPUT,immutable,verify,engine_loader
from scripts.train.pcv_public_data import sha,read_rows
from scripts.test.pcv_public_qualification import verify_bindings

PROTOCOL=HERE/'dia-cpu-codec-protocol.json'


def freeze():
    verify()
    prior=read_rows(OUTPUT/'dia-generated.jsonl')+read_rows(OUTPUT/'dia-single-generated.jsonl')
    for row in prior:
        verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['native_audio']:row['native_sha256'],row['features']:row['features_sha256']})
    ids={r['phrase_id'] for r in prior}
    for kind,ext in [('audio','.wav'),('native','.wav'),('features','.npy')]:
        if {p.stem for p in (OUTPUT/kind/'dia').glob('*'+ext)}!=ids:raise ValueError('Orphan Dia output')
    snapshot=HERE/'dia-before-cpu-codec.json'
    immutable(snapshot,prior)
    paths=[Path(__file__),Path(parent.__file__),HERE/'protocol.json',HERE/'dia-single-protocol.json',
        snapshot,OUTPUT/'dia-single-run.log',OUTPUT/'dia-single-failed-attempts.jsonl',
        ROOT/'.venv/Lib/site-packages/transformers/models/dia/processing_dia.py']
    immutable(PROTOCOL,{'version':'dia-cpu-codec-v1','retained_cuda_codec_recordings':len(prior),
        'reason':'FP32 single-item CUDA waveform decoding again reached 15669/16376 MiB and had no completion for over five minutes',
        'interrupted_owned_process_ids':[22120,17236],'other_job_interrupted':False,
        'change':'Dia acoustic generator remains FP32 on CUDA; DAC voice encoding and waveform decoding move to FP32 CPU',
        'codec_equivalence_claim':False,'note':'CPU/GPU rounding can change anchor codes and generated audio; this is an explicit generation revision',
        'unchanged':['All phrases and labels','Voice WAV','All model/codec weights','Single-item seed rule','Token limit 1600','Temperature 1.8','Twenty attempts','Waveform and duration/CTC gates'],
        'preflight':'Two development voice phrases pass waveform checks, with at most three attempts each; no VCM inference',
        'test_inference_accessed':False,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths}})


@contextmanager
def cpu_codec_loader(engine):
    if engine!='dia':raise ValueError('Adapter is Dia-only')
    import torch
    with engine_loader(engine) as (original,captured,module):
        nonlocal_values=inspect.getclosurevars(original).nonlocals
        codec=nonlocal_values['processor'].audio_tokenizer
        codec.to('cpu')
        model=nonlocal_values['model']
        assert next(codec.parameters()).device.type=='cpu'
        assert next(model.parameters()).device.type=='cuda'
        assert next(codec.parameters()).dtype==next(model.parameters()).dtype==torch.float32
        gc.collect();torch.cuda.empty_cache()
        def generate(texts,voice):
            waves,rate=original(texts,voice)
            generate.last_metadata=[{**m,'codec_device':'cpu','acoustic_model_device':'cuda',
                'generation_revision_sha256':sha(PROTOCOL)} for m in original.last_metadata]
            return waves,rate
        yield generate,captured,module


def check():
    verify(assets=True)
    p=json.loads(PROTOCOL.read_text());verify_bindings(ROOT,p['bindings'])
    import torch
    torch.set_num_threads(2)
    rows=[]
    with cpu_codec_loader('dia') as (generator,_,__):
        from synthesis import VOICES,prepare_audio
        voice=next(v for v in VOICES if v['id']=='dev-au-m')
        for i,text in enumerate(['The system is ready.','Please set the brightness to sixty percent.']):
            for attempt in range(3):
                torch.manual_seed(91237+i*10+attempt)
                began=time.perf_counter()
                try:
                    waves,rate=generator([text],voice)
                    value=prepare_audio(waves[0],rate)
                    rows.append({'text':text,'voice':voice['id'],'attempt':attempt+1,'seconds':time.perf_counter()-began,
                        'native_metrics':value[2],'derived_metrics':value[3],'generation_metadata':generator.last_metadata[0]})
                    print('CPU_CODEC_DEVELOPMENT_PROBE',i+1,rows[-1]['seconds'],flush=True)
                    break
                except ValueError as exc:
                    print('CPU_CODEC_DEVELOPMENT_RETRY',i+1,attempt+1,str(exc),flush=True)
                    if attempt==2:raise
            gc.collect();torch.cuda.empty_cache()
        immutable(HERE/'dia-cpu-codec-preflight.json',{'pass':len(rows)==2,'rows':rows,
            'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved(),
            'protocol_sha256':sha(PROTOCOL),'text_listener_verified':False,'VCM_inference':False})


def generate():
    p=json.loads(PROTOCOL.read_text());verify_bindings(ROOT,p['bindings'])
    pilot=json.loads((HERE/'dia-cpu-codec-preflight.json').read_text())
    if not pilot['pass'] or pilot['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Missing matching development preflight')
    old_row=parent.generated_row
    old_immutable=parent.immutable
    def revision_row(engine,prompt,value,metadata):
        return old_row(engine,prompt,value,{**metadata,'parent_generation_protocol_sha256':metadata['protocol_sha256'],
            'protocol_sha256':sha(PROTOCOL),'source_version':'dia-cpu-codec-v1'})
    def revision_receipt(path,value):
        if Path(path)==HERE/'dia-generation-complete.json':
            value={**value,'single_batch_protocol_sha256':value['generation_revision_sha256'],
                'generation_revision_sha256':sha(PROTOCOL),'retained_cuda_codec_recordings':p['retained_cuda_codec_recordings'],
                'codec_device_for_remaining':'cpu','codec_dtype':'float32','preflight_sha256':sha(HERE/'dia-cpu-codec-preflight.json')}
        return old_immutable(path,value)
    parent.engine_loader=cpu_codec_loader
    parent.generated_row=revision_row
    parent.immutable=revision_receipt
    try:parent.generate()
    finally:
        parent.engine_loader=engine_loader;parent.generated_row=old_row;parent.immutable=old_immutable
    verify_bindings(ROOT,p['bindings'])


if __name__=='__main__':
    if sys.argv[1:] not in (['freeze'],['check'],['generate']):raise SystemExit('Use freeze, check or generate')
    globals()[sys.argv[1]]()
