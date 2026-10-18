"""Portable reconstruction and original selected-model fitting in a new folder."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ROOT, bootstrap, fetch, read, run_id, save, sha


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,help='Original seed is 53; prompted with that default')
    p.add_argument('--model-choice',choices=('full','compact'),default='full',help='Shipping 2M model by default; compact is comparison research')
    p.add_argument('--workspace',type=Path,help='New separate reproduction folder; never overwritten')
    p.add_argument('--prepare-only',action='store_true',help='Download, verify and extract features without fitting')
    p.add_argument('--verify',action='store_true',help='Verify submitted recipe and model identities without fitting')
    p.add_argument('--source-cache',type=Path,help='Optional existing research checkout for hash-verified read-only input reuse')
    p.add_argument('--gpu',type=int,help='Explicit CUDA device; otherwise use CUDA_VISIBLE_DEVICES/default device')
    p.add_argument('--no-benchmark',action='store_true');p.add_argument('--yes',action='store_true')
    p.add_argument('--no-setup',action='store_true',help='Use the current interpreter instead of uv setup')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return p


def source_paths():
    paths=set()
    for directory in ('scripts','vcm','experiments/phrase-contract-v1/pcvcm'):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts)
    for name in ('reproduce.py','inference.py','benchmark.py','pyproject.toml','uv.lock','LICENSE',
                 'experiments/pcv-synthetic-v2/wake_gate.py','experiments/pcv-synthetic-v2/lexicon.json'):
        if (ROOT/name).is_file():paths.add(name)
    for p in (ROOT/'submission').rglob('*'):
        if p.is_file():paths.add(p.relative_to(ROOT).as_posix())
    return sorted(paths)


def copy_source(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    paths=source_paths(); hashes={}
    for name in paths:
        source=ROOT/name;target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);hashes[name]=sha(target)
    save(destination/'source-receipt.json',dict(files=hashes,
        source_git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True).stdout.strip(),
        source_is_dirty=bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True).stdout),
        installed_artifacts_touched=False))
    return hashes


def load_recipe_rows(recipe, directory):
    path=Path(directory)/recipe['rows_file']
    if sha(path)!=recipe['rows_sha256']:raise ValueError('Portable row-map checksum mismatch')
    with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream if line.strip()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate portable sample identity')
    return rows


def reconstruct(args, recipe):
    """Retrieve original WAV bytes and verify PCM, split and frontend identities."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from threadpoolctl import threadpool_limits
    from scripts.test.speaker_demo_suite import pcm_digest
    from vcm.frontend import features
    recipe_dir=ROOT/('submission/reproduction-full' if args.model_choice=='full' else 'submission/reproduction')
    rows=load_recipe_rows(recipe,recipe_dir)
    by_id={r['id']:r for r in rows}
    for a,b in (('train','dev'),('train','calibration'),('train','test'),('train','holdout')):
        first={by_id[i]['speaker'] for i in recipe['views'][a]}
        second={by_id[i]['speaker'] for i in recipe['views'][b]}
        if first & second:raise ValueError('Declared speaker overlap: '+a+'/'+b)
        if {by_id[i]['pcm_sha256'] for i in recipe['views'][a]} & {by_id[i]['pcm_sha256'] for i in recipe['views'][b]}:
            raise ValueError('PCM overlap: '+a+'/'+b)
    target=ROOT/'.cache/reproduction/prepared';target.mkdir(parents=True,exist_ok=False)
    audio_dir=target/'audio';audio_dir.mkdir();feature_dir=target/'features';feature_dir.mkdir()
    groups=defaultdict(list);auxiliary=[]
    for row in rows:
        if 'auxiliary_file' in row['location']:auxiliary.append(row)
        else:groups[row['location']['sha256']].append(row)
    prepared={}; feature_mismatches=[]; sources=[]

    def extract(item):
        row,payload=item
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Audio payload mismatch: '+row['id'])
        audio=audio_dir/(row['sha256']+'.wav')
        # Multiple views may refer to one immutable recording.
        if not audio.exists():audio.write_bytes(payload)
        pcm,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=False)
        if pcm_digest(pcm,rate)!=row['pcm_sha256']:raise ValueError('Decoded PCM identity mismatch: '+row['id'])
        valid=400<=len(pcm)<=240000 and float(np.max(np.abs(pcm),initial=0))>=1e-6
        result={**row,'audio':str(audio),'command':row['expected'],'speaker_id':row['speaker'],'capture_fault':not valid}
        if valid:
            values=features(pcm);path=feature_dir/(row['pcm_sha256']+'.npy')
            with path.open('wb') as stream:np.save(stream,values,allow_pickle=False)
            actual=sha(path)
            if row.get('features_sha256') and actual!=row['features_sha256']:
                feature_mismatches.append(row['id'])
            result.update(features=str(path),features_sha256=actual)
        return row['id'],result

    with threadpool_limits(limits=64,user_api='blas'), ThreadPoolExecutor(max_workers=8) as pool:
        for records in groups.values():
            loc=records[0]['location'];path=None
            if args.source_cache:
                prefix=('.cache/pcv-source-refresh-v1/source' if loc['repository'].startswith('airimonda/')
                        else '.cache/fil-supplemental-v1/source')
                candidate=args.source_cache/prefix/loc['file']
                if candidate.is_file():
                    if sha(candidate)!=loc['sha256']:raise ValueError('Existing source cache checksum mismatch')
                    path=candidate
            if path is None:path=fetch(loc['url'],ROOT/'.cache/reproduction/downloads'/(loc['sha256']+'.parquet'),loc['sha256'])
            sources.append(dict(repository=loc['repository'],revision=loc['revision'],file=loc['file'],sha256=loc['sha256'],
                                input_reused=bool(args.source_cache and path.is_relative_to(args.source_cache))))
            wanted={r['location']['row']:r for r in records};offset=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['audio']):
                tasks=[(wanted[offset+i],v['bytes']) for i,v in enumerate(batch.column('audio').to_pylist()) if offset+i in wanted]
                for key,result in pool.map(extract,tasks):prepared[key]=result
                offset+=len(batch)
            print('RECONSTRUCTED',len(prepared),'/',len(rows),loc['file'],flush=True)
        auxiliary_source=read(recipe_dir/'auxiliary-source.json')
        local_aux={}
        if args.source_cache:
            for path in (args.source_cache/'data/audio/speaker-synthetic-completion-v1').glob('dev-*/*.wav'):
                local_aux[sha(path)]=path
        tasks=[]
        for row in auxiliary:
            path=local_aux.get(row['sha256'])
            if path is None:
                path=fetch(f"https://huggingface.co/datasets/{auxiliary_source['repository']}/resolve/{auxiliary_source['revision']}/{auxiliary_source['prefix']}/{row['location']['auxiliary_file']}",
                           ROOT/'.cache/reproduction/downloads'/row['location']['auxiliary_file'],row['sha256'])
            tasks.append((row,path.read_bytes()))
        for key,result in pool.map(extract,tasks):prepared[key]=result
    if len(prepared)!=len(rows):raise ValueError('Missing reconstructed source rows')
    for name,ids in recipe['views'].items():
        with (target/(name+'.jsonl')).open('x',encoding='utf-8') as stream:
            for identity in ids:stream.write(json.dumps(prepared[identity],separators=(',',':'))+'\n')
    if feature_mismatches:
        raise ValueError(f'{len(feature_mismatches)} original feature hashes differ; preserve this run and inspect frontend/environment before fitting.')
    norm=read(ROOT/recipe['normalization']['path']);statistics=[]
    for identity in recipe['views']['normalization']:
        values=np.load(prepared[identity]['features'],allow_pickle=False).astype('float64')
        statistics.append((values.mean(axis=1),(values**2).mean(axis=1)))
    mean=np.mean([x[0] for x in statistics],axis=0)
    std=np.sqrt(np.maximum(np.mean([x[1] for x in statistics],axis=0)-mean**2,1e-6))
    if not np.allclose(mean,norm['mean'],atol=1e-12,rtol=0) or not np.allclose(std,norm['std'],atol=1e-12,rtol=0):
        raise ValueError('Human-only normalization reconstruction differs')
    save(target/'audit.json',dict(rows=len(rows),counts={k:len(v) for k,v in recipe['views'].items()},
         source_shards=sources,feature_hash_mismatches=feature_mismatches,original_pcm_hashes_verified=True,
         normalization_reconstructed=True,speaker_ids_pseudonymized=True,fitting_frontend_blas_threads=64,
         source_cache=str(args.source_cache) if args.source_cache else None))
    return prepared,target


def fit(args,recipe,prepared):
    import numpy as np
    import torch
    from scripts.train import speaker_final_train as training
    from scripts.pcv_parallel import ParallelScorer
    from scripts.submission.recipe import jsonl
    norm=read(ROOT/recipe['normalization']['path']);entries=read(ROOT/recipe['catalog']['path'])['entries']
    train=[prepared[i] for i in recipe['views']['train']];dev=[prepared[i] for i in recipe['views']['dev']]
    for r in train:
        if r['split']!='train' or not r['training_eligible']:raise ValueError('Non-fitting row entered training')
    device=training.configure_training_device()
    print('Training device:',device,flush=True)
    training.HERE=ROOT/'results';training.CACHE=ROOT/'.cache/reproduction';training.ARCHITECTURE=recipe['architecture']
    cfg={**recipe,'device':device,'training':{**recipe['training'],'seed':args.seed}}
    save(training.HERE/'protocol.json',dict(cfg,requirements=['ME-01','ME-02','ME-03','ME-04','ME-06','ME-07','ME-08'],
        hypothesis='The frozen selected recipe reconstructs the selected numerical behavior from pinned recordings.',
        source_receipt_sha256=sha(ROOT/'source-receipt.json'), test_used_for_selection=False))
    # Preserve every development checkpoint while reusing the original fit loop.
    class CheckpointRecorder:
        def __getattr__(self,name):return getattr(torch,name)
        def save(self,payload,path,*a,**kw):
            torch.save(payload,path,*a,**kw)
            if Path(path).name=='last.tmp.pt' and payload['epoch']%cfg['training']['development_every']==0:
                torch.save(payload,Path(path).parent/f"reproduction-epoch-{payload['epoch']:03d}.pt")
    training.torch=CheckpointRecorder()
    arrays=training.load_arrays(train+dev,norm);pools=training.TrainingPools(train);began=time.perf_counter()
    with ParallelScorer(entries,workers=cfg['training']['decoder_workers'],return_scores=True) as scorer:
        foundation=training.fit_stage('human','shared',None,pools,dev,arrays,entries,scorer,cfg)
        mixed=training.fit_stage('mixed','control',ROOT/foundation['checkpoint'],pools,dev,arrays,entries,scorer,cfg)
    history=read(training.CACHE/'training/mixed-control/history.json')
    def recognition_key(row):
        h=row['development']['human'];s=row['development']['synthetic']
        return (h['raw_macro_speaker_recall'],h['raw_exact'],-h['raw_negative_errors'],s.get('raw_recall') or 0.,-row['epoch'])
    chosen=(next(r for r in history if r['epoch']==mixed['epoch']) if args.model_choice=='full'
            else max((r for r in history if 'development' in r),key=recognition_key))
    path=training.CACHE/'training/mixed-control'/f"reproduction-epoch-{chosen['epoch']:03d}.pt"
    selection=dict(foundation=foundation,mixed_acceptance_selection=mixed,recognition_epoch=chosen['epoch'],
                   checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),
                   seed=args.seed,seconds=time.perf_counter()-began,device=device,
                   gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                   torch=torch.__version__,cuda=torch.version.cuda,test_used_for_selection=False)
    save(ROOT/'results/selection-before-calibration.json',selection)
    return path,selection


def evaluation_arrays(prepared,recipe,norm,cache):
    """Match historical one-thread evaluation without changing frozen fit inputs."""
    import numpy as np
    import soundfile as sf
    from threadpoolctl import threadpool_limits
    from vcm.frontend import features
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=False)
    external={i for view in ('calibration','test','holdout') for i in recipe['views'][view]}
    wanted=external|set(recipe['views']['dev']);arrays={};files={}
    mean=np.asarray(norm['mean'],dtype='float32')[:,None]
    std=np.asarray(norm['std'],dtype='float32')[:,None]
    # BLAS limits are process-global: use one outer context, with no concurrent
    # extraction. INT8 decisions near a gate can amplify tiny frontend rounding.
    with threadpool_limits(limits=1,user_api='blas'):
        for identity,row in prepared.items():
            if identity not in wanted or row['capture_fault']:continue
            if identity in external:
                pcm,rate=sf.read(row['audio'],dtype='float32',always_2d=False)
                if rate!=16000:raise ValueError('Evaluation requires original 16 kHz PCM')
                raw=features(pcm);path=cache/(row['pcm_sha256']+'.npy')
                if not path.exists():
                    with path.open('xb') as stream:np.save(stream,raw,allow_pickle=False)
                files[identity]=dict(path=path.name,sha256=sha(path))
            else:
                raw=np.load(row['features'],allow_pickle=False)
            arrays[identity]=(raw-mean)/std
    save(cache/'manifest.json',dict(format='submission-evaluation-frontend-v1',blas_threads=1,
        fitting_features_unchanged=True,files=files))
    return arrays


def export_and_test(args,recipe,prepared,checkpoint,selection):
    import numpy as np
    import torch
    from scripts.models.pcv_release_pooling import PooledContextVCM
    from scripts.models.pcv_contextual import export_onnx
    from scripts.train import speaker_final_train as training
    from scripts.train.speaker_final_release import ONNXModel,calibrate_human,quantization_gate
    from scripts.train.speaker_compact_policy import fit_policy
    from scripts.runtime.pcv_final_runtime import BINDINGS,verify_bundle
    from scripts.runtime.speaker_compact import decorate_event
    from scripts.runtime.speaker_confidence import application_decision
    from scripts.pcv_parallel import ParallelScorer
    from scripts.runtime.pcv_pi_benchmark import metrics
    torch.set_num_threads(1)
    bundle=ROOT/'results/model';bundle.mkdir()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=PooledContextVCM(**recipe['architecture']);model.load_state_dict(payload['weights'])
    full=args.model_choice=='full'
    if full:
        from scripts.train.pcv_context_int8 import export_quantized
        exported=export_quantized(model,ROOT/'results/export')
        shutil.copyfile(ROOT/'results/export/model.int8.onnx',bundle/'model.onnx')
    else:
        exported=export_onnx(model,bundle/'model.onnx')
        if exported['max_absolute_error']>=2e-5 or exported['onnx_bytes']>=4*1024*1024:raise ValueError('FP32 export parity/size gate failed')
    for name in ('catalog.json','custom-slots.json','normalization.json','output.schema.json'):
        shutil.copyfile(ROOT/f'submission/models/{args.model_choice}'/name,bundle/name)
    binding={k:sha(bundle/v) for k,v in BINDINGS.items()}
    norm=read(bundle/'normalization.json');entries=read(bundle/'catalog.json')['entries']
    arrays=evaluation_arrays(prepared,recipe,norm,ROOT/'.cache/reproduction/evaluation-features-blas1')
    engine=ONNXModel(bundle/'model.onnx')
    def score(ids,scorer):
        rows=[prepared[i] for i in ids if prepared[i]['metric_eligible']]
        valid=[r for r in rows if not r['capture_fault']]
        scored=training.score_model(engine,valid,arrays,entries,scorer,device='cpu',batch_size=1)
        by_id={r['id']:r for r in scored}
        for r in rows:
            if r['capture_fault']:
                by_id[r['id']]=dict(id=r['id'],expected=r['expected'],synthetic=r['synthetic'],speaker=r['speaker'],source=r['source'],
                    candidate=None,capture_fault=True,score=None,margin=None,intent_index=0,intent_probability=0.,
                    top_intent_probability=0.,target_named_probability=0.)
        return [by_id[r['id']] for r in rows]
    with ParallelScorer(entries,workers=recipe['training']['decoder_workers'],return_scores=True) as scorer:
        dev=score(recipe['views']['dev'],scorer)
        if full:
            quantized_dev=dev
            engine=ONNXModel(ROOT/'results/export/model.fp32.onnx')
            floating_dev=score(recipe['views']['dev'],scorer)
            engine=ONNXModel(bundle/'model.onnx')
            fidelity=quantization_gate(training.summarize(floating_dev),training.summarize(quantized_dev),exported['int8_bytes'])
            save(ROOT/'results/export-development.json',dict(export=exported,fidelity=fidelity,
                floating=training.summarize(floating_dev),quantized=training.summarize(quantized_dev)))
            if not fidelity['pass']:raise ValueError('Original INT8 development fidelity gate failed; test remains unopened')
        human_dev=[r for r in dev if not r['synthetic']];calibration=score(recipe['views']['calibration'],scorer)
        qualified=calibrate_human(calibration);qualified['binding']=binding
        save(bundle/'policy.json',qualified)
        recall=None
        if not full:
            recall=fit_policy(human_dev,calibration,binding);save(bundle/'recall-policy.json',recall)
        original_recipe=read(ROOT/f'submission/models/{args.model_choice}/recipe.json')
        new_recipe={**original_recipe,'binding':binding,'policy_enabled':qualified['enabled'],'parameters':recipe['parameters']}
        if recall:new_recipe['recall_policy_sha256']=sha(bundle/'recall-policy.json')
        save(bundle/'recipe.json',new_recipe);verify_bundle(bundle,diagnostic=True)
        save(ROOT/'results/selection-before-test.json',dict(selection=selection,binding=binding,
            recall_policy_sha256=sha(bundle/'recall-policy.json') if recall else None,policy_sha256=sha(bundle/'policy.json'),test_used=False))
        results={}
        for name in ('test','holdout'):
            records=score(recipe['views'][name],scorer)
            for row in records:
                event={'output':{'accepted':False,'command':None,'reason':'capture_fault' if row.get('capture_fault') else 'unqualified_model'},'diagnostic_candidate':row}
                application=(application_decision(event,dict(mode='demo',intent_threshold=.95,slot_threshold=.9))
                             if full else decorate_event(event,recall,mode='demo')['application'])
                row['demo']=application['command'];row['raw']=row['candidate']
            results[name]={kind:{key:metrics([r for r in records if r['synthetic']==(kind=='synthetic')],key)
                                for key in ('raw','demo')} for kind in ('human','synthetic')}
            save(ROOT/'results'/f'{name}-predictions.json',records)
    original=torch.load(ROOT/f'submission/checkpoints/{args.model_choice}-selected.pt',map_location='cpu',weights_only=True)['weights']
    differences={k:float((v-original[k]).abs().max()) for k,v in payload['weights'].items() if v.dtype.is_floating_point}
    comparison=dict(seed=args.seed,original_seed=recipe['original_seed'],recipe_model=args.model_choice,
        same_human_epoch=selection['foundation']['epoch']==recipe['selected_human_epoch'],
        same_mixed_epoch=selection['recognition_epoch']==recipe['selected_mixed_epoch'],
        weights_bitwise_equal=all(torch.equal(v,original[k]) for k,v in payload['weights'].items()),
        maximum_weight_absolute_difference=max(differences.values()),
        original_checkpoint_sha256=recipe['selected_checkpoint_sha256'],reproduced_checkpoint_sha256=sha(checkpoint),
        onnx_sha256=sha(bundle/'model.onnx'),export_parity=exported,
        strict_qualification_enabled=qualified['enabled'],results=results,
        original_human_test=dict(raw_exact=516 if full else 489,demo_exact=274 if full else 238,positive_n=636,
                                 unknown_accepts=3 if full else 1,negative_n=229),
        physical_pi_measured=False,original_protocol_reproduced=args.seed==recipe['original_seed'],
        limitations=['CUDA CTC is seeded with warn-only determinism; equal seeds do not promise identical weights.',
                     'Changed seed is a new run, not reproduction of original weights.'])
    save(ROOT/'results/comparison.json',comparison)
    return comparison


def benchmark_command(root, executable):
    root=Path(root)
    return [str(executable),str(root/'benchmark.py'),'--target','pc','--model',str(root/'results/model'),
            '--output',str(root/'results/benchmark'),'--no-setup']


def benchmark_reproduction(root, executable):
    result=subprocess.call(benchmark_command(root,executable),cwd=root)
    if result==0:
        from .report import render_reproduction
        render_reproduction(root)
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    recipe_dir=ROOT/('submission/reproduction-full' if args.model_choice=='full' else 'submission/reproduction')
    recipe=read(recipe_dir/'recipe.json')
    if args.verify:
        rows=load_recipe_rows(recipe,recipe_dir)
        for field in ('normalization','catalog'):
            if sha(ROOT/recipe[field]['path'])!=recipe[field]['sha256']:raise ValueError('Recipe asset identity changed')
        expected=('3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e' if args.model_choice=='full'
                  else 'fc87c98c242ccb6f37ed9fcd84bad3869b8c90eaae6267fdab2a1a5af2737f6d')
        if sha(ROOT/f'submission/models/{args.model_choice}/model.onnx')!=expected:raise ValueError('Selected ONNX changed')
        print(f'Verified recipe: {len(rows)} original rows, selected model and normalization. No training performed.');return 0
    if args.seed is None:
        args.seed=int(input('Training seed [53]: ').strip() or '53') if sys.stdin.isatty() and not args.yes else 53
    if not 0<=args.seed<=2**32-101:
        raise ValueError('Seed must be 0..4294967195; the original mixed stage adds 100')
    if args.source_cache:args.source_cache=args.source_cache.resolve()
    if not args.worker:
        destination=(args.workspace or ROOT/'.cache/submission/reproductions'/run_id()).resolve()
        copy_source(destination)
        command=[str(sys.executable if args.no_setup else bootstrap(group='training',root=destination)),
                 str(destination/'reproduce.py'),'--worker','--no-setup','--seed',str(args.seed),'--model-choice',args.model_choice]
        if args.prepare_only:command+=['--prepare-only']
        if args.source_cache:command+=['--source-cache',str(args.source_cache)]
        if args.no_benchmark:command+=['--no-benchmark']
        if args.yes:command+=['--yes']
        if args.gpu is not None:command+=['--gpu',str(args.gpu)]
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        result=subprocess.run(command,cwd=destination,env=env)
        print('Separate reproduction folder:',destination,flush=True)
        return result.returncode
    if args.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    prepared,_=reconstruct(args,recipe)
    if args.prepare_only:
        print('Prepared pinned data and features; no training or model reproduction claim.');return 0
    checkpoint,selection=fit(args,recipe,prepared)
    comparison=export_and_test(args,recipe,prepared,checkpoint,selection)
    from .report import render_reproduction
    render_reproduction(ROOT)
    print(json.dumps(comparison,indent=2),flush=True)
    if not args.no_benchmark and sys.stdin.isatty() and input('Benchmark the regenerated model on this PC with benchmark.py? [y/N]: ').strip().lower() in ('y','yes'):
        return benchmark_reproduction(ROOT,sys.executable)
    return 0
