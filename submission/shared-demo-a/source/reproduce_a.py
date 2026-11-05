"""Fresh main acquisition and original baseline-A full-fit reproduction.

The historical fitter is imported unchanged from this isolated snapshot. No
application imports, fallback model, relabeling or post-TEST selection exists.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
REPOSITORY = 'justinruaya123/ai231-smart-speaker'
PREFIX = 'shared-demo-a-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def read_rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_file(path,record):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or ('bytes' in record and path.stat().st_size!=record['bytes']) or sha(path)!=record['sha256']:
        raise ValueError('File identity mismatch: '+str(path))
    return path


def validate_rows(views):
    identities={k:{} for k in ('id','sha256','pcm_sha256','speaker')}
    total=0
    for view,rows in views.items():
        expected='dev' if view.endswith('dev') else 'holdout' if view=='official' else view
        for row in rows:
            if row['split']!=expected:
                raise ValueError('Invalid declared split: '+row['id'])
            for field in identities:
                if field=='speaker' and row['kind']!='human':
                    continue
                value=row.get(field)
                if not value:
                    raise ValueError('Missing recording identity: '+field)
                if field in ('sha256','pcm_sha256') and not re.fullmatch('[a-f0-9]{64}',value):
                    raise ValueError('Malformed recording identity: '+field)
                previous=identities[field].get(value)
                if previous is not None and (field=='id' or previous!=expected):
                    raise ValueError('Duplicate/cross-split identity: '+field)
                identities[field][value]=expected
            total+=1
    return total


def extract_audio(archive,destination,expected):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    seen=set()
    with tarfile.open(archive,'r') as stream:
        for member in stream:
            path=PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or len(path.parts)!=1 or
                    '\\' in member.name or not re.fullmatch('[a-f0-9]{64}\\.wav',member.name)):
                raise ValueError('Unsafe audio archive member: '+member.name)
            digest=path.stem
            if digest not in expected or digest in seen:
                raise ValueError('Unexpected/duplicate audio archive identity')
            target=destination/member.name
            with stream.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source,output)
            verify_file(target,dict(sha256=digest)); seen.add(digest)
    if seen!=set(expected):
        raise ValueError('Missing audio archive identity')
    return seen


def tensor_identity(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


def compare_curves(original,repeat):
    strip=lambda events:[{k:v for k,v in e.items() if k!='elapsed_seconds'} for e in events]
    if strip(original)!=strip(repeat):
        raise ValueError('Full training/DEV curve differs from original')
    return True


def verify_reference(row,original):
    if any(row.get(k)!=original.get(k) for k in ('id','class_index','sha256','pcm_sha256')):
        raise ValueError('Original evaluation reference identity/label differs')
    return True


def verify_preparation_source(receipt,current):
    if receipt.get('source')!=current:
        raise ValueError('Prepared source/reference identity changed')
    return True


def verify_source():
    snapshot=json.loads((HERE/'snapshot.json').read_text())
    for name,record in snapshot['bindings'].items():
        verify_file(ROOT/name,record)
    closure=json.loads((HERE/'training-source.json').read_text())
    for name,record in closure.items():
        verify_file(Path(__file__).parent/name,record)
    return dict(snapshot_sha256=sha(HERE/'snapshot.json'),training_source_sha256=sha(HERE/'training-source.json'),
        reference_sha256=sha(HERE/'reproduction-reference.json'),runner_sha256=sha(__file__),
        dataset_lock_sha256=sha(HERE/'dataset.lock.json'))


def dataset(out):
    from huggingface_hub import HfApi, hf_hub_download
    lock=json.loads((HERE/'dataset.lock.json').read_text())
    revision=HfApi().dataset_info(REPOSITORY,revision='main').sha
    def fetch(name,record):
        path=Path(hf_hub_download(REPOSITORY,PREFIX+'/'+name,repo_type='dataset',revision=revision,
            cache_dir=str(out/'hf-cache'),local_dir=str(out/'dataset')))
        return verify_file(path,record)
    manifest_path=fetch('manifest.json',dict(sha256=lock['manifest_sha256']))
    manifest=json.loads(manifest_path.read_text())
    paths={name:fetch(name,record) for name,record in manifest['files'].items()}
    views={name:read_rows(paths[name+'.jsonl.gz']) for name in manifest['views']}
    for name,rows in views.items():
        if len(rows)!=manifest['views'][name]['rows']:
            raise ValueError('Missing dataset rows: '+name)
    validate_rows(views)
    receipt=dict(repository=REPOSITORY,requested_branch='main',resolved_revision=revision,
        publication_revision=lock['publication_revision'],manifest_sha256=sha(manifest_path),
        files={name:dict(sha256=sha(path),bytes=path.stat().st_size) for name,path in paths.items()})
    write(out/'dataset-acquisition.json',receipt)
    return manifest,views,paths,revision


def acquire_audio(out,rows,revision,local):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest
    audio=out/'audio'; audio.mkdir(exist_ok=True)
    paths={}
    groups=defaultdict(list); singles=[]
    for row in rows:
        loc=row['location']
        if loc.get('local_synthesis'):
            paths[row['id']]=local/(row['sha256']+'.wav')
        elif loc.get('audio_file'):
            singles.append(row)
        else:
            groups[(loc['repository'],loc['revision'],loc['file'],loc['sha256'])].append(row)
    def admit(row,payload):
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError('Original WAV identity mismatch: '+row['id'])
        pcm,rate=sf.read(io.BytesIO(payload),dtype='float32')
        if pcm_digest(pcm,rate)!=row['pcm_sha256']:
            raise ValueError('Original PCM identity mismatch: '+row['id'])
        target=audio/(row['sha256']+'.wav')
        if not target.exists():
            with target.open('xb') as stream: stream.write(payload)
        verify_file(target,dict(sha256=row['sha256']))
        paths[row['id']]=target
    for (repo,rev,name,digest),members in groups.items():
        path=Path(hf_hub_download(repo,name,repo_type='dataset',revision=rev,cache_dir=str(out/'hf-cache')))
        verify_file(path,dict(sha256=digest))
        wanted=defaultdict(list)
        for row in members: wanted[row['location']['row']].append(row)
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['audio']):
            for index,cell in enumerate(batch.column('audio').to_pylist()):
                for row in wanted.get(offset+index,[]): admit(row,cell['bytes'])
            offset+=len(batch)
        print(json.dumps(dict(stage='source_audio',recordings=len(paths),total=len(rows),source=repo,file=name)),flush=True)
    def single(row):
        loc=row['location']
        path=Path(hf_hub_download(loc['repository'],loc['file'],repo_type='dataset',revision=loc['revision'],
            cache_dir=str(out/'hf-cache')))
        admit(row,path.read_bytes())
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(single,singles))
    if set(paths)!={r['id'] for r in rows}:
        raise ValueError('Missing original recording reconstruction')
    return [{**r,'audio':str(paths[r['id']].resolve())} for r in rows]


def extract_features(rows,verify=True):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from vcm.frontend import features
    from scripts.test.speaker_demo_suite import read_audio
    from scripts.train.speaker_optionb_tiny import array_digest
    def one(row):
        values=features(read_audio(row))
        if verify and (array_digest(values)!=row['feature_sha256'] or list(values.shape)!=row['feature_shape']):
            raise ValueError('Frozen feature identity differs: '+row['id'])
        return row['id'],values
    with threadpool_limits(limits=1,user_api='blas'):
        features(np.zeros(400,dtype=np.float32))
        result={}
        with ThreadPoolExecutor(max_workers=12) as pool:
            for identifier,values in pool.map(one,rows):
                result[identifier]=values
                if len(result)%2000==0: print('verified_features='+str(len(result)),flush=True)
    return result


def prepare(out):
    import torch
    from scripts.train.speaker_optionb_tiny import train_normalization
    binding=verify_source()
    manifest,views,paths,revision=dataset(out)
    expected={r['sha256'] for rows in views.values() for r in rows if r['location'].get('local_synthesis')}
    extract_audio(paths['authored-synthesis.tar'],out/'local-synthesis',expected)
    fitrows=views['train']+views['original-dev']+views['new-dev']
    prepared=acquire_audio(out,fitrows,revision,out/'local-synthesis')
    arrays=extract_features(prepared)
    human=[r for r in prepared if r['split']=='train' and r['kind']=='human']
    norm=train_normalization(human,arrays)
    expected_norm=json.loads(paths['normalization.json'].read_text())
    if norm!=expected_norm or len(human)!=1889:
        raise ValueError('Human TRAIN-only normalization differs')
    torch.save(arrays,out/'train-dev-features.pt')
    write(out/'prepared-train-dev.json',prepared)
    write(out/'normalization.json',norm)
    write(out/'preparation.json',dict(passed=True,feature_rows=len(arrays),human_normalization_rows=len(human),
        test_features_extracted=False,optimizer_steps=0,source=binding,
        prepared_sha256=sha(out/'prepared-train-dev.json'),features_sha256=sha(out/'train-dev-features.pt'),
        normalization_sha256=sha(out/'normalization.json')))
    return views


def train(out,smoke=False):
    import torch
    from scripts.train.demo_slots_fit import fit
    verify_source()
    receipt=json.loads((out/'preparation.json').read_text())
    verify_preparation_source(receipt,verify_source())
    for key,name in [('prepared','prepared-train-dev.json'),('features','train-dev-features.pt'),('normalization','normalization.json')]:
        verify_file(out/name,dict(sha256=receipt[key+'_sha256']))
    rows=json.loads((out/'prepared-train-dev.json').read_text())
    arrays=torch.load(out/'train-dev-features.pt',weights_only=False,map_location='cpu')
    norm=json.loads((out/'normalization.json').read_text())
    reference=json.loads((HERE/'reproduction-reference.json').read_text())
    config=reference['config']
    if smoke: config={**config,'foundation_epochs':1,'mixed_epochs':1,'steps_per_epoch':2,'software_smoke':True}
    result=fit([r for r in rows if r['split']=='train'],[r for r in rows if r['split']=='dev'],arrays,norm,
        out/('optimizer-smoke' if smoke else 'heads_s53'),config)
    if smoke:
        if result['optimizer_steps']!=4 or not result['weights_changed']:
            raise ValueError('Real optimizer smoke did not complete')
        return result
    best=torch.load(out/'heads_s53/best.pt',weights_only=False,map_location='cpu')
    last=torch.load(out/'heads_s53/last.pt',weights_only=False,map_location='cpu')
    fields=['draws','optimizer_steps','sequence_sha256','initial_encoder_sha256','source_draws','selected_epoch']
    checks={key:result[key]==reference[key] for key in fields}
    original=torch.load(ROOT/'submission/checkpoints/shared-demo-a-selected.pt',weights_only=False,map_location='cpu')
    checks['original_selected_tensors']=(set(best['state_dict'])==set(original['state_dict']) and
        all(torch.equal(value,original['state_dict'][key]) for key,value in best['state_dict'].items()))
    checks['selected_tensors']=tensor_identity(best['state_dict'])==reference['selected_tensor_sha256']
    checks['final_tensors']=tensor_identity(last['state_dict'])==reference['final_tensor_sha256']
    for tag,counts in [('selected',best['sample_counts']),('final',result['sample_counts'])]:
        checks[tag+'_sample_counts']=hashlib.sha256(json.dumps(counts,sort_keys=True,separators=(',',':')).encode()).hexdigest()==reference[tag+'_sample_counts_sha256']
    try: compare_curves(reference['curve'],read_rows(out/'heads_s53/training-curve.jsonl')); checks['full_curve']=True
    except ValueError: checks['full_curve']=False
    comparison=dict(passed=all(checks.values()),checks=checks,selected_epoch=result['selected_epoch'],
        parameters=result['parameters'],checkpoint_sha256=sha(out/'heads_s53/best.pt'),
        selected_tensor_sha256=tensor_identity(best['state_dict']),qualified=False,actions_enabled=False)
    write(out/'training-comparison.json',comparison)
    if not comparison['passed']:
        raise ValueError('Actual full fit completed but exact reproduction comparison failed; preserve evidence')
    write(out/'selection-before-test.json',dict(checkpoint_sha256=sha(out/'heads_s53/best.pt'),
        reference_sha256=sha(HERE/'reproduction-reference.json'),dataset_acquisition_sha256=sha(out/'dataset-acquisition.json'),
        source=verify_source(),training_comparison_sha256=sha(out/'training-comparison.json'),
        normalization_sha256=sha(out/'normalization.json'),
        policy='unchanged action-authoritative argmax, missing required slots reject',test_features_extracted=False))
    return comparison


def evaluate(out):
    import numpy as np
    import torch
    import onnxruntime as ort
    from scripts.models.demo_slots_v1 import DemoSlotVCM
    from scripts.train.demo_slots_fit import infer_logits, predictions, metrics
    lock=json.loads((out/'selection-before-test.json').read_text())
    verify_file(out/'heads_s53/best.pt',dict(sha256=lock['checkpoint_sha256']))
    if verify_source()!=lock['source'] or sha(out/'dataset-acquisition.json')!=lock['dataset_acquisition_sha256']:
        raise ValueError('Selection lock inputs changed')
    acquisition=json.loads((out/'dataset-acquisition.json').read_text())
    dataset_dir=out/'dataset'/PREFIX
    verify_file(out/'normalization.json',dict(sha256=lock['normalization_sha256']))
    for name,record in acquisition['files'].items():
        verify_file(dataset_dir/name,record)
    views={n:read_rows(dataset_dir/(n+'.jsonl.gz')) for n in ('test','official')}
    rows=views['test']+views['official']
    prepared=acquire_audio(out,rows,acquisition['resolved_revision'],out/'local-synthesis')
    arrays=extract_features(prepared,verify=False)
    norm=json.loads((out/'normalization.json').read_text())
    torch.set_num_threads(1)
    checkpoint=torch.load(out/'heads_s53/best.pt',weights_only=False,map_location='cpu')
    model=DemoSlotVCM('heads').eval(); model.load_state_dict(checkpoint['state_dict'])
    scores=infer_logits(model,prepared,arrays,norm,torch.device('cpu'))
    from scripts.train.speaker_optionb_tiny import normalized,tensor_batch
    x,n=tensor_batch([normalized(arrays[prepared[0]['id']],norm)])
    graph=out/'reproduced.onnx'
    torch.onnx.export(model,(x,n),str(graph),input_names=['features','lengths'],output_names=['logits'],
        dynamic_axes={'features':{0:'batch',2:'frames'},'lengths':{0:'batch'},'logits':{0:'batch'}},
        opset_version=17,dynamo=False)
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
    reference=json.loads((HERE/'reproduction-reference.json').read_text())
    expected={r['id']:r for r in reference['predictions']}
    pred=predictions(scores,'heads'); maximum=0.; records=[]
    for row,values,decision in zip(prepared,scores,pred):
        x,n=tensor_batch([normalized(arrays[row['id']],norm)])
        actual=session.run(None,dict(features=x.numpy(),lengths=n.numpy()))[0][0]
        maximum=max(maximum,float(np.max(np.abs(values-actual))))
        if not np.allclose(values,actual,rtol=1e-5,atol=3e-5) or predictions(actual[None],'heads')[0]!=decision:
            raise ValueError('Reproduced Torch/ONNX parity differs')
        prior=expected[row['id']]
        verify_reference(row,prior)
        if row['sha256']!=prior['sha256'] or row['pcm_sha256']!=prior['pcm_sha256'] or int(decision)!=prior['predicted_class']:
            raise ValueError('Original versus reproduced complete-command decision differs: '+row['id'])
        records.append(dict(id=row['id'],class_index=row['class_index'],predicted_class=int(decision),
            sha256=row['sha256'],pcm_sha256=row['pcm_sha256']))
    if set(expected)!={r['id'] for r in prepared} or len(records)!=642:
        raise ValueError('Missing/duplicated original evaluation population')
    result=dict(passed=True,rows=len(records),new_test=metrics(prepared[:440],pred[:440]),
        exact_test=metrics([r for r in prepared[:440] if r.get('form')=='clean'],
            [int(p) for r,p in zip(prepared[:440],pred[:440]) if r.get('form')=='clean']),
        official=metrics(prepared[440:],pred[440:]),
        human_official=metrics([r for r in prepared[440:] if r['kind']=='human'],
            [int(p) for r,p in zip(prepared[440:],pred[440:]) if r['kind']=='human']),
        original_complete_decision_parity=True,torch_onnx_parity=True,max_absolute_logit_difference=maximum,
        graph_sha256=sha(graph),qualified=False,actions_enabled=False,exposed_regression=True)
    write(out/'reproduced-predictions.json',records); write(out/'evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('new_test','exact_test','official','human_official')}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','smoke','train','evaluate','all'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output.resolve()
    if args.stage in ('prepare','all'):
        out.mkdir(parents=True,exist_ok=False)
        write(out/'source-acquisition.json',dict(git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=verify_source()))
        prepare(out)
    if args.stage in ('smoke','all'): train(out,smoke=True)
    if args.stage in ('train','all'): train(out)
    if args.stage in ('evaluate','all'): evaluate(out)


if __name__=='__main__': main()
