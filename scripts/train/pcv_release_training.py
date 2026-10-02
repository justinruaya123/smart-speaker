"""Fresh source-only release experiment; offline teacher never ships.

Frozen implementations are reused through process-local adapters. New source,
labels, normalization, finite teacher heads and student weights have independent
receipts. A completed stage may be verified/reused; partial fitting is preserved.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_qualification import verify_bindings

HERE=ROOT/'experiments/pcv-release-train-v1'
CACHE=ROOT/'.cache/pcv-release-train-v1'
SOURCE=ROOT/'experiments/pcv-source-refresh-v1'
OLD_TEACHER=ROOT/'.cache/pcv-updated-distill-v1'


def valid_run_id(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',value):raise ValueError('Unsafe run ID')
    return value


def validate_rows(rows,allowed,fitting=False,human_only=False):
    allowed={(ROOT/p).resolve() for p in allowed}
    for row in rows:
        if (ROOT/row['parquet']).resolve() not in allowed:raise ValueError('Recording is not from the pinned supplied source')
        if row['split'] not in ('train','dev'):raise ValueError('Held-out audio cannot enter teacher or fitting')
        if fitting and (row['split']!='train' or row['published_split'] not in ('train','numerals') or not row['training_eligible']):
            raise ValueError('Held-out or excluded recording cannot enter fitting')
        if human_only and row['synthetic']:raise ValueError('Synthetic audio cannot enter human foundation')


def completed_stage(directory,epochs,protocol_hash=None):
    directory=Path(directory)
    if not directory.exists():return None
    history=directory/'history.json';selected=directory/'selection.json'
    if not history.exists() or not selected.exists() or len(json.loads(history.read_text()))!=epochs:
        raise ValueError('Preserved incomplete training stage; choose a new --run-id: '+str(directory))
    value=json.loads(selected.read_text());verify_bindings(ROOT,{value['checkpoint']:value['checkpoint_sha256']})
    if protocol_hash:
        import torch
        checkpoint=torch.load(ROOT/value['checkpoint'],map_location='cpu',weights_only=True)
        if checkpoint['protocol_sha256']!=protocol_hash:raise ValueError('Completed checkpoint protocol differs')
    return value


def configure():
    import torch
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def select_arm(arms):
    # Epoch breaks ties within a run, not between the matched hard/teacher arms.
    return max(arms,key=lambda name:arms[name]['key'][:2])


def prepared_rows():
    from scripts.train.pcv_updated_training import development_subset
    receipt=json.loads((HERE/'data-freeze.json').read_text());verify_bindings(ROOT,receipt['bindings'])
    verify_bindings(ROOT,{r['path']:r['sha256'] for r in receipt['manifests'].values()})
    source=json.loads((SOURCE/'acquisition-complete.json').read_text())
    allowed={r['path'] for r in source['files']}
    train=[r for r in read_rows(CACHE/'prepared/train.jsonl') if r['training_eligible']]
    dev=development_subset(read_rows(CACHE/'prepared/dev.jsonl'))
    validate_rows(train,allowed,fitting=True);validate_rows(dev,allowed)
    return train,dev,allowed


def freeze(run_id,rows,dev):
    import torch
    directory=HERE/'runs'/run_id;path=directory/'protocol.json'
    if path.exists():
        value=json.loads(path.read_text());verify_bindings(ROOT,value['bindings']);return value
    files=[Path(__file__),ROOT/'scripts/models/pcv_release_pooling.py',ROOT/'scripts/models/pcv_contextual.py',
        ROOT/'scripts/train/pcv_context_training.py',ROOT/'scripts/train/pcv_context_int8.py',
        ROOT/'scripts/train/pcv_teacher_train.py',ROOT/'scripts/train/pcv_teacher_data.py',
        ROOT/'scripts/train/pcv_student_distill.py',ROOT/'scripts/train/pcv_distillation.py',
        ROOT/'scripts/models/pcv_teacher.py',ROOT/'scripts/models/pcv_residual.py',
        ROOT/'scripts/train/pcv_updated_heads.py',ROOT/'scripts/train/pcv_updated_training.py',
        ROOT/'scripts/train/pcv_public_training.py',ROOT/'scripts/test/pcv_public_eval.py',
        ROOT/'scripts/test/pcv_public_qualification.py',ROOT/'scripts/pcv_parallel.py',
        ROOT/'scripts/test/test_pcv_release_training.py',ROOT/'scripts/test/test_pcv_release_pooling.py',
        ROOT/'experiments/phrase-contract-v1/pcvcm/decoder.py',ROOT/'scripts/runtime/pcv_schema.py',
        HERE/'data-freeze.json',HERE/'catalog.json',HERE/'model-preflight.json',SOURCE/'acquisition-complete.json',SOURCE/'revision-difference.json',
        CACHE/'prepared/normalization.json',ROOT/'experiments/pcv-updated-distill-v1/teacher-acquisition.json']
    protocol={'experiment':'pcv-release-train-v1','run_id':run_id,'seed':1102,
        'requirements':['ME-02','ME-03','ME-04','ME-06','ME-07','ME-08'],
        'hypothesis':'Fresh source revision, conservative training-label repairs, additional capacity and learned semantic pooling improve exact command development recognition. Matched hard/distilled arms test teacher usefulness.',
        'initialization':'Random student and finite teacher heads. No prior source-fitted checkpoint or normalization is reused because published splits moved.',
        'source_revision':json.loads((SOURCE/'publisher-metadata.json').read_text())['sha'],
        'fit_ids':[r['id'] for r in rows],'development_ids':[r['id'] for r in dev],
        'architecture':{'width':192,'layers':6,'heads':4,'feedforward':384,'dropout':.1},
        'architecture_class':'PooledContextVCM','maximum_graph_bytes':4*1024*1024,
        'resource_note':'New experiment budget 4MiB before fitting. Prior 2MiB studies remain frozen; no measured Pi latency claim.',
        'training':{'human_epochs':40,'mixed_epochs':48,'steps_per_epoch':96,'batch_size':32,'microbatch_size':8,
            'human_learning_rate':.0008,'mixed_learning_rate':.00025,'minimum_learning_rate':.00002,
            'weight_decay':.001,'gradient_clip':5.,'intent_weight':.3,'target_weight':.15,
            'sequence_weight':.3,'frame_weight':.1,'semantic_weight':.1,'sequence_temperature':.03,
            'head_temperature':2.,'candidate_count':8,'development_every_epochs':4,
            'human_batch':{'human_positive':24,'human_negative':4,'numeral':4,'synthetic':0},
            'mixed_batch':{'human_positive':12,'human_negative':4,'numeral':4,'synthetic':12}},
        'teacher_training':{'human_epochs':40,'mixed_epochs':32,'steps_per_epoch':96,'batch_size':32,'microbatch_size':8,
            'human_learning_rate':.001,'mixed_learning_rate':.0003,'minimum_learning_rate':.00002,
            'weight_decay':.0001,'gradient_clip':5.,'intent_weight':.3,'target_weight':.15,
            'human_batch':{'human_positive':28,'human_negative':4,'synthetic':0},
            'mixed_batch':{'human_positive':12,'human_negative':4,'synthetic':16},'development_every_epochs':4},
        'teacher_model':{'width':256,'dilations':[1,2,4,8,1,2]},
        'teacher_gate':{'human_recall_min':.85,'synthetic_recall_min':.95},
        'arms':['hard','distilled'],'selection':'Mean human/synthetic exact raw development recall, then fewer human negative errors. Ties retain hard arm. Human foundation uses human recall. Test never selects.',
        'quality_gates':{'precision':.99,'recall':.90,'human_far':.01,'all_18_intents_in_human_calibration':True},
        'quantization_gate':{'max_human_exact_loss':1,'max_synthetic_exact_loss':1,'max_extra_human_negative':1},
        'audio':'Original pinned HF waveforms only; no generated/spliced/augmented audio. Supplied synthetic stage follows human stage.',
        'teacher':'Frozen WavLM 4c66d4806a428f2e922ccfa1a962776e232d487b; authorized external pretrained weights only; mean layers6/9/12, no waveform normalization; never shipped.',
        'test_exposure':'Old aggregate results known. New test metadata/source integrity audit only; no new test predictions before selection lock.',
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'installed_artifact_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts').iterdir() if p.is_file()},
        'hardware':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda},
        'physical_pi_test':False,'installation_authorized':False}
    immutable(path,protocol);return protocol


def teacher_features(rows,dev,allowed):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from einops import rearrange,reduce
    from transformers import WavLMModel
    from scripts.train.pcv_teacher_data import verified_pcm,teacher_frames
    wanted_rows=[r for r in rows if r['reference_status']!='auxiliary']+dev
    validate_rows(wanted_rows,allowed)
    receipt_path=HERE/'teacher-feature-freeze.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());verify_bindings(ROOT,{receipt['index']:receipt['sha256']})
        return receipt
    acquisition=json.loads((ROOT/'experiments/pcv-updated-distill-v1/teacher-acquisition.json').read_text())
    verify_bindings(ROOT,{r['path']:r['sha256'] for r in acquisition['files']})
    old_receipt=json.loads((ROOT/'experiments/pcv-updated-distill-v1/teacher-feature-freeze.json').read_text())
    verify_bindings(ROOT,{old_receipt['index']:old_receipt['sha256']})
    by_pcm={r['pcm_sha256']:r for r in read_rows(ROOT/old_receipt['index'])}
    index=CACHE/'teacher-feature-index.jsonl';existing={r['pcm_sha256']:r for r in read_rows(index)} if index.exists() else {}
    by_pcm.update(existing);completed=[];missing=defaultdict(dict);reused=0;began=time.perf_counter()
    for row in wanted_rows:
        item=by_pcm.get(row['pcm_sha256'])
        if item:
            verify_bindings(ROOT,{item['path']:item['sha256']});completed.append({**item,'id':row['id'],'split':row['split']});reused+=1
        else:missing[row['parquet']][row['parquet_row']]=row
    model=None
    if missing:
        model=WavLMModel.from_pretrained(str(OLD_TEACHER/'teacher'),local_files_only=True).cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
    target=CACHE/'teacher-features';target.mkdir(parents=True,exist_ok=True)
    with index.open('w',encoding='utf8',newline='\n') as stream:
        for item in completed:stream.write(json.dumps(item)+'\n')
        stream.flush()
        for parquet,selected in missing.items():
            position=0
            for batch in pq.ParquetFile(ROOT/parquet).iter_batches(batch_size=64,columns=['audio']):
                for value in batch.to_pylist():
                    row=selected.get(position);position+=1
                    if row is None:continue
                    pcm=verified_pcm(row,value['audio']['bytes'])
                    with torch.inference_mode():
                        outputs=model(rearrange(torch.from_numpy(pcm).cuda(),'t -> 1 t'),output_hidden_states=True)
                        feature=reduce([outputs.hidden_states[i] for i in (6,9,12)],'l b t c -> b t c','mean')
                        feature=rearrange(feature,'1 t c -> c t').cpu().numpy().astype('float16')
                    if feature.shape!=(768,teacher_frames(len(pcm))) or not np.isfinite(feature).all():raise ValueError('Invalid teacher feature')
                    path=target/(row['pcm_sha256']+'.npy');np.save(path,feature,allow_pickle=False)
                    item={'id':row['id'],'split':row['split'],'pcm_sha256':row['pcm_sha256'],
                        'path':str(path.relative_to(ROOT)),'sha256':sha(path),'frames':feature.shape[1]}
                    completed.append(item);stream.write(json.dumps(item)+'\n');stream.flush()
                    if len(completed)%100==0:print('REFRESHED_TEACHER_FEATURES',len(completed),len(wanted_rows),flush=True)
    if len(completed)!=len(wanted_rows):raise ValueError('Missing teacher features')
    receipt={'index':str(index.relative_to(ROOT)),'sha256':sha(index),'rows':len(completed),'reused_by_pcm':reused,
        'new':len(completed)-reused,'seconds':time.perf_counter()-began,'test_inference_accessed':False,
        'reuse_note':'Only frozen pretrained encoder outputs reused, never prior source-fitted heads or targets.',
        'source_data_freeze_sha256':sha(HERE/'data-freeze.json')}
    immutable(receipt_path,receipt)
    del model;torch.cuda.empty_cache();return receipt


def run(run_id='release-v1'):
    import numpy as np
    import torch
    from scripts.train import pcv_teacher_train as teacher,pcv_context_training as student,pcv_student_distill as distill
    from scripts.models.pcv_release_pooling import PooledContextVCM
    from scripts.train.pcv_public_training import load_arrays
    valid_run_id(run_id);configure();rows,dev,allowed=prepared_rows()
    protocol=freeze(run_id,rows,dev);directory=HERE/'runs'/run_id;output=CACHE/'runs'/run_id
    selection_path=directory/'selection-before-calibration.json'
    if selection_path.exists():
        value=json.loads(selection_path.read_text());verify_bindings(ROOT,value['bindings']);print('VERIFIED_COMPLETE_TRAINING',run_id,flush=True);return value
    teacher_features(rows,dev,allowed)
    guard=lambda values,fitting=False,human_only=False:validate_rows(values,allowed,fitting,human_only)
    teacher.HERE=HERE;arrays=teacher.feature_arrays([r for r in rows if r['reference_status']!='auxiliary']+dev)
    probe=[r for r in rows if r['reference_status']!='auxiliary' and teacher.ctc_eligible(r,arrays[r['id']].shape[1])]
    entries=json.loads((HERE/'catalog.json').read_text())['entries']
    teacher.HERE=directory;teacher.CACHE=output/'teacher';teacher.check_rows=guard
    probe_ids={r['id'] for r in probe}
    tp={'seed':protocol['seed']+10,'model':protocol['teacher_model'],'training':protocol['teacher_training'],
        'bindings':{str((directory/'protocol.json').relative_to(ROOT)):sha(directory/'protocol.json')},
        'fit_ids':[r['id'] for r in probe],'ctc_frame_exclusions':[r['id'] for r in rows if r['reference_status']!='auxiliary' and r['id'] not in probe_ids]}
    immutable(directory/'teacher-training-protocol.json',tp);ph=sha(directory/'teacher-training-protocol.json')
    h=completed_stage(teacher.CACHE/'human',tp['training']['human_epochs'],ph) or teacher.stage('human',None,[r for r in probe if not r['synthetic']],dev,arrays,entries,tp)
    m=completed_stage(teacher.CACHE/'mixed',tp['training']['mixed_epochs'],ph) or teacher.stage('mixed',ROOT/h['checkpoint'],probe,dev,arrays,entries,tp)
    gate=protocol['teacher_gate'];permitted=m['development']['human']['recall']>=gate['human_recall_min'] and m['development']['synthetic']['recall']>=gate['synthetic_recall_min']
    immutable(directory/'teacher-selection.json',{'human':h,'mixed':m,'distillation_gate_pass':permitted,'test_inference_accessed':False})
    del arrays;torch.cuda.empty_cache()
    sp={**protocol,'teacher':m};immutable(directory/'student-protocol.json',sp)
    targets={};frames={}
    if permitted:
        distill.HERE=directory;distill.CACHE=output;distill.PARENT=HERE;distill.PARENT_CACHE=CACHE
        distill.freeze=lambda:sp;distill.rows_for_probe=lambda:(probe,dev);distill.check_rows=guard
        def fresh_arrays(values):
            teacher.HERE=HERE
            try:return teacher.feature_arrays(values)
            finally:teacher.HERE=directory
        distill.feature_arrays=fresh_arrays
        if not (directory/'targets-freeze.json').exists():distill.make_targets()
        frozen=json.loads((directory/'targets-freeze.json').read_text());verify_bindings(ROOT,{frozen['path']:frozen['sha256']})
        for item in read_rows(ROOT/frozen['path']):
            verify_bindings(ROOT,{item['frame_path']:item['frame_sha256']})
            targets[item['id']]=item;frames[item['id']]=np.load(ROOT/item['frame_path'],allow_pickle=False).astype('float32')
    student_rows=[r for r in rows if teacher.ctc_eligible(r,(r['frames']+1)//2)]
    by_id={r['id']:r for r in student_rows}
    targets={k:v for k,v in targets.items() if k in by_id}
    for key,item in targets.items():
        if by_id[key]['pcm_sha256']!=item['pcm_sha256']:raise ValueError('Fresh teacher target source mismatch')
    norm=json.loads((CACHE/'prepared/normalization.json').read_text());arrays=load_arrays(student_rows+dev,norm)
    student.HERE=directory;student.CACHE=output/'student';student.ContextualVCM=PooledContextVCM
    student.validate_training_rows=lambda values,human_only=False:guard(values,True,human_only)
    ph=sha(directory/'protocol.json')
    human=completed_stage(student.CACHE/'human',protocol['training']['human_epochs'],ph) or student.stage('human',None,[r for r in student_rows if not r['synthetic']],dev,arrays,{}, {},norm,entries,protocol)
    arms={}
    for arm in protocol['arms']:
        if arm=='distilled' and not permitted:continue
        student.CACHE=output/arm
        arms[arm]=completed_stage(student.CACHE/'mixed',protocol['training']['mixed_epochs'],ph) or student.stage('mixed',ROOT/human['checkpoint'],student_rows,dev,arrays,targets if arm=='distilled' else {},frames if arm=='distilled' else {},norm,entries,protocol)
    winner=select_arm(arms);selected=arms[winner]
    verify_bindings(ROOT,protocol['bindings']);verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    files=[directory/'protocol.json',HERE/'data-freeze.json',SOURCE/'acquisition-complete.json',directory/'teacher-selection.json',ROOT/selected['checkpoint']]
    receipt={'winner':winner,'selected':selected,'human':human,'arms':arms,'teacher_gate_pass':permitted,
        'checkpoint':selected['checkpoint'],'protocol':str((directory/'protocol.json').relative_to(ROOT)),
        'source_receipt':str((SOURCE/'acquisition-complete.json').relative_to(ROOT)),
        'test_inference_accessed':False,'test_used_for_selection':False,'source_revision':protocol['source_revision'],
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    immutable(selection_path,receipt);print('RELEASE_TRAINING_COMPLETE',winner,flush=True);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='release-v1')
    run(parser.parse_args().run_id)
