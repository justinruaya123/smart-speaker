"""Isolated synthetic continuation of the existing project-trained causal CNN."""
import json
from pathlib import Path
import numpy as np
from einops import rearrange, reduce
from study import CACHE, HERE, ROOT, file_hash, identity, read_rows, write_json, write_rows
from synthbench import compile_synthetic_catalog, load_split, audit_all_splits
from pcvcm.compute import ComputeBudget
from pcvcm.data import batch_arrays
from pcvcm.training import development, sampling_weights, select_checkpoint


def require_training_split(rows):
    if not rows or any(r['split'] not in ('train','dev') for r in rows):
        raise ValueError('Only training and development may enter model fitting')


def fit_normalization(rows, arrays):
    from vcm.frontend import SPEC
    total, squares, frames = np.zeros(40), np.zeros(40), 0
    for r in rows:
        if r['split'] != 'train':
            continue
        x=arrays[r['recording_id']].astype(np.float64)
        total += reduce(x,'c t -> c','sum')
        squares += reduce(x*x,'c t -> c','sum')
        frames += x.shape[1]
    if not frames:
        raise ValueError('No training frames for normalization')
    mean=total/frames
    std=np.sqrt(np.maximum(squares/frames-mean*mean,1e-4))
    return dict(mean=mean.tolist(),std=std.tolist(),fit_split='train',frontend_spec=SPEC,
                frames=frames,synthetic=True)


def prepare():
    import soundfile as sf
    from vcm.frontend import features
    output=CACHE/'prepared'
    if output.exists():
        raise FileExistsError('Prepared inputs are immutable')
    global_audit=audit_all_splits()
    rows=load_split('train')+load_split('dev')
    require_training_split(rows)
    catalog=compile_synthetic_catalog(read_rows(HERE/'verbatims.jsonl'))
    phone_map={r['id']:r['phones'] for r in catalog['entries']}
    output.mkdir()
    arrays={}
    prepared=[]
    for index,row in enumerate(rows):
        pcm,_=sf.read(CACHE/row['audio'],dtype='float32')
        x=features(pcm)
        phones=phone_map[row['verbatim_id']]
        if x.shape[1]<len(phones)+sum(a==b for a,b in zip(phones,phones[1:])):
            raise ValueError('CTC-infeasible generated clip; preserve failure')
        path=output/'features'/f'{index:07}.npy'
        path.parent.mkdir(exist_ok=True)
        np.save(path,x)
        arrays[row['recording_id']]=x
        prepared.append({**row,'phones':phones,'features':str(path.relative_to(output)),
            'features_sha256':file_hash(path),'phrase_id':row['verbatim_id'],
            'speaker_id':row['voice_group']})
    write_json(output/'normalization.json',fit_normalization(rows,arrays))
    write_json(output/'catalog.json',catalog)
    write_json(output/'global-audit.json',global_audit)
    for split in ('train','dev'):
        write_rows(output/f'{split}.jsonl',[r for r in prepared if r['split']==split])
    freeze={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'freeze.json',dict(files=freeze,protocol_sha256=file_hash(HERE/'protocol.json'),
        test_assignment_sha256=file_hash(HERE/'generation-plan.jsonl'),test_audio_used=False))
    return freeze


def train(output):
    import torch
    from vcm.model import CausalVCM
    from pcvcm.data import verified_features
    output=Path(output)
    if output.exists():
        raise FileExistsError('Choose a new run; interrupted evidence is retained')
    prepared=CACHE/'prepared'
    freeze=json.loads((prepared/'freeze.json').read_text())
    for name,digest in freeze['files'].items():
        if file_hash(prepared/name)!=digest:
            raise ValueError('Frozen training inputs changed')
    protocol=json.loads((HERE/'protocol.json').read_text())
    if freeze['protocol_sha256']!=file_hash(HERE/'protocol.json'):
        raise ValueError('Protocol changed after preparation')
    train_rows=read_rows(prepared/'train.jsonl');dev=read_rows(prepared/'dev.jsonl')
    require_training_split(train_rows+dev)
    norm=json.loads((prepared/'normalization.json').read_text())
    mean=rearrange(np.array(norm['mean'],dtype=np.float32),'c -> c 1')
    std=rearrange(np.array(norm['std'],dtype=np.float32),'c -> c 1')
    arrays={r['recording_id']:(verified_features(r,prepared)-mean)/std for r in train_rows+dev}
    catalog=json.loads((prepared/'catalog.json').read_text())
    baseline=ROOT/'.cache/semifinal-v1/longer/seed-71/selected.pt'
    baseline_selection=ROOT/'.cache/semifinal-v1/longer/selection-before-test.json'
    selected=json.loads(baseline_selection.read_text())
    if Path(selected['winner']['checkpoint']).as_posix()!=baseline.relative_to(ROOT).as_posix():
        # Windows paths are serialized with backslashes in the historical receipt.
        if selected['winner']['checkpoint'].replace('\\','/')!=baseline.relative_to(ROOT).as_posix():
            raise ValueError('Baseline differs from original development selection')
    initial=torch.load(baseline,map_location='cpu',weights_only=True)['weights']
    probe=CausalVCM();probe.load_state_dict(initial,strict=True)
    if sum(p.numel() for p in probe.parameters())!=113064:
        raise ValueError('Baseline architecture mismatch')
    output.mkdir(parents=True)
    run_protocol={**protocol,'initialization':'continuation_of_project_trained_semifinal',
        'baseline_sha256':file_hash(baseline),'baseline_selection_sha256':file_hash(baseline_selection),
        'prepared_freeze_sha256':file_hash(prepared/'freeze.json'),
        'selection_epochs':[4,8,12,16,20,24,28,32,36,40],
        'batch_size':64,'learning_rate':.001,'weight_decay':.0001,'augmentation':'none'}
    write_json(output/'protocol.json',run_protocol)
    write_json(output/'normalization.json',norm)
    write_json(output/'catalog.json',catalog)
    if not torch.cuda.is_available():
        raise RuntimeError('Declared CUDA device unavailable')
    torch.set_num_threads(4)
    history=[];candidates=[]
    weights=sampling_weights(train_rows,.8,1.)
    with ComputeBudget(CACHE/'compute-budget.json',output) as budget:
        # Compare against the previous pipeline's fitted normalization, not its
        # historical score on a different corpus. The finite catalog is shared.
        old_recipe=json.loads((ROOT/'.cache/semifinal-v1/longer/bundle/recipe.json').read_text())
        old_mean=rearrange(np.array(old_recipe['frontend']['mean'],dtype=np.float32),'c -> c 1')
        old_std=rearrange(np.array(old_recipe['frontend']['std'],dtype=np.float32),'c -> c 1')
        baseline_arrays={r['recording_id']:((arrays[r['recording_id']]*std+mean)-old_mean)/old_std for r in dev}
        probe=probe.to('cuda')
        baseline_scores,baseline_predictions=development(probe,dev,baseline_arrays,catalog['entries'],'cuda',budget.check)
        write_json(output/'baseline-development.json',dict(metrics=baseline_scores,predictions=baseline_predictions,
            checkpoint_sha256=file_hash(baseline),normalization_sha256=identity(old_recipe['frontend'])))
        del baseline_arrays,probe
        torch.cuda.empty_cache()
        for seed in protocol['training_seeds']:
            torch.manual_seed(seed);np_rng=np.random.default_rng(seed)
            model=CausalVCM().to('cuda');model.load_state_dict(initial)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            seed_dir=output/f'seed-{seed}';seed_dir.mkdir()
            for epoch in range(1,41):
                budget.start_epoch(f'{seed}/{epoch}')
                model.train();losses=[]
                order=np_rng.choice(len(train_rows),size=len(train_rows),p=weights,replace=True)
                for offset in range(0,len(order),64):
                    budget.check()
                    batch=[train_rows[int(i)] for i in order[offset:offset+64]]
                    x,lengths=batch_arrays([arrays[r['recording_id']] for r in batch])
                    logits,_=model(torch.from_numpy(x).to('cuda'))
                    targets=torch.tensor([p for r in batch for p in r['phones']],device='cuda')
                    loss=torch.nn.functional.ctc_loss(rearrange(logits.log_softmax(1),'b c t -> t b c'),
                        targets,torch.tensor(lengths),torch.tensor([len(r['phones']) for r in batch]),zero_infinity=False)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite CTC loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    losses.append(float(loss.detach()))
                checkpoint=seed_dir/f'epoch-{epoch:03}.pt'
                torch.save(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,
                    epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np_rng.bit_generator.state,protocol_sha256=identity(run_protocol)),checkpoint)
                record=dict(seed=seed,epoch=epoch,loss=float(np.mean(losses)),checkpoint=str(checkpoint),
                            checkpoint_sha256=file_hash(checkpoint))
                if epoch in run_protocol['selection_epochs']:
                    scores,predictions=development(model,dev,arrays,catalog['entries'],'cuda',budget.check)
                    record.update(scores)
                    path=seed_dir/f'dev-{epoch:03}.json'
                    write_json(path,predictions)
                    record.update(predictions=str(path),predictions_sha256=file_hash(path))
                    candidates.append(dict(record))
                history.append(record);write_json(output/'history.json',history)
                print(json.dumps(record),flush=True)
    selection=dict(winner=select_checkpoint(candidates),candidates=candidates,
        test_accessed=False,calibration_accessed=False,synthetic_only=True,
        protocol_sha256=file_hash(output/'protocol.json'))
    write_json(output/'selection.json',selection)
    selected_rows=json.loads(Path(selection['winner']['predictions']).read_text())
    before={r['recording_id']:r for r in baseline_predictions}
    gains=[r['recording_id'] for r in selected_rows if r['raw_exact'] and not before[r['recording_id']]['raw_exact']]
    losses=[r['recording_id'] for r in selected_rows if not r['raw_exact'] and before[r['recording_id']]['raw_exact']]
    write_json(output/'development-improvement.json',dict(baseline=baseline_scores,
        selected={k:selection['winner'][k] for k in baseline_scores},gained=gains,lost=losses,
        baseline_predictions_sha256=file_hash(output/'baseline-development.json'),
        selected_predictions_sha256=selection['winner']['predictions_sha256']))
    return selection
