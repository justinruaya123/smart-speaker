"""Independent reconciliation of tiny-model membership, metrics and identities."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.runtime.optionb_contract import COMMANDS, class_index
from scripts.runtime.pcv_schema import baseline_cases, normalize_text
from scripts.test.speaker_demo_suite import read_json, read_jsonl, read_audio, file_digest, pcm_digest, write_new
from scripts.train.speaker_optionb_tiny import check_bindings, extract_arrays, array_digest

STUDY = ROOT / 'experiments/speaker-optionb-tiny-v1'


def metrics(rows, field):
    tp = sum(r['class_index'] > 0 and r[field] == r['class_index'] for r in rows)
    fp = sum(r[field] > 0 and r[field] != r['class_index'] for r in rows)
    fn = sum(r['class_index'] > 0 and r[field] != r['class_index'] for r in rows)
    negative = [r for r in rows if r['class_index'] == 0]
    nfa = sum(r[field] > 0 for r in negative)
    return dict(rows=len(rows), positives=tp + fn, negatives=len(negative), tp=tp, fp=fp, fn=fn,
        correct_accepted=tp, wrong_accepted=fp, precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        negative_false_accepts=nfa, negative_false_accept_rate=nfa/len(negative) if negative else None)


def main():
    protocol, freeze = read_json(STUDY/'protocol.json'), read_json(STUDY/'data-freeze.json')
    run = ROOT / protocol['outputs']['run']
    lock, published = read_json(STUDY/'selection-before-test.json'), read_json(STUDY/'results.json')
    for bindings in (freeze['bindings'], freeze['historical_bindings'], lock['bindings']):
        check_bindings(bindings)
    views = {s: read_jsonl(run/f'{s}.jsonl') for s in ('train','dev','calibration','test','holdout')}
    reserved = [r for s in ('dev','calibration','test','holdout') for r in views[s]]
    for field in ('id','sha256','pcm_sha256'):
        assert not ({r[field] for r in views['train']} & {r[field] for r in reserved})
    for field in ('speaker','pair','source_speaker_key','session_id','parent_recording_id','speaker_alias_group'):
        left = {str(r[field]) for r in views['train'] if r['kind']=='human' and r.get(field) not in (None,'','unknown')}
        right = {str(r[field]) for r in reserved if r['kind']=='human' and r.get(field) not in (None,'','unknown')}
        assert not left & right, field
    norm = read_json(run/'normalization.json')
    human = [r for r in views['train'] if r['kind']=='human']
    assert set(norm['ids']) == {r['id'] for r in human} and len(norm['ids']) == len(human)
    raw = read_jsonl(run/'raw-feature-inputs.jsonl')
    assert len(raw) == len(views['train'])+len(views['dev'])
    assert {r['id'] for r in raw} == {r['id'] for r in [*views['train'],*views['dev']]}
    raw_by_id = {r['id']:r for r in raw}
    arrays = extract_arrays(human)
    means, seconds = [], []
    for row in human:
        array = arrays[row['id']]
        assert array_digest(array) == norm['raw_feature_tensor_sha256'][row['id']] == raw_by_id[row['id']]['raw_feature_tensor_sha256']
        values = array.astype(np.float64)
        means.append(values.mean(axis=1));seconds.append(np.square(values).mean(axis=1))
    mean = np.mean(means,axis=0)
    std = np.sqrt(np.maximum(np.mean(seconds,axis=0)-mean**2,1e-8))
    np.testing.assert_array_equal(np.asarray(norm['mean'],dtype=np.float32),mean.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(norm['std'],dtype=np.float32),std.astype(np.float32))
    result = read_json(run/'training-result.json')
    config = protocol['training']
    assert result['optimizer_steps'] == (config['human_epochs']+config['mixed_epochs'])*config['steps_per_epoch']
    assert result['weights_changed'] and result['parameters']==258912 and result['fitting_splits']==['train']
    assert set(result['sample_counts']) <= {r['id'] for r in views['train']}
    assert sum(result['sample_counts'].values()) == result['optimizer_steps']*config['batch_size']
    policy_rows = read_jsonl(run/'calibration-predictions.jsonl')
    assert {r['id'] for r in policy_rows} == {r['id'] for s in ('dev','calibration') for r in views[s] if r['kind']=='human'}
    assert not {r['id'] for r in policy_rows} & {r['id'] for s in ('test','holdout') for r in views[s]}
    records = read_jsonl(run/'final-predictions.jsonl')
    parents = {r['id']:r for s in ('test','holdout') for r in views[s]}
    eligible_holdout = [r for r in views['holdout'] if r['kind']=='human' and r['metric_eligible']]
    assert len(records)==len(parents)+3*len(eligible_holdout)==published['prediction_rows']
    assert len({r['id'] for r in records})==len(records)
    assert file_digest(run/'final-predictions.jsonl')==published['predictions_sha256']
    for row in records:
        parent = parents[row['parent_id']]
        assert row['class_index']==class_index(parent['expected'])
        assert row['expected']==parent['expected'] and row['scope']==parent['scope']
        assert row['metric_eligible']==parent['metric_eligible'] and row['kind']==parent['kind'] and row['split']==parent['split']
        assert row['source_audio_sha256']==parent['sha256'] and row['source_pcm_sha256']==parent['pcm_sha256']
        if row['variant']=='original':assert row['actual_input_pcm_sha256']==parent['pcm_sha256']
        else:
            assert parent in eligible_holdout
            pcm=read_audio(parent);z=np.zeros(16000,dtype=np.float32)
            transformed={'silence_prefix_1s':lambda:np.concatenate([z,pcm]),'silence_suffix_1s':lambda:np.concatenate([pcm,z]),'silence_both_1s':lambda:np.concatenate([z,pcm,z])}[row['variant']]()
            assert row['actual_input_pcm_sha256']==pcm_digest(transformed)
        event=row['event']
        assert event['actions_enabled'] is False and event['qualified_output']['accepted'] is False
        if 'logits' not in event:
            assert row['raw_index']==row['accepted_index']==row['fixed_index']==0
            continue
        logits=np.asarray(event['logits'],dtype=np.float64);assert logits.shape==(32,) and np.isfinite(logits).all()
        p=np.exp(logits-logits.max());p/=p.sum();index=int(p.argmax());margin=float(p[index]-np.partition(p,-2)[-2])
        policy=lock['policy'];accept=bool(policy['enabled'] and index and p[index]>=policy['min_probability'] and margin>=policy['min_margin'])
        assert row['raw_index']==index and row['accepted_index']==(index if accept else 0)
        assert row['fixed_index']==(index if index and p[index]>=.95 else 0)
        assert event['output']['accepted']==accept
        assert event['output']['command']==(COMMANDS[index] if accept else None)
    for key, value in published['results'].items():
        split,kind,variant=key.split('/')
        all_rows=[r for r in records if (r['split'],r['kind'],r['variant'])==(split,kind,variant)]
        rows=[r for r in all_rows if r['metric_eligible']]
        assert value['total_rows']==len(all_rows) and value['ambiguous_unscored']==len(all_rows)-len(rows)
        for arm,field in (('raw','raw_index'),('fixed_095','fixed_index'),('selected','accepted_index')):
            assert value['metrics'][arm]==metrics(rows,field),(key,arm)
    for filename, expected_n in (('per-command.csv',31),('per-intent.csv',18)):
        table=list(csv.DictReader((STUDY/filename).open(encoding='utf-8',newline='')))
        assert len(table)==expected_n*len(published['results'])
        for cell in table:
            split,kind,variant=cell['cohort'].split('/')
            rows=[r for r in records if r['metric_eligible'] and (r['split'],r['kind'],r['variant'])==(split,kind,variant)]
            indices={int(cell['class_index'])} if 'class_index' in cell else {i for i,c in enumerate(COMMANDS) if c and c['intent']==cell['intent']}
            n=sum(r['class_index'] in indices for r in rows)
            tp=sum(r['class_index'] in indices and r['accepted_index']==r['class_index'] for r in rows)
            fp=sum(r['accepted_index'] in indices and r['accepted_index']!=r['class_index'] for r in rows)
            fn=n-tp
            assert [int(cell[k]) for k in ('n','tp','fp','fn')]==[n,tp,fp,fn]
            for field,actual in (('precision',tp/(tp+fp) if tp+fp else None),('recall',tp/n if n else None),('f1',2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)):
                assert cell[field]=='' if actual is None else math.isclose(float(cell[field]),actual,rel_tol=1e-12)
    verbatim=list(csv.DictReader((STUDY/'per-verbatim.csv').open(encoding='utf-8',newline='')))
    assert {r['id'] for r in verbatim}=={r['id'] for r in baseline_cases()} and len(verbatim)==93
    export=read_json(STUDY/'export-verification.json')
    assert export['initializer_elements']<=300000 and export['recorded_dev_calibration_parity_rows']==len(policy_rows)
    receipt=dict(status='verified',parameters=258912,exact_train_normalization_recomputed=True,
        historical_bindings_unchanged=len(freeze['historical_bindings']),prediction_rows=len(records),
        final_metric_groups=len(published['results']),normalization_human_rows=len(human),
        no_fitting_or_normalization_holdout_overlap=True,selection_before_test_sha256=file_digest(STUDY/'selection-before-test.json'),
        results_sha256=file_digest(STUDY/'results.json'))
    if '--write-receipt' in sys.argv:write_new(STUDY/'verification.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
