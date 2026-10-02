"""Versioned, integrity-enforced qualification of unchanged trained candidates."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scripts.runtime.pcv_schema import INTENTS, canonical, normalize_text, output
from scripts.train.pcv_public_data import ROOT, STUDY, CACHE, sha, save, read_rows
from scripts.train.pcv_calibration import CHILD, WORK, freeze_child
from scripts.test.pcv_public_eval import accepted_candidate as parent_candidate, score


def verify_bindings(root, bindings):
    for path, expected in bindings.items():
        if sha(Path(root)/path) != expected:
            raise ValueError('Frozen identity mismatch: ' + str(path))


def precise_metrics(records, key='candidate'):
    records = list(records)
    positives = [r for r in records if r['expected'] is not None]
    negatives = [r for r in records if r['expected'] is None]
    exact = lambda r: r[key] is not None and canonical(r[key]) == canonical(r['expected'])
    correct = sum(exact(r) for r in positives)
    accepted = sum(r[key] is not None for r in records)
    fa = sum(r[key] is not None for r in negatives)
    overall = {'n':len(records), 'positive_n':len(positives), 'negative_n':len(negatives), 'exact':correct,
               'accepted':accepted, 'precision':correct/accepted if accepted else None,
               'recall':correct/len(positives) if positives else None, 'negative_errors':fa,
               'false_accept_rate':fa/len(negatives) if negatives else None}
    per_intent = {}
    for intent in INTENTS:
        refs = [r for r in records if r['expected'] and r['expected']['intent']==intent]
        preds = [r for r in records if r[key] and r[key]['intent']==intent]
        full_tp = sum(exact(r) for r in refs)
        intent_tp = sum(bool(r['expected'] and r['expected']['intent']==intent) for r in preds)
        per_intent[intent] = {'reference_n':len(refs), 'predicted_n':len(preds), 'complete_correct':full_tp,
            'intent_correct':intent_tp, 'complete_precision':full_tp/len(preds) if preds else None,
            'complete_recall':full_tp/len(refs) if refs else None,
            'intent_precision':intent_tp/len(preds) if preds else None,
            'intent_recall':intent_tp/len(refs) if refs else None}
    return {'overall':overall, 'per_intent':per_intent}


def accepted_candidate(record, policy):
    candidate = parent_candidate(record, policy)
    if candidate is not None and candidate['intent'] in {'LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR'}:
        named = candidate['slots']['light_id'] != 'light.all'
        p = record['target_named_probability']
        if (named and p < .9) or (not named and p > .1):
            return None
    return candidate


def decision(record, policy):
    if not policy['enabled']:
        return output(None, 'unqualified_model')
    candidate = accepted_candidate(record, policy)
    if candidate is not None:
        return output(candidate)
    intent = INTENTS[record['intent_index']-1] if record['intent_index'] else 'OOV'
    confident = record['top_intent_probability'] >= max(.7, policy['min_intent_probability'])
    needs_argument = intent in {'LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR','CALL','CREATE_REMINDER','TIMER','ALARM','TEMPERATURE','PLAY_MUSIC','VOLUME'}
    return output(None, 'invalid_argument' if confident and needs_argument else 'OOV')


def verify_parent():
    frozen = json.loads((STUDY/'training-freeze.json').read_text())
    verify_bindings(ROOT, frozen['source_hashes'])
    verify_bindings(ROOT, {str(STUDY.relative_to(ROOT)/'protocol.json'):frozen['protocol_sha256'],
                          str(STUDY.relative_to(ROOT)/'data-freeze.json'):frozen['data_freeze_sha256']})
    data = frozen['data']
    verify_bindings(ROOT, {entry['path']:entry['sha256'] for entry in data['manifests'].values()})
    verify_bindings(ROOT, {str(STUDY.relative_to(ROOT)/'catalog.json'):data['catalog_sha256'],
        str(STUDY.relative_to(ROOT)/'data-audit.json'):data['audit_sha256'],
        str(CACHE.relative_to(ROOT)/'prepared/normalization.json'):data['normalization_sha256']})
    verify_bindings(ROOT, frozen['installed_artifact_hashes'])
    return frozen


def verify_rows(rows):
    for i, row in enumerate(rows, 1):
        if sha(row['features']) != row['features_sha256'] or sha(row['audio']) != row['audio_sha256']:
            raise ValueError('Evaluation feature/audio identity mismatch: ' + row['id'])
        if i % 5000 == 0:
            print('VERIFIED_EVALUATION_RECORDS', i, '/', len(rows), flush=True)


def calibration_policy(records):
    values = [r['score'] for r in records if r['score'] is not None and r['candidate'] is not None]
    scores = sorted(set([-100.] + [float(x) for x in np.quantile(values, np.linspace(0,1,11))])) if values else [0.]
    present = {r['expected']['intent'] for r in records if r['expected']}
    missing = sorted(set(INTENTS)-present)
    best, best_key, eligible = None, None, []
    # Cache semantic equality; candidate identity never changes while thresholds are searched.
    correct = np.asarray([bool(r['expected']) and canonical(r['candidate'])==canonical(r['expected']) for r in records])
    positive = np.asarray([r['expected'] is not None for r in records])
    real = np.asarray([not r['synthetic'] for r in records])
    for threshold in scores:
        for margin in (0.,.002,.005,.01,.02,.05,.1):
            for probability in (0.,.5,.7,.9):
                policy = {'min_score':threshold,'min_margin':margin,'min_intent_probability':probability}
                accepted = np.asarray([accepted_candidate(r,policy) is not None for r in records])
                strata = {}
                passing = not missing
                for label, mask in [('human',real),('synthetic',~real)]:
                    p = int(np.sum(positive & mask)); n = int(np.sum(~positive & mask))
                    a = int(np.sum(accepted & mask)); c = int(np.sum(accepted & correct & mask)); e = int(np.sum(accepted & ~positive & mask))
                    precision, recall, far = c/a if a else 0., c/p if p else 0., e/n if n else 1.
                    strata[label] = {'positive_n':p,'negative_n':n,'accepted':a,'exact':c,'negative_errors':e,
                                     'precision':precision,'recall':recall,'false_accept_rate':far}
                    passing &= bool(p and n and precision>=.99 and recall>=.90 and far<=.01)
                safe = all(m['precision']>=.99 and m['false_accept_rate']<=.01 for m in strata.values())
                key = (int(safe), min(m['recall'] for m in strata.values()), sum(m['recall'] for m in strata.values()),
                       min(m['precision'] for m in strata.values()))
                item = {**policy, 'calibration_strata':strata}
                if best_key is None or key > best_key:
                    best, best_key = item, key
                if passing:
                    eligible.append((key,item))
    chosen = max(eligible,key=lambda pair:pair[0])[1] if eligible else best
    return {**chosen,'enabled':bool(eligible),'missing_positive_intents':missing,
            'reason':'calibration_gates_passed' if eligible else 'calibration_quality_gates_failed',
            'grid_size':len(scores)*7*4, 'test_used':False}


def report_metrics(records):
    return {'all':precise_metrics(records),
        'human':precise_metrics(r for r in records if not r['synthetic']),
        'synthetic':precise_metrics(r for r in records if r['synthetic']),
        'per_source':{source:precise_metrics(r for r in records if r['source']==source) for source in sorted({r['source'] for r in records})}}


def evaluate():
    import torch
    import onnxruntime as ort
    from scripts.models.pcv_residual import ResidualVCM, export_onnx
    child_protocol = freeze_child()
    verify_parent()
    selection_path = STUDY/'selection-before-test.json'
    selection = json.loads(selection_path.read_text())
    extension = json.loads((CHILD/'calibration-freeze.json').read_text())
    verify_bindings(ROOT, {extension['manifest']:extension['sha256']})
    if extension['generation_protocol_sha256'] != sha(CHILD/'protocol.json') or extension['parent_test_sha256'] != sha(CACHE/'prepared/test.jsonl'):
        raise ValueError('Calibration lineage identity mismatch')
    cal = read_rows(ROOT/extension['manifest'])
    test = read_rows(CACHE/'prepared/test.jsonl')
    verify_rows(cal+test)
    save(CHILD/'before-inference.json', {'selection_sha256':sha(selection_path), 'qualification_protocol_sha256':sha(CHILD/'protocol.json'),
        'calibration_freeze_sha256':sha(CHILD/'calibration-freeze.json'), 'audio_feature_hashes_verified':len(cal)+len(test),
        'all_parent_bindings_verified':True, 'test_inference_accessed':False})
    norm = json.loads((CACHE/'prepared/normalization.json').read_text())
    catalog = json.loads((CHILD/'catalog.json').read_text())
    baseline = json.loads((STUDY/'baseline.json').read_text())
    test_by_id = {r['id']:r for r in test}
    results = {}
    for arm, selected in selection['arms'].items():
        directory = WORK/'evaluation'/arm
        if directory.exists():
            raise FileExistsError('Preserve existing evaluation output; choose a new receipt run')
        directory.mkdir(parents=True)
        checkpoint = ROOT/selected['checkpoint']
        verify_bindings(ROOT, {selected['checkpoint']:selected['checkpoint_sha256']})
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        model = ResidualVCM(**payload['model_config'])
        model.load_state_dict(payload['weights'])
        torch.set_num_threads(2)
        bundle = directory/'bundle'
        bundle.mkdir()
        export = export_onnx(model,bundle/'model.onnx')
        save(directory/'export.json',export)
        if not export['pass']:
            raise ValueError('Trained ONNX graph failed parity/size gate; receipt preserved')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(bundle/'model.onnx'),options,providers=['CPUExecutionProvider'])
        calibration = score(session,cal,norm,catalog['entries'])
        save(directory/'calibration-predictions.json',calibration)
        policy = calibration_policy(calibration)
        binding = {'model_sha256':sha(bundle/'model.onnx'),'catalog_sha256':sha(CHILD/'catalog.json'),
                   'custom_slots_sha256':sha(CHILD/'custom-slots.json'),'normalization_sha256':sha(CACHE/'prepared/normalization.json')}
        policy['binding'] = binding
        save(bundle/'policy.json',policy)
        for source in [CHILD/'catalog.json',CHILD/'custom-slots.json',CACHE/'prepared/normalization.json',STUDY/'output.schema.json']:
            shutil.copyfile(source,bundle/source.name)
        lock = {'checkpoint':selected,'binding':binding,'policy_sha256':sha(bundle/'policy.json'),
            'qualification_protocol_sha256':sha(CHILD/'protocol.json'),'test_manifest_sha256':sha(CACHE/'prepared/test.jsonl'),
            'calibration_predictions_sha256':sha(directory/'calibration-predictions.json'),'test_inference_accessed':False}
        save(directory/'selection-before-test.json',lock)
        verify_parent()
        freeze_child()
        predicted = score(session,test,norm,catalog['entries'])
        for row in predicted:
            row['policy_candidate'] = accepted_candidate(row,policy)
            row['output'] = decision(row,policy)
            row['accepted_command'] = row['output']['command']
        save(directory/'test-predictions.json',predicted)
        cells = []
        for case in baseline:
            text = normalize_text(case.get('spoken_text',case['text']))
            matching = [r for r in predicted if r['text']==text and canonical(r['expected'])==canonical(case['command'])]
            cells.append({'id':case['id'],'text':case['text'],'expected':case['command'],'n':len(matching),
                'raw_exact':sum(canonical(r['candidate'])==canonical(case['command']) for r in matching),
                'accepted_exact':sum(canonical(r['accepted_command'])==canonical(case['command']) for r in matching),
                'strata':{s:{'n':sum(r['source']==s for r in matching),'raw_exact':sum(r['source']==s and canonical(r['candidate'])==canonical(case['command']) for r in matching)} for s in sorted({r['source'] for r in matching})}})
        raw = report_metrics(predicted)
        accepted = {name:precise_metrics(rows,'accepted_command') for name,rows in
                    [('all',predicted),('human',[r for r in predicted if not r['synthetic']]),('synthetic',[r for r in predicted if r['synthetic']])]}
        point = {name:precise_metrics(rows,'policy_candidate') for name,rows in
                 [('all',predicted),('human',[r for r in predicted if not r['synthetic']]),('synthetic',[r for r in predicted if r['synthetic']])]}
        sapi_baseline = [r for r in predicted if r['source']=='SAPI' and (r.get('phrase_id') or '').startswith('B1-')]
        slots = defaultdict(lambda: {'n':0,'raw_correct':0,'accepted_correct':0})
        negative_classes = defaultdict(list)
        for row in predicted:
            if row['expected'] is None:
                negative_classes[test_by_id[row['id']]['reference_class']].append(row)
            else:
                for role,value in row['expected']['slots'].items():
                    counter = slots[row['source']+'/'+row['expected']['intent']+'/'+role]
                    counter['n'] += 1
                    for field,metric in [('candidate','raw_correct'),('accepted_command','accepted_correct')]:
                        actual = row[field]
                        counter[metric] += int(actual is not None and actual['intent']==row['expected']['intent'] and canonical(actual['slots'].get(role))==canonical(value))
        result = {'version': 'qualification1', 'arm':arm,'selected_checkpoint':selected,'export':export,'policy':policy,
            'raw':raw,'accepted':accepted,'diagnostic_policy_point':point,'baseline_cells':cells,
            'sapi_baseline_raw':precise_metrics(sapi_baseline),'slots':dict(slots),
            'negative_classes_raw':{k:precise_metrics(v)['overall'] for k,v in negative_classes.items()},
            'qualification_pass':False,'qualification_limitations':['SAPI has only one test voice; prescribed multi-engine baseline evidence absent',
                'Only 21/93 exact baseline phrases have any original human evidence','Generated spoken text is not independently listened',
                'No physical Pi or live-microphone evidence','No installed bundle changed'],
            'test_selected_model_or_thresholds':False}
        save(CHILD/f'results-{arm}.json',result)
        save(bundle/'recipe.json',{'schema_version':3,'status':'research_unqualified','state_shapes':model.state_shapes,
            'heads':['ctc','intent','target'],'binding':binding,'selection_lock_sha256':sha(directory/'selection-before-test.json'),
            'policy_enabled':policy['enabled'],'installation_authorized':False,'release_qualified':False})
        results[arm] = {'human_raw':raw['human']['overall'],'synthetic_raw':raw['synthetic']['overall'],
            'sapi_baseline_raw':result['sapi_baseline_raw']['overall'],'policy_enabled':policy['enabled'],
            'human_policy_point':point['human']['overall'],'synthetic_policy_point':point['synthetic']['overall']}
        print('QUALIFICATION_RESULT',arm,json.dumps(results[arm]),flush=True)
    verify_parent()
    save(CHILD/'comparison.json',{'development_selected_arm':selection['winner'],'arms':results,'qualified':False,
        'test_used_for_selection':False,'parent_bindings_preserved':True})
