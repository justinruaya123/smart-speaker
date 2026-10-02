"""Versioned denominator repair over unchanged predictions; no inference."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.pcv_three_engine import HERE,OUTPUT,ENGINES,immutable,verify
from scripts.train.pcv_public_data import sha
from scripts.test.pcv_public_qualification import precise_metrics,verify_bindings
from scripts.runtime.pcv_schema import canonical


def metrics_with_coverage(rows,key='candidate'):
    rows=list(rows)
    result=precise_metrics(rows,key)
    value=result['overall']
    missing=[r for r in rows if not r['present']]
    mp=sum(r['expected'] is not None for r in missing)
    mn=len(missing)-mp
    value.update(measured_n=len(rows)-len(missing),missing_n=len(missing),
        missing_positive=mp,missing_negative=mn,measured_negative_n=value['negative_n']-mn,
        measured_positive_n=value['positive_n']-mp)
    if value['negative_n']:
        value['false_accept_rate_bounds']=[value['negative_errors']/value['negative_n'],
            (value['negative_errors']+mn)/value['negative_n']]
    else:
        value['false_accept_rate_bounds']=None
    if mn:
        value['false_accept_rate']=None
    for intent,metric in result['per_intent'].items():
        absent=sum(r['expected'] is not None and r['expected']['intent']==intent for r in missing)
        metric.update(missing_reference_n=absent,measured_reference_n=metric['reference_n']-absent)
    return result


def compile_report(arm,predicted,prompts,engines=ENGINES):
    known={e+'/'+p['id'] for e in engines for p in prompts}
    by_id={r['id']:r for r in predicted}
    if len(by_id)!=len(predicted) or set(by_id)-known:
        raise ValueError('Unexpected or duplicate prediction cell')
    result={'version':'report-v2','arm':arm,'development_selected':arm=='mixed',
        'test_used_for_selection':False,'qualified':False,'engines':{},'cells':[]}
    for engine in engines:
        rows=[]
        for p in prompts:
            rid=engine+'/'+p['id']
            row=by_id.get(rid)
            if row is not None and canonical(row['expected'])!=canonical(p['command']):
                raise ValueError('Prediction reference differs from frozen prompt')
            r={**(row or {}),'id':rid,'phrase_id':p['id'],'expected':p['command'],'present':row is not None,
                'candidate':row['candidate'] if row else None,
                'policy_candidate':row['policy_candidate'] if row else None,
                'accepted_command':row['accepted_command'] if row else None}
            rows.append(r)
            result['cells'].append({'engine':engine,'phrase_id':p['id'],'text':p.get('spoken_text',p['text']),
                'expected':p['command'],'present':r['present'],'candidate':r['candidate'],
                'raw_exact':r['present'] and canonical(r['candidate'])==canonical(p['command']),
                'accepted_command':r['accepted_command'],'output':row['output'] if row else None})
        baseline=[r for r in rows if r['phrase_id'].startswith('B1-')]
        extras=[r for r in rows if not r['phrase_id'].startswith('B1-')]
        slots={}
        for r in rows:
            if r['expected'] is None:continue
            for role,value in r['expected']['slots'].items():
                counter=slots.setdefault(r['expected']['intent']+'/'+role,{'n':0,'measured_n':0,'missing_n':0,'raw_correct':0,'accepted_correct':0})
                counter['n']+=1;counter['measured_n']+=r['present'];counter['missing_n']+=not r['present']
                for field,key in [('candidate','raw_correct'),('accepted_command','accepted_correct')]:
                    actual=r[field]
                    counter[key]+=int(actual is not None and actual['intent']==r['expected']['intent']
                        and canonical(actual['slots'].get(role))==canonical(value))
        result['engines'][engine]={'raw':metrics_with_coverage(rows),'baseline_raw':metrics_with_coverage(baseline),
            'extra_raw':metrics_with_coverage(extras),'diagnostic_policy_point':metrics_with_coverage(rows,'policy_candidate'),
            'actual_outputs':metrics_with_coverage(rows,'accepted_command'),
            'baseline_actual':metrics_with_coverage(baseline,'accepted_command'),'slots':slots,
            'missing':[r['id'] for r in rows if not r['present']]}
    return result


def main():
    verify()
    protocol=json.loads((HERE/'report-v2-protocol.json').read_text())
    verify_bindings(ROOT,protocol['bindings'])
    prompts=json.loads((HERE/'prompts.json').read_text())
    audit=json.loads((HERE/'data-audit.json').read_text())
    for arm in ('human','mixed'):
        old=json.loads((HERE/f'results-{arm}.json').read_text())
        path=OUTPUT/f'predictions-{arm}.json'
        assert sha(path)==old['predictions_sha256']
        report=compile_report(arm,json.loads(path.read_text()),prompts)
        report.update(predictions_sha256=sha(path),policy_enabled=old['policy_enabled'],
            data_audit_sha256=sha(HERE/'data-audit.json'),complete=audit['complete'],
            reporting_protocol_sha256=sha(HERE/'report-v2-protocol.json'))
        immutable(HERE/f'results-{arm}-v2.json',report)
        print('REPORT_V2',arm,json.dumps({e:r['baseline_raw']['overall'] for e,r in report['engines'].items()}),flush=True)


if __name__=='__main__':main()
