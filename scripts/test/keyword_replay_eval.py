"""Supplemental frozen-policy replay; never calibrates or changes a model."""
from datetime import datetime,timezone
from collections import Counter
import json
import numpy as np
from einops import pack
from .keyword_data import ROOT,DATA,CACHE,REPORTS,HEADS,rows,dump,digest
from .keyword_runtime import KeywordPipeline
from .synthetic_eval import command_json
from .audio import load_audio


def summarize(records):
    def summarize_subset(subset,emitted_family=None):
        positive=sum(r['reference']['ready']=='READY' for r in subset)
        emitted=[];correct=0;exact=0;negative_fp=0
        for row in subset:
            accepted=[o for o in row['outputs'] if o['status']=='accepted']
            expected=command_json(row['reference']).get('command')
            matches=[o for o in accepted if row['reference']['ready']=='READY' and o['intent']==row['reference']['intent'] and o['command']==expected]
            exact+=int(len(accepted)==1 and len(matches)==1)
            negative_fp+=int(row['reference']['ready']!='READY' and bool(accepted))
            eligible=[o for o in accepted if emitted_family is None or o['intent']==emitted_family]
            emitted.extend(eligible)
            correct+=int(bool(matches) and (emitted_family is None or row['reference']['intent']==emitted_family))
        return {'n':len(subset),'positive_n':positive,'accepted_outputs':len(emitted),'correct_outputs':correct,
                'output_precision':correct/len(emitted) if emitted else None,'exact_commands':exact,
                'exact_command_recall':exact/positive if positive else None,
                'negative_n':len(subset)-positive,'negative_false_accepts':negative_fp,
                'negative_false_accept_rate':negative_fp/(len(subset)-positive) if len(subset)>positive else None}
    per={}
    for family in HEADS['intent']:
        true=[r for r in records if r['reference']['intent']==family]
        recall=summarize_subset(true);precision=summarize_subset(records,family)
        per[family]={**recall,'accepted_outputs':precision['accepted_outputs'],'correct_outputs':precision['correct_outputs'],'output_precision':precision['output_precision']}
    return {'overall':summarize_subset(records),'per_intent':per,
            'segment_counts':dict(Counter(sum(o.get('reason')!='no_speech' for o in r['outputs']) for r in records))}


def wake_boundary_software_check(pipeline,pcm):
    # Non-speech marker tests cropping only; never enters training or benchmark audio.
    class Spy:
        def __init__(self):self.calls=[]
        def predict(self,wave,offset_ms=0.):
            self.calls.append((wave,offset_ms));return {'status':'rejected','reason':'software_fixture'}
    actual=pipeline.recognizer;spy=Spy();pipeline.recognizer=spy
    marker=np.full(8000,.95,dtype=np.float32)
    combined,_=pack([marker,pcm],'*')
    try:
        output=pipeline.process(combined,wake_end_ms=500.)
        reference=pipeline.vad.segment(pcm)
        assert len(spy.calls)==len(reference)
        for (wave,offset),(expected,timeout) in zip(spy.calls,reference):
            assert not timeout
            np.testing.assert_array_equal(wave,expected)
            assert offset>=500.
        assert all(o.get('speech_start_ms',500.)>=500. for o in output)
        return {'passed':True,'wake_end_ms':500,'pre_boundary_marker_excluded':True,'post_boundary_pcm_preserved':True,
                'kind':'software cropping fixture with original speech; not a trained or tested wake detector'}
    finally:pipeline.recognizer=actual


def run():
    for mode in ('mixed','synthetic'):
        out=REPORTS[mode]
        if (out/'endpoint-results.json').exists():raise ValueError('Replay already completed; preserve evidence')
        specification={'created_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':digest(out/'selection-before-test.json'),
            'evaluation_source_sha256':digest(ROOT/'scripts/test/keyword_replay_eval.py'),
            'purpose':'Supplemental deployment-path check of the already frozen policy. Historical presegmented results have been seen; no model, threshold, grammar or selection is changed.',
            'input':'Each original historical test file, mono 16 kHz. Wake boundary zero because no annotated wake words exist in these files.',
            'exact_rule':'Exactly one accepted command matching the reference typed JSON; extra accepted segments make utterance incorrect.',
            'precision_rule':'At most one correct output per reference; duplicates and wrong accepted outputs count as errors.',
            'no_calibration':True,'no_live_microphone':True,'no_wake_detector':True}
        dump(out/'endpoint-protocol-before-inference.json',specification)
        pipeline=KeywordPipeline(mode);result={'protocol':specification,'tests':{}}
        for source in ('human','synthetic'):
            data=[r for r in rows(DATA/(source+'.jsonl')) if r['split']=='test']
            expected=json.loads((CACHE/mode/(source+'-test-predictions.json')).read_text())
            direct=[];endpoint=[];disagreements=[]
            for i,(row,saved) in enumerate(zip(data,expected)):
                assert row['id']==saved['id']
                pcm,_=load_audio(ROOT/row['audio'])
                answer=pipeline.recognizer.predict(pcm)
                same_accept=(answer['status']=='accepted')==saved['accepted']
                same_command=not saved['accepted'] or answer.get('command')==saved['command']['command']
                if not (same_accept and same_command):disagreements.append(row['id'])
                direct.append({'id':row['id'],'reference':row['labels'],'outputs':[answer]})
                endpoint.append({'id':row['id'],'reference':row['labels'],'outputs':pipeline.process(pcm)})
                if i%400==0:print('REPLAY',mode,source,i,'/',len(data),flush=True)
            if source=='human':
                fixture=next(r for r in rows(DATA/'human.jsonl') if r['split']=='dev' and r['labels']['intent']=='POWER_ON')
                pcm,_=load_audio(ROOT/fixture['audio']);result['wake_boundary_software_check']=wake_boundary_software_check(pipeline,pcm)
            result['tests'][source]={'presegmented_runtime':summarize(direct),'vad_endpoint_runtime':summarize(endpoint),
                'batched_vs_single_recording_decision_mismatches':disagreements}
            dump(CACHE/mode/(source+'-endpoint-predictions.json'),endpoint)
            dump(CACHE/mode/(source+'-direct-runtime-predictions.json'),direct)
        dump(out/'endpoint-results.json',result)
        print('REPLAY RESULT',mode,json.dumps({s:v['vad_endpoint_runtime']['overall'] for s,v in result['tests'].items()}),flush=True)
