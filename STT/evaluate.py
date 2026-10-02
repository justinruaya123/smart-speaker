"""Held-out STT, strict numbers and frozen-catalog/dynamic-catalog ablations."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from common import ROOT, read_jsonl, write_json, write_jsonl, extract_number, parse_number, say_number, edit_distance, digest
from model import AcousticCTC, greedy_decode
from train import collate, number_compositions, SEED
from catalog import TYPES, build_catalogs, parse_music_request, rank_fields, accept_fields, score_candidates

def transcription_metrics(rows):
    words=sum(len(r['reference'].split()) for r in rows)
    chars=sum(len(r['reference']) for r in rows)
    return {'n':len(rows),'wer':sum(edit_distance(r['reference'].split(),r['hypothesis'].split()) for r in rows)/max(1,words),
            'cer':sum(edit_distance(r['reference'],r['hypothesis']) for r in rows)/max(1,chars),
            'exact':sum(r['reference']==r['hypothesis'] for r in rows)/max(1,len(rows))}

def entity_metrics(rows,mode,threshold=0,margin=0,subset=None):
    tp=fp=fn=total=exact=accepted=0
    for row in rows:
        refs=row['references']
        if subset is not None and row['category']!=subset:continue
        pred=row['raw_fields'] if mode=='off' else accept_fields(row[mode],threshold,margin)
        refset=set(refs.items());predset=set(pred.items())
        tp+=len(refset&predset);fp+=len(predset-refset);fn+=len(refset-predset)
        total+=1;exact+=refset==predset;accepted+=bool(predset)
    return {'n':total,'correct_entities':tp,'wrong_entities':fp,'missed_entities':fn,
        'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),
        'exact_frame_accuracy':exact/max(1,total),'accepted_requests':accepted}

def choose_threshold(dev):
    trials=[]
    for threshold in [.45,.55,.65,.75,.80,.85,.90,.95]:
        for margin in [0,.03,.06,.10,.15]:
            metrics=entity_metrics(dev,'dynamic',threshold,margin)
            trials.append({'threshold':threshold,'margin':margin,**metrics})
    feasible=[t for t in trials if t['precision']>=.90 and t['correct_entities']>=5]
    if feasible:
        chosen=max(feasible,key=lambda t:(t['recall'],t['precision'],t['threshold']))
        reason='Highest development recall at >=90% entity precision, with >=5 correct entities.'
    else:
        chosen=max(trials,key=lambda t:(t['f1'],t['precision'],t['threshold']))
        reason='Development 90% precision gate unmet; diagnostic best-F1 threshold, not an operational acceptance gate.'
    return chosen,reason,trials

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=ROOT/'runs/scratch_ctc'
    checkpoint_path=out/'commands_best.pt'
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    model=AcousticCTC().to(device);model.load_state_dict(checkpoint['state']);model.eval()
    manifest=read_jsonl(ROOT/'data/manifest.jsonl')
    checkpoint_manifest=checkpoint['config']['manifest_sha256']
    actual_manifest=hashlib.sha256((ROOT/'data/manifest.jsonl').read_bytes()).hexdigest()
    if checkpoint_manifest!=actual_manifest:raise RuntimeError('Evaluation manifest differs from the training manifest')
    fixed,dynamic,added=build_catalogs(manifest)
    catalog_dir=ROOT/'catalogs'
    write_json(catalog_dir/'fixed.json',fixed)
    write_json(catalog_dir/'dynamic_fixture.json',dynamic)
    write_json(catalog_dir/'added_fixture.json',added)
    write_json(catalog_dir/'protocol.json',{'provenance':'Simulated user-supplied catalog: training entities plus deterministic half of unseen dev/test entity names.',
        'restriction':'Entity names only, not audio or full reference sentences. Optimistic oracle catalog membership simulation, not a real playlist evaluation.',
        'fixed_entries':len(fixed),'dynamic_entries':len(dynamic),'added_entries':len(added),
        'acoustic_retraining':False,'spotify_integration':False,'filepath_integration':False})
    evaluation=[r for r in manifest if r['split'] in {'dev','test'}]
    predictions=[];music=[];latencies=[]
    fixed_set=set(map(tuple,fixed));dynamic_set=set(map(tuple,dynamic))
    for split in ['dev','test']:
        selected=[r for r in evaluation if r['split']==split]
        for start in range(0,len(selected),48):
            batch=selected[start:start+48]
            items=[(torch.from_numpy(np.load(ROOT/r['feature']).astype(np.float32)),r['text']) for r in batch]
            x,l,_,_=collate(items)
            logits,ll=model(x.to(device),l)
            decoded=greedy_decode(logits,ll)
            probs=logits.log_softmax(-1)
            for i,(r,hyp) in enumerate(zip(batch,decoded)):
                prediction={'id':r['id'],'source':r['source'],'split':split,'reference':r['text'],'hypothesis':hyp,
                    'intent':r.get('intent'),'speaker':r['speaker'],'reference_number':extract_number(r['text']),
                    'predicted_number':extract_number(hyp),'kind':r['kind']}
                predictions.append(prediction)
                refs={e['type']:e['text'] for e in r.get('entities',[]) if e['type'] in TYPES}
                # Include all annotated music entities, plus a deterministic negative
                # sample of natural requests without these entities.
                negative=int(digest(r['id'])[:8],16)%6==0
                if not refs and not negative:continue
                lp=probs[i,:int(ll[i])]
                raw={k:v['text'] for k,v in parse_music_request(hyp).items()}
                start_time=time.perf_counter()
                ranked_fixed=rank_fields(hyp,lp,fixed)
                ranked_dynamic=rank_fields(hyp,lp,dynamic)
                latency=time.perf_counter()-start_time
                latencies.append(latency)
                if not refs:category='negative'
                elif all((k,v) in fixed_set for k,v in refs.items()):category='known'
                elif all((k,v) in dynamic_set for k,v in refs.items()):category='newly_added'
                else:category='out_of_catalog'
                music.append({'id':r['id'],'split':split,'source':r['source'],'reference':r['text'],'hypothesis':hyp,
                    'references':refs,'raw_fields':raw,'fixed':ranked_fixed,'dynamic':ranked_dynamic,
                    'category':category,'catalog_seconds':latency})
            if start%480==0:print('Evaluation',split,start,'/',len(selected),flush=True)
    dev=[r for r in music if r['split']=='dev']
    test=[r for r in music if r['split']=='test']
    chosen,reason,trials=choose_threshold(dev)
    threshold=chosen['threshold'];margin=chosen['margin']
    write_json(out/'catalog_calibration.json',{'selected':chosen,'reason':reason,'trials':trials,'selection_split':'dev'})
    catalog_metrics={mode:{subset:entity_metrics(test,mode,threshold,margin,None if subset=='all' else subset)
        for subset in ['all','known','newly_added','out_of_catalog','negative']} for mode in ['off','fixed','dynamic']}
    for r in music:
        r['accepted_fixed']=accept_fields(r['fixed'],threshold,margin)
        r['accepted_dynamic']=accept_fields(r['dynamic'],threshold,margin)
    heldout=[r for r in predictions if r['split']=='test']
    numeric={}
    groups={
        'isolated_human_numbers':[r for r in heldout if r['kind']=='human_word' and r['reference_number'] is not None],
        'natural_volume_commands':[r for r in heldout if 'volume' in (r.get('intent') or '').lower() and r['reference_number'] is not None],
        'natural_light_number_transfer':[r for r in heldout if r['source']=='SnipsSmartLights' and r['reference_number'] is not None],
        'other_natural_numbers':[r for r in heldout if r['kind']!='human_word' and r['reference_number'] is not None],
    }
    for name,rr in groups.items():
        by_value={str(v):{'n':sum(r['reference_number']==v for r in rr),'correct':sum(r['reference_number']==v and r['predicted_number']==v for r in rr)} for v in range(101)}
        numeric[name]={'n':len(rr),'correct':sum(r['reference_number']==r['predicted_number'] for r in rr),
            'exact_value_accuracy':sum(r['reference_number']==r['predicted_number'] for r in rr)/max(1,len(rr)),
            'values_covered':sorted({r['reference_number'] for r in rr}),
            'missing_values':[v for v in range(101) if not any(r['reference_number']==v for r in rr)],'per_value':by_value}
    by_source={source:transcription_metrics([r for r in heldout if r['source']==source]) for source in sorted({r['source'] for r in heldout})}
    results={'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),'selected_epoch':checkpoint['epoch'],
        'test_transcription':transcription_metrics(heldout),'test_by_source':by_source,'catalog':catalog_metrics,
        'catalog_threshold':threshold,'catalog_margin':margin,'calibration_reason':reason,
        'catalog_sizes':{'fixed':len(fixed),'dynamic':len(dynamic),'additions':len(added)},
        'catalog_both_conditions_latency_ms_median':float(np.median(latencies)*1000),
        'numeric':numeric,'text_grammar_roundtrips_0_to_100':sum(parse_number(say_number(n))==n for n in range(101)),
        'test_speakers':len({r['speaker'] for r in heldout}),
        'limitations':['SLURP text is the expected prompted transcript, not independently re-transcribed here.',
                       'Catalog addition identities are derived from held-out entity metadata to simulate user-supplied titles.',
                       'No physical Spotify control, microphone, Raspberry Pi, or Philippine English acceptance test.']}
    write_jsonl(out/'predictions.jsonl',predictions)
    write_jsonl(out/'catalog_predictions.jsonl',music)
    write_json(out/'evaluation.json',results)
    print(json.dumps({k:v for k,v in results.items() if k!='numeric'},indent=2),flush=True)

if __name__=='__main__':main()
