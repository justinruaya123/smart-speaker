"""Independent checks for numeric grammar, batching, and catalog-only updates."""
import hashlib
import json
import time
from difflib import SequenceMatcher
import numpy as np
import torch
from common import ROOT,say_number,parse_number,extract_number,read_jsonl,write_json
from model import AcousticCTC
from catalog import parse_music_request,build_catalogs

def catalog_text_prediction(text,catalog):
    out={}
    for kind,field in parse_music_request(text).items():
        candidates=[(SequenceMatcher(None,field['text'],name,autojunk=False).ratio(),k,name) for k,name in catalog if k==kind or kind=='song_name']
        if candidates:
            score,k,name=max(candidates)
            # Perfect-match text control, no acoustic evaluation and no fuzzy correction.
            if score==1:out[k]=name
    return out

def main():
    checks=[]
    for value in range(101):
        assert parse_number(say_number(value))==value
        assert parse_number(str(value))==value
        checks.append('numeric_'+str(value))
    for text in ['101','minus one','-1','one hundred one','twenty thirteen','one two','fifty point five','50.5','one hundred and one']:
        assert parse_number(text) is None,text
    for text in ['set volume to -1','set volume to 50.5','set volume to twenty and thirty','set volume to one hundred one']:
        assert extract_number(text) is None,text
    assert extract_number('set volume to ninety nine percent')==99
    assert extract_number('volume one hundred')==100
    checks+=['out_of_range_and_ambiguous_rejection','numeric_command_text_parsing']
    torch.set_num_threads(2)
    model=AcousticCTC().eval()
    a=torch.randn(1,80,80);b=torch.randn(1,130,80)
    with torch.inference_mode():
        alone,l=model(a,torch.tensor([80]))
        both=torch.zeros(2,130,80);both[0,:80]=a[0];both[1]=b[0]
        batched,_=model(both,torch.tensor([80,130]))
    diff=float((alone[0]-batched[0,:40]).abs().max())
    interior=float((alone[0,:35]-batched[0,:35]).abs().max())
    assert diff<1e-5,('single versus batch mismatch',diff)
    checks.append('single_batch_padding_invariance')
    manifest=read_jsonl(ROOT/'data/manifest.jsonl')
    fixed,dynamic,added=build_catalogs(manifest)
    fixedset=set(map(tuple,fixed));dynamicset=set(map(tuple,dynamic))
    test=[r for r in manifest if r['split']=='test' and any(e['type'] in {'song_name','artist_name'} for e in r.get('entities',[]))]
    results={}
    for mode,catalog in [('off',None),('fixed',fixed),('dynamic',dynamic)]:
        tp=fp=fn=0;newtp=newtotal=0
        for r in test:
            refs={(e['type'],e['text']) for e in r['entities'] if e['type'] in {'song_name','artist_name'}}
            pred={(k,v['text']) for k,v in parse_music_request(r['text']).items()} if catalog is None else set(catalog_text_prediction(r['text'],catalog).items())
            tp+=len(refs&pred);fp+=len(pred-refs);fn+=len(refs-pred)
            new={x for x in refs if x not in fixedset and x in dynamicset}
            newtp+=len(pred&new);newtotal+=len(new)
        results[mode]={'n':len(test),'correct_entities':tp,'wrong_entities':fp,'missed_entities':fn,
            'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'new_catalog_entities_correct':newtp,'new_catalog_entities_total':newtotal}
    timings={}
    for size in [100,1000,10000]:
        corpus=[{'title':f'example song {i}','artist':f'example artist {i%100}'} for i in range(size)]
        times=[]
        for _ in range(10):
            start=time.perf_counter();index={(r['title'],r['artist']):i for i,r in enumerate(corpus)};times.append((time.perf_counter()-start)*1000)
        timings[str(size)]={'in_memory_index_build_ms_median':float(np.median(times)),'entries':len(index)}
    checkpoint=ROOT/'runs/scratch_ctc/commands_best.pt'
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(ROOT/'evidence/protocol_checks.json',{'checks_passed':len(checks),'numeric_text_cases':101,
        'batch_single_logit_max_difference':diff,'batch_single_interior_max_difference':interior,
        'text_only_oracle_transcript_control':results,'catalog_index_build':timings,
        'checkpoint_sha256_after_catalog_experiments':checkpoint_hash,
        'limitations':'Text-only control uses correct reference transcripts and cannot be reported as speech recognition accuracy. Index timing is an in-memory dictionary microbenchmark, not end-to-end decoder latency.'})
    print(json.dumps({'checks':len(checks),'batch_difference':diff,'text_control':results,'index':timings},indent=2))

if __name__=='__main__':main()
