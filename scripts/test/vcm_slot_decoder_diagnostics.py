"""Post-hoc E19 failure accounting and text-index checks; never fits a policy."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/vcm_slot_decoder'))
import run  # establishes only the isolated import environment
from collections import defaultdict
import json
import numpy as np
from vcm_slot_decoder import EmbeddingModel, catalog_fingerprint, decision
from vcm_slot_decoder_eval import EXP, CACHE, read, write, validate_frozen, digest
from tinyvcm.parameter_study_data import phones
from threadpoolctl import threadpool_limits


def main():
    validate_frozen()
    if (EXP/'diagnostics.json').exists():
        raise ValueError('Post-hoc evidence exists; preserve it')
    model=EmbeddingModel()
    sets={
        'lights':['bedroom lamp','bedroom light','bedroom lights','bed lamp','bedside lamp'],
        'names':['the weeknd','weekend','adele','adell','beatles','the beatles','acdc','daily mix one','bjork'],
    }
    evidence=[]
    for label,names in sets.items():
        entries=[dict(id='fixture:'+n,role='device' if label=='lights' else 'artist',aliases=[n]) for n in names]
        index=model.index(entries)
        evidence.append(dict(name=label,entries=entries,unavailable=index['unavailable'],kind='dictionary/index coverage only; no acoustic recognition claim'))
    pronunciation=phones('adele',model.dictionary)
    homophones=[dict(id='artist:adele',role='artist',aliases=[],pronunciations=[pronunciation]),
                dict(id='artist:adell',role='artist',aliases=[],pronunciations=[pronunciation])]
    index=model.index(homophones)
    scores=model.rank(index['matrix'][:1],index)
    gate=dict(passed=True,threshold=.5,margin=1e-6,catalog_version=catalog_fingerprint(homophones))
    result=decision(homophones,scores,'MEDIA_PLAY','artist',gate)
    assert result['status']=='abstain'
    evidence.append(dict(name='explicit_homophones',entries=homophones,scores=scores,result=result,
                         kind='text-index collision control using an indexed vector as query, not audio'))
    predictions=read(CACHE/'test-predictions.json')
    rows={r['id']:r for r in read(EXP/'recordings.json')}
    groups={}
    for group in ('natural_number','isolated_number','time','artist','song','device'):
        records=[r for r in predictions if r['variant']=='expanded' and r['group']==group]
        known=[r for r in records if r['reference']]
        by_source=defaultdict(list)
        for record in known:by_source[rows[record['row_id']]['source']].append(record)
        groups[group]=dict(
            missing_pronunciation_references=sum(r['oov_reference'] for r in known),
            representable=len([r for r in known if not r['oov_reference']]),
            raw_errors_with_pronunciation=sum(not r['oov_reference'] and (not r['scores'] or r['scores'][0][0]!=r['reference']) for r in known),
            by_source={source:dict(n=len(items),top1=sum(r['scores'] and r['scores'][0][0]==r['reference'] for r in items)) for source,items in by_source.items()},
            margin_quantiles={label:np.percentile([r['scores'][0][1]-r['scores'][1][1] for r in items if len(r['scores'])>1],[0,50,95,100]).tolist()
                for label,items in [('known',known),('unknown',[r for r in records if r['reference'] is None])] if items})
    result=dict(post_hoc=True,no_threshold_or_model_changes=True,index_controls=evidence,groups=groups,
                source_sha256=digest(Path(__file__)),
                interpretation='Dictionary failures explain only the unavailable portion. Remaining errors cannot be causally assigned to architecture, data or localization without an ablation.')
    write(EXP/'hard-negatives.json',evidence)
    write(EXP/'diagnostics.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
