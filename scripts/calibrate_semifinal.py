"""Fit acceptance on the declared calibration partition before sealed test replay."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vcm.integrity import load,save,digest,identity
from vcm.data import read_manifest
from vcm.evaluation import replay,write_reports
from vcm.acceptance import candidate_reason


def acceptance_mask(rows,threshold,margin):
    return np.array([candidate_reason(r.get('candidate'),r['score'] if r.get('score') is not None else float('nan'),
                                     r['margin'] if r.get('margin') is not None else float('nan'),threshold,margin)=='accepted'
                     for r in rows],dtype=bool)


def select_thresholds(rows):
    scores=np.array([r.get('score') if r.get('score') is not None else -1e9 for r in rows])
    margins=np.array([r.get('margin') or 0. for r in rows])
    correct=np.array([r['expected'] is not None and identity(r['expected'])==identity(r.get('candidate')) for r in rows])
    valid=np.array([r['valid_request'] for r in rows]); negative=~valid
    best=None
    finite=np.isfinite(scores)&np.isfinite(margins)
    if not finite.any(): return None
    for threshold in np.unique(np.quantile(scores[finite],np.linspace(0,1,31))):
        for margin in np.unique(np.quantile(margins[finite],np.linspace(0,1,21))):
            accept=acceptance_mask(rows,threshold,margin)
            emitted=int(accept.sum()); good=int((accept&correct).sum())
            precision=good/emitted if emitted else 0.; recall=good/int(valid.sum()) if valid.sum() else 0.
            far=int((accept&negative).sum())/int(negative.sum()) if negative.sum() else 1.
            if emitted and precision>=.99 and far<=.01 and (best is None or recall>best['recall']):
                best={'min_score':float(threshold),'min_margin':float(margin),'precision':precision,'recall':recall,'far':far,'emitted':emitted,'correct':good}
    return best


def calibrate(bundle,manifest,output):
    records=read_manifest(manifest)
    if not records or any(r['split']!='calibration' for r in records): raise ValueError('Calibration partition only')
    if output.exists(): raise FileExistsError(output)
    rows,binding=replay(bundle,manifest); write_reports(rows,output,binding)
    best=select_thresholds(rows)
    valid=np.array([r['valid_request'] for r in rows]); negative=~valid
    enabled=bool(best and best['recall']>=.9 and int(negative.sum())>=100)
    recipe=load(bundle/'recipe.json'); policy=load(bundle/'policy.json')
    policy.update(enabled=enabled,reason='calibrated' if enabled else 'calibration quality/coverage gates failed',
                  min_score=best['min_score'] if best else 0.,min_margin=best['min_margin'] if best else 0.,
                  calibration_manifest=digest(manifest),calibration_best=best)
    save(bundle/'policy.json',policy); recipe['files']['policy.json']=digest(bundle/'policy.json'); save(bundle/'recipe.json',recipe)
    save(output/'selection-before-test.json',{'policy_sha256':digest(bundle/'policy.json'),'recipe_sha256':digest(bundle/'recipe.json'),
         'calibration_manifest':digest(manifest),'calibration_best':best,'acceptance_enabled':enabled,'negative_count':int(negative.sum()),
         'valid_count':int(valid.sum()),'test_accessed':False,'policy':policy})
    print(policy)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); calibrate(a.bundle,a.manifest,a.output)
