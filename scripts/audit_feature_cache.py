"""Add an immutable, post-run feature audit without rewriting frozen manifests."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vcm.data import read_manifest
from vcm.frontend import features,SPEC
from vcm.integrity import digest,identity,load,save


def audit(prepared,output,atol=0.):
    prepared=Path(prepared); output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    failures=[]; count=0; total=np.zeros(40); squares=np.zeros(40); frames=0
    cached_total=np.zeros(40); cached_squares=np.zeros(40); exact_count=0; max_difference=0.
    manifests={s:digest(prepared/f'{s}.jsonl') for s in ('train','dev','calibration','test')}
    with (output/'features.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','split','features_sha256','frontend_spec','audio_match','pcm_match','features_exact','max_absolute_difference'])
        writer.writeheader()
        for split in manifests:
            for r in read_manifest(prepared/f'{split}.jsonl'):
                pcm,rate=sf.read(r['audio'],dtype='float32')
                cached=np.load(r['features'],allow_pickle=False); rebuilt=features(pcm)
                result={'id':r['id'],'split':split,'features_sha256':digest(r['features']),'frontend_spec':identity(SPEC),
                        'audio_match':digest(r['audio'])==r['audio_sha256'],
                        'pcm_match':rate==16000 and hashlib.sha256(pcm.tobytes()).hexdigest()==r['pcm_sha256'],
                        'features_exact':cached.dtype==rebuilt.dtype and np.array_equal(cached,rebuilt),
                        'max_absolute_difference':float(np.max(np.abs(cached-rebuilt))) if cached.shape==rebuilt.shape else float('inf')}
                writer.writerow(result)
                exact_count+=result['features_exact']; max_difference=max(max_difference,result['max_absolute_difference'])
                if not result['audio_match'] or not result['pcm_match'] or not np.isfinite(cached).all() or result['max_absolute_difference']>atol: failures.append(r['id'])
                if split=='train':
                    f=rebuilt.astype('float64'); total+=f.sum(1); squares+=(f*f).sum(1); frames+=f.shape[1]
                    c=cached.astype('float64'); cached_total+=c.sum(1); cached_squares+=(c*c).sum(1)
                count+=1
                if count%5000==0: print('Audited',count,flush=True)
    mean=total/frames; std=np.sqrt(np.maximum(squares/frames-mean**2,1e-4)); norm=load(prepared/'normalization.json')
    normalization_exact=np.array_equal(mean,norm['mean']) and np.array_equal(std,norm['std']) and frames==norm['frames']
    cached_mean=cached_total/frames; cached_std=np.sqrt(np.maximum(cached_squares/frames-cached_mean**2,1e-4))
    cached_normalization_exact=np.array_equal(cached_mean,norm['mean']) and np.array_equal(cached_std,norm['std']) and frames==norm['frames']
    report={'rows':count,'failures':failures,'normalization_exact':normalization_exact,'train_frames':frames,
            'cached_normalization_exact':cached_normalization_exact,'features_exact_count':exact_count,'max_absolute_feature_difference':max_difference,
            'exploratory_absolute_tolerance':atol,'mean_max_absolute_difference':float(np.max(np.abs(mean-norm['mean']))),
            'std_max_absolute_difference':float(np.max(np.abs(std-norm['std']))),
            'frontend_spec':identity(SPEC),'manifests':manifests,'normalization_sha256':digest(prepared/'normalization.json'),
            'features_csv_sha256':digest(output/'features.csv'),'audit_source_sha256':digest(__file__),
            'scope':'Post-run exact recomputation from recorded source WAVs; original training receipt unchanged; no new fitting or selection',
            'temporal_limit':'Verifies current cache equivalence; historical cache bytes were not independently hashed before fitting.'}
    save(output/'summary.json',report); print(report,flush=True)
    if failures or not cached_normalization_exact or (atol==0. and not normalization_exact): raise ValueError('Feature provenance audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--prepared',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--atol',type=float,default=0.,help='Exploratory absolute tolerance; zero requires exact equality')
    a=p.parse_args(); audit(a.prepared,a.output,a.atol)
