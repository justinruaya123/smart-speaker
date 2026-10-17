"""Check the selected VCM's Torch/ONNX/ncnn decisions on fixed audio cohorts.

This is action-free CPU validation. It never fits a policy or chooses weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def verify(root=ROOT,output=None):
    root=Path(root).resolve()
    if root!=ROOT.resolve():
        raise ValueError('Parity --root must be the executing source checkout; run from the reconstructed checkout')
    import soundfile as sf
    import torch
    from scripts.iris_release import read,resolve_audio,sha
    from scripts.models.mvp_static import StaticCommandNet
    from scripts.runtime.mvp_static import Runtime,decide
    from scripts.runtime.pcv_schema import output as typed_output
    from vcm.frontend import features

    here=root/'experiments/iris-release-v1'
    cache=root/'.cache/iris-release-v1'
    destination=Path(output) if output else here/'decision-parity.json'
    choice=read(here/'release-choice.json')
    selected=read(here/'development-selection.json')
    checkpoint=root/(selected['checkpoint'] if choice['promoted'] else
        '.cache/iris-release-v1/inputs/'+choice['width']+'-initializer.pt')
    bundle=cache/'diversity/release-export'
    manifests={split:cache/'diversity/views'/f'{split}.jsonl'
               for split in ('holdout','personal-test','voice-test')}
    sources=['scripts/test/iris_decision_parity.py','scripts/models/mvp_static.py',
             'scripts/runtime/mvp_static.py','scripts/runtime/pcv_schema.py','vcm/frontend.py']
    bindings={name:sha(root/name) for name in sources}
    model_files=[bundle/'bundle.json',*[bundle/name for name in read(bundle/'bundle.json')['files']],
        bundle/'ncnn/engine.json',*[bundle/'ncnn'/name for name in read(bundle/'ncnn/engine.json')['files']]]
    for path in [here/'release-choice.json',here/'development-selection.json',checkpoint,
                 *model_files,*manifests.values()]:
        if path.is_file():bindings[path.relative_to(root).as_posix()]=sha(path)
    if destination.exists():
        previous=read(destination)
        if previous['bindings']!=bindings:raise ValueError('Parity inputs changed; preserve the receipt and use a new output')
        if not previous.get('passed'):raise ValueError('Preserved decision parity failed')
        return previous

    began=time.perf_counter();torch.set_num_threads(1)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=StaticCommandNet(**saved['config']);net.load_state_dict(saved['weights']);net.eval()
    runtimes={engine:Runtime(bundle,threads=1,engine=engine) for engine in ('onnxruntime','ncnn')}
    reference=runtimes['onnxruntime'];counts={};differences=[];maximum={engine:0. for engine in runtimes}
    for split,manifest in manifests.items():
        rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        counts[split]=len(rows)
        for row in rows:
            pcm,rate=sf.read(resolve_audio(row,root=root),dtype='float32')
            if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Invalid fixed parity audio')
            logits=None;index=None;expected=typed_output(None,'capture_fault')
            if 400<=len(pcm)<=240000 and np.max(np.abs(pcm),initial=0)>=1e-6:
                array=features(pcm,reference.norm['mean'],reference.norm['std'])
                with torch.no_grad():
                    logits=net(torch.from_numpy(array[None]),torch.tensor([array.shape[1]])).numpy().reshape(-1)
                decision=decide(logits,reference.catalog,reference.policy)
                index=decision['class_index'];expected=typed_output(decision['command'],decision.get('reason'))
            for engine,runtime in runtimes.items():
                event=runtime.recognize(pcm,sample_rate=rate)
                diagnostic=event['diagnostic_candidate'] or {}
                actual_index=diagnostic.get('class_index')
                if logits is not None:
                    error=float(np.max(np.abs(logits-np.asarray(diagnostic['logits']))))
                    maximum[engine]=max(maximum[engine],error)
                if actual_index!=index or event['output']!=expected:
                    differences.append(dict(split=split,id=row['id'],engine=engine,
                        torch_index=index,engine_index=actual_index,torch_output=expected,engine_output=event['output']))
        print('PARITY',split,len(rows),'differences',len(differences),flush=True)
    if any(sha(root/name)!=value for name,value in bindings.items()):raise ValueError('Parity inputs changed during replay')
    result=dict(experiment='iris-release-v1',evidence='actual selected-model original PCM CPU replay',
        passed=not differences,bindings=bindings,cohorts=counts,recordings=sum(counts.values()),
        engine_comparisons=2*sum(counts.values()),raw_argmax_and_accepted_output_equal=not differences,
        max_absolute_logit_error=maximum,differences=differences,seconds=time.perf_counter()-began,
        torch_version=torch.__version__,threads=1,actions_executed=False,physical_pi_measured=False,
        thresholds_changed=False,weight_selection_changed=False)
    destination.parent.mkdir(exist_ok=True,parents=True)
    with destination.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if differences:raise ValueError(f'Decision parity failed: {len(differences)} differences; see {destination}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=verify(args.root,args.output)
    print(json.dumps({key:result[key] for key in ('passed','recordings','engine_comparisons','seconds')}))


if __name__=='__main__':main()
