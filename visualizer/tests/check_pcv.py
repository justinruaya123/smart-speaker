"""Reproduce portable-model WASM/frontend/CPU ONNX parity on a named WAV."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
import numpy as np
import soundfile as sf
from einops import rearrange
from pcvcm.contract import file_hash,write_json
from scripts.runtime.pcv_deploy.bundle import STATE_SHAPES,cpu_session
from vcm.frontend import StreamingFrontend


def check(module_dir,assets,wav,output):
    if output.exists():raise FileExistsError('Keep previous simulator parity receipts')
    output.mkdir(parents=True)
    recipe=json.loads((assets/'recipe.json').read_text(encoding='utf-8'))
    if file_hash(assets/'model.onnx')!=recipe['source_onnx_sha256']:raise ValueError('Inspector graph changed')
    subprocess.run(['node',str(ROOT/'visualizer/tests/pcv_parity.mjs'),str(module_dir),str(assets),str(wav),str(output/'wasm.json')],check=True)
    measured=json.loads((output/'wasm.json').read_text(encoding='utf-8'))
    pcm,rate=sf.read(wav,dtype='float32')
    if rate!=16000 or pcm.ndim!=1:raise ValueError('Mono 16 kHz WAV required')
    frontend=StreamingFrontend(recipe['mean'],recipe['std']);features=frontend.feed(pcm)
    session=cpu_session(assets/'model.onnx')
    states={f'state_{i}':np.zeros(shape,dtype=np.float32) for i,shape in enumerate(STATE_SHAPES)}
    maximum=frontend_max=0.;offset=0
    for chunk in measured['chunks']:
        shape=dict(c=40,t=recipe['chunk_frames'])
        actual_features=rearrange(np.asarray(chunk['features'],dtype=np.float32),'(c t) -> c t',**shape)
        valid=chunk['valid'];reference=features[:,offset:offset+valid]
        frontend_max=max(frontend_max,float(np.max(np.abs(reference-actual_features[:,:valid]))))
        np.testing.assert_allclose(reference,actual_features[:,:valid],atol=2e-5,rtol=1e-5)
        inputs=np.zeros((40,recipe['chunk_frames']),dtype=np.float32);inputs[:,:valid]=reference
        values=session.run(None,dict(features=rearrange(inputs,'c t -> 1 c t'),**states))
        states={f'state_{i}':x for i,x in enumerate(values[1:])}
        reference_logits=rearrange(values[0],'1 t c -> c t')
        wasm_logits=rearrange(np.asarray(chunk['logits'],dtype=np.float32),'(c t) -> c t',**shape)
        maximum=max(maximum,float(np.max(np.abs(reference_logits[:,:valid]-wasm_logits[:,:valid]))))
        np.testing.assert_allclose(reference_logits[:,:valid],wasm_logits[:,:valid],atol=2e-5,rtol=1e-5)
        offset+=valid
    assert offset==features.shape[1] and measured['final']['status']=='needs_clarification'
    result=dict(passed=True,frames=offset,chunks=len(measured['chunks']),frontend_max_error=frontend_max,
        logits_max_error=maximum,source_onnx_sha256=recipe['source_onnx_sha256'],audio_sha256=file_hash(wav),
        wasm_results_sha256=file_hash(output/'wasm.json'),speech_accuracy_measured=False,physical_pi_measured=False)
    write_json(output/'results.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('module-dir','assets','wav','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(check(args.module_dir,args.assets,args.wav,args.output),indent=2))
