"""Export the selected project checkpoint into the optional inspector recipe."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def export(bundle,checkpoint,output):
    import shutil
    import numpy as np
    import torch
    from einops import rearrange,pack
    from vcm.model import CausalVCM
    from vcm.runtime import Runtime
    from vcm.frontend import StreamingFrontend
    from vcm.integrity import load,save,digest,identity
    if output.exists(): raise FileExistsError(output)
    runtime=Runtime(bundle)
    selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=CausalVCM().eval(); model.load_state_dict(selected['weights'])
    # Verify checkpoint versus the selected interchange graph before exporting weights.
    generator=torch.Generator().manual_seed(92)
    states=[torch.randn(s,generator=generator) for s in runtime.recipe['state_shapes']]
    x=torch.randn(1,40,4,generator=generator)
    with torch.no_grad(): expected,_=model(x,states)
    actual=runtime.session.run(None,{'features':x.numpy(),**{f'state_{i}':s.numpy() for i,s in enumerate(states)}})[0]
    np.testing.assert_allclose(actual,rearrange(expected.numpy(),'b c t -> b t c'),atol=2e-5,rtol=1e-5)
    output.mkdir(parents=True)
    tensors=[]; layers=[]; offset=0
    for i,layer in enumerate([*model.layers,model.projection]):
        weight=layer.weight.detach().numpy().astype('<f4'); bias=layer.bias.detach().numpy().astype('<f4')
        co,ci,kernel=weight.shape; dilation=layer.dilation[0]
        layers.append({'id':'stem' if i==0 else 'ctc' if i==6 else f'conv{i}','type':'Conv1d','in_channels':ci,'out_channels':co,
                       'kernel':kernel,'dilation':dilation,'stride':1,'activation':'linear' if i==6 else 'relu',
                       'weight_offset':offset,'bias_offset':offset+weight.size,'parameters':weight.size+bias.size,
                       'history_frames':(kernel-1)*dilation})
        tensors.extend([rearrange(weight,'o i k -> (o i k)'),bias]); offset+=weight.size+bias.size
    flat,_=pack(tensors,'*'); (output/'weights.bin').write_bytes(flat.tobytes())
    shutil.copyfile(bundle/'model.onnx',output/'model.onnx')
    graph={'phone_inventory':runtime.catalog['phone_inventory'],'forms':[{'target':c['phones'],'command':c['command'],'phrase':c['phrase']} for c in runtime.catalog['commands']]}
    graph['graph_hash']=identity(graph); save(output/'catalog.json',graph)
    f=StreamingFrontend()
    recipe={'schema_version':'vcm-inspector-1','name':'Semifinal causal VCM - diagnostic only',
            'source_onnx':'model.onnx','source_onnx_sha256':digest(output/'model.onnx'),'weights':'weights.bin',
            'weights_sha256':digest(output/'weights.bin'),'catalog':'catalog.json','graph_hash':graph['graph_hash'],
            'catalog_body_sha256':identity(graph),'sample_rate':16000,'window':400,'hop':160,'fft':512,
            'mel_bins':40,'fmin':20,'fmax':7600,'log_floor':1e-8,'chunk_frames':4,'maximum_seconds':8,
            'mean':runtime.recipe['frontend']['mean'],'std':runtime.recipe['frontend']['std'],
            'mel_filters':rearrange(f.bank,'f c -> (c f)').tolist(),'window_coefficients':f.window.tolist(),
            'vocabulary':runtime.catalog['phone_inventory'],'layers':layers,'acceptance_enabled':False,'action_execution_enabled':False,
            'notes':'Inspection only. No historical waveform examples substituted. Browser/Pi timings require separate measurement.'}
    save(output/'recipe.json',recipe); save(output/'examples.json',[])
    return recipe


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); export(a.bundle,a.checkpoint,a.output)
