"""Export the selected project checkpoint into the optional inspector recipe."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def export(bundle,checkpoint,output,*,pcv_runtime=False,evidence=None):
    import shutil
    import numpy as np
    import torch
    from einops import rearrange,pack
    from vcm.model import CausalVCM
    from vcm.runtime import Runtime
    from vcm.frontend import StreamingFrontend
    from vcm.integrity import load,save,digest,identity
    if output.exists(): raise FileExistsError(output)
    source={}
    if pcv_runtime:
        from scripts.pcv_delivery import verify_bundle
        from scripts.runtime.pcv_deploy.bundle import STATE_SHAPES,cpu_session
        from pcvcm.contract import PHONES
        content=verify_bundle(bundle,allow_diagnostic=True)
        session=cpu_session(bundle/'model.onnx');state_shapes=STATE_SHAPES
        normalization=content['normalization'];inventory=['<blank>',*PHONES]
        forms=[{'target':c['phones'],'command':c['command'],'phrase':c['text']} for c in content['catalog']['entries']]
        source={'source_binding':content['manifest'].get('source_binding'),
                'source_deployment_binding':content['binding'],'source_profile':content['manifest']['status']}
    else:
        runtime=Runtime(bundle);session=runtime.session;state_shapes=runtime.recipe['state_shapes']
        normalization=runtime.recipe['frontend'];inventory=runtime.catalog['phone_inventory']
        forms=[{'target':c['phones'],'command':c['command'],'phrase':c['phrase']} for c in runtime.catalog['commands']]
    saved_evidence=None
    if evidence is not None:
        if not pcv_runtime:raise ValueError('Measured demo evidence requires a PCV deployment bundle')
        measured=load(evidence)
        if not source['source_binding'] or measured.get('binding')!=source['source_binding']:raise ValueError('Evidence belongs to another model')
        saved_evidence={'binding':measured['binding'],'deployment_binding':source['source_deployment_binding'],
                        'source_sha256':digest(evidence),'rows':[]}
        for command in measured['commands_with_results']:
            ref=command['reference']
            for trial in command['outputs']:
                raw=trial.get('candidate');expected=ref['expected_command'];accepted=trial['output']['accepted']
                exact=identity(raw)==identity(expected)
                saved_evidence['rows'].append(dict(trial='pcv-synthetic-v2',dataset='sealed strict demo',
                    voice=trial['engine']+'/'+trial['voice_group'],id=trial['recording_id'],verbatim=ref['text'],
                    expected=expected,raw_output=raw,raw_exact=exact,accepted=accepted,
                    outcome='correct_accepted' if accepted and exact else 'wrong_accept' if accepted else 'false_reject'))
    selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=CausalVCM().eval(); model.load_state_dict(selected['weights'])
    # Verify checkpoint versus the selected interchange graph before exporting weights.
    generator=torch.Generator().manual_seed(92)
    states=[torch.randn(s,generator=generator) for s in state_shapes]
    x=torch.randn(1,40,4,generator=generator)
    with torch.no_grad(): expected,_=model(x,states)
    actual=session.run(None,{'features':x.numpy(),**{f'state_{i}':s.numpy() for i,s in enumerate(states)}})[0]
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
    graph={'phone_inventory':inventory,'forms':forms}
    graph['graph_hash']=identity(graph); save(output/'catalog.json',graph)
    f=StreamingFrontend()
    recipe={'schema_version':'vcm-inspector-1','name':'Causal VCM - layer inspection',**source,
            'source_onnx':'model.onnx','source_onnx_sha256':digest(output/'model.onnx'),'weights':'weights.bin',
            'weights_sha256':digest(output/'weights.bin'),'catalog':'catalog.json','graph_hash':graph['graph_hash'],
            'catalog_body_sha256':identity(graph),'sample_rate':16000,'window':400,'hop':160,'fft':512,
            'mel_bins':40,'fmin':20,'fmax':7600,'log_floor':1e-8,'chunk_frames':4,'maximum_seconds':8,
            'mean':normalization['mean'],'std':normalization['std'],
            'mel_filters':rearrange(f.bank,'f c -> (c f)').tolist(),'window_coefficients':f.window.tolist(),
            'vocabulary':inventory,'layers':layers,'acceptance_enabled':False,'action_execution_enabled':False,
            'notes':'Inspection only. No historical waveform examples substituted. Browser/Pi timings require separate measurement.'}
    if saved_evidence is not None:
        save(output/'evidence.json',saved_evidence);recipe['evidence_sha256']=digest(output/'evidence.json')
    save(output/'recipe.json',recipe); save(output/'examples.json',[])
    return recipe


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pcv-runtime',action='store_true');p.add_argument('--evidence',type=Path)
    a=p.parse_args(); export(a.bundle,a.checkpoint,a.output,pcv_runtime=a.pcv_runtime,evidence=a.evidence)
