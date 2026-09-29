"""Synthetic inputs and random-weight fixtures prove contracts, not recognition."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    from vcm.model import CausalVCM,export_model,STATE_SHAPES
    from vcm.catalog import PHONES
    from vcm.frontend import SPEC
    from vcm.integrity import save,digest,identity
    b=tmp_path_factory.mktemp('software-contract-fixture')
    export_model(CausalVCM(),b/'model.onnx')
    catalog={'phone_inventory':['<blank>']+PHONES,'commands':[{'phrase':'on','phones':[1,2],'command':{'intent':'POWER_ON','entity_id':'light.a'}},
                                                         {'phrase':'off','phones':[3,4],'command':{'intent':'POWER_OFF','entity_id':'light.a'}}]}
    front={'spec':SPEC,'mean':[0]*40,'std':[1]*40}
    binding={'model':digest(b/'model.onnx'),'catalog':identity(catalog),'frontend':identity(front)}
    save(b/'catalog.json',catalog); save(b/'policy.json',{'binding':binding,'enabled':False,'min_score':0,'min_margin':0})
    save(b/'recipe.json',{'status':'software_fixture_not_acoustic_evidence','frontend':front,'state_shapes':STATE_SHAPES,'chunk_frames':4,
                          'files':{p.name:digest(p) for p in b.iterdir()}})
    return b


def test_runtime_variable_chunks_hashes_and_faults(bundle,tmp_path):
    import shutil
    from vcm.runtime import Runtime
    r=Runtime(bundle); pcm=np.random.default_rng(13).normal(0,.1,7001).astype('float32')
    a=r.predict(pcm,chunk_samples=137); b=r.predict(pcm,chunk_samples=1280)
    assert a['candidate']==b['candidate'] and not a['accepted']
    assert a['score']==pytest.approx(b['score'],abs=1e-5)
    with pytest.raises(ValueError,match='Reset'): r.feed(pcm,0)
    r.reset(); r.feed(pcm[:640],0)
    assert r.feed(pcm[640:1280],700)['reason']=='capture_gap'
    assert r.feed(np.empty(0),1280,endpoint=True)['candidate'] is None
    damaged=tmp_path/'damaged'; shutil.copytree(bundle,damaged)
    (damaged/'catalog.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'): Runtime(damaged)


def test_extract_run_without_torch_or_network(bundle,tmp_path):
    from vcm.packaging import publish
    stale=bundle/'asr.onnx'; stale.write_bytes(b'obsolete, never allowed')
    output=tmp_path/'diagnostic.zip'; publish(bundle,output,diagnostic=True)
    extracted=tmp_path/'extracted'
    with zipfile.ZipFile(output) as z:
        assert len([p for p in z.namelist() if p.endswith('.onnx')])==1
        assert not any('asr' in p or 'legacy' in p or 'training' in p for p in z.namelist())
        z.extractall(extracted)
    code='''
import sys, socket, importlib.abc
sys.path.insert(0, '.')
class DenyTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','whisper','research','scripts'}:
            raise RuntimeError('Forbidden runtime dependency: '+fullname)
sys.meta_path.insert(0,DenyTraining())
def deny(*a,**k): raise RuntimeError('Network forbidden during recognition')
socket.socket.connect=deny
socket.create_connection=deny
import numpy as np
from vcm.runtime import Runtime
r=Runtime('models/selected')
out=r.predict(np.sin(np.arange(4000,dtype='float32')*.1)*.1)
assert not out['accepted'] and out['reason']=='uncalibrated'
print('extracted runtime passed with training imports and socket connections denied')
'''
    result=subprocess.run([sys.executable,'-I','-c',code],cwd=extracted,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_diagnostic_rejects_enabled_identity_consistent_policy(bundle,tmp_path):
    import shutil
    from vcm.integrity import load,save,digest
    from vcm.packaging import publish
    enabled=tmp_path/'enabled'; shutil.copytree(bundle,enabled)
    policy=load(enabled/'policy.json'); policy['enabled']=True; save(enabled/'policy.json',policy)
    recipe=load(enabled/'recipe.json'); recipe['files']['policy.json']=digest(enabled/'policy.json'); save(enabled/'recipe.json',recipe)
    with pytest.raises(ValueError,match='disabled'):
        publish(enabled,tmp_path/'diagnostic.zip',diagnostic=True)
    assert not (tmp_path/'diagnostic.zip').exists()
