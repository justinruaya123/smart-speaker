import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from test_bundle import fixture_bundle,ROOT,qualify_fixture
import pytest
from scripts.pcv_delivery import package


def test_clean_archive_checks_cpu_runtime_without_training_or_corpus(fixture_bundle,tmp_path):
    output=tmp_path/'pi.zip';receipt=package(fixture_bundle,output,diagnostic=True)
    assert receipt['diagnostic'] and receipt['target']['machine']=='aarch64'
    extracted=tmp_path/'extracted'
    with zipfile.ZipFile(output) as archive:
        names=archive.namelist()
        assert {'setup.sh','serve.sh','verify_package.py','MANIFEST.json'}<=set(names)
        assert not any('torch' in n or '/training/' in n or 'synthbench' in n for n in names)
        archive.extractall(extracted)
    environment=dict(os.environ);environment.pop('PYTHONPATH',None);environment['PYTHONNOUSERSITE']='1'
    check=subprocess.run([sys.executable,'verify_package.py'],cwd=extracted,env=environment,text=True,capture_output=True)
    assert check.returncode==0,check.stderr
    run=subprocess.run([sys.executable,'-c',
        "import sys; sys.argv=['pcv','check','--bundle','models/default','--diagnostic']; from scripts.runtime.pcv_deploy.__main__ import main; main(); assert 'torch' not in sys.modules"],
        cwd=extracted,env=environment,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    report=json.loads(run.stdout);assert report['provider']=='CPUExecutionProvider' and report['threads']==1
    target=extracted/'scripts/runtime/pcv_deploy/stream.py';target.write_text(target.read_text()+'\n# tampered\n')
    check=subprocess.run([sys.executable,'verify_package.py'],cwd=extracted,env=environment,text=True,capture_output=True)
    assert check.returncode!=0 and 'integrity mismatch' in check.stderr


def test_publication_rejects_changed_recognition_implementation(fixture_bundle,tmp_path,monkeypatch):
    qualify_fixture(fixture_bundle)
    original=Path.read_bytes
    def changed(path):
        result=original(path)
        return result+b'\n# substitution' if path==ROOT/'vcm/frontend.py' else result
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(ValueError,match='Recognition source'):
        package(fixture_bundle,tmp_path/'must-not-publish.zip')
    assert not (tmp_path/'must-not-publish.zip').exists()


def test_enabled_publication_requires_portable_waveform_parity(fixture_bundle,tmp_path):
    qualify_fixture(fixture_bundle)
    with pytest.raises(ValueError,match='portable parity'):
        package(fixture_bundle,tmp_path/'missing-parity.zip')
    assert not (tmp_path/'missing-parity.zip').exists()


def test_child_inherited_wake_source_passes_source_check_but_still_requires_parity(fixture_bundle,tmp_path):
    qualify_fixture(fixture_bundle,inherited_wake_source=True)
    with pytest.raises(ValueError,match='portable parity'):
        package(fixture_bundle,tmp_path/'child-missing-parity.zip')
    assert not (tmp_path/'child-missing-parity.zip').exists()
