"""CPU software fixture only: no acceptance or trained-model evidence."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study
import synthbench
import finishing
import reporting
import fitting
from vcm.model import CausalVCM


def test_export_replay_parity_calibration_and_report_preserve_failed_acceptance(tmp_path,monkeypatch):
    # Only corpus acquisition/auditing is replaced. The model, ONNX sessions,
    # streaming frontend/decoder, endpoint, policy and reports execute normally.
    for module in (study,synthbench,reporting):monkeypatch.setattr(module,'CACHE',tmp_path)
    prepared=tmp_path/'prepared';prepared.mkdir()
    manifests={}
    for engine in study.ENGINES:
        path=tmp_path/f'{engine}-manifest.jsonl';path.write_text('')
        manifests[engine]=study.file_hash(path)
    audit=dict(manifests=manifests,software_fixture_only=True)
    study.write_json(prepared/'global-audit.json',audit)
    refs=study.read_rows(study.HERE/'verbatims.jsonl')
    chosen=[next(r for r in refs if r['expected_command'] is not None),
            next(r for r in refs if r['expected_command'] is None)]
    plans=[];splits={}
    for split in ('dev','calibration','test'):
        splits[split]=[]
        for index,ref in enumerate(chosen):
            relative=f'{split}-{index}.wav';path=tmp_path/relative
            sf.write(path,.1*np.sin(np.arange(16000)*(.03+.01*index)),16000,subtype='PCM_16')
            row={**ref,'recording_id':f'fixture/{split}/{ref["id"]}',
                'verbatim_id':ref['id'],'audio':relative,'split':split,
                'engine':'qwen','voice_group':f'fixture-{split}',
                'accent_requested':'software tone fixture','audio_sha256':study.file_hash(path),
                'pcm_sha256':'software-fixture-not-a-corpus-receipt'}
            splits[split].append(row);plans.append(row)
    original_read=study.read_rows
    def fixture_read(path):
        if Path(path)==study.HERE/'generation-plan.jsonl':return plans
        return original_read(path)
    monkeypatch.setattr(synthbench,'read_rows',fixture_read)
    monkeypatch.setattr(reporting,'read_rows',fixture_read)
    monkeypatch.setattr(synthbench,'load_split',lambda split:splits[split])
    monkeypatch.setattr(synthbench,'audit_all_splits',lambda:audit)
    run=tmp_path/'training';run.mkdir()
    torch.manual_seed(53);torch.set_num_threads(1)
    checkpoint=run/'software-fixture.pt'
    torch.save({'weights':CausalVCM().state_dict()},checkpoint)
    study.write_json(run/'selection.json',{'winner':{'checkpoint':str(checkpoint),
        'checkpoint_sha256':study.file_hash(checkpoint)}})
    study.write_json(run/'normalization.json',{'mean':[0.]*40,'std':[1.]*40})
    study.write_json(run/'catalog.json',synthbench.compile_synthetic_catalog(refs))
    bundle=tmp_path/'bundle'
    exported=finishing.export(run,bundle)
    assert exported['parity']['parameters']==113064
    assert exported['parity']['max_absolute_error']<2e-5
    parity=finishing.waveform_parity(run,bundle,tmp_path/'parity')
    assert parity['complete'] and parity['recordings']==2
    policy=finishing.calibrate(bundle,tmp_path/'calibration',tmp_path/'parity/results.json')
    assert policy['enabled'] is False
    assert policy['status']=='SYNTHETIC_CALIBRATION_FAILED'
    final=synthbench.evaluate(tmp_path/'calibration/bundle','test',tmp_path/'final-test')
    assert final['complete'] and final['overall']['trials']==2
    assert final['synthetic_gates_passed'] is False
    assert final['timing']['ort_provider']=='CPUExecutionProvider'
    with pytest.raises(FileExistsError):
        synthbench.evaluate(tmp_path/'calibration/bundle','test',tmp_path/'illegal-test-rerun')
    report=reporting.report(tmp_path/'report')
    assert report['production_ready'] is False and report['synthetic_gates_passed'] is False
    observed=study.read_rows(tmp_path/'report/all-verbatims-with-outputs.jsonl')
    assert sum(r['status']=='MEASURED' for r in observed)==2
    assert any(r['status']=='NOT_RUN' for r in observed)
    assert all(not x['actual_output']['accepted'] for r in observed for x in r['observations'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA continuation smoke requires the declared device')
def test_training_continuation_writes_baseline_selection_and_improvement_on_artificial_features(tmp_path,monkeypatch):
    monkeypatch.setattr(fitting,'CACHE',tmp_path)
    prepared=tmp_path/'prepared';prepared.mkdir()
    refs=study.read_rows(study.HERE/'verbatims.jsonl')
    chosen=[next(r for r in refs if r['expected_command'] is not None),
            next(r for r in refs if r['expected_command'] is None)]
    catalog=synthbench.compile_synthetic_catalog(chosen)
    phones={r['id']:r['phones'] for r in catalog['entries']}
    rng=np.random.default_rng(912)
    for split in ('train','dev'):
        rows=[]
        for index,ref in enumerate(chosen):
            path=prepared/f'{split}-{index}.npy'
            np.save(path,rng.normal(size=(40,100)).astype(np.float32))
            rows.append({**ref,'split':split,'recording_id':f'fixture/{split}/{index}',
                'phrase_id':ref['id'],'speaker_id':f'fixture-{split}',
                'phones':phones[ref['id']],'features':path.name,
                'features_sha256':study.file_hash(path)})
        study.write_rows(prepared/f'{split}.jsonl',rows)
    study.write_json(prepared/'normalization.json',{'mean':[0.]*40,'std':[1.]*40})
    study.write_json(prepared/'catalog.json',catalog)
    study.write_json(prepared/'freeze.json',dict(
        files={p.name:study.file_hash(p) for p in prepared.iterdir() if p.suffix!='.npy'},
        protocol_sha256=study.file_hash(study.HERE/'protocol.json')))
    run=tmp_path/'training'
    selected=fitting.train(run)
    assert selected['test_accessed'] is False and selected['calibration_accessed'] is False
    assert len(selected['candidates'])==20
    assert len(list(run.glob('seed-*/epoch-*.pt')))==80
    assert len(json.loads((tmp_path/'compute-budget.json').read_text())['epochs'])==80
    improvement=json.loads((run/'development-improvement.json').read_text())
    assert set(improvement['gained']).isdisjoint(improvement['lost'])
    assert study.file_hash(selected['winner']['checkpoint'])==selected['winner']['checkpoint_sha256']
