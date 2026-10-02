import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision
import corpus


def test_reuse_requires_exact_original_or_explicit_pretest_amendment():
    old={r['id']:r for r in revision.read_rows(revision.V1/'verbatims.jsonl')}
    rows=revision.build_verbatims()
    for row in rows:
        if row['id'] in old:corpus.validate_reuse_reference(old[row['id']],row)
    base=next(r for r in rows if r['expected_command'])
    with pytest.raises(ValueError):
        corpus.validate_reuse_reference(base,{**base,'text':'different spoken words'})
    with pytest.raises(ValueError):
        corpus.validate_reuse_reference(base,{**base,'expected_command':None})


def test_completed_generation_receipt_must_bind_full_manifest(tmp_path):
    manifest=tmp_path/'qwen-manifest.jsonl';manifest.write_text('[]\n')
    receipt=tmp_path/'qwen-generation.json'
    revision.write_json(receipt,dict(complete=True,generated=1,planned=1,manifest_sha256=revision.file_hash(manifest)))
    assert corpus.generation_complete(tmp_path,'qwen',expected=1)
    manifest.write_text('changed\n')
    with pytest.raises(ValueError):corpus.generation_complete(tmp_path,'qwen',expected=1)


def test_reuse_receipt_cannot_substitute_generation_metadata():
    source={'recording_id':'one','audio_sha256':'abc','generation_attempt':1}
    plan={'recording_id':'one','audio':'absolute.wav'}
    row={**source,**plan,'source_record_sha256':revision.identity(source)}
    corpus.validate_reused_record(row,plan,source)
    with pytest.raises(ValueError):corpus.validate_reused_record({**row,'generation_attempt':2},plan,source)


def test_auxiliary_wake_cases_are_continuous_utterances_and_separate_from_commands():
    commands=revision.build_verbatims();rows,plan=corpus.wake_assignments(commands)
    assert {r['use_case'] for r in rows if r['case_type']=='wake_command'}=={f'UC-{i:02}' for i in range(1,11)}
    assert all(r['text'].lower().startswith('hey jarvis.') for r in rows if r['case_type']=='wake_command')
    assert {'wrong_wake','wake_only','no_pause'}<={r['case_type'] for r in rows}
    assert {r['split'] for r in plan}=={'dev','calibration','test'}
    assert not {r['id'] for r in rows}&{r['id'] for r in commands}
    assert len({r['recording_id'] for r in plan})==len(plan)


def test_generation_persists_real_wav_and_rejects_resume_metadata_drift(tmp_path,monkeypatch):
    import json
    import numpy as np
    import torch
    import synthesis
    import assets
    import soundfile as sf
    inputs=tmp_path/'inputs';cache=tmp_path/'cache';old=tmp_path/'old'
    for path in (inputs,cache,old):path.mkdir()
    voice=dict(id='fixture',split='dev',accent='British',parler='fixture')
    refs=[dict(id=f'fixture-{i}',text=f'fixture {i}') for i in range(5)]
    plan=[dict(recording_id=f'parler/fixture/{r["id"]}',verbatim_id=r['id'],engine='parler',
        voice_group='fixture',split='dev',source_version='v2',seed=i,synthetic=True,
        audio=str(cache/'audio'/f'{i}.wav'),native_audio=str(cache/'native'/f'{i}.wav')) for i,r in enumerate(refs)]
    for name,rows in [('verbatims.jsonl',refs),('generation-plan.jsonl',plan),('wake-verbatims.jsonl',[]),('wake-generation-plan.jsonl',[])]:
        revision.write_rows(inputs/name,rows)
    revision.write_json(inputs/'protocol.json',{'fixture':True})
    revision.write_json(old/'parler-model-identity.json',{'fixture':True})
    revision.write_json(old/'parler-recipe.json',{'fixture':True})
    for name,value in [('HERE',inputs),('CACHE',cache),('OLD_CACHE',old),('VOICES',[voice])]:monkeypatch.setattr(corpus,name,value)
    monkeypatch.setattr(revision,'validate',lambda:None)
    monkeypatch.setattr(assets,'verify_assets',lambda _: 'fixture-assets')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    def generator(texts,_voice):
        return [(.2*np.sin(2*np.pi*(200+int(t.split()[-1])*20)*np.arange(24000)/24000)).astype('float32') for t in texts],24000
    monkeypatch.setattr(synthesis,'load_engine',lambda _: (generator,{'fixture':True}))
    corpus.generate('parler')
    rows=revision.read_rows(cache/'generated-parler.jsonl')
    assert len(rows)==5 and json.loads((cache/'parler-v2-generation.json').read_text())['complete']
    for row in rows:
        pcm,rate=sf.read(row['audio']);assert rate==16000 and len(pcm)==16000
    corpus.generate('parler')
    assert revision.read_rows(cache/'generated-parler.jsonl')==rows
    rows[0]['verbatim_sha256']='tampered-reference'
    (cache/'generated-parler.jsonl').write_text(''.join(revision.canonical(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='provenance'):
        corpus.generate('parler')
