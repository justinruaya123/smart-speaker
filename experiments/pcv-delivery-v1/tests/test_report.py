import json
from pathlib import Path
import pytest
from scripts import pcv_delivery as delivery
from pcvcm.contract import canonical,file_hash,write_json


def fixture_study(tmp_path,monkeypatch,*,reject=False):
    source=delivery.ROOT
    root=tmp_path/'repo';definitions=root/'experiments/pcv-delivery-v1';definitions.mkdir(parents=True)
    for name in ('demo.json','protocol.json'):(definitions/name).write_bytes((source/'experiments/pcv-delivery-v1'/name).read_bytes())
    commands=json.loads((definitions/'demo.json').read_text())['commands']
    refs=root/'experiments/pcv-synthetic-v2';refs.mkdir()
    (refs/'verbatims.jsonl').write_text(''.join(canonical(r)+'\n' for r in commands),encoding='utf-8')
    cache=tmp_path/'cache';(cache/'final-test').mkdir(parents=True)
    rows=[];recordings={e:[] for e in ('qwen','parler','dia')}
    for reference in commands:
        for engine in ('qwen','parler'):
            for voice in ('test-a','test-b'):
                rid=f'{engine}/{voice}/{reference["id"]}'
                rows.append(dict(recording_id=rid,verbatim_id=reference['id'],engine=engine,voice_group=voice,
                    expected_command=reference['expected_command'],candidate={'intent':'TIME_QUERY'} if reject else reference['expected_command'],
                    output=dict(accepted=not reject,command=None if reject else reference['expected_command'],reason='fixture'),audio_sha256='artificial'))
                recordings[engine].append(dict(recording_id=rid,audio=rid+'.wav',audio_sha256='artificial'))
    for e,items in recordings.items():(cache/f'{e}-manifest.jsonl').write_text(''.join(canonical(r)+'\n' for r in items),encoding='utf-8')
    predictions=cache/'final-test/predictions.jsonl';predictions.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    write_json(cache/'final-test/results.json',dict(binding='artificial',complete=True,predictions_sha256=file_hash(predictions)))
    monkeypatch.setattr(delivery,'ROOT',root)
    return cache


def test_declared_demo_preserves_all_phrase_denominators_and_audio_links(tmp_path,monkeypatch):
    cache=fixture_study(tmp_path,monkeypatch);out=tmp_path/'report'
    result=delivery.report(cache,out)
    assert result==dict(demo_commands=35,demo_gates_passed=True,mistakes=0)
    measured=json.loads((out/'demo-results.json').read_text())
    assert len(measured['use_cases'])==10
    assert all(r['trials']==r['correct']==4 for r in measured['commands_with_results'])
    assert all('audio' in t for r in measured['commands_with_results'] for t in r['outputs'])
    with pytest.raises(FileExistsError):delivery.report(cache,out)


def test_all_reject_demo_fails_and_lists_verbatim_parsed_intent(tmp_path,monkeypatch):
    cache=fixture_study(tmp_path,monkeypatch,reject=True);out=tmp_path/'report'
    result=delivery.report(cache,out)
    assert not result['demo_gates_passed'] and result['mistakes']==140
    wrong=[json.loads(line) for line in (out/'mistakes.jsonl').read_text().splitlines()]
    assert all(r['verbatim'] and r['parsed_intent']=='TIME_QUERY' and r['audio'] for r in wrong)
