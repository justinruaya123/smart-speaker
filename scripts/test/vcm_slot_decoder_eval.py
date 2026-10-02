"""E19 frozen-model feasibility probe. Original audio; no training or pipeline changes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest
import numpy as np
import soundfile as sf
from vcm_slot_decoder import (ROOT, EmbeddingModel, number_entries, time_entries,
                             catalog_fingerprint, calibrate, eligible, decision)

EXP = ROOT / 'experiments/vcm_slot_decoder'
CACHE = ROOT / '.cache/vcm-slot-decoder-v1'
MODEL_PATHS = ['.cache/speechembedding-v1/dual_kws-audio.onnx', '.cache/speechembedding-v1/dual_kws-text.onnx']
SOURCE_PATHS = ['scripts/models/vcm_slot_decoder.py', 'scripts/test/vcm_slot_decoder_eval.py',
                'scripts/test/vcm_slot_decoder_checks.py', 'experiments/vcm_slot_decoder/run.py',
                'scripts/runtime/stt_runtime.py', 'scripts/train/parameter_study_data.py',
                'scripts/runtime/levels.py', 'train.py', 'scripts/modules.json', 'requirements.lock']
TIME_LABELS = {
    'audio-1489153935.flac': '03:00:pm', 'audio-1490007398.flac': '06:00:am',
    'audio-1490201794.flac': '05:00:am', 'audio-1490204049.flac': '08:00:am',
    'audio-1497879676.flac': '05:00:am', 'audio-1497887325.flac': '06:00:am',
    'audio-1488564008.flac': '06:00:am', 'audio-1497862421-headset.flac': '10:00:am',
    'audio-1502199609-headset.flac': '06:00:am', 'audio-1502200628.flac': '10:00:pm',
    'audio-1502893711.flac': '08:00:pm',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def rows_at(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def all_catalogs():
    saved = read(ROOT / 'data/parameter-study/catalogs.json')
    catalogs = {}
    for variant in ('fixed','expanded'):
        catalogs[variant] = dict(number=number_entries(), time=time_entries())
        for role in ('artist','song','device'):
            catalogs[variant][role] = [dict(id=role+':'+v, role=role, name=v, aliases=[v]) for v in saved[variant][role]]
    return catalogs


def prepare():
    if (EXP/'protocol.json').exists():
        validate_frozen()
        print('Existing immutable E19 inputs verified.'); return
    original = rows_at(ROOT/'data/parameter-study/manifest.jsonl')
    phonetic = rows_at(ROOT/'data/phonetic-v2/manifest.jsonl')
    phone_by_hash = {r['audio_sha256']: r for r in phonetic}
    training = [r for r in original+phonetic if r['split']=='train']
    train_hashes = {r['audio_sha256'] for r in training}
    train_speakers = {r['speaker'] for r in training}
    training_text = '\n'.join(r['text'] for r in training)
    queries, selected = [], {}
    for split in ('dev','test'):
        isolated = Counter()
        candidates = sorted([r for r in original if r['split']==split], key=lambda r:r['id'])
        for row in candidates:
            peer = phone_by_hash.get(row['audio_sha256'])
            if not peer or peer['split'] != split:
                continue
            row_queries = []
            for target in row['targets']:
                role, value = target['role'], target['value']
                if role not in ('number','artist','song','device'):
                    continue
                if role == 'number' and row['kind']=='word':
                    if isolated[value] >= 5:
                        continue
                    isolated[value] += 1
                group = ('isolated_number' if row['kind']=='word' else 'natural_number') if role=='number' else role
                unseen = role in ('artist','song','device') and not re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)', training_text)
                row_queries.append(dict(role=role, reference=f'{role}:{value}', group=group,
                                        value=value, unseen_in_training_transcripts=unseen))
            time_key = TIME_LABELS.get(Path(row['audio']).name)
            if time_key:
                row_queries.append(dict(role='time', reference='time:'+time_key, group='time',
                                        value=time_key, unseen_in_training_transcripts=False))
            for query in row_queries:
                queries.append(dict(query, row_id=row['id'], split=split))
                selected[row['id']] = row
        # Deterministic unrelated real speech, shared across roles; avoid unannotated alarms.
        negatives = [r for r in candidates if not r['targets'] and not any(w in r['text'] for w in ('alarm','timer','brightness','volume'))
                     and phone_by_hash.get(r['audio_sha256'],{}).get('split')==split]
        negatives.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
        for row in negatives[:80]:
            selected[row['id']] = row
            for role in ('number','artist','song','device','time'):
                groups = ['isolated_number','natural_number'] if role=='number' else [role]
                for group in groups:
                    queries.append(dict(row_id=row['id'], split=split, role=role, group=group,
                                        reference=None, value=None, unseen_in_training_transcripts=False))
    audit = dict(recordings=len(selected), conditional_trials=len(queries),
                 source_split_counts=dict(Counter(r['source']+'/'+r['split'] for r in selected.values())),
                 group_split_counts=dict(Counter(q['group']+'/'+q['split'] for q in queries if q['reference'])),
                 time_label_policy='Manual text-label review before predictions; clock slot only; date/context unresolved; no timestamp annotations',
                 heldout_label_review=[dict(id=r['id'],text=r['text'],label=TIME_LABELS[Path(r['audio']).name]) for r in selected.values() if Path(r['audio']).name in TIME_LABELS],
                 synthetic_audio=0, copied_audio=0, training_hash_overlap=[], training_speaker_overlap=[],
                 verified_recordings=[], failures=[])
    hashes, speakers = defaultdict(set), defaultdict(set)
    for row in selected.values():
        path = ROOT / row['audio']
        if row['audio_sha256'] in train_hashes: audit['training_hash_overlap'].append(row['id'])
        if row['speaker'] in train_speakers: audit['training_speaker_overlap'].append(row['id'])
        hashes[row['split']].add(row['audio_sha256']); speakers[row['split']].add(row['speaker'])
        actual = digest(path)
        wave, rate = sf.read(path, dtype='float32')
        if actual != row['audio_sha256'] or not np.isfinite(wave).all() or not len(wave):
            audit['failures'].append(row['id'])
        audit['verified_recordings'].append(dict(id=row['id'],sha256=actual,frames=len(wave),rate=rate))
    audit['dev_test_hash_overlap'] = sorted(hashes['dev'] & hashes['test'])
    audit['dev_test_speaker_overlap'] = sorted(speakers['dev'] & speakers['test'])
    if any(audit[k] for k in ('failures','training_hash_overlap','training_speaker_overlap','dev_test_hash_overlap','dev_test_speaker_overlap')):
        raise ValueError('Selected audio audit failed: '+str({k:audit[k] for k in ('failures','training_hash_overlap','training_speaker_overlap','dev_test_hash_overlap','dev_test_speaker_overlap')}))
    paths = SOURCE_PATHS + MODEL_PATHS + ['data/parameter-study/manifest.jsonl','data/parameter-study/catalogs.json',
             'data/phonetic-v2/manifest.jsonl','data/speechembedding-v1/protocol.json',
             '.cache/speechembedding-v1/dual_kws-training.json', 'STT/alternatives/sources/cmudict.dict']
    protocol = dict(id='E19', version='vcm-slot-decoder-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    purpose='Frozen-model slot-decoder feasibility POC; no integration', owner='current Codex task',
                    requirements=['ME-03','ME-04','ME-06','ME-07','ME-08','UC-01','UC-03','UC-04','UC-06','UC-08'],
                    hypothesis='Compact existing speech/pronunciation embeddings can retrieve structured values and unseen entity IDs with useful precision and rejection.',
                    allowed_paths=['experiments/vcm_slot_decoder','scripts/models/vcm_slot_decoder.py','scripts/test/vcm_slot_decoder*',
                                   '.cache/vcm-slot-decoder-v1','docs/EXPERIMENT-vcm-slot-decoder-v1.md','EXPERIMENTS.md','REQUIREMENTS.md'],
                    implemented_arms=['embedding'], phonetic_arms='awaiting research-only exception; historical evidence only until authorized',
                    training='none; existing random-initialized dual_kws frozen checkpoint; not a controlled architecture-training comparison',
                    evidence='historical-holdout regression with new decoder policy; not new blind confirmation or complete-command/Pi evidence',
                    role_source='annotation-supplied; full utterances, no oracle crops or verified VCM slot boundaries',
                    selection='Each catalog and group: dev only; >=10 accepted, >=95% precision, <=1% false acceptance on >=20 unknown trials; disabled otherwise',
                    split='Preserve shared old splits. Select all natural slot queries, <=5 isolated examples/value/split by ID, 80 unrelated negatives/split by ID hash.',
                    scale='20/100/500/1000/5000/10000 generated text entries; no synthesized speech; timing only',
                    time_support='hours 1..12; minutes 00/15/30/45; explicit AM/PM or clarification; other time forms unsupported',
                    unsupported=['full 0..100 human acoustic coverage','fresh unseen speaker collection','new entity Bedroom Lamp heldout query',
                                 'accents/noise strata','VCM-to-slot timing','Pi latency/RAM','runtime integration','device actions'],
                    ignore_gate='passed: existing commit 794d88e; .cache ignored, protocol and docs visible; no new rules/commit',
                    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    inputs={p:digest(ROOT/p) for p in paths},
                    installed_before={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'artifacts').glob('*') if p.is_file()},
                    protected_before={p:digest(ROOT/p) for p in ('pipeline.toml','publish.py','scripts/modules.json')})
    CACHE.mkdir(parents=True,exist_ok=True)
    (CACHE/'baseline.diff').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    protocol['baseline_diff_sha256'] = digest(CACHE/'baseline.diff')
    write(EXP/'queries.json',queries); write(EXP/'recordings.json',list(selected.values()))
    write(EXP/'catalogs.json',all_catalogs()); write(EXP/'audit.json',audit)
    for name in ('queries.json','recordings.json','catalogs.json','audit.json'):
        protocol['inputs']['experiments/vcm_slot_decoder/'+name] = digest(EXP/name)
    write(EXP/'protocol.json',protocol)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('verified_recordings','heldout_label_review')},indent=2))


def validate_frozen():
    protocol = read(EXP/'protocol.json')
    for path, expected in protocol['inputs'].items():
        if digest(ROOT/path) != expected: raise ValueError('Frozen input changed: '+path)
    return protocol


def outputs_for(model, rows, split):
    outputs = {}
    for row in rows:
        if row['split'] != split: continue
        path = CACHE / 'audio' / (row['audio_sha256']+'.npz')
        if path.exists():
            data = np.load(path)
            outputs[row['id']] = (data['windows'],read(path.with_suffix('.json')))
        else:
            windows, _, timing = model.audio(ROOT/row['audio'])
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,windows=windows); write(path.with_suffix('.json'),timing)
            outputs[row['id']] = (windows,timing)
    return outputs


def score_queries(model, outputs, queries, catalogs, split):
    records = []
    for variant, roles in catalogs.items():
        for role, entries in roles.items():
            start=time.perf_counter(); index=model.index(entries)
            print('index',variant,role,len(entries),'unavailable',len(index['unavailable']),flush=True)
            valid = {e['id'] for e in entries}
            for query in queries:
                if query['split']!=split or query['role']!=role: continue
                start=time.perf_counter(); scores=model.rank(outputs[query['row_id']][0],index)
                records.append(dict(query, variant=variant, method='embedding', scores=scores[:20],
                                    reference=query['reference'] if query['reference'] in valid else None,
                                    original_reference=query['reference'], catalog_version=index['version'],
                                    oov_reference=query['reference'] in index['unavailable'],
                                    decoder_ms=(time.perf_counter()-start)*1000))
    return records


def key(r):
    return '/'.join((r['method'],r['variant'],r['group']))


def summarize(records, gates):
    def metric(items):
        known=[r for r in items if r['reference'] is not None]
        unknown=[r for r in items if r['reference'] is None]
        accept=[r for r in items if gates[key(r)]['passed'] and eligible(r['scores'],gates[key(r)]['threshold'],gates[key(r)]['margin'])]
        correct=sum(r['reference'] is not None and r['scores'][0][0]==r['reference'] for r in accept)
        result=dict(known=len(known),unknown=len(unknown), accepted=len(accept),accepted_correct=correct,
                    accepted_precision=correct/len(accept) if accept else None,
                    exact_recall=correct/len(known) if known else None,
                    unknown_false_accepts=sum(r['reference'] is None for r in accept),
                    known_false_rejects=sum(r['reference'] is not None and r not in accept for r in items))
        for k in (1,3,5,20):
            result['top'+str(k)]=sum(r['reference'] in [p for p,_ in r['scores'][:k]] for r in known)
        result['unknown_false_accept_rate']=result['unknown_false_accepts']/len(unknown) if unknown else None
        result['known_false_reject_rate']=result['known_false_rejects']/len(known) if known else None
        return result
    groups={k:metric([r for r in records if key(r)==k]) for k in sorted({key(r) for r in records})}
    for variant in ('fixed','expanded'):
        for role in ('artist','song','device'):
            selected=[r for r in records if r['variant']==variant and r['role']==role and r['original_reference']]
            for unseen in (False,True):
                groups[f'embedding/{variant}/{role}/'+('unseen' if unseen else 'seen')]=metric([r for r in selected if r['unseen_in_training_transcripts']==unseen])
    for value in range(101):
        items=[r for r in records if r['variant']=='expanded' and r['group']=='natural_number' and r['value']==str(value)]
        if items: groups[f'embedding/expanded/natural_number/value:{value}']=metric(items)
    return groups


def evaluate():
    validate_frozen()
    if (EXP/'selection-before-test.json').exists() or (EXP/'results.json').exists():
        raise ValueError('E19 evaluation is immutable; use a new experiment version for another policy')
    model=EmbeddingModel(); rows=read(EXP/'recordings.json'); queries=read(EXP/'queries.json'); catalogs=read(EXP/'catalogs.json')
    dev_outputs=outputs_for(model,rows,'dev')
    dev=score_queries(model,dev_outputs,queries,catalogs,'dev'); write(CACHE/'dev-predictions.json',dev)
    gates={k:calibrate([r for r in dev if key(r)==k],next(r['catalog_version'] for r in dev if key(r)==k)) for k in sorted({key(r) for r in dev})}
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(EXP/'protocol.json'),
              dev_predictions_sha256=digest(CACHE/'dev-predictions.json'),gates=gates)
    write(EXP/'selection-before-test.json',lock)
    print('Development policies locked before test audio inference.',flush=True)
    test_outputs=outputs_for(model,rows,'test')
    test=score_queries(model,test_outputs,queries,catalogs,'test'); write(CACHE/'test-predictions.json',test)
    result=dict(protocol_sha256=lock['protocol_sha256'], selection_sha256=digest(EXP/'selection-before-test.json'),
                test_predictions_sha256=digest(CACHE/'test-predictions.json'),metrics=summarize(test,gates),
                desktop=dict(platform=platform.platform(),python=sys.version,onnx_threads=2,blas_threads=1,
                             model_bytes=sum(p.stat().st_size for p in model.paths),startup_ms=model.startup_ms),
                no_training=True,no_pipeline_integration=True,complete_command_accuracy=None,pi_measurements=None)
    write(EXP/'results.json',result)
    print(json.dumps(result,indent=2))


def checks():
    from vcm_slot_decoder_checks import SlotDecoderChecks
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SlotDecoderChecks))
    if not result.wasSuccessful(): raise SystemExit(1)


def query(args):
    validate_frozen(); model=EmbeddingModel()
    entries=read(Path(args.catalog)) if args.catalog else read(EXP/'catalogs.json')[args.variant][args.role]
    if isinstance(entries,dict): entries=entries['entities']
    index=model.index(entries); windows,_,timing=model.audio(Path(args.audio).resolve()); scores=model.rank(windows,index)
    group='natural_number' if args.role=='number' else args.role
    gate=read(EXP/'selection-before-test.json')['gates'].get('embedding/'+args.variant+'/'+group,{})
    result=decision(entries,scores,args.intent,args.role,gate)
    print(json.dumps(dict(result,raw_top5=scores[:5],raw_candidates_are_not_commands=True,timing=timing,
                         unavailable=index['unavailable'],conditional_on_caller_intent=True),indent=2))


def benchmark():
    validate_frozen()
    if (EXP/'benchmark.json').exists(): raise ValueError('Benchmark receipt exists; do not overwrite')
    model=EmbeddingModel(); rows=read(EXP/'recordings.json')
    sample=sorted([r for r in rows if r['split']=='dev' and any(t['role']=='artist' for t in r['targets'])],key=lambda r:r['id'])[:4]
    outputs=[model.audio(ROOT/r['audio']) for r in sample]
    scale=[]
    adjectives='amber blue bright calm clear cool dark deep fair fast fine fresh gold good gray green kind light little long new old orange pink quiet red short silver slow small soft still sweet tall warm white wild young'.split()
    nouns='bird book bridge car cat cloud coast day dog door dream field fire flower garden hill home house lake lamp light moon night ocean park path rain river road rock room sky song star stone street sun tree voice water wind wood world'.split()
    phrases=[a+' '+b+' '+c for a in adjectives for b in nouns for c in nouns if b!=c]
    for size in (20,100,500,1000,5000,10000):
        entries=[dict(id=f'synthetic:{i}',role='artist',aliases=[name]) for i,name in enumerate(phrases[:size])]
        start=time.perf_counter(); index=model.index(entries); build=(time.perf_counter()-start)*1000
        times=[]
        for _ in range(5):
            for windows,_,_ in outputs:
                start=time.perf_counter(); model.rank(windows,index); times.append((time.perf_counter()-start)*1000)
        scale.append(dict(entities=size,index_build_ms=build,index_matrix_bytes=index['matrix'].nbytes,
                          index_json_bytes=len(json.dumps(entries).encode()),unavailable=len(index['unavailable']),
                          queries=len(times),rank_p50_ms=float(np.percentile(times,50)),rank_p95_ms=float(np.percentile(times,95))))
        print('scale',scale[-1],flush=True)
    # Updates while the SAME sessions stay loaded. These are diagnostics, not acceptance claims.
    before={str(p.relative_to(ROOT)):digest(p) for p in model.paths}
    catalogs=read(EXP/'catalogs.json'); demos=[]
    test=read(CACHE/'test-predictions.json')
    for role, name in [('device','bedroom lamp'),('artist','adele')]:
        candidates=[r for r in rows if any(t['role']==role and t['value']==name for t in r['targets']) and r['split']=='test']
        if role=='device':
            # Exact phrase exists in training only; never claim this is held out.
            candidates=[r for r in rows_at(ROOT/'data/parameter-study/manifest.jsonl') if 'bedroom lamp' in r['text']][:1]
        old=[e for e in catalogs['fixed'][role] if e['id']!=role+':'+name]
        updated=old+[dict(id=role+':'+name,role=role,name=name,aliases=[name])]
        old_index=model.index(old); new_index=model.index(updated)
        demo=dict(role=role,name=name,old_version=old_index['version'],new_version=new_index['version'],queries=[])
        for row in candidates[:3]:
            windows,_,_=model.audio(ROOT/row['audio'])
            demo['queries'].append(dict(id=row['id'],split=row['split'],text=row['text'],
                                       heldout=row['split']=='test',old_top5=model.rank(windows,old_index)[:5],new_top5=model.rank(windows,new_index)[:5]))
        demo['acceptance']='requires calibration of changed catalog; raw ranks only'
        demos.append(demo)
    after={str(p.relative_to(ROOT)):digest(p) for p in model.paths}
    import psutil
    result=dict(platform=platform.platform(),pi=False,model_bytes=sum(p.stat().st_size for p in model.paths),
                startup_ms=model.startup_ms,rss_bytes=psutil.Process().memory_info().rss,
                sample_ids=[r['id'] for r in sample],frontend_inference_samples=[timing for _,_,timing in outputs],
                scale=scale,dynamic_demo=demos,models_before=before,models_after=after,models_unchanged=before==after,
                caveats=['single process resident RSS, not peak','4 desktop dev recordings; 20 rank timings/size',
                         'generated dictionary-word text catalogs; no large-catalog accuracy claim','no Pi measurements'])
    write(EXP/'benchmark.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','check','evaluate','benchmark','query'])
    parser.add_argument('--audio');parser.add_argument('--intent',default='UNKNOWN')
    parser.add_argument('--role',choices=['number','time','artist','song','device'],default='number')
    parser.add_argument('--variant',choices=['fixed','expanded'],default='expanded');parser.add_argument('--catalog')
    args=parser.parse_args()
    if args.action=='query':
        if not args.audio: parser.error('query requires --audio')
        query(args)
    else: {'prepare':prepare,'check':checks,'evaluate':evaluate,'benchmark':benchmark}[args.action]()
