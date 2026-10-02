"""Versioned fixed-keyword experiment; preserve historical data and holdouts."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import re
import csv
import io
import zipfile
from pathlib import Path
import numpy as np
from .synthetic_data import ROOT, HEADS, rows, dump, digest, write_rows
from .levels import NUMBER_WORDS, spoken_number
from .routing import COLORS
from .audio import FrontendConfig, features, load_audio

DATA = ROOT / 'data/keyword-v4'
CACHE = ROOT / '.cache/keyword-v4'
REPORTS = {m: ROOT / 'experiments' / m / 'keyword-v4' for m in ('synthetic', 'mixed')}
FRONTEND = FrontendConfig(mel_bins=80)
KEYWORDS = sorted(NUMBER_WORDS | COLORS | set('''set turn switch put on off brightness bright brighter brighten dim dimmer dimming darker darken light lights lighting lamp lamps volume sound audio up down increase decrease reduce raise lower higher more less louder quieter softer boost to by at level percent percentage per cent a and or not never cancel instead unless if when then after before until in seconds minutes hours point decimal minus negative plus half quarter max maximum minimum full medium double halve all way color colour warm cool play start music resume continue playing playback time date day today current what which whether show settings status how loud much here there do don t broken mathematical discussed weather alarm stop pause'''.split()))
VOCAB = ['<blank>', '<other>', *KEYWORDS]
INDEX = {v: i for i, v in enumerate(VOCAB)}


def tokens(text):
    text = text.lower().replace('%', ' percent ').replace("don't", 'do not').replace('’', "'")
    text = re.sub(r'(?<=\d)\.(?=\d)', ' point ', text)
    text = re.sub(r'-(?=\d)', 'minus ', text)
    def number(m):
        value = int(m[0])
        if value <= 100: return spoken_number(value)
        if value < 1000:
            return spoken_number(value // 100) + ' hundred' + (' and ' + spoken_number(value % 100) if value % 100 else '')
        return 'thousand'
    text = re.sub(r'\b\d+\b', number, text)
    result = []
    for word in re.findall('[a-z]+', text):
        word = word if word in INDEX else '<other>'
        if word != '<other>' or not result or result[-1] != word: result.append(word)
    return result or ['<other>']


def prepare():
    DATA.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    for p in REPORTS.values(): p.mkdir(parents=True, exist_ok=True)
    if (DATA/'protocol.json').exists():
        protocol = json.loads((DATA/'protocol.json').read_text())
        for name, expected in protocol['manifests'].items():
            if digest(DATA/name) != expected: raise ValueError('Frozen manifest changed: '+name)
        return protocol
    human = rows(ROOT/'data/tts/human.jsonl'); synth = rows(ROOT/'data/tts/manifest.jsonl')
    for group in (human, synth):
        for r in group: r.update(task='command', keyword_tokens=tokens(r['transcript']))
    forbidden_speakers = {r['speaker_id'] for r in human if r['split'] != 'train'}
    forbidden_hashes = {r['audio_sha256'] for r in human}
    aux = []; seen = set(forbidden_hashes)
    for r in rows(ROOT/'data/phonetic-v2/manifest.jsonl'):
        if r['split'] != 'train' or r['kind'] != 'word' or r['speaker'] in forbidden_speakers: continue
        sequence = tokens(r['text'])
        if '<other>' in sequence or r['audio_sha256'] in seen: continue
        if r['source'] not in ('FSDD','MLEndSpokenNumerals','MSWC-English','GoogleSpeechCommandsV2'): continue
        seen.add(r['audio_sha256'])
        aux.append({**r, 'speaker_id':r['speaker'], 'transcript':r['text'], 'keyword_tokens':sequence,
                    'task':'word', 'labels':None, 'kind':'human'})
    existing=rows(ROOT/'data/vcm.jsonl');transcription=rows(ROOT/'data/asr.jsonl')
    # A speaker held out by any prior study cannot enter auxiliary training.
    forbidden_speakers.update(r['speaker_id'] for r in existing+transcription if r['split']!='train')
    phonetic=rows(ROOT/'data/phonetic-v2/manifest.jsonl')
    forbidden_speakers.update(r['speaker'] for r in phonetic if r['split'] in ('dev','test','calibration'))
    aux=[r for r in aux if r['speaker_id'] not in forbidden_speakers]
    seen=set(forbidden_hashes)|{r['audio_sha256'] for r in aux}
    added,fluent_receipt=fluent_training(existing,forbidden_speakers,seen)
    # Original connected speech supplies fixed cue supervision, without name outputs.
    context=[];groups=set();existing_ids={r['id'] for r in existing}
    for r in existing+transcription:
        if r['split']!='train' or r['speaker_id'] in forbidden_speakers or r.get('augmentation') or 'components' in r:continue
        if r['source'] not in ('FluentSpeechCommands','SLURP-real','SnipsSmartLights'):continue
        group=r.get('paired_group',r['id'])
        if group in groups:continue
        groups.add(group)
        path=ROOT/r['audio']
        if not path.exists():continue
        h=digest(path)
        if h in seen:continue
        sequence=tokens(r['transcript'])
        if sequence==['<other>']:continue
        seen.add(h);command=r.get('command')
        known=command in HEADS['intent'] and command!='UNKNOWN' and r['id'] in existing_ids
        labels={'intent':command,'ready':'REJECT','operation':'NONE','amount':'NONE','color':'NONE'} if known else None
        context.append({**r,'audio_sha256':h,'keyword_tokens':sequence,'task':'intent' if known else 'context','labels':labels,'kind':'human'})
    for name, group in [('human.jsonl',human),('synthetic.jsonl',synth),('words.jsonl',aux),('context.jsonl',added+context)]: write_rows(DATA/name,group)
    original = {str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'data/tts/human.jsonl', ROOT/'data/tts/manifest.jsonl', ROOT/'data/phonetic-v2/manifest.jsonl']}
    preserved = {str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'pipeline.toml', *sorted((ROOT/'artifacts').glob('*'))] if p.is_file()}
    prior = {str(p.relative_to(ROOT)):digest(p) for mode in REPORTS for p in (ROOT/'experiments'/mode).glob('*.json')}
    protocol = {'version':'fixed-keyword-v4','seed':23,'epochs':48,'batch_size':64,'width':128,
        'vocabulary':VOCAB,'heads':HEADS,'frontend':FRONTEND.__dict__,
        'manifests':{name:digest(DATA/name) for name in ('human.jsonl','synthetic.jsonl','words.jsonl','context.jsonl')},
        'source_manifests':original,'installed_before':preserved,'prior_results_before':prior,
        'architecture':'Two small frequency/time Conv2D layers and two bidirectional LSTM layers; VCM intent/readiness heads plus fixed-word CTC. No general transcript or open names.',
        'training':'Random initialization. Joint sampling: original complete commands 45%, generated commands 25%, original words 10%, additional intent-only speech 15%, connected cue-only speech 5%. Synthetic uses generated audio only.',
        'selection':'Human development selects mixed checkpoints; synthetic development selects synthetic checkpoints. Human development alone selects per-intent confidence thresholds. Same historical test is reused for comparison, not a new untouched confirmation.',
        'checkpoint_score':'0.5 macro family accuracy + 0.5 macro exact finite-command accuracy',
        'acceptance_targets':{'precision':.95,'negative_false_accept_rate':.01,'minimum_accepts':10},
        'no_new_word_test':True,'no_names':True,'wake_boundary':'External wake detector supplies end sample; fixed model sees following speech only.',
        'numbers':'Strict ordered English word grammar, absolute 0..100; UP/DOWN is +/-10. No amount from class priors.',
        'counts':{'human':dict(Counter(r['split'] for r in human)),'synthetic':dict(Counter(r['split'] for r in synth)),'auxiliary_original_words':len(aux),'additional_original_context':dict(Counter(r['task'] for r in added+context))},
        'requested_corpus':fluent_receipt,'prior_cnn_candidate':'experiments/mixed/keyword-v2/development-candidate.json',
        'reference_repository':{'url':'https://github.com/AIWintermuteAI/Speech-to-Intent-Micro','commit':'7e6674117c55145d0071d39768c183e63e917479','use':'Study of shared small Conv2D encoder and separate semantic heads; independently implemented, not a reproduction of its benchmark.'},
        'label_quality':'Inherited silver human semantic labels; TTS intended text; MLEnd word categories. No new acoustic adjudication.'}
    dump(DATA/'protocol.json',protocol)
    for out in REPORTS.values(): dump(out/'protocol.json',protocol)
    for out in REPORTS.values():dump(out/'fluent-incorporation.json',fluent_receipt)
    print(json.dumps(protocol['counts']),flush=True)
    return protocol


def feature_list(data):
    def load(r):
        key=r['audio_sha256']; path=CACHE/'features'/key[:2]/(key+'.npy')
        if path.exists(): return np.load(path)
        old=ROOT/'.cache/keyword-v2/features'/key[:2]/(key+'.npy')
        if old.exists(): return np.load(old)  # Identical 80-bin frontend, audio-SHA cache.
        old=ROOT/'.cache/keyword-v3/features'/key[:2]/(key+'.npy')
        if old.exists(): return np.load(old)
        pcm,_=load_audio(ROOT/r['audio']); array=features(pcm,FRONTEND)
        path.parent.mkdir(parents=True,exist_ok=True); np.save(path,array,allow_pickle=False)
        return array
    with ThreadPoolExecutor(max_workers=6) as pool: return list(pool.map(load,data))


def dataset(mode, split):
    human=[r for r in rows(DATA/'human.jsonl') if r['split']==split]
    synthetic=[r for r in rows(DATA/'synthetic.jsonl') if r['split']==split]
    if mode=='synthetic':return synthetic
    return human+synthetic+rows(DATA/'words.jsonl')+rows(DATA/'context.jsonl') if split=='train' else human


def fluent_training(existing,forbidden_speakers,seen):
    """Reuse identical existing audio and extract only additional safe train rows."""
    archive=ROOT/'.cache/keyword-study-sources/kaggle-fluent-v1.zip'
    source=json.loads(archive.with_suffix('.json').read_text())
    original={r['id']:r for r in existing};added=[];counts=Counter();split_counts={};overlap_checked=0
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();audio={n[n.index('wavs/'):]:n for n in names if 'wavs/' in n and n.endswith('.wav')}
        for split in ('train','valid','test'):
            match=[n for n in names if n.endswith('/'+split+'_data.csv') or n==split+'_data.csv']
            if len(match)!=1:raise ValueError('Missing publisher metadata '+split+': '+str(match))
            content=z.read(match[0]);(DATA/(split+'_data.csv')).write_bytes(content)
            table=list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))));split_counts[split]={'n':len(table),'speakers':len({r['speakerId'] for r in table})}
            for r in table:
                path=r['path'].replace('\\','/').lstrip('./');key='fluent/'+path;speaker='fluent/'+r['speakerId']
                local=ROOT/'data/audio/fluent/fluent_speech_commands_dataset'/path
                if local.exists() and overlap_checked<64:
                    if digest(local)!=__import__('hashlib').sha256(z.read(audio[path])).hexdigest():raise ValueError('Corpus mirror differs: '+key)
                    overlap_checked+=1
                if split!='train' or speaker in forbidden_speakers:counts['excluded_nontrain_or_held_speaker']+=1;continue
                if not local.exists():
                    resolved=local.resolve();base=(ROOT/'data/audio/fluent').resolve()
                    if not resolved.is_relative_to(base):raise ValueError('Unsafe archive path')
                    local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(z.read(audio[path]));counts['new_audio_files']+=1
                h=digest(local)
                if h in seen:counts['already_present_or_duplicate']+=1;continue
                seen.add(h);text=r.get('transcription',r.get('transcript',''))
                if key in original:family=original[key]['command']
                else:
                    action,obj=r['action'],r['object'];family='UNKNOWN'
                    if obj in ('lights','lamp'):
                        family={'activate':'POWER_ON','deactivate':'POWER_OFF','increase':'BRIGHTNESS','decrease':'BRIGHTNESS'}.get(action,'UNKNOWN')
                    elif obj=='volume' and action in ('increase','decrease'):family='VOLUME'
                    elif obj=='music' and action=='activate':family='MEDIA_RESUME' if re.search(r'\b(resume|continue)\b',text,re.I) else 'MEDIA_PLAY'
                labs={'intent':family,'ready':'REJECT','operation':'NONE','amount':'NONE','color':'NONE'}
                added.append({'id':key,'audio':local.relative_to(ROOT).as_posix(),'audio_sha256':h,'speaker_id':speaker,'transcript':text,
                    'labels':labs,'keyword_tokens':tokens(text),'task':'intent','kind':'human','source':'FluentSpeechCommands','split':'train','source_split':split,
                    'license':'Fluent Speech Commands Public License; academic research only','source_intent':'/'.join(r[k] for k in ('action','object','location')),
                    'requested_source':'https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus','readiness_supervised':False})
                counts['additional_intent_training']+=1
    receipt={**source,'publisher_splits':split_counts,'overlapping_audio_hashes_matched':overlap_checked,'counts':dict(counts),
        'incorporation':'Original local FSC plus additional Kaggle publisher-train utterances with held speakers and hashes excluded. Location/name words are filler; no new semantic classes or test corpus.',
        'test_speakers_used_for_training':False}
    print('KAGGLE INCORPORATION',json.dumps(receipt),flush=True)
    return added,receipt
