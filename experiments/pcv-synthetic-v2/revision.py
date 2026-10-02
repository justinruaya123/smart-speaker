"""Frozen v2 contract and reduced third-generator assignments; v1 stays intact."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V1=ROOT/'experiments/pcv-synthetic-v1'
OLD_CACHE=ROOT/'.cache/pcv-synthetic-v1'
CACHE=ROOT/'.cache/pcv-synthetic-v2'
PCV=ROOT/'experiments/phrase-contract-v1'
sys.path.insert(0,str(V1));sys.path.insert(0,str(PCV));sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))
from study import VOICES,read_rows,write_rows
from pcvcm.contract import canonical,file_hash,identity,write_json,pronounce,PHONES

SONGS=(('dynamite','BTS'),('butter','BTS'),('stay','BTS'),('stay','Blackpink'),
       ('pink venom','Blackpink'),('how you like that','Blackpink'),('fancy','Twice'),
       ('feel special','Twice'),('cheer up','Twice'),('dance the night away','Twice'))
COLORS=('red','green','blue','white','yellow','orange','purple','pink','cyan','magenta','lime','gold')


def number(n):
    units='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    if n<20:return units[n]
    if n==100:return 'one hundred'
    tens={20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
    return tens[n//10*10]+(' '+units[n%10] if n%10 else '')


def demo_catalog():
    original=json.loads((PCV/'spec/finite_catalogs.json').read_text())
    return dict(version='pcv-synthetic-v2',songs=[dict(id='track.'+re.sub(r'[^a-z0-9]+','_',f'{artist} {title}'.lower()),
        title=title,artist=artist) for title,artist in SONGS],entities=original['entities'],
        colors=list(COLORS),brightness_percent=list(range(101)),
        reminder_minutes=list(range(1,61)),reminder_hours=list(range(1,13)),
        reminder_clock_times=[f'{hour:02}:{minute:02}' for hour in range(24) for minute in (0,30)],
        timezone=original['timezone'],wake_phrase='hey jarvis',free_form_text=False,
        unknown_or_ambiguous_name='reject',runtime_catalog_mutation=False)


def build_verbatims():
    rows=read_rows(V1/'verbatims.jsonl');by_text={r['text'].lower():r for r in rows}
    additions=[]
    def add(uc,text,command,tag,critical=False):
        if text.lower() in by_text:
            previous=by_text[text.lower()]
            if canonical(previous['expected_command'])!=canonical(command):
                if previous['expected_command'] is not None or command is None:
                    raise ValueError(f'Conflicting positive command amendment: {text}')
                # The user widened the domain before any model/test inference.
                # Preserve the v1 file and bind every promoted rejection explicitly.
                for row in rows:
                    if row['parent_id']==previous['parent_id']:
                        row['supersedes_reference_sha256']=identity(row)
                        row.update(expected_command=command,reference_class='supported_positive',critical_zero_accept=False)
            return
        key=f'V2-{uc.replace("-","")}-{tag}'
        for suffix,phase,sentence in (('',1,text),('-N',2,
                f'could you {text}, please' if command else f'{text}, please')):
            if sentence.lower() in by_text:continue
            row=dict(id=key+suffix,parent_id=key,phase=phase,use_case=uc,text=sentence,
                expected_command=command,reference_class='supported_positive' if command else 'unsupported_or_ambiguous',
                critical_zero_accept=critical if command is None else False)
            by_text[sentence.lower()]=row;additions.append(row)
    catalog=demo_catalog();titles=Counter(s['title'] for s in catalog['songs'])
    for i,song in enumerate(catalog['songs']):
        command=dict(intent='MEDIA_PLAY',entity_id='media.main',track_id=song['id'])
        add('UC-01',f'play {song["title"]} by {song["artist"]}',command,f'SONG-{i:02}-ARTIST')
        if titles[song['title']]==1:add('UC-01',f'play {song["title"]}',command,f'SONG-{i:02}-TITLE')
    for i,text in enumerate(['play stay','play butter by Blackpink','play dynamite by Twice',
            'play unknown song','play fancy by BTS','play stay by Twice','play stay by BTS and Blackpink',
            'do not play butter','play butter and dynamite']):
        add('UC-01',text,None,f'SONG-NEG-{i:02}',True)
    for color in COLORS:
        add('UC-04',f'set kitchen lamp to {color}',dict(intent='COLOR_SET',entity_id='light.kitchen',color=color),f'COLOR-{color.upper()}')
    for i,text in enumerate(['turn on bathroom lamp','turn on lamp three','turn kitchen lamp on and off',
            'turn on kitchen lamp and bedroom lamp']):add('UC-03',text,None,f'ENTITY-NEG-{i:02}',True)
    for i,text in enumerate(['set kitchen lamp to one hundred one percent','set kitchen lamp to minus one percent',
            'set kitchen lamp to thirty seven and seventy three percent','set kitchen lamp to infrared',
            'set bedroom lamp to cyan']):add('UC-04',text,None,f'LIGHT-NEG-{i:02}',True)
    for n,unit,seconds in [(n,'minute' if n==1 else 'minutes',n*60) for n in range(1,61)]+[
                           (n,'hour' if n==1 else 'hours',n*3600) for n in range(1,13)]:
        phrase=f'{number(n)} {unit}';tag=f'{unit.upper()}-{n:02}'
        add('UC-05',f'set a timer for {phrase}',dict(intent='TIMER_SET',timer_id='timer.primary',duration_seconds=seconds),f'TIMER-{tag}')
        add('UC-09',f'set a reminder for {phrase}',dict(intent='REMINDER_SET',reminder_id='reminder.timer',delay_seconds=seconds),f'REM-{tag}')
    for hour in range(24):
        for minute in (0,30):
            spoken=number(hour%12 or 12)+(' thirty' if minute else '')+(' a m' if hour<12 else ' p m')
            add('UC-09',f'set a reminder at {spoken}',dict(intent='REMINDER_SET',reminder_id='reminder.timer',
                time_local=f'{hour:02}:{minute:02}',timezone=catalog['timezone']),f'CLOCK-{hour:02}{minute:02}')
    for i,text in enumerate(['set a reminder at seven','set a reminder for zero minutes','set a reminder for minus one minute',
            'set a reminder for sixty one minutes','set a reminder for thirteen hours','set a reminder at thirteen p m',
            'set a reminder for five minutes and two hours','do not set a reminder for ten minutes']):
        add('UC-09',text,None,f'REM-NEG-{i:02}',True)
    result=rows+additions
    if len({r['id'] for r in result})!=len(result):raise ValueError('Duplicate phrase ID')
    return result


class RevisionContract:
    """Exact typed membership for this separately versioned finite extension."""
    def __init__(self,_legacy_root=None,*,rows=None):
        rows=rows if rows is not None else read_rows(HERE/'verbatims.jsonl')
        self.root=HERE;self.rows=rows
        self.commands=list({canonical(r['expected_command']):r['expected_command'] for r in rows if r['expected_command'] is not None}.values())
        self.allowed={canonical(c) for c in self.commands}
        self.intent_use_case={r['expected_command']['intent']:r['use_case'] for r in rows if r['expected_command'] is not None}
        self.sha256=identity(dict(version='pcv-synthetic-v2',verbatims=rows,catalog=demo_catalog()))
    def validate(self,command):
        if not isinstance(command,dict) or canonical(command) not in self.allowed:
            raise ValueError('Not an exact supported v2 typed command')


def lexicon(rows):
    from synthbench import EXTRA_WORDS
    words={**json.loads((PCV/'configs/lexicon.json').read_text())['words'],**EXTRA_WORDS}
    source=ROOT/'data/raw/cmudict/cmudict.dict'
    receipt=json.loads((ROOT/'data/sources/cmudict.source.json').read_text())
    if file_hash(source)!=receipt['files'][0]['sha256']:raise ValueError('CMU dictionary identity changed')
    cmu={}
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith(';;;'):continue
        token,*phones=line.split();token=re.sub(r'\(\d+\)$','',token.lower())
        phones=[re.sub(r'\d','',p) for p in phones]
        if phones and all(p in PHONES for p in phones):cmu.setdefault(token,phones)
    custom={'bts':['B','IY','T','IY','EH','S'],'blackpink':['B','L','AE','K','P','IH','NG','K']}
    needed={token for row in rows for token in re.findall(r"[a-z]+(?:'[a-z]+)?",row['text'].lower())}
    for token in needed-words.keys():
        value=custom.get(token,cmu.get(token))
        if not value:raise ValueError(f'Missing pronunciation: {token}')
        words[token]=value
    return dict(words=words,custom_entries=custom,source_sha256=file_hash(source),acoustic_evidence=False)


def compile_catalog(rows,*,words=None):
    if words is None:
        expected=json.loads((HERE/'protocol.json').read_text())['input_hashes']['lexicon.json']
        if file_hash(HERE/'lexicon.json')!=expected:raise ValueError('Frozen pronunciation lexicon changed')
        words=json.loads((HERE/'lexicon.json').read_text())
    contract=RevisionContract(rows=rows);seen={};entries=[]
    for row in rows:
        phones=pronounce(row['text'],words['words']);key=tuple(phones)
        if key in seen and seen[key]!=canonical(row['expected_command']):raise ValueError('Acoustic command collision')
        seen[key]=canonical(row['expected_command'])
        entries.append(dict(id=row['id'],text=row['text'],use_case=row['use_case'],phones=phones,command=row['expected_command']))
    return dict(version='pcv-synthetic-v2',entries=entries,base_contract_sha256=contract.sha256,
        pronunciation_sha256=identity(words),accent_or_pronunciation_verified=False)


def dia_verbatims(rows):
    def priority(row):
        c=row['expected_command'] or {};text=row['text'].lower()
        edge=(c.get('value') in (0,100,37,73) or c.get('time_local') in ('00:00','12:00') or
              c.get('delay_seconds') in (60,3600,43200) or 'stay' in text or 'butter by blackpink' in text)
        return (not edge,not row['id'].startswith('V2-'),identity(row['id']))
    chosen=[]
    for phase in (1,2):
        for uc in range(1,11):
            for positive,n in ((True,3),(False,2)):
                pool=[r for r in rows if r['phase']==phase and r['use_case']==f'UC-{uc:02}'
                      and (r['expected_command'] is not None)==positive]
                selected=sorted(pool,key=priority)[:n]
                if len(selected)!=n:raise ValueError('Insufficient Dia stratum')
                chosen.extend(r['id'] for r in selected)
    return set(chosen)


def generation_plan(rows):
    selected=dia_verbatims(rows);result=[]
    old_ids={r['id'] for r in read_rows(V1/'verbatims.jsonl')}
    for engine in ('qwen','parler','dia'):
        for voice in VOICES:
            for row in rows:
                if engine=='dia' and row['id'] not in selected:continue
                key=f'{engine}/{voice["id"]}/{row["id"]}'
                reuse=row['id'] in old_ids and engine in ('qwen','parler')
                result.append(dict(recording_id=key,verbatim_id=row['id'],engine=engine,
                    voice_group=voice['id'],split=voice['split'],accent_requested=voice['accent'],accent_verified=False,
                    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16),synthetic=True,
                    audio=str((OLD_CACHE if reuse else CACHE)/'audio'/f'{key}.wav'),
                    native_audio=str((OLD_CACHE if reuse else CACHE)/'native'/f'{key}.wav'),
                    source_version='v1' if reuse else 'v2'))
    return result


def freeze():
    rows=build_verbatims();plan=generation_plan(rows);words=lexicon(rows)
    compile_catalog(rows,words=words)
    from corpus import wake_assignments
    wake_rows,wake_plan=wake_assignments(rows)
    write_rows(HERE/'verbatims.jsonl',rows);write_rows(HERE/'generation-plan.jsonl',plan)
    write_rows(HERE/'wake-verbatims.jsonl',wake_rows);write_rows(HERE/'wake-generation-plan.jsonl',wake_plan)
    write_json(HERE/'demo-catalog.json',demo_catalog());write_json(HERE/'lexicon.json',words)
    write_json(HERE/'contract-amendments.json',[dict(id=r['id'],previous_reference_sha256=r['supersedes_reference_sha256'],
        new_reference_sha256=identity(r),expected_command=r['expected_command'],
        reason='User-authorized parameter domain expansion before any fitting or final test')
        for r in rows if 'supersedes_reference_sha256' in r])
    protocol=json.loads((V1/'protocol.json').read_text())
    protocol.update(version='pcv-synthetic-v2',expected_clips=len(plan),expected_verbatims=len(rows),
        verbatims_sha256=file_hash(HERE/'verbatims.jsonl'),generation_plan_sha256=file_hash(HERE/'generation-plan.jsonl'),
        contract_sha256=RevisionContract(rows=rows).sha256,base_protocol_sha256=file_hash(V1/'protocol.json'),
        dia_clips=600,dia_verbatims=100,third_engine_exhaustive=False,
        wake_clips=len(wake_plan),wake_threshold=.5,wake_release_ms=320,
        wake_release_timeout_seconds=3.,wake_command_timeout_seconds=5.,
        input_hashes={name:file_hash(HERE/name) for name in ('demo-catalog.json','lexicon.json','contract-amendments.json',
            'wake-verbatims.jsonl','wake-generation-plan.jsonl')},
        wake_assets={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'artifacts'/name for name in
            ('wake-melspectrogram.onnx','wake-embedding.onnx','wake-hey-jarvis.onnx','silero-vad.onnx','wake-models.json')]},
        shared_sources={str(p.relative_to(ROOT)):file_hash(p) for p in [*V1.glob('*.py'),*PCV.joinpath('pcvcm').glob('*.py'),
            *ROOT.joinpath('vcm').glob('*.py'),ROOT/'scripts/runtime/wake_models.py']},
        revision_reason='User authorized smaller Dia, timer-style reminders, K-pop catalog, common colors and wake gating')
    protocol.pop('compatible_previous_verbatims',None)
    path=HERE/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise FileExistsError('V2 protocol is frozen')
    write_json(path,protocol)
    CACHE.mkdir(parents=True,exist_ok=True)
    return dict(verbatims=len(rows),commands=len(RevisionContract(rows=rows).commands),clips=len(plan),dia_clips=600,wake_clips=len(wake_plan))


def validate():
    p=json.loads((HERE/'protocol.json').read_text())
    for field,name in [('verbatims_sha256','verbatims.jsonl'),('generation_plan_sha256','generation-plan.jsonl')]:
        if file_hash(HERE/name)!=p[field]:raise ValueError('Frozen v2 input changed')
    if RevisionContract().sha256!=p['contract_sha256']:raise ValueError('V2 contract changed')
    for name,digest in p['input_hashes'].items():
        if file_hash(HERE/name)!=digest:raise ValueError('V2 auxiliary input changed')
    for name,digest in p['wake_assets'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Wake model identity changed')
    for name,digest in p['shared_sources'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Shared v1 implementation changed')
    return p


if __name__=='__main__':print(json.dumps(freeze(),indent=2))
