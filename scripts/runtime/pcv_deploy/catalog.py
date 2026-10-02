"""CSV inventories become finite audio-to-command graphs, never transcripts."""
from collections import Counter
import csv
import re
from pcvcm.contract import canonical,file_hash,identity,pronounce

COLUMNS={'kind','id','name','artist','aliases','capabilities'}
COLORS=('red','green','blue','white','yellow','orange','purple','pink','cyan','magenta','lime','gold')


def number(n):
    units='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    if n<20:return units[n]
    if n==100:return 'one hundred'
    tens={20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
    return tens[n//10*10]+(' '+units[n%10] if n%10 else '')


def name(value):
    value=' '.join(value.lower().split())
    if not re.fullmatch(r"[a-z][a-z '\-]{0,79}",value):raise ValueError('Names require bounded English words')
    return value


class CatalogContract:
    """Canonical JSON keeps integers distinct from booleans, strings and floats."""
    def __init__(self,catalog):
        self.allowed={canonical(r['command']) for r in catalog['entries'] if r['command'] is not None}
        self.sha256=identity(sorted(self.allowed))
    def validate(self,command):
        if not isinstance(command,dict) or canonical(command) not in self.allowed:
            raise ValueError('Unsupported exact typed catalog command')


def compile_csv(path,lexicon,*,max_entries=4096,max_nodes=60000):
    """Compile bounded commands; output is a candidate requiring new calibration.

    lexicon maps words to the fixed 39-phone inventory. Unknown pronunciations
    fail; adding a CSV row never adds an acoustic class or a neural fallback.
    """
    if type(max_entries) is not int or not 1<=max_entries<=4096 or type(max_nodes) is not int or not 1<=max_nodes<=60000:
        raise ValueError('Invalid Pi catalog budget')
    with open(path,newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if set(reader.fieldnames or [])!=COLUMNS or len(reader.fieldnames)!=len(COLUMNS):
            raise ValueError('CSV header must contain exactly: '+','.join(sorted(COLUMNS)))
        rows=list(reader)
    if not rows or len(rows)>128:raise ValueError('Inventory row budget exceeded')
    ids=set();normalized=[]
    for row in rows:
        if None in row or any(v is None for v in row.values()):raise ValueError('Malformed CSV row')
        row={k:v.strip() for k,v in row.items()};kind=row['kind']
        prefix={'song':'track.','playlist':'playlist.','light':'light.'}.get(kind)
        if (prefix is None or not row['id'].startswith(prefix) or not re.fullmatch(r'[a-z][a-z0-9_.-]{0,95}',row['id'])
                or row['id'] in ids):raise ValueError('Invalid or duplicate stable ID')
        ids.add(row['id']);row['name']=name(row['name'])
        row['artist']=name(row['artist']) if row['artist'] else ''
        row['aliases']=sorted({name(a) for a in row['aliases'].split('|') if a.strip()}|{row['name']})
        row['capabilities']=set(row['capabilities'].split('|')) if row['capabilities'] else set()
        if len(row['aliases'])>4:raise ValueError('Alias budget exceeded')
        if kind=='light':
            if row['artist'] or not row['capabilities'] or not row['capabilities']<={'power','brightness','color'}:
                raise ValueError('Invalid light capabilities')
        elif row['capabilities'] or (kind=='playlist' and row['artist']):raise ValueError('Unexpected inventory parameter')
        normalized.append(row)
    title_counts=Counter(alias for row in normalized if row['kind']=='song' for alias in row['aliases'])
    entries={};phones_seen={};prefixes=set()
    def add(text,command,use_case):
        phones=pronounce(text,lexicon)
        previous=entries.get(text)
        if previous is not None:
            if canonical(previous['command'])!=canonical(command):raise ValueError('Conflicting named command')
            return
        key=tuple(phones)
        if key in phones_seen and phones_seen[key]!=canonical(command):raise ValueError('Ambiguous pronunciation requires a distinct alias')
        phones_seen[key]=canonical(command)
        for i in range(1,len(key)+1):prefixes.add(key[:i])
        if len(entries)+1>max_entries or len(prefixes)+1>max_nodes:raise ValueError('Pi catalog budget exceeded')
        entries[text]=dict(id='csv-'+identity(dict(text=text,command=command))[:20],text=text,
            phones=phones,command=command,use_case=use_case)
    for row in sorted(normalized,key=lambda r:r['id']):
        for alias in row['aliases']:
            if row['kind']=='song':
                command=dict(intent='MEDIA_PLAY',entity_id='media.main',track_id=row['id'])
                add('play '+alias,command if title_counts[alias]==1 else None,'UC-01')
                if row['artist']:add(f'play {alias} by {row["artist"]}',command,'UC-01')
            elif row['kind']=='playlist':
                command=dict(intent='MUSIC_PLAYLIST',entity_id='media.main',playlist_id=row['id'])
                add(f'play {alias} playlist',command,'UC-01')
                add(f'play playlist {alias}',command,'UC-01')
            else:
                if 'power' in row['capabilities']:
                    for state in ('on','off'):add(f'turn {state} {alias}',dict(intent='POWER_'+state.upper(),entity_id=row['id']),'UC-03')
                if 'brightness' in row['capabilities']:
                    for n in range(101):add(f'set {alias} to {number(n)} percent',dict(intent='BRIGHTNESS',entity_id=row['id'],value=n),'UC-04')
                if 'color' in row['capabilities']:
                    for color in COLORS:add(f'set {alias} to {color}',dict(intent='COLOR_SET',entity_id=row['id'],color=color),'UC-04')
    result=dict(version='pcv-csv-v1',entries=[entries[k] for k in sorted(entries)],source_csv_sha256=file_hash(path),
        pronunciation_sha256=identity(lexicon),requires_calibration=True,acoustic_generalization_measured=False,
        node_count=len(prefixes)+1,budgets=dict(max_entries=max_entries,max_nodes=max_nodes),
        integration='inventory only; no service calls or device actions')
    result['catalog_sha256']=identity(result)
    return result


def merge_inventory(base,inventory):
    """Replace catalog-backed positive commands, retain other UCs and negatives.

    A new name conflicting with a former explicit rejection needs an explicit
    protocol amendment; never silently convert a frozen negative to a positive.
    The returned candidate has no enabled policy and cannot be hot-loaded.
    """
    dynamic={'MEDIA_PLAY','MUSIC_PLAYLIST','POWER_ON','POWER_OFF','BRIGHTNESS','COLOR_SET'}
    entries={};phone_commands={};ids=set();prefixes=set()
    kept=[r for r in base['entries'] if r['command'] is None or r['command'].get('intent') not in dynamic]
    for row in [*kept,*inventory['entries']]:
        key=tuple(row['phones']);typed=canonical(row['command'])
        if key in phone_commands and phone_commands[key]!=typed:
            raise ValueError('Inventory conflicts with retained command/rejection; requires an explicit protocol amendment')
        unique=(key,typed)
        if unique in entries:continue
        if row['id'] in ids:raise ValueError('Catalog ID collision')
        ids.add(row['id']);phone_commands[key]=typed;entries[unique]=row
        for i in range(1,len(key)+1):prefixes.add(key[:i])
        if len(entries)>4096 or len(prefixes)+1>60000:raise ValueError('Pi catalog budget exceeded')
    result=dict(version='pcv-csv-merged-v1',entries=list(entries.values()),base_catalog_sha256=identity(base),
        inventory_sha256=identity(inventory),requires_calibration=True,acoustic_generalization_measured=False,
        node_count=len(prefixes)+1,removed_intents=sorted(dynamic))
    result['catalog_sha256']=identity(result)
    return result
