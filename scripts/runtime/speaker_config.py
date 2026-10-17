"""One root configuration for the launcher, native worker and gateway."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import tomllib

DEFAULTS = {
    'inference': {'mode':'demo', 'engine':'ncnn', 'intent_threshold':.95, 'slot_threshold':.90, 'threads':1,'profile':'benchmark-compatible'},
    'audio': {'microphone':-1},
    'server': {'host':'127.0.0.1', 'port':3000, 'origin':''},
    'ui': {'debug':True, 'readout':True},
    'actions': {'enabled':True},
    'benchmark': {'cpu':0, 'output':'results'},
    'providers': {'ha_enabled':False, 'ha_url':'', 'ha_entity_ids':'', 'spotify_enabled':False,
                  'spotify_redirect_uri':'', 'spotify_device_id':'', 'local_media_dir':'media', 'data_dir':'data','timezone':'Asia/Manila'},
    'bindings': {'lights':{}, 'songs':{}, 'climate':{}, 'weather':{}, 'media_player':{},'contacts':{},'messages':{}},
}
ENV_KEYS = {
    'INFERENCE_ENGINE':('inference','engine'),
    'RECOGNITION_PROFILE':('inference','profile'),
    'EXECUTION_MODE':('inference','mode'), 'INTENT_THRESHOLD':('inference','intent_threshold'),
    'SLOT_THRESHOLD':('inference','slot_threshold'), 'SMART_SPEAKER_THREADS':('inference','threads'),
    'MICROPHONE_INDEX':('audio','microphone'), 'HOST':('server','host'), 'PORT':('server','port'),
    'APP_ORIGIN':('server','origin'), 'DEBUG_ENABLED':('ui','debug'), 'INFERENCE_READOUT':('ui','readout'),
    'ACTIONS_ENABLED':('actions','enabled'), 'BENCHMARK_CPU':('benchmark','cpu'),
    'BENCHMARK_OUTPUT':('benchmark','output'),
    **{key.upper():('providers',key) for key in DEFAULTS['providers']},
}


def dotenv(path):
    values={}
    if not path.exists():return values
    for number,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'):continue
        if line.startswith('export '):line=line[7:].strip()
        key,sep,value=line.partition('=');key=key.strip();value=value.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',key):
            raise ValueError(f'.env line {number}: expected NAME=value')
        if value.startswith(('"',"'")):
            quote=value[0];end=value.find(quote,1)
            if end<0 or (value[end+1:].strip() and not value[end+1:].strip().startswith('#')):
                raise ValueError(f'.env line {number}: invalid quoted value')
            value=value[1:end]
        else:value=re.split(r'\s+#',value,maxsplit=1)[0].rstrip()
        values[key]=value
    return values


def resolve_config(root, *, environ=None, overrides=None, profile='inference'):
    root=Path(root).resolve();cfg=deepcopy(DEFAULTS)
    if profile in ('benchmark','standalone'):
        cfg['actions']['enabled']=False;cfg['inference']['mode']='qualified'
    path=root/'config.toml'
    if path.exists():
        supplied=tomllib.loads(path.read_text(encoding='utf-8-sig'))
        for section,values in supplied.items():
            if section not in cfg or not isinstance(values,dict):raise ValueError(f'Unknown config section: {section}')
            for key,value in values.items():
                if key not in cfg[section]:raise ValueError(f'Unknown config field: {section}.{key}')
                cfg[section][key]=value
    env=dotenv(root/'.env')
    if not (root/'.env').exists() and (root/'SpeakerUI/.env').exists():
        raise ValueError('Move your private SpeakerUI/.env to .env beside start.py; the legacy file has been preserved.')
    env.update(os.environ if environ is None else environ)
    for name,(section,key) in ENV_KEYS.items():
        if name not in env or env[name]=='':continue
        value=env[name];default=DEFAULTS[section][key]
        try:
            if type(default) is bool:
                if value not in ('true','false'):raise ValueError()
                value=value=='true'
            elif type(default) is int:value=int(value)
            elif type(default) is float:value=float(value)
        except ValueError:raise ValueError(f'Invalid configuration value for {name}') from None
        cfg[section][key]=value
    for key,value in (overrides or {}).items():
        if value is None:continue
        section,field={'engine':('inference','engine'),'threads':('inference','threads'),'port':('server','port'),'mic':('audio','microphone'),'cpu':('benchmark','cpu'),'profile':('inference','profile')}[key]
        cfg[section][field]=value
    if env.get('SMART_SPEAKER_BINDINGS'):
        try: bindings=json.loads(env['SMART_SPEAKER_BINDINGS'])
        except ValueError:raise ValueError('Invalid SMART_SPEAKER_BINDINGS JSON') from None
        if not isinstance(bindings,dict) or set(bindings)-set(cfg['bindings']):raise ValueError('Unknown SMART_SPEAKER_BINDINGS fields')
        cfg['bindings'].update(bindings)
    for section,values in cfg.items():
        for key,value in values.items():
            expected=DEFAULTS[section][key]
            if isinstance(expected,dict):
                def identifier(item):return isinstance(item,str) and bool(item)
                def valid_binding(k,v):
                    if not isinstance(k,str):return False
                    if section=='bindings' and key=='lights' and isinstance(v,list):
                        return 0<len(v)<=100 and all(identifier(i) for i in v) and len(set(v))==len(v)
                    return identifier(v)
                if not isinstance(value,dict) or any(not valid_binding(k,v) for k,v in value.items()):
                    raise ValueError(f'{section}.{key} must map names to nonempty identifiers')
            elif type(expected) is float:
                if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError(f'{section}.{key} must be 0..1')
            elif type(value) is not type(expected):raise ValueError(f'Invalid type for {section}.{key}')
    for section,key,low,high in [('inference','threads',1,8),('audio','microphone',-1,65535),('server','port',1,65535),('benchmark','cpu',0,65535)]:
        if not low<=cfg[section][key]<=high:raise ValueError(f'{section}.{key} must be {low}..{high}')
    if cfg['inference']['mode'] not in ('demo','qualified'):raise ValueError('inference.mode must be demo or qualified')
    if cfg['inference']['engine'] not in ('ncnn','onnxruntime'):raise ValueError('inference.engine must be ncnn or onnxruntime')
    if cfg['inference']['profile'] not in ('benchmark-compatible','owner-protected'):raise ValueError('Unknown recognition profile')
    deployment=root/'deployment.json'
    if deployment.exists() and json.loads(deployment.read_text()).get('actions_enabled') is False:
        cfg['actions']['enabled']=False
    if not cfg['server']['origin']:
        host=cfg['server']['host'];host='127.0.0.1' if host in ('0.0.0.0','::') else host
        cfg['server']['origin']=f"http://{host}:{cfg['server']['port']}"
    for name,(section,key) in ENV_KEYS.items():
        value=cfg[section][key];env[name]=str(value).lower() if type(value) is bool else str(value)
    if not env.get('SPOTIFY_REDIRECT_URI'):env['SPOTIFY_REDIRECT_URI']=cfg['server']['origin']+'/auth/spotify/callback'
    for key in ('TLS_CERT_PATH','TLS_KEY_PATH'):
        if env.get(key):env[key]=str((root/env[key]).resolve())
    for key in ('LOCAL_MEDIA_DIR','DATA_DIR'):
        env[key]=str((root/env[key]).resolve())
    env['SMART_SPEAKER_BINDINGS']=json.dumps(cfg['bindings'])
    return cfg,env


def initialize_config(root):
    root=Path(root);template=root/'config.example.toml';path=root/'config.toml'
    if template.exists() and not path.exists():
        try:
            with path.open('x',encoding='utf8') as stream:stream.write(template.read_text(encoding='utf8'))
        except FileExistsError:pass
