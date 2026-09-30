"""Complete local TTS asset identity, including tokenizer binary files."""
import argparse
import json
from study import CACHE, HERE, ROOT, file_hash, write_json


def folders():
    return {'qwen':ROOT/'.cache/tts/models/Qwen3-TTS-12Hz-1.7B-Base',
        'voice_design':ROOT/'.cache/tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
        'parler':CACHE/'models/parler-tts-large-v1',
        'dia':CACHE/'models/Dia-1.6B-0626','codec':CACHE/'models/dac_44khz'}


def fingerprint(folder):
    return {p.relative_to(folder).as_posix():file_hash(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and not any(part.startswith('.') for part in p.relative_to(folder).parts)}


def freeze_assets():
    receipt={k:{'directory':str(v.relative_to(ROOT)), 'files':fingerprint(v)} for k,v in folders().items()}
    path=HERE/'tts-assets.json'
    if path.exists() and json.loads(path.read_text())!=receipt:
        raise ValueError('TTS asset drift; preserve the original identity')
    write_json(path,receipt)
    return receipt


def verify_assets(engine):
    receipt=json.loads((HERE/'tts-assets.json').read_text())
    names=[engine]+(['codec'] if engine=='dia' else [])
    if engine=='qwen':names.append('voice_design')
    for name in names:
        expected=receipt[name]
        if fingerprint(ROOT/expected['directory'])!=expected['files']:
            raise ValueError(f'TTS weights/tokenizer/config changed: {name}')
    return file_hash(HERE/'tts-assets.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze_assets()
    else:
        for engine in ('qwen','parler','dia'):verify_assets(engine)
