"""Standard-library package verification and transactional model-only updates."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        temporary = Path(stream.name)
    try: os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def safe_name(name):
    parts = PurePosixPath(name).parts
    if (not parts or name != '/'.join(parts) or name.startswith('/') or '\\' in name or ':' in name
            or any(p in ('.','..') or p.endswith(('.', ' ')) or re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?',p) for p in parts)):
        raise ValueError('Unsafe package path: ' + name)
    return name


def inspect_archive(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist(); names = [i.filename for i in infos]
        if len(names) > 20000 or sum(i.file_size for i in infos) > 2 * 1024**3:
            raise ValueError('Package exceeds size budget')
        if len({n.casefold() for n in names}) != len(names): raise ValueError('Duplicate/case-colliding member')
        for i in infos:
            safe_name(i.filename)
            if (i.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError('Symlink in package')
        manifest = json.loads(archive.read('MANIFEST.json'))
        if manifest.get('format') != 'smart-speaker-v1': raise ValueError('Unsupported package format')
        if set(names) != set(manifest['files']) | {'MANIFEST.json'}: raise ValueError('Package inventory mismatch')
        for name, expected in manifest['files'].items():
            with archive.open(name) as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
            if h.hexdigest() != expected: raise ValueError('Package checksum mismatch: ' + name)
        return manifest


def verify_install(root):
    root = Path(root).resolve(); manifest = read(root/'MANIFEST.json')
    if manifest.get('format') != 'smart-speaker-v1': raise ValueError('Unsupported installation')
    for name, expected in manifest['files'].items():
        path = (root/safe_name(name)).resolve()
        if not path.is_relative_to(root) or digest(path) != expected: raise ValueError('Installed file changed: ' + name)
    return manifest


def installed_bundle(root):
    root=Path(root).resolve(); state=root/'state/install.json'
    name=read(state)['bundle'] if state.exists() else 'models/default'
    path=(root/safe_name(name)).resolve()
    if not path.is_relative_to(root): raise ValueError('Model pointer escapes installation')
    return path


def apply_update(root, archive_path, current, load_check):
    root=Path(root).resolve(); pointer=root/'state/install.json'
    previous=read(pointer) if pointer.exists() else {'bundle':'models/default','applied_zip':None}
    zip_hash=digest(archive_path)
    if previous.get('applied_zip')==zip_hash:return previous
    update=inspect_archive(archive_path)
    if update.get('startup_action')!='install-model' or update.get('target') not in ('pi','full','inference','benchmark'):
        raise ValueError('ZIP is not a flagged Pi model update')
    if not current.get('runtime_id') or update.get('runtime_id')!=current['runtime_id']:
        raise ValueError('Runtime/UI dependency contract changed. Extract the new full release in a new folder and run start.py.')
    name='.runtime/models/'+zip_hash
    destination=root/name
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        models=[n for n in update['files'] if n.startswith('models/default/')]
        if not models:raise ValueError('No complete model payload')
        for member in models:
            target=destination/member.removeprefix('models/default/')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(member))
    load_check(destination)  # All graph/recipe/policy checks happen before the pointer changes.
    value={'bundle':name,'applied_zip':zip_hash,'previous_bundle':previous['bundle'],'bundle_id':update.get('bundle_id')}
    write(pointer,value)
    return value
