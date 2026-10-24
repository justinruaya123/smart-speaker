"""One checked deterministic ZIP writer, independent of deployment code."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(name):
    """Canonical portable member name (shared producer path primitive)."""
    if not isinstance(name, str):
        raise ValueError('Unsafe package path: ' + repr(name))
    parts = PurePosixPath(name).parts
    if (not parts or name != '/'.join(parts) or name.startswith('/') or '\\' in name or ':' in name
            or any(ord(c) < 32 or c in '<>"|?*' or ord(c) == 127 for c in name)
            or any(p in ('.', '..') or p.endswith(('.', ' '))
                   or re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?', p)
                   for p in parts)):
        raise ValueError('Unsafe package path: ' + name)
    return name


def write_archive(entries, output, metadata):
    """Write exact bytes once; derive the inventory instead of trusting callers."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('ZIP outputs are immutable')
    entries = dict(entries)
    seen = {'manifest.json'}
    for name, data in entries.items():
        safe_name(name)
        if name.casefold() in seen:
            raise ValueError('Duplicate/case-colliding or reserved member: ' + name)
        seen.add(name.casefold())
        if not isinstance(data, bytes):
            raise TypeError('Archive payload must be bytes: ' + name)
    for name in entries:
        if any(str(parent).casefold() in seen for parent in PurePosixPath(name).parents if str(parent) != '.'):
            raise ValueError('File/directory collision: ' + name)
    manifest = {**metadata,
                'files': {n: hashlib.sha256(b).hexdigest() for n, b in sorted(entries.items())},
                'sizes': {n: len(b) for n, b in sorted(entries.items())}}
    payload = {**entries, 'MANIFEST.json': (json.dumps(manifest, indent=2, sort_keys=True,
                                                     allow_nan=False) + '\n').encode()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC failed')
    return {'path': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size}
