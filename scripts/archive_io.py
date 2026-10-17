"""One checked deterministic ZIP writer for Iris releases."""
from pathlib import Path
import hashlib
import json
import zipfile

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_archive(entries,output,metadata):
    """One checked ZIP writer shared by endpoint and direct-command packages."""
    from scripts.runtime.speaker_package import safe_name
    output=Path(output)
    if output.exists():raise FileExistsError('ZIP outputs are immutable')
    for name in entries:safe_name(name)
    payload=dict(entries)
    payload['MANIFEST.json']=(json.dumps({**metadata,'files':{n:hashlib.sha256(b).hexdigest() for n,b in entries.items()}},indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;archive.writestr(info,data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
    return {'path':str(output),'sha256':sha(output),'bytes':output.stat().st_size}
