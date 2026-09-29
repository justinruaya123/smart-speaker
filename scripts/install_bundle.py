"""Restore the versioned diagnostic artifact without overwriting local changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vcm.integrity import digest
from vcm.packaging import BUNDLE_FILES,safe_members


def restore(archive,output,expected):
    archive=Path(archive); output=Path(output)
    if digest(archive)!=expected: raise ValueError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); safe_members(names)
        if len(names)!=len(set(names)): raise ValueError('Duplicate ZIP paths')
        receipt=json.loads(z.read('manifest.json'))
        if set(names)!=set(receipt['files'])|{'manifest.json'}: raise ValueError('Unexpected ZIP entries')
        for name,value in receipt['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=value: raise ValueError('Inner checksum mismatch')
        payload={name:z.read('models/selected/'+name) for name in BUNDLE_FILES}
    for name,data in payload.items():
        p=output/name
        if p.is_symlink(): raise ValueError('Refusing symlink destination')
        if p.exists() and p.read_bytes()!=data: raise FileExistsError(f'Refusing to overwrite {p}')
    output.mkdir(parents=True,exist_ok=True)
    for name,data in payload.items():
        p=output/name
        if not p.exists():
            with p.open('xb') as stream: stream.write(data)
    print('Verified diagnostic bundle restored:',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('models/selected'))
    p.add_argument('--sha256',help='Defaults to adjacent .receipt.json from the versioned release')
    a=p.parse_args(); expected=a.sha256 or json.loads(a.archive.with_suffix('.receipt.json').read_text())['zip_sha256']
    restore(a.archive,a.output,expected)
