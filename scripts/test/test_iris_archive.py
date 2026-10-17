import json
import zipfile
import pytest


def test_shared_publisher_preserves_payload_and_refuses_overwrite(tmp_path):
    from scripts.archive_io import write_archive
    path=tmp_path/'candidate.zip'
    write_archive({'example.txt':b'payload'},path,{'format':'smart-speaker-v1'})
    with zipfile.ZipFile(path) as archive:
        assert archive.read('example.txt')==b'payload'
        assert set(archive.namelist())=={'example.txt','MANIFEST.json'}
        assert set(json.loads(archive.read('MANIFEST.json'))['files'])=={'example.txt'}
    with pytest.raises(FileExistsError):write_archive({'other.txt':b'other'},path,{})


def test_shared_publisher_rejects_escaping_paths(tmp_path):
    from scripts.archive_io import write_archive
    with pytest.raises(ValueError):write_archive({'../outside':b'bad'},tmp_path/'candidate.zip',{})
