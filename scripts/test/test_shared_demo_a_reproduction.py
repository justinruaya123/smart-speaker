"""Portable treated-data and exact-reproduction guards (no corpus required)."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'submission/shared-demo-a/source/reproduce_a.py'


def engine():
    assert MODULE.is_file(), 'Portable baseline-A reproduction engine is missing'
    spec = importlib.util.spec_from_file_location('reproduce_a_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(identifier='one', split='train'):
    return dict(id=identifier, split=split, kind='human', speaker='person-'+split,
                sha256='a'*64, pcm_sha256='b'*64, class_index=0,
                action=0, percent=0, alarm=0, expected=None)


def test_requires_real_portable_engine():
    assert callable(engine().validate_rows)


def test_accepts_complete_unique_split_identity():
    assert engine().validate_rows({'train':[row()], 'dev':[]}) == 1


@pytest.mark.parametrize('field', ['id','sha256','pcm_sha256','speaker'])
def test_rejects_cross_split_identity(field):
    first, second = row(), row('two','dev')
    second.update(sha256='c'*64, pcm_sha256='d'*64)
    second[field] = first[field]
    with pytest.raises(ValueError, match='identity|Duplicate'):
        engine().validate_rows({'train':[first], 'dev':[second]})


def test_rejects_undeclared_split():
    with pytest.raises(ValueError, match='split'):
        engine().validate_rows({'train':[row(split='test')]})


def test_verifies_downloaded_bytes(tmp_path):
    path=tmp_path/'payload'; path.write_bytes(b'actual')
    record=dict(sha256=hashlib.sha256(b'actual').hexdigest(), bytes=6)
    assert engine().verify_file(path,record)==path
    path.write_bytes(b'forged')
    with pytest.raises(ValueError, match='identity'):
        engine().verify_file(path,record)


@pytest.mark.parametrize('name', ['../outside.wav','/absolute.wav','C:/outside.wav','audio/../../escape.wav'])
def test_rejects_unsafe_archive_members(tmp_path,name):
    archive=tmp_path/'audio.tar'
    with tarfile.open(archive,'w') as stream:
        member=tarfile.TarInfo(name); member.size=1
        stream.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe'):
        engine().extract_audio(archive,tmp_path/'out',{})


def test_rejects_missing_archive_audio(tmp_path):
    archive=tmp_path/'empty.tar'
    with tarfile.open(archive,'w'): pass
    with pytest.raises(ValueError,match='Missing'):
        engine().extract_audio(archive,tmp_path/'out',{'a'*64})


def test_rejects_tensor_substitution():
    import torch
    original={'weight':torch.tensor([1.,2.])}
    assert engine().tensor_identity(original)==engine().tensor_identity(original)
    assert engine().tensor_identity(original)!=engine().tensor_identity({'weight':torch.tensor([1.,3.])})


def test_compare_curves_ignores_only_elapsed_time():
    original=[dict(epoch=1,loss=.25,human_correct=3,elapsed_seconds=1)]
    repeat=[dict(epoch=1,loss=.25,human_correct=3,elapsed_seconds=2)]
    assert engine().compare_curves(original,repeat)
    repeat[0]['human_correct']=2
    with pytest.raises(ValueError,match='curve'):
        engine().compare_curves(original,repeat)


@pytest.mark.parametrize('field,value', [('id','substituted'),('class_index',2),('sha256','c'*64),('pcm_sha256','d'*64)])
def test_original_evaluation_reference_cannot_be_substituted(field,value):
    original=row(); actual=dict(original); actual[field]=value
    with pytest.raises(ValueError,match='reference'):
        engine().verify_reference(actual,original)


def test_original_evaluation_reference_positive_case():
    original=row()
    assert engine().verify_reference(dict(original),original)


def test_preparation_source_cannot_change():
    with pytest.raises(ValueError,match='source'):
        engine().verify_preparation_source({'source':{'reference':'old'}},{'reference':'new'})


def test_preparation_source_positive_case():
    assert engine().verify_preparation_source({'source':{'reference':'same'}},{'reference':'same'})


def test_relative_output_is_anchored_to_repository():
    module=engine()
    assert module.output_directory(Path('.cache/run'))==module.ROOT/'.cache/run'


def test_output_cannot_escape_ignored_root():
    with pytest.raises(ValueError,match='output'):
        engine().output_directory(Path('../outside'))


def test_accepts_hash_verified_hf_cache_symlink(tmp_path):
    cache=tmp_path/'cache'; cache.mkdir()
    blob=cache/'blob'; blob.write_bytes(b'actual')
    link=cache/'snapshot'; link.symlink_to(blob)
    record=dict(sha256=hashlib.sha256(b'actual').hexdigest(),bytes=6)
    assert engine().verify_file(link,record,cache_root=cache)==link
    blob.write_bytes(b'forged')
    with pytest.raises(ValueError,match='identity'):
        engine().verify_file(link,record,cache_root=cache)


def test_rejects_cache_symlink_escape(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    outside=tmp_path/'outside';outside.write_bytes(b'actual')
    link=cache/'snapshot';link.symlink_to(outside)
    record=dict(sha256=hashlib.sha256(b'actual').hexdigest())
    with pytest.raises(ValueError,match='identity'):
        engine().verify_file(link,record,cache_root=cache)
