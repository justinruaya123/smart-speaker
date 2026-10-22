"""Source-reference metadata fixtures; no audio/annotation/model changes."""
from copy import deepcopy
import importlib

import pytest


def module():
    try:
        return importlib.import_module('scripts.test.speaker_optionb_tiny_demo_metadata')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.test.speaker_optionb_tiny_demo_metadata':
            raise
        pytest.fail('Pinned full-transcript wording enrichment is missing')


def fixture():
    location = dict(repository='fixture/repository', revision='revision-exact', file='data/holdout.parquet',
                    sha256='a' * 64, row=7)
    reference = dict(id='original-id', split='holdout', kind='human', synthetic=False,
                     class_index=20, expected=dict(intent='PLAY_MUSIC', slots={}), location=location)
    key = tuple(location[k] for k in ('repository', 'revision', 'file', 'sha256', 'row'))
    metadata = dict(transcript='Play a song', variation='Play music', is_synthetic=0, transcript_source='prompt')
    return {('holdout', 'original-id'): reference}, {key: metadata}


def test_pinned_full_transcript_not_variation_enriches_reporting_only():
    refs, meta = fixture()
    before = deepcopy(refs)
    enriched = module().enrich_references(refs, meta)
    row = enriched['holdout', 'original-id']
    assert row['reference_text'] == 'Play a song'
    assert row['reference_text'] != meta[next(iter(meta))]['variation']
    assert row['reference_text_provenance'] == 'publisher_full_transcript_not_independently_listened'
    assert row['expected'] == before['holdout', 'original-id']['expected']
    assert refs == before and 'reference_text' not in refs['holdout', 'original-id']


@pytest.mark.parametrize('change', ['repository', 'revision', 'row', 'sha256', 'synthetic', 'nontext'])
def test_unmatched_source_identity_or_provenance_is_rejected(change):
    refs, meta = fixture()
    if change in ('repository', 'revision', 'row', 'sha256'):
        refs['holdout', 'original-id']['location'][change] = 8 if change == 'row' else 'different'
    elif change == 'synthetic':
        meta[next(iter(meta))]['is_synthetic'] = 1
    else:
        meta[next(iter(meta))]['transcript'] = 123
    with pytest.raises(ValueError):
        module().enrich_references(refs, meta)
