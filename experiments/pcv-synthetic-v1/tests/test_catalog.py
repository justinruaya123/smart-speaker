import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from study import build_verbatims, generation_plan, validate_plan


def test_fixed_commands_survive_two_phases():
    rows = build_verbatims()
    assert len(rows) == 2310
    assert len({r['id'] for r in rows}) == 2310
    assert len({r['text'] for r in rows}) == 2310
    base = {r['parent_id']: r for r in rows if r['phase'] == 1}
    assert len(base) == 770
    for row in rows:
        assert row['expected_command'] == base[row['parent_id']]['expected_command']
        assert row['use_case'] == base[row['parent_id']]['use_case']
    assert all('could you what' not in r['text'] for r in rows)


def test_every_verbatim_has_three_engines_and_six_sealed_test_clips():
    rows = build_verbatims()
    plan = generation_plan(rows)
    assert len(plan) == 41580
    validate_plan(plan, rows)
    assert plan == generation_plan(rows)


def test_reject_missing_engine_and_cross_split_voice():
    rows = build_verbatims()[:1]
    plan = generation_plan(rows)
    with pytest.raises(ValueError, match='coverage'):
        validate_plan(plan[1:], rows)
    broken = [dict(r) for r in plan]
    broken[-1]['voice_group'] = broken[0]['voice_group']
    with pytest.raises(ValueError, match='voice'):
        validate_plan(broken, rows)


def test_short_media_commands_expand_to_grammatical_sentences():
    rows=build_verbatims()
    for row in rows:
        if row['phase']==2:
            assert 'you next track' not in row['text']
            assert 'to next track' not in row['text']
            assert 'you previous track' not in row['text']
            assert 'to volume up' not in row['text']
            assert 'you volume down' not in row['text']
