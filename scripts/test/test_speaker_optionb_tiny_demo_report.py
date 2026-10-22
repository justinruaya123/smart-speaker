"""Metadata wording-table fixtures, not listened/acoustic accuracy evidence."""
import importlib

import pytest

from scripts.runtime.optionb_contract import class_index
from scripts.runtime.pcv_schema import baseline_cases


def module():
    try:
        return importlib.import_module('scripts.test.speaker_optionb_tiny_demo_report')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.test.speaker_optionb_tiny_demo_report':
            raise
        pytest.fail('All93 OptionB demo wording report is missing')


def test_all93_definitions_retained_for_four_origin_cohorts_even_without_evidence():
    rows = module().wording_rows([], {})
    cohorts = [('test', 'human'), ('test', 'synthetic'), ('holdout', 'human'), ('holdout', 'synthetic')]
    assert len(rows) == len(baseline_cases()) * len(cohorts)
    assert {(r['id'], r['split'], r['kind']) for r in rows} == {
        (case['id'], split, kind) for case in baseline_cases() for split, kind in cohorts}
    assert all(r['metadata_exact_references'] == 0 and r['accepted_recall'] is None for r in rows)


def test_exact_wording_report_counts_wrong_complete_commands_and_rejections():
    case = baseline_cases()[0]
    index = class_index(case['command'])
    wrong = next(i for i in range(1, 32) if i != index)
    refs, records = {}, []
    for i, (raw, accepted) in enumerate(((index, index), (wrong, wrong), (index, 0))):
        identifier = f'fixture-{i}'
        refs['holdout', identifier] = dict(text=case['text'])
        records.append(dict(parent_id=identifier, split='holdout', kind='human', variant='original',
                            metric_eligible=True, class_index=index, raw_index=raw, accepted_index=accepted))
    rows = module().wording_rows(records, refs)
    row = next(r for r in rows if r['id'] == case['id'] and r['split'] == 'holdout' and r['kind'] == 'human')
    assert row['metadata_exact_references'] == 3
    assert row['raw_correct'] == 2
    assert row['correct_accepted'] == row['wrong_accepted'] == row['rejected'] == 1
    assert row['accepted_recall'] == 1 / 3


def test_publisher_prompt_bucket_is_not_full_spoken_reference_evidence():
    case = baseline_cases()[0]
    index = class_index(case['command'])
    ref = dict(text='a completely different full request', variation=case['text'])
    record = dict(parent_id='fixture', split='holdout', kind='human', variant='original',
                  metric_eligible=True, class_index=index, raw_index=index, accepted_index=index)
    rows = module().wording_rows([record], {('holdout', 'fixture'): ref})
    row = next(r for r in rows if r['id'] == case['id'] and r['split'] == 'holdout' and r['kind'] == 'human')
    assert row['metadata_exact_references'] == 0
