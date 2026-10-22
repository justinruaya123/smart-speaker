"""Exact full-reference OptionB wording tables; no inference-time text lookup."""
import json

from scripts.runtime.optionb_contract import class_index
from scripts.runtime.pcv_schema import baseline_cases, normalize_text

COHORTS = (('test', 'human'), ('test', 'synthetic'), ('holdout', 'human'), ('holdout', 'synthetic'))


def wording_rows(records, references):
    """Retain every definition even where original-recording evidence is absent."""
    output = []
    originals = [r for r in records if r['variant'] == 'original' and r['metric_eligible']]
    for case in baseline_cases():
        index = class_index(case['command'])
        allowed = {normalize_text(case[field]) for field in ('text', 'spoken_text') if case.get(field)}
        for split, kind in COHORTS:
            matched = [r for r in originals if r['split'] == split and r['kind'] == kind and r['class_index'] == index
                       and any(normalize_text(references[split, r['parent_id']].get(field) or '') in allowed
                               for field in ('text', 'reference_text', 'spoken_text'))]
            correct = sum(r['accepted_index'] == index for r in matched)
            output.append(dict(id=case['id'], text=case['text'], command=json.dumps(case['command'], sort_keys=True),
                split=split, kind=kind, metadata_exact_references=len(matched),
                raw_correct=sum(r['raw_index'] == index for r in matched), correct_accepted=correct,
                wrong_accepted=sum(r['accepted_index'] > 0 and r['accepted_index'] != index for r in matched),
                rejected=sum(r['accepted_index'] == 0 for r in matched),
                accepted_recall=correct / len(matched) if matched else None))
    return output


def verify_report(study=None, *, write_receipt=False):
    """Reuse the reviewed independent guards, then replay actual graph decisions."""
    import csv
    import importlib.util
    import numpy as np
    from pathlib import Path
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.test.speaker_demo_suite import read_json, read_jsonl, read_audio, file_digest, pcm_digest, write_new
    from scripts.train.speaker_optionb_tiny import check_bindings
    from scripts.test.speaker_optionb_tiny_demo import STUDY, validated_cached_scores

    study = Path(study) if study is not None else STUDY
    if write_receipt and (study / 'verification.json').exists():
        raise FileExistsError('Preserve the earlier demo verification')
    protocol, freeze, lock = (read_json(study / name) for name in
                             ('protocol.json', 'data-freeze.json', 'selection-before-test.json'))
    run = Path(protocol['outputs']['run'])
    for bindings in (freeze['bindings'], freeze['historical_bindings'], lock['bindings']):
        check_bindings(bindings)

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError('Frozen independent verifier unavailable')
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded

    v2 = load(Path('experiments/speaker-optionb-tiny-direct-ft-v1/verify-results-v2.py'), 'demo_frozen_v2_guards')
    helpers = load(Path('experiments/speaker-optionb-tiny-ft-v1/verify-results.py'), 'demo_frozen_metric_guards')
    data = {s: read_jsonl(run / (s + '.jsonl')) for s in ('dev', 'calibration', 'test', 'holdout')}
    parent = Path(protocol['parent']['run'])
    for split in data:
        assert file_digest(run / (split + '.jsonl')) == file_digest(parent / (split + '.jsonl'))
    for name in ('model.onnx', 'commands.json', 'normalization.json'):
        assert file_digest(run / 'bundle' / name) == file_digest(Path(protocol['parent']['bundle']) / name)
    humans = [r for s in ('dev', 'calibration') for r in data[s] if r['kind'] == 'human']
    scores = validated_cached_scores(humans, read_jsonl(run / 'calibration-predictions.jsonl'))
    selection = read_json(study / 'policy-selection.json')
    expected_grid = {(p, m) for p in protocol['selection']['probability_grid'] for m in protocol['selection']['margin_grid']}
    observed_grid = [(point['policy']['min_probability'], point['policy']['min_margin']) for point in selection['grid']]
    assert len(observed_grid) == len(set(observed_grid)) == len(expected_grid) and set(observed_grid) == expected_grid
    for point in selection['grid']:
        policy = point['policy']
        assert policy['enabled'] is True
        reconstructed = [dict(class_index=r['class_index'], split=r['split'],
                              accepted_index=helpers.predicted(scores[r['id']], policy)[1])
                         for r in humans if r['metric_eligible']]
        metrics = {s: helpers.metrics([r for r in reconstructed if r['split'] == s], 'accepted_index')
                   for s in ('dev', 'calibration')}
        metrics['pooled'] = helpers.metrics(reconstructed, 'accepted_index')
        assert metrics == point['metrics']
        pooled = metrics['pooled']
        rank = (pooled['f1'] or 0., min(metrics[s]['f1'] or 0. for s in ('dev', 'calibration')),
                pooled['precision'] or 0., pooled['recall'] or 0., policy['min_probability'], policy['min_margin'])
        assert tuple(point['rank']) == rank
    best = max(selection['grid'], key=lambda p: tuple(p['rank']))
    assert best['policy'] == selection['policy'] == lock['policy']
    assert best['metrics'] == selection['metrics']
    final = read_json(study / 'results.json')
    final_bindings = v2.verify_final_bindings(study, run, final)
    records, groups = v2.verify_prediction_artifact(run / 'final-predictions.jsonl', data, lock['policy'], final, helpers)
    v2.verify_csv_counts(study, final, group_count=len(groups))
    references = {(r['split'], r['id']): r for s in ('test', 'holdout') for r in data[s]}
    table = wording_rows(records, references)
    table_path = study / 'per-verbatim-cohorts.csv'
    if not table_path.exists():
        if not write_receipt:
            raise FileNotFoundError('Create the exclusive wording report with --write-receipt')
        with table_path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    with table_path.open(encoding='utf-8', newline='') as stream:
        assert list(csv.DictReader(stream)) == [{k: '' if v is None else str(v) for k, v in r.items()} for r in table]
    assert len(table) == len(baseline_cases()) * len(COHORTS)
    runtime = Runtime(run / 'bundle')
    for index in np.random.default_rng(109).choice(len(records), 32, replace=False):
        record = records[int(index)]
        pcm = read_audio(references[record['split'], record['parent_id']])
        silence = np.zeros(16000, dtype=np.float32)
        if record['variant'] in ('silence_prefix_1s', 'silence_both_1s'):
            pcm = np.concatenate([silence, pcm])
        if record['variant'] in ('silence_suffix_1s', 'silence_both_1s'):
            pcm = np.concatenate([pcm, silence])
        assert pcm_digest(pcm) == record['actual_input_pcm_sha256']
        event = runtime.recognize(pcm)
        assert event['actions_enabled'] is False and event['model_qualified'] is False
        assert event['output'] == record['event']['output']
        if 'logits' in event:
            np.testing.assert_allclose(event['logits'], record['event']['logits'], rtol=3e-5, atol=3e-5)
    assert v2.verify_final_bindings(study, run, final) == final_bindings
    for bindings in (freeze['bindings'], freeze['historical_bindings'], lock['bindings']):
        check_bindings(bindings)
    wording_summary = {}
    for split, kind in COHORTS:
        chosen = [r for r in table if r['split'] == split and r['kind'] == kind]
        wording_summary[split + '/' + kind] = dict(definitions=len(chosen), represented=sum(r['metadata_exact_references'] > 0 for r in chosen),
            metadata_reference_matches=sum(r['metadata_exact_references'] for r in chosen),
            correct_accepted=sum(r['correct_accepted'] for r in chosen), wrong_accepted=sum(r['wrong_accepted'] for r in chosen),
            rejected=sum(r['rejected'] for r in chosen))
    extra = [table_path, study / 'policy-selection.json', Path(__file__), Path('scripts/test/test_speaker_optionb_tiny_demo_report.py')]
    receipt = dict(status='verified', optimizer_updates=0, parameters=lock['parameters'], policy=lock['policy'],
        inference_rows=len(records), independent_actual_graph_spot_replays=32, protected_bindings=len(freeze['historical_bindings']),
        exact_canonical_prediction_coverage=True, independently_reconstructed_policy_grid=len(expected_grid),
        waveform_metadata_labels_not_independently_listened=True, model_normalizer_ontology_unchanged=True,
        current_reserved_membership_unchanged=True, installed=False, actions_enabled=False,
        exposed_regression_not_blind=True, wording_rows=len(table), wording_summary=wording_summary,
        wording_is_full_reference_metadata_matching_not_inference_lookup=True,
        bindings={**final_bindings, **{p.as_posix(): file_digest(p) for p in extra}})
    if write_receipt:
        write_new(study / 'verification.json', receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['verify'])
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    verify_report(write_receipt=args.write_receipt)
