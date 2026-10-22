"""Bounded OptionB-only holdout diagnosis; never fits or selects a policy/model."""
from __future__ import annotations

import argparse
from collections import Counter

import csv
from datetime import datetime
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.runtime.pcv_schema import INTENTS, baseline_cases, canonical
from scripts.runtime.pcv_final_runtime import Runtime, verify_bundle
from pcvcm.decoder import PrefixCTC
from scripts.runtime.speaker_confidence import application_decision
from scripts.test.speaker_demo_suite import file_digest, pcm_digest, read_audio, read_json, read_jsonl, write_new
from scripts.test.speaker_human_scorecard import wilson_interval
from scripts.test.speaker_optionb_acceptance import OPTIONB_COMMANDS, complete_metrics, gate_audit, optionb_catalog, scope_records
from scripts.test.speaker_optionb_holdout import cue_match
from scripts.train.pcv_updated_prepare import spoken

D = ROOT / 'experiments/speaker-optionb-acceptance-v1'
R = ROOT / '.cache/speaker-optionb-acceptance-v1/replay-01'
PARENT = ROOT / 'experiments/speaker-optionb-holdout-v1'
PREDICTIONS = ROOT / '.cache/speaker-optionb-holdout-v1/replay-01'
FULL = ROOT / 'submission/models/full'
ARMS = ('full_catalog_fixed', 'full_catalog_optionb_output_scope', 'optionb_catalog_fixed')
VARIANTS = {'silence_prefix_1s': (16000, 0), 'silence_suffix_1s': (0, 16000),
            'silence_both_1s': (16000, 16000)}
PARITY = dict(atol=1e-7, rtol=1e-6)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verify(bindings):
    for name, digest in bindings.items():
        if file_digest(ROOT / name) != digest:
            raise ValueError('Protected identity changed: ' + name)


def fitting_reference_audit():
    """Read the actual full-model frozen reproduction views, not an invented cache."""
    recipe_path = ROOT / 'submission/reproduction-full/recipe.json'
    recipe = read_json(recipe_path)
    rows_path = recipe_path.parent / recipe['rows_file']
    assert file_digest(rows_path) == recipe['rows_sha256']
    with gzip.open(rows_path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    index = {r['id']: r for r in rows}
    assert len(index) == len(rows)
    views = recipe['views']
    fit_ids = set(views['train'])
    reserved = set().union(*(set(views[s]) for s in ('dev', 'calibration', 'test', 'holdout')))
    assert not fit_ids & reserved
    assert not set(views['normalization']) & reserved
    assert all(index[i]['split'] == 'train' for i in views['normalization'])
    assert {r['id'] for r in read_jsonl(PREDICTIONS / 'full-original.jsonl')} == set(views['holdout'])
    result: dict[str, Any] = dict(recipe=relative(recipe_path), rows=relative(rows_path), rows_digest_verified=True,
                  original_gpu_prepared_cache_present=(ROOT / '.cache/speaker-synthetic-completion-v1/prepared-v1/train.jsonl').is_file(),
                  source='Selected full-model frozen reproduction manifest and ordered views; no fitting audio read or modification.',
                  train_reserved_id_overlap=0, normalization_reserved_id_overlap=0,
                  physical_speaker_aliases_independently_verified=False, views={}, human_train_reserved_overlap={})
    for split in ('train', 'dev', 'calibration', 'test', 'holdout'):
        members = [index[i] for i in views[split]]
        assert len({r['id'] for r in members}) == len(members)
        assert all(r['split'] == split for r in members)
        if split == 'train':
            assert all(r['training_eligible'] is True for r in members)
        human = [r for r in members if r['kind'] == 'human' and r['synthetic'] is False]
        intent_bad, target_bad = [], []
        for row in members:
            command = row['expected']
            intent = INTENTS.index(command['intent']) + 1 if command is not None else 0
            target = int(command['slots']['light_id'] != 'light.all') if command and 'light_id' in command['slots'] else -100
            if type(row['intent_index']) is not int or row['intent_index'] != intent:
                intent_bad.append(row['id'])
            if type(row['target_index']) is not int or row['target_index'] != target:
                target_bad.append(row['id'])
        result['views'][split] = dict(rows=len(members), kind=dict(Counter(r['kind'] for r in members)),
            human_intent_counts=dict(Counter((r['expected'] or {}).get('intent', 'UNKNOWN') for r in human)),
            intent_index_disagreements=len(intent_bad), target_index_disagreements=len(target_bad),
            intent_disagreement_ids=intent_bad, target_disagreement_ids=target_bad,
            ordered_ids_sha256=hashlib.sha256(canonical(views[split]).encode()).hexdigest())
    train_human = [index[i] for i in views['train'] if index[i]['synthetic'] is False]
    for split in ('dev', 'calibration', 'test', 'holdout'):
        other = [index[i] for i in views[split] if index[i]['synthetic'] is False]
        result['human_train_reserved_overlap'][split] = {}
        for field in ('id', 'speaker', 'sha256', 'pcm_sha256'):
            left = {r[field] for r in train_human if r.get(field)}
            right = {r[field] for r in other if r.get(field)}
            overlap = left & right
            assert not overlap, (split, field, sorted(overlap))
            result['human_train_reserved_overlap'][split][field] = len(overlap)
    result['normalization_rows'] = len(views['normalization'])
    return result


def freeze():
    parent = read_json(PARENT / 'selection-before-holdout.json')
    verify(parent['bindings'])
    protocol = read_json(D / 'protocol.json')
    assert len(baseline_cases()) == 93 and len(OPTIONB_COMMANDS) == 31
    recipe, policy, norm, entries = verify_bundle(FULL, diagnostic=True)
    selected = optionb_catalog(entries)
    assert [e for e in selected if e['command'] is None] == [e for e in entries if e['command'] is None]
    records = read_jsonl(PREDICTIONS / 'full-original.jsonl')
    stress = read_jsonl(PREDICTIONS / 'full-silence-stress.jsonl')
    assert len(records) == 196 and len(stress) == 282
    scope_records(records)  # Fail closed before assigning a benchmark scope.
    catalog = dict(format='optionb-frozen-positive-subset-v1', schema_version=3,
                   parent_catalog_sha256=recipe['binding']['catalog_sha256'],
                   entries=selected, acoustic_qualification=False,
                   positive_command_membership='All 31 full typed OptionB commands; every inherited alias retained.',
                   negative_competition='Every inherited null-command entry retained unchanged; no holdout aliases.')
    fit_audit = fitting_reference_audit()
    write_new(D / 'optionb-catalog.json', catalog)
    write_new(D / 'fit-reference-audit.json', fit_audit)
    paths = set(parent['bindings'])
    paths.update(relative(row['audio']) for row in records)
    paths.update(relative(path) for path in (
        D / 'protocol.json', D / 'optionb-catalog.json', D / 'fit-reference-audit.json', Path(__file__),
        PARENT / 'verification.json', PARENT / 'selection-before-holdout.json',
        PREDICTIONS / 'full-original.jsonl', PREDICTIONS / 'full-silence-stress.jsonl',
        ROOT / 'submission/reproduction-full/recipe.json', ROOT / 'submission/reproduction-full/rows.jsonl.gz',
        ROOT / 'scripts/test/speaker_optionb_acceptance.py', ROOT / 'scripts/test/test_speaker_optionb_acceptance.py',
        ROOT / 'scripts/test/speaker_demo_suite.py', ROOT / 'scripts/test/pcv_public_qualification.py',
        ROOT / 'scripts/test/speaker_optionb_holdout.py', ROOT / 'scripts/runtime/pcv_final_runtime.py',
        ROOT / 'scripts/runtime/speaker_confidence.py', ROOT / 'scripts/modules.json',
        ROOT / 'experiments/phrase-contract-v1/pcvcm/decoder.py'))
    bindings = {name: file_digest(ROOT / name) for name in sorted(paths)}
    selection = dict(format='optionb-acceptance-before-replay-v1', created=datetime.now().astimezone().isoformat(),
        protocol_sha256=file_digest(D / 'protocol.json'), bindings=bindings,
        baseline_git_head=subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip(),
        dirty_baseline=True, model_sha256=recipe['binding']['model_sha256'],
        catalog_counts=dict(full_entries=len(entries), restricted_entries=len(selected),
            restricted_positive_entries=sum(e['command'] is not None for e in selected),
            preserved_negative_entries=sum(e['command'] is None for e in selected),
            restricted_typed_commands=len({canonical(e['command']) for e in selected if e['command'] is not None})),
        policy=protocol['policy'], variants=VARIANTS, arms=list(ARMS),
        baseline_parity=PARITY, inference_threads=2, blas_threads=1,
        environment={p: importlib.metadata.version(p) for p in ('numpy', 'onnxruntime', 'soundfile', 'threadpoolctl')},
        training=False, holdout_used_to_fit_or_select=False, fresh_blind_holdout=False,
        original_rows=len(records), stress_rows=len(stress),
        no_truth_dependent_candidate_filter='The one shared restricted catalog is fixed before all recordings; expected labels are used only for scoring.')
    write_new(D / 'selection-before-replay.json', selection)
    print(json.dumps({k: selection[k] for k in ('format', 'catalog_counts', 'original_rows', 'stress_rows', 'model_sha256')}, indent=2))
    print('PROTECTED_INPUTS', len(bindings), flush=True)


def evaluate(runtime, row, pcm, *, arm, catalog_sha):
    event = runtime.recognize(pcm, 16000)
    application = application_decision(event, read_json(D / 'protocol.json')['policy'])
    diagnostic = event.get('diagnostic_candidate') or {}
    clean = {key: value for key, value in row.items() if key not in
             ('candidate', 'diagnostic_candidate', 'application_command', 'application_reason',
              'qualified_command', 'model_sha256', 'timing', 'wall_ms')}
    return {**clean, 'candidate': diagnostic.get('candidate'), 'diagnostic_candidate': diagnostic,
            'application_command': application['command'], 'application_reason': application['reason'],
            'qualified_command': event['output']['command'], 'model_sha256': event['model_sha256'],
            'research_catalog_sha256': catalog_sha, 'arm': arm, 'timing': event['timing']}


def assert_baseline_parity(actual, historical):
    for key in ('id', 'expected', 'candidate', 'application_command', 'application_reason', 'qualified_command'):
        assert actual[key] == historical[key], (actual['id'], key)
    a, b = actual['diagnostic_candidate'], historical['diagnostic_candidate']
    assert a.get('intent_index') == b.get('intent_index')
    for key in ('intent_probability', 'top_intent_probability', 'target_named_probability', 'score', 'margin'):
        if a.get(key) is None or b.get(key) is None:
            assert a.get(key) == b.get(key), (actual['id'], key)
        else:
            assert np.isclose(a[key], b[key], **PARITY), (actual['id'], key, a[key], b[key])
    assert a.get('slot_confidence', {}).keys() == b.get('slot_confidence', {}).keys()
    for key, value in a.get('slot_confidence', {}).items():
        assert np.isclose(value, b['slot_confidence'][key], **PARITY), (actual['id'], key)


def scoped_measure(records, *, output_allowlist=False):
    scoped = scope_records(records, output_allowlist=output_allowlist)
    application = complete_metrics(scoped)
    raw = complete_metrics(scoped, 'candidate')
    m = application['overall']
    m['accepted_recall_wilson95'] = wilson_interval(m['tp'], m['positive_n'])
    m['negative_false_accept_wilson95'] = wilson_interval(m['negative_errors'], m['negative_n'])
    observed = [v['f1'] for v in application['per_intent'].values() if v['reference_n']]
    m['macro_f1_observed_intents'] = float(np.mean(observed)) if observed else None
    m['observed_intents'] = len(observed)
    return dict(application=application, raw=raw,
                reference_scope_counts=dict(Counter(r['reference_scope'] for r in scoped)),
                interpretation='Exact complete-command micro metrics; Wilson intervals are recording-level, not speaker-adjusted.')


def summarize(outputs):
    result: dict[str, Any] = dict(arms={}, training=False, threshold_search=False, installation=False,
                  parent_regression_reproduced=True, graph_semantic_heads_unchanged_across_catalogs=True,
                  blind=False, independent_listening=False, qualified=False)
    original = outputs['full_catalog_fixed']['original']
    restricted = outputs['optionb_catalog_fixed']['original']
    result['paired_catalog_changes'] = dict(
        raw=sum(a['candidate'] != b['candidate'] for a, b in zip(original, restricted, strict=True)),
        application=sum(a['application_command'] != b['application_command'] for a, b in zip(original, restricted, strict=True)))
    for arm in ARMS:
        source_arm = 'full_catalog_fixed' if arm == 'full_catalog_optionb_output_scope' else arm
        records = outputs[source_arm]['original']
        stress = outputs[source_arm]['stress']
        allowlist = arm == 'full_catalog_optionb_output_scope'
        result['arms'][arm] = dict(original={}, stress={})
        for kind in ('human', 'synthetic'):
            members = [r for r in records if r['kind'] == kind]
            measured = scoped_measure(members, output_allowlist=allowlist)
            audit = gate_audit(members, read_json(D / 'protocol.json')['policy'])
            write_new(R / (arm + '-' + kind + '-gate-audit.jsonl'), audit.pop('records'), jsonl=True)
            measured['gate_audit_before_output_scope'] = audit
            result['arms'][arm]['original'][kind] = measured
        for variant in VARIANTS:
            members = [r for r in stress if r['variant'] == variant]
            measured = scoped_measure(members, output_allowlist=allowlist)
            audit = gate_audit(members, read_json(D / 'protocol.json')['policy'])
            write_new(R / (arm + '-' + variant + '-gate-audit.jsonl'), audit.pop('records'), jsonl=True)
            measured['gate_audit_before_output_scope'] = audit
            measured['evidence'] = 'Paired artificial silence probe; not independent original-human evidence.'
            result['arms'][arm]['stress'][variant] = measured
    return result


def export_tables(outputs):
    per_intent: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    verbatims: list[dict[str, Any]] = []
    definitions = baseline_cases()
    for arm in ARMS:
        source = outputs['full_catalog_fixed' if arm == 'full_catalog_optionb_output_scope' else arm]['original']
        all_scoped = scope_records(source, output_allowlist=arm == 'full_catalog_optionb_output_scope')
        for kind in ('human', 'synthetic'):
            rows = [r for r in all_scoped if r['kind'] == kind and r['metric_eligible']]
            raw = complete_metrics(rows, 'candidate')
            accepted = complete_metrics(rows)
            for intent in INTENTS:
                m = accepted['per_intent'][intent]
                per_intent.append(dict(arm=arm, kind=kind, intent=intent, reference_n=m['reference_n'],
                    raw_correct=raw['per_intent'][intent]['complete_correct'], accepted_correct=m['tp'],
                    fp=m['fp'], fn=m['fn'], precision=m['complete_precision'], recall=m['complete_recall'], f1=m['f1']))
            for signature in sorted(OPTIONB_COMMANDS):
                command = json.loads(signature)
                refs = [r for r in rows if r['expected'] == command]
                tp = sum(r['application_command'] == command for r in refs)
                predictions = sum(r['application_command'] == command for r in rows)
                fp, fn = predictions - tp, len(refs) - tp
                denominator = 2 * tp + fp + fn
                commands.append(dict(arm=arm, kind=kind, command=signature, reference_n=len(refs),
                    raw_correct=sum(r['candidate'] == command for r in refs), accepted_correct=tp, fp=fp, fn=fn,
                    precision=tp / predictions if predictions else None,
                    recall=tp / len(refs) if refs else None, f1=2 * tp / denominator if denominator else None))
            for definition in definitions:
                refs = [r for r in rows if r['reference_scope'] == 'optionb' and r['expected'] == definition['command']]
                exact = [r for r in refs if spoken(r.get('text', '')) == spoken(definition['spoken_text'])]
                cue = [r for r in refs if cue_match(r.get('text', ''), definition)]
                verbatims.append(dict(arm=arm, kind=kind, case_id=definition['id'], verbatim=definition['text'],
                    command=canonical(definition['command']), exact_reference_n=len(exact),
                    raw_correct=sum(r['candidate'] == r['expected'] for r in exact),
                    accepted_correct=sum(r['application_command'] == r['expected'] for r in exact),
                    wrong_accepted=sum(r['application_command'] is not None and r['application_command'] != r['expected'] for r in exact),
                    cue_compatible_n=len(cue), cue_correct_accepted=sum(r['application_command'] == r['expected'] for r in cue),
                    untested=not exact))
    assert len(per_intent) == len(ARMS) * 2 * len(INTENTS)
    assert len(commands) == len(ARMS) * 2 * 31
    assert len(verbatims) == len(ARMS) * 2 * 93
    for name, rows in (('per-intent.csv', per_intent), ('per-command.csv', commands), ('per-verbatim.csv', verbatims)):
        with (D / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return dict(per_intent=len(per_intent), per_command=len(commands), per_verbatim=len(verbatims))


def replay():
    selection = read_json(D / 'selection-before-replay.json')
    verify(selection['bindings'])
    records = read_jsonl(PREDICTIONS / 'full-original.jsonl')
    historical_stress = read_jsonl(PREDICTIONS / 'full-silence-stress.jsonl')
    stress_index = {r['id']: r for r in historical_stress}
    assert len(stress_index) == len(historical_stress)
    outputs, raw_files = {}, {}
    R.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=selection['blas_threads'], user_api='blas'):
        for arm in ('full_catalog_fixed', 'optionb_catalog_fixed'):
            runtime = Runtime(FULL, diagnostic=True, threads=selection['inference_threads'])
            if arm == 'optionb_catalog_fixed':
                runtime.entries = read_json(D / 'optionb-catalog.json')['entries']
                runtime.decoder = PrefixCTC([e['phones'] for e in runtime.entries])
                catalog_sha = file_digest(D / 'optionb-catalog.json')
            else:
                catalog_sha = runtime.recipe['binding']['catalog_sha256']
            original, stress = [], []
            paths = {name: R / (arm + '-' + name + '.jsonl') for name in ('original', 'stress')}
            with paths['original'].open('x', encoding='utf-8') as stream, paths['stress'].open('x', encoding='utf-8') as ss:
                for i, row in enumerate(records, 1):
                    pcm = read_audio(row, ROOT)
                    actual = evaluate(runtime, row, pcm, arm=arm, catalog_sha=catalog_sha)
                    if arm == 'full_catalog_fixed':
                        assert_baseline_parity(actual, row)
                    original.append(actual); stream.write(canonical(actual) + '\n')
                    if row['kind'] == 'human' and row['metric_eligible']:
                        for variant, padding in VARIANTS.items():
                            previous = stress_index[row['id'] + '#' + variant]
                            assert previous['parent_id'] == row['id'] and previous['training_eligible'] is False
                            assert previous['parent_pcm_sha256'] == row['pcm_sha256']
                            assert previous['parent_audio_sha256'] == row['sha256']
                            assert list(padding) == previous['padding_samples']
                            modified = np.pad(pcm, padding)
                            assert pcm_digest(modified, 16000) == previous['pcm_sha256']
                            assert previous['audio'] is None and previous['features'] is None
                            trial = evaluate(runtime, previous, modified, arm=arm, catalog_sha=catalog_sha)
                            if arm == 'full_catalog_fixed':
                                assert_baseline_parity(trial, previous)
                            stress.append(trial); ss.write(canonical(trial) + '\n')
                    if i % 32 == 0:
                        stream.flush(); ss.flush(); print(arm, i, '/', len(records), flush=True)
            assert len(original) == selection['original_rows'] and len(stress) == selection['stress_rows']
            outputs[arm] = dict(original=original, stress=stress)
            raw_files[arm] = {name: dict(path=relative(path), sha256=file_digest(path),
                              rows=len(original if name == 'original' else stress)) for name, path in paths.items()}
            del runtime
    for cohort in ('original', 'stress'):
        a, b = outputs['full_catalog_fixed'][cohort], outputs['optionb_catalog_fixed'][cohort]
        for left, right in zip(a, b, strict=True):
            assert left['id'] == right['id'] and left['expected'] == right['expected']
            x, y = left['diagnostic_candidate'], right['diagnostic_candidate']
            assert x.get('intent_index') == y.get('intent_index')
            for key in ('top_intent_probability', 'target_named_probability'):
                if x.get(key) is None or y.get(key) is None:
                    assert x.get(key) == y.get(key)
                else:
                    assert np.isclose(x[key], y[key], **PARITY), (left['id'], key)
    result = summarize(outputs)
    result.update(selection_sha256=file_digest(D / 'selection-before-replay.json'), raw_files=raw_files,
                  csv_rows=export_tables(outputs), protected_inputs_verified=len(selection['bindings']))
    verify(selection['bindings'])
    write_new(D / 'results.json', result)
    write_new(R / 'receipt.json', dict(selection_sha256=result['selection_sha256'], raw_files=raw_files,
        protected_inputs_verified=len(selection['bindings']), original_rows_per_arm=selection['original_rows'],
        stress_rows_per_arm=selection['stress_rows'], parent_baseline_parity=True,
        semantic_head_parity=True, training=False, threshold_search=False, installation=False, actions=False))
    print(json.dumps({arm: value['original']['human']['application']['overall'] for arm, value in result['arms'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'replay'))
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else replay()
