"""Source-fit-gated, action-free replay of frozen exposed OptionB regression.

Run ``python -m scripts.submission.optionb_replay replay --output .cache/.../run``
only after optionb_fit has sealed that same run's successful exact fit. This
module never installs, downloads dependencies, fits, reselects, qualifies or
runs Pi/native/UI/PC benchmarks. It calls the unchanged canonical evaluator
through an additive portable staging view; original locks remain untouched.

A failed attempt is exclusive evidence. The canonical evaluator buffers its
predictions until completion, so a mid-inference exception has no promised
partial-prediction journal; the stage, error, inputs and written files survive.
The ``verify`` operation reconciles stored decisions without reading PCM and
explicitly does not claim another fresh acoustic replay.
"""
from pathlib import Path
from typing import Any
import subprocess

from scripts.submission.common import ROOT, read, sha
from scripts.submission.optionb_reproduce import SNAPSHOT, pinned_path, read_snapshot_rows, verify_snapshot

SPLITS = ('train', 'dev', 'calibration', 'test', 'holdout')
METRIC_HELPER = 'experiments/speaker-optionb-tiny-ft-v1/verify-results.py'
COHORT_HELPER = 'experiments/speaker-optionb-tiny-direct-ft-v1/verify-results-v2.py'
TOLERANCE: dict[str, Any] = dict(atol=3e-5, rtol=3e-5, decisions='exact', metrics='exact')
REPLAY_SOURCES: tuple[str, ...] = (
    'scripts/submission/common.py', 'scripts/submission/optionb_reproduce.py',
    'scripts/submission/optionb_replay.py', 'scripts/test/optionb_release_check.py',
    'scripts/test/optionb_pc_benchmark.py', 'scripts/test/speaker_demo_suite.py',
    'scripts/test/speaker_optionb_tiny_demo_report.py', METRIC_HELPER,
    'scripts/test/test_speaker_optionb_tiny.py',
    'experiments/speaker-optionb-tiny-direct-ft-v1/verify-results-v2.py',
    'experiments/speaker-optionb-tiny-demo-v1/data-freeze.json',
    'scripts/runtime/speaker_worker.py',
    'SpeakerUI/scripts/optionb-source-parity.mjs', 'SpeakerUI/scripts/optionb-source-inputs.mjs',
    'SpeakerUI/scripts/optionb-parity-inputs.mjs',
    'experiments/speaker-optionb-release-v1/protocol-v3.json',
    'experiments/speaker-optionb-tiny-demo-v1/per-verbatim-cohorts-v2.csv',
)


def validate_selection(root, output) -> dict[str, Any]:
    """Read and independently bind the exact-fit receipt; never read PCM here."""
    from scripts.submission.optionb_fit import _assert_bindings, _fit_target
    from scripts.test.speaker_demo_suite import read_json

    root = Path(root).resolve()
    run = _run(root, output)
    snapshot = verify_snapshot(root)
    name = output + '/fit/selection-before-test.json'
    lock = read_json(_fit_target(run, 'fit/selection-before-test.json'))
    bindings = dict(lock['bindings'])
    for relative in bindings:
        pinned_path(root, relative)
        _fit_target(root, relative)
    independent = {path: record['sha256'] for path, record in snapshot['files'].items()}
    independent.update(snapshot['source_bindings'])
    independent[SNAPSHOT] = sha(pinned_path(root, SNAPSHOT))
    for path, digest in independent.items():
        if bindings.get(path) != digest:
            raise ValueError('Exact-fit binding conflicts with snapshot: ' + path)
    from scripts.test.speaker_demo_suite import canonical
    config = snapshot['training']
    steps = (config['human_epochs'] + config['mixed_epochs']) * config['steps_per_epoch']
    contract = dict(format='optionb-source-fit-selection-before-test-v1', status='selection_locked',
        evidence_scope=('software_fixture_not_corpus_reproduction' if snapshot.get('software_fixture_not_corpus_evidence')
                        else 'source_only_frozen_corpus_fit_comparison'),
        snapshot_sha256=independent[SNAPSHOT], selected_epoch=snapshot['selected_epoch'],
        parameters=snapshot['parameters'], policy=snapshot['policy'], optimizer_steps=steps,
        fitting_splits=['train'], checkpoint_selected_on='dev', selected_on=['dev', 'calibration'],
        normalization_split='original_human_train_only', fit_comparison=output + '/fit/fit-comparison.json',
        test_holdout_used_for_selection=False, cached_scores_reused=False,
        heldout_pcm_decoded_or_inferred_before_selection=False, exposed_regression_not_blind=True,
        installed_defaults_changed=False, actions_enabled=False, qualified=False, installed=False)
    for field, expected in contract.items():
        if canonical(lock.get(field)) != canonical(expected):
            raise ValueError('Exact-fit selection contract changed: ' + field)
    required = {output + '/fit/' + p for p in ('fit-comparison.json', 'input-comparison.json')}
    required.update(('scripts/submission/optionb_fit.py', 'scripts/submission/optionb_reproduce.py'))
    if required - bindings.keys():
        raise ValueError('Exact-fit selection has unbound comparison/source evidence')
    _assert_bindings(root, bindings)
    comparison = read_json(pinned_path(root, lock['fit_comparison']))
    input_path = output + '/fit/input-comparison.json'
    checks = comparison.get('training', {}).get('checks', {})
    export = comparison.get('export', {})
    valid = (comparison.get('exact') is True and comparison.get('protected_inputs_unchanged') is True
        and comparison.get('training', {}).get('exact') is True
        and bool(checks) and all(value is True for value in checks.values())
        and export.get('exact') is True and export.get('graph_bytes_equal') is True
        and export.get('policy_equal') is True and export.get('cached_scores_reused') is False
        and export.get('graph_sha256') == independent[snapshot['bundle'] + '/model.onnx']
        and canonical(export.get('selected_policy')) == canonical(snapshot['policy'])
        and comparison.get('snapshot_sha256') == independent[SNAPSHOT]
        and comparison.get('optimizer_steps') == steps
        and canonical(comparison.get('original_training_config')) == canonical(config)
        and comparison.get('input_comparison_sha256') == bindings[input_path]
        and read_json(pinned_path(root, input_path)).get('exact') is True
        and all(comparison.get(field) is False for field in
                ('heldout_pcm_decoded_or_inferred_before_selection', 'test_holdout_used_for_selection',
                 'actions_enabled', 'qualified', 'installed')))
    if not valid:
        raise ValueError('Exact-fit selection requires successful exact input/training/export comparisons')
    bindings[name] = sha(pinned_path(root, name))
    return dict(snapshot=snapshot, lock=lock, bindings=bindings)


def _run(root, output):
    from scripts.submission.optionb_fit import _fit_target
    pinned_path(root, output)
    path = _fit_target(Path(root).resolve(), output)
    parts = path.relative_to(Path(root).resolve()).parts
    if len(parts) < 2 or parts[0] != '.cache':
        raise ValueError('Replay outputs must be a named run under .cache/')
    return path


def _write_checked(run, relative, value, *, jsonl=False):
    from scripts.submission.optionb_fit import _fit_target
    from scripts.test.speaker_demo_suite import canonical, read_json, read_jsonl, write_new
    path = _fit_target(run, relative)
    write_new(path, value, jsonl=jsonl)
    actual = (read_jsonl if jsonl else read_json)(_fit_target(run, relative))
    if canonical(actual) != canonical(value):
        raise ValueError('Replay artifact readback mismatch: ' + relative)
    return path


def _expected_export_bindings(root) -> dict[str, str]:
    """Follow recorded metadata receipts only; never resolve predecessor cache paths."""
    from scripts.submission.optionb_fit import _assert_bindings
    from scripts.test.speaker_demo_suite import read_json
    legacy = 'experiments/speaker-optionb-release-v1/protocol-v3.json'
    study = 'experiments/speaker-optionb-tiny-demo-v1/'
    predecessor = read_json(pinned_path(root, legacy))['bindings']
    bindings = {legacy: sha(pinned_path(root, legacy))}
    for name in ('verification.json', 'verification-wording-v2.json'):
        relative = study + name
        bindings[relative] = predecessor[relative]
    _assert_bindings(root, bindings)
    ordinary = read_json(pinned_path(root, study + 'verification.json'))['bindings']
    wording = read_json(pinned_path(root, study + 'verification-wording-v2.json'))['bindings']
    for name in ('per-command.csv', 'per-intent.csv', 'per-verbatim.csv', 'per-verbatim-cohorts-v2.csv'):
        relative = study + name
        bindings[relative] = (wording if name.endswith('cohorts-v2.csv') else ordinary)[relative]
    _assert_bindings(root, bindings)
    return bindings


def _compare_csv_exports(root, replay) -> dict[str, Any]:
    import csv
    expected_bindings = _expected_export_bindings(root)
    differences, comparisons = [], {}
    for name in ('per-command.csv', 'per-intent.csv', 'per-verbatim.csv', 'per-verbatim-cohorts-v2.csv'):
        relative = 'experiments/speaker-optionb-tiny-demo-v1/' + name
        with pinned_path(root, relative).open(encoding='utf8', newline='') as stream:
            reader = csv.DictReader(stream)
            expected = list(reader)
            fields = reader.fieldnames
        with (replay / name).open(encoding='utf8', newline='') as stream:
            reader = csv.DictReader(stream)
            actual = list(reader)
            actual_fields = reader.fieldnames
        found = _differences(fields, actual_fields, '/csv/' + name + '/columns')
        found.extend(_differences(expected, actual, '/csv/' + name + '/rows'))
        comparisons[name] = not found
        differences.extend(found)
    return dict(csv_comparisons=comparisons, differences=differences, bindings=expected_bindings)


def _reader_bindings(root, checked) -> dict[str, str]:
    """Bind current portable/UI readers, but keep scientific helpers historically pinned."""
    from scripts.submission.optionb_fit import _fit_target
    from scripts.test.speaker_demo_suite import read_json
    bound = set(REPLAY_SOURCES)
    legacy = 'experiments/speaker-optionb-release-v1/protocol-v3.json'
    if legacy in bound:
        bound.update(name for name in read_json(pinned_path(root, legacy))['bindings']
                     if name.startswith('SpeakerUI/'))
    freeze = 'experiments/speaker-optionb-tiny-demo-v1/data-freeze.json'
    scientific = {METRIC_HELPER, COHORT_HELPER, 'scripts/test/test_speaker_optionb_tiny.py'} & bound
    if scientific:
        if freeze not in checked['bindings']:
            raise ValueError('Frozen replay helper manifest is not bound by the exact fit')
        historical = read_json(pinned_path(root, freeze))['historical_bindings']
        for name in scientific:
            expected = {digest for path, digest in historical.items()
                        if path.replace('\\', '/') == name or path.replace('\\', '/').endswith('/' + name)}
            if len(expected) != 1 or sha(pinned_path(root, name)) not in expected:
                raise ValueError('Frozen replay helper identity changed: ' + name)
    if 'scripts/test/test_speaker_optionb_tiny.py' in bound:
        bound.update(_expected_export_bindings(root))
    result = {}
    for name in sorted(bound):
        pinned_path(root, name)
        result[name] = sha(_fit_target(Path(root).resolve(), name))
    return result


def _merge_bindings(*groups) -> dict[str, str]:
    bindings = {}
    for group in groups:
        for name, digest in group.items():
            if name in bindings and bindings[name] != digest:
                raise ValueError('Conflicting replay binding: ' + name)
            bindings[name] = digest
    return bindings


def prepare_evaluation(root, output) -> dict[str, Any]:
    """Preserve a failed hydration attempt; retry only with another source run."""
    from scripts.submission.optionb_fit import _fit_target
    run = _run(root, output)
    failure = _fit_target(run, 'evaluation-preparation-failure.json')
    if failure.exists():
        raise FileExistsError('Preserve the earlier reserved source preparation failure')
    try:
        return _prepare_evaluation(root, output)
    except BaseException as exc:
        _write_checked(run, 'evaluation-preparation-failure.json', dict(
            stage='selection_gate_or_reserved_source_preparation', error_type=type(exc).__name__, error=str(exc),
            selection_gate_written=_fit_target(run, 'evaluation-selection-gate.json').is_file(),
            source_receipt_written=_fit_target(run, 'prepare-test-holdout.json').is_file(),
            actions_enabled=False, qualified=False, installed=False))
        raise


def _prepare_evaluation(root, output) -> dict[str, Any]:
    """Only TEST/HOLDOUT source hydration, after a verified exact fit lock."""
    from scripts.submission.optionb_fit import _assert_bindings, _fit_target
    from scripts.submission.optionb_reproduce import prepare_source
    from scripts.test.speaker_demo_suite import read_json, read_jsonl
    root = Path(root).resolve()
    run = _run(root, output)
    checked = validate_selection(root, output)  # Before any source preparation.
    snapshot = checked['snapshot']
    gate_path = _fit_target(run, 'evaluation-selection-gate.json')
    source_path = _fit_target(run, 'prepare-test-holdout.json')
    receipt_path = _fit_target(run, 'evaluation-preparation.json')
    bindings = _merge_bindings(checked['bindings'], _reader_bindings(root, checked))
    gate = dict(format='optionb-evaluation-selection-gate-v1', bindings=bindings, tolerance=dict(TOLERANCE),
        snapshot_sha256=bindings[SNAPSHOT], selected_epoch=snapshot['selected_epoch'], policy=snapshot['policy'],
        heldout_pcm_decoded_or_inferred_before_selection=False, cached_scores_reused=False,
        test_holdout_used_for_selection=False, exposed_regression_not_blind=True, actions_enabled=False)
    if gate_path.exists():
        if read_json(gate_path) != gate:
            raise ValueError('Evaluation selection gate identity/bindings changed')
    else:
        if source_path.exists() or any(_fit_target(run, f'{role}/{s}.jsonl').exists()
                for role in ('prepared', 'canonical') for s in ('test', 'holdout')):
            raise ValueError('Evaluation preparation predates this selection gate; preserve evidence')
        _write_checked(run, 'evaluation-selection-gate.json', gate)
    _assert_bindings(root, bindings)
    bindings[gate_path.relative_to(root).as_posix()] = sha(gate_path)
    # A reused receipt does not establish current audio path containment.
    # Check both directory and individual links without opening WAV/PCM.
    _fit_target(run, 'audio')
    for split in ('test', 'holdout'):
        for row in read_snapshot_rows(root, snapshot['views'][split]):
            _fit_target(run, 'audio/' + row['sha256'] + '.wav')
    if not receipt_path.exists():
        # The reviewed preparer owns hydration and puts its shard download cache
        # beside this run. Preflight every eventual writer target without PCM I/O.
        downloads = _fit_target(root, (run.parent / 'downloads').relative_to(root).as_posix())
        for split in ('test', 'holdout'):
            for row in read_snapshot_rows(root, snapshot['views'][split]):
                location = row['location']
                if 'auxiliary_file' in location:
                    name = 'auxiliary/' + location['auxiliary_file']
                else:
                    name = location['sha256'] + '.parquet'
                pinned_path(root, (downloads / name).relative_to(root).as_posix())
                _fit_target(downloads, name)
        prepare_source(root, output, splits=('test', 'holdout'))
    source = read_json(source_path)
    counts = {s: snapshot['views'][s]['rows'] for s in ('test', 'holdout')}
    if (source.get('snapshot_sha256') != bindings[SNAPSHOT] or source.get('counts') != counts
            or source.get('wav_and_pcm_verified') is not True
            or source.get('old_audio_or_features_copied') is not False
            or source.get('derived_feature_cache_reused') is not False
            or source.get('heldout_pcm_decoded') is not True
            or source.get('original_reference_order_and_labels_preserved') is not True):
        raise ValueError('Reserved source preparation receipt changed')
    _assert_bindings(root, bindings)
    _assert_bindings(root, source['bindings'])
    for split in ('test', 'holdout'):
        expected = [{**r, 'audio': f"{output}/audio/{r['sha256']}.wav"}
                    for r in read_snapshot_rows(root, snapshot['views'][split])]
        for role in ('prepared', 'canonical'):
            path = _fit_target(run, f'{role}/{split}.jsonl')
            if not expected or read_jsonl(path) != expected:
                raise ValueError('Reserved frozen ordered membership changed: ' + split)
            bindings[path.relative_to(root).as_posix()] = sha(path)
    bindings[source_path.relative_to(root).as_posix()] = sha(source_path)
    receipt = dict(format='optionb-evaluation-preparation-v1', counts=counts, bindings=bindings,
        source_preparation=source_path.relative_to(root).as_posix(),
        selection_gate=gate_path.relative_to(root).as_posix(), cached_scores_reused=False,
        heldout_pcm_decoded_or_inferred_before_selection=False,
        test_holdout_used_for_selection=False, old_audio_or_features_copied=False,
        exposed_regression_not_blind=True, actions_enabled=False, qualified=False, installed=False)
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ValueError('Reserved evaluation preparation identity/bindings changed')
    else:
        _write_checked(run, 'evaluation-preparation.json', receipt)
    _assert_bindings(root, bindings)
    return receipt


def _load_module(root, relative):
    import importlib.util
    spec = importlib.util.spec_from_file_location('optionb_replay_' + Path(relative).stem.replace('-', '_'),
                                                  pinned_path(root, relative))
    if spec is None or spec.loader is None:
        raise ImportError('Frozen replay helper unavailable: ' + relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_results(records, helpers) -> dict[str, Any]:
    """Group bookkeeping only; the original independent helper owns all metrics."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in records:
        groups['/'.join((row['split'], row['kind'], row['variant']))].append(row)
    results = {}
    for key, values in sorted(groups.items()):
        eligible = [row for row in values if row['metric_eligible']]
        negatives = {}
        for scope in ('original_unknown', 'outside_optionb'):
            selected = [row for row in eligible if row['scope'] == scope]
            negatives[scope] = dict(rows=len(selected),
                raw_false_accepts=sum(row['raw_index'] > 0 for row in selected),
                selected_false_accepts=sum(row['accepted_index'] > 0 for row in selected))
        results[key] = dict(total_rows=len(values), ambiguous_unscored=len(values) - len(eligible),
            metrics={arm: helpers.metrics(eligible, field) for arm, field in
                     (('raw', 'raw_index'), ('fixed_095', 'fixed_index'), ('selected', 'accepted_index'))},
            negative_groups=negatives)
    return results


def _differences(expected, actual, path=''):
    """Exact discrete/schema comparisons; tolerance only for runtime score values."""
    import math
    import numpy as np
    from scripts.test.speaker_demo_suite import canonical
    if path.endswith('/latency_ms'):
        valid = type(actual) in (int, float) and math.isfinite(actual) and actual >= 0
        return [] if valid else [dict(path=path, error='Invalid fresh timing', actual=repr(actual))]
    if any(path.endswith('/event/' + name) for name in
           ('logits', 'probabilities', 'command_probability', 'command_margin')):
        try:
            left, right = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
            equal = (left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
                     and np.allclose(left, right, atol=TOLERANCE['atol'], rtol=TOLERANCE['rtol']))
        except (ValueError, TypeError):
            equal = False
        return [] if equal else [dict(path=path, expected=repr(expected), actual=repr(actual),
                                       error='Nonfinite, malformed or out-of-tolerance score')]
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        if set(expected) != set(actual):
            differences.append(dict(path=path, expected_keys=sorted(expected), actual_keys=sorted(actual)))
        for key in sorted(set(expected) & set(actual)):
            differences.extend(_differences(expected[key], actual[key], path + '/' + key))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = [] if len(expected) == len(actual) else [dict(path=path, expected_rows=len(expected), actual_rows=len(actual))]
        for i, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(_differences(left, right, path + '/' + str(i)))
        return differences
    try:
        equal = canonical(expected) == canonical(actual)
    except (TypeError, ValueError):
        equal = False
    return [] if equal else [dict(path=path, expected=repr(expected), actual=repr(actual))]


def compare_predictions(root, snapshot, actual) -> dict[str, Any]:
    """Goldens are expectations only; compare every fresh original/stress record."""
    from scripts.test.speaker_demo_suite import canonical
    if canonical(verify_snapshot(root)) != canonical(snapshot):
        raise ValueError('Frozen snapshot metadata changed before reconciliation')
    expected = read_snapshot_rows(root, snapshot['golden_evaluation_predictions'])
    data = {s: read_snapshot_rows(root, snapshot['views'][s]) for s in ('test', 'holdout')}
    if not expected or any(not rows for rows in data.values()):
        raise ValueError('Empty frozen evaluation cohort')
    helpers, guards = _load_module(root, METRIC_HELPER), _load_module(root, COHORT_HELPER)
    expected_results = _metric_results(expected, helpers)
    final = dict(prediction_rows=len(expected), results=expected_results)
    guards.reconcile_predictions(expected, data, snapshot['policy'], final, helpers)
    differences = _differences(expected, actual, '/predictions')
    actual_results = None
    try:
        actual_results = _metric_results(actual, helpers)
        guards.reconcile_predictions(actual, data, snapshot['policy'], final, helpers)
        differences.extend(_differences(expected_results, actual_results, '/metrics'))
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        differences.append(dict(path='/independent_reconciliation', error_type=type(exc).__name__, error=str(exc)))
    return dict(exact=not differences, prediction_rows=len(actual), expected_prediction_rows=len(expected),
        tolerance=dict(TOLERANCE), differences=differences, results=actual_results,
        expected_results=expected_results, cached_scores_reused=False,
        expectations_source='independently_frozen_snapshot_golden_metadata_not_inference_inputs')


def _write_csv_checked(run, relative, rows):
    import csv
    from scripts.submission.optionb_fit import _fit_target
    path = _fit_target(run, relative)
    if not rows:
        raise ValueError('Empty wording export')
    with path.open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with _fit_target(run, relative).open(encoding='utf8', newline='') as stream:
        actual = list(csv.DictReader(stream))
    if actual != [{k: '' if v is None else str(v) for k, v in row.items()} for row in rows]:
        raise ValueError('Wording CSV readback mismatch')
    return path


def replay_evaluation(root, output, *, source_commit=None) -> dict[str, Any]:
    """Preserve actual failure evidence; never resume or overwrite an inference run."""
    from scripts.submission.optionb_fit import _fit_files, _fit_target
    root = Path(root).resolve()
    run = _run(root, output)
    replay = _fit_target(run, 'replay')
    if replay.exists():
        raise FileExistsError('Preserve the earlier replay, including any failure evidence')
    replay.mkdir(parents=True)
    progress = dict(stage='selection_gate', selection_validated=False, inference_started=False)
    try:
        return _execute_replay(root, output, source_commit=source_commit, progress=progress)
    except BaseException as exc:
        preserved, collection_error = {}, None
        try:
            preserved = {p.relative_to(root).as_posix(): sha(p) for p in _fit_files(replay)}
        except (ValueError, OSError) as collection:
            collection_error = str(collection)
        _write_checked(replay, 'failure.json', dict(**progress,
            error_type=type(exc).__name__, error=str(exc), preserved_bindings=preserved,
            artifact_collection_error=collection_error,
            canonical_evaluator_buffers_predictions_until_complete=True,
            attempted_inference_rows=None if progress['inference_started'] else 0,
            comparison_written=_fit_target(replay, 'comparison.json').is_file(),
            verification_written=_fit_target(replay, 'verification.json').is_file(),
            actions_enabled=False, qualified=False, installed=False,
            exposed_regression_not_blind=True, pi_validation='excluded_by_user'))
        raise


def _execute_replay(root, output, *, source_commit, progress) -> dict[str, Any]:
    """Run the unchanged original evaluator; goldens never supply actual scores."""
    from contextlib import chdir, redirect_stdout
    from scripts.submission.optionb_fit import _assert_bindings, _fit_files, _fit_target
    from scripts.test.speaker_demo_suite import read_json, read_jsonl
    root = Path(root).resolve()
    run = _run(root, output)
    replay = _fit_target(run, 'replay')
    checked = validate_selection(root, output)
    expected_exports = _expected_export_bindings(root)
    progress.update(stage='source_preparation', selection_validated=True)
    preparation = prepare_evaluation(root, output)
    progress['stage'] = 'portable_protocols'
    protocols = write_replay_protocols(root, output, source_commit=source_commit, reuse=True)
    snapshot = checked['snapshot']
    release = read_json(pinned_path(root, protocols['release_protocol']))
    pc = read_json(pinned_path(root, protocols['pc_protocol']))
    bindings = _merge_bindings(release['bindings'], preparation['bindings'], pc['bindings'], expected_exports)
    _assert_bindings(root, bindings)  # Validate the sealed digests before adding current readers.
    for name in (protocols['release_protocol'], protocols['pc_protocol'], output + '/evaluation-preparation.json',
                 METRIC_HELPER, COHORT_HELPER, 'scripts/test/test_speaker_optionb_tiny.py',
                 'scripts/test/speaker_optionb_tiny_demo_report.py'):
        bindings = _merge_bindings(bindings, {name: sha(pinned_path(root, name))})
    _assert_bindings(root, bindings)
    for split in ('test', 'holdout'):
        values = read_jsonl(pinned_path(root, release['manifest_files'][split]))
        path = _write_checked(replay, split + '.jsonl', values, jsonl=True)
        bindings[path.relative_to(root).as_posix()] = sha(path)
    bundle = _fit_target(replay, 'bundle')
    bundle.mkdir()
    for name in ('model.onnx', 'commands.json', 'normalization.json', 'policy.json', 'recipe.json'):
        source = pinned_path(root, snapshot['bundle'] + '/' + name)
        target = _fit_target(replay, 'bundle/' + name)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        digest = sha(target)
        if digest != snapshot['files'][snapshot['bundle'] + '/' + name]['sha256']:
            raise ValueError('Replay bundle copy readback mismatch: ' + name)
        bindings[target.relative_to(root).as_posix()] = digest
    protocol = dict(outputs=dict(run=output + '/replay'), tolerance=dict(TOLERANCE),
        scope='Unchanged canonical evaluator, portable exact-byte inputs; no old cache dependencies')
    path = _write_checked(replay, 'protocol.json', protocol)
    bindings[path.relative_to(root).as_posix()] = sha(path)
    freeze = dict(bindings=dict(bindings), historical_bindings={},
        original_historical_locks_unchanged=True, portable_bindings_replace_only_consumed_paths=True)
    path = _write_checked(replay, 'data-freeze.json', freeze)
    bindings[path.relative_to(root).as_posix()] = sha(path)
    path = _write_checked(replay, 'selection-before-test.json', {**checked['lock'], 'bindings': dict(bindings),
        'source_fit_selection_lock': output + '/fit/selection-before-test.json'})
    bindings[path.relative_to(root).as_posix()] = sha(path)
    _assert_bindings(root, bindings)
    writer = _load_module(root, 'scripts/test/test_speaker_optionb_tiny.py')
    _assert_bindings(root, bindings)
    # The legacy API resolves relative protocol/audio paths from cwd. Confine that
    # compatibility boundary here rather than patching the frozen evaluator.
    progress.update(stage='canonical_inference', inference_started=True)
    with _fit_target(replay, 'canonical-evaluator.log').open('x', encoding='utf8') as log:
        with chdir(root), redirect_stdout(log):
            final = writer.evaluate_candidate(replay)
    _assert_bindings(root, bindings)
    progress['stage'] = 'reconciliation'
    if read_json(_fit_target(replay, 'results.json')) != final:
        raise ValueError('Canonical results readback mismatch')
    records = read_jsonl(_fit_target(replay, 'final-predictions.jsonl'))
    comparison = compare_predictions(root, snapshot, records)
    contract = dict(parameters=snapshot['parameters'], selected_epoch=snapshot['selected_epoch'],
        policy=snapshot['policy'], prediction_rows=len(records),
        predictions_sha256=sha(_fit_target(replay, 'final-predictions.jsonl')),
        selection_sha256=sha(_fit_target(replay, 'selection-before-test.json')),
        frozen_inputs_unchanged=True, exposed_regression_not_blind=True,
        independent_human_rows_are_original_only=True, installed=False, qualified=False)
    comparison['differences'].extend(_differences(contract, {k: final.get(k) for k in contract}, '/result_contract'))
    stable = {key: {field: value for field, value in cohort.items() if field != 'timing'}
              for key, cohort in final['results'].items()}
    comparison['differences'].extend(_differences(comparison['results'], stable, '/canonical_results'))
    references = read_snapshot_rows(root, snapshot['references'])
    metadata = {(r['split'], r['id']): r for r in references}
    expected_keys = {(s, r['id']) for s in ('test', 'holdout')
                     for r in read_snapshot_rows(root, snapshot['views'][s])}
    if len(metadata) != len(references) or set(metadata) != expected_keys:
        raise ValueError('Frozen full-reference wording membership changed')
    report = _load_module(root, 'scripts/test/speaker_optionb_tiny_demo_report.py')
    wording = report.wording_rows(records, metadata)
    golden_wording = report.wording_rows(read_snapshot_rows(root, snapshot['golden_evaluation_predictions']), metadata)
    comparison['differences'].extend(_differences(golden_wording, wording, '/wording'))
    _write_csv_checked(replay, 'per-verbatim-cohorts-v2.csv', wording)
    guards = _load_module(root, COHORT_HELPER)
    guards.verify_csv_counts(replay, final, group_count=len(comparison['expected_results']))
    exports = _compare_csv_exports(root, replay)
    comparison['csv_comparisons'] = exports['csv_comparisons']
    comparison['differences'].extend(exports['differences'])
    comparison['timing_values_compared_to_historical'] = False
    comparison['exact'] = not comparison['differences']
    _write_checked(replay, 'comparison.json', comparison)
    if not comparison['exact']:
        raise ValueError('Complete replay comparison mismatch; evidence preserved')
    _assert_bindings(root, bindings)
    verify_snapshot(root)
    for path in _fit_files(replay):
        bindings[path.relative_to(root).as_posix()] = sha(path)
    receipt = dict(format='optionb-source-inference-replay-v1', exact=True, status='verified',
        evidence_scope=checked['lock']['evidence_scope'], inference_rows=len(records),
        canonical_case_feature_and_decision_path_reused=True, complete_original_and_stress_cohort=True,
        tolerance=dict(TOLERANCE), cached_scores_reused=False, golden_scores_used_for_inference=False,
        heldout_pcm_decoded_or_inferred_before_selection=False, test_holdout_used_for_selection=False,
        exposed_regression_not_blind=True, independent_listening_completed=False,
        fresh_blind_human_check='pending_no_new_recordings', pi_validation='excluded_by_user',
        optimizer_updates=0, selection_changes=0, actions_enabled=False, qualified=False, installed=False,
        installed_defaults_changed=False, native_js_executed=False, pc_benchmark_executed=False,
        release_check_executed=False, release_protocol=protocols['release_protocol'],
        pc_protocol=protocols['pc_protocol'], bindings=bindings,
        wording_rows=len(wording), output=output + '/replay')
    _write_checked(replay, 'verification.json', receipt)
    _assert_bindings(root, bindings)
    return receipt


def write_replay_protocols(root, output, *, source_commit=None, reuse=False) -> dict[str, Any]:
    """Rebind portable paths; derive canonical expectations from Git-tracked views."""
    from scripts.test.speaker_demo_suite import read_jsonl
    from scripts.test.optionb_release_check import validate_inputs
    from scripts.submission.optionb_fit import _fit_target

    root = Path(root).resolve()
    snapshot = verify_snapshot(root)
    destination = _run(root, output)
    release_path = _fit_target(destination, 'locks/release.json')
    pc_path = _fit_target(destination, 'locks/pc.json')
    existing = release_path.exists() or pc_path.exists()
    if existing and (not reuse or not release_path.is_file() or not pc_path.is_file()):
        raise FileExistsError('Preserve earlier portable locks')
    checked = validate_selection(root, output)
    manifests, originals = {}, {}
    for split in SPLITS:
        expected = [{**row, 'audio': f"{output}/audio/{row['sha256']}.wav"}
                    for row in read_snapshot_rows(root, snapshot['views'][split])]
        for role, table in (('prepared', manifests), ('canonical', originals)):
            path = _fit_target(destination, f'{role}/{split}.jsonl')
            if read_jsonl(path) != expected:
                raise ValueError('Portable view differs from frozen ordered membership: ' + split + '/' + role)
            table[split] = path.relative_to(root).as_posix()
    metadata = _fit_target(destination, 'locks/full-reference-metadata.jsonl')
    predictions = _fit_target(destination, 'locks/golden-evaluation-predictions.jsonl')
    for path, rows in ((metadata, read_snapshot_rows(root, snapshot['references'])),
                       (predictions, read_snapshot_rows(root, snapshot['golden_evaluation_predictions']))):
        if existing:
            if read_jsonl(path) != rows:
                raise ValueError('Portable frozen metadata identity changed: ' + str(path))
        else:
            _write_checked(destination, path.relative_to(destination).as_posix(), rows, jsonl=True)
    metadata_relative = metadata.relative_to(root).as_posix()
    predictions_relative = predictions.relative_to(root).as_posix()
    bound = {SNAPSHOT, *snapshot['files'], *snapshot['source_bindings'], *REPLAY_SOURCES,
             *manifests.values(), *originals.values(), metadata_relative, predictions_relative}
    bindings = _merge_bindings(checked['bindings'], _reader_bindings(root, checked),
                              {path: sha(pinned_path(root, path)) for path in sorted(bound)})
    if existing and source_commit is None:
        source_commit = read(release_path)['source_commit']
    if source_commit is None:
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    release = dict(format='optionb-release-freeze-v2', source_commit=source_commit,
        scope='Portable consumed paths; exact frozen OptionB model, policy, references and exposed regression cohorts',
        parameters=snapshot['parameters'], selected_epoch=snapshot['selected_epoch'],
        architecture=snapshot['architecture'], policy=snapshot['policy'],
        bundle=snapshot['bundle'], checkpoint=snapshot['checkpoint'],
        manifest_files=manifests, canonical_manifests=originals,
        canonical_human_counts={split: snapshot['views'][split]['kinds']['human'] for split in ('test', 'holdout')},
        wording_metadata=metadata_relative, cached_predictions=predictions_relative, bindings=bindings,
        snapshot_sha256=sha(pinned_path(root, SNAPSHOT)),
        actions_enabled=False, promoted=False, qualified=False, original_locks_unchanged=True)
    validate_inputs(release, root=root)
    if existing:
        if read(release_path) != release:
            raise ValueError('Portable release protocol identity/bindings changed')
    else:
        _write_checked(destination, 'locks/release.json', release)
    if read(release_path) != release:
        raise ValueError('Portable release protocol readback mismatch')
    release_relative = release_path.relative_to(root).as_posix()
    pc = dict(format='optionb-pc-benchmark-v1', target='pc', threads=1, repeats=3, warmup_calls=10,
        order_seed=53, release_protocol=release_relative, metric_helper=METRIC_HELPER,
        bindings={release_relative: sha(release_path), METRIC_HELPER: sha(pinned_path(root, METRIC_HELPER))},
        scope='Same source-derived original human regression cohort and unchanged CPU timing scope',
        actions_enabled=False, fresh_blind_human_check='pending_no_new_recordings', pi_validation='excluded_by_user')
    if existing:
        if read(pc_path) != pc:
            raise ValueError('Portable PC protocol identity/bindings changed')
    else:
        _write_checked(destination, 'locks/pc.json', pc)
    if read(pc_path) != pc:
        raise ValueError('Portable PC protocol readback mismatch')
    from scripts.submission.optionb_fit import _assert_bindings
    _assert_bindings(root, release['bindings'])
    _assert_bindings(root, pc['bindings'])
    verify_snapshot(root)
    return dict(release_protocol=release_relative, pc_protocol=pc_path.relative_to(root).as_posix(),
                canonical_counts=release['canonical_human_counts'], consumed_bindings=len(bindings))


def verify_replay(root, output) -> dict[str, Any]:
    """Metadata/logit reconciliation only; this operation does not read WAV/PCM."""
    from scripts.submission.optionb_fit import _assert_bindings, _fit_files, _fit_target
    from scripts.test.speaker_demo_suite import canonical, read_json, read_jsonl
    root = Path(root).resolve()
    checked = validate_selection(root, output)
    replay = _fit_target(_run(root, output), 'replay')
    if _fit_target(replay, 'failure.json').exists():
        raise ValueError('A failed replay cannot be verified; preserve its evidence')
    receipt_path = _fit_target(replay, 'verification.json')
    receipt = read_json(receipt_path)
    required = {output + '/replay/' + name for name in (
        'test.jsonl', 'holdout.jsonl', 'protocol.json', 'data-freeze.json', 'selection-before-test.json',
        'canonical-evaluator.log', 'final-predictions.jsonl', 'results.json', 'comparison.json',
        'per-command.csv', 'per-intent.csv', 'per-verbatim.csv', 'per-verbatim-cohorts-v2.csv',
        'bundle/model.onnx', 'bundle/commands.json', 'bundle/normalization.json', 'bundle/policy.json', 'bundle/recipe.json')}
    required.update(output + '/' + name for name in (
        'evaluation-preparation.json', 'evaluation-selection-gate.json', 'prepare-test-holdout.json',
        'locks/release.json', 'locks/pc.json', 'locks/full-reference-metadata.jsonl',
        'locks/golden-evaluation-predictions.jsonl'))
    required.update(checked['bindings'])
    required.update(p.relative_to(root).as_posix() for p in _fit_files(replay) if p != receipt_path)
    if required - receipt['bindings'].keys():
        raise ValueError('Replay verification has unbound required evidence artifacts')
    for name in required:
        if not _fit_target(root, name).is_file():
            raise ValueError('Missing required replay evidence artifact: ' + name)
    contract = dict(format='optionb-source-inference-replay-v1', exact=True, status='verified',
        evidence_scope=checked['lock']['evidence_scope'], cached_scores_reused=False,
        canonical_case_feature_and_decision_path_reused=True, complete_original_and_stress_cohort=True,
        golden_scores_used_for_inference=False, heldout_pcm_decoded_or_inferred_before_selection=False,
        test_holdout_used_for_selection=False, exposed_regression_not_blind=True,
        independent_listening_completed=False, fresh_blind_human_check='pending_no_new_recordings',
        optimizer_updates=0, selection_changes=0, actions_enabled=False, qualified=False, installed=False,
        installed_defaults_changed=False, pi_validation='excluded_by_user', tolerance=dict(TOLERANCE),
        native_js_executed=False, pc_benchmark_executed=False, release_check_executed=False,
        release_protocol=output + '/locks/release.json', pc_protocol=output + '/locks/pc.json', output=output + '/replay')
    if any(canonical(receipt.get(k)) != canonical(v) for k, v in contract.items()):
        raise ValueError('Replay verification contract changed')
    _assert_bindings(root, receipt['bindings'])
    write_replay_protocols(root, output, reuse=True)
    # All preparation/lock artifacts were required above: these reuse checks
    # cannot hydrate, write missing evidence, or decode WAV/PCM.
    preparation = _prepare_evaluation(root, output)
    release = read_json(_fit_target(root, receipt['release_protocol']))
    pc = read_json(_fit_target(root, receipt['pc_protocol']))
    expected_bindings = _merge_bindings(release['bindings'], preparation['bindings'], pc['bindings'],
                                        _expected_export_bindings(root))
    for name in (receipt['release_protocol'], receipt['pc_protocol'], output + '/evaluation-preparation.json',
                 METRIC_HELPER, COHORT_HELPER, 'scripts/test/test_speaker_optionb_tiny.py',
                 'scripts/test/speaker_optionb_tiny_demo_report.py'):
        expected_bindings = _merge_bindings(expected_bindings, {name: sha(_fit_target(root, name))})
    for split in ('test', 'holdout'):
        name = output + '/replay/' + split + '.jsonl'
        original = release['manifest_files'][split]
        if read_jsonl(_fit_target(root, name)) != read_jsonl(_fit_target(root, original)):
            raise ValueError('Stored replay manifest provenance changed: ' + split)
        expected_bindings[name] = release['bindings'][original]
    snapshot = checked['snapshot']
    for name in ('model.onnx', 'commands.json', 'normalization.json', 'policy.json', 'recipe.json'):
        expected_bindings[output + '/replay/bundle/' + name] = snapshot['files'][snapshot['bundle'] + '/' + name]['sha256']
    protocol = dict(outputs=dict(run=output + '/replay'), tolerance=dict(TOLERANCE),
        scope='Unchanged canonical evaluator, portable exact-byte inputs; no old cache dependencies')
    if canonical(read_json(_fit_target(replay, 'protocol.json'))) != canonical(protocol):
        raise ValueError('Stored replay protocol provenance changed')
    expected_bindings[output + '/replay/protocol.json'] = sha(_fit_target(replay, 'protocol.json'))
    freeze = dict(bindings=dict(expected_bindings), historical_bindings={},
        original_historical_locks_unchanged=True, portable_bindings_replace_only_consumed_paths=True)
    if canonical(read_json(_fit_target(replay, 'data-freeze.json'))) != canonical(freeze):
        raise ValueError('Stored replay data-freeze provenance changed')
    expected_bindings[output + '/replay/data-freeze.json'] = sha(_fit_target(replay, 'data-freeze.json'))
    selection = {**checked['lock'], 'bindings': dict(expected_bindings),
                 'source_fit_selection_lock': output + '/fit/selection-before-test.json'}
    if canonical(read_json(_fit_target(replay, 'selection-before-test.json'))) != canonical(selection):
        raise ValueError('Stored replay selection provenance changed')
    expected_bindings[output + '/replay/selection-before-test.json'] = sha(_fit_target(replay, 'selection-before-test.json'))
    if any(receipt['bindings'].get(name) != digest for name, digest in expected_bindings.items()):
        raise ValueError('Stored replay binding provenance changed')
    _assert_bindings(root, expected_bindings)
    records = read_jsonl(_fit_target(replay, 'final-predictions.jsonl'))
    comparison = compare_predictions(root, snapshot, records)
    exports = _compare_csv_exports(root, replay)
    final = read_json(_fit_target(replay, 'results.json'))
    result_contract = dict(parameters=snapshot['parameters'], selected_epoch=snapshot['selected_epoch'],
        policy=snapshot['policy'], prediction_rows=len(records),
        predictions_sha256=sha(_fit_target(replay, 'final-predictions.jsonl')),
        selection_sha256=sha(_fit_target(replay, 'selection-before-test.json')),
        frozen_inputs_unchanged=True, exposed_regression_not_blind=True,
        independent_human_rows_are_original_only=True, installed=False, qualified=False)
    if any(canonical(final.get(k)) != canonical(v) for k, v in result_contract.items()):
        raise ValueError('Stored replay result contract changed')
    stable = {key: {k: v for k, v in value.items() if k != 'timing'} for key, value in final['results'].items()}
    differences = _differences(comparison['results'], stable, '/canonical_results')
    comparison.update(csv_comparisons=exports['csv_comparisons'], timing_values_compared_to_historical=False)
    import csv
    with _fit_target(replay, 'per-verbatim-cohorts-v2.csv').open(encoding='utf8', newline='') as stream:
        wording_rows = len(list(csv.DictReader(stream)))
    if (not comparison['exact'] or exports['differences'] or differences
            or canonical(receipt.get('inference_rows')) != canonical(len(records))
            or canonical(receipt.get('wording_rows')) != canonical(wording_rows)
            or canonical(read_json(_fit_target(replay, 'comparison.json'))) != canonical(comparison)):
        raise ValueError('Stored replay comparison changed')
    guards = _load_module(root, COHORT_HELPER)
    guards.verify_csv_counts(replay, final, group_count=len(comparison['expected_results']))
    _assert_bindings(root, receipt['bindings'])
    return dict(status='verified_existing_replay', inference_rows=len(records),
        fresh_inference_performed=False, cached_scores_read_for_reconciliation=True,
        wav_or_pcm_read_for_this_verification=False, exposed_regression_not_blind=True,
        verification_sha256=sha(receipt_path), actions_enabled=False, qualified=False,
        release_protocol=receipt['release_protocol'], pc_protocol=receipt['pc_protocol'])


def main(argv=None, *, root=ROOT) -> dict[str, Any]:
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('selection', 'prepare', 'protocols', 'replay', 'verify'))
    parser.add_argument('--output', default='.cache/optionb-source-repro-v1/run-01',
                        help='The selected source-fit run under .cache/; replay/ is exclusive')
    parser.add_argument('--source-commit', help='Optional provenance label; defaults to current HEAD (read-only)')
    args = parser.parse_args(argv)
    if args.operation == 'selection':
        checked = validate_selection(root, args.output)
        result = dict(status='exact_fit_selection_verified', selected_epoch=checked['lock']['selected_epoch'],
                      wav_or_pcm_read=False, actions_enabled=False)
    elif args.operation == 'prepare':
        result = prepare_evaluation(root, args.output)
    elif args.operation == 'protocols':
        result = write_replay_protocols(root, args.output, source_commit=args.source_commit, reuse=True)
    elif args.operation == 'verify':
        result = verify_replay(root, args.output)
    else:
        result = replay_evaluation(root, args.output, source_commit=args.source_commit)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
