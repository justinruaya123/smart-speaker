"""The tiny candidate is one complete-command head, not partial intent recovery."""
from copy import deepcopy
import importlib
from typing import Any

import pytest
import torch


def export_cpu_model(model, destination) -> dict[str, Any]:
    import warnings
    import numpy as np
    import onnx
    model.eval().cpu()
    values = torch.zeros(1, 40, 121)
    lengths = torch.tensor([121], dtype=torch.long)
    with warnings.catch_warnings(record=True) as recorded:
        torch.onnx.export(model, (values, lengths), str(destination), dynamo=False,
            input_names=['features', 'lengths'], output_names=['logits'], opset_version=17,
            dynamic_axes={'features': {0: 'batch', 2: 'frames'}, 'lengths': {0: 'batch'}, 'logits': {0: 'batch'}})
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    return dict(initializer_elements=sum(int(np.prod(t.dims)) for t in graph.graph.initializer),
                export_warnings=[str(w.message) for w in recorded])


def select_policy(rows, scores, config):
    from scripts.runtime.speaker_optionb_tiny import decision
    from scripts.train.speaker_optionb_tiny import complete_class_metrics
    if (set(scores) != {r['id'] for r in rows} or len(scores) != len(rows)
            or any(r['split'] not in {'dev', 'calibration'} or r['kind'] != 'human'
                   or r.get('training_eligible') is True for r in rows)):
        raise ValueError('Policy selection requires only nonfitting human dev/calibration')
    rows = [r for r in rows if r['metric_eligible']]

    def metrics(policy):
        result = {}
        for split in ('dev', 'calibration'):
            selected = [r for r in rows if r['split'] == split]
            events = [decision(scores[r['id']], policy) for r in selected]
            predicted = [e['command_index'] if e['output']['accepted'] else 0 for e in events]
            result[split] = complete_class_metrics(selected, predicted)
        return result

    grid, feasible = [], []
    for probability in config['probability_grid']:
        for margin in config['margin_grid']:
            policy = dict(enabled=True, min_probability=probability, min_margin=margin)
            measured = metrics(policy)
            okay = all(m['tp'] > 0 and m['precision'] is not None and m['precision'] >= .99
                       and m['negative_false_accept_rate'] is not None and m['negative_false_accept_rate'] <= .01
                       for m in measured.values())
            result = dict(policy=policy, metrics=measured, feasible=okay)
            grid.append(result)
            if okay:
                recalls = [m['recall'] for m in measured.values()]
                feasible.append(((min(recalls), sum(recalls) / 2, probability, margin), result))
    if feasible:
        chosen = max(feasible, key=lambda item: item[0])[1]
    else:
        policy = dict(enabled=False, min_probability=.99, min_margin=.3)
        chosen = dict(policy=policy, metrics=metrics(policy), feasible=False)
    return {**chosen, 'grid': grid, 'selected_on': ['dev', 'calibration'],
            'fitting_splits': [], 'holdout_used': False}


def tiny_module():
    try:
        return importlib.import_module('scripts.models.speaker_optionb_tiny')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.models.speaker_optionb_tiny':
            raise
        raise AssertionError('The bounded OptionB joint-command model is missing') from exc


def test_joint_command_model_is_below_300k_and_has_no_auxiliary_rescue_head():
    tiny = tiny_module()
    torch.set_num_threads(2)
    model = tiny.OptionBTinyVCM().eval()
    assert sum(p.numel() for p in model.parameters()) <= 300_000
    assert len(tiny.COMMANDS) == 32 and tiny.COMMANDS[0] is None
    values = torch.randn(2, 40, 121)
    with torch.inference_mode():
        logits = model(values, torch.tensor([121, 81]))
    assert logits.shape == (2, 32) and torch.isfinite(logits).all()
    assert not hasattr(model.network, 'keyword_head')
    assert not hasattr(model.network, 'ctc')
    assert not hasattr(model.network, 'attention')


def test_shared_command_inventory_preserves_all_values_and_rejects_outside_scope():
    from scripts.runtime.pcv_schema import baseline_cases, canonical
    try:
        contract = importlib.import_module('scripts.runtime.optionb_contract')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.runtime.optionb_contract':
            raise
        raise AssertionError('Shared finite OptionB class contract is missing') from exc
    definitions = baseline_cases()
    assert len(definitions) == 93 and len(contract.COMMANDS) == 32
    assert contract.COMMANDS == tiny_module().COMMANDS
    assert {canonical(c) for c in contract.COMMANDS[1:]} == {canonical(d['command']) for d in definitions}
    for definition in definitions:
        command = deepcopy(definition['command'])
        index = contract.class_index(command)
        assert type(index) is int and 1 <= index < 32
        restored = contract.command_for_index(index)
        assert restored == command
        restored['slots'].clear()
        assert contract.command_for_index(index) == command
    assert contract.class_index(None) == 0 and contract.command_for_index(0) is None
    assert contract.class_index(dict(intent='LIGHT_ON', slots={'light_id': 'light.kitchen'})) == 0
    assert contract.class_index(dict(intent='BRIGHTNESS', slots={'light_id': 'light.all', 'percent': 21})) == 0
    with pytest.raises(ValueError):
        contract.class_index(dict(intent='TIMER', slots={}))
    with pytest.raises(ValueError):
        contract.command_for_index(True)


def training_module():
    try:
        return importlib.import_module('scripts.train.speaker_optionb_tiny')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.train.speaker_optionb_tiny':
            raise
        raise AssertionError('Treated-data tiny training adapter is missing') from exc


def test_treated_training_view_is_fail_closed_and_preserves_original_references():
    train = training_module()
    row = dict(id='h1', speaker='speaker1', split='train', kind='human', synthetic=False,
               expected=dict(intent='LIGHT_ON', slots={'light_id': 'light.kitchen'}),
               metric_eligible=True, training_eligible=True, sha256='a', pcm_sha256='b')
    reserved = {**row, 'id': 'r1', 'speaker': 'speaker2', 'split': 'holdout',
                'sha256': 'c', 'pcm_sha256': 'd', 'training_eligible': False}
    original = deepcopy(row)
    result = train.training_view([row], [reserved])
    assert row == original and result[0]['expected'] == original['expected']
    assert result[0]['class_index'] == 0 and result[0]['scope'] == 'outside_optionb'
    assert result[0]['optionb_command'] is None
    for change in ({'split': 'holdout'}, {'training_eligible': False},
                   {'metric_eligible': False}, {'synthetic': True}, {'synthetic': 0}):
        with pytest.raises(ValueError):
            train.training_view([{**row, **change}], [reserved])
    for field in ('id', 'speaker', 'sha256', 'pcm_sha256', 'session_id'):
        overlap = {**reserved, field: row.get(field, 'shared')}
        fitting = {**row, field: row.get(field, 'shared')}
        with pytest.raises(ValueError):
            train.training_view([fitting], [overlap])


def test_normalization_uses_exact_human_training_membership_and_equal_utterance_weight():
    import numpy as np
    train = training_module()
    base = dict(split='train', kind='human', synthetic=False, expected=None,
                metric_eligible=True, training_eligible=True)
    rows = [{**base, 'id': 'a', 'speaker': 'x', 'sha256': 'a', 'pcm_sha256': 'a'},
            {**base, 'id': 'b', 'speaker': 'y', 'sha256': 'b', 'pcm_sha256': 'b'}]
    arrays = {'a': np.zeros((40, 2), dtype=np.float32),
              'b': np.full((40, 6), 4, dtype=np.float32),
              'holdout': np.full((40, 100), 999, dtype=np.float32)}
    norm = train.train_normalization(rows, arrays)
    np.testing.assert_allclose(norm['mean'], 2)
    np.testing.assert_allclose(norm['std'], 2)
    assert norm['ids'] == ['a', 'b'] and norm['fit_split'] == 'train'
    with pytest.raises(ValueError):
        train.train_normalization([{**rows[0], 'split': 'holdout'}], arrays)


def test_context_padding_preserves_every_original_frame_without_mutation():
    import numpy as np
    train = training_module()
    raw = np.arange(40 * 31, dtype=np.float32).reshape(40, 31)
    original = raw.copy()
    padded = train.pad_context(raw, 7, 11, np.full(40, -18.42, dtype=np.float32))
    assert padded.shape == (40, 49)
    np.testing.assert_array_equal(padded[:, 7:38], original)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(padded[:, :7], np.full((40, 7), -18.42, dtype=np.float32))


def test_real_optimizer_and_development_selection_path_on_explicit_software_fixture(tmp_path):
    """Random features test plumbing, never speech accuracy or human evidence."""
    import numpy as np
    train = training_module()
    base = dict(split='train', kind='human', synthetic=False, metric_eligible=True,
                training_eligible=True, speaker='fit-speaker')
    rows = [{**base, 'id': 'h-positive', 'sha256': 'a', 'pcm_sha256': 'a',
             'expected': dict(intent='WEATHER', slots={})},
            {**base, 'id': 'h-negative', 'sha256': 'b', 'pcm_sha256': 'b', 'expected': None},
            {**base, 'id': 's-positive', 'sha256': 'c', 'pcm_sha256': 'c', 'kind': 'synthetic',
             'synthetic': True, 'speaker': 'tts-voice', 'expected': dict(intent='WEATHER', slots={})}]
    dev = [{**rows[0], 'id': 'dev-p', 'split': 'dev', 'speaker': 'dev-speaker',
            'sha256': 'd', 'pcm_sha256': 'd', 'training_eligible': False},
           {**rows[1], 'id': 'dev-n', 'split': 'dev', 'speaker': 'dev-speaker',
            'sha256': 'e', 'pcm_sha256': 'e', 'training_eligible': False}]
    rows = train.training_view(rows, dev)
    dev = [train.projected_row(r) for r in dev]
    rng = np.random.default_rng(7)
    arrays = {r['id']: rng.normal(-5, 1, (40, 61)).astype(np.float32) for r in [*rows, *dev]}
    config = dict(seed=53, human_epochs=1, mixed_epochs=1, steps_per_epoch=1,
                  batch_size=32, learning_rate=.002, weight_decay=.0001, label_smoothing=.02,
                  threads=2, context_max_frames=8)
    result = train.fit_model(rows, dev, arrays, tmp_path, config)
    assert result['optimizer_steps'] == 2 and result['selected_on'] == 'dev'
    assert set(result['normalization_ids']) == {'h-positive', 'h-negative'}
    assert (tmp_path / 'best.pt').is_file() and (tmp_path / 'training-result.json').is_file()
    assert np.isfinite(result['last_loss'])
    assert result['weights_changed'] is True
    assert result['parameters'] <= 300_000
    assert result['fitting_splits'] == ['train']


def test_joint_runtime_decides_atomic_command_confidence_without_auxiliary_heads():
    import numpy as np
    try:
        runtime = importlib.import_module('scripts.runtime.speaker_optionb_tiny')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.runtime.speaker_optionb_tiny':
            raise
        raise AssertionError('Torch-free joint-command CPU runtime is missing') from exc
    from scripts.runtime.optionb_contract import class_index
    index = class_index(dict(intent='TIMER', slots={'duration_seconds': 10}))
    wrong_value = class_index(dict(intent='TIMER', slots={'duration_seconds': 30}))
    logits = np.full(32, -10., dtype=np.float32)
    logits[index], logits[wrong_value] = 7., 1.
    policy = dict(enabled=True, min_probability=.9, min_margin=.1)
    event = runtime.decision(logits, policy)
    assert event['output']['accepted'] is True
    assert event['output']['command'] == dict(intent='TIMER', slots={'duration_seconds': 10})
    assert event['actions_enabled'] is False and event['qualified_output']['accepted'] is False
    logits[wrong_value] = 7.
    assert runtime.decision(logits, policy)['output']['accepted'] is False
    logits[:] = -10.
    logits[0] = 8.
    assert runtime.decision(logits, policy)['output']['accepted'] is False
    logits[0], logits[index] = -10., 8.
    assert runtime.decision(logits, {**policy, 'enabled': False})['output']['accepted'] is False
    with pytest.raises(ValueError):
        runtime.decision(np.full(32, np.nan), policy)


def test_exported_cpu_runtime_variable_length_parity_and_binding_guards(tmp_path):
    """Random model/tone fixture tests real ONNX execution, not speech accuracy."""
    import hashlib
    import json
    import numpy as np
    import onnx
    from scripts.runtime.optionb_contract import COMMANDS
    from scripts.runtime.speaker_optionb_tiny import Runtime
    torch.manual_seed(53)
    torch.set_num_threads(2)
    model = tiny_module().OptionBTinyVCM().eval()
    info = export_cpu_model(model, tmp_path / 'model.onnx')
    assert info['initializer_elements'] <= 300_000
    assert onnx.load(str(tmp_path / 'model.onnx')).graph.output[0].name == 'logits'
    for name, value in {'commands.json': list(COMMANDS),
                        'normalization.json': dict(mean=[0.] * 40, std=[1.] * 40),
                        'policy.json': dict(enabled=True, min_probability=0., min_margin=0.)}.items():
        (tmp_path / name).write_text(json.dumps(value), encoding='utf-8')
    binding = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
               for name in ('model.onnx', 'commands.json', 'normalization.json', 'policy.json')}
    (tmp_path / 'recipe.json').write_text(json.dumps(dict(architecture='OptionBTinyVCM-1',
        parameters=258912, binding=binding, research_only=True)), encoding='utf-8')
    runtime = Runtime(tmp_path)
    for batch, frames in ((1, 97), (2, 258), (1, 401)):
        values = torch.randn(batch, 40, frames)
        lengths = torch.tensor([frames - 13 * i for i in range(batch)])
        with torch.inference_mode():
            expected = model(values, lengths).numpy()
        actual = runtime.session.run(None, {'features': values.numpy(), 'lengths': lengths.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    event = runtime.recognize(np.sin(np.arange(16000) * .04).astype(np.float32) * .1)
    assert event['actions_enabled'] is False and event['model_qualified'] is False
    assert runtime.recognize(np.zeros(16000, dtype=np.float32))['output']['accepted'] is False
    assert runtime.recognize(np.full(16000, np.nan, dtype=np.float32))['output']['accepted'] is False
    with (tmp_path / 'model.onnx').open('ab') as stream:
        stream.write(b'\x00')
    with pytest.raises(ValueError, match='identity'):
        Runtime(tmp_path)


def test_global_policy_selection_uses_only_nonfitting_human_dev_calibration():
    import numpy as np
    from scripts.runtime.optionb_contract import class_index
    index = class_index(dict(intent='WEATHER', slots={}))
    rows = [dict(id=f'{split}-{i}', split=split, kind='human', metric_eligible=True,
                 training_eligible=False, class_index=index if i == 0 else 0)
            for split in ('dev', 'calibration') for i in range(2)]
    scores = {r['id']: np.full(32, -10.) for r in rows}
    for row in rows:
        scores[row['id']][index if row['class_index'] else 0] = 8.
    scores['dev-1'][0], scores['dev-1'][index] = 0., 4.
    config = dict(probability_grid=[.5, .95, .99], margin_grid=[0., .2])
    result = select_policy(rows, scores, config)
    assert result['policy']['enabled'] is True
    assert result['policy']['min_probability'] == .99
    assert result['metrics']['dev']['recall'] == 1.
    assert result['metrics']['calibration']['negative_false_accepts'] == 0
    with pytest.raises(ValueError):
        select_policy([{**rows[0], 'split': 'holdout'}, *rows[1:]], scores, config)
    impossible = {key: np.full(32, -10.) for key in scores}
    for values in impossible.values():
        values[index] = 8.
    assert select_policy(rows, impossible, config)['policy']['enabled'] is False


def test_selection_and_evaluation_fail_closed_without_completed_stage_receipts(tmp_path):
    import json
    (tmp_path / 'protocol.json').write_text(json.dumps({'outputs': {'run': str(tmp_path / 'run')}}))
    with pytest.raises(FileNotFoundError, match='completed fit'):
        select_candidate(tmp_path)
    with pytest.raises(FileNotFoundError, match='selection'):
        evaluate_candidate(tmp_path)


def select_candidate(study=None):
    """Export and calibrate once, then lock everything before final replay."""
    import json
    import platform
    from pathlib import Path
    import numpy as np
    import onnxruntime as ort
    from scripts.runtime.optionb_contract import COMMANDS
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.test.speaker_demo_suite import read_json, read_jsonl, write_new, file_digest
    from scripts.train.speaker_optionb_tiny import STUDY, check_bindings, extract_arrays, normalized
    study = Path(study) if study is not None else STUDY
    protocol = read_json(study / 'protocol.json')
    run = Path(protocol['outputs']['run'])
    if not (run / 'fit-verification.json').is_file():
        raise FileNotFoundError('A completed fit and its verification receipt are required')
    freeze = read_json(study / 'data-freeze.json')
    check_bindings(freeze['bindings'])
    check_bindings(freeze['historical_bindings'])
    if (study / 'selection-before-test.json').exists():
        raise FileExistsError('Preserve the existing selection lock')
    result = read_json(run / 'training-result.json')
    expected_steps = (protocol['training']['human_epochs'] + protocol['training']['mixed_epochs']) * protocol['training']['steps_per_epoch']
    if result['optimizer_steps'] != expected_steps:
        raise ValueError('The declared full fit is incomplete')
    torch.set_num_threads(1)
    model = tiny_module().OptionBTinyVCM().eval()
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != result['selected_epoch']:
        raise ValueError('Selected epoch does not match the fitting result')
    model.load_state_dict(checkpoint['state_dict'])
    bundle = run / 'bundle'
    bundle.mkdir(exist_ok=True)
    if (bundle / 'model.onnx').exists():
        raise FileExistsError('Preserve the earlier export attempt')
    export = export_cpu_model(model, bundle / 'model.onnx')
    if export['initializer_elements'] > 300_000:
        raise ValueError('Export exceeds the complete parameter budget')
    norm = read_json(run / 'normalization.json')
    rows = [r for split in ('dev', 'calibration') for r in read_jsonl(run / f'{split}.jsonl') if r['kind'] == 'human']
    arrays = extract_arrays(rows)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(bundle / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    scores, parity, prediction_rows = {}, [], []
    for row in rows:
        values = normalized(arrays[row['id']], norm)[None]
        lengths = np.asarray([values.shape[-1]], dtype=np.int64)
        actual = session.run(['logits'], {'features': values, 'lengths': lengths})[0][0]
        with torch.inference_mode():
            expected = model(torch.from_numpy(values), torch.from_numpy(lengths)).numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        if int(np.argmax(actual)) != int(np.argmax(expected)):
            raise ValueError('Export changes a development/calibration class')
        parity.append(float(np.max(np.abs(actual - expected))))
        scores[row['id']] = actual
        prediction_rows.append(dict(id=row['id'], split=row['split'], class_index=row['class_index'], logits=actual.tolist()))
    selected = select_policy(rows, scores, protocol['selection'])
    write_new(study / 'policy-selection.json', selected)
    write_new(run / 'calibration-predictions.jsonl', prediction_rows, jsonl=True)
    write_new(bundle / 'normalization.json', dict(mean=norm['mean'], std=norm['std'], source_sha256=file_digest(run / 'normalization.json')))
    write_new(bundle / 'commands.json', list(COMMANDS))
    write_new(bundle / 'policy.json', selected['policy'])
    names = ('model.onnx', 'commands.json', 'normalization.json', 'policy.json')
    binding = {name: file_digest(bundle / name) for name in names}
    recipe = dict(architecture='OptionBTinyVCM-1', parameters=sum(p.numel() for p in model.parameters()),
        binding=binding, research_only=True, selected_epoch=checkpoint['epoch'],
        no_asr_llm_transcript=True, qualified=False, wake_vad_not_in_vcm_parameter_count=True)
    write_new(bundle / 'recipe.json', recipe)
    runtime = Runtime(bundle)
    export.update(recorded_dev_calibration_parity_rows=len(rows), max_abs_logit_error=max(parity),
        model_bytes=(bundle / 'model.onnx').stat().st_size,
        bundle_bytes=sum(p.stat().st_size for p in bundle.iterdir() if p.is_file()),
        session_load_ms=runtime.load_ms, python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, onnxruntime=ort.__version__, numpy=np.__version__)
    write_new(study / 'export-verification.json', export)
    paths = [*[bundle / name for name in (*names, 'recipe.json')], run / 'best.pt', run / 'normalization.json',
        run / 'training-result.json', run / 'training-curve.jsonl', run / 'raw-feature-inputs.jsonl', run / 'fit-verification.json',
        run / 'calibration-predictions.jsonl', study / 'protocol.json', study / 'data-freeze.json',
        study / 'policy-selection.json', study / 'export-verification.json', Path(__file__),
        Path('scripts/runtime/speaker_optionb_tiny.py')]
    lock = dict(bindings={str(p).replace('\\', '/'): file_digest(p) for p in paths},
        selected_epoch=checkpoint['epoch'], policy=selected['policy'], selected_on=['dev', 'calibration'],
        fitting_splits=['train'], normalization_split='train', test_holdout_used_for_selection=False,
        heldout_payload_bytes_hashed_for_integrity=True, heldout_pcm_decoded_or_inferred_before_selection=False,
        historical_fit_receipt_heldout_audio_read_means_no_decode_or_inference=True,
        complete_command_classes=32, parameters=recipe['parameters'], exposed_regression_not_blind=True)
    check_bindings(freeze['bindings'])
    check_bindings(freeze['historical_bindings'])
    write_new(study / 'selection-before-test.json', lock)
    print(json.dumps(dict(status='selection_locked', epoch=checkpoint['epoch'], export=export, policy=selected['policy'],
                          dev_calibration=selected['metrics']), indent=2), flush=True)
    return lock


def evaluate_candidate(study=None):
    """Read-only final replay: no updates, threshold search or re-selection."""
    import csv
    import json
    from pathlib import Path
    import numpy as np
    from scripts.runtime.optionb_contract import COMMANDS, class_index
    from scripts.runtime.pcv_schema import baseline_cases, normalize_text, INTENTS
    from scripts.runtime.speaker_optionb_tiny import Runtime, decision
    from scripts.test.speaker_demo_suite import read_json, read_jsonl, write_new, read_audio, pcm_digest, file_digest
    from scripts.train.speaker_optionb_tiny import STUDY, check_bindings, complete_class_metrics
    study = Path(study) if study is not None else STUDY
    if not (study / 'selection-before-test.json').is_file():
        raise FileNotFoundError('A selection lock is required before evaluation')
    lock = read_json(study / 'selection-before-test.json')
    protocol, freeze = read_json(study / 'protocol.json'), read_json(study / 'data-freeze.json')
    check_bindings(lock['bindings'])
    check_bindings(freeze['bindings'])
    check_bindings(freeze['historical_bindings'])
    run = Path(protocol['outputs']['run'])
    if (run / 'final-predictions.jsonl').exists():
        raise FileExistsError('Preserve the earlier final evaluation')
    runtime = Runtime(run / 'bundle')
    records = []
    fixed = dict(enabled=True, min_probability=.95, min_margin=0.)

    def replay(row, pcm, variant='original'):
        event = runtime.recognize(pcm)
        raw_index = int(event.get('command_index', 0))
        fixed_event = decision(event['logits'], fixed) if 'logits' in event else event
        record = dict(id=row['id'] if variant == 'original' else row['id'] + '::' + variant,
            parent_id=row['id'], split=row['split'], kind=row['kind'], metric_eligible=row['metric_eligible'],
            class_index=row['class_index'], expected=row['expected'], optionb_command=row['optionb_command'],
            scope=row['scope'], variant=variant, source_audio_sha256=row['sha256'], source_pcm_sha256=row['pcm_sha256'],
            actual_input_pcm_sha256=pcm_digest(pcm), raw_index=raw_index,
            accepted_index=raw_index if event['output']['accepted'] else 0,
            fixed_index=raw_index if fixed_event['output']['accepted'] else 0,
            latency_ms=event['latency_ms'], event=event)
        records.append(record)

    for split in ('test', 'holdout'):
        for row in read_jsonl(run / f'{split}.jsonl'):
            pcm = read_audio(row)
            replay(row, pcm)
            if split == 'holdout' and row['kind'] == 'human' and row['metric_eligible']:
                silence = np.zeros(16000, dtype=np.float32)
                replay(row, np.concatenate([silence, pcm]), 'silence_prefix_1s')
                replay(row, np.concatenate([pcm, silence]), 'silence_suffix_1s')
                replay(row, np.concatenate([silence, pcm, silence]), 'silence_both_1s')
    write_new(run / 'final-predictions.jsonl', records, jsonl=True)
    results, per_command, per_intent = {}, [], []
    groups = sorted({(r['split'], r['kind'], r['variant']) for r in records})
    for split, kind, variant in groups:
        all_rows = [r for r in records if (r['split'], r['kind'], r['variant']) == (split, kind, variant)]
        rows = [r for r in all_rows if r['metric_eligible']]
        key = '/'.join((split, kind, variant))
        metrics = {arm: complete_class_metrics(rows, [r[field] for r in rows])
                   for arm, field in (('raw', 'raw_index'), ('fixed_095', 'fixed_index'), ('selected', 'accepted_index'))}
        negative_groups = {}
        for scope in ('original_unknown', 'outside_optionb'):
            chosen = [r for r in rows if r['scope'] == scope]
            negative_groups[scope] = dict(rows=len(chosen), raw_false_accepts=sum(r['raw_index'] > 0 for r in chosen),
                                         selected_false_accepts=sum(r['accepted_index'] > 0 for r in chosen))
        timing = np.asarray([r['latency_ms'] for r in rows])
        results[key] = dict(total_rows=len(all_rows), ambiguous_unscored=len(all_rows) - len(rows),
            metrics=metrics, negative_groups=negative_groups,
            timing=dict(rows=len(rows), p50_ms=float(np.percentile(timing, 50)), p95_ms=float(np.percentile(timing, 95))))

        def cell(indices):
            support = sum(r['class_index'] in indices for r in rows)
            tp = sum(r['class_index'] in indices and r['accepted_index'] == r['class_index'] for r in rows)
            fp = sum(r['accepted_index'] in indices and r['accepted_index'] != r['class_index'] for r in rows)
            fn = support - tp
            return dict(cohort=key, n=support, raw_correct=sum(r['class_index'] in indices and r['raw_index'] == r['class_index'] for r in rows),
                tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/support if support else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
        for index, command in enumerate(COMMANDS[1:], 1):
            per_command.append(dict(class_index=index, intent=command['intent'], command=json.dumps(command, sort_keys=True), **cell({index})))
        for intent in INTENTS:
            indices = {i for i, command in enumerate(COMMANDS) if command and command['intent'] == intent}
            per_intent.append(dict(intent=intent, **cell(indices)))
    original_holdout = [r for r in records if r['split'] == 'holdout' and r['kind'] == 'human' and r['variant'] == 'original']
    source_holdout = {r['id']: r for r in read_jsonl(run / 'holdout.jsonl')}
    per_verbatim = []
    for definition in baseline_cases():
        references = {normalize_text(definition[key]) for key in ('text', 'spoken_text') if definition.get(key)}
        matched = [r for r in original_holdout if r['metric_eligible'] and r['class_index'] == class_index(definition['command']) and any(normalize_text(source_holdout[r['id']].get(field) or '') in references
                    for field in ('text', 'reference_text', 'spoken_text'))]
        per_verbatim.append(dict(id=definition['id'], text=definition['text'], command=json.dumps(definition['command'], sort_keys=True),
            metadata_exact_human_references=len(matched), raw_correct=sum(r['raw_index'] == r['class_index'] for r in matched),
            correct_accepted=sum(r['accepted_index'] == r['class_index'] and r['class_index'] > 0 for r in matched)))
    for name, rows in (('per-command.csv', per_command), ('per-intent.csv', per_intent), ('per-verbatim.csv', per_verbatim)):
        with (study / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    check_bindings(lock['bindings'])
    check_bindings(freeze['bindings'])
    check_bindings(freeze['historical_bindings'])
    payload = dict(results=results, parameters=lock['parameters'], policy=lock['policy'],
        selected_epoch=lock['selected_epoch'], independent_human_rows_are_original_only=True,
        predictions_sha256=file_digest(run / 'final-predictions.jsonl'),
        prediction_rows=len(records), protected_historical_inputs=len(freeze['historical_bindings']),
        frozen_inputs_unchanged=True, exposed_regression_not_blind=True, qualified=False, installed=False,
        selection_sha256=file_digest(study / 'selection-before-test.json'),
        csv_rows=dict(per_command=len(per_command), per_intent=len(per_intent), per_verbatim=len(per_verbatim)))
    write_new(study / 'results.json', payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tiny OptionB export/selection and immutable final replay')
    parser.add_argument('operation', choices=['select', 'evaluate'])
    arguments = parser.parse_args()
    if arguments.operation == 'select':
        select_candidate()
    else:
        evaluate_candidate()
