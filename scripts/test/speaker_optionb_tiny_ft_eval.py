"""Guarded export/calibration and immutable final replay of the tiny fine-tune."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from scripts.runtime.optionb_contract import COMMANDS
from scripts.runtime.speaker_optionb_tiny import Runtime, decision
from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl, write_new
from scripts.test.speaker_optionb_tiny_ft_audit import PARENT, STUDY, score_logits
from scripts.train.speaker_optionb_tiny import check_bindings, extract_arrays, normalized


def select_candidate(study=STUDY):
    import torch
    import onnxruntime as ort
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    from scripts.test.test_speaker_optionb_tiny import export_cpu_model, select_policy

    study = Path(study)
    protocol = read_json(study / 'protocol.json')
    run = Path(protocol['outputs']['run'])
    if not (run / 'fit-verification.json').is_file():
        raise FileNotFoundError('A completed fine-tune and its receipt are required')
    if (study / 'model-selection-before-calibration.json').exists() or (study / 'selection-before-test.json').exists():
        raise FileExistsError('Preserve the earlier selected model and policy')
    freeze, fit = read_json(study / 'data-freeze.json'), read_json(run / 'fit-verification.json')
    for bindings in (freeze['bindings'], freeze['historical_bindings'], fit['bindings']):
        check_bindings(bindings)
    expected_steps = protocol['training']['epochs'] * protocol['training']['steps_per_epoch']
    if (fit['completed_arms'] != len(protocol['arms']) or fit['full_updates_per_arm'] != expected_steps
            or fit['total_optimizer_steps'] != expected_steps * len(protocol['arms'])
            or fit.get('calibration_used_for_weights_or_arm') is not False):
        raise ValueError('The full predeclared development-only comparison is incomplete')
    arms = [read_json(run / 'arms' / arm['name'] / 'training-result.json') for arm in protocol['arms']]
    if any(r['optimizer_steps'] != expected_steps or r['selected_on'] != 'dev' for r in arms):
        raise ValueError('An arm is incomplete or selected outside development')
    chosen = max(arms, key=lambda r: tuple(r['selection_rank']))
    result = read_json(run / 'training-result.json')
    if (result['arm']['name'] != chosen['arm']['name'] or result['selected_epoch'] != chosen['selected_epoch']
            or fit['selected_arm'] != chosen['arm']['name'] or fit['selected_epoch'] != chosen['selected_epoch']):
        raise ValueError('Selected arm/epoch disagrees with frozen development rank')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    arm_checkpoint = torch.load(run / 'arms' / chosen['arm']['name'] / 'best.pt', map_location='cpu', weights_only=True)
    if (checkpoint['epoch'] != result['selected_epoch'] or checkpoint['arm'] != chosen['arm']['name']
            or checkpoint['state_dict'].keys() != arm_checkpoint['state_dict'].keys()
            or any(not torch.equal(value, arm_checkpoint['state_dict'][key]) for key, value in checkpoint['state_dict'].items())):
        raise ValueError('Export checkpoint is not the development-selected arm')
    paths = [run / 'best.pt', run / 'normalization.json', run / 'training-result.json',
             run / 'fit-verification.json', study / 'protocol.json', study / 'data-freeze.json', Path(__file__)]
    model_lock = dict(selected_arm=chosen['arm']['name'], selected_epoch=chosen['selected_epoch'],
        selected_on=['dev'], calibration_used_for_weights_or_arm=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        bindings={p.as_posix(): file_digest(p) for p in paths})
    write_new(study / 'model-selection-before-calibration.json', model_lock)
    torch.set_num_threads(1)
    model = OptionBTinyVCM().eval()
    model.load_state_dict(checkpoint['state_dict'])
    bundle = run / 'bundle'
    bundle.mkdir(exist_ok=True)
    if (bundle / 'model.onnx').exists():
        raise FileExistsError('Preserve the earlier export')
    export = export_cpu_model(model, bundle / 'model.onnx')
    parameters = sum(p.numel() for p in model.parameters())
    if export['initializer_elements'] != parameters or parameters > 300_000:
        raise ValueError('Export violates the full parameter budget')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(bundle / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    norm = read_json(run / 'normalization.json')
    rows = [r for split in ('dev', 'calibration') for r in read_jsonl(run / f'{split}.jsonl') if r['kind'] == 'human']
    arrays = extract_arrays(rows)
    scores, parity, predictions = {}, [], []
    for row in rows:
        values = normalized(arrays[row['id']], norm)[None]
        lengths = np.asarray([values.shape[-1]], dtype=np.int64)
        actual = np.asarray(session.run(['logits'], {'features': values, 'lengths': lengths})[0], dtype=np.float32)[0]
        with torch.inference_mode():
            expected = model(torch.from_numpy(values), torch.from_numpy(lengths)).numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        if int(np.argmax(actual)) != int(np.argmax(expected)):
            raise ValueError('Export changed an actual development/calibration class')
        parity.append(float(np.max(np.abs(actual - expected))))
        scores[row['id']] = actual
        predictions.append(dict(id=row['id'], split=row['split'], class_index=row['class_index'], logits=actual.tolist()))
    fixtures = []
    generator = torch.Generator().manual_seed(83)
    for batch, frames in ((1, 97), (2, 258), (1, 401)):
        values = torch.randn(batch, 40, frames, generator=generator)
        lengths = torch.tensor([frames - 13 * i for i in range(batch)], dtype=torch.long)
        with torch.inference_mode():
            expected = model(values, lengths).numpy()
        actual = np.asarray(session.run(['logits'], {'features': values.numpy(), 'lengths': lengths.numpy()})[0])
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        fixtures.append(dict(batch=batch, frames=frames, max_abs_error=float(np.max(np.abs(actual - expected)))))
    selected = select_policy(rows, scores, protocol['selection'])
    write_new(study / 'policy-selection.json', selected)
    write_new(run / 'calibration-predictions.jsonl', predictions, jsonl=True)
    write_new(bundle / 'normalization.json', dict(mean=norm['mean'], std=norm['std'], source_sha256=file_digest(run / 'normalization.json')))
    write_new(bundle / 'commands.json', list(COMMANDS))
    write_new(bundle / 'policy.json', selected['policy'])
    names = ('model.onnx', 'commands.json', 'normalization.json', 'policy.json')
    binding = {name: file_digest(bundle / name) for name in names}
    recipe = dict(architecture='OptionBTinyVCM-1', parameters=parameters, binding=binding,
        research_only=True, qualified=False, selected_epoch=checkpoint['epoch'],
        selected_arm=checkpoint['arm'], parent_epoch=checkpoint['parent_epoch'],
        parent_checkpoint_sha256=freeze['parent_checkpoint_sha256'],
        no_asr_llm_transcript=True, wake_vad_not_in_vcm_parameter_count=True)
    write_new(bundle / 'recipe.json', recipe)
    runtime = Runtime(bundle)
    export.update(recorded_dev_calibration_parity_rows=len(rows), max_abs_logit_error=max(parity),
        variable_batch_frame_fixtures=fixtures, model_bytes=(bundle / 'model.onnx').stat().st_size,
        bundle_bytes=sum(p.stat().st_size for p in bundle.iterdir() if p.is_file()),
        session_load_ms=runtime.load_ms, python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, onnxruntime=ort.__version__, numpy=np.__version__)
    write_new(study / 'export-verification.json', export)
    paths.extend([*[bundle / name for name in (*names, 'recipe.json')], run / 'training-curve.jsonl',
        run / 'raw-feature-inputs.jsonl', run / 'hard-negative-pool.json', run / 'calibration-predictions.jsonl',
        study / 'model-selection-before-calibration.json', study / 'policy-selection.json', study / 'export-verification.json'])
    lock = dict(bindings={p.as_posix(): file_digest(p) for p in paths}, selected_arm=checkpoint['arm'],
        selected_epoch=checkpoint['epoch'], policy=selected['policy'], parameters=parameters,
        selected_on=['dev', 'calibration'], model_selected_on=['dev'], policy_selected_on=['dev', 'calibration'],
        fitting_splits=['train'], normalization_split='train', test_holdout_used_for_selection=False,
        heldout_payload_bytes_hashed_for_integrity=True, heldout_pcm_decoded_or_inferred_before_selection=False,
        exposed_regression_not_blind=True, complete_command_classes=32,
        created_utc=datetime.now(timezone.utc).isoformat())
    for bindings in (freeze['bindings'], freeze['historical_bindings'], fit['bindings'], model_lock['bindings']):
        check_bindings(bindings)
    write_new(study / 'selection-before-test.json', lock)
    print(json.dumps(dict(status='selection_locked', selected_arm=checkpoint['arm'], epoch=checkpoint['epoch'],
        parameters=parameters, policy=selected['policy'], dev_calibration=selected['metrics'], export=export), indent=2), flush=True)
    return lock


def baseline_controls(study=STUDY):
    study = Path(study)
    if not (study / 'selection-before-test.json').is_file():
        raise FileNotFoundError('Selection is required before reading final baseline predictions')
    lock = read_json(study / 'selection-before-test.json')
    check_bindings(lock['bindings'])
    parent = read_jsonl(PARENT / 'final-predictions.jsonl')
    fine = read_json(study / 'parent-finer-grid-policy.json')['policy']
    original = read_json(PARENT / 'bundle/policy.json')
    cohorts = sorted({(r['split'], r['kind'], r['variant']) for r in parent})
    results = {}
    for cohort in cohorts:
        rows = [r for r in parent if (r['split'], r['kind'], r['variant']) == cohort and r['metric_eligible']]
        values = np.asarray([r['event'].get('logits', np.zeros(32).tolist()) for r in rows])
        results['/'.join(cohort)] = dict(original=score_logits(rows, values, original), finer_grid=score_logits(rows, values, fine))
    result = dict(policies=dict(original=original, finer_grid=fine), results=results,
        parent_weights_unchanged=True, scores_reused=True, fitting_splits=[],
        policy_selected_on=['dev', 'calibration'], exposed_regression_not_blind=True)
    write_new(study / 'baseline-controls.json', result)
    return result


def evaluate_candidate(study=STUDY):
    from scripts.test.test_speaker_optionb_tiny import evaluate_candidate as original_final_replay
    study = Path(study)
    if not (study / 'selection-before-test.json').is_file():
        raise FileNotFoundError('A frozen selection is required before final evaluation')
    result = original_final_replay(study)
    baseline_controls(study)
    run = Path(read_json(study / 'protocol.json')['outputs']['run'])
    rows = [r for r in read_jsonl(run / 'dev.jsonl') if r['kind'] == 'human' and r['metric_eligible']]
    raw = Runtime(run / 'bundle')
    events = [(row, raw.recognize(read_audio(row))) for row in rows]
    correct = [(r, e) for r, e in events if e.get('command_index') == r['class_index'] and r['class_index'] > 0]
    chosen_row, chosen_event = next(((r, e) for r, e in correct if e['output']['accepted']), correct[0]) if correct else events[0]
    write_new(run / 'cli-smoke-reference.json', dict(id=chosen_row['id'], audio=chosen_row['audio'],
        expected=chosen_row['expected'], class_index=chosen_row['class_index'],
        observed=chosen_event, selected_software_smoke_not_accuracy_estimate=True))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['select', 'evaluate'])
    arguments = parser.parse_args()
    if arguments.operation == 'select':
        select_candidate()
    else:
        evaluate_candidate()
