"""One fixed-size checkpoint-selection repair; preserve all parent evidence."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts.models.pcv_contextual import export_onnx
from scripts.models.pcv_release_pooling import PooledContextVCM
from scripts.runtime.pcv_final_runtime import BINDINGS, verify_bundle
from scripts.runtime.pcv_schema import INTENTS, canonical
from scripts.runtime.speaker_compact import load_bound_policy
from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.train import speaker_compact as compact
from scripts.train import speaker_compact_release as reporting
from scripts.train import speaker_synthetic_data as generation
from scripts.train.speaker_compact_policy import fit_policy

ROOT = compact.ROOT
HERE = ROOT / 'experiments/speaker-pipeline-repair-v1'
CACHE = ROOT / '.cache/speaker-pipeline-repair-v1'
CONTROL = compact.HERE / 'fp32-recovery-v2'
CONTROL_CACHE = compact.CACHE / 'fp32-recovery-v2'
SAVED = compact.CACHE / 'training/mixed-control/epoch-028.pt'
FOUNDATION = compact.CACHE / 'training/human-shared/last.pt'
FIT_BASELINE = dict(development=dict(correct=96, wrong=6, false_accepts=1),
                    calibration=dict(correct=40, wrong=0, false_accepts=0))


def recognition_key(metrics, epoch):
    human, synthetic = metrics['human'], metrics['synthetic']
    return (human['raw_macro_speaker_recall'], human['raw_exact'],
            -human['raw_negative_errors'], synthetic.get('raw_recall') or 0., -epoch)


def promotion_gate(candidate_views, baseline_views):
    reasons = []
    for view in ('development', 'calibration'):
        candidate, baseline = candidate_views[view], baseline_views[view]
        if candidate['correct'] < baseline['correct']:
            reasons.append(view + ': fewer correct accepts')
        for field in ('wrong', 'false_accepts'):
            if candidate[field] > baseline[field]:
                reasons.append(view + ': increased ' + field)
    gain = sum(candidate_views[v]['correct'] - baseline_views[v]['correct']
               for v in ('development', 'calibration'))
    if gain <= 0:
        reasons.append('no strict combined correct-accept gain')
    return dict(reasons=reasons, correct_gain=gain,
                candidate=candidate_views, baseline=baseline_views) | {'pass': not reasons}


def verify_comparable(candidate_rows, baseline_rows, *, human_only=True):
    def identities(rows):
        result = {}
        for row in rows:
            if row['id'] in result:
                raise ValueError('Duplicate comparison identity')
            if human_only and row.get('synthetic') is not False:
                raise ValueError('Only original human fitting views may determine promotion')
            result[row['id']] = (canonical(row['expected']), row.get('synthetic'), bool(row.get('capture_fault')))
        return result
    if identities(candidate_rows) != identities(baseline_rows):
        raise ValueError('Comparison population or typed references changed')


def regression_gate(human_test, human_holdout):
    reasons = []
    for name, metric, minimum, max_wrong, max_unknown in (
        ('test', human_test, 230, 21, 2), ('holdout', human_holdout, 24, 1, 1)
    ):
        if metric['exact'] < minimum:
            reasons.append(name + ': insufficient correct accepts')
        if metric['accepted'] - metric['exact'] > max_wrong:
            reasons.append(name + ': increased wrong accepts')
        if metric['negative_errors'] > max_unknown:
            reasons.append(name + ': increased UNKNOWN accepts')
    return {'pass': not reasons, 'reasons': reasons}


def release_context(target):
    spec = importlib.util.spec_from_file_location('scripts.train._repair_release', ROOT / 'scripts/train/speaker_final_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.CACHE = target
    module.PREPARED = compact.PREPARED
    module.training = compact.shared_trainer()
    return module


def freeze():
    control = read_json(CONTROL/'results.json')
    generation._verify(control['bindings'])
    path = HERE / 'protocol.json'
    if path.exists():
        cfg = read_json(path)
        generation._verify(cfg['bindings'])
        generation._verify(cfg['installed_metadata'])
        return cfg
    parent = compact.freeze()
    baseline = read_json(CONTROL / 'recall-policy.json')
    measured = {name: baseline['views'][name]['selected'] for name in FIT_BASELINE}
    if measured != FIT_BASELINE:
        raise ValueError('Frozen control acceptance baseline changed')
    for checkpoint, phase, epoch in ((SAVED, 'mixed', 28), (FOUNDATION, 'human', 24)):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['model_config'] != compact.ARCHITECTURE or payload['phase'] != phase or payload['epoch'] != epoch:
            raise ValueError('Saved checkpoint architecture or provenance mismatch')
    model = PooledContextVCM(**compact.ARCHITECTURE)
    if compact.check_parameters(model) != 935567:
        raise ValueError('Repair must retain exactly 935567 parameters')
    files = [Path(__file__), ROOT/'scripts/test/test_speaker_pipeline_repair.py',
             ROOT/'docs/PLAN-speaker-pipeline-repair-v1.md', compact.HERE/'protocol.json',
             compact.HERE/'selection-before-calibration.json', SAVED, FOUNDATION,
             compact.CACHE/'training/mixed-control/epoch-032.pt',
             compact.CACHE/'training/mixed-control/history.json',
             compact.CACHE/'training/human-shared/history.json',
             compact.CACHE/'prepared/train.jsonl', compact.CACHE/'prepared/dev.jsonl',
             CONTROL/'results.json', CONTROL/'recall-policy.json',
             CONTROL_CACHE/'development-fp32.json', CONTROL_CACHE/'calibration-predictions.json',
             CONTROL/'package-final.json', CONTROL/'package-final-verification.json']
    files += [ROOT/name for name in (
        'scripts/train/speaker_compact_policy.py', 'scripts/train/speaker_compact_release.py',
        'scripts/train/speaker_final_release.py', 'scripts/runtime/pcv_final_runtime.py',
        'scripts/runtime/speaker_compact.py', 'scripts/runtime/speaker_compact_benchmark.py',
        'scripts/runtime/speaker_config.py', 'scripts/runtime/pcv_pi_benchmark.py',
        'scripts/pcv_final_delivery.py', 'scripts/test/pcv_public_qualification.py',
        'experiments/pcv-human-slots-v1/output.schema.json',
        'experiments/pcv-release-train-v1/custom-slots.json',
        'data/sources/speaker-final-pi-assets.source.json',
        'experiments/phrase-contract-v1/configs/runtime-cp312-aarch64/dependencies.json')]
    files += [compact.PREPARED/'views'/name for name in (
        'human/calibration.jsonl', 'evaluation/test.jsonl', 'evaluation/holdout.jsonl')]
    cfg = copy.deepcopy(parent)
    cfg.update(experiment_id='speaker-pipeline-repair-v1',
               authorization='User approved the fixed-size saved-checkpoint comparison and at most one continuation',
               hypothesis='Recognition-based weight selection improves accepted complete commands without increasing model size or fitting-view errors',
               initialization='First saved mixed epoch28; if needed one mixed continuation from saved human epoch24',
               selection='Human raw macro speaker recall; raw exact; fewer raw negative candidates; synthetic raw recall; earlier epoch',
               saved_checkpoint=str(SAVED.relative_to(ROOT)), foundation=str(FOUNDATION.relative_to(ROOT)),
               continuation_limit=1, repeat_human_stage=False, parameters=935567,
               fit_baseline=FIT_BASELINE,
               promotion='Correct accepts nondecreasing on each human fitting view and strictly increasing in total; wrong and UNKNOWN accepts nonincreasing on each view',
               regression_gate=dict(test=dict(minimum_correct=230, maximum_wrong=21, maximum_unknown=2),
                                    holdout=dict(minimum_correct=24, maximum_wrong=1, maximum_unknown=1)),
               export=dict(precision='fp32', maximum_bytes=4*1024*1024, numerical_tolerance=2e-5),
               published_evaluation='Previously examined regression corpus, not untouched validation',
               qualification_unchanged=True, installed=False, physical_pi_measured=False,
               limitation='One seed and previously examined data; no isolated estimate of checkpoint, foundation or calibration effects')
    cfg.pop('maximum_int8_graph_bytes', None)
    cfg['bindings'].update(control['bindings'])
    cfg['bindings'].update({str(p.relative_to(ROOT)):file_digest(p) for p in files})
    HERE.mkdir(parents=True, exist_ok=True)
    references = read_json(compact.HERE/'references.json')
    references.update(study=cfg['experiment_id'], retrieved_date='2026-10-16',
                      repair_scope='Reuses augmentations and loss unchanged; cited calibration work motivates separation, not a promised VCM gain; no temperature scaling')
    write_new(HERE/'references.json', references)
    cfg['bindings'][str((HERE/'references.json').relative_to(ROOT))] = file_digest(HERE/'references.json')
    write_new(path, cfg)
    return cfg


def _development_inputs():
    dev = read_jsonl(compact.CACHE/'prepared/dev.jsonl')
    if any(row['split'] != 'dev' for row in dev):
        raise ValueError('Only development may select recognition weights')
    norm = read_json(compact.PREPARED/'normalization.json')
    entries = read_json(compact.PREPARED/'catalog.json')['entries']
    return dev, norm, entries


def evaluate_checkpoint(name, checkpoint):
    cfg = freeze()
    target = CACHE/name
    evidence = HERE/name
    if target.exists() or evidence.exists():
        raise FileExistsError('Preserve existing candidate comparison; do not overwrite or revise thresholds')
    target.mkdir(parents=True)
    evidence.mkdir()
    base = release_context(target)
    torch.set_num_threads(1)
    dev, norm, entries = _development_inputs()
    arrays = base.training.load_arrays(dev, norm)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['model_config'] != compact.ARCHITECTURE:
        raise ValueError('Candidate architecture changed')
    model = PooledContextVCM(**compact.ARCHITECTURE)
    if compact.check_parameters(model) != cfg['parameters']:
        raise ValueError('Candidate parameter count changed')
    model.load_state_dict(payload['weights'])
    bundle = target/'bundle'
    bundle.mkdir()
    exported = export_onnx(model, bundle/'model.onnx')
    if exported['max_absolute_error'] >= cfg['export']['numerical_tolerance'] or not 0 < exported['onnx_bytes'] < cfg['export']['maximum_bytes']:
        raise ValueError('FP32 numerical/resource gate failed')
    exported['legacy_two_mib_pass'] = exported.pop('pass')
    exported['repair_four_mib_numerical_pass'] = True
    for source, destination in (
        (compact.PREPARED/'catalog.json', 'catalog.json'),
        (compact.PREPARED/'normalization.json', 'normalization.json'),
        (ROOT/'experiments/pcv-human-slots-v1/output.schema.json', 'output.schema.json'),
        (ROOT/'experiments/pcv-release-train-v1/custom-slots.json', 'custom-slots.json'),
    ):
        shutil.copyfile(source, bundle/destination)
    binding = {key:file_digest(bundle/filename) for key, filename in BINDINGS.items()}
    write_new(evidence/'selection-before-calibration.json', dict(
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=file_digest(checkpoint),
        epoch=payload['epoch'], model_binding=binding, export=exported,
        protocol_sha256=file_digest(HERE/'protocol.json'), test_used_for_selection=False))
    views = base.frozen_evaluation_views(compact.PREPARED)
    # Warm frontend/einops serially before the existing threaded audio preparation.
    from vcm.frontend import features
    features(np.zeros(1600, dtype=np.float32))
    from scripts.pcv_parallel import ParallelScorer
    with ParallelScorer(entries, workers=6, return_scores=True) as scorer:
        engine = base.ONNXModel(bundle/'model.onnx')
        scored_dev = base.training.score_model(engine, dev, arrays, entries, scorer, device='cpu', batch_size=1)
        write_new(target/'development-fp32.json', scored_dev)
        cal_rows, excluded = base.partition_evaluation(read_jsonl(views['human/calibration']))
        prepared, cal_arrays = base.external_inputs(cal_rows, norm, 'calibration')
        calibration = base.score_external(engine, prepared, cal_arrays, entries, scorer)
        write_new(target/'calibration-predictions.json', calibration)
        write_new(target/'calibration-excluded.jsonl', excluded, jsonl=True)
    human_dev = [r for r in scored_dev if r['synthetic'] is False]
    generation._verify(cfg['bindings'])
    verify_comparable(human_dev, [r for r in read_json(CONTROL_CACHE/'development-fp32.json') if r['synthetic'] is False])
    verify_comparable(calibration, read_json(CONTROL_CACHE/'calibration-predictions.json'))
    recall = fit_policy(human_dev, calibration, binding)
    qualified = base.calibrate_human(calibration)
    qualified['binding'] = binding
    write_new(bundle/'policy.json', qualified)
    write_new(bundle/'recall-policy.json', recall)
    write_new(evidence/'recall-policy.json', recall)
    from vcm.catalog import PHONES
    recipe = dict(schema_version=3, model_config=compact.ARCHITECTURE, architecture_class='PooledContextVCM',
        parameters=cfg['parameters'], maximum_parameters=compact.MAX_PARAMETERS, precision='fp32',
        intent_labels=['OOV', *INTENTS], acoustic_labels=['<blank>', *PHONES], state_shapes=[], heads=['ctc','intent','target'],
        binding=binding, policy_enabled=qualified['enabled'], recall_policy_sha256=file_digest(bundle/'recall-policy.json'),
        input='float32 [1,40,input_frames]', output='raw [1,ceil(input_frames/2),classes]',
        feature_hop_ms=10, output_hop_ms=20, inference_mode='complete_utterance', teacher_shipped=False,
        installed=False, status='unqualified_demo_candidate')
    write_new(bundle/'recipe.json', recipe)
    verify_bundle(bundle, diagnostic=True)
    load_bound_policy(bundle, recipe)
    candidate_views = {view:recall['views'][view]['selected'] for view in FIT_BASELINE}
    decision = promotion_gate(candidate_views, cfg['fit_baseline'])
    decision.update(candidate=name, checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=file_digest(checkpoint), model_binding=binding,
        development=base.training.summarize(scored_dev), export=exported,
        bundle=str(bundle.relative_to(ROOT)), qualified=qualified['enabled'],
        test_used=False, human_fitting_views_only=True,
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in (
            HERE/'protocol.json', evidence/'selection-before-calibration.json',
            target/'development-fp32.json', target/'calibration-predictions.json',
            target/'calibration-excluded.jsonl',
            bundle/'model.onnx', bundle/'recipe.json', bundle/'policy.json', bundle/'recall-policy.json')})
    generation._verify(cfg['bindings'])
    generation._verify(cfg['installed_metadata'])
    write_new(evidence/'decision.json', decision)
    if decision['pass']:
        write_new(HERE/'selection-before-test.json', dict(candidate=name,
            decision=str((evidence/'decision.json').relative_to(ROOT)),
            decision_sha256=file_digest(evidence/'decision.json'), bundle=decision['bundle'],
            bindings=decision['bindings'], published_test_used_for_selection=False,
            evaluation_scope='One regression replay of already examined test and holdout; never new validation'))
    print('REPAIR_DEVELOPMENT_DECISION', json.dumps(decision), flush=True)
    return decision


def reuse():
    cfg = freeze()
    history = read_json(compact.CACHE/'training/mixed-control/history.json')
    selected = max((r for r in history if 'development' in r),
                   key=lambda r:recognition_key(r['development'], r['epoch']))
    if selected['epoch'] != 28:
        raise ValueError('Saved recognition-selected checkpoint changed from predeclared epoch28')
    return evaluate_checkpoint('saved-epoch28', ROOT/cfg['saved_checkpoint'])


def train():
    cfg = freeze()
    prior = read_json(HERE/'saved-epoch28/decision.json')
    generation._verify(prior['bindings'])
    if prior['pass'] or (HERE/'selection-before-test.json').exists():
        raise ValueError('Saved checkpoint already promoted; training is unnecessary')
    if (CACHE/'training').exists() or (HERE/'continuation-selection-before-calibration.json').exists():
        raise FileExistsError('At most one continuation; preserve partial training evidence')
    if not torch.cuda.is_available():
        raise RuntimeError('Use approved SSH host execution to expose NVIDIA CUDA')
    shared = compact.shared_trainer()
    shared.HERE = HERE
    shared.CACHE = CACHE
    shared.selection_key = recognition_key
    dev, norm, entries = _development_inputs()
    rows = read_jsonl(compact.CACHE/'prepared/train.jsonl')
    if {r['id'] for r in rows} & {r['id'] for r in dev}:
        raise ValueError('Training/development overlap')
    pools = shared.TrainingPools(rows)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    arrays = shared.load_arrays(rows+dev, norm)
    from scripts.pcv_parallel import ParallelScorer
    began = time.perf_counter()
    with ParallelScorer(entries, workers=cfg['training']['decoder_workers'], return_scores=True) as scorer:
        selected = shared.fit_stage('mixed', 'control', ROOT/cfg['foundation'], pools, dev, arrays, entries, scorer, cfg)
    generation._verify(cfg['bindings'])
    generation._verify(cfg['installed_metadata'])
    result = dict(experiment_id=cfg['experiment_id'], selected=selected, continuation_count=1,
        human_stage_repeated=False, initial_checkpoint=cfg['foundation'], parameters=cfg['parameters'],
        seconds=time.perf_counter()-began, seed=cfg['training']['seed'],
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in (HERE/'protocol.json', ROOT/selected['checkpoint'])})
    write_new(HERE/'continuation-selection-before-calibration.json', result)
    print('REPAIR_TRAINING_COMPLETE', json.dumps(result), flush=True)
    return result


def candidate():
    cfg = freeze()
    selection = read_json(HERE/'continuation-selection-before-calibration.json')
    generation._verify(selection['bindings'])
    if selection['continuation_count'] != cfg['continuation_limit']:
        raise ValueError('Unexpected training count')
    return evaluate_checkpoint('continuation', ROOT/selection['selected']['checkpoint'])


def regression():
    cfg = freeze()
    selection = read_json(HERE/'selection-before-test.json')
    generation._verify(selection['bindings'])
    generation._verify({selection['decision']:selection['decision_sha256']})
    if not read_json(ROOT/selection['decision'])['pass']:
        raise ValueError('Regression requires development promotion')
    write_new(HERE/'regression-started.json', dict(selection_sha256=file_digest(HERE/'selection-before-test.json'),
        test_used_for_tuning=False, one_replay=True))
    target = CACHE/'regression'
    target.mkdir(parents=True, exist_ok=False)
    base = release_context(target)
    bundle = ROOT/selection['bundle']
    recall, qualified = read_json(bundle/'recall-policy.json'), read_json(bundle/'policy.json')
    _, norm, entries = _development_inputs()
    views = base.frozen_evaluation_views(compact.PREPARED)
    torch.set_num_threads(1)
    from vcm.frontend import features
    features(np.zeros(1600, dtype=np.float32))
    from scripts.pcv_parallel import ParallelScorer
    results = {}
    with ParallelScorer(entries, workers=6, return_scores=True) as scorer:
        engine = base.ONNXModel(bundle/'model.onnx')
        for name in ('test', 'holdout'):
            source = read_jsonl(views['evaluation/'+name])
            eligible, excluded = base.partition_evaluation(source)
            prepared, arrays = base.external_inputs(eligible, norm, name)
            records = base.score_external(engine, prepared, arrays, entries, scorer)
            verify_comparable(records, read_json(CONTROL_CACHE/(name+'-predictions.json')), human_only=False)
            reporting.add_decisions(records, recall, qualified)
            write_new(target/(name+'-predictions.json'), records)
            write_new(target/(name+'-excluded.jsonl'), excluded, jsonl=True)
            results[name] = dict(published_rows=len(source), metric_eligible_rows=len(records), excluded_rows=len(excluded),
                raw=reporting.grouped_metrics(records, 'candidate'),
                fixed_demo=reporting.grouped_metrics(records, 'application_command'),
                relaxed_demo=reporting.grouped_metrics(records, 'relaxed_command'),
                qualified=reporting.grouped_metrics(records, 'qualified_command'),
                capture_fault_rows=sum(bool(r.get('capture_fault')) for r in records))
    gate = regression_gate(results['test']['relaxed_demo']['human']['overall'],
                           results['holdout']['relaxed_demo']['human']['overall'])
    generation._verify(cfg['bindings'])
    generation._verify(selection['bindings'])
    result = dict(experiment_id=cfg['experiment_id'], candidate=selection['candidate'],
        parameters=cfg['parameters'], maximum_parameters=compact.MAX_PARAMETERS,
        model_binding=read_json(bundle/'recipe.json')['binding'], graph_bytes=(bundle/'model.onnx').stat().st_size,
        precision='fp32', qualified=qualified['enabled'], installed=False,
        physical_pi_measured=False, published_data_previously_examined=True,
        test_used_for_tuning=False, regression_gate=gate, **results,
        bindings={**selection['bindings'], **{str(p.relative_to(ROOT)):file_digest(p) for p in (
            HERE/'selection-before-test.json', target/'test-predictions.json', target/'holdout-predictions.json',
            target/'test-excluded.jsonl', target/'holdout-excluded.jsonl')}})
    write_new(HERE/'results.json', result)
    print('REPAIR_REGRESSION_COMPLETE', json.dumps(dict(gate=gate, candidate=selection['candidate'])), flush=True)
    return result


def package_pi():
    cfg = freeze()
    result = read_json(HERE/'results.json')
    generation._verify(result['bindings'])
    if not result['regression_gate']['pass']:
        raise ValueError('Failed regression gate; preserve the current Pi archive')
    selection = read_json(HERE/'selection-before-test.json')
    bundle = ROOT/selection['bundle']
    base = release_context(CACHE/'publication')
    sample_dir = CACHE/'publication/pi-sample-source'
    sample_dir.mkdir(parents=True, exist_ok=False)
    source = base.sample_rows(read_jsonl(compact.PREPARED/'views/evaluation/test.jsonl'))
    manifest = sample_dir/'manifest.jsonl'
    write_new(manifest, source, jsonl=True)
    card = ROOT/'experiments/pcv-source-refresh-v1/source-card.txt'
    write_new(sample_dir/'protocol.json', dict(bindings={str(card.relative_to(ROOT)):file_digest(card)},
        selection='Same original licensed SLURP smoke samples; no quality selection'))
    ledger = read_json(ROOT/'data/sources/ai231-primary.source.json')
    allowed = {str(p['path']).replace('\\','/'):p['sha256'] for p in ledger['files']}
    files = []
    for name in sorted({r['parquet'] for r in source}):
        digest = file_digest(ROOT/name)
        if allowed.get(name) != digest:
            raise ValueError('Sample shard differs from pinned source')
        files.append(dict(path=name, sha256=digest))
    write_new(sample_dir/'acquisition.json', dict(repository='airimonda/ai231-me2-voice-commands',
        revision='25111444af3adff7588d86ab27c304bb080895cf',
        protocol_sha256=file_digest(sample_dir/'protocol.json'), files=files))
    shutil.copytree(ROOT/'.cache/speaker-final-refinement-v1/wake', bundle/'wake')
    from scripts.pcv_final_delivery import package
    packaged = package(bundle, CACHE/'publication/pi-base.zip', diagnostic=True,
        sample_manifest=manifest, source_receipt=sample_dir/'acquisition.json', source_card=card,
        extra_files={'docs/COMPACT-RECALL.md':ROOT/'docs/EXPERIMENT-speaker-pipeline-repair-v1.md',
            'evidence/results.json':HERE/'results.json', 'evidence/training-protocol.json':HERE/'protocol.json',
            'evidence/recall-policy.json':bundle/'recall-policy.json', 'evidence/references.json':HERE/'references.json'})
    output = ROOT/'dist/speaker-pipeline-repair-v1/speaker-pi4-pipeline-repair.zip'
    packaged.update(reporting.complete_archive(CACHE/'publication/pi-base.zip', output, bundle))
    generation._verify(cfg['installed_metadata'])
    write_new(HERE/'package.json', packaged)
    return packaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','reuse','train','candidate','regression','package_pi'])
    args = parser.parse_args()
    result = globals()[args.stage]()
    print(json.dumps(dict(stage=args.stage, status='complete', promoted=result.get('pass', False))))


if __name__ == '__main__':
    main()
