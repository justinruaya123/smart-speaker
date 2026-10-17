"""One bounded compact fit reusing the frozen trainer through isolated globals."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

from scripts.models.pcv_release_pooling import PooledContextVCM
from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.train import speaker_synthetic_data as generation

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'experiments/speaker-compact-recall-v1'
CACHE = ROOT/'.cache/speaker-compact-recall-v1'
PARENT = ROOT/'experiments/speaker-final-refinement-v1'
PREPARED = ROOT/'.cache/speaker-synthetic-completion-v1/prepared-v1'
ARCHITECTURE = dict(width=128, layers=6, heads=4, feedforward=256, dropout=.1)
MAX_PARAMETERS = 1200000


def check_parameters(model):
    count = sum(p.numel() for p in model.parameters())
    if count > MAX_PARAMETERS:
        raise ValueError('VCM parameter ceiling exceeded')
    return count


def control_rows(rows):
    accepted = []; excluded = []
    for row in rows:
        reason = ('new_qwen_not_control' if row['id'].startswith('qsc1/') else
                  'unresolved_slot_metadata_conflict' if row.get('slot_metadata_conflict') else None)
        if reason:
            excluded.append({**row, 'compact_exclusion':reason})
        else:
            accepted.append(row)
    return accepted, excluded


def shared_trainer():
    spec = importlib.util.spec_from_file_location('scripts.train._compact_shared', ROOT/'scripts/train/speaker_final_train.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.HERE = HERE; module.CACHE = CACHE; module.ARCHITECTURE = dict(ARCHITECTURE)
    return module


def freeze():
    path = HERE/'protocol.json'
    if path.exists():
        cfg = read_json(path); generation._verify(cfg['bindings']); return cfg
    parent = read_json(PARENT/'protocol.json')
    generation._verify(parent['bindings']); generation._verify(parent['installed_metadata'])
    shared = shared_trainer(); rows,dev,norm,entries = shared.load_inputs()
    train,excluded = control_rows(rows)
    HERE.mkdir(parents=True, exist_ok=True)
    write_new(CACHE/'prepared/train.jsonl', train, jsonl=True)
    write_new(CACHE/'prepared/dev.jsonl', dev, jsonl=True)
    write_new(CACHE/'prepared/excluded.jsonl', excluded, jsonl=True)
    cfg = copy.deepcopy(parent)
    cfg.update(experiment_id='speaker-compact-recall-v1', architecture=ARCHITECTURE,
               parameters=check_parameters(PooledContextVCM(**ARCHITECTURE)), maximum_parameters=MAX_PARAMETERS,
               arms=['control'], maximum_int8_graph_bytes=2*1024*1024,
               authorization='User approved compact fit <=1.2M and development/calibration-selected relaxation of correct-command rejection',
               hypothesis='A compact pooled encoder and a human-fitted demo intent policy retain useful complete-command recall within the smaller footprint',
               initialization='one random-init human foundation then one mixed-control continuation',
               normalization='Parent human-training-only normalization (1869 utterances), unchanged; excludes no evaluation data',
               human_qualified=False, physical_pi_measured=False,
               training_rows=len(train), exclusions=len(excluded),
               label_audit='Metadata conflict exclusion only; no independent listening or label corrections',
               recall_policy=dict(intent_thresholds=[.5,.6,.7,.8,.85,.9,.95],slot_threshold=.9,
                   fitting_views=['human/development','human/calibration'],
                   missing_class='retain .95',nonregression='no additional wrong or UNKNOWN accepts on either human view'),
               limitation='Single seed; compact width and supervision filtering change together; relaxed policy cannot certify missing human calibration classes')
    cfg['bindings'].update({str(p.relative_to(ROOT)):file_digest(p) for p in (
        PARENT/'protocol.json',Path(__file__),CACHE/'prepared/train.jsonl',
        CACHE/'prepared/dev.jsonl',CACHE/'prepared/excluded.jsonl')})
    cfg.pop('completion_batch', None)
    cfg['augmentation']['both_arms'] = False
    write_new(path,cfg)
    return cfg


def train():
    cfg = freeze(); shared = shared_trainer()
    if check_parameters(PooledContextVCM(**cfg['architecture'])) != cfg['parameters']:
        raise ValueError('Compact model parameter identity changed')
    original_load = shared.load_inputs
    def inputs():
        _,_,norm,entries = original_load()
        return read_jsonl(CACHE/'prepared/train.jsonl'), read_jsonl(CACHE/'prepared/dev.jsonl'), norm, entries
    shared.freeze = lambda: cfg
    shared.load_inputs = inputs
    return shared.train()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['freeze','train'])
    args = p.parse_args(); result = freeze() if args.stage=='freeze' else train()
    print(result['experiment_id'], args.stage, 'complete')


if __name__ == '__main__':
    main()
