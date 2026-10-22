"""Separate absent non-fitting supervision fields from actual label-index errors.

The preserved first audit counted absent integers as disagreements. This companion
check changes no source, split, label, prediction, selection or replay result.
"""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.runtime.pcv_schema import INTENTS, canonical
from scripts.test.speaker_demo_suite import file_digest, read_json, write_new

D = ROOT / 'experiments/speaker-optionb-acceptance-v1'
selection = read_json(D / 'selection-before-replay.json')
for name, digest in selection['bindings'].items():
    assert file_digest(ROOT / name) == digest, name
recipe_path = ROOT / 'submission/reproduction-full/recipe.json'
recipe = read_json(recipe_path)
rows_path = recipe_path.parent / recipe['rows_file']
assert file_digest(rows_path) == recipe['rows_sha256']
with gzip.open(rows_path, 'rt', encoding='utf-8') as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
index = {r['id']: r for r in rows}
report = dict(format='optionb-reference-field-availability-v1',
              preserved_initial_audit_sha256=file_digest(D / 'fit-reference-audit.json'),
              selection_sha256=file_digest(D / 'selection-before-replay.json'),
              driver_sha256=file_digest(__file__), views={},
              caveat='Missing evaluation-only training-head integers are not mislabeled audio or wrong predicted heads. Evaluation uses frozen typed references and actual diagnostic head predictions.',
              changes_to_source_labels_or_fitting=False)
for split in ('train', 'dev', 'calibration', 'test', 'holdout'):
    members = [index[i] for i in recipe['views'][split]]
    status = {field: Counter() for field in ('intent_index', 'target_index')}
    for row in members:
        command = row['expected']
        expected = dict(intent_index=INTENTS.index(command['intent']) + 1 if command else 0,
                        target_index=int(command['slots']['light_id'] != 'light.all')
                        if command and 'light_id' in command['slots'] else -100)
        for field, gold in expected.items():
            value = row.get(field)
            if value is None:
                label = 'not_supplied'
            elif type(value) is not int:
                label = 'invalid_present_type'
            elif value == gold:
                label = 'present_consistent'
            else:
                label = 'present_disagreement'
            status[field][label] += 1
    report['views'][split] = dict(rows=len(members), fields={key: dict(value) for key, value in status.items()})
train = report['views']['train']
assert all(values == {'present_consistent': train['rows']} for values in train['fields'].values())
assert all(sum(values.values()) == data['rows'] for data in report['views'].values() for values in data['fields'].values())
write_new(D / 'reference-field-check.json', report)
print(json.dumps(report, indent=2))
