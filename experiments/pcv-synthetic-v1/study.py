"""Versioned synthetic inputs; never changes the human-only PC-v1 contract."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PCV = ROOT / 'experiments/phrase-contract-v1'
CACHE = ROOT / '.cache/pcv-synthetic-v1'
sys.path.insert(0, str(PCV))
sys.path.insert(0, str(ROOT))
from pcvcm.contract import Contract, canonical, file_hash, identity, write_json

ENGINES = ('qwen', 'parler', 'dia')
VOICES = (
    dict(id='train-us-m', split='train', accent='American English', gender='man', parler='Jon', seed=7101),
    dict(id='train-gb-f', split='train', accent='British English', gender='woman', parler='Lea', seed=7102),
    dict(id='dev-au-m', split='dev', accent='Australian English', gender='man', parler='Gary', seed=7103),
    dict(id='cal-us-f', split='calibration', accent='American English', gender='woman', parler='Jenna', seed=7104),
    dict(id='test-in-f', split='test', accent='Indian English', gender='woman', parler='Laura', seed=7105),
    dict(id='test-gb-m', split='test', accent='British English', gender='man', parler='Mike', seed=7106),
)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ''.join(canonical(r) + '\n' for r in rows)
    if path.exists() and path.read_text(encoding='utf-8') != data:
        raise FileExistsError(f'Immutable study input changed: {path}')
    path.write_text(data, encoding='utf-8', newline='\n')


def variations(text, command):
    # Question word order is preserved rather than prefixing an imperative helper.
    if command and text.startswith('what'):
        query = text.replace('what time is it', 'what time it is').replace('what is the', 'the')
        query = query.replace("what's the", 'the')
        return [f'can you tell me {query}, please', f'I would like to know {query}']
    if command:
        imperative={'next track':'play the next track','previous track':'play the previous track',
                    'volume up':'turn the volume up','volume down':'turn the volume down'}.get(text,text)
        return [f'could you {imperative}, please', f'I would like you to {imperative} for me']
    return [f'hey assistant, {text}', f'{text}, please']


def build_verbatims():
    contract = Contract(PCV / 'spec')
    result = []
    for row in contract.positive + contract.negative:
        parent = row.get('phrase_id', row.get('rejection_id'))
        command = row.get('expected_command')
        for index, text in enumerate([row['exact_text'], *variations(row['exact_text'], command)]):
            result.append(dict(id=parent if index == 0 else f'{parent}-N{index}',
                parent_id=parent, phase=1 if index == 0 else 2,
                use_case=row['use_case'], text=text, expected_command=command,
                reference_class='supported_positive' if command else row['reference_class'],
                critical_zero_accept=row.get('critical_zero_accept', False)))
    if len({r['text'] for r in result}) != len(result):
        raise ValueError('Duplicate natural verbatim; review wording before freezing')
    return result


def generation_plan(rows):
    result = []
    for engine in ENGINES:
        for voice in VOICES:
            for row in rows:
                key = f'{engine}/{voice["id"]}/{row["id"]}'
                seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                result.append(dict(recording_id=key, verbatim_id=row['id'],
                    engine=engine, voice_group=voice['id'], split=voice['split'],
                    accent_requested=voice['accent'], accent_verified=False,
                    seed=seed, synthetic=True, audio=f'audio/{key}.wav',
                    native_audio=f'native/{key}.wav'))
    return result


def validate_plan(plan, rows):
    voice_splits = defaultdict(set)
    for row in plan:
        voice_splits[row['voice_group']].add(row['split'])
    if any(len(s) != 1 for s in voice_splits.values()):
        raise ValueError('Synthetic voice crosses data splits')
    counts = Counter((r['verbatim_id'], r['engine'], r['split']) for r in plan)
    expected = Counter({(r['id'], e, split): n for r in rows for e in ENGINES
                        for split, n in [('train', 2), ('dev', 1), ('calibration', 1), ('test', 2)]})
    if counts != expected or len({r['recording_id'] for r in plan}) != len(plan):
        raise ValueError('Incomplete or duplicate three-engine split coverage')


def validate_frozen_study():
    protocol=json.loads((HERE/'protocol.json').read_text())
    for field,name in [('verbatims_sha256','verbatims.jsonl'),('generation_plan_sha256','generation-plan.jsonl')]:
        if file_hash(HERE/name)!=protocol[field]:
            raise ValueError(f'Frozen study identity changed: {name}')
    if Contract(PCV/'spec').sha256!=protocol['base_contract']:
        raise ValueError('Original command contract changed')
    if protocol['voices']!=list(VOICES) or protocol['engine_families']!=list(ENGINES):
        raise ValueError('Frozen voice/engine assignments changed')
    return protocol


def freeze():
    rows = build_verbatims()
    plan = generation_plan(rows)
    validate_plan(plan, rows)
    write_rows(HERE / 'verbatims.jsonl', rows)
    write_rows(HERE / 'generation-plan.jsonl', plan)
    protocol = dict(version='pcv-synthetic-v1-r2', base_contract=Contract(PCV/'spec').sha256,
        verbatims_sha256=file_hash(HERE/'verbatims.jsonl'),
        generation_plan_sha256=file_hash(HERE/'generation-plan.jsonl'),
        expected_clips=len(plan), expected_verbatims=len(rows), voices=VOICES,
        engine_families=ENGINES, test_audio_used_for_selection=False,
        synthetic_acceptance_only=True, human_qualified=False,
        training_seeds=[53, 71], epochs_per_seed=40, max_total_epochs=200,
        max_training_seconds=86400, selection_split='dev', calibration_split='calibration',
        final_split='test', acceptance_precision=.99, acceptance_recall=.90, negative_far=.01,
        compatible_previous_verbatims={'revisions/r1/verbatims.jsonl':
            '90f404b1cfc04fe40907a1387b739cc88ae1afee98e2295e39c1481a4152263a'})
    path = HERE/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise FileExistsError('Protocol already frozen differently')
    write_json(path, protocol)
    lines = ['# Synthetic benchmark: full test verbatims', '',
             'Expected outputs only; generated audio and measured results are recorded separately.', '',
             '| ID | Phase | Use case | Say exactly | Expected command (null means reject) |',
             '| --- | --- | --- | --- | --- |']
    lines += [f'| {r["id"]} | {r["phase"]} | {r["use_case"]} | {r["text"]} | `{canonical(r["expected_command"])}` |' for r in rows]
    (ROOT/'docs/PCV-SYNTHETIC-VERBATIMS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return protocol


if __name__ == '__main__':
    print(json.dumps(freeze(), indent=2))
