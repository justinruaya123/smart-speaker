"""Finish the declared test stages after this task's local Dia process exits."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.pcv_three_engine import HERE, OUTPUT, ENGINES, immutable, verify
from scripts.train.pcv_public_data import read_rows, save, sha
from scripts.test.pcv_public_qualification import verify_bindings

PROTOCOL = HERE / 'completion-controller-protocol.json'
STATUS = OUTPUT / 'completion-controller-status.json'


def state(stage, **kwargs):
    save(STATUS, {'stage': stage, 'updated': datetime.now(timezone.utc).isoformat(), **kwargs})


def freeze(pid):
    import psutil
    verify()
    process = psutil.Process(pid)
    if not any('pcv_dia_bounded.py' in arg for arg in process.cmdline()):
        raise ValueError('The process is not this study\'s Dia generator')
    paths = [Path(__file__), HERE / 'protocol.json', HERE / 'report-v2-protocol.json',
             HERE / 'dia-single-protocol.json', HERE / 'dia-cpu-codec-protocol.json',
             HERE / 'dia-cpu-output-protocol.json', HERE / 'dia-bounded-protocol.json',
             HERE / 'dia-bounded-preflight.json', ROOT / 'scripts/train/pcv_dia_bounded.py',
             ROOT / 'scripts/test/pcv_three_engine_report.py']
    immutable(PROTOCOL, {'version': 'completion-controller-v1', 'dia_pid': pid,
        'dia_process_created': process.create_time(), 'other_processes_changed': False,
        'steps': ['wait for owned Dia completion and process exit', 'generate Parler',
                  'assemble and audit', 'freeze generation lineage', 'evaluate both frozen arms',
                  'compile report-v2', 'verify unchanged bindings'],
        'training_or_policy_changes': False, 'test_inference_accessed': False,
        'bindings': {str(p.relative_to(ROOT)): sha(p) for p in paths}})


def same_process_alive(pid, created):
    import psutil
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.create_time() == created
    except psutil.NoSuchProcess:
        return False


def check_controller_bindings(protocol):
    verify_bindings(ROOT, protocol['bindings'])
    for relative in protocol['bindings']:
        path = ROOT / relative
        if path.name.endswith('protocol.json'):
            child = json.loads(path.read_text())
            verify_bindings(ROOT, child.get('bindings', {}))


def step(name, args):
    log = OUTPUT / ('completion-' + name + '.log')
    with log.open('x', encoding='utf8') as stream:
        with subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT,
                              stdout=stream, stderr=subprocess.STDOUT) as process:
            state(name, child_pid=process.pid, log=str(log.relative_to(ROOT)))
            print('COMPLETION_STAGE', name, 'pid', process.pid, flush=True)
            code = process.wait()
    if code:
        raise RuntimeError(f'{name} failed with exit {code}; preserve {log}')


def lineage():
    paths = [PROTOCOL, HERE / 'data-audit.json']
    groups = {}
    for engine in ENGINES:
        receipt_path = HERE / (engine + '-generation-complete.json')
        receipt = json.loads(receipt_path.read_text())
        paths.append(receipt_path)
        rows = []
        if receipt['manifest_sha256']:
            verify_bindings(ROOT, {receipt['manifest']: receipt['manifest_sha256']})
            paths.append(ROOT / receipt['manifest'])
            rows += read_rows(ROOT / receipt['manifest'])
        if engine == 'qwen':
            paths.append(HERE / 'reused-qwen.json')
            rows += json.loads((HERE / 'reused-qwen.json').read_text())
        conditions = Counter()
        for row in rows:
            meta = row.get('generation_metadata', {})
            key = json.dumps({
                'source_version': row.get('source_version'),
                'generation_protocol_sha256': row.get('protocol_sha256'),
                'retained_parent_reuse': 'reused_parent_id' in row,
                'codec_device_recorded': meta.get('codec_device'),
                'max_new_tokens_recorded': meta.get('max_new_tokens'),
            }, sort_keys=True)
            conditions[key] += 1
        groups[engine] = {'recordings': len(rows), 'missing': receipt['missing'],
                          'conditions': [{**json.loads(k), 'n': n} for k, n in sorted(conditions.items())]}
    for name in ('parler-weights-before.json', 'parler-weights-after-first.json', 'parler-weights-final.json'):
        p = OUTPUT / name
        value = json.loads(p.read_text())
        if not value['verified_after'] or ('before' not in name and value['changed_before']):
            raise ValueError('Parler weight identity check failed')
        paths.append(p)
    immutable(HERE / 'generation-lineage.json', {'version': 'generation-lineage-v1',
        'engines': groups, 'all_inference_local': True, 'test_inference_accessed': False,
        'generation_revision_notes': 'First 15 Dia recordings use CUDA codec; later recordings use CPU codec. The final revision bounds new tokens at 1050. Full revision protocols are bound by the completion controller.',
        'metadata_null_means_not_recorded': True,
        'bindings': {str(p.relative_to(ROOT)): sha(p) for p in paths}})


def run():
    p = json.loads(PROTOCOL.read_text())
    check_controller_bindings(p)
    verify()
    state('waiting-for-dia', dia_pid=p['dia_pid'])
    while same_process_alive(p['dia_pid'], p['dia_process_created']):
        time.sleep(5)
    if not (HERE / 'dia-generation-complete.json').exists():
        raise RuntimeError('Dia exited without a completion receipt; no later stage started')
    step('parler', ['experiments/pcv-b1-three-engine-v1/run.py', 'generate', '--engine', 'parler'])
    step('assemble', ['experiments/pcv-b1-three-engine-v1/run.py', 'assemble'])
    lineage()
    step('evaluate', ['experiments/pcv-b1-three-engine-v1/run.py', 'evaluate'])
    step('report-v2', ['scripts/test/pcv_three_engine_report.py'])
    verify(assets=True)
    check_controller_bindings(p)
    proof = json.loads((HERE / 'generation-lineage.json').read_text())
    verify_bindings(ROOT, proof['bindings'])
    paths = [HERE / f'results-{arm}-v2.json' for arm in ('human', 'mixed')]
    immutable(HERE / 'completion-controller-result.json', {'stages_completed': True,
        'controller_protocol_sha256': sha(PROTOCOL),
        'results': {str(path.relative_to(ROOT)): sha(path) for path in paths},
        'policies_enabled': [json.loads(path.read_text())['policy_enabled'] for path in paths],
        'installed_artifacts_unchanged': True, 'device_actions': False})
    state('complete', result=str((HERE / 'completion-controller-result.json').relative_to(ROOT)))
    print('COMPLETION_FINISHED', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'freeze':
        freeze(int(sys.argv[2]))
    elif sys.argv[1:] == ['run']:
        try:
            run()
        except Exception as exc:
            state('failed', error=repr(exc))
            raise
    else:
        raise SystemExit('Use freeze <owned Dia PID> or run')
