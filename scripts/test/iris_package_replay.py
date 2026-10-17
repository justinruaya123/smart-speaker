"""Audit a pinned extracted Pi package offline using its existing replay CLI."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.iris_release import ROOT, read, sha, verify_package
from scripts.iris_reproduce import compare_bundle


CHILD = '''
import json, sys
from pathlib import Path
attempts = []
def audit(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.sendto', 'socket.getaddrinfo'):
        attempts.append(event)
        raise PermissionError('Offline replay denies outbound network: ' + event)
sys.addaudithook(audit)
sys.path.insert(0, str(Path.cwd()))
try:
    from scripts.runtime.iris_pi_benchmark import main
    main(['--bundle', 'models/default', '--manifest', 'samples/commands.jsonl',
          '--output', sys.argv[2], '--threads', '1', '--repeats', '3', '--engine', sys.argv[1]])
finally:
    modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
               if (name == 'scripts' or name.startswith('scripts.')) and getattr(module, '__file__', None)}
    Path(sys.argv[3]).write_text(json.dumps({'network_attempts': attempts,
        'audit_installed_before_runtime_import': True, 'runtime_modules': modules}) + '\\n')
'''


def replay(package, expected, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; preserve it and choose a new --output: ' + str(output))
    if sha(package) != expected.lower():
        raise ValueError('Package SHA-256 differs from --sha256')
    before = verify_package(package)
    reference = compare_bundle(ROOT / 'artifacts/iris', package)
    if not reference or not all(item['matches'] for item in reference.values()):
        raise ValueError('Package differs from the approved artifacts/iris bundle')
    output.mkdir(parents=True, exist_ok=False)
    install = output / 'install'
    extracted = verify_package(package, install)
    bindings = {**before['manifest']['files'], 'MANIFEST.json': sha(install / 'MANIFEST.json')}

    def verify_extracted():
        actual = {p.relative_to(install).as_posix() for p in install.rglob('*') if p.is_file()}
        if actual != set(bindings) or any(sha(install / name) != digest for name, digest in bindings.items()):
            raise ValueError('Extracted package inventory or hashes changed')
    verify_extracted()
    population = [json.loads(line) for line in (install / 'samples/commands.jsonl').read_text().splitlines() if line.strip()]
    expected_order = [(str(row['id']), repeat) for row in population for repeat in range(3)]
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[key] = '1'
    engines, decisions = {}, {}
    for engine in ('ncnn', 'onnxruntime'):
        folder, network_path = output / engine, output / (engine + '-network.json')
        with (output / (engine + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-B', '-c', CHILD, engine, str(folder), str(network_path)],
                           cwd=install, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics, network = read(folder / 'metrics.json'), read(network_path)
        modules = network['runtime_modules']
        if (network['network_attempts'] or not modules
                or any(not Path(path).is_relative_to(install) for path in modules.values())
                or modules.get('scripts.runtime.iris_pi_benchmark') != str(install / 'scripts/runtime/iris_pi_benchmark.py')):
            raise ValueError('Replay attempted network access or imported checkout runtime modules')
        if (metrics['actions_enabled'] is not False or metrics['threads'] != 1 or metrics['repeats'] != 3
                or metrics['unstable_output_source_n'] or metrics['source_n'] != len(population)
                or metrics['observation_n'] != len(expected_order)):
            raise ValueError('Replay actions, stability or population verification failed')
        trials = [json.loads(line) for line in (folder / 'trials.jsonl').read_text().splitlines()]
        if [(row['id'], row['repeat']) for row in trials] != expected_order:
            raise ValueError('Replay ordered IDs/repeats differ from the packaged population')
        decisions[engine] = [(row['id'], row['repeat'], json.dumps(
            [row['pcm_sha256'], row['raw'], row['output']], sort_keys=True)) for row in trials]
        for index in range(0, len(expected_order), 3):
            if len({row[2] for row in decisions[engine][index:index + 3]}) != 1:
                raise ValueError('PCM/raw/typed decisions changed between repeats')
        engines[engine] = {key: metrics[key] for key in ('accepted', 'raw', 'source_n', 'observation_n',
            'latency_ms', 'session_load_ms', 'peak_process_rss_mib', 'model_sha256', 'engine_model_sha256',
            'threads', 'frontend_thread_environment', 'physical_pi_measured', 'software_versions')}
        engines[engine].update(metrics_sha256=sha(folder / 'metrics.json'),
            trials_sha256=sha(folder / 'trials.jsonl'), network=network)
    if decisions['ncnn'] != decisions['onnxruntime']:
        raise ValueError('Engine PCM/raw/typed decisions differ')
    verify_extracted()
    after = verify_package(package)
    if any(item['sha256'] != expected.lower() for item in (before, extracted, after)):
        raise ValueError('Package SHA-256 changed during verification')
    result = {'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'package': {'path': str(package), 'sha256': after['sha256'], 'bytes': after['bytes'],
                    'manifest_sha256': bindings['MANIFEST.json'], 'files_verified_before_and_after': len(bindings) - 1},
        'approved_bundle_comparison': reference, 'engines': engines,
        'comparison': {'identical': True, 'raw_and_typed_decision_pairs': len(expected_order)},
        'population': {'manifest': 'samples/commands.jsonl', 'sha256': bindings['samples/commands.jsonl'],
                       'source_n': len(population), 'repeats': 3},
        'execution': {'interpreter': sys.executable, 'extracted_package_modules_only': True,
                      'actions_enabled': False, 'wake_tested': False, 'owner_gating_enabled': False,
                      'physical_microphone_measured': False, 'scope': 'Command-only packaged diagnostic replay'},
        'audit_source_sha256': sha(Path(__file__))}
    with (output / 'verification.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory; partial runs are preserved')
    args = parser.parse_args(argv)
    result = replay(args.package, args.sha256, args.output)
    print(json.dumps({'passed': result['passed'], 'output': str(args.output), 'comparison': result['comparison']}))


if __name__ == '__main__':
    main()
