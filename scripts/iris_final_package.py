"""Safe source-layout Iris publication ZIPs, via the single archive_io writer.

Packaging never fits, acquires models, captures, calls providers or promotes.
--build-ui may provision pinned tooling and run the unchanged UI check/test/build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()

# Deliberate scope, not a recursive checkout/archive of local research state.
RUNTIME_FILES = (
    'scripts/iris_final_package.py', 'scripts/archive_io.py',
    'inference.py', 'benchmark.py', 'scripts/__init__.py', 'scripts/runtime/__init__.py',
    'scripts/iris_inference.py', 'scripts/iris_epoch44_benchmark.py',
    'scripts/runtime/speaker_worker.py', 'scripts/runtime/iris_epoch44.py',
    'scripts/runtime/iris_pi_benchmark.py', 'scripts/runtime/pcv_schema.py',
    'scripts/runtime/pcv_benchmark_adapter.py', 'scripts/runtime/mvp_static.py',
    'scripts/runtime/mvp_wake.py', 'scripts/runtime/wake_gate.py',
    'scripts/runtime/wake_models.py', 'scripts/runtime/iris_owner.py',
    'scripts/runtime/speaker_session.py', 'scripts/runtime/audio.py',
    'scripts/runtime/iris_feedback.py', 'scripts/runtime/speaker_config.py',
    'scripts/runtime/speaker_start.py', 'scripts/runtime/speaker_tools.py',
    'scripts/runtime/speaker_package.py', 'scripts/runtime/pyproject.toml',
    'scripts/runtime/uv.lock', 'vcm/__init__.py', 'vcm/frontend.py',
)
MODEL_FILES = tuple('submission/models/optionb-tiny/' + name for name in (
    'model.onnx', 'commands.json', 'normalization.json', 'policy.json', 'recipe.json'))
COMPANION_FILES = tuple('artifacts/iris/' + name for name in (
    'wake/iris.json', 'wake/iris.onnx', 'wake/iris.ncnn.bin', 'wake/iris.ncnn.param',
    'wake/silero-vad.onnx', 'wake/wake-models.json',
    'owner/owner.json', 'owner/owner.onnx'))
FEEDBACK_FILES = ('submission/feedback/manifest.json',) + tuple(
    'submission/feedback/' + cue + suffix
    for cue in ('wake_yes', 'wake_hmm', 'wake_yeah', 'wake_here', 'intent', 'success', 'failure')
    for suffix in ('.mp3', '.wav'))
OWW_FILES = ('scripts/runtime/iris_openwakeword.py',) + tuple(
    'experiments/iris-openwakeword-publication-v1/' + name for name in (
        'LICENSE-ASHR12-MIT.txt', 'LICENSE-openWakeWord-Apache-2.0.txt',
        'baseline-cases.json', 'companion-test-cases.json', 'findings.json',
        'interfaces.json', 'parity.json', 'protocol.json', 'provenance.json',
        'regression-started.json', 'regression-verification.json', 'results.json',
        'selection-before-regression.json', 'verification.json'))
EVIDENCE_FILES = tuple('experiments/iris-publication-v1/' + name for name in (
    'desktop-product-20261024.json', 'reference-pc-under-load-20261024.json', 'wake-disposition.json'))
DOC_FILES = tuple('docs/' + name for name in (
    'IRIS-GETTING-STARTED.md', 'MODEL-CARD-iris-epoch44.md',
    'RELEASE-iris-publication-v1.md', 'REVERT-iris-publication-v1.md',
    'REPRODUCIBILITY-optionb-epoch44.md', 'DATASETS.md'))
NOTICE_FILES = ('artifacts/silero-LICENSE.txt', 'scripts/runtime/ncnn-LICENSE.txt')
GENERATED_FILES = ('START-HERE.md', 'BENCHMARK-PROTOCOL.json', 'config.example.toml', '.env.example', 'deployment.json',
                   'SOURCE-CLOSURE.json', 'RESEARCH-STATUS.json', 'UI-BUILD.json')
UI_ROOT_FILES = tuple('SpeakerUI/' + name for name in (
    'package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml', 'UPSTREAM.json',
    'index.html', 'components.json', 'vite.config.ts', 'svelte.config.js',
    'tsconfig.json', 'tsconfig.server.json'))
UI_SOURCE_PREFIXES = ('SpeakerUI/src/', 'SpeakerUI/server/', 'SpeakerUI/shared/',
                      'SpeakerUI/scripts/', 'SpeakerUI/public/')
UI_NOTICE_FILES = tuple('SpeakerUI/mockups/ambient-v6/licenses/' + name for name in (
    'AURA-NOTICE.txt', 'LIVEKIT-APACHE-2.0.txt', 'LUCIDE.txt'))
UI_BUILD_RECEIPT = '.cache/iris-final-package/ui-build.json'
FULL_EVIDENCE_PREFIX = 'benchmark-evidence/full-reference-20261024/'
FULL_EVIDENCE_FILES = (
    'audio-preflight.json', 'benchmark.md', 'benchmark.xlsx', 'build_full_report.py',
    'complete-command-confusion.json', 'component-observations.jsonl', 'full_benchmark.py',
    'graph-inspection.json', 'observations.jsonl', 'per-command-slots.csv', 'per-intent.csv',
    'per-phrase.csv', 'predictions.jsonl', 'protocol.json', 'resource-probe-inputs.json',
    'resource-results.json', 'resource_probe.py', 'results.json', 'slot-marginals.csv',
    'stress-predictions.jsonl', 'test_full_benchmark.py', 'wrong-accepts.csv',
)


def benchmark_evidence(directory):
    """Copy verified, existing as-run reporting evidence, never audio or fit state."""
    from scripts.runtime.iris_epoch44 import FROZEN_HASHES
    directory = Path(directory).resolve()
    verification_data = read_source(directory, 'verification.json')
    verification = json.loads(verification_data)
    if set(verification.get('output_files', {})) != set(FULL_EVIDENCE_FILES):
        raise ValueError('Unexpected full benchmark evidence inventory')
    entries = {}
    for name in FULL_EVIDENCE_FILES:
        data = read_source(directory, name)
        pin = verification['output_files'][name]
        if len(data) != pin['bytes'] or sha(data) != pin['sha256']:
            raise ValueError('Changed full benchmark evidence: ' + name)
        entries[FULL_EVIDENCE_PREFIX + name] = data
    def parsed(name):
        return json.loads(entries[FULL_EVIDENCE_PREFIX + name])
    result, protocol = parsed('results.json'), parsed('protocol.json')
    if (verification.get('status') != 'verified' or verification.get('actions_enabled') is not False
            or result.get('actions_enabled') is not False or protocol.get('no_training') is not True
            or result.get('original_recordings') != 4614 or result.get('timing_observations') != 13842
            or result.get('model', {}).get('bindings') != FROZEN_HASHES):
        raise ValueError('Full benchmark evidence is not the verified frozen epoch44 run')
    predictions = [json.loads(line) for line in entries[FULL_EVIDENCE_PREFIX + 'predictions.jsonl'].splitlines()]
    observations = [json.loads(line) for line in entries[FULL_EVIDENCE_PREFIX + 'observations.jsonl'].splitlines()]
    identities = [row['id'] for row in predictions]
    pairs = {(row['id'], row['repeat']) for row in observations}
    if (len(predictions) != 4614 or len(set(identities)) != 4614 or len(observations) != 13842
            or pairs != {(identity, repeat) for identity in identities for repeat in range(3)}):
        raise ValueError('Full benchmark evidence row membership/count mismatch')
    entries[FULL_EVIDENCE_PREFIX + 'verification.json'] = verification_data
    return entries


def ui_source_name(name):
    return (name in UI_ROOT_FILES + UI_NOTICE_FILES or
            name.startswith(UI_SOURCE_PREFIXES) and PurePosixPath(name).suffix in
            {'.ts', '.js', '.mjs', '.cjs', '.svelte', '.css', '.json', '.html', '.svg'})


def ui_compiled_name(name):
    return (name.startswith(('SpeakerUI/dist/', 'SpeakerUI/dist-server/'))
            and (name.startswith('SpeakerUI/dist/licenses/') and PurePosixPath(name).suffix == '.txt'
                 or PurePosixPath(name).suffix in {'.js', '.css', '.html', '.svg', '.png', '.ico',
                                                   '.webp', '.woff', '.woff2', '.json', '.gz', '.br'}))


def ui_source_names(root):
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', 'SpeakerUI']).decode().split('\0')
    return sorted(name for name in names if ui_source_name(name))


def ui_binding(root) -> dict:
    source = {name: sha(read_source(root, name)) for name in ui_source_names(root)}
    compiled = {file.relative_to(root).as_posix(): sha(read_source(root, file.relative_to(root).as_posix()))
                for folder in ('SpeakerUI/dist', 'SpeakerUI/dist-server')
                for file in (root / folder).rglob('*') if file.is_file()}
    if not {'SpeakerUI/dist/index.html', 'SpeakerUI/dist-server/server/main.js'} <= compiled.keys():
        raise ValueError('Build the combined SpeakerUI before packaging inference')
    return dict(lock_sha256=source['SpeakerUI/pnpm-lock.yaml'],
                source_sha256=source, compiled_sha256=compiled)


def build_ui(root):
    """Use pinned local tools and the unchanged UI scripts/lock, never Python setup."""
    from scripts.runtime.speaker_tools import node_tools
    import sys
    root = Path(root).resolve()
    node, pnpm, env = node_tools(root)
    env.update(SMART_SPEAKER_PYTHON=sys.executable, ACTIONS_ENABLED='false',
               HA_ENABLED='false', SPOTIFY_ENABLED='false', OPENWEATHER_ENABLED='false')
    lock_before = sha(read_source(root, 'SpeakerUI/pnpm-lock.yaml'))
    for task in ('check', 'test', 'build'):
        subprocess.run([str(node), str(pnpm), 'run', task], cwd=root/'SpeakerUI', env=env, check=True)
    receipt = dict(schema_version=1, **ui_binding(root), tasks_passed=['check', 'test', 'build'],
                   node=subprocess.check_output([str(node), '--version'], text=True).strip(),
                   pnpm=subprocess.check_output([str(node), str(pnpm), '--version'], env=env, text=True).strip())
    if receipt['lock_sha256'] != lock_before:
        raise ValueError('UI lock changed during build')
    destination = root / UI_BUILD_RECEIPT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(json_bytes(receipt))
    return receipt


def validate_entries(entries, *, target):
    """Reject private/unapproved paths even when they occur inside allowed trees."""
    if target not in ('inference', 'benchmark'):
        raise ValueError('Unknown package target')
    approved = set(RUNTIME_FILES + MODEL_FILES + COMPANION_FILES + FEEDBACK_FILES
                   + OWW_FILES + EVIDENCE_FILES + DOC_FILES + NOTICE_FILES + GENERATED_FILES)
    for name in entries:
        path = PurePosixPath(name)
        if (not isinstance(name, str) or not name or ':' in name or '\\' in name
                or path.is_absolute() or path.as_posix() != name or '..' in path.parts
                or (name != '.env.example' and any(part.startswith('.') for part in path.parts))
                or not (name in approved or target == 'inference' and
                        (ui_source_name(name) or ui_compiled_name(name))
                        or target == 'benchmark' and name in {
                            FULL_EVIDENCE_PREFIX + filename for filename in (*FULL_EVIDENCE_FILES, 'verification.json')})):
            raise ValueError('Unapproved package payload: ' + str(name))


def read_source(root, name):
    """Never follow links out of (or within) the approved source tree."""
    root = Path(root).resolve()
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('Linked package input: ' + name)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Package input leaves source root: ' + name)
    return path.read_bytes()


def git_state(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    return dict(source_revision=git('rev-parse', 'HEAD').decode().strip(),
                source_is_clean=not bool(git('status', '--porcelain', '--untracked-files=all')))


def assemble_files(root, *, target, allow_dirty=False, benchmark_evidence_dir=None,
                   source_state=None):
    """Read a deliberate source closure; no local state, scientific setup or fit."""
    from scripts.runtime.iris_epoch44 import FROZEN_HASHES
    root = Path(root).resolve()
    state = git_state(root) if source_state is None else dict(source_state)
    if not state['source_is_clean'] and not allow_dirty:
        raise ValueError('Commit final source before publication; --allow-dirty is verification-only')
    names = RUNTIME_FILES + MODEL_FILES + COMPANION_FILES + FEEDBACK_FILES
    names += OWW_FILES + EVIDENCE_FILES + DOC_FILES + NOTICE_FILES
    entries = {name: read_source(root, name) for name in names}
    for name, expected in FROZEN_HASHES.items():
        if sha(entries['submission/models/optionb-tiny/' + name]) != expected:
            raise ValueError('Frozen epoch44 identity mismatch: ' + name)
    findings = json.loads(entries['experiments/iris-openwakeword-publication-v1/findings.json'])
    if findings['decision'] != 'DO_NOT_PROMOTE' or findings['runtime']['integration_performed'] is not False:
        raise ValueError('Unreviewed OpenWakeWord promotion/status change')
    entries['config.example.toml'] = (
        '# Copy to private config.toml only after review. Never enables actions by default.\n'
        '[inference]\nengine = "onnxruntime"\nmode = "demo"\nprofile = "owner-protected"\nthreads = 1\n'
        '[actions]\nenabled = false\n[providers]\nha_enabled = false\n').encode()
    entries['.env.example'] = b'# Private local settings only. No credentials are supplied.\nACTIONS_ENABLED=false\nHA_ENABLED=false\n'
    entries['BENCHMARK-PROTOCOL.json'] = json_bytes(dict(schema_version=1,
        entrypoint='benchmark.py', engine='onnxruntime CPU', qualified=False, actions_enabled=False,
        original_mono_sample_rate=16000, no_mixing_or_resampling=True,
        no_fitting_or_policy_selection=True, repeats_increase_accuracy_denominators=False,
        full_reference=dict(original_recordings=4614, eligible_recordings=4427,
            ambiguous_unscored=187, ambiguous_test=185, ambiguous_holdout=2,
            timing_observations=13842, repeats=3,
            evidence=FULL_EVIDENCE_PREFIX + 'protocol.json',
            driver='archived as-run reference driver; not the current adapter'),
        adapter_human=dict(original_recordings=1146, eligible_recordings=959,
            ambiguous_unscored=187, positive_n=689, negative_n=270,
            evidence='experiments/iris-publication-v1/desktop-product-20261024.json',
            driver='current epoch44 adapter; separately measured desktop population'),
        manifest_required_fields=['id', 'audio (safe relative original WAV path)',
                                  'sha256 (original WAV bytes)', 'expected (typed command or null)'],
        ambiguous_reference_fields=dict(metric_eligible=False, reference_status='ambiguous'),
        timing_includes=['PCM validation', 'frontend/normalization', 'ONNX CPU', 'typed decision'],
        timing_excludes=['WAV reads/hash/decoding', 'utterance duration', 'wake/VAD waits',
                         'owner', 'UI/gateway', 'feedback', 'providers'],
        evidence_notice='Archived measurements are not a new ZIP performance run; non-quiescent desktop, not Pi'))
    entries['deployment.json'] = json_bytes(dict(schema_version=1, target=target,
        entrypoint='inference.py' if target == 'inference' else 'benchmark.py',
        actions_enabled=False, qualified=False, engine='onnxruntime',
        model='submission/models/optionb-tiny', parameters=258912, selected_epoch=44,
        wake_backend='native_experimental_failed_gate', openwakeword_installed=False,
        source_layout=True, training_performed=False))
    entries['RESEARCH-STATUS.json'] = json_bytes(dict(qualified=False, actions_enabled=False,
        wake=dict(status='FAILED_RELEASE_GATE', native_positive_n=1, companion_positive_n=21,
                  positive_n=32, companion_false_wake_n=2, negative_n=68, replacement_promoted=False),
        openwakeword=dict(decision='DO_NOT_PROMOTE', installed=False, promoted=False,
            third_party_weights_included=False, integration_performed=False,
            blocker='Failed acoustic gates; unresolved feature-weight redistribution; missing WakeGate.audio seam',
            findings='experiments/iris-openwakeword-publication-v1/findings.json'),
        physical_pi_measured=False, live_microphone_qualified=False,
        ownership='experimental; not authentication', fitting_performed=False))
    entries['START-HERE.md'] = (
        '# Iris epoch44 research publication\n\n'
        '258,912 parameters; frozen epoch 44. Qualification is FALSE and actions are disabled.\n'
        'Native and same-size companion wake gates FAILED. No replacement is promoted.\n'
        'OpenWakeWord is uninstalled research SOURCE ONLY: failed gates, missing live integration seam, '
        'and unresolved feature-weight redistribution. No third-party OpenWakeWord weights are included.\n\n'
        '## Check without setup\n\nUse an already installed compatible Python runtime:\n'
        '```sh\npython -B inference.py --check --no-setup\n```\n'
        'This loads the real VCM/native wake/VAD/owner, without microphone, playback or actions.\n'
        'For first-time Python setup, `python inference.py --check` uses the unchanged '
        '`scripts/runtime/pyproject.toml` and `uv.lock`. Network and compatible wheels are required.\n'
        'The inference ZIP contains compiled combined SpeakerUI plus its build sources and lock. '
        'First UI launch `python inference.py --mode ui --no-audio --no-browser` provisions pinned '
        'Node24.13.0/pnpm12.5.1 and dependencies; tools/node_modules are intentionally not redistributed.\n'
        'Do not use historical start.py/ncnn/969k instructions as this publication default.\n\n'
        '## Action-free benchmark\n\nSupply your own permitted, original hash-bound mono16kHz recordings:\n'
        '```sh\npython -B benchmark.py --model-choice epoch44 --manifest MANIFEST.jsonl '
        '--data-root ORIGINAL_ROOT --output NEW_RESULTS --repeats 3\n```\n'
        'No datasets, recordings, completed fits, credentials or caches are bundled. '
        'The only audio is the deliberately published 14 feedback MP3/WAV assets, not evaluation recordings.\n'
        'See docs/IRIS-GETTING-STARTED.md for exact original evaluation reconstruction and manifest schema. '
        'Reproduction/fitting wrappers are repository workflows, not part of this inference source closure.\n'
        'BENCHMARK-PROTOCOL.json defines the replay contract and distinct populations: '
        'full reference = 4,614 original recordings / 4,427 eligible / 187 ambiguous excluded '
        '(185 TEST + 2 HOLDOUT), with 13,842 timing observations from 3 repeats; '
        'current-adapter human subset = 1,146 recordings / 959 eligible / 187 ambiguous excluded. '
        'Repeats never increase accuracy denominators. '
        'The benchmark ZIP retains 23 hash-checked full-reference report/protocol/evidence files '
        'under benchmark-evidence/full-reference-20261024/. Their archived scripts bind the original '
        'snapshot environment and are not the default runnable driver; use the bundled benchmark.py.\n'
        'Published desktop evidence is archival as-run, under non-quiescent load, not a measurement '
        'of this ZIP, end-to-end latency, Pi, or wake/owner qualification.\n'
        'SOURCE-CLOSURE.json and MANIFEST.json bind the bytes actually bundled; source revision alone '
        'does not describe a dirty verification build.\n').encode()
    if target == 'inference':
        binding = ui_binding(root)
        try:
            receipt = json.loads(read_source(root, UI_BUILD_RECEIPT))
        except FileNotFoundError as error:
            raise ValueError('Missing UI build receipt; rebuild using --build-ui') from error
        if (any(receipt.get(key) != value for key, value in binding.items())
                or receipt.get('tasks_passed') != ['check', 'test', 'build']
                or receipt.get('node') != 'v24.13.0' or receipt.get('pnpm') != '12.5.1'):
            raise ValueError('UI build receipt is stale or not pinned; rebuild using --build-ui')
        for name in binding['source_sha256'].keys() | binding['compiled_sha256'].keys():
            entries[name] = read_source(root, name)
        entries['UI-BUILD.json'] = json_bytes(receipt)
        state['ui_build'] = receipt
    if target == 'benchmark' and benchmark_evidence_dir is not None:
        entries.update(benchmark_evidence(benchmark_evidence_dir))
    entries['SOURCE-CLOSURE.json'] = json_bytes(dict(schema_version=1, **state,
        verification_only=not state['source_is_clean'],
        omitted_optional_paths=['legacy research audio resolver', 'legacy start.py and legacy model weights',
                               'repository-only preparation/fitting workflows'],
        files={name: sha(data) for name, data in sorted(entries.items())}))
    validate_entries(entries, target=target)
    return entries


def build_archive(root, output, *, target, allow_dirty=False, benchmark_evidence_dir=None):
    """Use the existing immutable writer; never create another ZIP implementation."""
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('ZIP outputs are immutable')
    entries = assemble_files(root, target=target, allow_dirty=allow_dirty,
                             benchmark_evidence_dir=benchmark_evidence_dir)
    return _write_assembled_archive(entries, output, target=target)


def _write_assembled_archive(entries, output, *, target):
    """Write retained bytes only; archive_io remains the sole ZIP writer."""
    from scripts import archive_io
    return archive_io.write_archive(entries, output, dict(
        format='iris-final-source-layout-v1', target=target, qualified=False,
        actions_enabled=False, source=json.loads(entries['SOURCE-CLOSURE.json']),
        model_sha256=sha(entries['submission/models/optionb-tiny/model.onnx'])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='NEW immutable pair output directory')
    parser.add_argument('--benchmark-evidence', required=True, type=Path,
                        help='Existing verified full-reference 20261024 evidence directory; no audio')
    parser.add_argument('--build-ui', action='store_true', help='Run pinned unchanged UI check/test/build first')
    parser.add_argument('--allow-dirty', action='store_true', help='Explicitly labelled verification-only build')
    args = parser.parse_args(argv)
    output = args.output_dir.absolute()
    try:
        if output.exists():
            raise FileExistsError('Output directory is immutable; choose a NEW --output-dir')
        state = git_state(ROOT)
        if not args.allow_dirty and not state['source_is_clean']:
            raise ValueError('Commit final source before publication; --allow-dirty is verification-only')
        # Fail before creating an output tree or starting UI work if evidence is invalid.
        benchmark_evidence(args.benchmark_evidence)
        if args.build_ui:
            build_ui(ROOT)
            state = git_state(ROOT)
        # Freeze both closures and their common identity before any output write.
        payloads = {target: assemble_files(ROOT, target=target, allow_dirty=args.allow_dirty,
                        benchmark_evidence_dir=args.benchmark_evidence, source_state=state)
                    for target in ('inference', 'benchmark')}
        output.mkdir(parents=True, exist_ok=False)
        artifacts = [_write_assembled_archive(payloads[target], output / f'iris-final-{target}.zip',
                     target=target)
                     for target in ('inference', 'benchmark')]
        report = dict(schema_version=1, **state, actions_enabled=False, qualified=False,
                      verification_only=args.allow_dirty, artifacts=artifacts)
        (output / 'BUILD-REPORT.json').write_bytes(json_bytes(report))
        (output / 'SHA256SUMS.txt').write_text(''.join(
            f"{item['sha256']}  {Path(item['path']).name}\n" for item in artifacts), encoding='utf-8')
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
