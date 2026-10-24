"""Engineering extraction checks, not new acoustic or historical-fit evidence.

The oracle is the immutable pre-migration Git blob, never a copied expectation.
Only capture/gate/UI entry points are omitted from the training references.
"""
from __future__ import annotations

import ast
from dataclasses import asdict
from functools import lru_cache
import importlib
import json
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '94c0021d7e22c9aa719246ac97105d7140a9f2a2'
# None means every pure top-level definition/constant from that module.
EXTRACTIONS = (
    ('scripts/train/iris_schema.py', 'scripts/runtime/pcv_schema.py', None),
    ('scripts/train/iris_validation.py', 'scripts/runtime/mvp_static.py', None),
    ('scripts/train/iris_audio.py', 'scripts/runtime/audio.py', None),
    ('scripts/train/iris_owner_reference.py', 'scripts/runtime/iris_owner.py', None),
    ('scripts/train/iris_session.py', 'scripts/runtime/speaker_session.py', None),
    ('scripts/train/iris_wake_reference.py', 'scripts/runtime/mvp_wake.py', {'RollingWake', 'HeyIris'}),
    ('scripts/training_tools.py', 'scripts/runtime/speaker_tools.py',
     {'UV_VERSION', 'UV_WHEELS', 'run', 'find_uv', 'python_path', 'sync_python'}),
    ('scripts/train/iris_openwakeword_reference.py', 'scripts/runtime/iris_openwakeword.py', None),
    ('scripts/train/iris_benchmark_serializer.py', 'scripts/runtime/pcv_benchmark_adapter.py',
     {'class_prediction', 'log_record'}),
)
RELOCATIONS = {
    'scripts.runtime.pcv_schema': 'scripts.train.iris_schema',
    'scripts.runtime.mvp_static': 'scripts.train.iris_validation',
    'scripts.runtime.audio': 'scripts.train.iris_audio',
    'scripts.runtime.iris_owner': 'scripts.train.iris_owner_reference',
    'scripts.runtime.speaker_session': 'scripts.train.iris_session',
    'scripts.runtime.mvp_wake': 'scripts.train.iris_wake_reference',
    'scripts.runtime.iris_openwakeword': 'scripts.train.iris_openwakeword_reference',
    'scripts.runtime.pcv_benchmark_adapter': 'scripts.train.iris_benchmark_serializer',
}
CONSUMERS = (
    'scripts/train/iris_diversity.py',
    'scripts/train/iris_owner.py',
    'scripts/train/iris_wake_helpers.py',
    'scripts/test/iris_benchmark.py',
)


@lru_cache(maxsize=None)
def baseline_source(path):
    return subprocess.check_output(
        ['git', 'show', BASELINE + ':' + path], cwd=ROOT,
    ).decode('utf-8')


def module_name(path):
    return path.removesuffix('.py').replace('/', '.')


def load_reference(path):
    assert (ROOT / path).is_file(), 'Missing training reference: ' + path
    return importlib.import_module(module_name(path))


def semantic_body(tree):
    return [node for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
            and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                     and isinstance(node.value.value, str))]


def definition_name(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def projected_baseline(path, selected):
    tree = ast.parse(baseline_source(path))
    tree.body = semantic_body(tree)
    if selected is not None:
        tree.body = [node for node in tree.body if definition_name(node) in selected]
    if path == 'scripts/runtime/audio.py':
        tree.body = [node for node in tree.body if definition_name(node) != 'CaptureResampler']
    if path == 'scripts/runtime/mvp_static.py':
        # This is the one deployment-only method, not part of forward/export
        # validation. Its wake/VAD/capture closure must not enter the producer.
        runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Runtime')
        runtime.body = [node for node in runtime.body if definition_name(node) != 'make_gate']
    return tree


@pytest.mark.parametrize('new,old,selected', EXTRACTIONS, ids=[Path(row[0]).stem for row in EXTRACTIONS])
def test_reference_ast_preserved_against_frozen_git_blob(new, old, selected):
    assert (ROOT / new).is_file(), 'Missing training reference: ' + new
    actual = ast.parse((ROOT / new).read_text(encoding='utf-8'))
    actual.body = semantic_body(actual)
    expected = RelocateImportsAndSourcePaths().visit(projected_baseline(old, selected))
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


@pytest.mark.parametrize('path', [row[0] for row in EXTRACTIONS])
def test_reference_imports_do_not_reach_deployment(path):
    assert (ROOT / path).is_file(), 'Missing training reference: ' + path
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                 else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        assert not any(name == 'scripts.runtime' or name.startswith(('scripts.runtime.', 'SpeakerUI'))
                       or name in {'sounddevice', 'pyaudio'} for name in names), (path, names)
    assert not any(isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)
                   for node in tree.body), 'No service/capture/bootstrap CLI in reference modules'


def test_owner_reference_binds_training_audio_and_session_types():
    owner = load_reference('scripts/train/iris_owner_reference.py')
    audio = load_reference('scripts/train/iris_audio.py')
    session = load_reference('scripts/train/iris_session.py')
    assert owner.DEFAULT is audio.DEFAULT
    assert owner.features is audio.features
    assert owner.SpeakerEvidence is session.SpeakerEvidence
    assert owner.SpeakerPolicy is session.SpeakerPolicy


def test_python_bootstrap_excludes_node_and_ui():
    tools = load_reference('scripts/training_tools.py')
    assert not any(hasattr(tools, name) for name in ('NODE_VERSION', 'PNPM_VERSION', 'node_tools', 'prepare_ui'))
    text = (ROOT / 'scripts/training_tools.py').read_text(encoding='utf-8')
    assert 'nodejs.org' not in text and 'SpeakerUI' not in text and 'pnpm' not in text
    assert not hasattr(load_reference('scripts/train/iris_audio.py'), 'CaptureResampler')
    assert not hasattr(load_reference('scripts/train/iris_validation.py').Runtime, 'make_gate')


class RelocateImportsAndSourcePaths(ast.NodeTransformer):
    """Only the explicitly enumerated engineering relocation is permitted."""

    def visit_ImportFrom(self, node):
        if node.module is not None:
            node.module = RELOCATIONS.get(node.module, node.module)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            paths = {old.replace('.', '/') + '.py': new.replace('.', '/') + '.py'
                     for old, new in RELOCATIONS.items()}
            paths['scripts/runtime'] = 'scripts/train'
            node.value = paths.get(node.value, node.value)
        return node


@pytest.mark.parametrize('path', CONSUMERS)
def test_training_consumers_only_relocate_imports_and_source_bindings(path):
    expected = RelocateImportsAndSourcePaths().visit(ast.parse(baseline_source(path)))
    actual = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)
    old_modules = set(RELOCATIONS)
    old_paths = {name.replace('.', '/') + '.py' for name in old_modules}
    for node in ast.walk(actual):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in old_modules
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in old_paths, 'Dangling source-binding path: ' + node.value


@pytest.fixture(scope='module')
def frozen_modules():
    """Execute the original Git bytes in a temporary, import-only namespace.

    Internal imports are rebound to other original Git modules. No current
    runtime source is the oracle and no scientific expression is rewritten.
    """
    package_name = '_iris_reference_baseline'
    package = ModuleType(package_name)
    package.__path__ = []
    installed = {package_name: package}
    originals = {old.removesuffix('.py').replace('/', '.'): package_name + '.' + Path(new).stem
                 for new, old, _ in EXTRACTIONS}

    class OriginalImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.level and node.module in {'audio', 'speaker_session'}:
                node.level = 0
                node.module = originals['scripts.runtime.' + node.module]
            else:
                if node.module is not None:
                    node.module = originals.get(node.module, node.module)
            return node

    previous = {package_name: sys.modules.get(package_name)}
    sys.modules[package_name] = package
    try:
        # Owner's relative imports and validation's schema must resolve to the
        # independent baseline, not to the newly extracted implementations.
        for index in (0, 2, 4, 1, 3, 5, 6, 7, 8):
            new, old, _ = EXTRACTIONS[index]
            name = originals[module_name(old)]
            module = ModuleType(name)
            module.__file__ = BASELINE + ':' + old
            module.__package__ = package_name
            previous[name] = sys.modules.get(name)
            installed[name] = module
            sys.modules[name] = module
            tree = OriginalImports().visit(ast.parse(baseline_source(old)))
            exec(compile(tree, module.__file__, 'exec'), module.__dict__)
        yield {Path(new).stem: installed[originals[module_name(old)]] for new, old, _ in EXTRACTIONS}
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_official_serialization_preserves_original_class_and_log_records(frozen_modules):
    before = frozen_modules['iris_benchmark_serializer']
    after = load_reference('scripts/train/iris_benchmark_serializer.py')
    commands = [None] + [row['command'] for row in frozen_modules['iris_schema'].baseline_cases()]
    for command in commands:
        assert before.class_prediction(command) == after.class_prediction(command)
        event = {'output': {'accepted': command is not None, 'command': command},
                 'timing': {'total_ms': 1.25}, 'audio_seconds': .5}
        assert before.log_record(event) == after.log_record(event)
    for invalid in (-1, float('nan'), float('inf'), True, None):
        event = {'output': {'accepted': False, 'command': None},
                 'timing': {'total_ms': invalid}, 'audio_seconds': .5}
        for module in (before, after):
            with pytest.raises(ValueError):
                module.log_record(event)


def test_schema_catalog_and_rejection_math_match_git_baseline(frozen_modules):
    before = frozen_modules['iris_schema']
    after = load_reference('scripts/train/iris_schema.py')
    assert before.baseline_cases() == after.baseline_cases()
    assert before.extended_cases() == after.extended_cases()
    assert before.DEFAULT_CUSTOM == after.DEFAULT_CUSTOM and before.INTENTS == after.INTENTS
    for value in range(101):
        assert before.number_words(value) == after.number_words(value)
    for row in before.baseline_cases() + before.extended_cases():
        assert before.canonical(row) == after.canonical(row)
        assert before.normalize_text(row['text']) == after.normalize_text(row['text'])
        reason = row.get('reason') if row['command'] is None else None
        assert before.output(row['command'], reason) == after.output(row['command'], reason)
    for bad in (-1, 101, True, 1.5):
        for module in (before, after):
            with pytest.raises(ValueError):
                module.number_words(bad)
    for intent in ('CALL', 'LIGHT_ON'):
        for name in (None, 'Home' if intent == 'CALL' else 'kitchen lamp'):
            assert before.default_slots(intent, name) == after.default_slots(intent, name)
    for module in (before, after):
        with pytest.raises(ValueError):
            module.command('TIMER', duration_seconds=True)
        with pytest.raises(ValueError):
            module.default_slots('CALL', 'not a known contact')


@pytest.mark.parametrize('strategy', ['coarse', 'observed'])
@pytest.mark.parametrize('grouping', ['intent', 'class'])
def test_policy_fit_and_decisions_match_git_baseline(frozen_modules, strategy, grouping):
    before = frozen_modules['iris_validation']
    after = load_reference('scripts/train/iris_validation.py')
    entries = frozen_modules['iris_schema'].baseline_cases()
    catalog = before.build_catalog(entries)
    assert catalog == after.build_catalog(entries)
    size = len(catalog)
    rng = np.random.default_rng(94)
    rows = []
    for index in range(24):
        logits = rng.normal(size=size)
        logits[index % size] += 4
        rows.append(dict(logits=logits.tolist(), expected_index=index % size,
                         split='dev' if index % 2 else 'calibration', synthetic=bool(index % 3)))
    rows += [dict(logits=row['logits'], expected_index=0, split='dev', synthetic=False)
             for row in rows[:2]]
    trained = list(range(size))
    for group_catalog in (None, catalog):
        kwargs = dict(catalog=group_catalog, risk={'human': .03, 'synthetic': .01},
                      threshold_strategy=strategy, grouping=grouping, min_correct=2)
        original = before.fit_policy(rows, trained, **kwargs)
        relocated = after.fit_policy(rows, trained, **kwargs)
        assert original == relocated
        for row in rows:
            np.testing.assert_array_equal(before.probabilities(row['logits']), after.probabilities(row['logits']))
            assert before.decide(row['logits'], catalog, original) == after.decide(row['logits'], catalog, relocated)
    assert before.fit_policy([], trained) == after.fit_policy([], trained)
    for module in (before, after):
        with pytest.raises(ValueError, match='excludes'):
            module.fit_policy([{**rows[0], 'split': 'test'}], trained)
        with pytest.raises(ValueError, match='Ambiguous'):
            module.fit_policy([{**rows[0], 'expected_index': None}], trained)
        for logits in ([float('nan'), 1], [1], [[1, 2]]):
            with pytest.raises(ValueError):
                module.probabilities(logits)


@pytest.mark.parametrize('samples', [0, 37, 399, 400, 560, 16000])
def test_audio_feature_arrays_and_streaming_match_git_baseline(frozen_modules, samples):
    before = frozen_modules['iris_audio']
    after = load_reference('scripts/train/iris_audio.py')
    pcm = (.05 * np.sin(np.arange(samples) * .073)).astype('float32')
    assert asdict(before.DEFAULT) == asdict(after.DEFAULT)
    np.testing.assert_array_equal(before.mel_filters(), after.mel_filters())
    np.testing.assert_array_equal(before.features(pcm), after.features(pcm))
    np.testing.assert_array_equal(before.resample_audio(pcm, 16000), after.resample_audio(pcm, 16000))
    streams = (before.StreamingFrontend(), after.StreamingFrontend())
    for start in range(0, len(pcm), 317):
        outputs = [stream.push(pcm[start:start + 317]) for stream in streams]
        np.testing.assert_array_equal(*outputs)
        np.testing.assert_array_equal(streams[0].pending, streams[1].pending)


@pytest.mark.parametrize('rate', [32000, 44100, 48000])
def test_optional_non_native_resampling_matches_git_baseline(frozen_modules, rate):
    pytest.importorskip('scipy.signal', reason='Approved producer interpreter has no SciPy; no dependency installs allowed')
    before = frozen_modules['iris_audio']
    after = load_reference('scripts/train/iris_audio.py')
    pcm = (.05 * np.sin(np.arange(16000) * .073)).astype('float32')
    np.testing.assert_array_equal(before.resample_audio(pcm, rate), after.resample_audio(pcm, rate))


def test_audio_loading_and_feature_cache_match_git_baseline(frozen_modules, tmp_path, monkeypatch):
    before = frozen_modules['iris_audio']
    after = load_reference('scripts/train/iris_audio.py')
    pcm = (.03 * np.sin(np.arange(2000) * .091)).astype('float32')
    calls = []

    def read_fixture(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        return np.stack((pcm, -pcm), axis=1), 16000

    # SoundFile is an I/O seam only; the frozen and extracted math is real.
    monkeypatch.setattr(before.sf, 'read', read_fixture)
    assert before.sf is after.sf
    for channel in (0, 1):
        original, relocated = before.load_audio('fixture.wav', channel), after.load_audio('fixture.wav', channel)
        np.testing.assert_array_equal(original[0], relocated[0])
        assert original[1] == relocated[1]
    for module in (before, after):
        with pytest.raises(ValueError, match='Channel'):
            module.load_audio('fixture.wav', 2)
    record = {'id': 'bounded-feature-fixture', 'audio': 'fixture.wav', 'augmentation': {'gain': 0}}
    assert before.cache_path(record, tmp_path) == after.cache_path(record, tmp_path)
    original = before.cached_features(record, tmp_path)
    reads = len(calls)
    relocated = after.cached_features(record, tmp_path)
    np.testing.assert_array_equal(original, relocated)
    assert len(calls) == reads, 'The extracted helper must use the same existing feature-cache identity'


def test_python_bootstrap_cached_and_missing_paths_match_git_baseline(frozen_modules, tmp_path, monkeypatch):
    before = frozen_modules['training_tools']
    after = load_reference('scripts/training_tools.py')

    def forbidden(*args, **kwargs):
        pytest.fail('No download, environment synchronization or process is authorized')

    for module in (before, after):
        monkeypatch.setattr(module, 'run', forbidden)
        monkeypatch.setattr(module.urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(before.shutil, 'which', lambda name: 'existing-uv' if name == 'uv' else None)
    assert before.find_uv(tmp_path) == after.find_uv(tmp_path) == 'existing-uv'
    assert not (tmp_path / '.runtime').exists()
    assert before.python_path(tmp_path) == after.python_path(tmp_path)
    monkeypatch.setattr(before.shutil, 'which', lambda _: None)
    local = tmp_path / '.runtime/tools/uv' / ('uv.exe' if before.os.name == 'nt' else 'uv')
    local.parent.mkdir(parents=True)
    local.write_bytes(b'fixture for cached path lookup only; never executed')
    assert before.find_uv(tmp_path) == after.find_uv(tmp_path) == str(local)
    for source in (False, True):
        for module in (before, after):
            with pytest.raises(RuntimeError, match='pyproject.toml / uv.lock'):
                module.sync_python(tmp_path, source=source)


@pytest.mark.parametrize('samples', [0, 200, 400, 8000, 24000, 64000])
def test_owner_energy_features_and_coverage_match_git_baseline(frozen_modules, samples):
    before = frozen_modules['iris_owner_reference']
    after = load_reference('scripts/train/iris_owner_reference.py')
    pcm = (.03 * np.sin(np.arange(samples) * .091)).astype('float32')
    assert before.PCM_NORMALIZATION == after.PCM_NORMALIZATION
    for waveform in (pcm, np.zeros_like(pcm), np.pad(pcm, (400, 1700))):
        np.testing.assert_array_equal(before.speech_mask(waveform), after.speech_mask(waveform))
        np.testing.assert_array_equal(before.owner_features(waveform), after.owner_features(waveform))
        assert before.voiced_seconds(waveform) == after.voiced_seconds(waveform)
        assert before.continuity_spans(waveform) == after.continuity_spans(waveform)
    for module in (before, after):
        with pytest.raises(ValueError):
            module.owner_features([float('nan')])


@pytest.mark.parametrize('case', ['same', 'impostor', 'overlap', 'short', 'unchecked', 'timeout', 'model', 'shape', 'token', 'clock', 'no_windows'])
def test_session_acceptance_rejection_and_reset_match_git_baseline(frozen_modules, case):
    outcomes = []
    for module in (frozen_modules['iris_session'], load_reference('scripts/train/iris_session.py')):
        fields = dict(embedding=(1., 0.), model_id='a' * 64, voiced_seconds=1.2, overlap=False)
        reference = module.SpeakerEvidence(**fields)
        lock = module.SpeakerSession(module.SpeakerPolicy(threshold=.8))
        arm = lock.arm('reference', reference, now=2.)
        changes = {'impostor': {'embedding': (0., 1.)}, 'overlap': {'overlap': True},
                   'short': {'voiced_seconds': .1}, 'unchecked': {'overlap': None},
                   'model': {'model_id': 'b' * 64}, 'shape': {'embedding': (1., 0., 0.)}}.get(case, {})
        evidence = module.SpeakerEvidence(**{**fields, **changes})
        verdict = lock.finish('different' if case == 'token' else 'reference', evidence,
                              now=10. if case == 'timeout' else 1. if case == 'clock' else 3.,
                              windows=None if case == 'no_windows' else [reference])
        consumed = lock.finish('reference', reference, now=4.)
        application = {'accepted': True, 'command': {'intent': 'PAUSE', 'slots': {}}, 'reason': None}
        outcomes.append((asdict(arm), asdict(verdict), asdict(consumed), module.gate_application(application, verdict)))
    assert outcomes[0] == outcomes[1]


def test_rolling_wake_chunking_hysteresis_and_reset_match_git_baseline(frozen_modules):
    scores = [.95, .95, .96, .96] + [.1] * 20 + [.95, .95, .1, .95, .95]
    for chunk in (317, 1280, 7013):
        traces = []
        for module in (frozen_modules['iris_wake_reference'], load_reference('scripts/train/iris_wake_reference.py')):
            values = iter(scores)
            wake = module.RollingWake(lambda _: next(values), threshold=.8)
            pcm = np.zeros(24320 + 1280 * len(scores), dtype='float32')
            outputs = [wake.push(pcm[start:start + chunk]) for start in range(0, len(pcm), chunk)]
            traces.append((outputs, wake.hits, wake.latched, wake.cooldown, wake.audio.copy(), wake.pending.copy()))
            wake.reset()
            assert wake.hits == 0 and not wake.latched and wake.cooldown == 0
            assert not len(wake.audio) and not len(wake.pending)
            with pytest.raises(ValueError):
                wake.push([float('nan')])
        assert traces[0][:4] == traces[1][:4]
        np.testing.assert_array_equal(traces[0][4], traces[1][4])
        np.testing.assert_array_equal(traces[0][5], traces[1][5])


def test_real_frozen_command_graph_forward_matches_git_baseline(frozen_modules):
    before = frozen_modules['iris_validation'].Runtime(ROOT / 'artifacts/iris')
    after = load_reference('scripts/train/iris_validation.py').Runtime(ROOT / 'artifacts/iris')
    assert before.catalog == after.catalog and before.policy == after.policy and before.norm == after.norm
    assert before.recipe == after.recipe
    for pcm in (np.zeros(16000, dtype='float32'),
                (.03 * np.sin(np.arange(16000) * .071)).astype('float32'),
                (.03 * np.cos(np.arange(33777) * .053)).astype('float32')):
        original, relocated = before.recognize(pcm), after.recognize(pcm)
        # Wall-clock timing is deliberately not an equivalence claim.
        for result in (original, relocated):
            result.pop('timing')
            result.pop('frontend_ms', None)
        assert original == relocated
    before.mode = after.mode = 'qualified'
    pcm = np.full(1600, .01, dtype='float32')
    assert before.recognize(pcm)['output'] == after.recognize(pcm)['output']


def test_real_frozen_owner_graph_forward_matches_git_baseline(frozen_modules):
    before = frozen_modules['iris_owner_reference'].OwnerBackend(ROOT / 'artifacts/iris/owner')
    after = load_reference('scripts/train/iris_owner_reference.py').OwnerBackend(ROOT / 'artifacts/iris/owner')
    assert before.model_id == after.model_id and asdict(before.policy) == asdict(after.policy)
    pcm = (.03 * np.sin(np.arange(30000) * .067)).astype('float32')
    for waveform in (pcm, np.zeros_like(pcm)):
        original, relocated = before.predict(waveform), after.predict(waveform)
        np.testing.assert_array_equal(original[0], relocated[0])
        assert original[1] == relocated[1]
        evidence = (before.evidence(waveform), after.evidence(waveform))
        np.testing.assert_array_equal(evidence[0].embedding, evidence[1].embedding)
        for field in ('model_id', 'voiced_seconds', 'overlap', 'clipped'):
            assert getattr(evidence[0], field) == getattr(evidence[1], field)
        windows = (before.continuity(waveform), after.continuity(waveform))
        if windows[0] is None:
            assert windows[1] is None
        else:
            assert len(windows[0]) == len(windows[1])
            for original_window, relocated_window in zip(*windows):
                np.testing.assert_array_equal(original_window.embedding, relocated_window.embedding)
                assert original_window.voiced_seconds == relocated_window.voiced_seconds


def test_real_frozen_wake_graph_forward_matches_git_baseline(frozen_modules):
    before = frozen_modules['iris_wake_reference'].HeyIris(ROOT / 'artifacts/iris/wake')
    after = load_reference('scripts/train/iris_wake_reference.py').HeyIris(ROOT / 'artifacts/iris/wake')
    pcm = (.03 * np.sin(np.arange(32000) * .077)).astype('float32')
    assert before.meta == after.meta and before.norm == after.norm
    assert before.score(pcm[-25600:]) == after.score(pcm[-25600:])
    for chunk in (640, 317):
        before.reset()
        after.reset()
        for start in range(0, len(pcm), chunk):
            assert before.push(pcm[start:start + chunk]) == after.push(pcm[start:start + chunk])
            assert (before.hits, before.latched, before.cooldown) == (after.hits, after.latched, after.cooldown)
            np.testing.assert_array_equal(before.audio, after.audio)
            np.testing.assert_array_equal(before.pending, after.pending)


def test_openwakeword_identity_and_pcm_rejections_match_git_baseline(frozen_modules, tmp_path, monkeypatch):
    import onnxruntime as ort
    before = frozen_modules['iris_openwakeword_reference']
    after = load_reference('scripts/train/iris_openwakeword_reference.py')
    assert before.FILES == after.FILES
    monkeypatch.setattr(ort, 'InferenceSession', lambda *args, **kwargs: pytest.fail('Invalid identity reached ONNX'))
    for module in (before, after):
        with pytest.raises(ValueError, match='identity'):
            module.HeyIrisOpenWakeWord(tmp_path)
        # Invalid PCM is rejected before consulting graph state.
        detector = module.HeyIrisOpenWakeWord.__new__(module.HeyIrisOpenWakeWord)
        for invalid in (np.zeros((2, 3)), np.array([np.nan]), np.array([np.inf]),
                        np.array([1.01]), np.zeros(16001), np.zeros(3, np.int16)):
            with pytest.raises(ValueError, match='PCM'):
                detector.push(invalid)
            assert not detector.__dict__


def test_optional_real_openwakeword_graphs_match_git_baseline(frozen_modules):
    assets = ROOT / '.cache/iris-openwakeword-publication-v1/models'
    if not assets.is_dir():
        pytest.skip('Optional public openWakeWord graphs absent; no downloads allowed')
    before = frozen_modules['iris_openwakeword_reference'].HeyIrisOpenWakeWord(assets)
    after = load_reference('scripts/train/iris_openwakeword_reference.py').HeyIrisOpenWakeWord(assets)
    pcm = (.1 * np.sin(np.arange(32000) * .077)).astype('float32')
    for start in range(0, len(pcm), 731):
        assert before.push(pcm[start:start + 731]) == after.push(pcm[start:start + 731])
        assert before.last_scores == after.last_scores
        for field in ('mels', 'embeddings', 'raw', 'pending'):
            np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    before.reset()
    after.reset()
    np.testing.assert_array_equal(before.embeddings, after.embeddings)


def test_reference_and_consumer_imports_work_with_deployment_blocked():
    names = [module_name(row[0]) for row in EXTRACTIONS] + [module_name(path) for path in CONSUMERS]
    script = dedent('''
        import importlib, importlib.abc, json, socket, subprocess, sys
        # Resolve third-party initialization before installing action spies.
        import numpy, soundfile, einops, onnxruntime
        def forbidden(*args, **kwargs):
            raise AssertionError('No setup/network/process/capture action during reference imports')
        class Boundary(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'scripts.runtime' or fullname.startswith(('scripts.runtime.', 'SpeakerUI')) or fullname in {'sounddevice', 'pyaudio'}:
                    raise AssertionError('Deployment import: ' + fullname)
        sys.meta_path.insert(0, Boundary())
        socket.create_connection = forbidden
        socket.socket.connect = forbidden
        subprocess.run = subprocess.Popen = subprocess.check_call = forbidden
        import urllib.request
        urllib.request.urlopen = forbidden
        modules = {name: importlib.import_module(name) for name in json.loads(sys.argv[1])}
        schema = modules['scripts.train.iris_schema']
        validation = modules['scripts.train.iris_validation']
        diversity = modules['scripts.train.iris_diversity']
        catalog = validation.build_catalog(schema.baseline_cases())
        for row in catalog:
            row['training_recordings'] = 1
        logits = [0.] * len(catalog)
        logits[1] = 8.
        records = [dict(id=str(i), split='dev', synthetic=False, expected_index=1, logits=logits) for i in range(2)]
        policy = diversity._policy(records, catalog, {'demo_risk': {'human': .03, 'synthetic': .01}})
        assert diversity._apply(records, catalog, policy)[0]['accepted_index'] == 1
        assert diversity.generation_plan(catalog)
        wake_metrics = modules['scripts.train.iris_wake_helpers'].stream_metrics(
            [dict(positive=True, seconds=1., scores=[.95, .95])], .8)
        assert wake_metrics['detected'] == 1
        pcm = numpy.full(16000, .01, dtype='float32')
        assert validation.Runtime('artifacts/iris').recognize(pcm)['event'] == 'command'
        owner = modules['scripts.train.iris_owner_reference'].OwnerBackend('artifacts/iris/owner')
        assert owner.predict(pcm)[0].shape == (128,)
        wake = modules['scripts.train.iris_wake_reference'].HeyIris('artifacts/iris/wake')
        wake.push(numpy.zeros(32000, dtype='float32'))
        assert not any(name == 'scripts.runtime' or name.startswith('scripts.runtime.') for name in sys.modules)
        print('training reference imports isolated from deployment and actions')
    ''')
    result = subprocess.run([sys.executable, '-B', '-c', script, json.dumps(names)],
                            cwd=ROOT, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'isolated from deployment and actions' in result.stdout
