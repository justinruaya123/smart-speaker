"""Producer-owned model-only ZIPs. No deployment imports or export conversion."""
from pathlib import Path
import hashlib
import json
import math
import re

from scripts.archive_io import safe_name, write_archive
from scripts.optionb_epoch44 import ROOT, SNAPSHOT, SNAPSHOT_SHA256, SOURCE_REVISION

FROZEN_NAMES = {'commands.json', 'model.onnx', 'normalization.json', 'policy.json', 'recipe.json'}


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            nonfinite(value)
        return number
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_float)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _read(root, name):
    """Read regular contained files, never symlinks/junctions or normalized names."""
    safe_name(name)
    root = Path(root).absolute()
    path = root / name
    for part in (path, *path.parents):
        is_junction = getattr(part, 'is_junction', lambda: False)
        if part.is_symlink() or is_junction():
            raise ValueError('Linked model source: ' + str(part))
        if part == root:
            break
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Missing or unsafe model source: ' + name)
    return path.read_bytes()


def build_archive(bundle, output, *, source_revision, companions=None):
    """Package only the original five-file FP32 epoch44 export, byte for byte.

    source_revision declares the producer checkout (full Git object ID); the
    independently pinned snapshot binds the original model provenance.
    """
    if not isinstance(source_revision, str) or not re.fullmatch('[0-9a-f]{40}', source_revision):
        raise ValueError('source_revision must be a full lowercase Git revision')
    snapshot_bytes = _read(ROOT, SNAPSHOT)
    if _hash(snapshot_bytes) != SNAPSHOT_SHA256:
        raise ValueError('Frozen snapshot identity changed')
    snapshot = _json(snapshot_bytes)
    bundle = Path(bundle)
    if {p.name for p in bundle.iterdir()} != FROZEN_NAMES:
        raise ValueError('Frozen bundle requires the exact five-file inventory')
    entries = {}
    for name in sorted(FROZEN_NAMES):
        data = _read(bundle, name)
        expected = snapshot['files'][snapshot['bundle'] + '/' + name]
        if len(data) != expected['bytes'] or _hash(data) != expected['sha256']:
            raise ValueError('Frozen payload identity mismatch: ' + name)
        entries['model/' + name] = data
    contract = _json(entries['model/recipe.json'])
    if companions is None:
        companions = {}
    if (not isinstance(companions, dict) or set(companions) - {'wake', 'owner', 'vad'}
            or any(value is None for value in companions.values())):
        raise ValueError('companions must map wake/owner/vad roles to existing export folders')
    extra, source = _companions(companions.get('wake'), companions.get('owner'), companions.get('vad'),
                                prefix='model/')
    entries.update(extra)
    metadata = {'format': 'smart-speaker-model-v1', 'source_revision': source_revision,
                'original_source_revision': SOURCE_REVISION, 'snapshot_sha256': SNAPSHOT_SHA256,
                'source_bundle': snapshot['bundle'], 'contract': contract, 'companion_source': source,
                'qualified': False, 'actions_enabled': False}
    receipt = write_archive(entries, output, metadata)
    return {**receipt, 'qualified': False, 'files': len(entries)}


def _export(folder, metadata_name, required, prefix):
    """Exact exported closure: metadata cannot request additional payloads."""
    raw = _read(folder, metadata_name)
    meta = _json(raw)
    if not isinstance(meta, dict) or set(meta.get('files', {})) != set(required):
        raise ValueError('Unexpected export inventory: ' + metadata_name)
    entries = {prefix + metadata_name: raw}
    for name in sorted(required):
        expected = meta['files'][name]
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid export SHA-256: ' + name)
        data = _read(folder, name)
        if _hash(data) != expected:
            raise ValueError('Export identity mismatch: ' + name)
        if name.endswith('.json'):
            _json(data)
        entries[prefix + name] = data
    return entries, meta


def _onnx(data):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])


def _companions(wake_folder=None, owner_folder=None, vad_folder=None, *, prefix):
    """Verify companions without importing a deployed capture/session backend."""
    entries, bindings = {}, {}
    if wake_folder is not None:
        from scripts.train.iris_wake_reference import RollingWake
        from vcm.frontend import StreamingFrontend
        wake, meta = _export(wake_folder, 'iris.json',
                             {'iris.onnx', 'iris.ncnn.param', 'iris.ncnn.bin'}, prefix + 'wake/')
        if meta.get('qualified') is not False:
            raise ValueError('Wake qualification must remain false')
        norm = meta['normalization']
        StreamingFrontend(norm['mean'], norm['std'])
        RollingWake(lambda _: 0., threshold=meta['threshold'], release=meta['release'],
                    consecutive=meta['consecutive'], cooldown_samples=meta['cooldown_samples'])
        _onnx(wake[prefix + 'wake/iris.onnx'])
        entries.update(wake)
        bindings['wake_metadata_sha256'] = _hash(wake[prefix + 'wake/iris.json'])
    if owner_folder is not None:
        from scripts.train.iris_owner_reference import OwnerBackend
        owner, meta = _export(owner_folder, 'owner.json', {'owner.onnx'}, prefix + 'owner/')
        model = meta['model']
        data = owner[prefix + 'owner/owner.onnx']
        if (model.get('path') != 'owner.onnx' or model.get('sha256') != meta['files']['owner.onnx']
                or model.get('bytes') != len(data)):
            raise ValueError('Owner model binding mismatch')
        OwnerBackend(owner_folder, threads=1)  # Original frontend, policy, provenance and graph checks.
        entries.update(owner)
        bindings['owner_metadata_sha256'] = _hash(owner[prefix + 'owner/owner.json'])
    if vad_folder is not None:
        raw = _read(ROOT / 'artifacts', 'wake-models.json')
        meta = _json(raw)
        if set(meta.get('files', {})) != {'silero-vad.onnx'}:
            raise ValueError('Unexpected VAD inventory')
        record = meta['files']['silero-vad.onnx']
        data = _read(vad_folder, 'silero-vad.onnx')
        if _hash(data) != record['sha256'] or len(data) != record['bytes']:
            raise ValueError('VAD source identity mismatch')
        _onnx(data)
        entries[prefix + 'wake/silero-vad.onnx'] = data
        # This derived metadata is the historical reproduction layout, not an
        # ONNX conversion or replacement for any source export identity.
        entries[prefix + 'wake/wake-models.json'] = (
            json.dumps({'files': {'silero-vad.onnx': record}}, indent=2) + '\n').encode()
        entries['licenses/silero-LICENSE.txt'] = _read(ROOT / 'artifacts', 'silero-LICENSE.txt')
        bindings['vad_source_metadata_sha256'] = _hash(raw)
    return entries, bindings


def _extras(extra_files):
    """Accept model evidence/notices, not an open-ended application payload."""
    entries, excluded = {}, []
    for name, data in (extra_files or {}).items():
        safe_name(name)
        if name == 'docs/HANDOFF-iris-rpi-v1.md':
            excluded.append(name)  # Existing caller's application-only operator manual.
            continue
        allowed = (name in {'MODEL_CARD.md', 'docs/DATASETS.md', 'NOTICES.txt'}
                   or re.fullmatch(r'docs/EXPERIMENT-[a-z0-9-]+\.md', name)
                   or re.fullmatch(r'evidence/[a-zA-Z0-9_-]+\.json', name))
        if not allowed or not isinstance(data, bytes):
            raise ValueError('Not model evidence or a notice: ' + name)
        if name.startswith('evidence/'):
            _json(data)
        entries[name] = data
    return entries, sorted(excluded)


def package_release(bundle, output, *, wake_folder, owner_folder, vad_folder=None, extra_files=None):
    """Legacy release call shape; only exported models, notices and evidence ship.

    NCNN payloads require their exact hash inventory and passed ONNX-bound parity
    receipt. No conversion, runtime loader, audio corpus or application is used.
    Graph loading here is CPU ONNX validation, not fresh NCNN/acoustic parity.
    """
    from scripts.train.iris_validation import Runtime as ExportReference
    if wake_folder is None or owner_folder is None:
        raise ValueError('A locked wake and independent owner export are required')
    prefix = 'models/default/'
    entries, meta = _export(bundle, 'bundle.json',
                            {'model.onnx', 'catalog.json', 'normalization.json', 'policy.json'}, prefix)
    if meta.get('format') != 'mvp-static-v1' or meta.get('qualified') is not False:
        raise ValueError('Unsupported or qualified command bundle')
    ncnn, conversion = _export(Path(bundle) / 'ncnn', 'engine.json',
                               {'model.ncnn.param', 'model.ncnn.bin', 'position-frequency.json'},
                               prefix + 'ncnn/')
    if (conversion.get('onnx_sha256') != meta['files']['model.onnx']
            or conversion.get('parity', {}).get('passed') is not True):
        raise ValueError('Unverified ncnn conversion')
    parity = conversion['parity']
    if (any(type(parity.get(key)) not in (int, float) or not math.isfinite(parity[key])
            for key in ('max_absolute_error', 'tolerance'))
            or not 0 <= parity['max_absolute_error'] <= parity['tolerance']
            or parity['tolerance'] <= 0):
        raise ValueError('Contradictory ncnn parity receipt')
    # Preserve the real semantic/ONNX checks formerly performed by deployment.
    reference = ExportReference(bundle, engine='onnxruntime')
    entries.update(ncnn)
    companions, source = _companions(wake_folder, owner_folder,
                                     vad_folder if vad_folder is not None else ROOT / 'artifacts',
                                     prefix=prefix)
    entries.update(companions)
    extras, excluded = _extras(extra_files)
    entries.update(extras)
    release = _json(extras.get('evidence/reproduction-release.json', b'{}'))
    revision = release.get('source_version', {}).get('commit')
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision)):
        raise ValueError('Invalid release source revision')
    metadata = {'format': 'smart-speaker-model-v1', 'source_revision': revision,
                'source': {**source, 'bundle_metadata_sha256': _hash(entries[prefix + 'bundle.json'])},
                'contract': {'bundle_format': meta['format'], 'output_schema_version': 3,
                             'model_binding': reference.recipe['binding'],
                             'ncnn_parity_receipt_sha256': _hash(ncnn[prefix + 'ncnn/engine.json']),
                             'ncnn_native_load_checked': False},
                'qualified': False, 'actions_enabled': False}
    receipt = write_archive(entries, output, metadata)
    return {**receipt, 'qualified': False, 'actions_enabled': False, 'files': len(entries),
            'model_sha256': meta['files']['model.onnx'], 'parameters': meta['parameters'],
            'profile': 'model-only', 'excluded_application_files': excluded}
