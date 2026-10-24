"""Ownership of source inputs for NEW model-only engineering receipts.

Historical protocols/input manifests stay byte-for-byte unchanged. Exact old
application-containing releases are reconstructed at their immutable Git source,
not silently reclassified as model-only releases at the new head.
"""
from collections.abc import Iterable

REFERENCE_RELOCATIONS = {
    'scripts/runtime/pcv_schema.py': 'scripts/train/iris_schema.py',
    'scripts/runtime/mvp_static.py': 'scripts/train/iris_validation.py',
    'scripts/runtime/audio.py': 'scripts/train/iris_audio.py',
    'scripts/runtime/iris_owner.py': 'scripts/train/iris_owner_reference.py',
    'scripts/runtime/speaker_session.py': 'scripts/train/iris_session.py',
    'scripts/runtime/mvp_wake.py': 'scripts/train/iris_wake_reference.py',
    'scripts/runtime/iris_openwakeword.py': 'scripts/train/iris_openwakeword_reference.py',
    'scripts/runtime/speaker_tools.py': 'scripts/training_tools.py',
    'scripts/runtime/pcv_benchmark_adapter.py': 'scripts/train/iris_benchmark_serializer.py',
    'scripts/runtime/pyproject.toml': 'scripts/train/pyproject.toml',
    'scripts/runtime/uv.lock': 'scripts/train/uv.lock',
    'scripts/mvp_delivery.py': 'scripts/model_package.py',
    'scripts/speaker_publish.py': 'scripts/model_package.py',
}
APPLICATION_SOURCES = frozenset((
    'start.py', 'inference.py', 'benchmark.py', 'config.example.toml',
    'scripts/iris_inference.py', 'scripts/iris_epoch44_benchmark.py',
    'scripts/iris_final_package.py', 'scripts/build_iris_feedback.py',
    'scripts/runtime/__init__.py', 'scripts/runtime/config.example.toml',
    'scripts/runtime/iris_pi_benchmark.py', 'scripts/runtime/iris_epoch44.py',
    'scripts/runtime/iris_benchmark_assets.py', 'scripts/runtime/iris_feedback.py',
    'scripts/runtime/epoch44_setup.py', 'scripts/runtime/ncnn-LICENSE.txt',

    'scripts/runtime/speaker.env.example', 'scripts/runtime/speaker_config.py',
    'scripts/runtime/speaker_package.py', 'scripts/runtime/speaker_start.py',
    'scripts/runtime/speaker_worker.py',
    'scripts/runtime/wake_gate.py', 'scripts/runtime/wake_models.py',
    'scripts/test/iris_resources.py', 'scripts/test/iris_package_replay.py',
    'scripts/test/test_iris_configuration.py', 'scripts/test/test_iris_worker.py',
    'scripts/test/test_iris_pi_benchmark.py', 'scripts/test/test_iris_resources.py',
))


def model_source_files(names: Iterable[str]) -> list[str]:
    """Map declared ownership, never filter by existence or alter frozen input.

    Unknown/missing model sources are retained so freeze_release still rejects
    them. Only explicitly migrated application paths are omitted.
    """
    return sorted({REFERENCE_RELOCATIONS.get(name, name) for name in names
                   if not name.startswith('SpeakerUI/') and name not in APPLICATION_SOURCES
                   and not name.startswith('scripts/runtime/epoch44/')})
