"""Source-only platform gates must not become publication or training jobs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_native_arm64_workflow_exercises_public_source_commands_only():
    path = ROOT/'.github/workflows/iris-source-arm64.yml'
    assert path.is_file(), 'A native Linux ARM64 source-entrypoint check is required'
    text = path.read_text(encoding='utf8')
    assert 'runs-on: ubuntu-24.04-arm' in text
    assert 'python3 -B inference.py --check' in text
    assert 'python3 -B benchmark.py --output' in text
    assert 'contents: read' in text
    assert 'persist-credentials: false' in text
    assert 'train.py' not in text and '--publish' not in text and '--mic ' not in text
    assert 'pages: write' not in text and 'contents: write' not in text


def test_pi_quickstart_documents_shallow_clone_and_bounded_default_benchmark():
    from presentation.build_readme import PREFIX
    assert 'Raspberry Pi 4' in PREFIX and '64-bit' in PREFIX
    assert 'git clone --depth 1' in PREFIX
    assert 'python3 inference.py --check' in PREFIX
    assert 'python3 benchmark.py --output' in PREFIX
    assert '14.55 MB' in PREFIX and '196' in PREFIX
