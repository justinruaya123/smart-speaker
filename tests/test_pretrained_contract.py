"""Keep external pretrained systems confined to separate benchmarks."""
import ast

from test_vcm_only import ROOT, modules, publisher_module


def test_active_sources_do_not_fetch_pretrained_weights(modules):
    """Allow dependency installation, not external model initialization."""
    sources = dict(modules)
    for name in ("train.py", "publish.py"):
        sources[name] = (ROOT / name).read_text()
    forbidden_calls = {
        "from_pretrained", "load_state_dict_from_url", "hf_hub_download",
        "snapshot_download", "load_model", "WhisperModel",
    }
    forbidden_imports = {"transformers", "faster_whisper", "whisper"}
    for module, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in forbidden_imports
                               for alias in node.names), module
            if isinstance(node, ast.ImportFrom):
                package = (node.module or "").split(".")[0]
                assert package not in forbidden_imports, module
            if isinstance(node, ast.Call):
                call = node.func
                name = call.attr if isinstance(call, ast.Attribute) else (
                    call.id if isinstance(call, ast.Name) else "")
                assert name not in forbidden_calls, (module, name)
                assert ast.unparse(call) != "torch.hub.load", module


def test_publisher_remains_one_acoustic_graph():
    publisher = publisher_module()
    graphs = [name for name in publisher.MODEL_FILES if name.endswith(".onnx")]
    assert graphs == ["vcm.fp32.onnx"]
    assert not any("benchmark" in name for name in publisher.MODEL_FILES)
