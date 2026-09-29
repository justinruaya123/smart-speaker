import ast
import math
from pathlib import Path
import tomllib

def resolve_switches(values):
    """Interpret only boolean names, and/or/not and parentheses. Never eval()."""
    resolved, visiting = {}, set()
    def switch(name):
        if name in resolved:
            return resolved[name]
        if name not in values:
            raise ValueError(f"Unknown boolean switch: {name}")
        if name in visiting:
            raise ValueError(f"Cyclic boolean expression: {name}")
        visiting.add(name)
        value = values[name]
        if type(value) is bool:
            result = value
        elif isinstance(value, str):
            def walk(node):
                if isinstance(node, ast.Name):
                    if node.id.lower() in {"true", "false"}:
                        return node.id.lower() == "true"
                    return switch(node.id)
                if isinstance(node, ast.Constant) and type(node.value) is bool:
                    return node.value
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    return not walk(node.operand)
                if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                    results = [walk(item) for item in node.values]
                    return all(results) if isinstance(node.op, ast.And) else any(results)
                raise ValueError("Use only booleans, switch names, and/or/not and parentheses")
            result = walk(ast.parse(value, mode="eval").body)
        else:
            raise ValueError(f"Switch {name} must be a boolean or boolean expression")
        visiting.remove(name)
        resolved[name] = result
        return result
    for name in values:
        switch(name)
    return resolved

def load_config(path):
    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"workflow", "tests", "runtime", "validation", "training", "publish", "audio"}
    if set(config) != expected:
        raise ValueError(f"Configuration sections must be {sorted(expected)}")
    for section, keys in {"workflow": {"setup", "train", "test", "validate", "report"},
                          "tests": {"vad", "vcm", "pipeline"},
                          "runtime": {"vad", "vcm"}}.items():
        if set(config[section]) != keys:
            raise ValueError(f"Unknown or missing {section} switches")
        config[section] = resolve_switches(config[section])
    if any(k in config["training"] for k in ("asr", "asr_epochs")):
        raise ValueError("Transcription training switches have been removed")
    for key in ("vcm", "update_artifacts"):
        if type(config["training"][key]) is not bool:
            raise ValueError(f"training.{key} must be true or false")
    for section, key in (("validation", "require_quality_gate"), ("publish", "include_wheels")):
        if type(config[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be true or false")
    if config["validation"]["split"] not in {"dev", "calibration", "test"}:
        raise ValueError("Validation split must be dev, calibration or test")
    if type(config["validation"]["threads"]) is not int or not 1 <= config["validation"]["threads"] <= 4:
        raise ValueError("Use one to four inference threads")
    for section, keys in {"validation": ("max_examples",), "training": ("max_train_examples", "max_dev_examples")}.items():
        if any(type(config[section][k]) is not int or config[section][k] < 0 for k in keys):
            raise ValueError("Example limits must be non-negative integers")
    if any(type(config["training"][k]) is not int or config["training"][k] < 1
           for k in ("vcm_epochs", "batch_size", "patience")):
        raise ValueError("Epochs, batch size and patience must be positive integers")
    if config["training"]["device"] not in {"cpu", "cuda"}:
        raise ValueError("training.device must be cpu or cuda")
    if type(config["training"]["seed"]) is not int or config["training"]["seed"] < 0:
        raise ValueError("training.seed must be a nonnegative integer")
    learning_rate = config["training"]["learning_rate"]
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be a finite positive number")
    versions = config["publish"]["python_versions"]
    if not versions or len(set(versions)) != len(versions) or set(versions) - {"3.11", "3.12", "3.13"}:
        raise ValueError("Choose unique publish Python versions from 3.11, 3.12 and 3.13")
    if type(config["publish"]["sample_per_intent"]) is not int or config["publish"]["sample_per_intent"] < 1:
        raise ValueError("publish.sample_per_intent must be positive")
    audio = config["audio"]
    if audio["sample_rate"] != 16000 or audio["block_ms"] != 40:
        raise ValueError("The current checkpoint uses 16 kHz and 40 ms capture blocks")
    if any(not isinstance(audio[k], (int, float)) or type(audio[k]) is bool or not math.isfinite(audio[k]) or audio[k] <= 0
           for k in ("preroll_ms", "silence_ms", "max_utterance_seconds", "energy_floor", "noise_multiplier")):
        raise ValueError("Audio parameters must be positive numbers")
    if audio["max_utterance_seconds"] > 8 or audio["silence_ms"] >= audio["max_utterance_seconds"] * 1000:
        raise ValueError("Endpointing must stay inside the current eight-second model contract")
    return config
