import ast
import copy
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
    if any(any(key=='asr' or key.startswith('asr_') for key in config.get(section,{}))
           for section in ('runtime','tests','training')) or config.get('parameters',{}).get('backend')=='asr':
        raise ValueError('The ASR model is retired. Use parameters enabled=false/backend="none" for standalone VCM.')
    config.setdefault("parameters", {"enabled": False, "backend": "none"})
    config.setdefault("wake", {})
    wake_defaults = {"enabled": False, "phrase": "hey jarvis", "threshold": .5,
                     "release_silence_ms": 160, "release_timeout_seconds": 2.,
                     "command_timeout_seconds": 5., "cooldown_ms": 1000}
    for key, value in wake_defaults.items():
        config['wake'].setdefault(key, value)
    config["tests"].setdefault("stt", False)
    config["training"].setdefault("stt", False)
    expected = {"workflow", "tests", "runtime", "parameters", "validation", "training", "publish", "audio", "wake"}
    if set(config) != expected:
        raise ValueError(f"Configuration sections must be {sorted(expected)}")
    for section, keys in {"workflow": {"setup", "train", "test", "validate", "report"},
                          "tests": {"vad", "vcm", "stt", "pipeline"},
                          "runtime": {"vad", "vcm"}}.items():
        if set(config[section]) != keys:
            raise ValueError(f"Unknown or missing {section} switches")
        config[section] = resolve_switches(config[section])
    if (set(config['parameters']) != {'enabled','backend'} or type(config['parameters']['enabled']) is not bool
            or config['parameters']['backend'] not in {'none', 'stt'}
            or config['parameters']['enabled'] and config['parameters']['backend'] == 'none'):
        raise ValueError('Use parameters enabled=false/backend="none" for the standalone VCM')
    wake = config['wake']
    if set(wake) != set(wake_defaults) or type(wake['enabled']) is not bool or wake['phrase'] != 'hey jarvis':
        raise ValueError('The installed pretrained wake phrase is hey jarvis')
    for key in ('threshold','release_silence_ms','release_timeout_seconds','command_timeout_seconds','cooldown_ms'):
        if type(wake[key]) not in (int, float) or not math.isfinite(wake[key]) or wake[key] <= 0:
            raise ValueError(f'wake.{key} must be finite and positive')
    if (not 0 < wake['threshold'] < 1 or not 64 <= wake['release_silence_ms'] <= 1000
            or not .5 <= wake['release_timeout_seconds'] <= 5 or not 1 <= wake['command_timeout_seconds'] <= 15
            or not 400 <= wake['cooldown_ms'] <= 5000):
        raise ValueError('Wake thresholds/timeouts are outside the supported bounded range')
    if wake['enabled'] and (not all(config['runtime'].values()) or config['parameters']['enabled']):
        raise ValueError('Wake capture requires VAD and standalone VCM without a transcription backend')
    for key in ("vcm", "stt", "update_artifacts"):
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
    for key in ('stt_width','stt_layers','stt_word_epochs','stt_epochs','stt_batch_size','stt_patience'):
        if key in config['training'] and (type(config['training'][key]) is not int or config['training'][key]<1):
            raise ValueError(f'training.{key} must be a positive integer')
    if 'stt_learning_rate' in config['training'] and (type(config['training']['stt_learning_rate']) not in (int,float)
            or not math.isfinite(config['training']['stt_learning_rate']) or config['training']['stt_learning_rate']<=0):
        raise ValueError('training.stt_learning_rate must be finite and positive')
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
    audio.setdefault('vad_backend', 'energy')
    audio.setdefault('speech_threshold', .5)
    if audio['vad_backend'] not in {'energy','silero'} or type(audio['speech_threshold']) not in (int,float) or not .2 <= audio['speech_threshold'] <= .9:
        raise ValueError('Select energy/silero VAD and a speech threshold between 0.2 and 0.9')
    if wake['enabled'] and audio['vad_backend'] != 'silero':
        raise ValueError('Wake capture uses the installed Silero VAD')
    audio.setdefault("calibration_ms", 1000)  # Older Pi configs keep working.
    if audio["sample_rate"] != 16000 or audio["block_ms"] != 40:
        raise ValueError("The current checkpoint uses 16 kHz and 40 ms capture blocks")
    if any(not isinstance(audio[k], (int, float)) or type(audio[k]) is bool or not math.isfinite(audio[k]) or audio[k] <= 0
           for k in ("preroll_ms", "silence_ms", "max_utterance_seconds", "energy_floor", "noise_multiplier", "calibration_ms")):
        raise ValueError("Audio parameters must be positive numbers")
    if not audio["block_ms"] <= audio["calibration_ms"] <= 5000 or audio["noise_multiplier"] <= 1:
        raise ValueError("Use 40-5000 ms calibration and noise_multiplier greater than one")
    if audio["max_utterance_seconds"] > 8 or audio["silence_ms"] >= audio["max_utterance_seconds"] * 1000:
        raise ValueError("Endpointing must stay inside the current eight-second model contract")
    return config


def select_backend(config, backend):
    if backend not in {'none','stt'}:raise ValueError('The legacy ASR model is retired')
    selected=copy.deepcopy(config)
    selected['parameters']={'enabled':backend != 'none','backend':backend}
    selected['tests'].update(stt=backend != 'none')
    return selected


def config_text(config):
    """Serialize our flat-section, scalar/list TOML configuration for shipping."""
    import json
    return '\n\n'.join('['+section+']\n'+'\n'.join(key+' = '+json.dumps(value) for key,value in values.items())
                       for section,values in config.items())+'\n'
