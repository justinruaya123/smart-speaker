"""Current component checks and held-out evaluation; no historical test tree."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .audio import load_audio
from .pipeline import EnergyVAD, Pipeline


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def run_checks(config, bundle):
    """Real graph smoke checks plus explicitly labelled routing fixtures."""
    from .config import resolve_switches
    checks = {}
    def require(condition, message):
        if not condition:
            raise AssertionError(message)
    require(resolve_switches({"vad": True, "vcm": True, "pipeline": "vad and vcm"})["pipeline"],
            "Boolean expression resolution")
    for bad in ({"a": "__import__('os')"}, {"a": "b", "b": "a"}, {"a": "missing"}, {"a": 1}):
        try:
            resolve_switches(bad)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError("Invalid boolean expression accepted")
    checks["config"] = {"status": "passed", "safe_boolean_expressions": True}
    pcm = (.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    if config["tests"]["vad"]:
        vad = EnergyVAD(config["audio"])
        require(not vad.segment(np.zeros(16000, dtype=np.float32)), "Silence triggered VAD")
        sequence = np.pad(pcm, (8000, 8000))
        from einops import pack
        twice, _ = pack([sequence, sequence], "*")
        found = vad.segment(twice)
        require(len(found) == 2 and not any(timeout for _, timeout in found), "Repeated VAD endpointing")
        from einops import repeat
        found = vad.segment(repeat(pcm, "t -> (r t)", r=10))
        require(found and found[0][1], "Overlong speech did not time out")
        from .capture import resolve_microphone
        devices = [{"max_input_channels": 0}, {"max_input_channels": 2}]
        require(resolve_microphone(devices, [], 1)[0] == 1, "Numeric microphone selection")
        try:
            resolve_microphone(devices, [], 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Output-only device accepted")
        # Simulated PortAudio callbacks exercise the actual live-capture loop at
        # 48 kHz, including channel selection, fallback, overflow and two endpoints.
        import contextlib
        import io
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from .capture import listen
        rate = 48000
        tone = (.1 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
        native, _ = pack([np.pad(tone, (rate//2, rate//2)), np.pad(tone, (rate//2, rate//2))], "*")
        block = round(rate * config["audio"]["block_ms"] / 1000)
        class FakePortAudioError(Exception):
            pass
        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs, self.offset = kwargs, 0
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            @property
            def active(self):
                if self.offset >= len(native):
                    return False
                audio = np.zeros((min(block, len(native)-self.offset), 2), dtype=np.float32)
                audio[:, 1] = native[self.offset:self.offset+block]
                self.kwargs["callback"](audio, len(audio), None, self.offset == 0)
                self.offset += len(audio)
                return True
        def input_settings(**kwargs):
            if kwargs["samplerate"] == 16000:
                raise FakePortAudioError("native rate only")
        fake = SimpleNamespace(InputStream=FakeStream, PortAudioError=FakePortAudioError,
            check_input_settings=input_settings, default=SimpleNamespace(device=(1, 1)),
            query_hostapis=lambda: [{"name": "Simulated"}],
            query_devices=lambda: [{"max_input_channels": 0}, {"max_input_channels": 2,
                "name": "Simulated input", "hostapi": 0, "default_samplerate": rate}])
        lengths = []
        cfg = copy.deepcopy(config); cfg["runtime"]["vad"] = True
        stub = SimpleNamespace(config=cfg, utterance=lambda audio, timeout:
            (lengths.append(len(audio)) or {"status": "fixture", "timeout": timeout}))
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"sounddevice": fake}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            listen(stub, SimpleNamespace(mic=1, channel=1, capture_rate=None))
        output = [json.loads(line) for line in stdout.getvalue().splitlines()]
        require(len(lengths) == 2 and all(16000 <= n < 32000 for n in lengths), "Native microphone resampling/endpoints")
        require(any(item.get("reason") == "audio_overflow" for item in output), "Overflow must be rejected")
        require(json.loads(stderr.getvalue())["capture_rate"] == 48000, "Native-rate fallback")
        checks["vad"] = {"status": "passed", "silence": True, "repeated_utterances": 2,
                         "timeout": True, "numeric_microphone": True, "simulated_native_capture_rate": 48000,
                         "simulated_capture_endpoints": len(lengths), "overflow_rejected": True, "live_microphone": False}
    if config["tests"]["vcm"]:
        from .inference import Recognizer
        model = Recognizer(bundle, threads=config["validation"]["threads"])
        scores = model.classify(pcm)
        require(np.isfinite(scores).all() and abs(float(scores.sum()) - 1) < 1e-6, "VCM probabilities")
        silence = model.predict(np.zeros(16000, dtype=np.float32), stream=False)
        require(silence["status"] == "rejected" , "Silence must abstain")
        require(model.stream is None, "VCM-only path loaded another model")
        checks["vcm"] = {"status": "passed", "labels": model.labels, "silence": silence["status"],
                         "providers": model.vcm.get_providers()}
    if config["tests"]["pipeline"]:
        cfg = copy.deepcopy(config)
        cfg["runtime"] = {"vad": True, "vcm": True}
        pipeline = Pipeline(bundle, cfg)
        require(pipeline.process(np.zeros(16000, dtype=np.float32))[0]["reason"] == "no_speech", "VAD bypass")
        results = pipeline.process(pcm)
        require(results and all("status" in r for r in results), "Full pipeline result")
        # Inject probabilities into an independent in-memory copy to check handoff.
        # Production artifacts and their disabled custom-media policy remain untouched.
        model = pipeline.recognizer
        original = copy.deepcopy(model.metadata)
        vocab = model.metadata["acoustic_slots"]["vocabulary"]
        def onehot(values, value):
            p = np.zeros(len(values)); p[values.index(value)] = 1.; return p
        slot = {name: onehot(values, values[0]) for name, values in vocab.items()}
        slot["guard"] = onehot(vocab["guard"], "READY")
        slot["media_mode"] = onehot(vocab["media_mode"], "CUSTOM")
        slot["media"] = onehot(vocab["media"], "OTHER")
        family = ["MEDIA_PLAY"]
        model.classify_with_slots = lambda audio: (onehot(model.labels, family[0]), slot, None)
        result = model.predict(pcm, stream=False)
        require(result["status"] == "needs_parameters", "Custom media must clarify")
        require(result["reason"] == "catalog_decoder_required", "Unexpected fallback")
        require(not hasattr(model, "transcribe"), "Transcription API reintroduced")
        family[0] = "UNKNOWN"
        require(model.predict(pcm, stream=False)["status"] == "rejected", "Unknown accepted")
        model.metadata = original
        checks["pipeline"] = {"status": "passed", "real_vad_vcm_replay": True,
            "routing_fixtures": ["custom media clarifies", "UNKNOWN rejected"],
            "production_thresholds_modified": False}
    return checks


def audit_data(root):
    root = Path(root)
    descriptor = json.loads((root / "data/dataset.json").read_text(encoding="utf-8"))
    index = json.loads((root / "data/audio-index.json").read_text(encoding="utf-8"))
    splits = {}
    required_audio = set()
    for task in ("vcm",):
        path = root / f"data/{task}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["current_manifests"][task]:
            raise ValueError(f"Changed {task} manifest; update the reviewed dataset descriptor first")
        rows = read_rows(path)
        splits[task] = dict(Counter(r["split"] for r in rows))
        required_audio.update(row["audio"] for row in rows)
        for field in ("id", "audio", "speaker_id", "paired_group"):
            groups = defaultdict(set)
            for row in rows:
                if row.get(field):
                    groups[row[field]].add(row["split"])
            if any(len(s) > 1 for s in groups.values()):
                raise ValueError(f"{task}: {field} leaks across dataset splits")
        if any(row["audio"] not in index for row in rows):
            raise ValueError("Manifest audio missing from the integrity index")
    for relative in sorted(required_audio):
        expected = index[relative]
        path = (root / relative).resolve()
        if not path.is_relative_to((root / "data/audio").resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid current audio path: {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Changed audio: {relative}")
    if hashlib.sha256((root / "data/references.jsonl").read_bytes()).hexdigest() != descriptor["references_sha256"]:
        raise ValueError("Reference labels changed")
    return {"audio_files_verified": len(required_audio), "split_counts": splits,
            "recording_speaker_pair_split_leaks": 0, "reference_quality": "programmatic silver"}


def evaluate(config, bundle, rows, root, references):
    from .metrics import metrics
    from .text import normalize_text
    started = time.perf_counter()
    selected = config["tests"]
    vcm = pipeline = None
    if selected["vcm"]:
        from .inference import Recognizer
        vcm = Recognizer(bundle, threads=config["validation"]["threads"])
    if selected["pipeline"]:
        cfg = copy.deepcopy(config); cfg["runtime"] = {"vad": True, "vcm": True}
        pipeline = Pipeline(bundle, cfg)
    vad = EnergyVAD(config["audio"]) if selected["vad"] else None
    records, endpoint_records = [], []
    speech_count = segments_count = timeouts = 0
    command_counts = {name: Counter() for name in ("vcm", "pipeline")}
    predictions = []
    for i, row in enumerate(rows, 1):
        pcm, rate = load_audio(Path(root) / row["audio"])
        entry = {"id": row["id"], "gold": row["command"]}
        if vad:
            segments = vad.segment(pcm)
            speech_count += bool(segments); segments_count += len(segments)
            timeouts += sum(int(timeout) for _, timeout in segments)
        reference = references.get(row["id"], {}).get("reference")
        finite = reference and reference.get("status") == "accepted" and not (
            set(reference["command"]["arguments"]) & {"title", "artist", "selection_text"})
        for name, engine, output in (("vcm", vcm, records), ("pipeline", pipeline, endpoint_records)):
            if engine is None:
                continue
            results = [engine.predict(pcm, stream=False)] if name == "vcm" else engine.process(pcm)
            # All segments count as emissions; multiple endpoints cannot earn a complete-command success.
            result = results[0] if len(results) == 1 else {"status": "rejected", "reason": "multiple_endpoints"}
            output.append({"gold": row["command"], "predicted": result.get("vcm_command", "UNKNOWN"),
                           "accepted": result.get("intent_accepted", False)})
            counts = command_counts[name]
            counts["finite_references"] += bool(finite)
            counts["emitted"] += sum(r.get("status") == "accepted" and "command" in r for r in results)
            counts["correct_complete"] += bool(finite and result.get("status") == "accepted"
                and result.get("vcm_command") == row["command"] and result.get("command") == reference["command"])
            entry[name] = results
        predictions.append(entry)
        if i % 200 == 0:
            print(f"VALIDATE {i}/{len(rows)} elapsed={time.perf_counter()-started:.0f}s", flush=True)
    report = {"examples": len(rows), "elapsed_seconds": time.perf_counter()-started,
              "quality_gate_passed": False, "target_hardware_measured": False}
    for name, engine, output in (("vcm", vcm, records), ("pipeline", pipeline, endpoint_records)):
        if engine is not None:
            labels = (engine if name == "vcm" else engine.recognizer).labels
            counts = command_counts[name]
            report[name] = {"intent": metrics(output, labels), "commands": dict(counts),
                "exact_finite_command_recall": counts["correct_complete"] / max(1, counts["finite_references"]),
                "accepted_command_precision": counts["correct_complete"] / max(1, counts["emitted"])}
    if vad:
        report["vad"] = {"recordings_with_speech": speech_count, "endpoints": segments_count, "timeouts": timeouts,
                         "boundary_accuracy_assessed": False}
    command_paths = [report[name] for name in ("vcm", "pipeline") if name in report]
    report["quality_gate_passed"] = bool(command_paths) and all(
        part["intent"]["gates"]["inherited_pass"]
        and not part["intent"]["gates"]["additional_low_per_intent_precision"]
        and part["accepted_command_precision"] >= .95
        and part["exact_finite_command_recall"] >= .60 for part in command_paths)
    report["quality_gate_criteria"] = {
        "per_family_raw_recall": .80, "per_family_accepted_recall": .60,
        "global_and_per_family_accepted_precision": .95, "maximum_unknown_false_accept_rate": .01,
        "finite_command_recall": .60, "finite_command_precision": .95,
        "scope": "selected command paths only; VAD-only does not establish command quality"}
    report["quality_gate_note"] = "Software quality gates are separate from production readiness. Target-device, live microphone, reviewed-reference and dynamic-catalog acceptance remain unestablished."
    return report, predictions


def validate_dataset(config, root):
    root = Path(root)
    audit = audit_data(root)
    split, maximum = config["validation"]["split"], config["validation"]["max_examples"]
    rows = [r for r in read_rows(root / "data/vcm.jsonl") if r["split"] == split]
    if maximum:
        rows = rows[:maximum]
    refs = {r["id"]: r for r in read_rows(root / "data/references.jsonl")}
    report, predictions = evaluate(config, root / "artifacts", rows, root, refs)
    report.update(split=split, limited=bool(maximum), data_audit=audit)
    out = root / ".cache/validation/predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8")
    report["prediction_log"] = ".cache/validation/predictions.jsonl"
    return report


def validate_samples(config, bundle, sample_root):
    sample_root = Path(sample_root)
    rows = read_rows(sample_root / "samples.jsonl")
    refs = {r["id"]: r["annotation"] for r in rows}
    report, _ = evaluate(config, bundle, rows, sample_root, refs)
    report["scope"] = "packaged held-out smoke samples; not the full validation set"
    return report
