"""Six-voice, evaluation-only speech generation and bound VCM service replay.

Never trains, changes thresholds, enables actions, or certifies Pi performance.
Expected text and commands are used by synthesis and scoring, not inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name("tts-protocol-v2.json")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def bound_json(item):
    if sha(item["path"]) != item["sha256"]:
        raise ValueError("Changed input binding: " + item["path"])
    return read_json(item["path"])


def cache_path(path):
    result = Path(path).resolve()
    if not result.is_relative_to((ROOT / ".cache").resolve()):
        raise ValueError("Generated payloads must stay under repo .cache")
    return result


def trial_id(voice, text, speed=None):
    """Keep v1 identities; a declared speed is part of a v2 waveform identity."""
    identity = [voice, text]
    if speed is not None:
        identity.append(float(speed))
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def protocol_grid(spec):
    voices = spec.get("voices", [])
    ids = [voice.get("id") for voice in voices]
    if (not ids or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Empty/duplicate/invalid voice grid")
    speeds = spec.get("speeds", [spec.get("speed", 1.0)])
    if (not isinstance(speeds, list) or not speeds
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   or value <= 0 for value in speeds)
            or len(set(speeds)) != len(speeds)):
        raise ValueError("Empty/duplicate/invalid speed grid")
    return voices, [float(speed) for speed in speeds]


def expand_grid(cases, spec):
    """Expand every source reference; only identical voice/speed/text shares audio."""
    voices, speeds = protocol_grid(spec)
    audios, trials = {}, []
    for case in cases:
        for voice in voices:
            for speed in speeds:
                identifier = trial_id(voice["id"], case["spoken_text"],
                                      speed if "speeds" in spec else None)
                audios[identifier] = {"audio_id": identifier,
                                      "text": case["spoken_text"], **voice,
                                      "speed": speed}
                trials.append({**case, "audio_id": identifier,
                               "voice": voice["id"], "speed": speed})
    return list(audios.values()), trials


def validate_plan(plan, contract, spec):
    """Fail closed on dropped grid cells, changed verbatims or changed references."""
    audio, trials = expand_grid(merge_cases(contract), spec)
    default_speed = spec.get("speed", 1.0)
    for key, expected in (("audio", audio), ("trials", trials)):
        actual = [{**row, "speed": row.get("speed", default_speed)}
                  for row in plan.get(key, [])]
        if sorted(map(canonical, actual)) != sorted(map(canonical, expected)):
            raise ValueError("Changed/incomplete Cartesian grid coverage/reference: " + key)
    counts = {key: len(contract[key]) for key in ("cases", "catalog_entries")}
    if plan.get("source_counts") != counts:
        raise ValueError("Changed source coverage counts")
    baseline_trials = sum(row["track"] == "optionb" for row in trials)
    if plan.get("optionb_trials") != baseline_trials:
        raise ValueError("Changed OptionB grid coverage count")


def merge_cases(contract):
    """Keep all case references; share audio only for identical spoken text."""
    result, seen, meanings = [], set(), {}
    for source in ("cases", "catalog_entries"):
        if not contract.get(source):
            raise ValueError("Missing full contract/catalog: " + source)
        for number, row in enumerate(contract[source]):
            text = row.get("spoken_text", row.get("text"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Missing exact spoken text")
            if "command" not in row:
                raise ValueError("Missing reviewed command reference")
            identifier = source + "/" + str(row.get("id", number))
            if identifier in seen:
                raise ValueError("Duplicate source case identifier")
            seen.add(identifier)
            meaning = canonical([row["command"], row.get("reason")])
            if text in meanings and meanings[text] != meaning:
                raise ValueError("Conflicting commands for identical speech")
            meanings[text] = meaning
            track = "selected_catalog"
            if source == "cases":
                track = ("optionb" if row.get("tier") == "baseline"
                         else "personal_extensions")
            result.append({"case_id": identifier, "spoken_text": text,
                           "command": row["command"], "track": track,
                           "reason": row.get("reason")})
    return result


def asset_paths(directory, spec):
    result = {}
    for name, expected in spec["files"].items():
        path = (Path(directory) / name).resolve()
        if (not path.is_relative_to(Path(directory).resolve())
                or not path.is_file() or sha(path) != expected):
            raise ValueError("Missing/mismatched TTS asset hash: " + name)
        result[name] = path
    return result


def to_pcm16(audio, sample_rate):
    """Anti-aliased resampling; never trim or change the spoken content."""
    from scipy.signal import resample_poly

    audio = np.asarray(audio, dtype=np.float32)
    if (sample_rate != 24000 or audio.ndim != 1 or audio.size == 0
            or not np.isfinite(audio).all() or np.max(abs(audio)) >= 1
            or np.sqrt(np.mean(audio.astype(float) ** 2)) < 1e-6):
        raise ValueError("Invalid, clipped or silent native TTS audio")
    result = resample_poly(audio, 2, 3).astype(np.float32)
    if (not 400 <= len(result) <= 128000
            or not np.isfinite(result).all() or np.max(abs(result)) >= 1):
        raise ValueError("TTS audio violates unchanged VCM capture contract")
    return result


def review_complete(records, review):
    """Require independent listening for every WAV before VCM evaluation."""
    if review.get("blind_to_vcm") is not True:
        raise ValueError("Review must be completed without VCM outcomes")
    items = review.get("items", [])
    ids = [item["audio_id"] for item in items]
    expected = [item["audio_id"] for item in records]
    if (not records or len(set(expected)) != len(expected)
            or len(ids) != len(set(ids)) or set(ids) != set(expected)):
        raise ValueError("Incomplete/duplicate quality review")
    indexed = {item["audio_id"]: item for item in items}
    for row in records:
        item = indexed[row["audio_id"]]
        if (row.get("status") != "generated"
                or item.get("sha256") != row.get("sha256")
                or item.get("heard_exact") is not True
                or type(item.get("quality")) is not int
                or not 4 <= item["quality"] <= 5
                or not item.get("reviewer") or not item.get("reviewed_at")):
            raise ValueError("Unqualified TTS review: " + row["audio_id"])


def outcome(trial, output):
    """Classify only the authoritative application decision, never a candidate."""
    if output is None:
        return "missing"
    if not isinstance(output, dict) or output.get("fault"):
        return "fault"
    app = output.get("final_application")
    if not isinstance(app, dict):
        return "fault"
    accepted, actual, reason = app.get("accepted"), app.get("command"), app.get("reason")
    if (type(accepted) is not bool or accepted != (actual is not None)
            or reason in ("capture_fault", "unqualified_model")
            or (reason is not None and not isinstance(reason, str))
            or (accepted and reason is not None)
            or (accepted and (not isinstance(actual, dict)
                              or not isinstance(actual.get("slots"), dict)
                              or not isinstance(actual.get("intent"), str)
                              or not actual["intent"]))):
        return "fault"
    expected = trial["command"]
    if expected is not None:
        if not accepted:
            return "positive_rejection"
        try:
            return ("correct_positive" if canonical(actual) == canonical(expected)
                    else "wrong_command")
        except (TypeError, ValueError):
            return "fault"
    if accepted:
        return "false_accept"
    correct = bool(reason) and (trial.get("reason") is None or reason == trial["reason"])
    return "correct_rejection" if correct else "wrong_rejection"


def _metrics(trials, outputs):
    result = {key: 0 for key in (
        "trials", "supported", "negative", "accepted", "correct_positive",
        "correct_rejections", "strict_correct", "false_accepts", "faults",
        "missing", "supported_faults", "negative_faults", "supported_missing",
        "negative_missing", "positive_rejections", "wrong_accepted_positive",
        "incorrect_rejections")}
    slots = defaultdict(lambda: {"n": 0, "correct_accepted": 0})
    joint = {"n": 0, "correct_accepted": 0}
    for trial in trials:
        result["trials"] += 1
        expected = trial["command"]
        positive = expected is not None
        result["supported" if positive else "negative"] += 1
        supplied = (expected or {}).get("slots", {})
        joint["n"] += int(bool(supplied))
        for field in supplied:
            slots[field]["n"] += 1
        output = outputs.get(trial["audio_id"])
        kind = outcome(trial, output)
        if kind in ("missing", "fault"):
            field = "missing" if kind == "missing" else "faults"
            result[field] += 1
            result[("supported_" if positive else "negative_") + field] += 1
            continue
        app = output["final_application"]
        accepted, actual = app.get("accepted"), app.get("command")
        result["accepted"] += int(accepted)
        if positive:
            result["correct_positive"] += int(kind == "correct_positive")
            result["positive_rejections"] += int(kind == "positive_rejection")
            result["wrong_accepted_positive"] += int(kind == "wrong_command")
            matches = []
            for field, value in supplied.items():
                matched = (
                    accepted and actual.get("intent") == expected["intent"]
                    and canonical(actual.get("slots", {}).get(field))
                    == canonical(value))
                slots[field]["correct_accepted"] += int(matched)
                matches.append(matched)
            joint["correct_accepted"] += int(bool(supplied) and all(matches))
        else:
            result["false_accepts"] += int(accepted)
            result["correct_rejections"] += int(kind == "correct_rejection")
            result["incorrect_rejections"] += int(kind == "wrong_rejection")
        result["strict_correct"] += int(kind in ("correct_positive", "correct_rejection"))
    result["recall"] = (result["correct_positive"] / result["supported"]
                        if result["supported"] else None)
    result["precision"] = (result["correct_positive"] / result["accepted"]
                           if result["accepted"] else None)
    result["false_accept_rate"] = (result["false_accepts"] / result["negative"]
                                   if result["negative"] else None)
    result["complete_command_recall"] = result["recall"]
    result["complete_command_precision"] = result["precision"]
    result["strict_accuracy"] = (result["strict_correct"] / result["trials"]
                                 if result["trials"] else None)
    result["correct_rejection_rate"] = (result["correct_rejections"] / result["negative"]
                                         if result["negative"] else None)
    joint["accuracy"] = joint["correct_accepted"] / joint["n"] if joint["n"] else None
    for value in slots.values():
        value["accuracy"] = value["correct_accepted"] / value["n"]
    result["supplied_slots"] = dict(slots)
    result["joint_supplied_slots"] = joint
    return result


def score_cases(trials, results):
    indexed = {row["audio_id"]: row for row in results}
    expected_ids = {row["audio_id"] for row in trials}
    if len(indexed) != len(results) or set(indexed) - expected_ids:
        raise ValueError("Duplicate/unexpected inference output")
    unique, voices, tracks = {}, defaultdict(dict), defaultdict(list)
    speeds = defaultdict(dict)
    voice_speeds = defaultdict(lambda: defaultdict(dict))
    for row in trials:
        prior = unique.get(row["audio_id"])
        if prior is not None:
            fields = ("voice", "spoken_text", "command", "reason")
            if (any(canonical(prior.get(key)) != canonical(row.get(key)) for key in fields)
                    or prior.get("speed", 1.0) != row.get("speed", 1.0)):
                raise ValueError("Conflicting references for one waveform")
        unique.setdefault(row["audio_id"], row)
        voices[row["voice"]].setdefault(row["audio_id"], row)
        speed = str(float(row.get("speed", 1.0)))
        speeds[speed].setdefault(row["audio_id"], row)
        voice_speeds[row["voice"]][speed].setdefault(row["audio_id"], row)
        tracks[row["track"]].append(row)
    intents, cases = defaultdict(list), defaultdict(list)
    for row in unique.values():
        intents[(row["command"] or {}).get("intent", "OOV")].append(row)
    for row in trials:
        cases[row["case_id"]].append(row)
    total = _metrics(list(unique.values()), indexed)
    track_metrics = {key: _metrics(value, indexed)
                     for key, value in tracks.items()}
    misses = []
    for row in trials:
        output = indexed.get(row["audio_id"])
        kind = outcome(row, output)
        if kind not in ("correct_positive", "correct_rejection"):
            misses.append({"case_id": row["case_id"], "audio_id": row["audio_id"],
                           "voice": row["voice"], "speed": row.get("speed", 1.0),
                           "track": row["track"], "spoken_text": row["spoken_text"],
                           "expected_command": row["command"],
                           "expected_reason": row.get("reason"), "outcome": kind,
                           "final_application": (output or {}).get("final_application"),
                           "fault": (output or {}).get("fault")})
    return {
        "case_references": len(trials), "unique_audio": total,
        "case_reference_metrics": _metrics(trials, indexed),
        "per_voice": {key: _metrics(list(value.values()), indexed)
                      for key, value in voices.items()},
        "per_speed": {key: _metrics(list(value.values()), indexed)
                      for key, value in speeds.items()},
        "per_voice_speed": {voice: {
            speed: _metrics(list(rows.values()), indexed) for speed, rows in group.items()}
            for voice, group in voice_speeds.items()},
        "per_track": track_metrics,
        "per_intent": {key: _metrics(value, indexed)
                       for key, value in intents.items()},
        "per_case": {key: _metrics(value, indexed)
                     for key, value in cases.items()},
        "metric_definitions": {
            "complete_command": "Accepted command exactly equals the entire typed reference, with no extra or missing slots",
            "joint_supplied_slots": "Supported references with nonempty typed slots: accepted correct intent and every reference slot correct; extra slots still fail complete-command correctness",
            "slot_reference_scope": "All typed reference slots, including declared defaults and schema constants; not a claim that each value was independently spoken",
            "denominators": "unique_audio, per_voice, per_speed, per_voice_speed and per_intent deduplicate shared waveforms; case_reference_metrics, per_track and per_case retain every source reference",
        },
        "missed_cases": misses,
        "pass": bool(total["supported"] and all(
            item["strict_correct"] == item["trials"]
            for item in [total, *track_metrics.values()])),
        "human_generalization_established": False,
        "ownership_established": False, "pi_evidence": False,
    }


def export_missed_verbatims(path, metrics):
    """Retain exact reference text and every failed case reference, without overwrites."""
    items = metrics["missed_cases"]
    write_json(path, {"format": "tts-exact-missed-verbatims-v1",
                      "case_references": len(items),
                      "unique_audio": len({row["audio_id"] for row in items}),
                      "items": items})


def make_plan(contract_path, output, protocol_path=SPEC):
    sys.path.insert(0, str(ROOT))
    from scripts.runtime.pcv_schema import baseline_cases, validate_command
    from scripts.runtime.vcm_intent_slots_contract import verify_contract

    contract, spec = read_json(contract_path), read_json(protocol_path)
    verify_contract(contract)
    baseline = baseline_cases()
    rows = {row["id"]: row for row in contract["cases"]}
    if len(baseline) != 93:
        raise ValueError("Unexpected frozen OptionB definition count")
    for expected in baseline:
        actual = rows.get(expected["id"], {})
        for key in ("text", "spoken_text", "command"):
            if canonical(actual.get(key)) != canonical(expected.get(key)):
                raise ValueError("Changed OptionB verbatim/reference: " + key)
    cases = merge_cases(contract)
    for case in cases:
        if case["command"] is not None:
            validate_command(case["command"])
    audios, trials = expand_grid(cases, spec)
    value = {"format": spec["experiment_id"],
             "contract": binding(contract_path), "protocol": binding(protocol_path),
             "source_counts": {key: len(contract[key])
                               for key in ("cases", "catalog_entries")},
             "optionb_trials": sum(row["track"] == "optionb" for row in trials),
             "audio": audios, "trials": trials,
             "fit_allowed": False, "selection_allowed": False,
             "calibration_allowed": False, "enrollment_allowed": False}
    validate_plan(value, contract, spec)
    write_json(cache_path(output), value)
    return {"status": "planned_not_generated", "audio_n": len(audios),
            "case_trials": len(trials),
            "optionb_trials": value["optionb_trials"]}


def fetch_assets(directory, protocol_path=SPEC):
    from huggingface_hub import hf_hub_download

    spec = read_json(protocol_path)
    directory = cache_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in spec["files"]:
        hf_hub_download(repo_id=spec["repository"], filename=filename,
                        revision=spec["revision"], local_dir=str(directory))
    files = asset_paths(directory, spec)
    receipt = {"repository": spec["repository"], "revision": spec["revision"],
               "protocol": binding(protocol_path),
               "files": {key: binding(path) for key, path in files.items()}}
    destination = directory / "asset-receipt.json"
    if destination.exists():
        if read_json(destination) != receipt:
            raise ValueError("Existing asset receipt changed")
    else:
        write_json(destination, receipt)
    return {"status": "downloaded_hash_verified", "files": len(files)}


def _package_versions():
    versions = {name: importlib.metadata.version(name) for name in (
        "kokoro-onnx", "numpy", "scipy", "soundfile", "espeakng-loader", "phonemizer-fork")}
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            versions[name] = importlib.metadata.version(name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    else:
        raise ValueError("Missing ONNX Runtime distribution metadata")
    return versions


def _tts_runtime(spec, files, provider, *, start_method="inline"):
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    if importlib.metadata.version("kokoro-onnx") != spec["kokoro_onnx_version"]:
        raise ValueError("Use the frozen TTS package version")
    if provider not in ("cpu", "cuda"):
        raise ValueError("TTS provider must be cpu or cuda")
    requested = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    if provider == "cuda":
        # The GPU wheel can preload CUDA/cuDNN from the shared Torch installation.
        ort.preload_dlls()
    available = ort.get_available_providers()
    if requested not in available:
        raise ValueError("Requested TTS provider unavailable: " + requested)
    options = ort.SessionOptions()
    options.intra_op_num_threads = spec["threads"]
    options.inter_op_num_threads = 1
    ort.set_seed(spec["seed"])
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if provider == "cuda" else ["CPUExecutionProvider"])
    session = ort.InferenceSession(str(files["kokoro-v1.0.onnx"]), options,
                                   providers=providers)
    if requested not in session.get_providers():
        raise ValueError("Requested TTS provider did not initialize: " + requested)
    if provider == "cuda":
        session.disable_fallback()
    tts = Kokoro.from_session(session, str(files["voices-v1.0.bin"]))
    if not {v["id"] for v in spec["voices"]} <= set(tts.voices):
        raise ValueError("Selected six distinct voice packs are unavailable")
    return {"tts": tts, "session": session, "info": {
        "process_id": os.getpid(), "start_method": start_method, "seed": spec["seed"],
        "providers": session.get_providers(), "execution_provider": requested,
        "requested_provider": provider, "available_providers": available,
        "whole_provider_fallback_disabled": provider == "cuda",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "onnxruntime_module_version": getattr(ort, "__version__", "unknown")}}


def _base_generation_record(item, spec):
    return {"audio_id": item["audio_id"], "voice": item["id"],
            "speed": item.get("speed", spec.get("speed", 1.0)),
            "spoken_text": item["text"], "kind": "synthetic",
            "status": "generation_fault", "failure_stage": "phonemization"}


def _synthesize_record(item, spec, output, runtime):
    import soundfile as sf

    output = Path(output)
    row = _base_generation_record(item, spec)
    tts = runtime["tts"]
    requested = runtime["info"]["execution_provider"]
    try:
        row["failure_stage"] = "provider_validation"
        if requested not in runtime["session"].get_providers():
            raise ValueError("Requested TTS provider was lost; no CPU downgrade")
        row["failure_stage"] = "phonemization"
        phonemes = tts.tokenizer.phonemize(item["text"], item["lang"])
        if not phonemes or len(phonemes) > 490:
            raise ValueError("Empty/overlong TTS phonemes")
        row["failure_stage"] = "synthesis"
        audio, rate = tts.create(item["text"], voice=item["id"], lang=item["lang"],
                                 speed=row["speed"], trim=False)
        row["failure_stage"] = "native_preservation"
        native_array = output / "native" / (item["audio_id"] + ".npy")
        with native_array.open("xb") as stream:
            np.save(stream, np.asarray(audio), allow_pickle=False)
        row.update(native_array=binding(native_array),
                   native_sample_rate=(int(rate) if isinstance(rate, (int, np.integer))
                                       else str(rate)))
        native = output / "native" / (item["audio_id"] + ".wav")
        target = output / "audio" / (item["audio_id"] + ".wav")
        row["failure_stage"] = "native_wav_format"
        sf.write(native, audio, rate, subtype="FLOAT")
        row["native"] = binding(native)
        row["failure_stage"] = "provider_validation"
        if requested not in runtime["session"].get_providers():
            raise ValueError("Requested TTS provider was lost during synthesis")
        row["failure_stage"] = "capture_qa"
        pcm = to_pcm16(audio, rate)
        row["failure_stage"] = "pcm_write"
        sf.write(target, pcm, 16000, subtype="PCM_16")
        row["written_wav"] = binding(target)
        row["failure_stage"] = "pcm_format_qa"
        decoded, actual_rate = sf.read(target, dtype="float32")
        if (actual_rate != 16000 or decoded.ndim != 1 or len(decoded) != len(pcm)
                or sf.info(target).subtype != "PCM_16"):
            raise ValueError("Written WAV format mismatch")
        row.update(status="generated", path=str(target), sha256=sha(target),
                   failure_stage=None, pcm_sha256=hashlib.sha256(
                       decoded.astype("<f4").tobytes()).hexdigest(),
                   samples=len(decoded), sample_rate=16000)
    except Exception as error:
        row["fault"] = type(error).__name__ + ": " + str(error)
    receipt = output / "worker-records" / (item["audio_id"] + ".json")
    providers = runtime["session"].get_providers()
    info = {**runtime["info"], "providers": providers,
            "execution_provider": requested if requested in providers else None}
    value = {"record": row, "worker": info}
    write_json(receipt, value)
    row["record_receipt"] = binding(receipt)
    return value


_WORKER_RUNTIME = None


def _initialize_worker(spec, files, output, provider, source, gpu_pin):
    global _WORKER_RUNTIME
    if sha(__file__) != source["sha256"]:
        raise ValueError("Changed worker source before generation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_pin:
        raise ValueError("Worker did not inherit the selected GPU environment")
    if any(sha(path) != spec["files"][name] for name, path in files.items()):
        raise ValueError("Changed worker TTS asset")
    runtime = _tts_runtime(spec, files, provider, start_method="spawn")
    _WORKER_RUNTIME = {**runtime, "spec": spec, "output": output}


def _worker_record(item):
    if _WORKER_RUNTIME is None:
        raise ValueError("TTS worker was not initialized")
    return _synthesize_record(item, _WORKER_RUNTIME["spec"],
                              _WORKER_RUNTIME["output"], _WORKER_RUNTIME)


def _recover_worker_record(item, spec, output, error):
    """Recover completed metadata after transport failure; never synthesize a second take."""
    receipt = output / "worker-records" / (item["audio_id"] + ".json")
    try:
        value = read_json(receipt)
        base = _base_generation_record(item, spec)
        if any(value["record"].get(key) != base[key] for key in (
                "audio_id", "voice", "speed", "spoken_text", "kind")):
            raise ValueError("Changed worker waveform identity")
        value["record"]["record_receipt"] = binding(receipt)
        return value, True
    except (OSError, ValueError, KeyError, TypeError):
        row = {**_base_generation_record(item, spec), "failure_stage": "worker_execution",
               "fault": type(error).__name__ + ": " + str(error)}
        for key, directory, suffix in (("native_array", "native", ".npy"),
                                       ("native", "native", ".wav"),
                                       ("written_wav", "audio", ".wav")):
            path = output / directory / (item["audio_id"] + suffix)
            if path.is_file():
                row[key] = binding(path)
        return {"record": row, "worker": None}, False


def generate(plan_path, assets, output, provider="cpu", workers=1):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("TTS workers must be an integer from 1 through 8")
    plan = read_json(plan_path)
    spec = bound_json(plan["protocol"])
    contract = bound_json(plan["contract"])
    validate_plan(plan, contract, spec)
    files = asset_paths(assets, spec)
    versions, source = _package_versions(), binding(__file__)
    output = cache_path(output)
    if output.exists():
        raise FileExistsError(output)
    if provider not in ("cpu", "cuda"):
        raise ValueError("TTS provider must be cpu or cuda")
    gpu_pin = os.environ.get("CUDA_VISIBLE_DEVICES")
    if workers > 1 and provider == "cuda" and (
            not gpu_pin or len(gpu_pin.split(",")) != 1 or gpu_pin.strip() in ("", "-1")):
        raise ValueError("CUDA workers require one explicit visible GPU pin")
    # The existing single-process path verifies its session before creating files.
    runtime = _tts_runtime(spec, files, provider) if workers == 1 else None
    output.mkdir(parents=True, exist_ok=False)
    for name in ("audio", "native", "worker-records"):
        (output / name).mkdir()
    items = sorted(plan["audio"], key=lambda item: item["audio_id"])
    received, worker_receipts, transport_faults = {}, {}, []
    write_json(output / "run-identity.json", {
        "plan": binding(plan_path), "protocol": plan["protocol"], "source": source,
        "assets": {key: binding(path) for key, path in files.items()},
        "versions": versions, "workers": workers, "provider": provider,
        "cuda_visible_devices": gpu_pin, "seed": spec["seed"],
        "fit_allowed": False, "human_reviewed": False})
    with (output / "generation.jsonl").open("x", encoding="utf-8") as log:
        def retain(value):
            row, worker = value["record"], value["worker"]
            received[row["audio_id"]] = row
            if worker is not None:
                worker_receipts[worker["process_id"]] = worker
            log.write(canonical(row) + "\n")
            log.flush()

        def failed(item, error):
            value, recovered = _recover_worker_record(item, spec, output, error)
            transport_faults.append({"audio_id": item["audio_id"],
                                     "fault": type(error).__name__ + ": " + str(error),
                                     "recovered_persisted_record": recovered})
            retain(value)

        if workers == 1:
            for item in items:
                retain(_synthesize_record(item, spec, output, runtime))
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=_initialize_worker,
                        initargs=(spec, files, str(output), provider, source, gpu_pin)) as pool:
                    pending = {}
                    for item in items:
                        try:
                            pending[pool.submit(_worker_record, item)] = item
                        except Exception as error:
                            failed(item, error)
                    for future in as_completed(pending):
                        item = pending[future]
                        try:
                            retain(future.result())
                        except Exception as error:
                            failed(item, error)
            except Exception as error:
                for item in items:
                    if item["audio_id"] not in received:
                        failed(item, error)
        for item in items:
            if item["audio_id"] not in received:
                failed(item, ValueError("Missing worker result; no regeneration"))
    records = [received[item["audio_id"]] for item in items]
    receipts = [worker_receipts[key] for key in sorted(worker_receipts)]
    info = receipts[0] if receipts else {}
    manifest = {"plan": binding(plan_path), "records": records,
                "assets": {key: binding(path) for key, path in files.items()},
                "versions": versions, "python": sys.version,
                "platform": platform.platform(), "source": source,
                "source_changed_during_generation": sha(__file__) != source["sha256"],
                "seed": spec["seed"], "providers": info.get("providers", []),
                "execution_provider": info.get("execution_provider"), "requested_provider": provider,
                "available_providers": info.get("available_providers", []),
                "cuda_visible_devices": gpu_pin, "requested_workers": workers,
                "worker_receipts": receipts, "worker_transport_faults": transport_faults,
                "record_order": "audio_id",
                "fit_allowed": False, "selection_allowed": False,
                "calibration_allowed": False, "enrollment_allowed": False,
                "human_reviewed": False}
    write_json(output / "manifest.json", manifest)
    write_json(output / "review-template.json", {
        "blind_to_vcm": True, "items": [{
            "audio_id": row["audio_id"], "sha256": row.get("sha256"),
            "voice": row["voice"], "speed": row["speed"],
            "spoken_text": row["spoken_text"], "path": row.get("path"),
            "status": row["status"], "native": row.get("native"),
            "native_array": row.get("native_array"),
            "heard_exact": None, "quality": None, "reviewer": "",
            "reviewed_at": "", "notes": ""} for row in records]})
    faults = sum(row["status"] != "generated" for row in records)
    return {"status": "generated_needs_listening", "attempted": len(records),
            "generated": len(records) - faults,
            "generation_faults": faults,
            "pass": faults == 0 and not manifest["source_changed_during_generation"]}


def checked_manifest(path):
    import soundfile as sf

    manifest = read_json(path)
    plan = bound_json(manifest["plan"])
    contract = bound_json(plan["contract"])
    spec = bound_json(plan["protocol"])
    validate_plan(plan, contract, spec)
    if sha(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("Generator implementation changed")
    ids = [row["audio_id"] for row in manifest["records"]]
    if (len(ids) != len(set(ids))
            or set(ids) != {row["audio_id"] for row in plan["audio"]}):
        raise ValueError("Incomplete generated coverage")
    if set(manifest["assets"]) != set(spec["files"]):
        raise ValueError("Incomplete TTS asset bindings")
    for name, item in manifest["assets"].items():
        if sha(item["path"]) != spec["files"][name]:
            raise ValueError("Changed TTS model/voice asset")
    planned = {row["audio_id"]: row for row in plan["audio"]}
    for row in manifest["records"]:
        item = planned[row["audio_id"]]
        if (row.get("voice") != item["id"]
                or row.get("speed", spec.get("speed", 1.0))
                != item.get("speed", spec.get("speed", 1.0))
                or ("speeds" in spec and row.get("spoken_text") != item["text"])):
            raise ValueError("Changed generated voice/speed/text identity")
        if row["status"] != "generated" or sha(row["path"]) != row["sha256"]:
            raise ValueError("Missing/failed/changed generated waveform")
        if sha(row["native"]["path"]) != row["native"]["sha256"]:
            raise ValueError("Changed native waveform")
        if "speeds" in spec and "native_array" not in row:
            raise ValueError("Missing original native synthesis array")
        if ("native_array" in row
                and sha(row["native_array"]["path"]) != row["native_array"]["sha256"]):
            raise ValueError("Changed native synthesis array")
        pcm, rate = sf.read(row["path"], dtype="float32")
        if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row["samples"]
                or hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
                != row["pcm_sha256"]):
            raise ValueError("Decoded PCM identity mismatch")
    return manifest, plan


def seal(manifest_path, review_path, bundle, parent_selection, output):
    sys.path.insert(0, str(ROOT))
    from scripts.runtime.vcm_intent_slots_runtime import IntentSlotsRuntime
    from scripts.test.vcm_intent_slots_eval import (
        freeze_selection, verify_evaluation_context, verify_selection,
    )

    manifest, plan = checked_manifest(manifest_path)
    review_complete(manifest["records"], read_json(review_path))
    runtime = IntentSlotsRuntime(bundle, "ncnn", 1)
    parent = verify_selection(parent_selection)
    verify_evaluation_context(parent, runtime,
                              parent["files"]["manifest"]["path"],
                              parent["files"]["protocol"]["path"])
    if sha(runtime.root / "contract.json") != plan["contract"]["sha256"]:
        raise ValueError("Planned catalog differs from the selected candidate")
    files = {"protocol": plan["protocol"]["path"], "manifest": manifest_path,
             "checkpoint": runtime.root / "reference.pt",
             "contract": runtime.root / "contract.json",
             "catalog": runtime.root / "catalog.json",
             "normalization": runtime.root / "normalization.json",
             "calibration": runtime.root / "policy.json",
             "frontend": ROOT / "vcm/frontend.py",
             "runtime": runtime.root / "recipe.json",
             "parent_selection": parent_selection,
             "quality_review": review_path,
             "plan": manifest["plan"]["path"], "tts_runner": __file__}
    freeze_selection(files, cache_path(output))
    return {"status": "sealed_not_evaluated",
            "audio": len(manifest["records"])}


def replay(lock_path, bundle, engine, output):
    sys.path.insert(0, str(ROOT))
    from scripts.runtime.speaker_worker import AudioService
    from scripts.runtime.vcm_intent_slots_runtime import IntentSlotsRuntime
    from scripts.test.vcm_intent_slots_eval import (
        verify_evaluation_context, verify_selection,
    )

    lock = verify_selection(lock_path)
    if lock["files"].get("tts_runner", {}).get("sha256") != sha(__file__):
        raise ValueError("Replay implementation is not locked")
    path = lock["files"]["manifest"]["path"]
    manifest, plan = checked_manifest(path)
    review_complete(manifest["records"], read_json(
        lock["files"]["quality_review"]["path"]))
    runtime = IntentSlotsRuntime(bundle, engine, 1)
    verify_evaluation_context(lock, runtime, path,
                              lock["files"]["protocol"]["path"])
    if getattr(runtime, "action_free", False) is not True:
        raise ValueError("Action-free runtime required")
    output = cache_path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, events = [], []
    with (output / "events.jsonl").open("x", encoding="utf-8") as log:
        def emit(event):
            events.append(event)
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

        service = AudioService(runtime, emit)
        for row in manifest["records"]:
            events.clear()
            result = {"audio_id": row["audio_id"]}
            try:
                # Only WAV bytes enter the application, never reference labels.
                service.wav(Path(row["path"]).read_bytes(), wake=False)
                commands = [e for e in events if e.get("event") == "command"]
                if len(commands) != 1 or commands[0].get("actions_enabled"):
                    raise ValueError("Missing/multiple command or actions")
                event = commands[0]
                app = event["final_application"]
                if app != event["application"] or app != event["output"]:
                    raise ValueError("Non-authoritative application decision")
                result.update(final_application=app,
                              benchmark=event.get("benchmark"),
                              timing=event.get("timing"))
            except Exception as error:
                result["fault"] = type(error).__name__ + ": " + str(error)
            results.append(result)
        service.stop(announce=False)
    write_json(output / "predictions.json", results)
    metrics = score_cases(plan["trials"], results)
    metrics.update(lock=binding(lock_path), engine=engine,
                   checkpoint_sha256=runtime.recipe[
                       "checkpoint_source_sha256"],
                   calibration_id=runtime.policy["calibration_id"],
                   track="synthetic_command_wav_service_not_room_playback",
                   actions_enabled=False)
    write_json(output / "metrics.json", metrics)
    export_missed_verbatims(output / "missed-verbatims.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "fetch", "generate",
                                            "seal", "replay"])
    for name in ("contract", "plan", "assets", "output", "manifest",
                 "review", "bundle", "parent-selection", "lock"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--engine", choices=["ncnn", "onnxruntime", "pytorch"],
                        default="ncnn")
    parser.add_argument("--protocol", type=Path, default=SPEC,
                        help="Protocol for plan/fetch; generation/replay use the bound plan")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu",
                        help="TTS generation provider; CUDA requires a working GPU ORT session")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1,
                        help="Independent spawned TTS worker processes (default: 1)")
    args = parser.parse_args()
    required = {"plan": ("contract", "output", "protocol"),
                "fetch": ("assets", "protocol"),
                "generate": ("plan", "assets", "output", "provider", "workers"),
                "seal": ("manifest", "review", "bundle", "parent_selection",
                         "output"),
                "replay": ("lock", "bundle", "engine", "output")}
    values = [getattr(args, name) for name in required[args.command]]
    if any(value is None for value in values):
        parser.error("Missing fields: " + ", ".join(required[args.command]))
    functions = {"plan": make_plan, "fetch": fetch_assets,
                 "generate": generate,
                 "seal": seal, "replay": replay}
    try:
        result = functions[args.command](*values)
        print(json.dumps(result, allow_nan=False))
        return 0 if result.get("pass", True) else 2
    except Exception as error:
        print(json.dumps({"status": "blocked", "error": type(error).__name__,
                          "reason": str(error), "acoustic_pass": False}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
