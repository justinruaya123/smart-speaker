"""Six-voice, evaluation-only speech generation and bound VCM service replay.

Never trains, changes thresholds, enables actions, or certifies Pi performance.
Expected text and commands are used by synthesis and scoring, not inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name("tts-protocol.json")


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


def trial_id(voice, text):
    return hashlib.sha256(canonical([voice, text]).encode()).hexdigest()


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
            meaning = canonical(row["command"])
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


def _metrics(trials, outputs):
    result = {key: 0 for key in (
        "trials", "supported", "negative", "accepted", "correct_positive",
        "correct_rejections", "strict_correct", "false_accepts", "faults",
        "missing")}
    slots = defaultdict(lambda: {"n": 0, "correct_accepted": 0})
    for trial in trials:
        result["trials"] += 1
        expected = trial["command"]
        positive = expected is not None
        result["supported" if positive else "negative"] += 1
        for field in (expected or {}).get("slots", {}):
            slots[field]["n"] += 1
        output = outputs.get(trial["audio_id"])
        if output is None:
            result["missing"] += 1
            continue
        if output.get("fault"):
            result["faults"] += 1
            continue
        app = output.get("final_application", {})
        accepted, actual = app.get("accepted"), app.get("command")
        if (type(accepted) is not bool or accepted != (actual is not None)
                or app.get("reason") in ("capture_fault", "unqualified_model")
                or (accepted and (
                    not isinstance(actual, dict)
                    or not isinstance(actual.get("slots"), dict)))):
            result["faults"] += 1
            continue
        result["accepted"] += int(accepted)
        if positive:
            correct = accepted and canonical(actual) == canonical(expected)
            result["correct_positive"] += int(correct)
            for field, value in expected.get("slots", {}).items():
                matched = (
                    accepted and actual.get("intent") == expected["intent"]
                    and canonical(actual.get("slots", {}).get(field))
                    == canonical(value))
                slots[field]["correct_accepted"] += int(matched)
        else:
            result["false_accepts"] += int(accepted)
            correct = not accepted and bool(app.get("reason"))
            if trial.get("reason") is not None:
                correct = correct and app.get("reason") == trial["reason"]
            result["correct_rejections"] += int(correct)
        result["strict_correct"] += int(correct)
    result["recall"] = (result["correct_positive"] / result["supported"]
                        if result["supported"] else None)
    result["precision"] = (result["correct_positive"] / result["accepted"]
                           if result["accepted"] else None)
    result["false_accept_rate"] = (result["false_accepts"] / result["negative"]
                                   if result["negative"] else None)
    result["supplied_slots"] = dict(slots)
    return result


def score_cases(trials, results):
    indexed = {row["audio_id"]: row for row in results}
    expected_ids = {row["audio_id"] for row in trials}
    if len(indexed) != len(results) or set(indexed) - expected_ids:
        raise ValueError("Duplicate/unexpected inference output")
    unique, voices, tracks = {}, defaultdict(dict), defaultdict(list)
    for row in trials:
        unique.setdefault(row["audio_id"], row)
        voices[row["voice"]].setdefault(row["audio_id"], row)
        tracks[row["track"]].append(row)
    intents, cases = defaultdict(list), defaultdict(list)
    for row in unique.values():
        intents[(row["command"] or {}).get("intent", "OOV")].append(row)
    for row in trials:
        cases[row["case_id"]].append(row)
    total = _metrics(list(unique.values()), indexed)
    track_metrics = {key: _metrics(value, indexed)
                     for key, value in tracks.items()}
    return {
        "case_references": len(trials), "unique_audio": total,
        "per_voice": {key: _metrics(list(value.values()), indexed)
                      for key, value in voices.items()},
        "per_track": track_metrics,
        "per_intent": {key: _metrics(value, indexed)
                       for key, value in intents.items()},
        "per_case": {key: _metrics(value, indexed)
                     for key, value in cases.items()},
        "pass": bool(total["supported"] and all(
            item["strict_correct"] == item["trials"]
            for item in [total, *track_metrics.values()])),
        "human_generalization_established": False,
        "ownership_established": False, "pi_evidence": False,
    }


def make_plan(contract_path, output):
    sys.path.insert(0, str(ROOT))
    from scripts.runtime.pcv_schema import baseline_cases, validate_command
    from scripts.runtime.vcm_intent_slots_contract import verify_contract

    contract, spec = read_json(contract_path), read_json(SPEC)
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
    cases, audios, trials = merge_cases(contract), {}, []
    for case in cases:
        if case["command"] is not None:
            validate_command(case["command"])
        for voice in spec["voices"]:
            identifier = trial_id(voice["id"], case["spoken_text"])
            audios[identifier] = {"audio_id": identifier,
                                  "text": case["spoken_text"], **voice}
            trials.append({**case, "audio_id": identifier,
                           "voice": voice["id"]})
    value = {"format": spec["experiment_id"],
             "contract": binding(contract_path), "protocol": binding(SPEC),
             "source_counts": {key: len(contract[key])
                               for key in ("cases", "catalog_entries")},
             "optionb_trials": 93 * len(spec["voices"]),
             "audio": list(audios.values()), "trials": trials,
             "fit_allowed": False, "enrollment_allowed": False}
    write_json(cache_path(output), value)
    return {"status": "planned_not_generated", "audio_n": len(audios),
            "case_trials": len(trials),
            "optionb_trials": value["optionb_trials"]}


def fetch_assets(directory):
    from huggingface_hub import hf_hub_download

    spec = read_json(SPEC)
    directory = cache_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in spec["files"]:
        hf_hub_download(repo_id=spec["repository"], filename=filename,
                        revision=spec["revision"], local_dir=str(directory))
    files = asset_paths(directory, spec)
    receipt = {"repository": spec["repository"], "revision": spec["revision"],
               "protocol": binding(SPEC),
               "files": {key: binding(path) for key, path in files.items()}}
    destination = directory / "asset-receipt.json"
    if destination.exists():
        if read_json(destination) != receipt:
            raise ValueError("Existing asset receipt changed")
    else:
        write_json(destination, receipt)
    return {"status": "downloaded_hash_verified", "files": len(files)}


def generate(plan_path, assets, output):
    import onnxruntime as ort
    import soundfile as sf
    from kokoro_onnx import Kokoro

    plan = read_json(plan_path)
    spec = bound_json(plan["protocol"])
    bound_json(plan["contract"])
    version = importlib.metadata.version("kokoro-onnx")
    if version != spec["kokoro_onnx_version"]:
        raise ValueError("Use the frozen TTS package version")
    files = asset_paths(assets, spec)
    output = cache_path(output)
    output.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = spec["threads"]
    options.inter_op_num_threads = 1
    ort.set_seed(spec["seed"])
    session = ort.InferenceSession(str(files["kokoro-v1.0.onnx"]), options,
                                   providers=["CPUExecutionProvider"])
    tts = Kokoro.from_session(session, str(files["voices-v1.0.bin"]))
    if not {v["id"] for v in spec["voices"]} <= set(tts.voices):
        raise ValueError("Selected six distinct voice packs are unavailable")
    records = []
    (output / "audio").mkdir()
    (output / "native").mkdir()
    with (output / "generation.jsonl").open("x", encoding="utf-8") as log:
        for item in plan["audio"]:
            row = {"audio_id": item["audio_id"], "voice": item["id"],
                   "kind": "synthetic", "status": "generation_fault"}
            try:
                phonemes = tts.tokenizer.phonemize(item["text"], item["lang"])
                if not phonemes or len(phonemes) > 490:
                    raise ValueError("Empty/overlong TTS phonemes")
                audio, rate = tts.create(item["text"], voice=item["id"],
                                         lang=item["lang"], speed=1.0,
                                         trim=False)
                pcm = to_pcm16(audio, rate)
                native = output / "native" / (item["audio_id"] + ".wav")
                target = output / "audio" / (item["audio_id"] + ".wav")
                sf.write(native, audio, rate, subtype="FLOAT")
                sf.write(target, pcm, 16000, subtype="PCM_16")
                decoded, actual_rate = sf.read(target, dtype="float32")
                if actual_rate != 16000 or len(decoded) != len(pcm):
                    raise ValueError("Written WAV format mismatch")
                row.update(status="generated", path=str(target),
                           sha256=sha(target), native=binding(native),
                           pcm_sha256=hashlib.sha256(
                               decoded.astype("<f4").tobytes()).hexdigest(),
                           samples=len(decoded), sample_rate=16000)
            except Exception as error:
                row["fault"] = type(error).__name__ + ": " + str(error)
            records.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    versions = {name: importlib.metadata.version(name) for name in (
        "kokoro-onnx", "onnxruntime", "numpy", "scipy", "soundfile",
        "espeakng-loader", "phonemizer-fork")}
    manifest = {"plan": binding(plan_path), "records": records,
                "assets": {key: binding(path) for key, path in files.items()},
                "versions": versions, "python": sys.version,
                "platform": platform.platform(), "source": binding(__file__),
                "seed": spec["seed"], "providers": session.get_providers(),
                "fit_allowed": False, "human_reviewed": False}
    write_json(output / "manifest.json", manifest)
    write_json(output / "review-template.json", {
        "blind_to_vcm": True, "items": [{
            "audio_id": row["audio_id"], "sha256": row.get("sha256"),
            "heard_exact": None, "quality": None, "reviewer": "",
            "reviewed_at": "", "notes": ""} for row in records]})
    faults = sum(row["status"] != "generated" for row in records)
    return {"status": "generated_needs_listening", "generated": len(records),
            "generation_faults": faults, "pass": faults == 0}


def checked_manifest(path):
    import soundfile as sf

    manifest = read_json(path)
    plan = bound_json(manifest["plan"])
    bound_json(plan["contract"])
    spec = bound_json(plan["protocol"])
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
    for row in manifest["records"]:
        if row["status"] != "generated" or sha(row["path"]) != row["sha256"]:
            raise ValueError("Missing/failed/changed generated waveform")
        if sha(row["native"]["path"]) != row["native"]["sha256"]:
            raise ValueError("Changed native waveform")
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
    args = parser.parse_args()
    required = {"plan": ("contract", "output"), "fetch": ("assets",),
                "generate": ("plan", "assets", "output"),
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
