"""Versioned semantic curation from public, human-transcribed recordings."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ["UNKNOWN", "POWER_ON", "POWER_OFF", "BRIGHTNESS_SET", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN", "COLOR_SET", "VOLUME_SET", "MEDIA_PLAY", "MEDIA_PAUSE", "MEDIA_RESUME", "MEDIA_STOP", "TIME_QUERY", "DATE_QUERY", "TIMER_SET"]
MAPPING_VERSION = "2026-09-15-v1"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_number(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:12], 16)


def normalize_text(text):
    text = text.lower().replace("’", "'")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", text)).strip()


def command_label(text, source_intent=""):
    """Conservative, inspectable curation. This is never an inference-time input."""
    t = normalize_text(text)
    if re.search(r"\b(don't|do not|never|not|instead|unless)\b", t):
        return "UNKNOWN"
    if re.search(r"\b(set|start|create)\b.*\btimer\b", t) and "alarm" not in t:
        return "TIMER_SET"
    if re.search(r"\b(pause)\b.*\b(music|song|track|playback)\b", t):
        return "MEDIA_PAUSE"
    if re.search(r"\b(resume|continue)\b.*\b(music|song|track|playback)\b", t):
        return "MEDIA_RESUME"
    if re.search(r"\bstop\b.*\b(music|song|track|playback)\b", t):
        return "MEDIA_STOP"
    light = bool(re.search(r"\b(light|lights|lamp|lamps|brightness)\b", t))
    power_target = light or bool(re.search(r"\b(plug|socket|switch|fan|wemo)\b", t))
    if power_target and re.search(r"\b(turn|switch|put)\b", t):
        on, off = bool(re.search(r"\bon\b", t)), bool(re.search(r"\boff\b", t))
        if on != off:
            return "POWER_ON" if on else "POWER_OFF"
    if light and re.search(r"\b(colou?r|red|orange|yellow|green|blue|purple|pink|cyan|white)\b", t):
        return "COLOR_SET"
    if light and ("percent" in t and re.search(r"\b(to|at)\b", t) or "set" in t and "brightness" in t):
        return "BRIGHTNESS_SET"
    if light and re.search(r"\b(dim|dimmer|dimming|reduce|lower|down|less)\b", t):
        return "BRIGHTNESS_DOWN"
    if light and re.search(r"\b(brighten|brighter|increase|raise|up)\b", t):
        return "BRIGHTNESS_UP"
    if power_target:
        on, off = bool(re.search(r"\bon\b", t)), bool(re.search(r"\boff\b", t))
        if on != off:
            return "POWER_ON" if on else "POWER_OFF"
    if "volume" in t and re.search(r"\b(set|to|at)\b", t) and re.search(r"\b(percent|zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b", t):
        return "VOLUME_SET"
    if (source_intent in {"play_music", "music"} or not source_intent) and re.search(r"\b(play|listen)\b", t):
        if not re.search(r"\b(radio|podcast|audiobook|book|game|video)\b", t):
            return "MEDIA_PLAY"
    if re.search(r"\b(what|tell|give)\b.*\b(date|day)\b", t) and not re.search(r"\b(weather|alarm|calendar|event|holiday|christmas|birthday|tomorrow|yesterday)\b", t):
        return "DATE_QUERY"
    if re.search(r"\b(what|tell|give)\b.*\btime\b", t) and not re.search(r"\b(alarm|timer|zone|flight|train|meeting|until|ahead|difference|tomorrow|yesterday)\b", t):
        return "TIME_QUERY"
    return "UNKNOWN"


def prepare_slurp():
    raw = ROOT / "data/raw/slurp"
    metadata = json.loads((raw / "metadata.json").read_text(encoding="utf-8"))
    audio_paths = {p.name: p for p in (raw / "slurp_real").rglob("*.flac")}
    if not audio_paths:
        raise RuntimeError("SLURP real audio has not been extracted")
    import soundfile as sf
    rows, excluded = [], []
    for source_split, name in [("train", "train.jsonl"), ("dev", "devel.jsonl"), ("test", "test.jsonl")]:
        for annotation in read_jsonl(raw / name):
            sid = str(annotation["slurp_id"])
            text = annotation["sentence"]
            intent = annotation["intent"]
            label = command_label(text, intent)
            # Split calibration by complete source prompt, not individual microphones.
            split = "calibration" if source_split == "dev" and stable_number(sid) % 4 == 0 else source_split
            for recording in annotation["recordings"]:
                filename = recording["file"]
                path = audio_paths.get(filename)
                if path is None:
                    excluded.append({"id": filename, "reason": "missing_audio"})
                    continue
                meta = metadata.get(sid, {}).get("recordings", {}).get(filename, {})
                info = sf.info(path)
                if not 0.35 <= info.duration <= 16:
                    excluded.append({"id": filename, "reason": "duration_out_of_0.35_16s", "duration": info.duration})
                    continue
                rows.append({"id": "slurp/" + filename, "audio": path.relative_to(ROOT).as_posix(), "source": "SLURP-real", "source_id": sid, "source_split": source_split, "split": split, "speaker_id": "slurp/" + str(meta.get("usrid", "unknown")), "paired_group": "slurp/" + str(meta.get("recid", filename.replace("-headset", ""))), "transcript": text, "source_intent": intent, "command": label, "entities": annotation["entities"], "tokens": annotation["tokens"], "duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels, "license": "CC-BY-NC-4.0", "timing_verified": False, "mapping_version": MAPPING_VERSION})
    write_jsonl(ROOT / "data/manifests/slurp.all.jsonl", rows)
    write_json(ROOT / "reports/slurp_exclusions.json", excluded)
    return rows


def prepare_libri():
    folder = ROOT / "data/raw/librispeech/LibriSpeech"
    import soundfile as sf
    rows = []
    for subset, split in [("train-clean-100", "train"), ("dev-clean", "dev"), ("test-clean", "test")]:
        for transcript_file in sorted((folder / subset).rglob("*.trans.txt")):
            for line in transcript_file.read_text(encoding="utf-8").splitlines():
                rid, text = line.split(" ", 1)
                path = transcript_file.parent / (rid + ".flac")
                info = sf.info(path)
                rows.append({"id": "libri/" + rid, "audio": path.relative_to(ROOT).as_posix(), "source": "LibriSpeech", "source_id": rid, "source_split": subset, "split": split, "speaker_id": "libri/" + rid.split("-")[0], "paired_group": "libri/" + rid, "transcript": text.lower(), "duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels, "license": "CC-BY-4.0", "timing_verified": False})
    write_jsonl(ROOT / "data/manifests/libri.all.jsonl", rows)
    return rows


def prepare_task_manifests():
    original_path = ROOT / "data/manifests/slurp.source_records.jsonl"
    current_path = ROOT / "data/manifests/slurp.all.jsonl"
    if not original_path.exists():
        original_path.write_bytes(current_path.read_bytes())
    original = read_jsonl(original_path)
    groups, ids = defaultdict(list), defaultdict(list)
    for r in original:
        groups[r["paired_group"]].append(r)
        ids[r["id"]].append(r)
    leaking = {key for key, records in groups.items() if len({r["source_split"] for r in records}) > 1}
    conflicting = {key for key, records in ids.items() if len({r["transcript"] for r in records}) > 1}
    slurp = [r for r in original if r["paired_group"] not in leaking and r["id"] not in conflicting]
    slurp = list({r["id"]: r for r in slurp}.values())
    # Connect corrected prompt IDs sharing a microphone pair before calibrating.
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    for r in slurp:
        if r["source_split"] == "dev":
            a, b = find("prompt/" + r["source_id"]), find(r["paired_group"])
            parent[max(a, b)] = min(a, b)
    for r in slurp:
        if r["source_split"] == "dev":
            r["split"] = "calibration" if stable_number(find("prompt/" + r["source_id"])) % 4 == 0 else "dev"
    write_jsonl(current_path, slurp)
    write_json(ROOT / "reports/leakage_quarantine.json", {"paired_groups_crossing_official_splits": sorted(leaking), "conflicting_transcript_recording_ids": sorted(conflicting), "removed_records": len(original)-len(slurp), "original_records": len(original), "retained_records": len(slurp), "policy": "Remove all views of leaking pairs; preserve original source splits on retained rows."})
    counts = {s: Counter(r["command"] for r in slurp if r["split"] == s and r["duration"] <= 8) for s in ["train", "dev", "calibration", "test"]}
    # Class inclusion uses train and development/calibration support, never test performance.
    supported = [c for c in COMMANDS if c == "UNKNOWN" or counts["train"][c] >= 40 and counts["dev"][c] + counts["calibration"][c] >= 10]
    vcm = []
    for split in counts:
        eligible = [r for r in slurp if r["split"] == split and r["duration"] <= 8 and r["command"] in supported]
        positives = [r for r in eligible if r["command"] != "UNKNOWN"]
        negatives = sorted((r for r in eligible if r["command"] == "UNKNOWN"), key=lambda r: stable_number(r["id"]))
        # Freeze a documented cohort; report its class distribution rather than all-corpus accuracy.
        limit = max(1000, len(positives)) if split == "train" else max(500, len(positives))
        vcm.extend(positives + negatives[:limit])
    write_jsonl(ROOT / "data/manifests/vcm.jsonl", vcm)
    write_json(ROOT / "data/manifests/labels.json", supported)
    coverage = {c: {**{s: counts[s][c] for s in counts}, "trained": c in supported, "early_timing_verified": False} for c in COMMANDS}
    write_json(ROOT / "reports/coverage.json", coverage)
    speakers = {s: set(r["speaker_id"] for r in slurp if r["source_split"] == s) for s in ["train", "dev", "test"]}
    paired = {s: set(r["paired_group"] for r in slurp if r["source_split"] == s) for s in speakers}
    overlaps = {f"{a}:{b}": {"speakers": sorted(speakers[a] & speakers[b]), "paired_recordings": sorted(paired[a] & paired[b])} for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]}
    if any(v["paired_recordings"] for v in overlaps.values()):
        raise RuntimeError("Paired recording leakage across official source splits")
    write_json(ROOT / "reports/split_audit.json", {"official_split_overlaps": overlaps, "vcm_cohort_counts": {s: dict(Counter(r["command"] for r in vcm if r["split"] == s)) for s in counts}, "note": "Official source splits retained; speaker overlaps disclosed rather than relabeled speaker-disjoint."})
    annotation_queue = sorted((r for r in vcm if r["command"] != "UNKNOWN"), key=lambda r: stable_number(r["id"]))[:500]
    write_jsonl(ROOT / "data/annotations/review_queue.jsonl", [{"id": r["id"], "audio": r["audio"], "transcript": r["transcript"], "command": r["command"], "split": r["split"], "intent_ready_ms": None, "parameter_start_ms": None, "parameter_transcript": None, "verified": False} for r in annotation_queue])
    print(json.dumps({"vcm_examples": len(vcm), "labels": supported, "coverage": coverage}, indent=2), flush=True)
    return vcm


def prepare_asr_manifests():
    libri = read_jsonl(ROOT / "data/manifests/libri.all.jsonl")
    slurp = read_jsonl(ROOT / "data/manifests/slurp.all.jsonl")
    # Original transcripts are preserved; unverified token spans are never cut into pseudo-ground-truth tails.
    rows = [r for r in libri if r["duration"] <= 20]
    rows += [r for r in slurp if r["split"] in {"train", "dev", "test"} and r["duration"] <= 12]
    # Digits may be read as a number, a digit sequence, or an ordinal. Do not erase
    # them or invent a spoken form when no verified normalization is supplied.
    rejected = [r for r in rows if re.search(r"\d", r["transcript"]) or not normalize_text(r["transcript"])]
    rows = [r for r in rows if not re.search(r"\d", r["transcript"]) and normalize_text(r["transcript"])]
    write_json(ROOT / "reports/asr_transcript_exclusions.json", [{"id": r["id"], "reason": "unverified_numeric_or_empty_normalization", "transcript": r["transcript"]} for r in rejected])
    write_jsonl(ROOT / "data/manifests/asr.jsonl", rows)
    return rows
