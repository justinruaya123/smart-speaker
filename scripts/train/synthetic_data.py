"""Explicitly synthetic, fixed-command pilot; immutable human holdouts."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import re

from .family_contract import LABELS
from .levels import spoken_number, NUMBER_WORDS
from .routing import COLORS, parse_parameters

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/tts"
CACHE = ROOT / ".cache/synthetic-v1"
SYN = ROOT / "experiments/synthetic"
MIX = ROOT / "experiments/mixed"
HEADS = {"intent": LABELS, "ready": ["REJECT", "READY"],
         "operation": ["NONE", "SET", "INCREASE", "DECREASE"],
         "amount": ["NONE", *map(str, range(101))], "color": ["NONE", *sorted(COLORS)]}
WORDS = sorted(NUMBER_WORDS | COLORS | set("on off brightness volume light lights lamp lamps up down increase decrease raise lower brighten dim louder quieter play resume continue time date percent music".split()))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    temp.replace(path)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]


def write_rows(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf8")


def fraction(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16) / 16**12


def prepare_features(data):
    import numpy as np
    from .audio import cached_features
    def load(record):
        array=cached_features(record,ROOT)
        if array.shape[0]!=40 or not np.isfinite(array).all():raise ValueError("Invalid features")
        return array
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(load,data))


def catalog():
    c = json.loads((ROOT / "artifacts/bundle.json").read_text())["catalog"]
    return {**c, "entities": [e for e in c["entities"] if e["acoustic_key"] == "main_lights"]}


def label(command, text):
    """Silver labels reuse the existing strict grammar; never run on predictions."""
    parsed = parse_parameters(command, text, catalog())
    args = parsed.get("command", {}).get("arguments", {})
    if command == "MEDIA_PLAY" and args:
        return None  # Names and open media selections are outside this study.
    target = args.get("target")
    if target and target.get("id") != "example.main_lights":
        return None
    ready = parsed["status"] == "accepted" and command != "UNKNOWN"
    output = {"intent": command, "ready": "READY" if ready else "REJECT",
              "operation": "NONE", "amount": "NONE", "color": "NONE"}
    if ready:
        if "percent" in args:
            output.update(operation="SET", amount=str(args["percent"]))
        elif "delta_percentage_points" in args:
            delta = args["delta_percentage_points"]
            output.update(operation="INCREASE" if delta >= 0 else "DECREASE", amount=str(abs(delta)))
        if "color" in args:
            output["color"] = args["color"]
    return output


def annotate_text(text):
    spans = [{"start": m.start(), "end": m.end(), "text": m.group(),
              "role": "number" if m.group().lower() in NUMBER_WORDS or m.group().isdigit() else "keyword"}
             for m in re.finditer(r"[a-zA-Z]+|\d+", text) if m.group().lower() in WORDS or m.group().isdigit()]
    highlighted = text
    for s in reversed(spans):
        highlighted = highlighted[:s["start"]] + "**" + highlighted[s["start"]:s["end"]] + "**" + highlighted[s["end"]:]
    return {"keyword_spans": spans, "highlighted_text": highlighted, "timing_verified": False}


def voices():
    specifications = [
        ("train", "en-US", "A young adult woman with a clear American English accent, medium pitch and conversational pace."),
        ("train", "en-US", "A middle aged man with a low warm voice, general American English accent, measured pace."),
        ("train", "en-GB", "An adult British English woman, light husky voice, clear relaxed conversational delivery."),
        ("train", "en-AU", "An adult Australian English man, mid pitch, brisk but clear delivery."),
        ("train", "en-PH-requested", "A young adult Filipina woman speaking English with a natural Filipino accent, clear vowels, warm medium pitch and everyday conversational pace."),
        ("train", "en-PH-requested", "An adult Filipino man speaking English with a natural Manila Filipino accent, low resonant voice, measured relaxed pace."),
        ("train", "en-PH-requested", "A middle aged Filipina woman speaking English with a noticeable Philippine English accent, bright clear voice, slightly brisk conversational pace."),
        ("train", "en-PH-requested", "A young Filipino man speaking fluent English with a Filipino accent, soft mid-range voice, relaxed everyday delivery."),
        ("dev", "en-US", "An older woman with a gentle American English accent, slightly textured low voice and steady delivery."),
        ("dev", "en-PH-requested", "An older Filipino man speaking English with a Filipino accent, slightly gravelly medium voice and clear steady delivery."),
        ("test", "en-GB", "A young British English man with a bright high baritone, casual clear natural delivery."),
        ("test", "en-PH-requested", "An adult Filipina woman speaking English with a natural Filipino accent, low smooth voice and careful conversational delivery."),
    ]
    return [{"id": f"designed-{i:02d}", "split": split, "accent_requested": accent,
             "prompt": prompt + " Speak only the supplied English text. Clean dry recording without music.",
             "seed": 1700 + i, "accent_verified_by_human": False, "identity_source": "synthetic_voice_design"}
            for i, (split, accent, prompt) in enumerate(specifications)]


def texts(voice_index):
    fixed = {
        "POWER_ON": ["turn on the lights", "switch the lights on", "please turn the light on", "turn the lamp on", "switch on the lamps", "lights on"],
        "POWER_OFF": ["turn off the lights", "switch the lights off", "please turn the light off", "turn the lamp off", "switch off the lamps", "lights off"],
        "MEDIA_PLAY": ["play music", "please play music", "play some music", "start the music", "can you play music", "put on the music"],
        "MEDIA_RESUME": ["resume music", "resume playback", "continue playing", "please resume", "resume the music", "continue the music"],
        "TIME_QUERY": ["what time is it", "tell me the time", "what is the current time", "please tell me the time", "what's the time now", "give me the time"],
        "DATE_QUERY": ["what is the date", "tell me today's date", "what day is it today", "please tell me the date", "what is today's date", "give me the date"],
    }
    result = [(c, t) for c, values in fixed.items() for t in values]
    for color in sorted(COLORS):
        result += [("COLOR_SET", f"set the lights to {color}"), ("COLOR_SET", f"make the light {color}")]
    templates = ["set {family} to {number} percent", "{family} {number}", "please set the {family} to {number} percent", "change the {family} to {number}"]
    for family in ("BRIGHTNESS", "VOLUME"):
        for value in range(101):
            result.append((family, templates[(voice_index + value) % len(templates)].format(family=family.lower(), number=spoken_number(value))))
        for text in ["{f} up", "{f} down", "increase the {f}", "decrease the {f}", "raise the {f}", "lower the {f}"]:
            result.append((family, text.format(f=family.lower())))
        for value in (1, 10, 25, 50, 100):
            for direction in ("increase", "decrease"):
                result.append((family, f"{direction} the {family.lower()} by {spoken_number(value)} percent"))
        for invalid in ["minus five", "one hundred and one", "fifty point five", "twenty thirty", "fifty or sixty", "half"]:
            result.append((family, f"set {family.lower()} to {invalid} percent"))
        result += [(family, f"{family.lower()} up and down"), (family, f"set {family.lower()} percent")]
    unknown = ["hello there", "thank you very much", "how are you today", "what is the weather", "set an alarm", "tell me a joke", "cancel the timer", "open the door", "close the window", "please wait", "stop listening", "never mind", "turn on the television", "turn off the fan", "the light is broken", "I bought a blue shirt", "one two three four", "I will arrive at fifty past", "today was a good day", "I don't want anything", "do not turn on the lights", "do not play music", "set brightness to fifty and volume to twenty", "turn the lights on then off", "make the lights red and blue", "set the lights to blue tomorrow", "turn the lights off in ten minutes", "what does brightness mean", "volume is a mathematical quantity", "we discussed music yesterday", "please repeat that", "this is only a test"]
    result += [("UNKNOWN", t) for t in unknown]
    return result


def excluded_name(row):
    named_types = {"house_place", "place_name", "artist_name", "song_name", "playlist_name", "audiobook_name", "audiobook_author", "podcast_name", "radio_name", "person", "business_name", "time_zone"}
    if any(e.get("type") in named_types for e in row.get("entities", [])):
        return True
    return bool(re.search(r"\b(bed ?room|kitchen|bathroom|living room|lounge|office|hallway|garage|washroom|wemo)\b", row["transcript"], re.I))


def prepare_human():
    source = rows(ROOT / "data/vcm.jsonl")
    # Preserve previous held-out speakers, including the supplemental numeric holdout.
    historical = rows(ROOT / "data/phonetic-v2/manifest.jsonl")
    ranks = {"train": 0, "enroll": 0, "dev": 1, "calibration": 2, "test": 3}
    split_by_speaker = {}
    for r in source + [{"speaker_id": x["speaker"], "split": x["split"]} for x in historical]:
        speaker, split = r["speaker_id"], r["split"]
        if ranks[split] > ranks.get(split_by_speaker.get(speaker, "train"), 0):
            split_by_speaker[speaker] = split
        else:
            split_by_speaker.setdefault(speaker, "train")
    # Use all original SmartLights numeric commands, one close microphone per utterance.
    existing = {r["audio"] for r in source}
    for r in rows(ROOT / "data/asr.jsonl"):
        if r["source"] == "SnipsSmartLights" and r["audio"].endswith("_close.wav") and r["audio"] not in existing:
            source.append(r); existing.add(r["audio"])
    seen_groups = set(); selected = []; excluded = Counter()
    source.sort(key=lambda r: (r.get("paired_group", r["id"]), 0 if "headset" in r["audio"] or "_close.wav" in r["audio"] else 1, r["id"]))
    for r in source:
        group = r.get("paired_group", r["id"])
        if group in seen_groups:
            excluded["paired_recording"] += 1; continue
        seen_groups.add(group)
        if r.get("augmentation") or excluded_name(r):
            excluded["augmentation_or_named_entity"] += 1; continue
        if r.get("command") not in LABELS:
            excluded["outside_contract"] += 1; continue
        lab = label(r["command"], r["transcript"])
        if lab is None:
            excluded["open_parameter"] += 1; continue
        # Keep positive parameter data and a bounded deterministic background sample.
        if r["command"] == "UNKNOWN" and fraction("unknown/" + r["id"]) > .40:
            excluded["bounded_background"] += 1; continue
        speaker = r["speaker_id"]
        split = split_by_speaker.get(speaker, r["split"])
        if split == "calibration":
            excluded["root_calibration"] += 1; continue
        if split == "enroll": split = "train"
        if split not in ("train", "dev", "test"): raise ValueError(split)
        selected.append({**r, "split": split, "original_split": r["split"], "labels": lab,
            "kind": "human", "label_quality": "programmatic_silver", "audio_sha256": digest(ROOT / r["audio"]), **annotate_text(r["transcript"])})
    by_hash = defaultdict(list)
    for r in selected: by_hash[r["audio_sha256"]].append(r)
    final = []; conflicts = []
    for h, group in by_hash.items():
        if len({json.dumps(r["labels"], sort_keys=True) for r in group}) > 1:
            conflicts.append({"sha256": h, "ids": [r["id"] for r in group]}); continue
        chosen = max(group, key=lambda r: ranks[r["split"]])
        if len({r["split"] for r in group}) > 1:
            excluded["cross_split_duplicate"] += len(group); continue
        final.append(chosen)
    write_rows(DATA / "human.jsonl", final)
    dump(DATA / "human-audit.json", {"n": len(final), "excluded": dict(excluded), "quarantined": conflicts,
        "counts": dict(Counter(r["split"] + "/" + r["labels"]["intent"] for r in final)),
        "speaker_counts": {s: len({r["speaker_id"] for r in final if r["split"] == s}) for s in ("train", "dev", "test")},
        "labels": "Silver references from existing source labels and strict fixed grammar; not human-adjudicated.",
        "original_recordings_only": True, "manifest_sha256": digest(DATA / "human.jsonl")})
    return final


def prepare():
    for p in (DATA, CACHE, SYN, MIX): p.mkdir(parents=True, exist_ok=True)
    if (DATA / "protocol.json").exists():
        protocol = json.loads((DATA / "protocol.json").read_text())
        for name, value in protocol["manifest_sha256"].items():
            if digest(DATA / name) != value: raise ValueError("Frozen manifest changed: " + name)
        print("Existing frozen tts manifest verified", flush=True); return
    human = rows(DATA / "human.jsonl") if (DATA / "human.jsonl").exists() else prepare_human()
    generated = []
    for i, voice in enumerate(voices()):
        for j, (family, text) in enumerate(texts(i)):
            lab = label(family, text)
            if lab is None: raise ValueError("Unexpected open template: " + text)
            key = f"tts/{voice['id']}/{j:04d}"
            generated.append({"id": key, "audio": f"data/audio/{key}.wav", "transcript": text,
                "command": family, "labels": lab, "split": voice["split"], "speaker_id": "tts/" + voice["id"],
                "kind": "synthetic", "source": "Qwen3-TTS-1.7B", "license": "Apache-2.0 model; locally generated audio",
                "voice_id": voice["id"], "accent_requested": voice["accent_requested"], "accent_verified_by_human": False,
                "seed": 170000 + i * 1000 + j, "label_quality": "intended TTS text; acoustic fidelity not human-adjudicated",
                **annotate_text(text)})
    write_rows(DATA / "planned.jsonl", generated)
    dump(DATA / "voices.json", voices())
    protocol = {"version": "synthetic-fixed-v1", "dataset": "tts", "seed": 17,
        "heads": HEADS, "keyword_vocabulary": WORDS, "manifest_sha256": {f: digest(DATA / f) for f in ("planned.jsonl", "human.jsonl", "voices.json")},
        "synthetic_counts": dict(Counter(r["split"] for r in generated)), "human_counts": dict(Counter(r["split"] for r in human)),
        "scope": "Nine existing intent families plus UNKNOWN, generic lights, nine colors, brightness and volume integers 0..100 and +/-10 or explicit by amounts. No entity names or music selection.",
        "selection": "Random initialization; source-specific development chooses checkpoints. Human dev only tunes confidence and real/synthetic preference. Test scored once after selection receipt.",
        "mixed": "Two source-specific acoustic VCM experts; per-intent convex probability blend with Optuna alpha, bias and threshold. Parameter blend conditioned only on accepted VCM family.",
        "quality_targets": {"accepted_precision": .95, "unknown_false_accept_rate": .01, "minimum_dev_accepts_per_intent": 10},
        "epochs": 32, "batch_size": 48, "learning_rate": .001, "optuna_trials_per_intent": 100,
        "fixed_classes": True, "human_test_untouched_by_tuning": True, "synthetic_audio_authorized_by_user": True,
        "audio_augmentation": False, "physical_pi_measured": False, "live_microphone_measured": False,
        "synthetic_split": "Disjoint designed voice anchors; fixed corpus, not novel-word generalization. Voice identity and accent are not independently verified."}
    dump(DATA / "protocol.json", protocol)
    for folder in (SYN, MIX): dump(folder / "protocol.json", protocol)
    (SYN / "text-preview.md").write_text("# TTS label preview\n\nKeyword highlighting is textual, not an audio alignment.\n\n" + "\n".join(f"- `{r['command']}` {r['highlighted_text']} -> `{json.dumps(r['labels'])}`" for r in generated[:45]), encoding="utf8")
    print(json.dumps({"tts": len(generated), "human": len(human), **protocol["synthetic_counts"]}), flush=True)
