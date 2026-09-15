"""Extra locked-threshold tests for omitted intent classes and cancellations."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyvcm.audio import load_audio
from tinyvcm.data import read_jsonl, write_json, write_jsonl
from tinyvcm.inference import Recognizer

recognizer = Recognizer(ROOT / "artifacts/tinyvcm")
rows = [r for r in read_jsonl(ROOT / "data/manifests/slurp.all.jsonl") if r["split"] == "test" and .1 <= r["duration"] <= 8]
groups = {
    "omitted_command_classes": [r for r in rows if r["command"] not in recognizer.labels],
    "explicit_negation_or_cancellation": [r for r in rows if re.search(r"\b(do not|don't|never|cancel)\b", r["transcript"].lower())],
}
summary = {"thresholds_frozen": recognizer.metadata["routing"], "selection": "All final-test source clips in each textual category; no retraining or recalibration from these tests. Negation examples can include unrelated out-of-scope requests and are reported with full transcripts.", "groups": {}}
outputs = []
for name, subset in groups.items():
    passed = early = 0
    counts = Counter()
    for row in subset:
        pcm, _ = load_audio(ROOT / row["audio"])
        p = recognizer.classify(pcm)
        predicted = recognizer.labels[int(p.argmax())]
        final_pass = predicted != "UNKNOWN" and p.max() >= recognizer.metadata["routing"]["acceptance_threshold"]
        events = recognizer.stream_events(pcm)
        passed += final_pass
        early += bool(events)
        counts[row["command"]] += 1
        outputs.append({"group": name, "id": row["id"], "transcript": row["transcript"], "original_command": row["command"], "predicted": predicted, "confidence": float(p.max()), "passed_final_intent_threshold": bool(final_pass), "early_events": events})
    summary["groups"][name] = {"samples": len(subset), "source_classes": dict(counts), "passed_final_intent_threshold": int(passed), "final_intent_false_accept_rate": int(passed)/len(subset) if subset else None, "early_triggers": int(early)}
write_json(ROOT / "reports/open_set_metrics.json", summary)
write_jsonl(ROOT / "reports/predictions_open_set.jsonl", outputs)
print(json.dumps(summary, indent=2))
