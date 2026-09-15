"""Evaluation-only prompt agreement and negative-command replay for ten readers."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyvcm.audio import load_audio
from tinyvcm.data import read_jsonl, write_json, write_jsonl, normalize_text
from tinyvcm.inference import Recognizer


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


recognizer = Recognizer(ROOT / "artifacts/tinyvcm", asr=True)
rows = read_jsonl(ROOT / "data/manifests/philippine_english_diagnostic.jsonl")
outputs = []
for row in rows:
    pcm, source_rate = load_audio(ROOT / row["audio"])
    hypothesis = recognizer.transcribe(pcm)
    reference = normalize_text(row["reference"])
    trials = []
    for start in range(0, len(pcm), 120000):
        wave = pcm[start:start + 120000]
        if len(wave) < 8000:
            continue
        p = recognizer.classify(wave)
        label = recognizer.labels[int(p.argmax())]
        accepted = label != "UNKNOWN" and p.max() >= recognizer.metadata["routing"]["acceptance_threshold"]
        trials.append({"start_s": start/16000, "predicted": label, "confidence": float(p.max()), "passed_final_intent_threshold": bool(accepted)})
    outputs.append({"id": row["id"], "source_url": row["url"], "duration": len(pcm)/16000, "original_sample_rate": source_rate, "hypothesis": hypothesis, "reference_type": row["reference_type"], "word_errors_vs_prompt": distance(reference.split(), hypothesis.split()), "reference_words": len(reference.split()), "negative_vcm_windows": trials})
    print(f"ACCENT TEST {row['id']}: prompt error {outputs[-1]['word_errors_vs_prompt']}/{len(reference.split())}", flush=True)
write_jsonl(ROOT / "reports/predictions_accent.jsonl", outputs)
windows = [w for row in outputs for w in row["negative_vcm_windows"]]
write_json(ROOT / "reports/accent_metrics.json", {"speakers": len(outputs), "total_audio_seconds": sum(r["duration"] for r in outputs), "prompt_word_error_rate": sum(r["word_errors_vs_prompt"] for r in outputs)/sum(r["reference_words"] for r in outputs), "negative_vcm_windows": len(windows), "windows_passing_final_intent_threshold": sum(w["passed_final_intent_threshold"] for w in windows), "positive_command_samples": 0, "training_use": False, "interpretation": "Small evaluation-only read-prompt diagnostic. References are elicitation prompts rather than verified verbatim transcripts. Speakers are publisher-labelled as born in the Philippines; this is neither a representative accent benchmark nor a test of supported Philippine-English voice commands. The VCM windows contain out-of-scope requests/read speech; passing the intent threshold is a diagnostic false accept, not necessarily a completed frame."})
