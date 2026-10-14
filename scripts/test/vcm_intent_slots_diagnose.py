"""Exact-shipping full-reference versus independently recorded capture-crop attribution."""
from pathlib import Path
import json

from scripts.runtime.pcv_schema import canonical
from scripts.runtime.speaker_confidence import application_decision
from scripts.test.speaker_demo_suite import read_audio
from scripts.train.vcm_intent_slots_data import sha, write_new


def diagnose(inputs, data, output):
    shipping = inputs.get("shipping_bundle")
    crops = inputs.get("capture_crops")
    if not shipping or not crops:
        raise ValueError("Missing exact shipping checkpoint/graphs and reviewed original-PCM wake/VAD crop spans")
    root = Path(shipping["path"])
    for name, expected in shipping["files"].items():
        if sha(root/name) != expected:
            raise ValueError("Exact shipping binding mismatch: "+name)
    if not shipping.get("verified_shipping") or shipping.get("checkpoint_sha256") != (
            "c471f68f389bb01fa6da1c664d1878a25f0892397200a35a0d664b76b2cb468b"):
        raise ValueError("Only the recorded shipping baseline is comparable")
    from scripts.runtime.pcv_final_runtime import Runtime
    runtime = Runtime(root, diagnostic=True, threads=1, engine="ncnn")
    records = []
    for row in data:
        if row["split"] != "dev":
            raise ValueError("Attribution must use frozen development references")
        pcm = read_audio(row, Path(inputs["audio_root"]))
        crop = crops.get(row["id"])
        if not crop or crop["pcm_sha256"] != row["pcm_sha256"]:
            raise ValueError("Missing original-PCM crop binding: "+row["id"])
        for name, audio in [("reference", pcm), ("capture_crop", pcm[crop["start"]:crop["end"]])]:
            try:
                event = runtime.recognize(audio)
                app = application_decision(event, {"mode": "demo", "intent_threshold": .95, "slot_threshold": .90})
                expected = row["command"]
                candidate = event.get("diagnostic_candidate", {}).get("candidate")
                category = ("accepted_exact" if app["accepted"] and canonical(app["command"]) == canonical(expected)
                            else "confidence_or_validity" if canonical(candidate) == canonical(expected)
                            else "intent_or_slot_error")
                record = {"id": row["id"], "crop": name, "application": app,
                          "category": category, "raw_candidate": candidate}
            except Exception as error:
                record = {"id": row["id"], "crop": name, "fault": str(error)}
            records.append(record)
    write_new(Path(output)/"attribution.json", records)
    return {"records": len(records), "exact_shipping": True}
