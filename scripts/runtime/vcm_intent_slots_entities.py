"""Role-specific open-set acoustic memory; display strings never score audio."""
from __future__ import annotations

import math
import numpy as np

from scripts.runtime.vcm_intent_slots_contract import digest


def _vector(value):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (64,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
        raise ValueError("Invalid acoustic entity embedding")
    return v / np.linalg.norm(v)


def catalog_ids(catalog):
    entries = catalog.get("catalog_entries", catalog.get("entries", []))
    result = {r: set() for r in ("light", "song", "artist", "contact", "task")}
    for row in entries:
        for key, value in (row.get("command") or {}).get("slots", {}).items():
            if key.endswith("_id") and key[:-3] in result:
                result[key[:-3]].add(value)
    return result


def enroll_examples(examples, encoder_binding, catalog):
    for name in ("encoder", "frontend", "normalization"):
        value = encoder_binding.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("Missing encoder/frontend/normalization binding")
    allowed = catalog_ids(catalog)
    retained, seen = [], {}
    for ex in examples:
        role, entity = ex["role"], ex["entity_id"]
        if role not in allowed or entity not in allowed[role] or not entity.startswith(role+"."):
            raise ValueError("Entity role/catalog mismatch")
        for field in ("sha256", "pcm_sha256", "speaker_id", "source"):
            if not ex.get(field):
                raise ValueError("Missing enrollment provenance: " + field)
        if ex.get("kind") != "human":
            raise ValueError("Enrollment requires original human recordings")
        vector = _vector(ex["embedding"])
        key = (role, ex["pcm_sha256"])
        if key in seen:
            if seen[key] != entity:
                raise ValueError("Conflicting duplicate enrollment")
            continue
        seen[key] = entity
        retained.append({**ex, "embedding": vector.tolist()})
    memory = {"version": 1, "encoder_binding": dict(encoder_binding),
              "catalog_hash": catalog["catalog_hash"], "examples": retained}
    memory["memory_id"] = digest(memory)
    return memory


def match_entity(embedding, role, memory, policy):
    result = {"accepted": False, "entity_id": None, "confidence": None,
              "reason": None, "score_status": "uncalibrated_acoustic_similarity"}

    def reject(reason):
        return {**result, "reason": reason}

    if not memory or not policy:
        return reject("missing_enrollment_or_calibration")
    if (memory.get("encoder_binding") != policy.get("encoder_binding")
            or memory.get("catalog_hash") != policy.get("catalog_hash")
            or memory.get("memory_id") != digest({k: v for k, v in memory.items() if k != "memory_id"})):
        return reject("stale_entity_binding")
    try:
        query = _vector(embedding)
        scores = {}
        for ex in memory["examples"]:
            if ex["role"] == role:
                score = float(query @ _vector(ex["embedding"]))
                scores[ex["entity_id"]] = max(scores.get(ex["entity_id"], -1), score)
        absolute, margin = policy["absolute"], policy["margin"]
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (absolute, margin)):
            return reject("invalid_entity_policy")
        if not -1 <= absolute <= 1 or not 0 <= margin <= 2:
            return reject("invalid_entity_policy")
    except (ValueError, KeyError, TypeError):
        return reject("invalid_entity_evidence")
    if not scores:
        return reject("empty_entity_catalog")
    if len(scores) not in policy.get("catalog_sizes", []):
        return reject("uncalibrated_catalog_size")
    ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
    top = ranked[0][1]
    gap = top - ranked[1][1] if len(ranked) > 1 else None
    result.update(cosine=top, margin=gap, catalog_size=len(scores))
    if top < absolute or (gap is not None and (gap <= 0 or gap < margin)):
        return reject("unknown_or_ambiguous_entity")
    # This raw selection score is explicitly NOT a calibrated probability.
    confidence = (top + 1) / 2
    calibration = policy.get("calibration")
    if calibration:
        if calibration.get("memory_id") != memory["memory_id"]:
            return reject("stale_entity_calibration")
        z = (calibration["weights"][0]*top
             + calibration["weights"][1]*(gap if gap is not None else 0)
             + calibration["bias"])
        confidence = float(1 / (1 + np.exp(-np.clip(z, -700, 700))))
        result["score_status"] = "frozen_calibration"
    return {**result, "accepted": True, "entity_id": ranked[0][0],
            "confidence": confidence, "reason": None}
