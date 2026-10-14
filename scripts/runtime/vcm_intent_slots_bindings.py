"""Semantic binding checks shared by candidate export and runtime loading."""
from scripts.runtime.vcm_intent_slots_contract import digest


def verify_evidence_bindings(contract, policy, memory, *, encoder, frontend, normalization):
    expected = {"encoder": encoder, "frontend": frontend, "normalization": normalization}
    if policy.get("require_calibration"):
        payload = {k: v for k, v in policy.items() if k != "calibration_id"}
        binding = policy.get("binding", {})
        if (policy.get("calibration_id") != digest(payload)
                or policy.get("contract_hash") != contract["contract_hash"]
                or binding.get("contract_hash") != contract["contract_hash"]
                or binding.get("checkpoint") != encoder
                or binding.get("normalization") != normalization
                or binding.get("frontend") != frontend
                or policy.get("thresholds") != {"intent": .95, "slot": .90}):
            raise ValueError("Stale calibration/checkpoint/frontend/normalization binding")
    if memory is not None:
        if (memory.get("encoder_binding") != expected
                or memory.get("catalog_hash") != contract["catalog_hash"]
                or memory.get("memory_id") != digest({k:v for k,v in memory.items() if k != "memory_id"})):
            raise ValueError("Stale memory encoder/catalog binding")
    for role, matcher in policy.get("entities", {}).items():
        if (memory is None or role not in contract["entity_roles"]
                or matcher.get("encoder_binding") != expected
                or matcher.get("catalog_hash") != contract["catalog_hash"]
                or matcher.get("calibration", {}).get("memory_id", memory["memory_id"]) != memory["memory_id"]):
            raise ValueError("Stale entity policy/encoder binding")
