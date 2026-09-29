"""Select on matched development commands, never phoneme error alone.

This deliberately conservative research guard protects every previously correct
replay command. It is not a statistical guarantee or an acceptance calibrator.
"""
from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any


Record = dict[str, Any]


def _indexed(rows: list[Record]) -> dict[str, Record]:
    if not rows:
        raise ValueError("Nonempty development records are required")
    result = {}
    for row in rows:
        key = row.get("id")
        digest = row.get("pcm_sha256", "")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("Recording IDs must be nonempty and unique")
        if row.get("split") not in {"dev", "val", "development"}:
            raise ValueError("Selection accepts development records only")
        if (not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Every record requires its decoded PCM hash")
        if not isinstance(row.get("category"), str) or not row["category"]:
            raise ValueError("A task category is required")
        if "target" not in row or "candidate" not in row:
            raise ValueError("Explicit reference and candidate are required")
        for field in ("target", "candidate"):
            if row[field] is not None and not isinstance(row[field], dict):
                raise ValueError("Commands must be typed JSON objects or null")
            json.dumps(row[field], allow_nan=False)
        result[key] = row
    return result


def _exact(row: Record) -> bool:
    return (row["target"] is not None
            and json.dumps(row["target"], sort_keys=True, allow_nan=False)
            == json.dumps(row["candidate"], sort_keys=True, allow_nan=False)
            and not row.get("ties") and not row.get("ambiguous", False))


def _paired(before: list[Record], after: list[Record]) -> Record:
    base, proposed = _indexed(before), _indexed(after)
    if base.keys() != proposed.keys():
        raise ValueError("Compare the same complete development partition")
    counts: dict[str, Counter] = {}
    lost, gained = [], []
    for key, row in base.items():
        other = proposed[key]
        for field in ("pcm_sha256", "category", "target", "split"):
            left = json.dumps(row[field], sort_keys=True, allow_nan=False)
            right = json.dumps(other[field], sort_keys=True, allow_nan=False)
            if left != right:
                raise ValueError(f"Mismatched {field} for {key}")
        old, new = _exact(row), _exact(other)
        category = counts.setdefault(row["category"], Counter())
        category["recordings"] += 1
        category["valid"] += int(row["target"] is not None)
        category["parent_correct"] += int(old)
        category["candidate_correct"] += int(new)
        if old and not new:
            lost.append(key)
        if new and not old:
            gained.append(key)
    return {
        "recordings": len(base), "lost_ids": sorted(lost),
        "gained_ids": sorted(gained),
        "parent_correct": sum(_exact(row) for row in base.values()),
        "candidate_correct": sum(_exact(row) for row in proposed.values()),
        "categories": {key: dict(value) for key, value in counts.items()},
    }


def assess_candidate(
    parent_replay: list[Record], candidate_replay: list[Record],
    parent_external: list[Record], candidate_external: list[Record],
) -> Record:
    """Require an external gain with no previously correct replay losses."""
    replay = _paired(parent_replay, candidate_replay)
    external = _paired(parent_external, candidate_external)
    failures = []
    if replay["lost_ids"]:
        failures.append("lost_previously_correct_replay_commands")
    if external["candidate_correct"] <= external["parent_correct"]:
        failures.append("no_external_complete_command_gain")
    return {
        "eligible": not failures, "failed_requirements": failures,
        "replay": replay, "external": external,
        "acceptance_qualified": False,
        "note": ("Research selection only; "
                 "separate calibrated acceptance is required."),
    }
