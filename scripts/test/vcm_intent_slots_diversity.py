"""Audit supplied speaker metadata without changing splits or opening audio."""
from __future__ import annotations

from collections import Counter, defaultdict


ATTRIBUTES = ("gender", "sex", "age_range", "native_language", "language",
              "first_language", "work_school_language", "self_reported_fluency",
              "accent_group", "recording_device")
UNKNOWN = {"", "unknown", "unspecified", "not provided", "n/a", "none"}


def known_identity(value):
    return isinstance(value, str) and value.strip().lower() not in UNKNOWN


def _attribute(row, name):
    # Only named metadata fields count. Sex is never relabeled as gender;
    # voice IDs, names, pitch and accents never supply missing demographics.
    values = set()
    for source in (row.get("speaker_metadata", {}), row.get("publisher_metadata", {})):
        value = source.get(name)
        if isinstance(value, (str, int)) and str(value).strip().lower() not in UNKNOWN:
            values.add(str(value).strip())
    return next(iter(values)) if len(values) == 1 else "conflicting" if values else "unknown"


def _group(rows, kind):
    people = defaultdict(list)
    sources = defaultdict(list)
    for row in rows:
        if known_identity(row.get("speaker_id")):
            people[row["speaker_id"]].append(row)
        sources[row.get("publisher_metadata", {}).get("source") or "unknown"].append(row)
    result = {"recordings": len(rows),
              "declared_speaker_ids" if kind == "human" else "declared_voice_ids": len(people),
              "unidentified_recordings": sum(not known_identity(r.get("speaker_id")) for r in rows),
              "reference_status": dict(sorted(Counter(r.get("reference_status", "unknown") for r in rows).items())),
              "sources": {source: {"recordings": len(group), "declared_ids": len({
                  r["speaker_id"] for r in group if known_identity(r.get("speaker_id"))})}
                          for source, group in sorted(sources.items())}}
    if kind == "human":
        result["demographics"] = {}
        for name in ATTRIBUTES:
            counts = Counter()
            for group in people.values():
                values = {_attribute(r, name) for r in group} - {"unknown"}
                category = (next(iter(values)) if len(values) == 1 else
                            "conflicting" if values else "unknown")
                counts[category] += 1
            result["demographics"][name] = {
                "people": dict(sorted(counts.items())),
                "recordings": dict(sorted(Counter(_attribute(r, name) for r in rows).items())),
                "known_people": sum(n for value, n in counts.items() if value not in {"unknown", "conflicting"}),
                "conflicting_people": counts["conflicting"],
                "provenance": "supplied publisher/speaker metadata; collection method unverified"}
    return result


def audit_diversity(views, *, speaker_metadata=None):
    speaker_metadata = speaker_metadata or {}
    for person, attributes in speaker_metadata.items():
        if (not known_identity(person) or "/" not in person or not isinstance(attributes, dict)
                or set(attributes) - set(ATTRIBUTES)
                or any(type(v) not in (str, int) for v in attributes.values())):
            raise ValueError("Invalid source-qualified speaker metadata")
    identifiers = set()
    memberships = {name: defaultdict(set) for name in
                   ("human_speaker_id", "synthetic_voice_id", "pair_id", "pcm_sha256")}
    populations = {"human": [], "synthetic": []}
    splits = {}
    matched_ids, matched_recordings = set(), 0
    for split, rows in views.items():
        groups = {"human": [], "synthetic": []}
        for row in rows:
            if row["id"] in identifiers:
                raise ValueError("Duplicate trial ID in diversity views")
            identifiers.add(row["id"])
            if row.get("split") != split:
                raise ValueError("Diversity partition mismatch")
            kind = row.get("kind")
            if kind not in groups:
                raise ValueError("Unknown recording population")
            person = row.get("speaker_id")
            if kind == "human" and person in speaker_metadata:
                attributes = speaker_metadata[person]
                original = row.get("speaker_metadata", {})
                if any(name in original and original[name] != value for name, value in attributes.items()):
                    raise ValueError("Supplied metadata would overwrite existing speaker metadata")
                row = {**row, "speaker_metadata": {**original, **attributes}}
                matched_ids.add(person)
                matched_recordings += 1
            groups[kind].append(row)
            populations[kind].append(row)
            if known_identity(row.get("speaker_id")):
                key = "human_speaker_id" if kind == "human" else "synthetic_voice_id"
                memberships[key][row["speaker_id"]].add(split)
            for name in ("pair_id", "pcm_sha256"):
                if row.get(name):
                    memberships[name][row[name]].add(split)
        splits[split] = {kind: _group(group, kind) for kind, group in groups.items()}
    overlap = {name: sum(len(sets) > 1 for sets in groups.values())
               for name, groups in memberships.items()}
    return {"version": 1, "scope": "metadata-only audit of supplied frozen views",
            "identity_scope": "declared source-qualified IDs; cross-source aliases unresolved",
            "independent_human_identities_verified": False,
            "metadata_join": {"matched_human_recordings": matched_recordings,
                              "matched_human_ids": len(matched_ids),
                              "unmatched_metadata_ids": len(set(speaker_metadata) - matched_ids)},
            "splits": splits, "total_human": _group(populations["human"], "human"),
            "total_synthetic": _group(populations["synthetic"], "synthetic"),
            "cross_split_groups": overlap, "declared_split_disjoint": not any(overlap.values()),
            "split_policy": "Person-disjoint first; balance supplied demographics within each split in a new version only.",
            "limits": ["No voice-derived gender, age, identity or accent inference.",
                       "Synthetic voice/clone IDs are not independent human people.",
                       "Missing metadata stays unknown; source labels are not verified identity or demographics.",
                       "No listening, near-duplicate or cross-source alias validation performed by this metadata audit.",
                       "No audio scoring or frozen split changes performed by this metadata audit."]}
