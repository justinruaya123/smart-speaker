"""Two profile scopes and stream event invariants; no recognizer or actions.

A production adapter must first verify the whole bundle, schema, catalog, and
calibrated policy. This module does NOT turn raw head outputs into calibrated
probabilities, validate device availability, or authorize physical actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

PROFILES = {"benchmark", "demo"}
EMPTY = {"WEATHER", "TIME", "PAUSE", "STOP", "NEXT", "MESSAGE",
         "LIST_REMINDERS"}
COLORS = {"red", "blue", "green", "yellow", "white", "orange", "purple",
          "pink", "cyan", "magenta", "warm white", "cool white"}


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _identifier(value: Any, role: str) -> bool:
    return (type(value) is str
            and re.fullmatch(role + r"\.[a-z0-9_.-]+", value) is not None)


def profile_allows(profile: str, command: Any) -> bool:
    """Scope check. The demo still needs catalog identity/capability checks."""
    if profile not in PROFILES:
        raise ValueError("Unknown profile")
    if (not isinstance(command, dict)
            or set(command) != {"intent", "slots"}
            or not isinstance(command["slots"], dict)
            or not isinstance(command["intent"], str)):
        return False
    intent, slots = command["intent"], command["slots"]
    demo = profile == "demo"
    if intent in EMPTY:
        return slots == {}
    if intent == "PLAY_MUSIC":
        if not slots:
            return True
        return (demo and set(slots) in ({"song_id"}, {"song_id", "artist_id"})
                and _identifier(slots.get("song_id"), "song")
                and ("artist_id" not in slots
                     or _identifier(slots["artist_id"], "artist")))
    if intent in {"LIGHT_ON", "LIGHT_OFF"}:
        return (set(slots) == {"light_id"}
                and _identifier(slots["light_id"], "light")
                and (demo or slots["light_id"] == "light.all"))
    if intent == "CALL":
        return (set(slots) == {"contact_id"}
                and _identifier(slots["contact_id"], "contact")
                and slots["contact_id"] == "contact.home")
    if intent == "VOLUME":
        op = slots.get("operation")
        if set(slots) == {"operation"}:
            return op in ({"up", "down", "mute"} if demo else {"up", "down"})
        return (set(slots) == {"operation", "percent"} and op == "set"
                and _integer(slots["percent"], 0, 100))
    if intent == "TIMER":
        value = slots.get("duration_seconds")
        return (set(slots) == {"duration_seconds"}
                and _integer(value, 1, 86400)
                and value in {10, 30, 60})
    if intent == "ALARM":
        value = slots.get("time")
        return (set(slots) == {"time", "timezone"}
                and slots["timezone"] == "Asia/Manila"
                and isinstance(value, str)
                and re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value)
                is not None and value in {"06:00", "08:00", "21:00"})
    if intent == "TEMPERATURE":
        value = slots.get("degrees")
        return (set(slots) == {"degrees", "unit"} and slots["unit"] == "C"
                and _integer(value, -50, 60)
                and value in {18, 22, 26})
    if intent in {"BRIGHTNESS", "COLOR"}:
        field = "percent" if intent == "BRIGHTNESS" else "color"
        if (set(slots) != {"light_id", field}
                or not _identifier(slots["light_id"], "light")
                or (not demo and slots["light_id"] != "light.all")):
            return False
        value = slots[field]
        if field == "percent":
            return _integer(value, 0, 100) and (demo or value in {20, 60, 100})
        return isinstance(value, str) and value in {"red", "blue", "green"}
    if intent == "CREATE_REMINDER":
        return (set(slots) == {"task_id"}
                and slots["task_id"] in {"task.drink_water", "task.study",
                                          "task.exercise"})
    return False



@dataclass
class EventGate:
    """Testable lifecycle gate; qualification comes from a verified bundle.

    Send only final `command` events to the official benchmark log. Partial
    events belong to a separate UI channel and must not enter action dispatch.
    """

    profile: str
    model_hash: str
    qualified: bool = False
    ownership_required: bool = False
    utterance_id: str | None = None
    sample_end: int = 0
    finalized: bool = True

    def start(self, utterance_id: str) -> None:
        if (self.profile not in PROFILES or not utterance_id
                or type(self.qualified) is not bool
                or type(self.ownership_required) is not bool):
            raise ValueError("Invalid profile or utterance identity")
        if not self.finalized:
            raise ValueError("Active utterance must finish or be cancelled")
        self.utterance_id = utterance_id
        self.sample_end = 0
        self.finalized = False

    def partial(self, candidate: Any, sample_end: int) -> dict[str, Any]:
        if (self.finalized or type(sample_end) is not int
                or sample_end <= self.sample_end):
            raise ValueError("Invalid stream sequence or finalized utterance")
        self.sample_end = sample_end
        return {
            "event": "command_partial", "utterance_id": self.utterance_id,
            "profile": self.profile, "model_sha256": self.model_hash,
            "sample_end": sample_end, "accepted": False, "actionable": False,
            "command": None, "candidate": candidate, "final": False,
        }

    def finish(self, command: Any, *, endpoint_confirmed: bool,
               policy_pass: bool, owner_pass: bool = False,
               explicit_target_unresolved: bool = False
               ) -> dict[str, Any] | None:
        if self.finalized:
            return None
        if any(type(flag) is not bool for flag in (
                endpoint_confirmed, policy_pass, owner_pass,
                explicit_target_unresolved)):
            raise ValueError("Decision flags must be bool")
        if not endpoint_confirmed:
            raise ValueError("Final command requires a confirmed endpoint")
        self.finalized = True
        accepted = (self.qualified and policy_pass
                    and (not self.ownership_required or owner_pass)
                    and not explicit_target_unresolved
                    and profile_allows(self.profile, command))
        reason = None
        if not accepted:
            reason = "OOV"
            if not self.qualified:
                reason = "unqualified_model"
            elif self.ownership_required and not owner_pass:
                reason = "speaker_unverified"
            elif explicit_target_unresolved:
                reason = "invalid_argument"
        return {
            "event": "command", "utterance_id": self.utterance_id,
            "profile": self.profile, "model_sha256": self.model_hash,
            "accepted": accepted, "actionable": False, "final": True,
            "actions_enabled": False,
            "command": command if accepted else None, "reason": reason,
        }
