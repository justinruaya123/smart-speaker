"""Command selection belongs to the VCM; this module only validates arguments."""
from __future__ import annotations

import re
import unicodedata

PARAMETER_COMMANDS = {"POWER_ON", "POWER_OFF", "BRIGHTNESS_SET", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN", "COLOR_SET", "VOLUME_SET", "MEDIA_PLAY", "TIMER_SET"}
ACTION = {"POWER_ON": "SET_POWER", "POWER_OFF": "SET_POWER", "BRIGHTNESS_SET": "SET_BRIGHTNESS", "BRIGHTNESS_UP": "ADJUST_BRIGHTNESS", "BRIGHTNESS_DOWN": "ADJUST_BRIGHTNESS", "COLOR_SET": "SET_COLOR", "VOLUME_SET": "SET_VOLUME", "MEDIA_PLAY": "PLAY", "MEDIA_PAUSE": "PAUSE", "MEDIA_RESUME": "RESUME", "MEDIA_STOP": "STOP_MEDIA", "TIME_QUERY": "GET_TIME", "DATE_QUERY": "GET_DATE", "TIMER_SET": "CREATE_TIMER"}
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan", "white"}
ONES = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))


def number(text):
    tokens = re.findall(r"\d+|[a-z]+", text.lower())
    value = 0
    found = False
    for word in tokens:
        if word.isdigit():
            if found:
                raise ValueError("Ambiguous multiple numbers")
            value, found = int(word), True
        elif word in ONES:
            value, found = value + ONES[word], True
        elif word in TENS:
            value, found = value + TENS[word], True
        elif word == "hundred" and found:
            value *= 100
        elif word == "thousand" and found:
            value *= 1000
        elif word == "and":
            continue
        elif found:
            break
    if not found:
        raise ValueError("Missing numeric parameter")
    return value


def match_text(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def without_target(text, target):
    remainder = match_text(text)
    for alias in sorted([target["name"], *target.get("aliases", [])], key=len, reverse=True):
        remainder = re.sub(r"\b" + re.escape(match_text(alias)) + r"\b", " ", remainder)
    return remainder


def match_target(text, catalog):
    normalized = " " + match_text(text) + " "
    matches = []
    for entity in catalog.get("entities", []):
        aliases = [entity["name"], *entity.get("aliases", [])]
        if any(" " + match_text(alias) + " " in normalized for alias in aliases):
            matches.append(entity)
    if not matches:
        raise ValueError("Unknown or missing target name")
    # Exact full alias matching, with explicit ambiguity rather than nearest-name substitution.
    unique = {entry["id"]: entry for entry in matches}
    if len(unique) != 1:
        raise ValueError("Ambiguous target name")
    return next(iter(unique.values()))


def parse_parameters(command, text, catalog=None):
    if command not in ACTION:
        return {"status": "rejected", "reason": "unsupported_command"}
    catalog = catalog or {"entities": []}
    arguments = {}
    try:
        if re.search(r"\b(do not|don't|never|instead|cancel)\b", text.lower()):
            raise ValueError("Contradictory or cancelled parameter tail")
        if command in {"POWER_ON", "POWER_OFF", "BRIGHTNESS_SET", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN", "COLOR_SET"}:
            target = match_target(text, catalog)
            arguments["target"] = {"id": target["id"], "name": target["name"], "device_kind": target["kind"]}
            if command.startswith("POWER_"):
                contradiction = "off" if command == "POWER_ON" else "on"
                if re.search(r"\b" + contradiction + r"\b", without_target(text, target)):
                    raise ValueError("Parameter tail contradicts the VCM power intent")
                arguments["on"] = command == "POWER_ON"
            elif target["kind"] != "light":
                raise ValueError("Requested light property on a non-light target")
        if command in {"BRIGHTNESS_SET", "VOLUME_SET"}:
            numeric_text = text.lower()
            if command == "BRIGHTNESS_SET":
                numeric_text = without_target(text, target)
            value = number(numeric_text)
            if not 0 <= value <= 100:
                raise ValueError("Percentage outside 0..100")
            arguments["percent"] = value
        elif command in {"BRIGHTNESS_UP", "BRIGHTNESS_DOWN"}:
            numeric_text = without_target(text, target)
            try:
                value = number(numeric_text)
            except ValueError as exc:
                if "Missing" not in str(exc):
                    raise
                value = 10
            if not 0 < value <= 100:
                raise ValueError("Adjustment outside 1..100 percentage points")
            arguments["delta_percentage_points"] = value * (1 if command == "BRIGHTNESS_UP" else -1)
        elif command == "COLOR_SET":
            # Remove the matched target's name/aliases before identifying a color argument.
            remainder = without_target(text, target)
            colors = COLORS & set(re.findall(r"[a-z]+", remainder))
            if len(colors) != 1:
                raise ValueError("Missing or ambiguous color parameter")
            arguments["color"] = next(iter(colors))
        elif command == "TIMER_SET":
            value = number(text)
            units = [unit for unit in ("hour", "minute", "second") if unit in text.lower()]
            if len(units) != 1:
                raise ValueError("Specify one duration unit")
            value *= {"hour": 3600, "minute": 60, "second": 1}[units[0]]
            if not 0 < value <= 86400:
                raise ValueError("Duration outside 1..86400 seconds")
            arguments["duration_s"] = value
        elif command == "MEDIA_PLAY":
            cleaned = re.sub(r"^(please\s+)?(play|listen to)\s+", "", text.lower()).strip()
            cleaned = re.sub(r"\s+please$", "", cleaned)
            if not cleaned or cleaned in {"play", "please play", "listen to"}:
                raise ValueError("Missing media parameter or explicit generic music request")
            if cleaned and cleaned not in {"music", "some music", "the music"}:
                if " by " in cleaned:
                    title, artist = cleaned.rsplit(" by ", 1)
                    arguments = {"title": title.strip(), "artist": artist.strip()}
                else:
                    arguments = {"selection_text": cleaned}
        return {"status": "accepted", "command": {"action": ACTION[command], "arguments": arguments}}
    except ValueError as exc:
        return {"status": "needs_clarification", "vcm_command": command, "reason": str(exc)}


def parameter_slice(pcm, trigger_seconds, lookback_seconds=0.5, rate=16000):
    start = max(0, int(round((trigger_seconds - lookback_seconds) * rate)))
    if start >= len(pcm):
        return pcm[:0], start
    return pcm[start:], start
