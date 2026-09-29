"""Command selection belongs to the VCM; this module only validates arguments."""
from __future__ import annotations

import re
import unicodedata

from .levels import level_parameter, percentage

PARAMETER_COMMANDS = {"POWER_ON", "POWER_OFF", "BRIGHTNESS_SET", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN", "COLOR_SET", "VOLUME_SET", "MEDIA_PLAY", "TIMER_SET"}
ACTION = {"POWER_ON": "SET_POWER", "POWER_OFF": "SET_POWER", "BRIGHTNESS_SET": "SET_BRIGHTNESS", "BRIGHTNESS_UP": "ADJUST_BRIGHTNESS", "BRIGHTNESS_DOWN": "ADJUST_BRIGHTNESS", "COLOR_SET": "SET_COLOR", "VOLUME_SET": "SET_VOLUME", "MEDIA_PLAY": "PLAY", "MEDIA_PAUSE": "PAUSE", "MEDIA_RESUME": "RESUME", "MEDIA_STOP": "STOP_MEDIA", "TIME_QUERY": "GET_TIME", "DATE_QUERY": "GET_DATE", "TIMER_SET": "CREATE_TIMER"}
PARAMETER_COMMANDS.update({"BRIGHTNESS", "VOLUME"})
ACTION.update({"BRIGHTNESS": "SET_BRIGHTNESS", "VOLUME": "SET_VOLUME"})
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


def without_target(text, target, *, preserve_numbers=False):
    remainder = match_text(text) if not preserve_numbers else "".join(
        c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    for alias in sorted([target["name"], *target.get("aliases", [])], key=len, reverse=True):
        pattern = r"[^\w]+".join(re.escape(part) for part in match_text(alias).split())
        remainder = re.sub(r"\b" + pattern + r"\b", " ", remainder)
    return remainder


def match_target(text, catalog):
    normalized = " " + match_text(text) + " "
    matches = []
    for entity in catalog.get("entities", []):
        aliases = [entity["name"], *entity.get("aliases", [])]
        spans = [m.span() for alias in aliases for m in re.finditer(r"\b" + re.escape(match_text(alias)) + r"\b", normalized)]
        if spans:
            matches.append((entity, spans))
    if not matches:
        raise ValueError("Unknown or missing target name")
    # Exact full alias matching, with explicit ambiguity rather than nearest-name substitution.
    # A full alias owns an overlapping generic suffix ("bedroom lights" vs "lights").
    matches = [(entry, spans) for entry, spans in matches if not all(
        any(other["id"] != entry["id"] and a <= start and end <= b and (a, b) != (start, end)
            for other, other_spans in matches for a, b in other_spans)
        for start, end in spans)]
    if any(not entry.get("generic") for entry, _ in matches) and not re.search(r"\b(and|plus)\b", normalized):
        matches = [(entry, spans) for entry, spans in matches if not entry.get("generic")]
    unique = {entry["id"]: entry for entry, _ in matches}
    if len(unique) != 1:
        raise ValueError("Ambiguous target name")
    target = next(iter(unique.values()))
    if target.get("acoustic_key") and target["kind"] == "light" and re.search(r"\b(radio|fan|heater|thermostat|television|tv|computer)\b|\bair conditioner\b", normalized):
        raise ValueError("Spoken device conflicts with the registered light target")
    return target


def color_parameter(text, target=None):
    """Resolve one color independently of whether the complete command is valid."""
    remainder = without_target(text, target) if target else match_text(text)
    colors = COLORS & set(re.findall(r"[a-z]+", remainder))
    transitions = re.findall(r"\bfrom\s+(" + "|".join(sorted(COLORS)) + r")\s+to\s+(" + "|".join(sorted(COLORS)) + r")\b", remainder)
    if len(transitions) == 1 and colors == set(transitions[0]) and not re.search(r"\b(or|and)\b", remainder):
        return transitions[0][1]
    if len(colors) != 1:
        raise ValueError("Missing or ambiguous color parameter")
    return next(iter(colors))


def media_parameters(text):
    """Remove explicit request wording under a VCM-selected MEDIA_PLAY intent."""
    cleaned = text.lower().strip()
    cleaned = re.sub(r"\s+please$", "", cleaned)
    cleaned = re.sub(r"^please\s+", "", cleaned)
    if re.fullmatch(r"(?:resume|continue)(?:\s+(?:the\s+)?music)?", cleaned):
        raise ValueError("Parameter speech contradicts the VCM media intent")
    # Only strip these wrappers when followed by an explicit play request; names
    # such as 'I Want to Break Free' remain intact after an initial 'play'.
    cleaned = re.sub(r"^(?:(?:can|could|would|will) you (?:please )?|i (?:want|would like|would love) to )(?=(?:play|listen to|put on|start)\b)", "", cleaned)
    cleaned = re.sub(r"^(play|listen to|put on|start)\s+(?:for me\s+)?", "", cleaned)
    if re.fullmatch(r"(?:turn|switch)\s+(?:on\s+(?:the\s+)?music|(?:the\s+)?music\s+on)", cleaned):
        cleaned = "music"
    if not cleaned or cleaned in {"play", "listen to", "put on", "start"}:
        raise ValueError("Missing media parameter or explicit generic music request")
    if cleaned in {"music", "some music", "the music"}:
        return {}
    if " by " in cleaned:
        title, artist = cleaned.rsplit(" by ", 1)
        # An artist-only request does not name a song literally called 'songs'.
        if title.strip() not in {"music", "the music", "songs", "all songs", "some songs", "an album", "albums"}:
            return {"title": title.strip(), "artist": artist.strip()}
    return {"selection_text": cleaned}


def parse_parameters(command, text, catalog=None):
    if command not in ACTION:
        return {"status": "rejected", "reason": "unsupported_command"}
    catalog = catalog or {"entities": []}
    arguments = {}
    action = ACTION[command]
    try:
        if re.search(r"\b(do not|don't|never|instead|cancel)\b", text.lower()):
            raise ValueError("Contradictory or cancelled parameter tail")
        if command in {"POWER_ON", "POWER_OFF", "BRIGHTNESS", "BRIGHTNESS_SET", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN", "COLOR_SET"}:
            target = match_target(text, catalog)
            arguments["target"] = {"id": target["id"], "name": target["name"], "device_kind": target["kind"]}
            if command.startswith("POWER_"):
                contradiction = "off" if command == "POWER_ON" else "on"
                if re.search(r"\b" + contradiction + r"\b", without_target(text, target)):
                    raise ValueError("Parameter tail contradicts the VCM power intent")
                arguments["on"] = command == "POWER_ON"
            elif target["kind"] != "light":
                raise ValueError("Requested light property on a non-light target")
        if command in {"BRIGHTNESS", "VOLUME"}:
            opposite = r"\b(volume|sound|louder|quieter)\b" if command == "BRIGHTNESS" else r"\b(brightness|lights?|lamps?|brighten|dim)\b"
            if re.search(opposite, text.lower()):
                raise ValueError("Parameter speech contradicts the VCM device property")
            numeric_text = without_target(text, target, preserve_numbers=True) if command == "BRIGHTNESS" else text
            if command == "BRIGHTNESS" and COLORS & set(re.findall(r"[a-z]+", numeric_text)):
                raise ValueError("Color speech contradicts the VCM brightness family")
            parameter = level_parameter(numeric_text)
            arguments.update(parameter)
            action = ("SET_" if "percent" in parameter else "ADJUST_") + command
        elif command in {"BRIGHTNESS_SET", "VOLUME_SET"}:
            numeric_text = text.lower()
            if command == "BRIGHTNESS_SET":
                numeric_text = without_target(text, target, preserve_numbers=True)
            value = percentage(numeric_text)
            if value is None:
                raise ValueError("Missing numeric parameter")
            arguments["percent"] = value
        elif command in {"BRIGHTNESS_UP", "BRIGHTNESS_DOWN"}:
            numeric_text = without_target(text, target, preserve_numbers=True)
            number_words = "|".join([*ONES, *TENS])
            if re.search(r"\b(to|at|on)\s+(?:(?:a|the)\s+)?(?:level\s+)?(?:\d+|" + number_words + r"|max(?:imum)?|full|medium|half)\b|\bhalf way\b|\ball the way\b", numeric_text):
                raise ValueError("Absolute brightness cannot be executed as a relative adjustment")
            direction_text = re.sub(r"\bdown (here|there)\b", "here", numeric_text)
            direction_text = re.sub(r"\bless brighter\b", "less bright", direction_text)
            opposite = r"\b(dim|dimmer|decrease|reduce|lower|down|less|quieter|softer)\b" if command.endswith("_UP") else r"\b(brighten|brighter|increase|raise|up|higher|louder)\b"
            if re.search(opposite, direction_text):
                raise ValueError("Parameter speech contradicts the VCM brightness direction")
            value = percentage(numeric_text)
            value = 10 if value is None else value
            if not 0 < value <= 100:
                raise ValueError("Adjustment outside 1..100 percentage points")
            arguments["delta_percentage_points"] = value * (1 if command.endswith("_UP") else -1)
        elif command == "COLOR_SET":
            arguments["color"] = color_parameter(text, target)
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
            arguments = media_parameters(text)
        return validate_arguments(command, arguments, catalog, action=action)
    except ValueError as exc:
        return {"status": "needs_clarification", "vcm_command": command, "reason": str(exc)}




def validate_arguments(command, arguments, catalog=None, *, action=None):
    """Shared typed checks for typed acoustic-slot arguments."""
    catalog = catalog or {"entities": []}
    if command not in ACTION:
        return {"status": "rejected", "reason": "unsupported_command"}
    try:
        args = dict(arguments)
        allowed = {"POWER_ON": {"target", "on"}, "POWER_OFF": {"target", "on"},
                   "BRIGHTNESS_UP": {"target", "delta_percentage_points"}, "BRIGHTNESS_DOWN": {"target", "delta_percentage_points"},
                   "BRIGHTNESS": {"target", "percent", "delta_percentage_points"}, "VOLUME": {"percent", "delta_percentage_points"},
                   "BRIGHTNESS_SET": {"target", "percent"}, "VOLUME_SET": {"percent"}, "COLOR_SET": {"target", "color"},
                   "MEDIA_PLAY": {"selection_text", "title", "artist"}, "TIMER_SET": {"duration_s"}}
        if set(args) - allowed.get(command, set()):
            raise ValueError("Unexpected parameter for the VCM intent")
        if command in {"POWER_ON", "POWER_OFF", "BRIGHTNESS", "BRIGHTNESS_SET", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN", "COLOR_SET"}:
            target = args.get("target", {})
            if not isinstance(target, dict):
                raise ValueError("Invalid target binding")
            matches = [e for e in catalog.get("entities", []) if e["id"] == target.get("id")]
            if len(matches) != 1:
                raise ValueError("Unknown or ambiguous target binding")
            entity = matches[0]
            args["target"] = {"id": entity["id"], "name": entity["name"], "device_kind": entity["kind"]}
            if not command.startswith("POWER_") and entity["kind"] != "light":
                raise ValueError("Requested light property on a non-light target")
        if command.startswith("POWER_") and args.get("on") is not (command == "POWER_ON"):
            raise ValueError("Power parameter contradicts the VCM intent")
        if command == "COLOR_SET" and args.get("color") not in COLORS:
            raise ValueError("Missing or unsupported color")
        if command in {"BRIGHTNESS", "VOLUME", "BRIGHTNESS_SET", "VOLUME_SET"}:
            fields = {"percent", "delta_percentage_points"} & set(args)
            if len(fields) != 1 or command.endswith("_SET") and "percent" not in fields:
                raise ValueError("Specify exactly one supported level operation")
        if command == "MEDIA_PLAY" and (any(not isinstance(v, str) or not v.strip() for v in args.values())
                or "selection_text" in args and len(args) != 1):
            raise ValueError("Invalid media selection")
        if command in {"BRIGHTNESS_UP", "BRIGHTNESS_DOWN"}:
            value = args.get("delta_percentage_points")
            if type(value) is not int or not 0 < abs(value) <= 100 or (value > 0) != command.endswith("_UP"):
                raise ValueError("Relative parameter contradicts the VCM intent")
        if "percent" in args and (type(args["percent"]) is not int or not 0 <= args["percent"] <= 100):
            raise ValueError("Percentage outside 0..100")
        if "delta_percentage_points" in args and (type(args["delta_percentage_points"]) is not int or not 0 < abs(args["delta_percentage_points"]) <= 100):
            raise ValueError("Adjustment outside 1..100 percentage points")
        return {"status": "accepted", "command": {"action": action or ACTION[command], "arguments": args}}
    except ValueError as exc:
        return {"status": "needs_clarification", "vcm_command": command, "reason": str(exc)}


def parameter_slice(pcm, trigger_seconds, lookback_seconds=0.5, rate=16000):
    start = max(0, int(round((trigger_seconds - lookback_seconds) * rate)))
    if start >= len(pcm):
        return pcm[:0], start
    return pcm[start:], start
