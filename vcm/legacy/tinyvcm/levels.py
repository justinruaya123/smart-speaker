"""Bounded local grammar for brightness/volume slots; no intent selection."""
from __future__ import annotations

import re

ONES = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
NUMBER_WORDS = set(ONES) | set(TENS) | {"hundred", "thousand", "million"}
UP = r"\b(up|increase|raise|higher|more|brighten|brighter|louder|boost)\b"
DOWN = r"\b(down|decrease|reduce|lower|less|dim|dimmer|dimming|darker|darken|quieter|softer)\b"


def spoken_number(value: int) -> str:
    """Canonical English form, also used to enumerate the grammar in tests."""
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("Percentage outside 0..100")
    if value == 100:
        return "one hundred"
    if value < 20:
        return next(word for word, n in ONES.items() if n == value)
    tens = next(word for word, n in TENS.items() if n == value // 10 * 10)
    return tens + (" " + spoken_number(value % 10) if value % 10 else "")


def _integer(words):
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    if len(words) == 1 and words[0] in ONES:
        return ONES[words[0]]
    if len(words) == 1 and words[0] in TENS:
        return TENS[words[0]]
    if words in (["one", "hundred"], ["a", "hundred"], ["hundred"]):
        return 100
    if len(words) == 2 and words[0] in TENS and words[1] in ONES and 1 <= ONES[words[1]] <= 9:
        return TENS[words[0]] + ONES[words[1]]
    raise ValueError("Malformed or ambiguous numeric parameter")


def percentage(text: str) -> int | None:
    """Extract exactly one integer, preserving signs/decimals for rejection.

    None means no number was spoken. Invalid numbers must never become a
    default ten-point adjustment. Device names must be removed by the caller.
    """
    text = text.lower().replace("â€™", "'")
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    if re.search(r"\d[a-z]|[a-z]\d|\b(degrees?|decibels?|celsius|fahrenheit|watts?|seconds?|minutes?|hours?)\b", text):
        raise ValueError("Expected a percentage, not another unit or an ordinal")
    if re.search(r"\d\s*[.,/]\s*\d|[+\-âˆ’]\s*\d|\b(minus|negative|plus|point|decimal|half|quarter)\b", text):
        raise ValueError("Expected an unsigned whole percentage from 0 to 100")
    tokens = re.findall(r"\d+|[a-z]+|%", text)
    groups, current = [], []
    for i, token in enumerate(tokens):
        is_number = token.isdigit() or token in NUMBER_WORDS
        is_article = token == "a" and i + 1 < len(tokens) and tokens[i + 1] == "hundred"
        is_and = token == "and" and current and i + 1 < len(tokens) and tokens[i + 1] in NUMBER_WORDS
        if is_number or is_article or is_and:
            current.append(token)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if not groups:
        if re.search(r"\b(percent|percentage|per cent)\b|%", text):
            raise ValueError("Missing numeric parameter before percent")
        return None
    if len(groups) != 1:
        raise ValueError("Ambiguous multiple numbers")
    value = _integer(groups[0])
    if not 0 <= value <= 100:
        raise ValueError("Percentage outside 0..100")
    return value


def level_parameter(text: str) -> dict:
    """Resolve a level-family slot to an absolute value or relative delta.

    Bare UP/DOWN means +/-10 percentage points. A numeric relative amount
    requires 'by'; 'to fifty' always means an absolute level, including in
    'turn the volume down to fifty'. An optional percent suffix is equivalent.
    """
    text = text.lower().replace("â€™", "'")
    if re.search(r"\b(not|don't|never|cancel|instead|unless|if|when|then|or)\b", text):
        raise ValueError("Contradictory, conditional, or ambiguous level parameter")
    if re.search(r"\band\b.*\b(play|turn|switch|set|change|dim|brighten)\b", text):
        raise ValueError("Multiple actions in one level request")
    if re.search(r"\b(what|which|whether|show|settings|status|current|double|halve)\b|\bhow (loud|bright|much)\b", text):
        raise ValueError("Unsupported level query or operation")
    if re.search(r"\b(after|before|until)\b|\b(in|at)\b.*\b(seconds?|minutes?|hours?|am|pm)\b", text):
        raise ValueError("Scheduled level changes are unsupported")
    if re.search(r"\b(volume|sound)\b", text) and re.search(r"\b(brightness|lights?|lamps?)\b", text):
        raise ValueError("Multiple device properties in one request")
    text = re.sub(r"\bdown (here|there)\b", "here", text)
    text = re.sub(r"\bless brighter\b", "less bright", text)
    up, down = bool(re.search(UP, text)), bool(re.search(DOWN, text))
    if up and down:
        raise ValueError("Conflicting level directions")
    value = percentage(text)
    if value is not None:
        by = bool(re.search(r"\bby\b", text))
        absolute = bool(re.search(r"\b(to|at|level)\s+(?:(?:a|the)\s+)?(?:level\s+)?(?:\d+|" + "|".join(sorted(NUMBER_WORDS)) + r")\b", text))
        if by:
            if absolute or not (up or down) or value == 0:
                raise ValueError("Relative amount requires one direction and a positive value")
            return {"delta_percentage_points": value if up else -value}
        if (up or down) and not absolute:
            raise ValueError("Say 'to' for an absolute level or 'by' for a relative amount")
        return {"percent": value}
    if re.search(r"\b(by|to|percent|percentage|max|maximum|minimum|full|medium|all the way)\b", text):
        raise ValueError("Missing or unsupported numeric level")
    if up or down:
        return {"delta_percentage_points": 10 if up else -10}
    raise ValueError("Missing level: say UP, DOWN, or a whole number from 0 to 100")
