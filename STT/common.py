"""Independent STT experiment utilities; no imports from the speaker pipeline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
NUM_WORDS = set(ONES + TENS + ['hundred'])
CHARS = "_ abcdefghijklmnopqrstuvwxyz'"

def say_number(value):
    if not 0 <= value <= 100:
        raise ValueError('Only integer values 0 through 100 are supported')
    if value == 100:
        return 'one hundred'
    if value < 20:
        return ONES[value]
    return TENS[value // 10] + (' ' + ONES[value % 10] if value % 10 else '')

NUMBER_TEXT = {say_number(n): n for n in range(101)} | {'a hundred': 100, 'hundred': 100, 'oh': 0}

def normalize(text):
    text = unicodedata.normalize('NFKD', text.lower().replace('’', "'"))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'\b\d+\b', lambda m: say_number(int(m[0])) if int(m[0]) <= 100 else ' '.join(ONES[int(c)] for c in m[0]), text)
    return re.sub(r'\s+', ' ', re.sub("[^a-z' ]", ' ', text)).strip()

def parse_number(text):
    """Strict English cardinal slot. No fuzzy substitution of numerical values."""
    clean = text.lower().strip().replace('-', ' ')
    if re.fullmatch(r'\d{1,3}(?:\s*%|\s+percent)?', clean):
        value = int(re.match(r'\d+', clean)[0])
        return value if value <= 100 else None
    clean = re.sub(r'\s*(?:percent|%)$', '', clean).strip()
    return NUMBER_TEXT.get(re.sub(r'\s+', ' ', clean))

def extract_number(text):
    if re.search(r'[-−]\s*\d|\bminus\b|\bnegative\b|\d[.,]\d|\bpoint\b', text.lower()):
        return None
    words = re.findall(r'[a-z]+|\d+', text.lower())
    chunks, current = [], []
    for word in words + ['<end>']:
        if word in NUM_WORDS or word.isdigit() or (word == 'a' and current == []):
            current.append(word)
        else:
            if current and current != ['a']:
                chunks.append(' '.join(current))
            current = []
    values = [parse_number(c) for c in chunks]
    return values[0] if len(values) == 1 else None

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def split_speaker(speaker):
    bucket = int(digest('STT-v1/' + speaker)[:8], 16) % 100
    return 'train' if bucket < 75 else ('dev' if bucket < 87 else 'test')

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]

def write_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def write_jsonl(path, rows):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        new = [i]
        for j, bj in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (ai != bj)))
        row = new
    return row[-1]
