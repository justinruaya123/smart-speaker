"""Strict finite command membership and reproducible phone graph compilation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

PHONES = (
    'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG '
    'OW OY P R S SH T TH UH UW V W Y Z ZH'
).split()


def canonical(value: Any) -> str:
    """Preserve JSON numeric types rather than Python's True == 1 equality."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def write_json(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n', encoding='utf-8')


class Contract:
    """Only literal, fully typed commands in the frozen catalog are supported."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.positive = read_jsonl(self.root / 'phrase_catalog.jsonl')
        self.negative = read_jsonl(self.root / 'rejection_catalog.jsonl')
        self.commands = json.loads((self.root / 'allowed_commands.json').read_text())
        self.allowed = {canonical(c) for c in self.commands}
        self.by_id = {r['phrase_id']: r for r in self.positive}
        self.negative_by_id = {r['rejection_id']: r for r in self.negative}
        if (len(self.by_id), len(self.allowed), len(self.negative_by_id)) != (570, 402, 200):
            raise ValueError('PC-v1 required counts changed; version the contract')
        texts = [r['exact_text'] for r in self.positive + self.negative]
        if len(texts) != len(set(texts)) or any('{' in t or '}' in t for t in texts):
            raise ValueError('Duplicate or unexpanded phrase')
        self.intent_use_case = {}
        for row in self.positive:
            self.validate(row['expected_command'])
            self.intent_use_case[row['expected_command']['intent']] = row['use_case']
        names = ['phrase_catalog.jsonl', 'rejection_catalog.jsonl',
                 'allowed_commands.json', 'command.schema.json',
                 'output.schema.json', 'finite_catalogs.json']
        self.binding = {name: file_hash(self.root / name) for name in names}
        self.sha256 = identity(self.binding)

    def validate(self, command: dict) -> None:
        if not isinstance(command, dict) or canonical(command) not in self.allowed:
            raise ValueError('Not an exact supported typed command')


def pronounce(text: str, words: dict[str, list[str]]) -> list[int]:
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    if not tokens:
        raise ValueError('Empty text cannot be a command pronunciation')
    output = []
    for index, token in enumerate(tokens):
        # Letter A in AM is EY; the article a keeps AH. Do not global-replace.
        phones = ['EY'] if token == 'a' and tokens[index + 1:index + 2] == ['m'] else words.get(token)
        if not phones:
            raise ValueError(f'Missing reviewed pronunciation: {token}')
        output.extend(PHONES.index(phone) + 1 for phone in phones)
    return output


def compile_catalog(contract: Contract, lexicon_path: Path) -> dict:
    lexicon = json.loads(Path(lexicon_path).read_text())
    entries = []
    seen = {}
    collisions = []
    for row in contract.positive + contract.negative:
        command = row.get('expected_command')
        phones = pronounce(row['exact_text'], lexicon['words'])
        key = tuple(phones)
        signature = canonical(command)
        entry_id = row.get('phrase_id', row.get('rejection_id'))
        if key in seen and seen[key]['command'] != signature:
            collisions.append([seen[key]['id'], entry_id])
        seen[key] = {'command': signature, 'id': entry_id}
        entries.append({'id': entry_id, 'text': row['exact_text'],
                        'use_case': row['use_case'], 'phones': phones,
                        'command': command,
                        'critical_zero_accept': row.get('critical_zero_accept', False)})
    if collisions:
        raise ValueError(f'Acoustically indistinguishable targets: {collisions}')
    return {'format_version': 2, 'contract_sha256': contract.sha256,
            'lexicon_sha256': file_hash(lexicon_path),
            'phone_inventory': ['<blank>'] + PHONES,
            'entries': entries, 'collisions': collisions,
            'acoustic_qualification': False}
