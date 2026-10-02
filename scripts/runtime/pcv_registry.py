"""Versioned custom slots for research bundles; edits invalidate calibration."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, command, compile_inventory, extended_cases, normalize_text, canonical

KINDS = {'light_id': 'lights', 'song_id': 'songs', 'artist_id': 'artists', 'contact_id': 'contacts', 'task_id': 'tasks'}


def reconciled_registry(base_catalog, configured):
    registry = json.loads(json.dumps(configured))
    registry['version'] = 'PCV-B1-effective-1'
    for kind in KINDS.values():
        registry.setdefault(kind, [])
    known = {kind: {row['id'] for row in registry[kind]} for kind in KINDS.values()}
    for entry in base_catalog['entries']:
        c = entry['command']
        if c is None:
            continue
        for role, value in c['slots'].items():
            kind = KINDS.get(role)
            if kind and value not in known[kind]:
                registry[kind].append({'id': value, 'name': value, 'aliases': [],
                    'retained_from_frozen_catalog': True, 'display_name_verified': False,
                    'provenance': 'Opaque ID retained without changing frozen labels; spoken forms remain in catalog'})
                known[kind].add(value)
    return registry


def validate_registry(registry):
    if not isinstance(registry.get('version'), str) or not registry['version']:
        raise ValueError('Registry needs a version')
    for kind in KINDS.values():
        identifiers, aliases = set(), {}
        for row in registry.get(kind, []):
            if row['id'] in identifiers:
                raise ValueError('Duplicate registry ID')
            identifiers.add(row['id'])
            if not row.get('name'):
                raise ValueError('Registry needs a name')
            if row.get('retained_from_frozen_catalog'):
                continue
            for alias in [row['name'], *row.get('aliases', [])]:
                key = normalize_text(alias)
                if not key:
                    raise ValueError('Empty alias')
                if kind != 'songs' and key in aliases and aliases[key] != row['id']:
                    raise ValueError('Ambiguous registry alias')
                aliases[key] = row['id']


def load_registry(path):
    registry = json.loads(Path(path).read_text(encoding='utf8'))
    validate_registry(registry)
    return registry


def compile_registry(base_catalog, path, words):
    registry = load_registry(path)
    known = {kind: {r['id'] for r in registry.get(kind, [])} for kind in KINDS.values()}
    for entry in base_catalog['entries']:
        if entry['command']:
            for role, value in entry['command']['slots'].items():
                if role in KINDS and value not in known[KINDS[role]]:
                    raise ValueError('Unregistered catalog ID: ' + value)
    custom = {kind: [r for r in registry.get(kind, []) if not r.get('retained_from_frozen_catalog')]
              for kind in ('lights','songs','contacts','tasks')}
    rows = extended_cases(custom)
    for contact in custom['contacts']:
        for name in contact.get('aliases', []):
            rows.append({'id': 'CUSTOM-CALL-' + normalize_text(name).replace(' ', '-'), 'text': 'call ' + name,
                         'command': command('CALL', contact_id=contact['id']), 'tier': 'custom'})
    for task in custom['tasks']:
        for name in [task['name'], *task.get('aliases', [])]:
            for template in ('reminder {}', 'remind me to {}', 'create a reminder to {}'):
                text = template.format(name)
                rows.append({'id': 'CUSTOM-TASK-' + normalize_text(text).replace(' ', '-'), 'text': text,
                             'command': command('CREATE_REMINDER', task_id=task['id']), 'tier': 'custom'})
    catalog = compile_inventory([*base_catalog['entries'], *rows], words)
    catalog['requires_calibration'] = True
    return registry, catalog
