"""Pinned local tooling must also work inside nested package scripts."""
import os
from pathlib import Path

from scripts.runtime import speaker_tools


def test_local_pnpm_is_on_child_script_path(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_tools.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(speaker_tools.platform, 'machine', lambda: 'x86_64')
    tools = tmp_path / '.runtime/tools'
    node = tools / f'node-v{speaker_tools.NODE_VERSION}-linux-x64/bin/node'
    pnpm_root = tools / f'pnpm-{speaker_tools.PNPM_VERSION}'
    pnpm = pnpm_root / 'node_modules/pnpm/bin/pnpm.mjs'
    node.parent.mkdir(parents=True)
    pnpm.parent.mkdir(parents=True)
    node.write_text('fixture')
    pnpm.write_text('fixture')
    monkeypatch.setattr(speaker_tools, 'run', lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('no install')))
    _, _, env = speaker_tools.node_tools(tmp_path)
    path = env['PATH'].split(os.pathsep)
    assert str(pnpm_root / 'node_modules/.bin') in path
    assert path.index(str(node.parent)) < path.index(str(pnpm_root / 'node_modules/.bin'))
