import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, writeFile, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createLocalMedia, parseRange } from './local.js';

test('ranges support bounded, open-ended and suffix requests and reject invalid ranges', () => {
  assert.deepEqual(parseRange('bytes=1-3', 10), { start: 1, end: 3 });
  assert.deepEqual(parseRange('bytes=8-', 10), { start: 8, end: 9 });
  assert.deepEqual(parseRange('bytes=-3', 10), { start: 7, end: 9 });
  assert.deepEqual(parseRange('bytes=2-99', 10), { start: 2, end: 9 });
  for (const range of ['bytes=10-', 'bytes=4-2', 'bytes=-0', 'bytes=0-1,4-5', 'bytes=-', 'bad']) assert.throws(() => parseRange(range, 10));
});

test('local catalog handles empty directories, opaque IDs and ignores symlinks', async () => {
  const root = await mkdtemp(join(tmpdir(), 'speaker-media-'));
  try {
    await mkdir(join(root, 'music'));
    assert.deepEqual((await createLocalMedia(join(root, 'music'))).search(''), []);
    await writeFile(join(root, 'music', 'Soft Piano.mp3'), '0123456789');
    await writeFile(join(root, 'private.mp3'), 'private');
    await symlink(join(root, 'private.mp3'), join(root, 'music', 'escape.mp3'));
    const catalog = await createLocalMedia(join(root, 'music'));
    const tracks = catalog.search('PIANO');
    assert.equal(tracks.length, 1);
    assert.match(tracks[0]!.id, /^[a-f0-9]{32}$/);
    assert.equal(catalog.search('escape').length, 0);
    await assert.rejects(catalog.open('../private.mp3'));
    const opened = await catalog.open(tracks[0]!.id);
    assert.equal(opened.size, 10); await opened.handle.close();
  } finally { await rm(root, { recursive: true, force: true }); }
});
