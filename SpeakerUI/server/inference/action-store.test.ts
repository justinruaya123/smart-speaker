import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ActionStore } from './action-store.js';

test('one atomic store restores future jobs, marks expired jobs missed, and never replays an uncertain action', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'iris-store-'));
  try {
    const store = await ActionStore.open(directory, () => 1_000);
    assert.equal(await store.claim('s:external', { intent: 'TIME', slots: {} }), true);
    await store.addJob({ id: 'future', kind: 'timer', label: 'Timer', deadline: 10_000 });
    await store.addJob({ id: 'expired', kind: 'alarm', label: 'Alarm', deadline: 2_000 });
    const restored = await ActionStore.open(directory, () => 3_000);
    assert.equal(restored.receipt('s:external')?.status, 'unknown');
    assert.equal(await restored.claim('s:external', { intent: 'TIME', slots: {} }), false);
    assert.equal(restored.snapshot().jobs.find((job: any) => job.id === 'future')?.status, 'scheduled');
    assert.equal(restored.snapshot().jobs.find((job: any) => job.id === 'expired')?.status, 'missed');
    await restored.expire(11_000);
    assert.equal(restored.snapshot().jobs.find((job: any) => job.id === 'future')?.status, 'due');
    const json = JSON.parse(await readFile(join(directory, 'action-state.json'), 'utf8'));
    assert.equal(json.version, 1);
    assert.equal(json.jobs.length, 2);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('concurrent duplicate claims and job updates persist exactly once', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'iris-store-'));
  try {
    const store = await ActionStore.open(directory);
    assert.deepEqual(await Promise.all([store.claim('s:u', { intent: 'TIME', slots: {} }), store.claim('s:u', { intent: 'TIME', slots: {} })]), [true, false]);
    await store.addJob({ id: 'reminder', kind: 'reminder', label: 'Drink water', deadline: null });
    await store.updateJob('reminder', 'complete');
    assert.equal((await ActionStore.open(directory)).snapshot().jobs[0]?.status, 'completed');
  } finally { await rm(directory, { recursive: true, force: true }); }
});
