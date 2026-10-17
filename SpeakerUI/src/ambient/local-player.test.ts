import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createLocalPlayer } from './local-player.js';

test('local playback only responds once to a live requested execution; snapshots and nonowners cannot play', async () => {
  let plays = 0, pauses = 0;
  const audio: any = { src: '', currentTime: 12, volume: .5, muted: false, play: async () => { plays++; }, pause: () => { pauses++; } };
  const player = createLocalPlayer(audio);
  const execution = { id: 's:u', status: 'requested', local_media: { action: 'PLAY_MUSIC', url: '/api/media/local/' + 'a'.repeat(32), slots: {} } };
  assert.equal(await player.handle(execution, false), undefined);
  assert.equal(plays, 0);
  assert.equal((await player.handle(execution, true))?.status, 'succeeded');
  assert.equal(await player.handle(execution, true), undefined);
  assert.equal(plays, 1);
  await player.handle({ ...execution, id: 's:stop', local_media: { action: 'STOP', slots: {} } }, true);
  assert.equal(audio.currentTime, 0); assert.equal(pauses, 1);
  await player.handle({ ...execution, id: 's:volume', local_media: { action: 'VOLUME', slots: { operation: 'set', percent: 17 } } }, true);
  assert.equal(audio.volume, .17);
});

test('blocked or invalid local playback returns failure and never retries or substitutes a provider', async () => {
  let calls = 0;
  const audio: any = { src: '', play: async () => { calls++; throw new Error('Autoplay blocked'); } };
  const player = createLocalPlayer(audio);
  const execution = { id: 's:u', status: 'requested', local_media: { action: 'PLAY_MUSIC', url: '/api/media/local/' + 'a'.repeat(32), slots: {} } };
  assert.equal((await player.handle(execution, true))?.status, 'failed');
  assert.equal(await player.handle(execution, true), undefined);
  assert.equal(calls, 1);
  assert.equal((await player.handle({ ...execution, id: 's:unsafe', local_media: { ...execution.local_media, url: 'https://other.invalid/audio' } }, true))?.status, 'failed');
  assert.equal(calls, 1);
});
