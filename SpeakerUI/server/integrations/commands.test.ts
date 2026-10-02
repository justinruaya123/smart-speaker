import assert from 'node:assert/strict';
import { test } from 'node:test';
import { lightCommand, spotifyCommand } from './commands.js';

test('light controls reject unknown operations, targets and out-of-range values', () => {
  for (const value of [{ entityId: 'switch.a', action: 'on' }, { entityId: 'light.a', action: 'delete' }, { entityId: 'light.a', action: 'brightness', value: 101 }, { entityId: 'light.a', action: 'color', rgb: [255, -1, 0] }, { entityId: 'light.a', action: 'on', service: 'restart' }]) assert.throws(() => lightCommand(value), /light/i);
  assert.deepEqual(lightCommand({ entityId: 'light.a', action: 'brightness', value: 0 }), { entityId: 'light.a', action: 'brightness', value: 0 });
});

test('Spotify controls reject unknown actions and preserve valid zero/false values', () => {
  for (const value of [{ action: 'seek', value: -1 }, { action: 'volume', value: 101 }, { action: 'repeat', value: 'forever' }, { action: 'shuffle', value: 'false' }, { action: 'transfer' }, { action: 'play', device_id: 'foreign' }]) assert.throws(() => spotifyCommand(value), /Spotify/);
  assert.deepEqual(spotifyCommand({ action: 'volume', value: 0 }), { action: 'volume', value: 0 });
  assert.deepEqual(spotifyCommand({ action: 'shuffle', value: false }), { action: 'shuffle', value: false });
});
