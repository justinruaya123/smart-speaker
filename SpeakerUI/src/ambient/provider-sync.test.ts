import assert from 'node:assert/strict';
import { test } from 'node:test';
import { setImmediate as settle } from 'node:timers/promises';
import { connectProviderStates } from './provider-sync.js';

test('state readers pause when hidden, resume on visibility, and ignore responses after close', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  class Page extends EventTarget { hidden = true; }
  const page = new Page();
  let calls = 0, updates = 0;
  let resolveHa: (response: Response) => void = () => {};
  const pendingHa = new Promise<Response>(resolve => { resolveHa = resolve; });
  const readers = connectProviderStates({ ha: () => { updates++; }, spotify: () => { updates++; }, failed: () => assert.fail('Closed requests must not surface errors') }, async (path: string) => {
    calls++;
    return path.includes('home-assistant') ? pendingHa : Response.json({});
  }, page);
  assert.equal(calls, 0);
  page.hidden = false; page.dispatchEvent(new Event('visibilitychange')); await settle();
  assert.equal(calls, 2);
  assert.equal(updates, 1, 'Spotify is not blocked by a pending HA request');
  t.mock.timers.tick(1500); await settle();
  assert.equal(calls, 2, 'No overlapping read');
  page.hidden = true; page.dispatchEvent(new Event('visibilitychange'));
  t.mock.timers.tick(10000); await settle();
  assert.equal(calls, 2);
  readers.close(); resolveHa(Response.json({})); await settle();
  assert.equal(updates, 1);
  page.hidden = false; page.dispatchEvent(new Event('visibilitychange'));
  t.mock.timers.tick(10000); await settle();
  assert.equal(calls, 2);
});

test('a read begun before a control action cannot overwrite its result', async (t) => {
  class Page extends EventTarget { hidden = false; }
  const page = new Page();
  let updates = 0;
  let resolveHa: (response: Response) => void = () => {};
  const readers = connectProviderStates({ ha: () => { updates++; }, spotify: () => {}, failed: () => {} }, async (path: string) => {
    if (path.includes('home-assistant')) return new Promise<Response>(resolve => { resolveHa = resolve; });
    return Response.json({});
  }, page);
  t.after(() => readers.close());
  const release = readers.hold('ha');
  resolveHa(Response.json({ old: true })); await settle();
  assert.equal(updates, 0);
  release(); await settle();
  resolveHa(Response.json({ current: true })); await settle();
  assert.equal(updates, 1);
  readers.close();
});

test('Spotify action release briefly refreshes faster and then returns to the normal interval', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  class Page extends EventTarget { hidden = false; }
  let spotifyReads = 0;
  const readers = connectProviderStates({ ha() {}, spotify() {}, failed() { assert.fail('Unexpected read failure'); } }, async (path: string) => {
    if (path.includes('spotify')) spotifyReads++;
    return Response.json({});
  }, new Page());
  t.after(() => readers.close());
  await settle();
  readers.hold('spotify')(); await settle();
  const afterAction = spotifyReads;
  for (let i = 0; i < 6; i++) { t.mock.timers.tick(500); await settle(); }
  assert.equal(spotifyReads, afterAction + 6);
  t.mock.timers.tick(4999); await settle();
  assert.equal(spotifyReads, afterAction + 6);
  t.mock.timers.tick(1); await settle();
  assert.equal(spotifyReads, afterAction + 7);
});
