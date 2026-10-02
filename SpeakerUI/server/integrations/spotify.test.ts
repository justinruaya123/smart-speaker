import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { setImmediate as settle } from 'node:timers/promises';
import { createSpotify, spotifyFailure } from './spotify.js';
import { searchMedia } from '../media/search.js';
import { createLocalMedia } from '../media/local.js';

test('resume continues the Spotify session without a track URI, query or search', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'speaker-resume-'));
  try {
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'fixture', refreshToken: 'fixture', expiresAt: Date.now() + 3_600_000 }));
    for (const deviceId of [undefined, 'desk-speaker']) {
      const calls: { url: string; method: string | undefined; body: unknown }[] = [];
      const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId }, directory, async (url, init) => {
        calls.push({ url: String(url), method: init?.method, body: init?.body });
        return new Response(null, { status: 204 });
      });
      await spotify.control({ action: 'play' });
      assert.deepEqual(calls, [{ url: `https://api.spotify.com/v1/me/player/play${deviceId ? '?device_id=desk-speaker' : ''}`, method: 'PUT', body: undefined }]);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('Spotify error policy distinguishes authentication, quota, rate limit, and no device', async () => {
  const fixtures = JSON.parse(await readFile(new URL('./fixtures.json', import.meta.url), 'utf8'));
  for (const [key, expected] of [['spotifyUnauthorized','authentication'], ['spotifyRateLimit','rate_limited'], ['spotifyQuota','quota_exceeded'], ['spotifyNoDevice','no_device']]) {
    const fixture = fixtures[key!];
    assert.equal(spotifyFailure(fixture.status, fixture.body).code, expected);
  }
});

test('successful playback commands accept empty response bodies while reads still require JSON', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'speaker-empty-command-'));
  try {
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'fixture', refreshToken: 'fixture', expiresAt: Date.now() + 3_600_000 }));
    const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, async () => new Response('', { status: 200 }));
    await spotify.control({ action: 'next' });
    await spotify.control({ action: 'seek', value: 10000 });
    assert.equal((await spotify.snapshot()).status, 'invalid_response');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('playback refreshes promptly after commands rather than caching a settling state for five seconds', async (t) => {
  t.mock.timers.enable({ apis: ['Date'], now: 1000 });
  const directory = await mkdtemp(join(tmpdir(), 'speaker-settling-'));
  try {
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'fixture', refreshToken: 'fixture', expiresAt: 3600000 }));
    let reads = 0;
    const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, async (_url, init) => {
      if (init?.method !== 'GET') return new Response(null, { status: 204 });
      return Response.json({ is_playing: true, shuffle_state: false, repeat_state: 'off', progress_ms: ++reads === 1 ? 2000 : 70000, item: { name: 'Song', duration_ms: 120000 } });
    });
    await spotify.control({ action: 'seek', value: 70000 });
    assert.equal((await spotify.snapshot()).playback?.positionMs, 2000);
    t.mock.timers.tick(500);
    assert.equal((await spotify.snapshot()).playback?.positionMs, 70000);
    assert.equal(reads, 2);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('an older playback read neither delays a command nor overwrites its fresh result', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'speaker-playback-order-'));
  let resolveOld: (response: Response) => void = () => {};
  let started: () => void = () => {};
  const readStarted = new Promise<void>(resolve => { started = resolve; });
  const oldResponse = new Promise<Response>(resolve => { resolveOld = resolve; });
  const body = (name: string) => ({ is_playing: true, shuffle_state: false, repeat_state: 'off', item: { name, duration_ms: 120000 } });
  try {
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'fixture', refreshToken: 'fixture', expiresAt: Date.now() + 3_600_000 }));
    let reads = 0, writes = 0;
    const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, async (_url, init) => {
      if (init?.method !== 'GET') { writes++; return new Response(null, { status: 204 }); }
      if (++reads === 1) { started(); return oldResponse; }
      return Response.json(body('New track'));
    });
    const old = spotify.snapshot(); await readStarted;
    const command = spotify.control({ action: 'next' }); await settle();
    assert.equal(writes, 1, 'A slow earlier read must not delay sending next');
    await command;
    assert.equal((await spotify.snapshot()).playback?.track?.title, 'New track');
    resolveOld(Response.json(body('Old track'))); await old;
    assert.equal((await spotify.snapshot()).playback?.track?.title, 'New track');
  } finally { resolveOld(Response.json(body('Old track'))); await rm(directory, { recursive: true, force: true }); }
});

test('Retry-After suppresses repeated provider calls while local search remains available', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'speaker-spotify-'));
  try {
    await writeFile(join(directory, 'Piano.mp3'), 'fixture');
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'fixture', refreshToken: 'fixture', expiresAt: Date.now() + 3_600_000 }));
    let requests = 0;
    const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, async () => { requests++; return new Response(JSON.stringify({ error: { reason: 'QUOTA_EXCEEDED' } }), { status: 429, headers: { 'retry-after': '60' } }); });
    const local = await createLocalMedia(directory);
    const first = await searchMedia('piano', local, spotify);
    assert.equal(first.tracks[0]?.title, 'Piano'); assert.match(first.fallbackReason || '', /quota/);
    await searchMedia('piano', local, spotify);
    assert.equal(requests, 1);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('OAuth state is browser-bound and single-use before any token request', async () => {
  let requests = 0;
  const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, '/tmp/unused', async () => { requests++; throw new Error('Must not request tokens'); });
  const auth = spotify.authorize(); const state = new URL(auth.url).searchParams.get('state')!;
  await assert.rejects(spotify.callback(state, 'wrong-browser', 'code'), /state/);
  await assert.rejects(spotify.callback(state, auth.binding, 'code'), /state/);
  assert.equal(requests, 0);
});

test('playback polling stays offline when disabled and reports missing authorization', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'speaker-playback-'));
  const forbidden = async () => { assert.fail('No provider call before authorization'); };
  try {
    const disabled = createSpotify({ enabled: false }, directory, forbidden);
    assert.equal((await disabled.snapshot()).status, 'disabled');
    const enabled = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, forbidden);
    assert.equal((await enabled.snapshot()).status, 'not_authorized');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('playback readers share a short cache and keep rate limiting separate from no playback', async (t) => {
  t.mock.timers.enable({ apis: ['Date'], now: 1000 });
  const directory = await mkdtemp(join(tmpdir(), 'speaker-playback-cache-'));
  try {
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'fixture', refreshToken: 'fixture', expiresAt: 3600000 }));
    let requests = 0;
    const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, async () => {
      requests++;
      return requests === 1 ? new Response(null, { status: 204 }) : new Response('{}', { status: 429, headers: { 'retry-after': '60' } });
    });
    const states = await Promise.all([spotify.snapshot(), spotify.snapshot()]);
    assert.deepEqual(states.map(state => state.status), ['idle', 'idle']);
    assert.equal(requests, 1);
    t.mock.timers.tick(6000);
    assert.equal((await spotify.snapshot()).status, 'rate_limited');
    t.mock.timers.tick(6000);
    assert.equal((await spotify.snapshot()).status, 'rate_limited');
    assert.equal(requests, 2);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('concurrent first requests share token loading instead of losing authorization', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'speaker-token-load-'));
  try {
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'fixture', refreshToken: 'fixture', expiresAt: Date.now() + 3_600_000 }));
    const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, async () => Response.json({ tracks: { items: [] } }));
    const outcomes = await Promise.allSettled([spotify.search('first'), spotify.search('second')]);
    assert.deepEqual(outcomes.map((outcome) => outcome.status), ['fulfilled', 'fulfilled']);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('401 preserves the refresh credential for a later request without replaying a mutation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'speaker-token-refresh-'));
  try {
    await writeFile(join(directory, 'spotify-tokens.json'), JSON.stringify({ accessToken: 'expired', refreshToken: 'refresh-fixture', expiresAt: Date.now() + 3_600_000 }));
    let calls = 0;
    const spotify = createSpotify({ enabled: true, clientId: 'fixture', clientSecret: 'fixture', redirectUri: 'http://127.0.0.1/auth/spotify/callback', deviceId: undefined }, directory, async () => {
      calls++;
      if (calls === 1) return new Response('{}', { status: 401 });
      if (calls === 2) return Response.json({ access_token: 'fresh', expires_in: 3_600 });
      return Response.json({ tracks: { items: [] } });
    });
    await assert.rejects(spotify.pause(), /authorization/);
    assert.equal(calls, 1);
    await spotify.search('piano');
    assert.equal(calls, 3);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
