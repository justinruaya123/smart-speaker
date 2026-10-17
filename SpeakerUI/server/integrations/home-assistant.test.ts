import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test } from 'node:test';
import { setImmediate as settle } from 'node:timers/promises';
import { createHomeAssistant, type HomeAssistantSocket } from './home-assistant.js';

class Socket extends EventEmitter implements HomeAssistantSocket {
  terminated = false;
  send() {}
  ping() {}
  close() { if (!this.terminated) { this.terminated = true; this.emit('close'); } }
  terminate() { this.close(); }
  message(value: unknown) { this.emit('message', Buffer.from(JSON.stringify(value))); }
  ready() { this.emit('open'); this.message({ type: 'auth_ok' }); this.message({ type: 'result', id: 1, success: true }); }
}
const config = { enabled: true as const, url: 'http://ha.invalid', token: 'fixture', entityIds: ['light.desk'] };
const state = { entity_id: 'light.desk', state: 'on', last_updated: '2026-01-01T00:00:00Z', attributes: { friendly_name: 'Desk' } };

test('camera streaming requires an enabled, allowlisted camera before outbound I/O', async () => {
  const forbidden = async () => { assert.fail('Unselected cameras must stay offline'); };
  const signal = new AbortController().signal;
  const disabled = createHomeAssistant({ enabled: false, entityIds: [] }, forbidden);
  await assert.rejects(disabled.cameraStream('camera.front', signal), /disabled/);
  const ha = createHomeAssistant(config, forbidden);
  for (const id of ['camera.front', 'light.desk', '../camera.front']) {
    await assert.rejects(ha.cameraStream(id, signal), /allowlisted/);
  }
});

test('camera streaming preserves multipart boundaries and has no whole-stream timeout', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const lifetime = new AbortController();
  let upstreamSignal: AbortSignal | undefined;
  const contentType = 'multipart/x-mixed-replace; boundary=ha-frame';
  const ha = createHomeAssistant({ ...config, entityIds: ['camera.front'] }, async (url, init) => {
    assert.equal(url, 'http://ha.invalid/api/camera_proxy_stream/camera.front');
    assert.equal(new Headers(init?.headers).get('Authorization'), 'Bearer fixture');
    assert.equal(init?.redirect, 'error');
    upstreamSignal = init?.signal as AbortSignal;
    return new Response('--ha-frame\r\n', { headers: { 'content-type': contentType } });
  });
  const stream = await ha.cameraStream('camera.front', lifetime.signal);
  assert.equal(stream.contentType, contentType);
  t.mock.timers.tick(60_000);
  assert.equal(upstreamSignal?.aborted, false);
  lifetime.abort();
  assert.equal(upstreamSignal?.aborted, true);
  await stream.body.cancel();
});

test('camera setup times out and invalid streams are cancelled without forwarding content', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const cfg = { ...config, entityIds: ['camera.front'] };
  const stuck = createHomeAssistant(cfg, async (_url, init) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(new Error('Private upstream detail')), { once: true });
  }));
  const pending = stuck.cameraStream('camera.front', new AbortController().signal);
  const rejection = assert.rejects(pending, /Camera stream could not be opened/);
  t.mock.timers.tick(15_000);
  await rejection;
  for (const contentType of ['text/html', 'image/jpeg', 'multipart/x-mixed-replace']) {
    let cancelled = false;
    const ha = createHomeAssistant(cfg, async () => new Response(new ReadableStream({ cancel() { cancelled = true; } }), { headers: { 'content-type': contentType } }));
    await assert.rejects(ha.cameraStream('camera.front', new AbortController().signal), /MJPEG/);
    assert.equal(cancelled, true);
  }
});

test('light controls enforce the allowlist and use observed state after an accepted action', async () => {
  const socket = new Socket();
  let writes = 0;
  const ha = createHomeAssistant(config, async (_input, init) => {
    if (init?.method === 'POST') { writes++; return Response.json([]); }
    return Response.json({ ...state, attributes: { friendly_name: 'Desk', brightness: 128, rgb_color: [255, 0, 0], private_attribute: 'secret' } });
  }, () => socket);
  try {
    ha.start(); socket.ready(); await settle();
    await assert.rejects(ha.control({ entityId: 'light.other', action: 'off' }), /allowlisted/);
    assert.equal(writes, 0);
    await ha.control({ entityId: 'light.desk', action: 'off' });
    assert.equal(writes, 1);
    assert.equal(ha.snapshot().entities[0]?.state, 'on', 'acceptance must not invent an off state');
    assert.deepEqual(ha.snapshot().entities[0]?.attributes, { brightness: 128, rgb_color: [255, 0, 0] });
  } finally { ha.close(); }
});

test('silent Home Assistant connections expire and mark cached state stale', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  const sockets: Socket[] = [];
  const ha = createHomeAssistant(config, async () => Response.json(state), () => { const socket = new Socket(); sockets.push(socket); return socket; });
  try {
    ha.start(); sockets[0]!.ready(); await settle();
    assert.equal(ha.snapshot().entities[0]?.stale, false);
    t.mock.timers.tick(15_000); t.mock.timers.tick(15_000);
    assert.equal(sockets[0]!.terminated, true);
    assert.equal(ha.snapshot().entities[0]?.stale, true);
  } finally { ha.close(); }
});

test('an upgraded connection that never authenticates has a setup deadline', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  const socket = new Socket();
  const ha = createHomeAssistant(config, async () => { assert.fail('No snapshot before authentication'); }, () => socket);
  try { ha.start(); socket.emit('open'); t.mock.timers.tick(10_001); assert.equal(socket.terminated, true); }
  finally { ha.close(); }
});

for (const outcome of ['404', 'reject']) test(`old hydration ${outcome} cannot overwrite a reconnected cache`, async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  let resolveOld: (value: Response) => void = () => {};
  let rejectOld: (error: Error) => void = () => {};
  const old = new Promise<Response>((resolve, reject) => { resolveOld = resolve; rejectOld = reject; });
  let requests = 0;
  const sockets: Socket[] = [];
  const ha = createHomeAssistant(config, async () => ++requests === 1 ? old : Response.json(state), () => { const socket = new Socket(); sockets.push(socket); return socket; });
  try {
    ha.start(); sockets[0]!.ready(); await settle();
    sockets[0]!.terminate(); t.mock.timers.tick(1_000);
    sockets[1]!.ready(); await settle();
    assert.equal(ha.snapshot().entities[0]?.stale, false);
    if (outcome === '404') resolveOld(new Response('', { status: 404 })); else rejectOld(new Error('old timeout'));
    await settle();
    assert.equal(ha.snapshot().entities[0]?.state, 'on');
    assert.equal(ha.snapshot().entities[0]?.stale, false);
    assert.equal(ha.snapshot().status, 'connected');
  } finally { ha.close(); }
});

test('climate, media, weather and scripts enforce capability/allowlist checks and exact service payloads', async () => {
  const posts: any[] = [];
  const entities: Record<string, any> = {
    'climate.room': { supported_features: 1, temperature_unit: '°C', min_temp: 16, max_temp: 30 },
    'media_player.room': { supported_features: 512 | 16384 | 4 | 4096 },
    'weather.home': { temperature: 28, temperature_unit: '°C', humidity: 70, secret: 'hidden' },
    'script.call_home': {},
  };
  const ha = createHomeAssistant({ ...config, entityIds: Object.keys(entities) }, async (url, init) => {
    if (init?.method === 'POST') { posts.push([String(url), JSON.parse(String(init.body))]); return Response.json([]); }
    const id = decodeURIComponent(String(url).split('/').at(-1)!);
    return Response.json({ ...state, entity_id: id, state: id.startsWith('weather.') ? 'sunny' : 'on', attributes: entities[id] });
  });
  await (ha as any).climate('climate.room', 22);
  await (ha as any).media('media_player.room', { intent: 'PLAY_MUSIC', slots: { song_id: 'song.dynamite' } }, 'media-source://music/song');
  await (ha as any).media('media_player.room', { intent: 'VOLUME', slots: { operation: 'set', percent: 17 } });
  await (ha as any).media('media_player.room', { intent: 'STOP', slots: {} });
  await (ha as any).script('script.call_home');
  assert.deepEqual(posts.map(post => post[1]), [{ entity_id: 'climate.room', temperature: 22 },
    { entity_id: 'media_player.room', media_content_id: 'media-source://music/song', media_content_type: 'music' },
    { entity_id: 'media_player.room', volume_level: .17 }, { entity_id: 'media_player.room' }, { entity_id: 'script.call_home' }]);
  assert.match(posts[3][0], /media_stop$/);
  const weather = await (ha as any).weather('weather.home');
  assert.deepEqual(weather, { entity_id: 'weather.home', state: 'sunny', temperature: 28, unit: '°C', humidity: 70, observed_at: state.last_updated });
  await assert.rejects((ha as any).climate('climate.other', 22), /allowlisted/);
  await assert.rejects((ha as any).climate('climate.room', 40), /range/);
  await assert.rejects((ha as any).media('media_player.room', { intent: 'NEXT', slots: {} }), /support/);
  assert.equal(posts.length, 5);
});

test('a light group preflights every capability before making one group request', async () => {
  const posts: any[] = [];
  let dimmable = false;
  const ha = createHomeAssistant({ ...config, entityIds: ['light.a', 'light.b'] }, async (url, init) => {
    if (init?.method === 'POST') { posts.push(JSON.parse(String(init.body))); return Response.json([]); }
    return Response.json({ ...state, entity_id: String(url).split('/').at(-1), attributes: { supported_color_modes: String(url).endsWith('light.a') || dimmable ? ['brightness'] : ['onoff'] } });
  });
  const commands = [{ entityId: 'light.a', action: 'brightness', value: 37 }, { entityId: 'light.b', action: 'brightness', value: 37 }];
  await assert.rejects((ha as any).controlLights(commands), /support/);
  assert.equal(posts.length, 0);
  dimmable = true;
  await (ha as any).controlLights(commands);
  assert.deepEqual(posts, [{ entity_id: ['light.a', 'light.b'], brightness_pct: 37 }]);
});
