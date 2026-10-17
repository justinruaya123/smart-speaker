import assert from 'node:assert/strict';
import { test } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApp } from '../app.js';
import { readConfig } from '../config.js';
import type { InferenceFactory } from '../inference/adapter.js';
import type { ServerEvent } from '../../shared/protocol.js';

const config = () => readConfig({ APP_ORIGIN: 'http://127.0.0.1:3000', ACTIONS_ENABLED: 'true' }, '/tmp/speaker-fixtures-missing');
const headers = { host: '127.0.0.1:3000', origin: 'http://127.0.0.1:3000' };
const forbiddenFetch: typeof fetch = async () => { throw new Error('An offline test attempted outbound HTTP.'); };
async function nextEvent(events: ServerEvent[], predicate: (event: ServerEvent) => boolean) {
  for (let i = 0; i < 100; i++) {
    const event = events.find(predicate);
    if (event) return event;
    await delay(5);
  }
  assert.fail(`Expected event missing: ${JSON.stringify(events)}`);
}

test('disabled integrations do not connect; API rejects cross-origin changes', async () => {
  let calls = 0;
  const app = await createApp(config(), { fetcher: async () => { calls++; return forbiddenFetch('http://invalid'); } });
  try {
    const health = await app.inject({ url: '/api/health', headers });
    assert.equal(health.statusCode, 200);
    assert.equal(health.json().homeAssistant, 'disabled');
    const search = await app.inject({ url: '/api/media/search?q=piano', headers });
    assert.deepEqual(search.json().tracks, []);
    assert.match(search.json().fallbackReason, /disabled/);
    assert.equal((await app.inject({ method: 'POST', url: '/auth/spotify', headers: { ...headers, origin: 'https://other.invalid' } })).statusCode, 403);
    assert.equal((await app.inject({ method: 'POST', url: '/auth/spotify', headers })).statusCode, 503);
    assert.equal((await app.inject({ url: '/api/home-assistant/camera/camera.front/stream', headers })).statusCode, 503);
    assert.equal(calls, 0);
  } finally { await app.close(); }
});

test('live state remains readable while control routes enforce actions, origin and provider enablement', async () => {
  const app = await createApp({ ...config(), actionsEnabled: false }, { fetcher: forbiddenFetch });
  try {
    assert.equal((await app.inject({ url: '/api/spotify/state', headers })).json().status, 'disabled');
    assert.equal((await app.inject({ url: '/api/home-assistant/entities', headers })).json().enabled, false);
    for (const [url, payload] of [['/api/home-assistant/control', { entityId: 'light.desk', action: 'on' }], ['/api/spotify/control', { action: 'play' }]] as const) {
      assert.equal((await app.inject({ method: 'POST', url, headers, payload })).statusCode, 409);
      assert.equal((await app.inject({ method: 'POST', url, headers: { ...headers, origin: 'https://other.invalid' }, payload })).statusCode, 403);
    }
  } finally { await app.close(); }
  const enabledActions = await createApp(config(), { fetcher: forbiddenFetch });
  try {
    assert.equal((await enabledActions.inject({ method: 'POST', url: '/api/spotify/control', headers, payload: { action: 'play' } })).statusCode, 503);
    assert.equal((await enabledActions.inject({ method: 'POST', url: '/api/home-assistant/control', headers, payload: { entityId: 'light.desk', action: 'on' } })).statusCode, 503);
    assert.equal((await enabledActions.inject({ method: 'POST', url: '/api/spotify/control', headers, payload: { action: 'volume', value: 101 } })).statusCode, 400);
  } finally { await enabledActions.close(); }
});

test('streamed PCM yields partial/final mock events and cannot execute real actions', async () => {
  const app = await createApp(config(), { fetcher: forbiddenFetch });
  await app.ready();
  const ws = await app.injectWS('/ws', { headers });
  const events: ServerEvent[] = [];
  ws.on('message', (data) => events.push(JSON.parse(data.toString())));
  try {
    ws.send(JSON.stringify({ type: 'start', utteranceId: 'one' }));
    await nextEvent(events, (event) => event.type === 'state' && event.state === 'recording');
    for (let i = 0; i < 25; i++) ws.send(Buffer.alloc(640));
    ws.send(JSON.stringify({ type: 'end', utteranceId: 'one' }));
    const action = await nextEvent(events, (event) => event.type === 'action');
    assert.equal(action.type === 'action' && action.status, 'simulated');
    assert.equal(events.filter((event) => event.type === 'inference').length, 2);
    ws.send(JSON.stringify({ type: 'end', utteranceId: 'one' }));
    ws.send(JSON.stringify({ type: 'start', utteranceId: 'one' }));
    await nextEvent(events, (event) => event.type === 'error' && event.code === 'reused_id');
    assert.equal(events.filter((event) => event.type === 'action').length, 1);
  } finally { ws.terminate(); await app.close(); }
});

test('cancel ignores late inference and disconnect releases the global recording lease', async () => {
  let emit: (value: unknown) => void = () => {};
  const factory: InferenceFactory = (callback) => { emit = callback; return { simulated: true, start() {}, writeAudio() {}, end() {}, cancel() {} }; };
  const app = await createApp(config(), { fetcher: forbiddenFetch, inferenceFactory: factory });
  await app.ready();
  const first = await app.injectWS('/ws', { headers });
  const events: ServerEvent[] = [];
  first.on('message', (data) => events.push(JSON.parse(data.toString())));
  try {
    first.send(JSON.stringify({ type: 'start', utteranceId: 'cancelled' }));
    await nextEvent(events, (event) => event.type === 'state' && event.state === 'recording');
    first.send(JSON.stringify({ type: 'cancel', utteranceId: 'cancelled' }));
    await nextEvent(events, (event) => event.type === 'state' && event.state === 'idle');
    emit({ utteranceId: 'cancelled', revision: 2, final: true, intent: 'media.search', parameters: { query: 'demo' }, confidence: 1 });
    await delay(10);
    assert.equal(events.some((event) => event.type === 'action'), false);
    first.send(JSON.stringify({ type: 'start', utteranceId: 'disconnecting' }));
    await nextEvent(events, (event) => event.type === 'state' && event.utteranceId === 'disconnecting' && event.state === 'recording');
    first.terminate();
    const second = await app.injectWS('/ws', { headers });
    const secondEvents: ServerEvent[] = [];
    second.on('message', (data) => secondEvents.push(JSON.parse(data.toString())));
    second.send(JSON.stringify({ type: 'start', utteranceId: 'fresh' }));
    await nextEvent(secondEvents, (event) => event.type === 'state' && event.state === 'recording');
    second.terminate();
  } finally { first.terminate(); await app.close(); }
});

test('slow inference has a bounded queue and malformed audio cancels', async () => {
  const factory: InferenceFactory = () => ({ simulated: true, start() {}, writeAudio: () => new Promise<void>(() => {}), end() {}, cancel() {} });
  const app = await createApp(config(), { inferenceFactory: factory, fetcher: forbiddenFetch });
  await app.ready(); const ws = await app.injectWS('/ws', { headers }); const events: ServerEvent[] = [];
  ws.on('message', (data) => events.push(JSON.parse(data.toString())));
  try {
    ws.send(JSON.stringify({ type: 'start', utteranceId: 'slow' }));
    await nextEvent(events, (event) => event.type === 'state' && event.state === 'recording');
    for (let i = 0; i < 51; i++) ws.send(Buffer.alloc(640));
    await nextEvent(events, (event) => event.type === 'error' && event.code === 'audio_limit');
    ws.send(JSON.stringify({ type: 'start', utteranceId: 'invalid' }));
    await nextEvent(events, (event) => event.type === 'state' && event.utteranceId === 'invalid' && event.state === 'recording');
    ws.send(Buffer.alloc(3));
    await nextEvent(events, (event) => event.type === 'error' && event.code === 'invalid_audio');
  } finally { ws.terminate(); await app.close(); }
});

test('local HTTP playback honors ranges and does not expose arbitrary file paths', async () => {
  const root = await mkdtemp(join(tmpdir(), 'speaker-http-'));
  await writeFile(join(root, 'Piano.mp3'), '0123456789');
  const cfg = { ...config(), mediaDir: root };
  const app = await createApp(cfg, { fetcher: forbiddenFetch });
  try {
    const search = await app.inject({ url: '/api/media/search?q=piano', headers });
    const id = search.json().tracks[0].id;
    const range = await app.inject({ url: `/api/media/local/${id}`, headers: { ...headers, range: 'bytes=2-5' } });
    assert.equal(range.statusCode, 206); assert.equal(range.body, '2345');
    assert.equal(range.headers['content-range'], 'bytes 2-5/10');
    const invalid = await app.inject({ url: `/api/media/local/${id}`, headers: { ...headers, range: 'bytes=20-' } });
    assert.equal(invalid.statusCode, 416);
    assert.equal((await app.inject({ url: '/api/media/local/unknown', headers })).statusCode, 404);
  } finally { await app.close(); await rm(root, { recursive: true, force: true }); }
});

test('configuration fails closed for unknown inference and unsafe credential settings', () => {
  assert.throws(() => readConfig({ INFERENCE_MODE: 'remote', INFERENCE_URL: 'http://example.invalid' }, '/tmp'), /service contract/);
  assert.throws(() => readConfig({ INTENT_MIN_CONFIDENCE: 'NaN' }, '/tmp'), /CONFIDENCE/);
  assert.throws(() => readConfig({ HA_ENABLED: 'true' }, '/tmp'), /HA_URL/);
  assert.throws(() => readConfig({ HA_ENTITY_IDS: 'switch.other' }, '/tmp'), /HA_ENTITY_IDS/);
});

test('ambient fixture endpoint loads dummy runtime, validates input and never calls providers', async () => {
  let outbound = 0;
  const app = await createApp(config(), { fetcher: async () => { outbound++; return forbiddenFetch('http://invalid'); } });
  try {
    const health = (await app.inject({ url: '/api/health', headers })).json();
    assert.equal(health.runtime.engine, 'dummy-onnx');
    assert.equal(health.runtime.loaded, true);
    assert.equal(health.runtime.nativeSession, false);
    const result = await app.inject({ method: 'POST', url: '/api/ambient/infer', headers, payload: {fixture:3} });
    assert.equal(result.statusCode, 200);
    assert.equal(result.json().result.intent, 'timer.set');
    assert.equal(result.json().simulated, true);
    for (const payload of [{fixture:15}, {fixture:'3'}, {fixture:0, intent:'light.on'}, {}]) {
      assert.equal((await app.inject({method:'POST',url:'/api/ambient/infer',headers,payload})).statusCode,400);
    }
    assert.equal((await app.inject({method:'POST',url:'/api/ambient/infer',headers:{...headers,origin:'https://other.invalid'},payload:{fixture:0}})).statusCode,403);
    assert.equal(outbound,0);
  } finally { await app.close(); }
});

test('canonical touch actions persist jobs and dedup receipts, validate slots, and enforce the action switch', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'iris-actions-'));
  const cfg = { ...config(), dataDir: directory };
  let app = await createApp(cfg, { fetcher: forbiddenFetch });
  const payload = { request_id: 'timer-once', command: { intent: 'TIMER', slots: { duration_seconds: 123 } } };
  try {
    const result = await app.inject({ method: 'POST', url: '/api/actions', headers, payload });
    assert.equal(result.statusCode, 200);
    assert.equal(result.json().status, 'local');
    const again = await app.inject({ method: 'POST', url: '/api/actions', headers, payload });
    assert.equal(again.json().id, result.json().id);
    assert.equal((await app.inject({ url: '/api/actions/state', headers })).json().jobs.length, 1);
    assert.equal((await app.inject({ method: 'POST', url: '/api/actions', headers, payload: { ...payload, command: { intent: 'TIMER', slots: { duration_seconds: 1.2 } } } })).statusCode, 400);
    assert.equal((await app.inject({ method: 'POST', url: '/api/actions', headers: { ...headers, origin: 'https://other.invalid' }, payload })).statusCode, 403);
    await app.close();
    app = await createApp(cfg, { fetcher: forbiddenFetch });
    assert.equal((await app.inject({ url: '/api/actions/state', headers })).json().jobs[0].status, 'scheduled');
    await app.inject({ method: 'POST', url: '/api/actions', headers, payload });
    assert.equal((await app.inject({ url: '/api/actions/state', headers })).json().jobs.length, 1);
  } finally { await app.close(); await rm(directory, { recursive: true, force: true }); }
  const disabled = await createApp({ ...cfg, actionsEnabled: false }, { fetcher: forbiddenFetch });
  try { assert.equal((await disabled.inject({ method: 'POST', url: '/api/actions', headers, payload })).json().status, 'disabled'); }
  finally { await disabled.close(); await rm(directory, { recursive: true, force: true }); }
});

test('local media elects one player, records acknowledgment once, and duplicate touch requests do not replay playback', async () => {
  const directory=await mkdtemp(join(tmpdir(),'iris-local-player-'));
  await writeFile(join(directory,'Song.mp3'),'fixture payload; no audio playback in this server test');
  const cfg=readConfig({APP_ORIGIN:headers.origin,ACTIONS_ENABLED:'true',DATA_DIR:directory,LOCAL_MEDIA_DIR:directory,
    SMART_SPEAKER_BINDINGS:JSON.stringify({media_player:{provider:'local'}})},directory);
  const app=await createApp(cfg,{fetcher:forbiddenFetch});
  const events:any[][]=[[],[]];const sockets:any[]=[];
  try {
    const search=(await app.inject({url:'/api/media/search?q=Song',headers})).json();
    cfg.bindings.songs['song.dynamite']=search.tracks[0].id;
    for(let index=0;index<2;index++)sockets.push(await app.injectWS('/ws/actions',{headers},{onInit(socket){socket.on('message',bytes=>events[index]!.push(JSON.parse(bytes.toString())));}}));
    const first=events[0]!.find(e=>e.event==='action_state'),second=events[1]!.find(e=>e.event==='action_state');
    assert.equal(first.local_player,true);assert.equal(second.local_player,false);
    const payload={request_id:'local-once',command:{intent:'PLAY_MUSIC',slots:{song_id:'song.dynamite'}}};
    const receipt=(await app.inject({method:'POST',url:'/api/actions',headers,payload})).json();
    assert.equal(receipt.status,'requested');
    assert.equal((await app.inject({method:'POST',url:'/api/actions/local-receipt',headers,payload:{id:receipt.id,client_id:second.client_id,status:'succeeded'}})).statusCode,400);
    assert.equal((await app.inject({method:'POST',url:'/api/actions/local-receipt',headers,payload:{id:receipt.id,client_id:first.client_id,status:'succeeded'}})).json().status,'succeeded');
    const requestedBefore=events[0]!.filter(e=>e.event==='execution'&&e.execution.status==='requested').length;
    const duplicate=(await app.inject({method:'POST',url:'/api/actions',headers,payload})).json();
    assert.equal(duplicate.replayed,true);assert.equal(duplicate.status,'succeeded');
    await delay(10);
    assert.equal(events[0]!.filter(e=>e.event==='execution'&&e.execution.status==='requested').length,requestedBefore);
  } finally {for(const socket of sockets)socket.terminate();await app.close();await rm(directory,{recursive:true,force:true});}
});
