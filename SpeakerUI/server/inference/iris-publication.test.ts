import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RuntimeActions } from './runtime-actions.js';
import { readConfig } from '../config.js';
import { createApp } from '../app.js';
import { EventEmitter } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { irisUseCases } from '../../shared/canonical-command.js';
import { setTimeout as delay } from 'node:timers/promises';

const inventory = JSON.parse(readFileSync(new URL('../../../submission/models/optionb-tiny/commands.json', import.meta.url), 'utf8')).slice(1);
const lamp = 'light.hue_color_lamp_1';
const camera = 'camera.dining_area_main_camera_tc70_mainstream';
const config = { enabled: true, publication: true, profile: 'owner-protected', bindings: { lights: {}, songs: {}, contacts: { 'contact.home': 'script.not_a_phone' } } };
function event(command: unknown, id = 'u') { return { event: 'command', is_final: true, session_id: 's', utterance_id: id,
  source: 'microphone', actions_enabled: true, owner_result: { accepted: true }, application: { accepted: true, command } }; }
function providers(calls: any[]) { return { light: async (c: any) => { calls.push(['light', c]); }, spotify: async () => {}, song: async () => {},
  camera: async (id: string) => { calls.push(['camera', id]); return { entity_id: id, video_only: true, stream_url: `/api/home-assistant/camera/${id}/stream` }; },
  script: async () => assert.fail('CALL must not run a telephone script') }; }

test('publication routing is explicitly opt-in and preserves the exact entity allowlist and owner profile', () => {
  assert.equal(readConfig({}, '.').publication, false);
  const cfg = readConfig({ IRIS_PUBLICATION: 'true', HA_ENTITY_IDS: `${lamp},light.dining_light,${camera}` }, '.');
  assert.equal(cfg.publication, true);
  assert.deepEqual(cfg.ha.entityIds, [lamp, 'light.dining_light', camera]);
  assert.equal(readConfig({ IRIS_PUBLICATION: 'true', INFERENCE_MODE: 'onnx', SMART_SPEAKER_PYTHON: 'python', SMART_SPEAKER_ROOT: '.', SMART_SPEAKER_BUNDLE: '.', SMART_SPEAKER_TOKEN: 'test' }, '.').runtime?.profile, 'owner-protected');
  assert.throws(() => readConfig({ IRIS_PUBLICATION: 'yes' }, '.'), /IRIS_PUBLICATION/);
});

test('every frozen light command targets only the Hue lamp, including recognized named IDs', async () => {
  const calls: any[] = [], actions = new RuntimeActions(config, providers(calls));
  const lights = inventory.filter((command: any) => ['LIGHT_ON', 'LIGHT_OFF', 'BRIGHTNESS', 'COLOR'].includes(command.intent));
  assert.equal(lights.length, 8);
  for (const [index, command] of [...lights, { intent: 'LIGHT_ON', slots: { light_id: 'light.kitchen' } }].entries()) {
    const result = await actions.handle(event(command, String(index)));
    assert.equal(result?.status, 'succeeded');
  }
  assert.equal(calls.length, 9);
  for (const [, command] of calls) assert.equal(command.entityId, lamp);
  assert.deepEqual(calls.find(([, c]) => c.action === 'brightness')[1], { entityId: lamp, action: 'brightness', value: 100 });
  assert.deepEqual(calls.find(([, c]) => c.action === 'color')[1].rgb, [0, 0, 255]);
});

test('recognized CALL opens the dining camera video only, never a telephone or script', async () => {
  const calls: any[] = [], actions = new RuntimeActions(config, providers(calls));
  for (const [index, id] of ['contact.home', 'contact.mom'].entries()) {
    const result = await actions.handle(event({ intent: 'CALL', slots: { contact_id: id } }, String(index)));
    assert.equal(result?.status, 'observed');
    assert.equal(result?.camera.entity_id, camera);
    assert.equal(result?.camera.video_only, true);
  }
  assert.deepEqual(calls, [['camera', camera], ['camera', camera]]);
});

test('publication routes reject malformed slots, owner failures, replays and tentative events before any provider', async () => {
  const calls: any[] = [], actions = new RuntimeActions(config, providers(calls));
  for (const command of [{ intent: 'LIGHT_ON', slots: {} }, { intent: 'COLOR', slots: { light_id: 'light.all', color: 'invalid' } },
    { intent: 'CALL', slots: { contact_id: 'light.all' } }, { intent: 'CALL', slots: { contact_id: 'contact.home', text: 'hello' } }])
    assert.equal((await actions.handle(event(command, JSON.stringify(command))))?.status, 'rejected');
  const good = event({ intent: 'LIGHT_ON', slots: { light_id: 'light.all' } });
  assert.equal((await actions.handle({ ...good, source: 'wav' }))?.status, 'preview');
  assert.equal((await actions.handle({ ...good, actions_enabled: false }))?.status, 'disabled');
  assert.equal((await actions.handle({ ...good, owner_result: { accepted: false } }))?.status, 'rejected');
  assert.equal(await actions.handle({ ...good, event: 'tentative', is_final: false }), undefined);
  assert.equal(calls.length, 0);
});

test('the actual gateway uses HA light capability checks and same-origin dining camera proxies without leaking credentials', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'iris-routes-'));
  const calls: { url: string; body?: any }[] = [];
  const cfg = readConfig({ IRIS_PUBLICATION: 'true', ACTIONS_ENABLED: 'true', HA_ENABLED: 'true', HA_URL: 'http://offline.invalid', HA_TOKEN: 'offline-test-token',
    HA_ENTITY_IDS: `${lamp},light.dining_light,${camera}`, DATA_DIR: directory }, directory);
  const socket = Object.assign(new EventEmitter(), { send() {}, close() {}, terminate() {}, ping() {} });
  const app = await createApp(cfg, { haSocketFactory: () => socket, fetcher: async (input, init) => {
    const url = String(input); calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    if (url.includes('/api/states/')) return Response.json({ entity_id: lamp, state: 'on', attributes: { supported_color_modes: ['rgb'] } });
    if (url.includes('/api/services/')) return new Response(null, { status: 200 });
    if (url.includes('/api/camera_proxy/')) return new Response(new Uint8Array([255, 216, 255]), { headers: { 'Content-Type': 'image/jpeg' } });
    throw new Error('Unexpected offline URL');
  } });
  const headers = { host: '127.0.0.1:3000', origin: 'http://127.0.0.1:3000' };
  try {
    const light = await app.inject({ method: 'POST', url: '/api/actions', headers, payload: { request_id: 'hue', command: { intent: 'BRIGHTNESS', slots: { light_id: 'light.all', percent: 20 } } } });
    assert.equal(light.json().status, 'succeeded');
    assert.deepEqual(calls.find(c => c.url.includes('/api/services/'))?.body, { entity_id: lamp, brightness_pct: 20 });
    const beforeCamera = calls.length;
    assert.equal((await app.inject({ url: `/api/home-assistant/camera/${camera}`, headers })).statusCode, 409, 'Direct camera reads need explicit canonical CALL admission');
    assert.equal((await app.inject({ url: `/api/home-assistant/camera/${camera}/stream`, headers })).statusCode, 409);
    assert.equal(calls.length, beforeCamera);
    const view = (await app.inject({ method: 'POST', url: '/api/actions', headers, payload: { request_id: 'camera', command: { intent: 'CALL', slots: { contact_id: 'contact.home' } } } })).json();
    assert.equal(view.status, 'observed');
    assert.ok(view.camera.stream_url.startsWith(`/api/home-assistant/camera/${camera}/stream?admission=`));
    assert.equal(view.camera.video_only, true);
    assert.equal(JSON.stringify(view).includes('offline-test-token'), false);
    assert.ok(calls.some(c => c.url.endsWith('/api/camera_proxy/' + camera)));
    const state = (await app.inject({ url: '/api/actions/state', headers })).json();
    assert.equal(state.publication, true);
    assert.equal(state.bindings.lights['light.all'], lamp);
    assert.deepEqual(state.bindings.contacts, ['contact.home']);
    const before = calls.length;
    assert.equal((await app.inject({ method: 'POST', url: '/api/actions', headers: { ...headers, origin: 'https://other.invalid' }, payload: { request_id: 'foreign', command: { intent: 'LIGHT_ON', slots: { light_id: 'light.all' } } } })).statusCode, 403);
    assert.equal(calls.length, before);
    assert.equal((await app.inject({ url: '/api/home-assistant/camera/camera.not_allowed', headers })).statusCode, 409);
    const frame = await app.inject({ url: view.camera.snapshot_url, headers });
    assert.equal(frame.statusCode, 200);
    cfg.actionsEnabled = false;
    const beforeDisabled = calls.length;
    assert.equal((await app.inject({ url: view.camera.snapshot_url, headers })).statusCode, 409);
    assert.equal(calls.length, beforeDisabled);
  } finally { await app.close(); await rm(directory, { recursive: true, force: true }); }
});

test('all 31 frozen commands have real local or explicit synthetic completions across the ten categories', async () => {
  assert.equal(inventory.length, 31);
  assert.equal(new Set(inventory.map((c: unknown) => JSON.stringify(c))).size, 31);
  assert.equal(irisUseCases.length,10);
  for(const command of inventory)assert.equal(irisUseCases.filter(category=>category.intents.includes(command.intent as never)).length,1,JSON.stringify(command));
  const unavailable = async () => { throw new (await import('../integrations/http.js')).IntegrationError('disabled', 'Offline integration fixture.'); };
  const actions = new RuntimeActions({ ...config, bindings: { lights: {}, songs: {} } }, { light: unavailable, spotify: unavailable, song: unavailable });
  const local = new Set(['TIME', 'TIMER', 'ALARM', 'CREATE_REMINDER', 'LIST_REMINDERS']);
  for (const [index, command] of inventory.entries()) {
    const result = await actions.handle(event(command, 'catalog-' + index));
    assert.equal(result?.status, local.has(command.intent) ? 'local' : 'synthetic', JSON.stringify(command));
    assert.equal(result?.provenance.source, local.has(command.intent) ? 'local' : 'synthetic');
  }
  assert.equal(actions.store.snapshot().jobs.length, 9, 'three real timers, three alarms and three reminders');
  assert.equal(actions.store.snapshot().jobs.filter(j => j.kind === 'alarm').every(j => j.timezone === 'Asia/Manila'), true);
});

test('publication WEATHER selects OpenWeather before HA and keeps private API configuration out of action state', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'iris-weather-route-'));
  const cfg = readConfig({ IRIS_PUBLICATION: 'true', ACTIONS_ENABLED: 'true', DATA_DIR: directory,
    OPENWEATHER_ENABLED: 'true', OPENWEATHER_API_KEY: 'offline-secret-fixture', OPENWEATHER_LATITUDE: '14.6', OPENWEATHER_LONGITUDE: '121.1', OPENWEATHER_LOCATION: 'Metro Manila',
    SMART_SPEAKER_BINDINGS: JSON.stringify({ weather: { default: 'weather.not_used' } }) }, directory);
  let calls = 0;
  const app = await createApp(cfg, { fetcher: async url => {
    calls++; assert.equal(new URL(String(url)).hostname, 'api.openweathermap.org');
    return Response.json({ dt: Math.floor(Date.now() / 1000), main: { temp: 29 }, weather: [{ description: 'light rain' }] });
  } });
  const headers = { host: '127.0.0.1:3000', origin: 'http://127.0.0.1:3000' };
  try {
    const result = (await app.inject({ method: 'POST', url: '/api/actions', headers, payload: { request_id: 'weather', command: { intent: 'WEATHER', slots: {} } } })).json();
    assert.equal(result.status, 'observed'); assert.equal(result.provenance.source, 'openweather');
    assert.equal(calls, 1);
    const state = (await app.inject({ url: '/api/actions/state', headers })).json();
    assert.equal(JSON.stringify(state).includes('offline-secret-fixture'), false);
    assert.equal(JSON.stringify(state).includes('14.6'), false);
  } finally { await app.close(); await rm(directory, { recursive: true, force: true }); }
});

test('an admitted video stream is revoked when actions are disabled, without any provider mutation', async () => {
  const directory=await mkdtemp(join(tmpdir(),'iris-camera-revocation-'));
  const cfg=readConfig({IRIS_PUBLICATION:'true',ACTIONS_ENABLED:'true',HA_ENABLED:'true',HA_URL:'http://offline.invalid',HA_TOKEN:'offline-fixture',HA_ENTITY_IDS:camera,DATA_DIR:directory},directory);
  let upstreamAborted=false;
  const socket=Object.assign(new EventEmitter(),{send(){},close(){},terminate(){},ping(){}});
  const app=await createApp(cfg,{haSocketFactory:()=>socket,fetcher:async(url,init)=>{
    assert.equal(init?.method,undefined,'Camera admission is read-only');
    if(String(url).includes('/api/camera_proxy_stream/'))return new Response(new ReadableStream({start(controller){
      controller.enqueue(new TextEncoder().encode('--fixture\r\n'));
      init?.signal?.addEventListener('abort',()=>{upstreamAborted=true;try{controller.close();}catch{}},{once:true});
    }}),{headers:{'Content-Type':'multipart/x-mixed-replace; boundary=fixture'}});
    return new Response(new Uint8Array([255,216,255]),{headers:{'Content-Type':'image/jpeg'}});
  }});
  const cancel=new AbortController();
  try{
    await app.listen({host:'127.0.0.1',port:0});cfg.port=(app.server.address() as any).port;cfg.origin=`http://127.0.0.1:${cfg.port}`;
    const headers={host:new URL(cfg.origin).host,origin:cfg.origin};
    const receipt=(await app.inject({method:'POST',url:'/api/actions',headers,payload:{request_id:'admit',command:{intent:'CALL',slots:{contact_id:'contact.home'}}}})).json();
    const stream=await fetch(cfg.origin+receipt.camera.stream_url,{headers:{origin:cfg.origin},signal:cancel.signal});
    const reader=stream.body!.getReader();await reader.read();
    cfg.actionsEnabled=false;
    for(let i=0;i<150&&!upstreamAborted;i++)await delay(10);
    assert.equal(upstreamAborted,true,'Disabling actions must close the existing upstream video, not merely reject new views');
    await reader.cancel().catch(()=>{});
  }finally{cancel.abort();await app.close();await rm(directory,{recursive:true,force:true});}
});
