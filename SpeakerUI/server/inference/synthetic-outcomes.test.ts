import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RuntimeActions } from './runtime-actions.js';
import { IntegrationError } from '../integrations/http.js';
import { readConfig } from '../config.js';
import { createApp } from '../app.js';
import { EventEmitter } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ActionStore } from './action-store.js';

const bindings = { lights: {}, songs: {}, climate: { default: 'climate.real_device' } };
const event = (intent: string, slots: Record<string, any>, id = intent) => ({ event: 'command', is_final: true, session_id: 's', utterance_id: id,
  source: 'microphone', actions_enabled: true, owner_result: { accepted: true }, application: { accepted: true, command: { intent, slots } } });
const forbidden = async () => assert.fail('Synthetic target cannot call any real provider');

test('every temperature class completes only a synthetic target even with a real climate binding', async () => {
  const actions = new RuntimeActions({ enabled: true, publication: true, profile: 'owner-protected', bindings }, { light: forbidden, spotify: forbidden, song: forbidden, climate: forbidden });
  for (const degrees of [18, 22, 26]) {
    const result = await actions.handle(event('TEMPERATURE', { degrees, unit: 'C' }, String(degrees)));
    assert.equal(result?.status, 'synthetic');
    assert.deepEqual(result?.provenance, { source: 'synthetic', synthetic: true, provider_accepted: false, device_state_confirmed: false });
    assert.equal(result?.provider_accepted, false);
    assert.match(result!.detail, /Synthetic.*no device changed/i);
    assert.deepEqual(result?.synthetic_state.thermostat, { target: degrees, unit: 'C' });
    assert.equal(result?.observed_temperature, undefined);
  }
});

test('the configured observed home sensor stays read-only and independent of a synthetic target', async () => {
  assert.throws(() => readConfig({ HA_TEMPERATURE_SENSOR_ID: 'sensor.guessed' }, '.'), /HA_TEMPERATURE_SENSOR_ID/);
  assert.throws(() => readConfig({ HA_TEMPERATURE_SENSOR_ID: 'climate.room', HA_ENTITY_IDS: 'climate.room' }, '.'), /HA_TEMPERATURE_SENSOR_ID/);
  const directory = await mkdtemp(join(tmpdir(), 'iris-temperature-'));
  const calls: any[] = [];
  const cfg = readConfig({ IRIS_PUBLICATION: 'true', ACTIONS_ENABLED: 'true', HA_ENABLED: 'true', HA_URL: 'http://offline.invalid', HA_TOKEN: 'offline-fixture',
    HA_ENTITY_IDS: 'sensor.real_room,climate.real_device', HA_TEMPERATURE_SENSOR_ID: 'sensor.real_room', DATA_DIR: directory,
    SMART_SPEAKER_BINDINGS: JSON.stringify(bindings) }, directory);
  const socket = Object.assign(new EventEmitter(), { send() {}, close() {}, terminate() {}, ping() {} });
  const app = await createApp(cfg, { haSocketFactory: () => socket, fetcher: async (url, init) => {
    calls.push([String(url), init?.method || 'GET']);
    assert.equal(init?.method || 'GET', 'GET');
    assert.ok(String(url).endsWith('/api/states/sensor.real_room'));
    return Response.json({ entity_id: 'sensor.real_room', state: '29.5', last_updated: '2026-10-23T09:00:00Z',
      attributes: { device_class: 'temperature', unit_of_measurement: '°C', private_data: 'not public' } });
  } });
  const headers = { host: '127.0.0.1:3000', origin: 'http://127.0.0.1:3000' };
  try {
    const result = (await app.inject({ method: 'POST', url: '/api/actions', headers, payload: { request_id: 'target', command: { intent: 'TEMPERATURE', slots: { degrees: 22, unit: 'C' } } } })).json();
    assert.equal(result.status, 'synthetic');
    assert.equal(calls.length, 0);
    const read = (await app.inject({ url: '/api/home-assistant/entities', headers })).json();
    assert.equal(read.observed_temperature.status, 'observed');
    assert.equal(read.observed_temperature.temperature, 29.5);
    assert.equal(read.observed_temperature.provenance.source, 'home_assistant');
    assert.equal(read.observed_temperature.observed_at, '2026-10-23T09:00:00Z');
    assert.equal(JSON.stringify(read).includes('not public'), false);
    assert.equal(result.synthetic_state.thermostat.target, 22);
    assert.equal(calls.length, 1);
  } finally { await app.close(); await rm(directory, { recursive: true, force: true }); }
});

test('known unavailable integrations complete labeled simulations, never provider acknowledgments or observations', async () => {
  const unavailable = async () => { throw new IntegrationError('disabled', 'Integration is not configured.'); };
  const actions = new RuntimeActions({ enabled: true, publication: true, bindings: { lights: {}, songs: {} } },
    { light: unavailable, spotify: unavailable, song: unavailable, camera: unavailable });
  for (const [intent, slots] of [['LIGHT_ON', { light_id: 'light.all' }], ['PLAY_MUSIC', {}], ['MESSAGE', {}], ['CALL', { contact_id: 'contact.home' }]] as const) {
    const result = await actions.handle(event(intent, slots));
    assert.equal(result?.status, 'synthetic');
    assert.equal(result?.provenance.source, 'synthetic');
    assert.equal(result?.provider_accepted, false);
    assert.equal(result?.device_state_confirmed, false);
    assert.match(result!.detail, /Synthetic — no device changed/);
  }
  assert.equal(actions.syntheticState().lights['light.hue_color_lamp_1'].on, true);
  assert.equal(actions.syntheticState().playback.state, 'playing');
  const result = actions.store.receipt('s:CALL');
  assert.equal(result?.camera.video_only, true);
  assert.equal(result?.camera.stream_url, null);
  assert.equal(result?.camera.available, false);
  assert.equal(actions.store.receipt('s:MESSAGE')?.notification.delivered, false);
});

test('unknown outcomes never fall back, retry or replace separately observed/synthetic state', async () => {
  let calls = 0;
  const actions = new RuntimeActions({ enabled: true, publication: true, bindings: { lights: {}, songs: {} } }, {
    light: async () => { calls++; throw new IntegrationError('unavailable', 'Outcome unknown.', true); }, spotify: forbidden, song: forbidden });
  const command = event('LIGHT_ON', { light_id: 'light.all' });
  assert.equal((await actions.handle(command))?.status, 'unknown');
  assert.equal(await actions.handle(command), undefined);
  assert.equal(calls, 1);
  assert.deepEqual(actions.syntheticState(), {});
  for (const patch of [{ source: 'wav' }, { actions_enabled: false }, { application: { accepted: false, command: null } }])
    await actions.handle({ ...event('TEMPERATURE', { degrees: 22, unit: 'C' }, JSON.stringify(patch)), ...patch });
  assert.deepEqual(actions.syntheticState(), {});
  const disabled = new RuntimeActions({ enabled: false, publication: true, bindings }, { light: forbidden, spotify: forbidden, song: forbidden });
  assert.equal((await disabled.handle(event('TEMPERATURE', { degrees: 22, unit: 'C' })))?.status, 'disabled');
  assert.deepEqual(disabled.syntheticState(), {});
});

test('real accepted requests retain provider provenance without claiming an observed device state', async () => {
  const actions = new RuntimeActions({ enabled: true, publication: true, bindings: { lights: {}, songs: {} } }, { light: async () => {}, spotify: forbidden, song: forbidden });
  const receipt = await actions.handle(event('LIGHT_ON', { light_id: 'light.all' }));
  assert.equal(receipt?.status, 'succeeded');
  assert.deepEqual(receipt?.provenance, { source: 'home_assistant', synthetic: false, provider_accepted: true, device_state_confirmed: false });
  assert.deepEqual(actions.syntheticState(), {});
});

test('synthetic target state hydrates from durable receipts after restart without creating an observation or replaying an action', async () => {
  const directory=await mkdtemp(join(tmpdir(),'iris-synthetic-restart-'));
  const cfg={enabled:true,publication:true,bindings};
  const providers={light:forbidden,spotify:forbidden,song:forbidden,climate:forbidden};
  try {
    const first=new RuntimeActions(cfg,providers,await ActionStore.open(directory));
    await first.handle(event('TEMPERATURE',{degrees:22,unit:'C'}));
    for(let i=0;i<32;i++)await first.handle(event('TIME',{},'time-'+i));
    const second=new RuntimeActions(cfg,providers,await ActionStore.open(directory));
    assert.deepEqual(second.syntheticState().thermostat,{target:22,unit:'C'});
    assert.equal(second.syntheticState().observed_temperature,undefined);
    assert.equal(await second.handle(event('TEMPERATURE',{degrees:22,unit:'C'})),undefined);
  }finally{await rm(directory,{recursive:true,force:true});}
});
