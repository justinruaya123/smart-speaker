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
