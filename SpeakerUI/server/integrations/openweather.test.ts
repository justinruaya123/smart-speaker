import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createOpenWeather } from './openweather.js';
import { readConfig } from '../config.js';

const settings = { enabled: true as const, apiKey: 'offline-key-not-real', latitude: 14.123456, longitude: 121.987654, location: 'Metro Manila' };
const observation = { dt: 1792701000, main: { temp: 27.5, humidity: 82 }, weather: [{ main: 'Rain', description: 'light rain' }],
  coord: { lat: 14.123456, lon: 121.987654 }, name: 'private exact station' };

test('OpenWeather makes a bounded server-only metric request and exposes only a sanitized observation', async () => {
  let calls = 0;
  const now = Date.UTC(2026, 9, 23, 9);
  const weather = createOpenWeather(settings, async (input, init) => {
    calls++;
    const url = new URL(String(input));
    assert.equal(url.origin + url.pathname, 'https://api.openweathermap.org/data/2.5/weather');
    assert.equal(url.searchParams.get('appid'), settings.apiKey);
    assert.equal(url.searchParams.get('units'), 'metric');
    assert.equal(init?.redirect, 'error');
    assert.ok(init?.signal);
    return Response.json(observation);
  }, () => now);
  const result = await weather.read();
  assert.equal(calls, 1);
  assert.equal(result.status, 'observed');
  assert.equal(result.provenance.source, 'openweather');
  assert.equal(result.weather.temperature, 27.5);
  assert.equal(result.weather.location, 'Metro Manila');
  assert.equal(result.weather.observed_at, new Date(observation.dt * 1000).toISOString());
  assert.equal(result.weather.fetched_at, new Date(now).toISOString());
  for (const secret of [settings.apiKey, '14.123456', '121.987654', observation.name]) assert.equal(JSON.stringify(result).includes(secret), false);
});

test('cached weather keeps its actual times and age through cache hits and offline failure, never today relabeling', async () => {
  let now = Date.UTC(2026, 9, 23, 9), calls = 0;
  const weather = createOpenWeather(settings, async () => { if (++calls > 1) throw new Error('Offline'); return Response.json(observation); }, () => now);
  const first = await weather.read();
  now += 60000;
  const hit = await weather.read();
  assert.equal(hit.status, 'cached');
  assert.equal(calls, 1);
  now += 86400000;
  const stale = await weather.read();
  assert.equal(calls, 2);
  assert.equal(stale.status, 'cached');
  assert.equal(stale.weather.observed_at, first.weather.observed_at);
  assert.equal(stale.weather.fetched_at, first.weather.fetched_at);
  assert.equal(stale.weather.age_seconds, 86460);
  assert.equal(stale.weather.stale, true);
  assert.doesNotMatch(stale.detail, /today|current weather/i);
});

test('unconfigured or failed weather is visibly synthetic with null observation/retrieval times and no invented live location', async () => {
  let calls = 0;
  const disabled = createOpenWeather({ enabled: false }, async () => { calls++; assert.fail('Disabled weather cannot fetch'); });
  const fallback = await disabled.read();
  assert.equal(calls, 0);
  for (const result of [fallback, await createOpenWeather(settings, async () => new Response('private provider error', { status: 401 })).read(),
    await createOpenWeather(settings, async () => Response.json({ main: { temp: NaN } })).read()]) {
    assert.equal(result.status, 'synthetic');
    assert.equal(result.provenance.synthetic, true);
    assert.equal(result.provenance.provider_accepted, false);
    assert.equal(result.weather.observed_at, null);
    assert.equal(result.weather.fetched_at, null);
    assert.equal(result.weather.location, 'Illustrative sample');
    assert.match(result.detail, /Synthetic — no device changed/);
    assert.equal(JSON.stringify(result).includes('private provider error'), false);
  }
});

test('OpenWeather configuration is explicit, validates coordinates and never substitutes a guessed home sensor or HA weather', () => {
  assert.equal(readConfig({}, '.').openweather.enabled, false);
  for (const env of [{ OPENWEATHER_ENABLED: 'true' }, { OPENWEATHER_ENABLED: 'true', OPENWEATHER_API_KEY: 'fixture', OPENWEATHER_LATITUDE: '91', OPENWEATHER_LONGITUDE: '0' }])
    assert.throws(() => readConfig(env, '.'), /OPENWEATHER/);
  const cfg = readConfig({ OPENWEATHER_ENABLED: 'true', OPENWEATHER_API_KEY: 'fixture', OPENWEATHER_LATITUDE: '14.6', OPENWEATHER_LONGITUDE: '121', OPENWEATHER_LOCATION: 'Metro Manila' }, '.');
  assert.equal(cfg.openweather.enabled, true);
  assert.equal(cfg.temperatureSensor, undefined);
});
