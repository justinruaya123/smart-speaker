import assert from 'node:assert/strict';
import { test } from 'node:test';
import { homeAssistantView, spotifyView, playbackPosition } from './provider-state.js';

test('camera contacts use observed entities and distinguish unavailable cameras', () => {
  const view = homeAssistantView({ enabled: true, status: 'connected', entities: [
    { id: 'camera.front', name: 'Front camera', state: 'idle', stale: false, attributes: {} },
    { id: 'camera.back', name: 'Back camera', state: 'unavailable', stale: false, attributes: {} },
  ] }, null, null);
  assert.deepEqual(view.cameras, [
    { id: 'camera.front', name: 'Front camera', available: true },
    { id: 'camera.back', name: 'Back camera', available: false },
  ]);
});

test('live lights replace fixtures, preserve selection by entity, and normalize brightness', () => {
  const view = homeAssistantView({ enabled: true, status: 'connected', entities: [
    { id: 'light.second', name: 'Second', state: 'off', stale: false, attributes: {} },
    { id: 'light.hue_color_lamp_1', name: 'Ceiling', state: 'on', stale: false, attributes: { brightness: 128, rgb_color: [255, 0, 0], supported_color_modes: ['hs'] } },
  ] }, 'light.hue_color_lamp_1', null);
  assert.equal(view.selectedLight, 1);
  assert.equal(view.lights[1]?.brightness, 50);
  assert.equal(view.lights[1]?.color, 'Red');
  assert.equal(view.lights[0]?.brightness, null);
  assert.equal(view.lights.length, 2);
});

test('missing and stale lights are never interpreted as off or replaced by demos', () => {
  const empty = homeAssistantView({ enabled: true, status: 'connected', entities: [] }, 'light.deleted', null);
  assert.deepEqual(empty.lights, []);
  assert.equal(empty.mode, 'live');
  assert.equal(empty.current, null);
  const stale = homeAssistantView({ enabled: true, status: 'disconnected', entities: [{ id: 'light.a', name: 'A', state: 'on', stale: true, attributes: {} }] }, null, null);
  assert.equal(stale.lights[0]?.status, 'Unavailable');
  assert.equal(stale.lights[0]?.on, false);
  assert.equal(homeAssistantView({ enabled: false, status: 'disabled', entities: [] }, null, null).mode, 'demo');
});

test('color-temperature mode takes precedence over a retained RGB attribute', () => {
  const view = homeAssistantView({ enabled: true, status: 'connected', entities: [{ id: 'light.a', name: 'A', state: 'on', stale: false,
    attributes: { color_mode: 'color_temp', color_temp_kelvin: 2700, rgb_color: [0, 80, 255] } }] }, null, null);
  assert.equal(view.lights[0]?.color, 'Warm white');
  assert.equal(view.lights[0]?.colorCss, '#ffd6a1');
});

test('temperature selection stays read-only and unavailable sensors have no fabricated temperature', () => {
  const view = homeAssistantView({ enabled: true, status: 'connected', entities: [
    { id: 'sensor.room', name: 'Room', state: '77', stale: false, attributes: { device_class: 'temperature', unit_of_measurement: '°F' } },
  ] }, null, 'sensor.room');
  assert.equal(view.current, 25);
  assert.equal(view.temperatureSensor, 'sensor.room');
});

test('Spotify errors and idle playback clear old track data without enabling demo playback', () => {
  for (const status of ['not_authorized', 'idle', 'authentication', 'rate_limited']) {
    const view = spotifyView({ enabled: true, status, playback: null, updatedAt: 0 });
    assert.equal(view.mode, 'live');
    assert.equal(view.playing, false);
    assert.equal(view.mediaLoaded, false);
    assert.equal(view.position, 0);
    assert.equal(view.album, null);
  }
  assert.equal(spotifyView({ enabled: false, status: 'disabled', playback: null }).mode, 'demo');
});

test('live playback uses provider progress, clamps at the end, and never invents the next song', () => {
  const view = spotifyView({ enabled: true, status: 'connected', updatedAt: 1000, playback: {
    playing: true, positionMs: 8000, shuffle: true, repeat: 'context',
    track: { title: 'Live song', artist: 'Artist', album: 'Live album', artworkUrl: 'https://i.scdn.co/image/cover', durationMs: 10000 }, device: { name: 'Desk', volume: 0 },
  } });
  assert.equal(view.position, 8);
  assert.deepEqual(view.track, ['Live song', 'Artist', 10]);
  assert.equal(view.album, 'Live album');
  assert.equal(view.artworkUrl, 'https://i.scdn.co/image/cover');
  assert.equal(view.volume, 0);
  assert.equal(playbackPosition({ position: 8, duration: 10, playing: true, receivedAt: 1000 }, 6000), 10);
  assert.equal(playbackPosition({ position: 8, duration: 10, playing: false, receivedAt: 1000 }, 6000), 8);
});

test('artwork clears on missing metadata, unavailable playback and disabled Spotify', () => {
  const playback = { playing: true, positionMs: 0, shuffle: false, repeat: 'off', track: { title: 'Song', artist: 'Artist', artworkUrl: 'https://i.scdn.co/image/old', durationMs: 10000 } };
  assert.equal(spotifyView({ enabled: true, status: 'disconnected', playback }).artworkUrl, null);
  assert.equal(spotifyView({ enabled: false, status: 'disabled', playback }).artworkUrl, null);
  assert.equal(spotifyView({ enabled: true, status: 'connected', playback: { ...playback, track: { title: 'Song', artist: 'Artist', durationMs: 10000 } } }).artworkUrl, null);
});

test('album display clears on unavailable playback and accepts older snapshots without an album', () => {
  const playback = { playing: true, positionMs: 0, shuffle: false, repeat: 'off', track: { title: 'Song', artist: 'Artist', album: 'Previous album', durationMs: 10000 } };
  assert.equal(spotifyView({ enabled: true, status: 'disconnected', playback }).album, null);
  assert.equal(spotifyView({ enabled: false, status: 'disabled', playback }).album, null);
  assert.equal(spotifyView({ enabled: true, status: 'connected', playback: { ...playback, track: { title: 'Song', artist: 'Artist', durationMs: 10000 } } }).album, null);
});
