import assert from 'node:assert/strict';
import { test } from 'node:test';
import { lightAttributes, playbackState } from './state.js';

test('light state exposes only valid display attributes, preserving zero brightness', () => {
  assert.deepEqual(lightAttributes({ brightness: 0, rgb_color: [255, 0, 80], color_temp_kelvin: 2700, supported_color_modes: ['hs', 'color_temp'], access_token: 'private' }), {
    brightness: 0, rgb_color: [255, 0, 80], color_temp_kelvin: 2700, supported_color_modes: ['hs', 'color_temp'],
  });
  assert.deepEqual(lightAttributes({ brightness: 256, rgb_color: [1, -2, 3], color_temp_kelvin: NaN, supported_color_modes: 'rgb' }), {});
});

test('playback state bounds progress and strips provider fields unrelated to the display', () => {
  const result = playbackState({ is_playing: false, progress_ms: 300000, shuffle_state: true, repeat_state: 'track',
    item: { type: 'track', name: 'Song', duration_ms: 180000, artists: [{ name: 'Artist' }], album: { name: 'Album title', private_field: 'secret' }, private_field: 'secret' },
    device: { name: 'Desk', volume_percent: 0, id: 'private-device' }, access_token: 'secret' });
  assert.deepEqual(result, { playing: false, positionMs: 180000, shuffle: true, repeat: 'track',
    track: { title: 'Song', artist: 'Artist', album: 'Album title', artworkUrl: null, durationMs: 180000 }, device: { name: 'Desk', volume: 0 } });
});

test('playback keeps album artwork from Spotify without passing through unrelated image fields', () => {
  const result = playbackState({ is_playing: true, shuffle_state: false, repeat_state: 'off',
    item: { type: 'track', name: 'Song', duration_ms: 5000, album: { images: [
      { url: 'https://i.scdn.co/image/cover', width: 640, height: 640, private_field: 'secret' },
      { url: 'https://i.scdn.co/image/small', width: 64, height: 64 },
    ] } } });
  assert.equal(result?.track?.artworkUrl, 'https://i.scdn.co/image/cover');
  assert.ok(!JSON.stringify(result).includes('secret'));
});

test('absent or malformed artwork keeps playback usable and skips unsafe image URLs', () => {
  for (const images of [undefined, null, [], 'broken', [null, {}, { url: 42 }],
    [{ url: 'not a url' }], [{ url: 'javascript:alert(1)' }], [{ url: 'data:image/png;base64,abc' }],
    [{ url: 'http://i.scdn.co/image/cover' }], [{ url: 'https://secret@i.scdn.co/image/cover' }]]) {
    const result = playbackState({ is_playing: true, shuffle_state: false, repeat_state: 'off',
      item: { type: 'track', name: 'Song', duration_ms: 5000, album: { images } } });
    assert.equal(result?.track?.title, 'Song');
    assert.equal(result?.track?.artworkUrl, null);
  }
  const result = playbackState({ is_playing: true, shuffle_state: false, repeat_state: 'off',
    item: { type: 'track', name: 'Song', duration_ms: 5000, album: { images: [{ url: 'invalid' }, { url: 'https://i.scdn.co/image/valid' }] } } });
  assert.equal(result?.track?.artworkUrl, 'https://i.scdn.co/image/valid');
});

test('episodes use their own artwork and fall back to show artwork when absent', () => {
  const episode = { type: 'episode', name: 'Episode', duration_ms: 5000, show: { name: 'Podcast', images: [{ url: 'https://i.scdn.co/image/show' }] } };
  for (const [images, expected] of [[undefined, 'show'], [[{ url: 'https://i.scdn.co/image/episode' }], 'episode']] as const) {
    const result = playbackState({ is_playing: true, shuffle_state: false, repeat_state: 'off', item: { ...episode, images } });
    assert.equal(result?.track?.artworkUrl, `https://i.scdn.co/image/${expected}`);
  }
});

test('missing or invalid album metadata does not discard an otherwise playable track', () => {
  for (const album of [undefined, null, {}, { name: 42 }, { name: '  ' }]) {
    const result = playbackState({ is_playing: true, shuffle_state: false, repeat_state: 'off',
      item: { type: 'track', name: 'Song', duration_ms: 5000, artists: [{ name: 'Artist' }], album } });
    assert.equal(result?.track?.title, 'Song');
    assert.equal(result?.track?.album, null);
  }
});

test('no playback and unavailable metadata stay absent instead of inventing a track', () => {
  assert.equal(playbackState(undefined), null);
  assert.equal(playbackState(null), null);
  const result = playbackState({ is_playing: true, item: null, progress_ms: null, shuffle_state: false, repeat_state: 'off', device: null });
  assert.equal(result?.track, null);
  assert.equal(result?.positionMs, null);
  assert.equal(result?.device, null);
  assert.throws(() => playbackState({ is_playing: 'yes' }), /playback/);
});

test('episodes use the show name and malformed tracks are rejected', () => {
  const result = playbackState({ is_playing: true, progress_ms: 1000, shuffle_state: false, repeat_state: 'off',
    item: { type: 'episode', name: 'Episode', duration_ms: 5000, show: { name: 'Podcast' } } });
  assert.equal(result?.track?.artist, 'Podcast');
  assert.equal(result?.track?.album, null);
  assert.throws(() => playbackState({ is_playing: true, shuffle_state: false, repeat_state: 'off', item: { name: 'Broken' } }), /playback/);
});
