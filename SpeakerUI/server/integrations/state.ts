import { isRecord } from '../../shared/protocol.js';
import { IntegrationError } from './http.js';

export interface EntityAttributes {
  brightness?: number;
  rgb_color?: number[];
  color_temp_kelvin?: number;
  color_mode?: string;
  supported_color_modes?: string[];
  min_color_temp_kelvin?: number;
  max_color_temp_kelvin?: number;
  device_class?: string;
  unit_of_measurement?: string;
}

/** Only display attributes cross the gateway; arbitrary provider attributes stay private. */
export function lightAttributes(value: unknown): EntityAttributes {
  if (!isRecord(value)) return {};
  const result: EntityAttributes = {};
  if (typeof value.brightness === 'number' && Number.isFinite(value.brightness) && value.brightness >= 0 && value.brightness <= 255) result.brightness = value.brightness;
  if (Array.isArray(value.rgb_color) && value.rgb_color.length === 3 && value.rgb_color.every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 255)) result.rgb_color = [...value.rgb_color];
  for (const key of ['color_temp_kelvin', 'min_color_temp_kelvin', 'max_color_temp_kelvin'] as const) {
    const number = value[key];
    if (typeof number === 'number' && Number.isFinite(number) && number > 0) result[key] = number;
  }
  for (const key of ['color_mode', 'device_class', 'unit_of_measurement'] as const) {
    if (typeof value[key] === 'string') result[key] = value[key].slice(0, 80);
  }
  if (Array.isArray(value.supported_color_modes)) result.supported_color_modes = value.supported_color_modes.filter((mode): mode is string => typeof mode === 'string' && mode.length < 32).slice(0, 10);
  return result;
}

export interface Playback {
  playing: boolean;
  positionMs: number | null;
  shuffle: boolean;
  repeat: 'off' | 'track' | 'context';
  track: { title: string; artist: string; album: string | null; artworkUrl: string | null; durationMs: number } | null;
  device: { name: string; volume: number | null } | null;
}
export interface PlaybackSnapshot {
  enabled: boolean;
  status: string;
  playback: Playback | null;
  updatedAt: number;
}

function artworkUrl(images: unknown): string | null {
  if (!Array.isArray(images)) return null;
  for (const image of images) {
    if (!isRecord(image) || typeof image.url !== 'string' || image.url.length > 2048) continue;
    const url = URL.parse(image.url);
    if (url?.protocol === 'https:' && !url.username && !url.password) return url.href;
  }
  return null;
}

export function playbackState(value: unknown): Playback | null {
  if (value === undefined || value === null) return null;
  const invalid = () => new IntegrationError('invalid_response', 'Invalid Spotify playback response.');
  if (!isRecord(value) || typeof value.is_playing !== 'boolean' || typeof value.shuffle_state !== 'boolean' || !['off', 'track', 'context'].includes(String(value.repeat_state))) throw invalid();
  let track: Playback['track'] = null;
  if (value.item !== null && value.item !== undefined) {
    const item = value.item;
    if (!isRecord(item) || typeof item.name !== 'string' || typeof item.duration_ms !== 'number' || !Number.isFinite(item.duration_ms) || item.duration_ms < 0) throw invalid();
    const artist = item.type === 'episode' && isRecord(item.show) && typeof item.show.name === 'string' ? item.show.name :
      Array.isArray(item.artists) ? item.artists.filter(isRecord).map(a => a.name).filter((name): name is string => typeof name === 'string').join(', ') : '';
    let album: string | null = null;
    if (item.type !== 'episode' && isRecord(item.album) && typeof item.album.name === 'string') album = item.album.name.trim().slice(0, 200) || null;
    const artwork = item.type === 'episode'
      ? artworkUrl(item.images) ?? artworkUrl(isRecord(item.show) ? item.show.images : null)
      : artworkUrl(isRecord(item.album) ? item.album.images : null);
    track = { title: item.name.slice(0, 200), artist: artist.slice(0, 200), album, artworkUrl: artwork, durationMs: item.duration_ms };
  }
  let device: Playback['device'] = null;
  if (isRecord(value.device) && typeof value.device.name === 'string') {
    const volume = value.device.volume_percent;
    device = { name: value.device.name.slice(0, 200), volume: typeof volume === 'number' && Number.isFinite(volume) && volume >= 0 && volume <= 100 ? volume : null };
  }
  const position = value.progress_ms;
  const positionMs = typeof position === 'number' && Number.isFinite(position) && position >= 0 ? Math.min(position, track?.durationMs ?? position) : null;
  const repeat = value.repeat_state;
  if (repeat !== 'off' && repeat !== 'track' && repeat !== 'context') throw invalid();
  return { playing: value.is_playing, positionMs, shuffle: value.shuffle_state, repeat, track, device };
}
