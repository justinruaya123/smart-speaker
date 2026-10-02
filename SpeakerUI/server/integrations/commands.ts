import { isRecord } from '../../shared/protocol.js';
import { IntegrationError } from './http.js';

export type LightCommand = { entityId: string } & (
  { action: 'on' | 'off' } | { action: 'brightness'; value: number } |
  { action: 'color'; rgb: number[] } | { action: 'temperature'; value: number }
);
export type SpotifyCommand = { action: 'play' | 'pause' | 'next' | 'previous' } |
  { action: 'seek' | 'volume'; value: number } | { action: 'shuffle'; value: boolean } |
  { action: 'repeat'; value: 'off' | 'track' | 'context' };

export function lightCommand(value: unknown): LightCommand {
  const invalid = () => new IntegrationError('invalid_request', 'Invalid light command.');
  if (!isRecord(value) || typeof value.entityId !== 'string' || !/^light\.[a-z0-9_]+$/.test(value.entityId)) throw invalid();
  const { entityId, action } = value;
  const keys = action === 'color' ? ['entityId', 'action', 'rgb'] : action === 'brightness' || action === 'temperature' ? ['entityId', 'action', 'value'] : ['entityId', 'action'];
  if (Object.keys(value).some(key => !keys.includes(key))) throw invalid();
  if (action === 'on' || action === 'off') return { entityId, action };
  if (action === 'brightness' && typeof value.value === 'number' && Number.isInteger(value.value) && value.value >= 0 && value.value <= 100) return { entityId, action, value: value.value };
  if (action === 'temperature' && typeof value.value === 'number' && Number.isInteger(value.value) && value.value >= 1000 && value.value <= 40000) return { entityId, action, value: value.value };
  if (action === 'color' && Array.isArray(value.rgb) && value.rgb.length === 3 && value.rgb.every(v => Number.isInteger(v) && v >= 0 && v <= 255)) return { entityId, action, rgb: value.rgb };
  throw invalid();
}

export function spotifyCommand(value: unknown): SpotifyCommand {
  const invalid = () => new IntegrationError('invalid_request', 'Invalid Spotify command.');
  if (!isRecord(value)) throw invalid();
  const action = value.action;
  if (action === 'play' || action === 'pause' || action === 'next' || action === 'previous') {
    if (Object.keys(value).length !== 1) throw invalid();
    return { action };
  }
  if (Object.keys(value).some(key => key !== 'action' && key !== 'value')) throw invalid();
  if ((action === 'seek' || action === 'volume') && typeof value.value === 'number' && Number.isSafeInteger(value.value) && value.value >= 0 && (action === 'seek' || value.value <= 100)) return { action, value: value.value };
  if (action === 'shuffle' && typeof value.value === 'boolean') return { action, value: value.value };
  if (action === 'repeat' && (value.value === 'off' || value.value === 'track' || value.value === 'context')) return { action, value: value.value };
  throw invalid();
}
