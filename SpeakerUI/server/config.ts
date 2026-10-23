import { resolve } from 'node:path';
import { isRecord } from '../shared/protocol.js';

export interface ActionBindings {
  lights: Record<string, string | string[]>;
  songs: Record<string, string>;
  climate?: Record<string, string>;
  weather?: Record<string, string>;
  media_player?: { provider?: 'home_assistant' | 'local' | 'spotify'; entity_id?: string };
  contacts?: Record<string, string>;
  messages?: Record<string, string>;
}

function actionBindings(input: string | undefined): ActionBindings {
  const invalid = () => new Error('Invalid action bindings. Use explicit canonical IDs, HA entities and a supported media provider.');
  let value: unknown;
  try { value = JSON.parse(input || '{}'); } catch { throw invalid(); }
  if (!isRecord(value) || Object.keys(value).some(key => !['lights', 'songs', 'climate', 'weather', 'media_player', 'contacts', 'messages'].includes(key))) throw invalid();
  const result: Record<string, any> = {};
  for (const key of ['lights', 'songs', 'climate', 'weather', 'contacts', 'messages']) {
    const map = value[key] ?? {};
    if (!isRecord(map) || Object.keys(map).length > 100) throw invalid();
    result[key] = {};
    for (const [id, target] of Object.entries(map)) {
      const role = key === 'lights' ? 'light' : key === 'songs' ? 'song' : key === 'contacts' ? 'contact' : undefined;
      if (role ? !new RegExp('^' + role + '\\.[a-z0-9_.-]+$').test(id) : id !== 'default') throw invalid();
      const targets = key === 'lights' && Array.isArray(target) ? target : [target];
      if (!targets.length || targets.length > 100 || targets.some(item => typeof item !== 'string' || !item || item.length > 2048)) throw invalid();
      const domain = key === 'lights' ? 'light' : key === 'contacts' || key === 'messages' ? 'script' : key;
      if (key !== 'songs' && targets.some(item => !new RegExp('^' + domain + '\\.[a-z0-9_]+$').test(String(item)))) throw invalid();
      result[key][id] = Array.isArray(target) ? [...new Set(targets)] : target;
    }
  }
  const media = value.media_player ?? {};
  if (!isRecord(media) || Object.keys(media).some(key => !['provider', 'entity_id'].includes(key))) throw invalid();
  if (Object.keys(media).length && !['home_assistant', 'local', 'spotify'].includes(String(media.provider))) throw invalid();
  if (media.provider === 'home_assistant' ? typeof media.entity_id !== 'string' || !/^media_player\.[a-z0-9_]+$/.test(media.entity_id) : media.entity_id !== undefined) throw invalid();
  result.media_player = { ...media };
  return result as unknown as ActionBindings;
}

function enabled(value: string | undefined, key: string): boolean {
  if (value === undefined || value === '' || value === 'false') return false;
  if (value === 'true') return true;
  throw new Error(`${key} must be true or false.`);
}
function required(value: string | undefined, key: string): string {
  if (!value?.trim()) throw new Error(`${key} is required when this integration is enabled.`);
  return value.trim();
}
function httpUrl(value: string, key: string): string {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error(`${key} must be an HTTP(S) URL without credentials, query, or fragment.`);
  return url.toString().replace(/\/$/, '');
}

export function readConfig(env: NodeJS.ProcessEnv, cwd: string) {
  const bindings = actionBindings(env.SMART_SPEAKER_BINDINGS);
  const timezone = (env.TIMEZONE || 'Asia/Manila').trim();
  try {
    if (/^[+-]/.test(timezone)) throw new Error('Offset is not an IANA zone');
    new Intl.DateTimeFormat('en', { timeZone: timezone });
  } catch { throw new Error('TIMEZONE must be a supported IANA timezone.'); }
  const port = Number(env.PORT || 3000);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid PORT.');
  const origin = httpUrl(env.APP_ORIGIN || `http://127.0.0.1:${port}`, 'APP_ORIGIN');
  if (new URL(origin).origin !== origin) throw new Error('APP_ORIGIN must have no path.');
  const inferenceMode = env.INFERENCE_MODE || 'mock';
  if (inferenceMode !== 'mock' && inferenceMode !== 'onnx') throw new Error('Unsupported inference service contract.');
  const profile = env.RECOGNITION_PROFILE || 'owner-protected';
  if (!['owner-protected','benchmark-compatible'].includes(profile)) throw new Error('Unsupported RECOGNITION_PROFILE.');
  const runtime = inferenceMode === 'onnx' ? {
    python: required(env.SMART_SPEAKER_PYTHON, 'SMART_SPEAKER_PYTHON'),
    root: required(env.SMART_SPEAKER_ROOT, 'SMART_SPEAKER_ROOT'),
    bundle: required(env.SMART_SPEAKER_BUNDLE, 'SMART_SPEAKER_BUNDLE'),
    threads: Number(env.SMART_SPEAKER_THREADS || 1), token: required(env.SMART_SPEAKER_TOKEN, 'SMART_SPEAKER_TOKEN'), profile,
  } : undefined;
  if (runtime && (!Number.isInteger(runtime.threads) || runtime.threads < 1 || runtime.threads > 8)) throw new Error('Threads must be 1..8.');
  const threshold = env.INTENT_MIN_CONFIDENCE?.trim() ? Number(env.INTENT_MIN_CONFIDENCE) : undefined;
  if (threshold !== undefined && (!Number.isFinite(threshold) || threshold < 0 || threshold > 1)) throw new Error('INTENT_MIN_CONFIDENCE must be between 0 and 1.');
  const haEnabled = enabled(env.HA_ENABLED, 'HA_ENABLED');
  const spotifyEnabled = enabled(env.SPOTIFY_ENABLED, 'SPOTIFY_ENABLED');
  const entityIds = [...new Set((env.HA_ENTITY_IDS || '').split(',').map((id) => id.trim()).filter(Boolean))];
  if (entityIds.length > 100 || entityIds.some((id) => !/^(light|camera|sensor|climate|weather|media_player|script)\.[a-z0-9_]+$/.test(id))) throw new Error('HA_ENTITY_IDS supports at most 100 light/camera/sensor/climate/weather/media_player/script IDs.');
  const tls = env.TLS_CERT_PATH || env.TLS_KEY_PATH ? { cert: required(env.TLS_CERT_PATH, 'TLS_CERT_PATH'), key: required(env.TLS_KEY_PATH, 'TLS_KEY_PATH') } : undefined;
  const ha = haEnabled ? { enabled: true as const, url: httpUrl(required(env.HA_URL, 'HA_URL'), 'HA_URL'), token: required(env.HA_TOKEN, 'HA_TOKEN'), entityIds } : { enabled: false as const, entityIds };
  let spotify;
  if (spotifyEnabled) {
    const redirectUri = httpUrl(required(env.SPOTIFY_REDIRECT_URI, 'SPOTIFY_REDIRECT_URI'), 'SPOTIFY_REDIRECT_URI');
    const redirect = new URL(redirectUri);
    if (redirect.protocol !== 'https:' && !['127.0.0.1', '[::1]'].includes(redirect.hostname)) throw new Error('Spotify redirect requires HTTPS or an explicit loopback IP.');
    if (redirect.origin !== origin || redirect.pathname !== '/auth/spotify/callback') throw new Error('Spotify redirect must be APP_ORIGIN + /auth/spotify/callback.');
    spotify = { enabled: true as const, clientId: required(env.SPOTIFY_CLIENT_ID, 'SPOTIFY_CLIENT_ID'), clientSecret: required(env.SPOTIFY_CLIENT_SECRET, 'SPOTIFY_CLIENT_SECRET'), redirectUri, deviceId: env.SPOTIFY_DEVICE_ID || undefined };
  } else spotify = { enabled: false as const };
  return { host: env.HOST || '127.0.0.1', port, origin, tls, inferenceMode, runtime, threshold, bindings, timezone, publication: enabled(env.IRIS_PUBLICATION, 'IRIS_PUBLICATION'), actionsEnabled: enabled(env.ACTIONS_ENABLED, 'ACTIONS_ENABLED'), ha, spotify, mediaDir: resolve(cwd, env.LOCAL_MEDIA_DIR || 'media'), dataDir: resolve(cwd, env.DATA_DIR || 'data') };
}
export type Config = ReturnType<typeof readConfig>;
