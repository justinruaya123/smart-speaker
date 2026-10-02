import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { randomBytes } from 'node:crypto';
import type { Config } from '../config.js';
import { isRecord, MediaSource, type Track } from '../../shared/protocol.js';
import { IntegrationError, request, requireSuccess } from './http.js';
import { playbackState, type PlaybackSnapshot } from './state.js';
import type { SpotifyCommand } from './commands.js';

interface Tokens { accessToken: string; refreshToken: string; expiresAt: number }
export function spotifyFailure(status: number, body: unknown): IntegrationError {
  if (status === 429) {
    const quota = isRecord(body) && isRecord(body.error) && body.error.reason === 'QUOTA_EXCEEDED';
    return new IntegrationError(quota ? 'quota_exceeded' : 'rate_limited', quota ? 'Spotify quota is exhausted.' : 'Spotify is rate-limited.');
  }
  if (status === 401 || status === 403) return new IntegrationError('authentication', 'Spotify authorization, allowlist, or Premium access is unavailable.');
  if (status === 404) return new IntegrationError('no_device', 'No available Spotify playback device.');
  return new IntegrationError('provider_error', `Spotify returned HTTP ${status}.`);
}

export function createSpotify(config: Config['spotify'], dataDir: string, fetcher: typeof fetch) {
  let tokens: Tokens | undefined;
  let loaded = false;
  let loading: Promise<void> | undefined;
  let refreshing: Promise<string> | undefined;
  let retryAt = 0;
  let cooldownError = new IntegrationError('rate_limited', 'Spotify is rate-limited.');
  const tokenPath = join(dataDir, 'spotify-tokens.json');
  const pending = new Map<string, { binding: string; expiresAt: number }>();
  const selected = new Map<string, Track>();
  let playbackCache: PlaybackSnapshot | undefined;
  let playbackLoading: { revision: number; promise: Promise<PlaybackSnapshot> } | undefined;
  let playbackRevision = 0;
  let settlingUntil = 0;

  function enabledConfig() {
    if (!config.enabled) throw new IntegrationError('disabled', 'Spotify is disabled.');
    return config;
  }
  async function loadTokens() {
    if (loaded) return;
    if (!loading) loading = (async () => {
      try {
        const value: unknown = JSON.parse(await readFile(tokenPath, 'utf8'));
        if (!isRecord(value) || typeof value.accessToken !== 'string' || typeof value.refreshToken !== 'string' || typeof value.expiresAt !== 'number' || !Number.isFinite(value.expiresAt)) throw new Error('Invalid token file.');
        tokens = { accessToken: value.accessToken, refreshToken: value.refreshToken, expiresAt: value.expiresAt };
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new IntegrationError('token_storage', 'Spotify token file is unreadable. Reauthorize after fixing storage.');
      }
      loaded = true;
    })().finally(() => { loading = undefined; });
    await loading;
  }
  async function exchange(parameters: URLSearchParams): Promise<string> {
    const cfg = enabledConfig();
    const response = await request(fetcher, 'https://accounts.spotify.com/api/token', {
      method: 'POST', headers: { Authorization: `Basic ${Buffer.from(`${cfg.clientId}:${cfg.clientSecret}`).toString('base64')}`, 'Content-Type': 'application/x-www-form-urlencoded' }, body: parameters.toString(),
    });
    await requireSuccess(response);
    const body: unknown = await response.json();
    if (!isRecord(body) || typeof body.access_token !== 'string' || typeof body.expires_in !== 'number' || !Number.isFinite(body.expires_in) || body.expires_in <= 0) throw new IntegrationError('invalid_response', 'Invalid Spotify token response.');
    const refreshToken = typeof body.refresh_token === 'string' ? body.refresh_token : tokens?.refreshToken;
    if (!refreshToken) throw new IntegrationError('authentication', 'Spotify did not provide a refresh token.');
    tokens = { accessToken: body.access_token, refreshToken, expiresAt: Date.now() + body.expires_in * 1_000 };
    loaded = true;
    await mkdir(dataDir, { recursive: true, mode: 0o700 });
    const temporary = `${tokenPath}.${randomBytes(8).toString('hex')}.tmp`;
    await writeFile(temporary, JSON.stringify(tokens), { mode: 0o600, flag: 'wx' });
    await rename(temporary, tokenPath);
    return tokens.accessToken;
  }
  async function accessToken(): Promise<string> {
    enabledConfig(); await loadTokens();
    if (!tokens) throw new IntegrationError('not_authorized', 'Connect Spotify first.');
    if (tokens.expiresAt > Date.now() + 30_000) return tokens.accessToken;
    if (!refreshing) refreshing = exchange(new URLSearchParams({ grant_type: 'refresh_token', refresh_token: tokens.refreshToken })).finally(() => { refreshing = undefined; });
    return refreshing;
  }
  async function api(path: string, method = 'GET', body?: unknown): Promise<unknown> {
    enabledConfig();
    if (Date.now() < retryAt) throw cooldownError;
    const token = await accessToken();
    const response = await request(fetcher, `https://api.spotify.com/v1${path}`, {
      method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    }, method !== 'GET');
    // Player mutations return no data; a successful empty 200/202 is also valid.
    if (response.ok && method !== 'GET') { await response.body?.cancel(); return undefined; }
    let payload: unknown;
    try { payload = response.status === 204 ? undefined : await response.json(); }
    catch { if (response.ok) throw new IntegrationError('invalid_response', 'Spotify returned invalid JSON.', method !== 'GET'); }
    if (!response.ok) {
      const error = spotifyFailure(response.status, payload);
      if (response.status === 429) {
        const retryHeader = response.headers.get('retry-after');
        const seconds = retryHeader === null ? NaN : Number(retryHeader);
        const deadline = Number.isFinite(seconds) && seconds >= 0 ? Date.now() + seconds * 1_000 : Date.parse(retryHeader || '');
        retryAt = Number.isFinite(deadline) ? Math.max(Date.now() + 1_000, deadline) : Date.now() + 30_000;
        cooldownError = error;
      }
      if (response.status === 401 && tokens?.accessToken === token) tokens = { ...tokens, accessToken: '', expiresAt: 0 };
      throw error;
    }
    return payload;
  }
  return {
    enabled: config.enabled,
    async control(command: SpotifyCommand) {
      const cfg = enabledConfig();
      const query = new URLSearchParams();
      if (cfg.deviceId) query.set('device_id', cfg.deviceId);
      switch (command.action) {
        case 'seek': query.set('position_ms', String(command.value)); break;
        case 'volume': query.set('volume_percent', String(command.value)); break;
        case 'shuffle': case 'repeat': query.set('state', String(command.value)); break;
      }
      const suffix = query.size ? `?${query}` : '';
      try { await api(`/me/player/${command.action}${suffix}`, ['next', 'previous'].includes(command.action) ? 'POST' : 'PUT'); }
      finally { playbackRevision++; playbackCache = undefined; settlingUntil = Date.now() + 4000; }
    },
    async snapshot(): Promise<PlaybackSnapshot> {
      if (!config.enabled) return { enabled: false, status: 'disabled', playback: null, updatedAt: Date.now() };
      const cacheAge = Date.now() < settlingUntil ? 250 : 5000;
      if (playbackCache && Date.now() - playbackCache.updatedAt < cacheAge) return playbackCache;
      if (playbackLoading?.revision === playbackRevision) return playbackLoading.promise;
      const revision = playbackRevision;
      const promise = (async (): Promise<PlaybackSnapshot> => {
        let snapshot: PlaybackSnapshot;
        try {
          const playback = playbackState(await api('/me/player?additional_types=track,episode'));
          snapshot = { enabled: true, status: playback ? 'connected' : 'idle', playback, updatedAt: Date.now() };
        } catch (error) {
          if (!(error instanceof IntegrationError)) throw error;
          snapshot = { enabled: true, status: error.code, playback: null, updatedAt: Date.now() };
        }
        if (revision === playbackRevision) playbackCache = snapshot;
        return snapshot;
      })();
      playbackLoading = { revision, promise };
      try { return await promise; }
      finally { if (playbackLoading?.promise === promise) playbackLoading = undefined; }
    },
    authorize() {
      const cfg = enabledConfig();
      for (const [state, value] of pending) if (value.expiresAt < Date.now()) pending.delete(state);
      if (pending.size >= 16) throw new IntegrationError('busy', 'Too many pending authorization attempts.');
      const state = randomBytes(32).toString('hex');
      const binding = randomBytes(32).toString('hex');
      pending.set(state, { binding, expiresAt: Date.now() + 600_000 });
      const query = new URLSearchParams({ client_id: cfg.clientId, response_type: 'code', redirect_uri: cfg.redirectUri, state, scope: 'user-read-playback-state user-modify-playback-state' });
      return { url: `https://accounts.spotify.com/authorize?${query}`, binding };
    },
    async callback(state: string, binding: string | undefined, code: string | undefined) {
      const cfg = enabledConfig();
      const entry = pending.get(state); pending.delete(state);
      if (!entry || !binding || entry.binding !== binding || entry.expiresAt < Date.now()) throw new IntegrationError('invalid_state', 'Spotify authorization state is invalid or expired.');
      if (!code) throw new IntegrationError('authorization_denied', 'Spotify authorization was denied.');
      await exchange(new URLSearchParams({ grant_type: 'authorization_code', code, redirect_uri: cfg.redirectUri }));
      playbackCache = undefined;
    },
    async search(query: string): Promise<Track[]> {
      const value = await api(`/search?${new URLSearchParams({ q: query, type: 'track', limit: '5' })}`);
      if (!isRecord(value) || !isRecord(value.tracks) || !Array.isArray(value.tracks.items)) throw new IntegrationError('invalid_response', 'Invalid Spotify search response.');
      const tracks: Track[] = [];
      for (const item of value.tracks.items.slice(0,5)) {
        if (!isRecord(item) || typeof item.id !== 'string' || !/^[A-Za-z0-9]{22}$/.test(item.id) || typeof item.name !== 'string') continue;
        const artist = Array.isArray(item.artists) ? item.artists.filter(isRecord).map((a) => a.name).filter((name): name is string => typeof name === 'string').join(', ').slice(0,200) : undefined;
        const track: Track = { source: MediaSource.Spotify, id: item.id, title: item.name.slice(0,200), ...(artist ? { artist } : {}) };
        tracks.push(track); selected.delete(track.id); selected.set(track.id, track);
        if (selected.size > 100) { const oldest = selected.keys().next().value; if (oldest) selected.delete(oldest); }
      }
      return tracks;
    },
    find(id: string) { return selected.get(id); },
    async play(id: string) {
      const cfg = enabledConfig();
      if (!selected.has(id)) throw new IntegrationError('invalid_target', 'Search for and select a Spotify track first.');
      const query = cfg.deviceId ? `?${new URLSearchParams({ device_id: cfg.deviceId })}` : '';
      await api(`/me/player/play${query}`, 'PUT', { uris: [`spotify:track:${id}`] });
    },
    async playConfigured(uri: string) {
      const cfg=enabledConfig();
      if(!/^spotify:track:[A-Za-z0-9]{22}$/.test(uri))throw new IntegrationError('invalid_target','Configure an exact Spotify track URI.');
      const query=cfg.deviceId?`?${new URLSearchParams({device_id:cfg.deviceId})}`:'';
      await api(`/me/player/play${query}`,'PUT',{uris:[uri]});
      playbackRevision++;playbackCache=undefined;
    },
    async pause() {
      const cfg = enabledConfig();
      const query = cfg.deviceId ? `?${new URLSearchParams({ device_id: cfg.deviceId })}` : '';
      await api(`/me/player/pause${query}`, 'PUT');
    },
  };
}
export type Spotify = ReturnType<typeof createSpotify>;
