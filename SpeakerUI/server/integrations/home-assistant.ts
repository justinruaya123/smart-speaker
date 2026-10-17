import WebSocket from 'ws';
import type { Config } from '../config.js';
import { isRecord } from '../../shared/protocol.js';
import { IntegrationError, request, requireSuccess } from './http.js';
import { lightAttributes, type EntityAttributes } from './state.js';
import type { LightCommand } from './commands.js';
import type { CanonicalCommand } from '../../shared/canonical-command.js';

interface EntityState { id: string; name: string; state: string; updatedAt: string; stale: boolean; attributes: EntityAttributes }
export interface HomeAssistantSocket {
  on(event: 'message', listener: (data: Buffer) => void): void;
  on(event: 'open' | 'close' | 'error' | 'pong', listener: () => void): void;
  send(data: string): void;
  close(): void;
  terminate(): void;
  ping(): void;
}
const openSocket = (url: URL): HomeAssistantSocket => new WebSocket(url, { handshakeTimeout: 5_000, maxPayload: 256 * 1024 });

export function createHomeAssistant(config: Config['ha'], fetcher: typeof fetch, connectSocket = openSocket) {
  const cache = new Map<string, EntityState>();
  const allowed = new Set(config.entityIds);
  let socket: HomeAssistantSocket | undefined;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let stopped = true;
  let authenticated = false;
  let attempts = 0;
  let status = config.enabled ? 'disconnected' : 'disabled';

  function enabledConfig() {
    if (!config.enabled) throw new IntegrationError('disabled', 'Home Assistant is disabled.');
    return config;
  }
  function checkEntity(id: string, domain: string) {
    if (!allowed.has(id) || !id.startsWith(`${domain}.`)) throw new IntegrationError('invalid_target', 'Entity is not allowlisted for this operation.');
  }
  async function observe(id: string, domain: string) {
    const cfg = enabledConfig(); checkEntity(id, domain);
    const response = await request(fetcher, `${cfg.url}/api/states/${encodeURIComponent(id)}`, { headers: { Authorization: `Bearer ${cfg.token}` } });
    await requireSuccess(response);
    const value: unknown = await response.json();
    if (!isRecord(value) || value.entity_id !== id || typeof value.state !== 'string' || !isRecord(value.attributes)) throw new IntegrationError('invalid_response', 'Home Assistant returned invalid entity state.');
    if (['unavailable', 'unknown'].includes(value.state)) throw new IntegrationError('unavailable', 'The configured Home Assistant entity is unavailable.');
    return { state: value.state, attributes: value.attributes, observed_at: typeof value.last_updated === 'string' ? value.last_updated : null };
  }
  async function service(domain: string, name: string, ids: string[], data: Record<string, unknown> = {}) {
    const cfg = enabledConfig();
    for (const id of ids) checkEntity(id, domain);
    const response = await request(fetcher, `${cfg.url}/api/services/${domain}/${name}`, {
      method: 'POST', headers: { Authorization: `Bearer ${cfg.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ entity_id: ids.length === 1 ? ids[0] : ids, ...data }),
    }, true);
    if (response.status >= 500) { await response.body?.cancel(); throw new IntegrationError('provider_error', 'Home Assistant returned an error after dispatch; the outcome is unknown. Do not automatically retry.', true); }
    await requireSuccess(response); await response.body?.cancel();
    // Hydration is independent evidence; its failure cannot turn an accepted write into a second request.
    if (socket && authenticated) void hydrate(socket, ids);
  }
  function supports(attributes: Record<string, unknown>, feature: number) {
    return typeof attributes.supported_features === 'number' && Number.isSafeInteger(attributes.supported_features) && (attributes.supported_features & feature) === feature;
  }
  const unsupported = () => new IntegrationError('unsupported_capability', 'The configured entity does not report support for this action.');
  async function controlLights(commands: LightCommand[]) {
    if (!commands.length) throw new IntegrationError('invalid_request', 'Empty light group.');
    const first = commands[0]!;
    const signature = ({ entityId: _id, ...action }: LightCommand) => JSON.stringify(action);
    if (commands.some(command => signature(command) !== signature(first))) throw new IntegrationError('invalid_request', 'A light group requires one shared action.');
    for (const command of commands) {
      const { attributes: a } = await observe(command.entityId, 'light');
      const modes = Array.isArray(a.supported_color_modes) ? a.supported_color_modes : [];
      if (command.action === 'brightness' && command.value !== 0 && !modes.some(mode => !['onoff', 'unknown'].includes(String(mode)))) throw unsupported();
      if (command.action === 'color' && !modes.some(mode => ['hs', 'xy', 'rgb', 'rgbw', 'rgbww'].includes(String(mode)))) throw unsupported();
      if (command.action === 'temperature' && (!modes.includes('color_temp') ||
        typeof a.min_color_temp_kelvin === 'number' && command.value < a.min_color_temp_kelvin ||
        typeof a.max_color_temp_kelvin === 'number' && command.value > a.max_color_temp_kelvin)) throw unsupported();
    }
    const data: Record<string, unknown> = {};
    let name = first.action === 'off' || first.action === 'brightness' && first.value === 0 ? 'turn_off' : 'turn_on';
    if (first.action === 'brightness' && first.value !== 0) data.brightness_pct = first.value;
    if (first.action === 'color') data.rgb_color = first.rgb;
    if (first.action === 'temperature') data.color_temp_kelvin = first.value;
    await service('light', name, [...new Set(commands.map(command => command.entityId))], data);
  }
  function update(value: unknown) {
    if (!isRecord(value) || typeof value.entity_id !== 'string' || !allowed.has(value.entity_id) || typeof value.state !== 'string' || typeof value.last_updated !== 'string') return;
    const timestamp = Date.parse(value.last_updated);
    if (!Number.isFinite(timestamp)) return;
    const previous = cache.get(value.entity_id);
    if (previous && Date.parse(previous.updatedAt) > timestamp) return;
    const name = isRecord(value.attributes) && typeof value.attributes.friendly_name === 'string' ? value.attributes.friendly_name : value.entity_id;
    cache.set(value.entity_id, { id: value.entity_id, name: name.slice(0,200), state: value.state.slice(0,200), updatedAt: value.last_updated, stale: !authenticated, attributes: lightAttributes(value.attributes) });
  }
  async function hydrate(connection: HomeAssistantSocket, ids: Iterable<string> = allowed) {
    const cfg = enabledConfig();
    for (const id of ids) {
      if (stopped || socket !== connection || !authenticated) return;
      try {
        const response = await request(fetcher, `${cfg.url}/api/states/${encodeURIComponent(id)}`, { headers: { Authorization: `Bearer ${cfg.token}` } });
        if (stopped || socket !== connection || !authenticated) { await response.body?.cancel().catch(() => {}); return; }
        if (response.status === 404) { await response.body?.cancel().catch(() => {}); if (socket === connection && authenticated) cache.delete(id); continue; }
        await requireSuccess(response);
        const value: unknown = await response.json();
        if (socket === connection && authenticated) update(value);
      } catch {
        if (stopped || socket !== connection || !authenticated) return;
        const previous = cache.get(id);
        if (previous) previous.stale = true;
        if (socket === connection) status = 'degraded';
      }
    }
  }
  function connect() {
    if (!config.enabled || stopped || allowed.size === 0) return;
    const url = new URL(`${config.url}/api/websocket`);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    const connection = connectSocket(url);
    socket = connection;
    let authFailed = false;
    let alive = true;
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    const setupDeadline = setTimeout(() => { if (socket === connection && !stopped) connection.terminate(); }, 10_000);
    setupDeadline.unref();
    connection.on('open', () => {
      heartbeat = setInterval(() => {
        if (!alive) { connection.terminate(); return; }
        alive = false; connection.ping();
      }, 15_000);
      heartbeat.unref();
    });
    connection.on('pong', () => { alive = true; });
    status = 'connecting';
    connection.on('message', (bytes) => {
      if (socket !== connection || stopped) return;
      try {
        const message: unknown = JSON.parse(bytes.toString());
        if (!isRecord(message)) return;
        if (message.type === 'auth_required') connection.send(JSON.stringify({ type: 'auth', access_token: config.token }));
        if (message.type === 'auth_invalid') { authFailed = true; status = 'authentication_failed'; connection.close(); }
        if (message.type === 'auth_ok') {
          status = 'subscribing';
          connection.send(JSON.stringify({ id: 1, type: 'subscribe_events', event_type: 'state_changed' }));
        }
        if (message.type === 'result' && message.id === 1) {
          if (message.success === true) { clearTimeout(setupDeadline); authenticated = true; attempts = 0; status = 'connected'; void hydrate(connection); }
          else { status = 'subscription_failed'; authFailed = true; connection.close(); }
        }
        if (message.type === 'event' && isRecord(message.event) && isRecord(message.event.data)) {
          const data = message.event.data;
          if (data.new_state === null && typeof data.entity_id === 'string') cache.delete(data.entity_id);
          else update(data.new_state);
        }
      } catch { status = 'invalid_response'; connection.close(); }
    });
    connection.on('error', () => { status = 'disconnected'; });
    connection.on('close', () => {
      clearTimeout(setupDeadline); clearInterval(heartbeat);
      if (socket !== connection) return;
      authenticated = false;
      for (const entity of cache.values()) entity.stale = true;
      if (!authFailed) status = 'disconnected';
      if (!stopped && !authFailed) { retry = setTimeout(connect, Math.min(30_000, 1_000 * 2 ** Math.min(attempts++, 5))); retry.unref(); }
    });
  }
  return {
    start() { if (!config.enabled || !stopped) return; stopped = false; if (allowed.size === 0) status = 'no_entities_selected'; else connect(); },
    close() { stopped = true; authenticated = false; clearTimeout(retry); socket?.terminate(); for (const entity of cache.values()) entity.stale = true; },
    snapshot() { return { enabled: config.enabled, status, entities: [...cache.values()].map((entity) => ({ ...entity, attributes: structuredClone(entity.attributes) })) }; },
    controlLights,
    async climate(id: string, degrees: number) {
      const { attributes: a } = await observe(id, 'climate');
      if (!supports(a, 1)) throw unsupported();
      const unit = a.temperature_unit;
      if (!['°C', 'C', '°F', 'F'].includes(String(unit))) throw new IntegrationError('unsupported_capability', 'Climate temperature unit is not reported.');
      const temperature = unit === '°F' || unit === 'F' ? degrees * 9 / 5 + 32 : degrees;
      if (typeof a.min_temp !== 'number' || typeof a.max_temp !== 'number' || temperature < a.min_temp || temperature > a.max_temp) throw new IntegrationError('invalid_request', 'Requested temperature is outside the reported device range.');
      await service('climate', 'set_temperature', [id], { temperature });
    },
    async weather(id: string) {
      const value = await observe(id, 'weather'), a = value.attributes;
      return { entity_id: id, state: value.state.slice(0, 100),
        temperature: typeof a.temperature === 'number' && Number.isFinite(a.temperature) ? a.temperature : null,
        unit: typeof a.temperature_unit === 'string' ? a.temperature_unit.slice(0, 20) : null,
        humidity: typeof a.humidity === 'number' && a.humidity >= 0 && a.humidity <= 100 ? a.humidity : null,
        observed_at: value.observed_at };
    },
    async script(id: string) { await observe(id, 'script'); await service('script', 'turn_on', [id]); },
    async media(id: string, command: CanonicalCommand, content?: string) {
      const { attributes: a } = await observe(id, 'media_player');
      let name: string, feature: number, data: Record<string, unknown> = {};
      if (command.intent === 'PLAY_MUSIC') {
        name = content ? 'play_media' : 'media_play'; feature = content ? 512 : 16384;
        if (content) data = { media_content_id: content, media_content_type: 'music' };
      } else if (command.intent === 'PAUSE') { name = 'media_pause'; feature = 1; }
      else if (command.intent === 'STOP') { name = 'media_stop'; feature = 4096; }
      else if (command.intent === 'NEXT') { name = 'media_next_track'; feature = 32; }
      else if (command.intent === 'VOLUME' && command.slots.operation === 'set') { name = 'volume_set'; feature = 4; data.volume_level = Number(command.slots.percent) / 100; }
      else if (command.intent === 'VOLUME' && command.slots.operation === 'mute') { name = 'volume_mute'; feature = 8; data.is_volume_muted = true; }
      else if (command.intent === 'VOLUME') { name = command.slots.operation === 'up' ? 'volume_up' : 'volume_down'; feature = 1024; }
      else throw new IntegrationError('invalid_request', 'Invalid media command.');
      if (!supports(a, feature)) throw unsupported();
      await service('media_player', name, [id], data);
    },
    async control(command: LightCommand) {
      await controlLights([command]);
    },
    async setLight(id: string, on: boolean) {
      const cfg = enabledConfig(); checkEntity(id, 'light');
      const response = await request(fetcher, `${cfg.url}/api/services/light/${on ? 'turn_on' : 'turn_off'}`, {
        method: 'POST', headers: { Authorization: `Bearer ${cfg.token}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ entity_id: id }),
      }, true);
      await requireSuccess(response);
      await response.body?.cancel();
    },
    async camera(id: string) {
      const cfg = enabledConfig(); checkEntity(id, 'camera');
      const response = await request(fetcher, `${cfg.url}/api/camera_proxy/${encodeURIComponent(id)}`, { headers: { Authorization: `Bearer ${cfg.token}` } });
      await requireSuccess(response);
      const contentType = response.headers.get('content-type')?.split(';')[0];
      if (!contentType || !['image/jpeg', 'image/png', 'image/webp'].includes(contentType) || !response.body) { await response.body?.cancel(); throw new IntegrationError('invalid_response', 'Camera did not return a supported image.'); }
      return { body: response.body, contentType };
    },
    async cameraStream(id: string, signal: AbortSignal) {
      const cfg = enabledConfig(); checkEntity(id, 'camera');
      // Bound connection setup, not the lifetime of an open video stream.
      const setup = new AbortController();
      const deadline = setTimeout(() => setup.abort(), 15_000);
      deadline.unref();
      let response: Response;
      try {
        response = await fetcher(`${cfg.url}/api/camera_proxy_stream/${encodeURIComponent(id)}`, {
          headers: { Authorization: `Bearer ${cfg.token}` }, redirect: 'error', signal: AbortSignal.any([signal, setup.signal]),
        });
      } catch { throw new IntegrationError('unavailable', 'Camera stream could not be opened. Check the camera in Home Assistant.'); }
      finally { clearTimeout(deadline); }
      await requireSuccess(response);
      const contentType = response.headers.get('content-type');
      if (!contentType || !/^multipart\/x-mixed-replace\s*;/i.test(contentType) || !/;\s*boundary=(?:"[^"]+"|[^;\s]+)/i.test(contentType) || !response.body) {
        await response.body?.cancel();
        throw new IntegrationError('invalid_response', 'Camera did not return an MJPEG stream.');
      }
      return { body: response.body, contentType };
    },
  };
}
