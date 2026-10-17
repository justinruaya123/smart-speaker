import WebSocket from 'ws';
import type { Config } from '../config.js';
import { isRecord } from '../../shared/protocol.js';
import { IntegrationError, request, requireSuccess } from './http.js';
import { lightAttributes, type EntityAttributes } from './state.js';
import type { LightCommand } from './commands.js';

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
  function checkEntity(id: string, domain: 'light' | 'camera' | 'climate') {
    if (!allowed.has(id) || !id.startsWith(`${domain}.`)) throw new IntegrationError('invalid_target', 'Entity is not allowlisted for this operation.');
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
    async notify(entityId:string,message:string,title:string){
      const cfg=enabledConfig();
      if(!/^notify\.[a-z0-9_]+$/.test(entityId)||!message||message.length>500||title.length>100)throw new IntegrationError('invalid_request','Invalid notification target or content.');
      const response=await request(fetcher,`${cfg.url}/api/services/notify/send_message`,{method:'POST',headers:{Authorization:`Bearer ${cfg.token}`,'Content-Type':'application/json'},body:JSON.stringify({entity_id:entityId,message,title})},true);
      await requireSuccess(response);await response.body?.cancel();
    },
    async setTemperature(entityId:string,degrees:number){
      const cfg=enabledConfig();checkEntity(entityId,'climate');
      if(!Number.isInteger(degrees)||degrees < -50||degrees>60)throw new IntegrationError('invalid_request','Invalid thermostat temperature.');
      // Confirm the entity's native unit and range before sending a setting.
      const state=await request(fetcher,`${cfg.url}/api/states/${encodeURIComponent(entityId)}`,{headers:{Authorization:`Bearer ${cfg.token}`}});
      await requireSuccess(state);const value:any=await state.json();
      if(value?.attributes?.temperature_unit!=='°C')throw new IntegrationError('invalid_request','Configure a Celsius thermostat for this finite command model.');
      if((Number.isFinite(value.attributes.min_temp)&&degrees<value.attributes.min_temp)||(Number.isFinite(value.attributes.max_temp)&&degrees>value.attributes.max_temp))
        throw new IntegrationError('invalid_request','Requested temperature is outside the thermostat range.');
      const response=await request(fetcher,`${cfg.url}/api/services/climate/set_temperature`,{method:'POST',headers:{Authorization:`Bearer ${cfg.token}`,'Content-Type':'application/json'},body:JSON.stringify({entity_id:entityId,temperature:degrees})},true);
      await requireSuccess(response);await response.body?.cancel();
    },
    async control(command: LightCommand) {
      const cfg = enabledConfig(); checkEntity(command.entityId, 'light');
      const body: Record<string, unknown> = { entity_id: command.entityId };
      let service = 'turn_on';
      switch (command.action) {
        case 'off': service = 'turn_off'; break;
        case 'on': break;
        case 'brightness': if (command.value === 0) service = 'turn_off'; else body.brightness_pct = command.value; break;
        case 'color': body.rgb_color = command.rgb; break;
        case 'temperature': body.color_temp_kelvin = command.value; break;
      }
      const response = await request(fetcher, `${cfg.url}/api/services/light/${service}`, {
        method: 'POST', headers: { Authorization: `Bearer ${cfg.token}`, 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      }, true);
      await requireSuccess(response);
      await response.body?.cancel();
      // Re-read observed state; do not turn an accepted request into an invented light state.
      if (socket && authenticated) await hydrate(socket, [command.entityId]);
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
