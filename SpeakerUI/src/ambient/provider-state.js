import { LIGHT_COLORS, temperatureCelsius } from './device-contracts.js';

const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const messages = {
    connected: 'Live', connecting: 'Connecting to Home Assistant', subscribing: 'Connecting to Home Assistant',
    disconnected: 'Home Assistant disconnected', degraded: 'Some entities are unavailable',
    authentication_failed: 'Home Assistant authentication failed', no_entities_selected: 'No entities configured',
    not_authorized: 'Connect Spotify to see playback', authentication: 'Reconnect Spotify', idle: 'Nothing playing on Spotify',
    rate_limited: 'Spotify is rate-limited', quota_exceeded: 'Spotify quota is exhausted', no_device: 'No active Spotify device',
};

export function homeAssistantView(snapshot, selectedId, sensorId) {
    if (!record(snapshot) || typeof snapshot.enabled !== 'boolean' || typeof snapshot.status !== 'string' || !Array.isArray(snapshot.entities)) throw new Error('Invalid Home Assistant state');
    const lights = [], sensors = [], cameras = [];
    for (const entity of snapshot.entities) {
        if (!record(entity) || typeof entity.id !== 'string' || typeof entity.name !== 'string' || typeof entity.state !== 'string' || typeof entity.stale !== 'boolean' || !record(entity.attributes)) throw new Error('Invalid entity state');
        const a = entity.attributes, available = !entity.stale && snapshot.status === 'connected' && !['unavailable', 'unknown'].includes(entity.state);
        if (entity.id.startsWith('camera.')) cameras.push({ id: entity.id, name: entity.name, available });
        if (entity.id.startsWith('sensor.') && a.device_class === 'temperature') {
            sensors.push({ entity_id: entity.id, state: available ? entity.state : 'unavailable', attributes: { ...a, friendly_name: entity.name } });
        }
        if (!entity.id.startsWith('light.')) continue;
        const rgb = Array.isArray(a.rgb_color) && a.rgb_color.length === 3 && a.rgb_color.every(v => Number.isFinite(v) && v >= 0 && v <= 255) ? a.rgb_color : null;
        const kelvin = Number.isFinite(a.color_temp_kelvin) ? a.color_temp_kelvin : null;
        const color = a.color_mode === 'color_temp' && kelvin !== null ?
            Object.keys(LIGHT_COLORS).find(name => LIGHT_COLORS[name].color_temp_kelvin === kelvin) || `${kelvin} K` :
            rgb ? Object.keys(LIGHT_COLORS).find(name => JSON.stringify(LIGHT_COLORS[name].rgb_color) === JSON.stringify(rgb)) || 'Custom color' :
            kelvin !== null ? `${kelvin} K` : 'Color unavailable';
        const brightness = Number.isFinite(a.brightness) && a.brightness >= 0 && a.brightness <= 255 ? Math.round(a.brightness / 255 * 100) : null;
        const on = available && entity.state === 'on';
        lights.push({ id: entity.id, name: entity.name, room: entity.id, on, brightness, color,
            colorCss: a.color_mode === 'color_temp' ? (LIGHT_COLORS[color]?.hex || '#eff7ef') : rgb ? `rgb(${rgb.join(',')})` : (LIGHT_COLORS[color]?.hex || '#eff7ef'),
            status: !available ? 'Unavailable' : entity.state === 'off' ? 'Off' : on ? brightness === null ? 'On' : `${brightness}%` : entity.state,
            supported_color_modes: Array.isArray(a.supported_color_modes) ? a.supported_color_modes : [],
            min_color_temp_kelvin: a.min_color_temp_kelvin, max_color_temp_kelvin: a.max_color_temp_kelvin,
        });
    }
    const sensor = sensors.find(s => s.entity_id === sensorId) || sensors[0];
    return { mode: snapshot.enabled ? 'live' : 'demo', status: snapshot.status,
        label: messages[snapshot.status] || 'Home Assistant unavailable', lights, sensors, cameras,
        selectedLight: Math.max(0, lights.findIndex(l => l.id === selectedId)),
        temperatureSensor: sensor?.entity_id || '', current: temperatureCelsius(sensor) };
}

export function spotifyView(snapshot) {
    if (!record(snapshot) || typeof snapshot.enabled !== 'boolean' || typeof snapshot.status !== 'string') throw new Error('Invalid Spotify state');
    const p = snapshot.playback;
    if (p !== null && (!record(p) || typeof p.playing !== 'boolean' || typeof p.shuffle !== 'boolean' || !['off', 'track', 'context'].includes(p.repeat))) throw new Error('Invalid playback state');
    if (p?.track && (!record(p.track) || typeof p.track.title !== 'string' || typeof p.track.artist !== 'string' || !Number.isFinite(p.track.durationMs) || p.track.durationMs < 0)) throw new Error('Invalid track state');
    const live = snapshot.enabled && snapshot.status === 'connected' ? p : null;
    const track = live?.track;
    const label = messages[snapshot.status] || 'Spotify unavailable';
    return { mode: snapshot.enabled ? 'live' : 'demo', status: snapshot.status, label,
        track: track ? [track.title, track.artist, track.durationMs / 1000] : ['Spotify', label, 0],
        album: typeof track?.album === 'string' && track.album.trim() ? track.album : null,
        artworkUrl: typeof track?.artworkUrl === 'string' && track.artworkUrl ? track.artworkUrl : null,
        mediaLoaded: Boolean(track), playing: Boolean(track && live.playing), position: Number.isFinite(live?.positionMs) ? live.positionMs / 1000 : 0,
        progressKnown: Number.isFinite(live?.positionMs), shuffle: live?.shuffle || false, repeat: live?.repeat || 'off',
        volume: Number.isFinite(live?.device?.volume) ? live.device.volume : null, deviceName: live?.device?.name || '',
    };
}

export function playbackPosition(playback, now) {
    return Math.min(playback.duration, playback.position + (playback.playing ? Math.max(0, now - playback.receivedAt) / 1000 : 0));
}
