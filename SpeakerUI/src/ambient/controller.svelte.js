import { flushSync } from 'svelte';
import { installFullscreen } from './fullscreen.js';
import { connectDummyBackend } from './backend.js';
import { connectProviderStates } from './provider-sync.js';
import { homeAssistantView, spotifyView, playbackPosition } from './provider-state.js';
import { createAuraRenderer, AURA_STATES } from './aura-renderer.js';
import { LIGHT_COLORS, supportsColor, colorPayload, serviceRequest, TEMPERATURE_SENSORS, temperatureCelsius } from './device-contracts.js';
export function createAmbient() {
    let root, aura, backend, providers, cameraBackground, disposed = false;
    const handlers = [], timeouts = new Set();
    let commandLifetime = new AbortController();
    function on(event, handler, options) { handlers.push({ event, handler, options }); }
    function later(fn, ms) { const id = setTimeout(() => { timeouts.delete(id); if (!disposed)
        fn(); }, ms); timeouts.add(id); return id; }
    function cancelLater(id) { clearTimeout(id); timeouts.delete(id); }
    const q = s => root.querySelector(s);
    const colors = Object.fromEntries(Object.entries(LIGHT_COLORS).map(([name, c]) => [name, c.hex]));
    const localDateTime = ms => { const d = new Date(ms); return new Date(ms - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16); };
    const clockTime = ms => new Date(ms).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    const duration = seconds => { const t = Math.max(0, Math.ceil(seconds)), h = Math.floor(t / 3600); return (h ? h + ':' : '') + String(Math.floor(t / 60) % 60).padStart(2, '0') + ':' + String(t % 60).padStart(2, '0'); };
    function nextAlarm(time, now) { const [h, m] = time.split(':').map(Number); const d = new Date(now); d.setHours(h, m, 0, 0); if (d.getTime() <= now)
        d.setDate(d.getDate() + 1); return d.getTime(); }
    function until(ms) { const m = Math.max(0, Math.ceil((ms - Date.now()) / 60000)); return Math.floor(m / 60) + 'h ' + m % 60 + 'm'; }
    function setText(selector, text) { if (selector === '[data-announcement]')
        model.announcement = text; if (selector === '[data-auto-caption]')
        model.autoCaption = text; }
    const model = $state({ now: Date.now(), closeProgress: null, auraError: null, announcement: "", autoCaption: "", draggingLight: null, backend: "connecting", runtimeMode: false, runtimeEngine: "", runtimeConfig: null, runtimeUI: null, runtimeActions: false, runtimeReady: false, runtimeState: 'starting', runtimePolicy: false, runtimeResult: null, runtimeError: '', runtimeSession: '', runtimeSequence: -1, view: 'home', agent: 'idle', muted: false, remindersExpanded: false, remindersEditUntil: 0, alarmPickerOpen: false, reminderMenuOpen: false, reminderAccordionOpen: true, soundMuted: false, pendingReminder: null, history: [], historyIndex: 0, lastInteraction: Date.now(), event: null, receiptAt: 0, selectedLight: 0,
        lights: [{ id: 'light.living_room', name: 'Living room', room: 'Ceiling', on: true, brightness: 65, color: 'Warm white' }, { id: 'light.reading_lamp', name: 'Reading lamp', room: 'Living room', on: true, brightness: 35, color: 'Warm white' }, { id: 'light.kitchen', name: 'Kitchen', room: 'Pendant', on: false, brightness: 80, color: 'Cool white' }, { id: 'light.bedroom', name: 'Bedroom', room: 'Bedside', on: false, brightness: 25, color: 'Red' }],
        haMode: 'loading', haStatus: 'loading', haActions: false, haPending: false, haLabel: 'Connecting to Home Assistant', spotifyMode: 'loading', spotifyActions: false, spotifyPending: false, spotifyLabel: 'Connecting to Spotify', spotifyStatus: 'loading', spotifyDevice: '', spotifyLastVolume: 40, spotifyProgress: null,
        tracks: [['Spotify', 'Connecting to Spotify', 0]], spotifyAlbum: null, spotifyArtworkUrl: null, spotifySeeking: false, spotifyActionLabel: '', sensors: [], cameras: [],
        target: 24, current: 26, temperatureSensor: 'sensor.living_room_temperature', timer: null, alarm: null, timerDraft: 300, alarmDraft: '07:00', reminderDraft: '', reminderDeadline: '', editingReminder: null,
        reminders: [{ id: 1, text: 'Water the plants', deadline: null, done: false, unread: true, nearShown: false, dueShown: false }, { id: 2, text: 'Pick up oat milk', deadline: null, done: false, unread: false, nearShown: false, dueShown: false }],
        auto: null, toast: null, alert: null, alertQueue: [], call: null, cameraEnabled: true, playing: false, mediaLoaded: false, position: 0, shuffle: false, repeat: 'off', song: 0, volume: 40, request: 0, returnFocus: null });
    const tracks = [['Weightless', 'Marconi Union', 509], ['A Walk', 'Tycho', 316], ['First Breath', 'Helios', 432]];
    for (const l of model.lights) {
        l.supported_color_modes = l.id === 'light.kitchen' ? ['color_temp'] : ['rgb', 'color_temp'];
        l.min_color_temp_kelvin = 2200;
        l.max_color_temp_kelvin = 6500;
        l.lastBrightness = l.brightness;
    }
    const demoLights = $state.snapshot(model.lights);
    model.lights = [];
    model.current = null;
    model.target = null;
    model.temperatureSensor = '';
    let lastHaState = null, lastSpotifyState = null, lightRangeEditing = false, mediaEditing = false, pendingSeek = null;
    function applyHomeAssistant(snapshot) {
        const view = homeAssistantView(snapshot, model.lights[model.selectedLight]?.id, model.temperatureSensor);
        lastHaState = snapshot;
        if (lightDrag || lightRangeEditing || model.haPending) return;
        const previousMode = model.haMode;
        model.haMode = view.mode;
        model.haStatus = view.status;
        model.haActions = snapshot.actionsEnabled === true;
        model.haLabel = view.label + (view.mode === 'live' && !model.haActions ? ' · Actions disabled' : '');
        model.cameras = view.cameras;
        if (view.mode === 'demo') {
            if (previousMode !== 'demo') {
                model.lights = structuredClone(demoLights);
                model.selectedLight = 0;
                model.sensors = TEMPERATURE_SENSORS;
                model.temperatureSensor = TEMPERATURE_SENSORS[0].entity_id;
                model.current = temperatureCelsius(TEMPERATURE_SENSORS[0]);
                model.target = 24;
            }
            return;
        }
        model.lights = view.lights;
        model.selectedLight = view.selectedLight;
        model.sensors = view.sensors;
        model.temperatureSensor = view.temperatureSensor;
        model.current = view.current;
        model.target = null;
    }
    function applySpotify(snapshot) {
        const view = spotifyView(snapshot), previousMode = model.spotifyMode;
        lastSpotifyState = snapshot;
        if (mediaEditing || model.spotifyPending) return;
        model.spotifyMode = view.mode;
        model.spotifyActions = snapshot.actionsEnabled === true;
        model.spotifyLabel = view.label + (view.mode === 'live' && !model.spotifyActions ? ' · Actions disabled' : '');
        model.spotifyStatus = view.status;
        model.spotifyAlbum = view.album;
        model.spotifyArtworkUrl = view.artworkUrl;
        if (pendingSeek && (view.status !== 'connected' || Date.now() >= pendingSeek.until || Math.abs(view.position - pendingSeek.position) <= 2)) pendingSeek = null;
        model.spotifySeeking = pendingSeek !== null;
        if (view.mode === 'demo') {
            if (previousMode !== 'demo') {
                model.tracks = tracks;
                model.song = 0;
                model.position = 0;
                model.playing = false;
                model.mediaLoaded = false;
                model.volume = 40;
            }
            return;
        }
        model.tracks = [view.track];
        model.song = 0;
        model.playing = view.playing;
        model.mediaLoaded = view.mediaLoaded;
        model.position = pendingSeek ? pendingSeek.position : view.position;
        model.shuffle = view.shuffle;
        model.repeat = view.repeat;
        model.volume = view.volume;
        if (view.volume > 0) model.spotifyLastVolume = view.volume;
        model.spotifyDevice = view.deviceName;
        model.spotifyProgress = { position: model.position, duration: view.track[2], playing: view.playing && view.progressKnown && !pendingSeek, receivedAt: Date.now() };
    }
    function providerFailed(provider) {
        if (provider === 'ha' && model.haMode !== 'demo') applyHomeAssistant({ enabled: true, status: 'disconnected', entities: lastHaState?.entities || [] });
        if (provider === 'spotify' && model.spotifyMode !== 'demo') applySpotify({ enabled: true, status: 'disconnected', playback: null });
    }
    function canControlLights() { return model.haMode === 'demo' || (model.haMode === 'live' && model.haActions && model.haStatus === 'connected' && !model.haPending); }
    function canControlSpotify() { return model.spotifyMode === 'demo' || (model.spotifyMode === 'live' && model.spotifyActions && ['connected', 'idle'].includes(model.spotifyStatus) && !model.spotifyPending); }
    function canDim(light) { return Boolean(light && (model.haMode === 'demo' || light.brightness !== null || light.supported_color_modes.some(mode => mode !== 'onoff'))); }
    async function providerCommand(path, command) {
        const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(command), signal: AbortSignal.any([commandLifetime.signal, AbortSignal.timeout(15000)]) });
        const body = await response.json();
        if (!response.ok) throw new Error(body.message || 'The provider did not accept this request.');
        return body;
    }
    function actionReceipt(intent, parameters, path) {
        receipt(intent, parameters);
        model.event.simulated = false;
        model.event.source = 'touch';
        model.event.status = 'accepted';
        model.event.request = { method: 'POST', path, body: parameters };
    }
    async function sendLight(commands) {
        if (!canControlLights() || !commands.length) return;
        model.haPending = true;
        const release = providers.hold('ha');
        try {
            for (const command of commands) {
                const snapshot = await providerCommand('/api/home-assistant/control', command);
                if (disposed) return;
                lastHaState = snapshot;
                actionReceipt(`light.${command.action}`, command, '/api/home-assistant/control');
            }
        } catch (error) {
            if (!disposed) toast('circle-help', 'Light request failed', error.message, 'lights');
        } finally {
            model.haPending = false;
            lightRangeEditing = false;
            if (!disposed && lastHaState) applyHomeAssistant(lastHaState);
            release();
        }
    }
    async function sendSpotify(command) {
        if (!canControlSpotify()) return;
        model.spotifyPending = true;
        model.spotifyActionLabel = command.action === 'seek' ? 'Seeking…' : ['next', 'previous'].includes(command.action) ? 'Switching track…' : 'Updating playback…';
        if (['next', 'previous'].includes(command.action)) { pendingSeek = null; model.spotifySeeking = false; }
        if (command.action === 'seek') {
            pendingSeek = { position: command.value / 1000, until: Date.now() + 4000 };
            model.spotifySeeking = true;
            model.position = pendingSeek.position;
        }
        const release = providers.hold('spotify');
        try {
            const snapshot = await providerCommand('/api/spotify/control', command);
            if (disposed) return;
            lastSpotifyState = snapshot;
            if (pendingSeek) pendingSeek.until = Date.now() + 4000;
            actionReceipt(`media.${command.action}`, command, '/api/spotify/control');
        } catch (error) {
            pendingSeek = null;
            model.spotifySeeking = false;
            if (!disposed) toast('circle-help', 'Spotify request failed', error.message, 'music');
        } finally {
            model.spotifyPending = false;
            model.spotifyActionLabel = '';
            mediaEditing = false;
            if (!disposed && lastSpotifyState) applySpotify(lastSpotifyState);
            release();
        }
    }
    async function connectSpotify() {
        try {
            const response = await fetch('/auth/spotify', { method: 'POST', signal: AbortSignal.any([commandLifetime.signal, AbortSignal.timeout(8000)]) });
            if (!response.ok) throw new Error('Authorization unavailable');
            const value = await response.json(), url = new URL(value.url);
            if (url.origin !== 'https://accounts.spotify.com') throw new Error('Invalid authorization URL');
            if (!disposed) window.location.assign(url.href);
        } catch {
            if (!disposed) model.spotifyLabel = 'Could not connect Spotify. Try again.';
        }
    }
    let stateReturn = null, speechInterval = null, dialPointer = null;
    let reminderOrderDrag = null;
    for (const r of model.reminders) {
        r.pinned = false;
        r.visibleUntil = 0;
    }
    function setAgent(state, { settle = false } = {}) {
        if (!(state in AURA_STATES))
            throw new Error('Unknown state');
        model.agent = state;
        aura?.update({ state, muted: model.muted, volume: 0 });
        clearInterval(speechInterval);
        speechInterval = null;
        if (state === 'speaking' && !model.muted) {
            speechInterval = setInterval(() => { if (document.hidden)
                return; const t = performance.now() / 1000; const syllable = Math.pow(Math.max(0, Math.sin(t * 5.4) + .43 * Math.sin(t * 13.1)), 1.1); aura.update({ volume: Math.min(.95, .14 + syllable * .58) }); }, 80);
        }
        cancelLater(stateReturn);
        if (settle)
            stateReturn = later(() => setAgent('idle'), 1900);
    }
    function receipt(intent, parameters = {}, source = 'touch') { const timestamp = Date.now(), request = serviceRequest(intent, parameters); model.event = { type: source === 'onnx-simulated' ? 'inference' : 'interaction', simulated: true, source, timestamp: new Date(timestamp).toISOString(), result: { utteranceId: crypto.randomUUID(), revision: 0, final: true, intent, parameters, ...(source === 'onnx-simulated' ? { confidence: 1, latencyMs: 380 } : {}) }, ...(request ? { request } : {}) }; model.history.unshift(model.event); model.history = model.history.slice(0, 30); model.historyIndex = 0; model.receiptAt = timestamp; }
    function autoStart(action) { model.auto = { action, remaining: 5000, last: performance.now(), paused: false }; updateAuto(); }
    function stopAuto() { if (model.auto) {
        model.auto = null;
        model.autoCaption = 'Save when ready';
    } }
    function updateAuto() { if (model.auto)
        model.autoCaption = (model.auto.action === 'reminder' ? '“Reminder”' : model.auto.action === 'timer' ? duration(model.timerDraft) : clockTime(nextAlarm(model.alarmDraft, Date.now()))) + ' in ' + Math.ceil(model.auto.remaining / 1000) + 's'; }
    function toast(symbol, title, detail, target, undo = null) { model.toast = { symbol, title, detail, target, undo, until: Date.now() + 4200, remaining: 4200, held: false }; model.announcement = title + '. ' + detail; }
    function closeToast() { model.toast = null; }
    function refreshReminderSurfaces() { renderReminderCount(); }
    function completeReminder(index) { const r = model.reminders[index]; r.done = true; r.unread = false; model.pendingReminder = null; receipt('reminder.complete', { id: r.id, done: true }); refreshReminderSurfaces(); }
    function renderReminderCount() { if (model.remindersExpanded && Date.now() >= model.remindersEditUntil && !reminderOrderDrag && !reminderDrag) {
        model.remindersExpanded = false;
        model.pendingReminder = null;
    } }
    function revealReminders() { model.remindersExpanded = !model.remindersExpanded; model.remindersEditUntil = model.remindersExpanded ? Date.now() + 30000 : 0; model.pendingReminder = null; renderReminderCount(); }
    function restoreHome() { model.alarmPickerOpen = false; model.view = 'home'; model.closeProgress = null; stopAuto(); flushSync(); if (model.returnFocus?.isConnected)
        model.returnFocus.focus({ preventScroll: true }); model.lastInteraction = Date.now(); }
    function open(view, { automatic = false } = {}) { model.alarmPickerOpen = view === 'alarm'; if (model.alarmPickerOpen) {
        if (model.alarm && /^\d\d:\d\d$/.test(model.alarm.time))
            model.alarmDraft = model.alarm.time;
        view = 'control';
    } if (model.view === 'home')
        model.returnFocus = document.activeElement; stopAuto(); if (view !== 'control')
        closeToast(); model.lastInteraction = Date.now(); model.view = view; if (view === 'timer' && !model.timer)
        model.timerDraft = 300; if (view === 'new-reminder' && model.editingReminder === null) {
        model.reminderDraft = '';
        model.reminderDeadline = '';
    } renderView(); q('.close-task')?.focus({ preventScroll: true }); if (automatic)
        autoStart(view === 'new-reminder' ? 'reminder' : view); }
    function renderView() { flushSync(); }
    function selectTemperatureSensor(entityId) { const sensor = model.sensors.find(s => s.entity_id === entityId); if (!sensor)
        return; model.temperatureSensor = entityId; model.current = temperatureCelsius(sensor); receipt('display.temperature_sensor', { entity_id: entityId }); model.lastInteraction = Date.now(); }
    function adjustTemperature(value) { if (model.haMode !== 'demo') return; model.target = Math.round(Math.max(16, Math.min(30, value))); receipt('climate.set_temperature', { entity_id: 'climate.living_room', temperature: model.target }); }
    function dialAt(event) { const r = q('.thermostat').getBoundingClientRect(); const dx = event.clientX - (r.left + r.width / 2), dy = event.clientY - (r.top + r.height / 2); if (Math.hypot(dx, dy) < r.width * .29)
        return; let deg = Math.atan2(dy, dx) * 180 / Math.PI; let progress = (deg + 225) % 360; if (progress < 0)
        progress += 360; if (progress > 270)
        progress = progress > 315 ? 0 : 270; adjustTemperature(16 + progress / 270 * 14); }
    function timerRemaining() { return !model.timer ? 0 : model.timer.end ? Math.max(0, (model.timer.end - Date.now()) / 1000) : model.timer.remaining; }
    function saveTimer() { stopAuto(); model.timer = { total: model.timerDraft, remaining: model.timerDraft, end: Date.now() + model.timerDraft * 1000 }; receipt('timer.set', { seconds: model.timerDraft }); restoreHome(); toast('timer', 'Timer started', duration(model.timerDraft), 'timer'); setAgent('speaking', { settle: true }); }
    function saveAlarm() { if (!/^\d{2}:\d{2}$/.test(model.alarmDraft))
        return; stopAuto(); model.alarm = { time: model.alarmDraft, deadline: nextAlarm(model.alarmDraft, Date.now()) }; receipt('alarm.set', { time: model.alarm.time }); restoreHome(); toast('alarm-clock', clockTime(model.alarm.deadline), 'In ' + until(model.alarm.deadline), 'alarm'); setAgent('speaking', { settle: true }); }
    function saveReminder() { const parsed = model.reminderDeadline ? new Date(model.reminderDeadline).getTime() : null; if (parsed !== null && !Number.isFinite(parsed))
        return; stopAuto(); const text = model.reminderDraft.trim() || 'Reminder'; const r = { id: Date.now(), text, deadline: parsed, done: false, unread: true, nearShown: false, dueShown: false, visibleUntil: Date.now() + 30000 }; if (model.editingReminder === null)
        model.reminders.push({ ...r, pinned: false });
    else {
        const previous = model.reminders[model.editingReminder];
        model.reminders[model.editingReminder] = { ...previous, ...r, id: previous.id };
    } model.remindersExpanded = true; model.remindersEditUntil = Date.now() + 30000; model.editingReminder = null; receipt('reminder.add', { text, deadline: parsed ? new Date(parsed).toISOString() : null }); renderReminderCount(); restoreHome(); toast('check', 'Reminder saved', text, 'reminders'); setAgent('speaking', { settle: true }); }
    function alertNotification(alert) { if (model.alert) {
        model.alertQueue.push(alert);
        return;
    } model.alert = { ...alert, closesAt: Date.now() + 300000 }; stopAuto(); closeToast(); model.announcement = alert.title; setAgent('idle'); flushSync(); q('[data-action=alert-dismiss]')?.focus({ preventScroll: true }); }
    function dismissAlert(snooze) { const a = model.alert; if (!a)
        return; if (snooze) {
        if (a.kind === 'reminder' && a.reminder) {
            model.alertQueue = model.alertQueue.filter(queued => queued.reminder?.id !== a.reminder.id);
                a.reminder.deadline = Date.now() + 300000;
            a.reminder.nearShown = false;
            a.reminder.dueShown = false;
        }
        else if (a.kind === 'timer')
            model.timer = { total: 300, remaining: 300, end: Date.now() + 300000 };
        else if (a.kind === 'alarm') {
            const deadline = Date.now() + 300000, d = new Date(deadline);
            model.alarm = { time: String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0'), deadline };
        }
    } if (a.reminder)
        a.reminder.unread = false; model.alert = null; model.lastInteraction = Date.now(); if (model.alertQueue.length)
        alertNotification(model.alertQueue.shift());
    else {
        flushSync();
        q(model.view === 'home' ? '.wake-target' : '.close-task')?.focus({ preventScroll: true });
    } }
    function startCall(name) { if (model.call) {
        open('call');
        return;
    } model.call = { kind: 'demo', name, started: Date.now(), muted: false, speaker: true }; model.cameraEnabled = true; receipt('call.start', { contact: name, simulated: true }); open('call'); }
    function startCamera(id) {
        const camera = model.cameras.find(camera => camera.id === id);
        if (!camera?.available) { open('contacts'); return; }
        model.call = { kind: 'camera', entityId: camera.id, name: camera.name, started: Date.now() };
        model.cameraEnabled = true;
        receipt('camera.view', { entity_id: camera.id });
        model.event.simulated = false;
        model.event.status = 'requested';
        model.event.request = { method: 'GET', path: `/api/home-assistant/camera/${encodeURIComponent(camera.id)}/stream` };
        open('call');
    }
    function liveShortcut(command) {
        return !model.runtimeMode && ((command.intent === 'media.play' && model.spotifyMode === 'live') ||
            (command.intent === 'call.start' && command.parameters.contact === 'Front door' && model.haMode === 'live'));
    }
    function drawCamera() { const canvas = q('.camera-canvas'); if (!canvas)
        return; const c = canvas.getContext('2d'), w = canvas.width, h = canvas.height, t = Date.now() / 1000; if (cameraBackground?.complete && cameraBackground.naturalWidth)
        c.drawImage(cameraBackground, 0, 0, w, h);
    else {
        c.fillStyle = '#18282e';
        c.fillRect(0, 0, w, h);
    } c.fillStyle = '#d9edeb14'; c.fillRect((t * 30) % (w + 50) - 50, 0, 35, h); c.fillStyle = '#c3d8db'; c.font = `${w / 28}px monospace`; c.fillText(model.cameraEnabled ? 'CAM 01  /  DEMO' : 'CAMERA OFF', w * .05, h * .10); c.font = `${w / 36}px monospace`; c.fillText(new Date().toLocaleTimeString(), w * .62, h * .94); }
    // The VCM supplies structured intent slots. No natural-language parsing or ASR exists here.
    const DEMO = [
        ['Lights · 40%', 'lightbulb', { intent: 'light.brightness', parameters: { entity_id: 'light.living_room', brightness_pct: 40 } }],
        ['Light tray', 'layout-grid', { intent: 'light.list', parameters: {} }], ['Temperature', 'thermometer', { intent: 'climate.set_temperature', parameters: { entity_id: 'climate.living_room', temperature: 24 } }],
        ['Timer · 5 min', 'timer', { intent: 'timer.set', parameters: { seconds: 300 } }], ['Alarm · 7 am', 'alarm-clock', { intent: 'alarm.set', parameters: { time: '07:00' } }],
        ['New reminder', 'plus', { intent: 'reminder.create', parameters: {} }], ['Play music', 'music-2', { intent: 'media.play', parameters: {} }],
        ['Weather', 'cloud-sun', { intent: 'answer.weather', parameters: { location: 'Manila' } }], ['Call Mom', 'phone', { intent: 'call.start', parameters: { contact: 'Mom' } }],
        ['Camera call', 'video', { intent: 'call.start', parameters: { contact: 'Front door' } }], ['Reminder due', 'bell', { intent: 'demo.reminder_due', parameters: {} }],
        ['Timer done', 'hourglass', { intent: 'demo.timer_due', parameters: {} }], ['Alarm rings', 'alarm-clock', { intent: 'demo.alarm_due', parameters: {} }],
        ['Near deadline', 'clock-3', { intent: 'demo.reminder_near', parameters: {} }], ['Unknown light', 'circle-help', { intent: 'light.brightness', parameters: { entity_id: 'light.unknown', brightness_pct: 50 } }]
    ];
    function dispatch(event) {
        if (!event || typeof event.intent !== 'string' || (event.parameters !== undefined && (!event.parameters || typeof event.parameters !== 'object' || Array.isArray(event.parameters)))) {
            setAgent('failed');
            toast('circle-help', 'Incomplete intent', 'Choose a demo intent', 'demo');
            return;
        }
        if ((model.haMode !== 'demo' && ((event.intent.startsWith('light.') && event.intent !== 'light.list') || event.intent.startsWith('climate.'))) || (model.spotifyMode !== 'demo' && event.intent.startsWith('media.'))) {
            receipt('needs_input', { reason: 'Simulator actions do not operate live providers.' }, 'onnx-simulated');
            toast('circle-help', 'Simulator only', 'Use the screen controls to change live devices', event.intent.startsWith('media.') ? 'music' : 'lights');
            return;
        }
        const request = ++model.request;
        restoreHome();
        setAgent('thinking');
        later(() => {
            if (request !== model.request)
                return;
            const raw = event.parameters || {};
            const p = { ...raw };
            if (p.entity_id !== undefined)
                p.entityId = p.entity_id;
            if (p.brightness_pct !== undefined)
                p.brightness = p.brightness_pct;
            if (p.temperature !== undefined)
                p.target = p.temperature;
            if (event.intent === 'light.brightness' && (!model.lights.some(l => l.id === p.entityId) || !Number.isFinite(p.brightness) || p.brightness < 0 || p.brightness > 100)) {
                receipt('needs_input', { reason: 'Choose a known light and 0–100% brightness' });
                setAgent('failed');
                toast('circle-help', 'Which light?', 'Choose from your lights', 'lights');
                return;
            }
            receipt(event.intent, raw, 'onnx-simulated');
            switch (event.intent) {
                case 'light.brightness': {
                    const l = model.lights.find(l => l.id === p.entityId);
                    l.brightness = p.brightness;
                    l.on = p.brightness > 0;
                    model.selectedLight = model.lights.indexOf(l);
                    toast('lightbulb', l.name, `${l.brightness}% · ${l.color}`, 'lights');
                    break;
                }
                case 'light.on':
                case 'light.off':
                case 'light.toggle': {
                    const l = model.lights.find(l => l.id === p.entityId);
                    if (!l) {
                        setAgent('failed');
                        toast('circle-help', 'Which light?', 'Choose from your lights', 'lights');
                        return;
                    }
                    l.on = event.intent === 'light.toggle' ? !l.on : event.intent === 'light.on';
                    if (l.on && !l.brightness)
                        l.brightness = l.lastBrightness || 50;
                    model.selectedLight = model.lights.indexOf(l);
                    toast('lightbulb', l.name, l.on ? l.brightness + '% · ' + l.color : 'Off', 'lights');
                    break;
                }
                case 'light.color': {
                    const l = model.lights.find(l => l.id === p.entityId);
                    const chosen = p.color ?? Object.keys(LIGHT_COLORS).find(name => p.rgb_color ? JSON.stringify(LIGHT_COLORS[name].rgb_color) === JSON.stringify(p.rgb_color) : LIGHT_COLORS[name].color_temp_kelvin === p.color_temp_kelvin);
                    if (!l || !supportsColor(l, chosen)) {
                        setAgent('failed');
                        toast('circle-help', 'Color unavailable', 'Choose a supported light color', 'lights');
                        return;
                    }
                    l.color = chosen;
                    l.on = true;
                    if (!l.brightness)
                        l.brightness = l.lastBrightness || 50;
                    model.selectedLight = model.lights.indexOf(l);
                    toast('lightbulb', l.name, l.brightness + '% · ' + l.color, 'lights');
                    break;
                }
                case 'light.list':
                    open('lights');
                    break;
                case 'climate.set_temperature':
                    if (!Number.isFinite(p.target) || p.target < 16 || p.target > 30) {
                        setAgent('failed');
                        toast('thermometer', '16–30°C', 'Choose a target', 'climate');
                        return;
                    }
                    model.target = p.target;
                    toast('thermometer', p.target + '° target', model.current === null ? 'Sensor unavailable' : 'Inside ' + model.current + '°', 'climate');
                    break;
                case 'timer.set':
                    if (!Number.isFinite(p.seconds) || p.seconds < 1 || p.seconds > 86400) {
                        setAgent('failed');
                        return;
                    }
                    model.timerDraft = p.seconds;
                    model.timer = { total: p.seconds, remaining: p.seconds, end: Date.now() + p.seconds * 1000 };
                    toast('hourglass', 'Timer started', duration(p.seconds), 'timer');
                    break;
                case 'alarm.set':
                    if (!/^([01]\d|2[0-3]):[0-5]\d$/.test(p.time)) {
                        setAgent('failed');
                        return;
                    }
                    model.alarmDraft = p.time;
                    model.alarm = { time: p.time, deadline: nextAlarm(p.time, Date.now()) };
                    toast('alarm-clock', clockTime(model.alarm.deadline), 'In ' + until(model.alarm.deadline), 'alarm');
                    break;
                case 'reminder.create':
                    model.editingReminder = null;
                    open('new-reminder', { automatic: true });
                    break;
                case 'media.play':
                    model.song = 0;
                    model.playing = true;
                    model.mediaLoaded = true;
                    model.position = 0;
                    toast('music-2', tracks[0][0], tracks[0][1], 'music');
                    break;
                case 'answer.weather':
                    open('weather');
                    model.weatherCloseAt = null;
                    break;
                case 'call.start':
                    if (!['Mom', 'Alex', 'Front door'].includes(p.contact)) {
                        setAgent('failed');
                        return;
                    }
                    startCall(p.contact);
                    break;
                case 'demo.reminder_due':
                case 'demo.reminder_near': {
                    const r = { id: Date.now(), text: 'Check the oven', deadline: Date.now() + (event.intent.endsWith('near') ? 45000 : 0), done: false, unread: true, nearShown: false, dueShown: false };
                    model.reminders.push(r);
                    renderReminderCount();
                    checkDeadlines();
                    return;
                }
                case 'demo.timer_due':
                    model.timer = { total: 1, remaining: 0, end: Date.now() };
                    checkDeadlines();
                    return;
                case 'demo.alarm_due':
                    model.alarm = { time: 'Demo', deadline: Date.now() };
                    checkDeadlines();
                    return;
                default:
                    setAgent('failed');
                    toast('circle-help', 'Not available', 'Try a demo intent', 'demo');
                    return;
            }
            setAgent('speaking', { settle: true });
        }, 380);
    }
    function checkDeadlines() { const now = Date.now(); if (model.timer?.end && model.timer.end <= now) {
        model.timer = null;
        alertNotification({ kind: 'timer', title: 'Timer finished', subtitle: 'Time’s up.' });
    } if (model.alarm && model.alarm.deadline <= now) {
        const a = model.alarm;
        model.alarm = null;
        alertNotification({ kind: 'alarm', title: clockTime(a.deadline), subtitle: 'Alarm' });
    } for (const r of model.reminders) {
        if (r.done || !r.deadline)
            continue;
        if (r.deadline <= now && !r.dueShown) {
            r.dueShown = true;
            r.nearShown = true;
            r.unread = true;
            alertNotification({ kind: 'reminder', title: r.text, subtitle: 'Due now', reminder: r });
        }
        else if (r.deadline > now && r.deadline - now <= 60000 && !r.nearShown) {
            r.nearShown = true;
            r.unread = true;
            alertNotification({ kind: 'reminder', title: r.text, subtitle: 'In less than a minute', reminder: r });
        }
    } renderReminderCount(); }
    // Reordering is deliberately owned only by the grip, never by a row or its text.
    function reorderReminder(from, to) { if (from === to)
        return; const [r] = model.reminders.splice(from, 1); model.reminders.splice(to, 0, r); receipt('reminder.reorder', { ids: model.reminders.filter(r => !r.done).map(r => r.id) }); refreshReminderSurfaces(); }
    on('pointerdown', event => { const grip = event.target.closest('[data-reorder-reminder]'); if (!grip)
        return; event.preventDefault(); grip.setPointerCapture(event.pointerId); const row = grip.closest('[data-reminder-swipe]'); reminderOrderDrag = { pointer: event.pointerId, index: Number(grip.dataset.reorderReminder), target: Number(grip.dataset.reorderReminder), grip, row, y: event.clientY, moved: false }; });
    on('pointermove', event => { const g = reminderOrderDrag; if (!g || g.pointer !== event.pointerId)
        return; event.preventDefault(); const dy = event.clientY - g.y; if (Math.abs(dy) > 6)
        g.moved = true; if (!g.moved)
        return; model.lastInteraction = Date.now(); const rows = [...g.row.parentElement.querySelectorAll(':scope > [data-reminder-swipe]')]; let nearest = g.row, distance = Infinity; for (const row of rows) {
        const r = row.getBoundingClientRect(), d = Math.abs(event.clientY - r.y - r.height / 2);
        if (d < distance) {
            distance = d;
            nearest = row;
        }
        row.classList.remove('reorder-target');
    } g.target = Number(nearest.dataset.reminderSwipe); if (nearest !== g.row)
        nearest.classList.add('reorder-target'); g.row.classList.add('reordering'); });
    function clearReorderGesture() { if (!reminderOrderDrag)
        return; reminderOrderDrag.row.classList.remove('reordering'); for (const row of root.querySelectorAll('.reorder-target'))
        row.classList.remove('reorder-target'); reminderOrderDrag = null; }
    on('pointerup', event => { const g = reminderOrderDrag; if (!g || g.pointer !== event.pointerId)
        return; clearReorderGesture(); if (g.moved) {
        suppressClickUntil = Date.now() + 450;
        reorderReminder(g.index, g.target);
    } });
    on('pointercancel', clearReorderGesture);
    on('keydown', event => { const grip = event.target.closest('[data-reorder-reminder]'); if (!grip || !['ArrowUp', 'ArrowDown'].includes(event.key))
        return; event.preventDefault(); const from = Number(grip.dataset.reorderReminder), id = model.reminders[from].id, active = model.reminders.map((r, i) => ({ r, i })).filter(({ r }) => !r.done), position = active.findIndex(item => item.i === from), next = active[position + (event.key === 'ArrowUp' ? -1 : 1)]; if (!next)
        return; const surface = grip.closest('.cc-reminder-menu') ? '.cc-reminder-menu' : grip.closest('.home-reminders') ? '.home-reminders' : '.reminders-layout'; reorderReminder(from, next.i); q(`${surface} [data-reminder-id="${id}"]`)?.focus({ preventScroll: true }); });
    // Reveal the action continuously; completion belongs to the swiped row on release.
    let reminderDrag = null;
    on('pointerdown', event => { const row = event.target.closest('[data-reminder-swipe]'); if (!row || event.target.closest('[data-pin-reminder],[data-reorder-reminder],[data-complete-reminder]'))
        return; const label = row.querySelector('.reminder-copy strong,.quiet-reminder strong'), bounds = row.getBoundingClientRect(); row.style.setProperty('--swipe-label-x', Math.max(8, Math.min(bounds.width - 128, (label?.getBoundingClientRect().left ?? bounds.left + 18) - bounds.left)) + 'px'); reminderDrag = { pointer: event.pointerId, index: Number(row.dataset.reminderSwipe), row, x: event.clientX, y: event.clientY, moved: false, ready: false, samples: [{ x: event.clientX, t: event.timeStamp }] }; });
    on('pointermove', event => { if (reminderDrag?.pointer !== event.pointerId)
        return; const g = reminderDrag, dx = event.clientX - g.x, dy = event.clientY - g.y; sampleReminderSwipe(g, event.clientX, event.timeStamp); if (!g.moved && Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 10) {
        reminderDrag = null;
        return;
    } if (!g.moved && Math.abs(dx) > 12) {
        g.moved = true;
        g.row.setPointerCapture(event.pointerId);
    } if (g.moved) {
        event.preventDefault();
        model.lastInteraction = Date.now();
        const distance = Math.max(0, Math.min(g.row.clientWidth, dx)), progress = distance / g.row.clientWidth;
        g.row.style.setProperty('--reminder-offset', distance + 'px');
        g.row.style.setProperty('--swipe-reveal', Math.max(0, Math.min(160, distance - 7)) + 'px');
        g.row.style.setProperty('--swipe-opacity', String(Math.min(1, distance / 54)));
        g.row.classList.add('dragging');
        const ready = progress >= .8;
        g.row.classList.toggle('swipe-ready', ready);
        if (ready && !g.ready)
            setText('[data-announcement]', 'Release to mark as done');
        g.ready = ready;
    } });
    function sampleReminderSwipe(g, x, t) { g.samples = g.samples.filter(sample => t - sample.t <= 100); g.samples.push({ x, t }); }
    function reminderSwipeCompletes(g, x, t) { sampleReminderSwipe(g, x, t); const first = g.samples[0], elapsed = t - first.t, velocity = elapsed >= 8 ? (x - first.x) / elapsed : 0, distance = x - g.x; return distance >= g.row.clientWidth * .8 || (distance >= 44 && velocity >= .65); }
    function resetReminderSwipe(g) { g.row.style.removeProperty('--reminder-offset'); g.row.style.removeProperty('--swipe-reveal'); g.row.style.removeProperty('--swipe-opacity'); g.row.classList.remove('dragging', 'swipe-ready'); }
    on('pointerup', event => { if (reminderDrag?.pointer !== event.pointerId)
        return; const g = reminderDrag; reminderDrag = null; if (!g.moved)
        return; suppressClickUntil = Date.now() + 450; if (reminderSwipeCompletes(g, event.clientX, event.timeStamp)) {
        completeReminder(g.index);
        setText('[data-announcement]', 'Reminder completed');
    }
    else
        resetReminderSwipe(g); });
    on('pointercancel', () => { if (reminderDrag)
        resetReminderSwipe(reminderDrag); reminderDrag = null; });
    // All gestures mutate the local model first. One service payload is recorded on release.
    let lightDrag = null, cornerSwipe = null, suppressClickUntil = 0, lastLightTap = null, lightSelectTimer = null, lastPlaybackAt = Date.now();
    function setLightBrightness(index, value) { if (!canControlLights() || !canDim(model.lights[index])) return; const l = model.lights[index]; l.brightness = Math.max(0, Math.min(100, Math.round(value))); l.on = l.brightness > 0; if (l.on)
        l.lastBrightness = l.brightness; }
    function recordLightBrightness(index) { const l = model.lights[index]; if (model.haMode === 'live') { void sendLight([{ entityId: l.id, action: 'brightness', value: l.brightness }]); return; } if (model.haMode !== 'demo') return; receipt('light.brightness', { entity_id: l.id, brightness_pct: l.brightness }); }
    function toggleLight(index) { const l = model.lights[index]; if (!l || !canControlLights() || l.status === 'Unavailable') return; if (model.haMode === 'live') { void sendLight([{ entityId: l.id, action: l.on ? 'off' : 'on' }]); return; } l.on = !l.on; if (l.on && !l.brightness)
        l.brightness = l.lastBrightness || 50; receipt(l.on ? 'light.on' : 'light.off', { entity_id: l.id }); if (model.view === 'control')
        renderView(); }
    function advancePlayback(now) { const dt = Math.min(1, (now - lastPlaybackAt) / 1000); lastPlaybackAt = now; if (model.spotifyMode !== 'demo') {
        if (pendingSeek && !model.spotifyPending && now >= pendingSeek.until) { pendingSeek = null; if (lastSpotifyState) applySpotify(lastSpotifyState); }
        if (model.spotifyProgress && !mediaEditing && !model.spotifyPending) model.position = playbackPosition(model.spotifyProgress, now);
        return;
    } if (model.playing) {
        model.position += dt;
        if (model.position >= tracks[model.song][2]) {
            if (model.repeat === 'track')
                model.position = 0;
            else if (model.repeat === 'context' || model.song < tracks.length - 1) {
                model.song = model.shuffle ? (model.song + 1 + Math.floor(Math.random() * 2)) % 3 : (model.song + 1) % 3;
                model.position = 0;
            }
            else {
                model.playing = false;
                model.position = tracks[model.song][2];
            }
            if (model.view === 'music' || model.view === 'control')
                renderView();
        }
    } }
    on('pointerdown', event => {
        const row = event.target.closest('[data-light-row]');
        if (row && canControlLights() && canDim(model.lights[Number(row.dataset.lightRow)]) && model.lights[Number(row.dataset.lightRow)].status !== 'Unavailable') {
            const index = Number(row.dataset.lightRow), l = model.lights[index];
            model.draggingLight = index;
            lightDrag = { pointer: event.pointerId, index, x: event.clientX, y: event.clientY, brightness: l.brightness, on: l.on, lastBrightness: l.lastBrightness, moved: false, row };
        }
        if (model.view === 'home' && event.target.closest('.corner-swipe-zone,.control-trigger') && !event.target.closest('.home-reminders'))
            cornerSwipe = { pointer: event.pointerId, x: event.clientX, y: event.clientY };
    });
    on('pointermove', event => {
        if (cornerSwipe?.pointer === event.pointerId && !reminderDrag && !reminderOrderDrag) {
            const dx = event.clientX - cornerSwipe.x, dy = event.clientY - cornerSwipe.y;
            if (dy > 45 && dy > Math.abs(dx) * 1.3) {
                cornerSwipe = null;
                suppressClickUntil = Date.now() + 450;
                open('control');
            }
        }
        if (lightDrag?.pointer !== event.pointerId)
            return;
        const g = lightDrag, dx = event.clientX - g.x, dy = event.clientY - g.y;
        if (!g.moved && Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 10) {
            lightDrag = null;
            model.draggingLight = null;
            return;
        }
        if (Math.abs(dx) > 10 && !g.moved) {
            g.moved = true;
            g.row.setPointerCapture(event.pointerId);
            cancelLater(lightSelectTimer);
            lastLightTap = null;
        }
        if (g.moved) {
            event.preventDefault();
            model.lastInteraction = Date.now();
            setLightBrightness(g.index, g.brightness + dx / g.row.clientWidth * 100);
        }
    });
    on('pointerup', event => { cornerSwipe = null; if (lightDrag?.pointer === event.pointerId) {
        if (lightDrag.moved) {
            recordLightBrightness(lightDrag.index);
            if (model.view === 'control' && !model.lights[lightDrag.index].on)
                renderView();
            suppressClickUntil = Date.now() + 450;
        }
        lightDrag = null;
        model.draggingLight = null;
    } });
    on('pointercancel', () => { cornerSwipe = null; if (lightDrag?.moved) {
        const l = model.lights[lightDrag.index];
        l.brightness = lightDrag.brightness;
        l.on = lightDrag.on;
        l.lastBrightness = lightDrag.lastBrightness;
    } lightDrag = null; model.draggingLight = null; });
    on('click', event => { if (model.view === 'control' && event.target === q('.task-view'))
        restoreHome(); });
    on('pointerdown', event => { model.lastInteraction = Date.now(); if (model.remindersExpanded && event.target.closest('.home-reminders'))
        model.remindersEditUntil = Date.now() + 30000; if (q('.task-content').contains(event.target))
        stopAuto(); const dial = event.target.closest('.thermostat'); if (dial && model.haMode === 'demo') {
        const r = dial.getBoundingClientRect();
        if (Math.hypot(event.clientX - r.left - r.width / 2, event.clientY - r.top - r.height / 2) < r.width * .29) {
            adjustTemperature(model.target + (event.clientX > r.left + r.width / 2 ? 1 : -1));
        }
        else {
            dialPointer = event.pointerId;
            dial.setPointerCapture(dialPointer);
            dialAt(event);
        }
    } });
    on('pointermove', event => { if (event.pointerId === dialPointer) {
        model.lastInteraction = Date.now();
        dialAt(event);
    } });
    on('pointerup', event => { if (event.pointerId === dialPointer) {
        dialPointer = null;
    } });
    on('pointercancel', () => { dialPointer = null; });
    on('focusin', event => { if (q('.task-content').contains(event.target))
        stopAuto(); });
    on('click', event => {
        const b = event.target.closest('button');
        if (!b || b.disabled)
            return;
        model.lastInteraction = Date.now();
        if (Date.now() < suppressClickUntil)
            return;
        if (b.dataset.open) {
            if (b.dataset.open === 'alarm' && model.alarm && /^\d\d:\d\d$/.test(model.alarm.time))
                model.alarmDraft = model.alarm.time;
            open(b.dataset.open);
            return;
        }
        if (b.dataset.demo !== undefined) {
            const index = Number(b.dataset.demo);
            const command = DEMO[index][2];
            if(model.runtimeMode){receipt(command.intent,command.parameters,'sample-fixture');model.event.actionsEnabled=false;model.event.status='preview';open('intent');return;}
            if (command.intent === 'media.play' && model.spotifyMode !== 'demo') {
                open('music');
                if (canControlSpotify()) void sendSpotify({ action: 'play' });
                else if (!['not_authorized', 'authentication'].includes(model.spotifyStatus)) toast('music-2', 'Spotify', model.spotifyLabel, 'music');
                return;
            }
            if (command.intent === 'call.start' && command.parameters.contact === 'Front door' && model.haMode !== 'demo') {
                if (model.cameras.length === 1) startCamera(model.cameras[0].id);
                else open('contacts');
                return;
            }
            void backend.infer(index, command);
            return;
        }
        if (b.dataset.agent) {
            restoreHome();
            setAgent(b.dataset.agent);
            return;
        }
        if (b.dataset.quickLight !== undefined) {
            model.selectedLight = Number(b.dataset.quickLight);
            open('lights');
            return;
        }
        if (b.dataset.lightToggle !== undefined) {
            toggleLight(Number(b.dataset.lightToggle));
            return;
        }
        if (b.dataset.lightSelect !== undefined) {
            const i = Number(b.dataset.lightSelect);
            if (event.detail === 0) {
                model.selectedLight = i;
                renderView();
                return;
            }
            if (lastLightTap?.index === i && Date.now() - lastLightTap.at < 340) {
                cancelLater(lightSelectTimer);
                lastLightTap = null;
                toggleLight(i);
            }
            else {
                lastLightTap = { index: i, at: Date.now() };
                cancelLater(lightSelectTimer);
                lightSelectTimer = later(() => { if (model.view === 'lights') {
                    model.selectedLight = i;
                    renderView();
                } lastLightTap = null; }, 340);
            }
            return;
        }
        if (b.dataset.completeReminder !== undefined) {
            completeReminder(Number(b.dataset.completeReminder));
            return;
        }
        if (b.dataset.pinReminder !== undefined) {
            const r = model.reminders[Number(b.dataset.pinReminder)];
            r.pinned = !r.pinned;
            if (b.closest('.home-reminders')) {
                model.remindersExpanded = true;
                model.remindersEditUntil = Date.now() + 30000;
            }
            if (!r.pinned)
                r.visibleUntil = Date.now() + 30000;
            receipt('reminder.pin', { id: r.id, pinned: r.pinned });
            renderReminderCount();
            if (model.view === 'control' || model.view === 'reminders')
                renderView();
            return;
        }
        if (b.dataset.history !== undefined) {
            model.historyIndex = Number(b.dataset.history);
            renderView();
            return;
        }
        if (b.dataset.color) {
            if (!canControlLights()) return;
            const l = model.lights[model.selectedLight];
            if (!supportsColor(l, b.dataset.color))
                return;
            if (model.haMode === 'live') { const color = colorPayload(l, b.dataset.color); void sendLight([{ entityId: l.id, ...(color.rgb_color ? { action: 'color', rgb: color.rgb_color } : { action: 'temperature', value: color.color_temp_kelvin }) }]); return; }
            l.color = b.dataset.color;
            l.on = true;
            if (!l.brightness)
                l.brightness = l.lastBrightness || 50;
            receipt('light.color', { entity_id: l.id, ...colorPayload(l, l.color) });
            renderView();
            return;
        }
        if (b.dataset.timerPreset) {
            model.timerDraft = Number(b.dataset.timerPreset);
            renderView();
            return;
        }
        if (b.dataset.check !== undefined) {
            const r = model.reminders[Number(b.dataset.check)];
            r.done = !r.done;
            r.unread = false;
            renderReminderCount();
            renderView();
            receipt('reminder.complete', { id: r.id, done: r.done });
            return;
        }
        if (b.dataset.reminder !== undefined) {
            const i = Number(b.dataset.reminder), r = model.reminders[i];
            r.unread = false;
            model.editingReminder = i;
            model.reminderDraft = r.text;
            model.reminderDeadline = r.deadline ? localDateTime(r.deadline) : '';
            renderReminderCount();
            open('new-reminder');
            return;
        }
        if (b.dataset.camera) { startCamera(b.dataset.camera); return; }
        if (b.dataset.contact) {
            startCall(b.dataset.contact);
            return;
        }
        switch (b.dataset.action) {
            case 'close':
                ++model.request;
                model.weatherCloseAt = null;
                restoreHome();
                break;
            case 'back':
                open(model.view === 'call' ? 'contacts' : 'reminders');
                break;
            case 'mute':
                if (model.runtimeMode) { void fetch('/api/runtime/mute', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}); break; }
                model.muted = !model.muted;
                setAgent('idle');
                if (model.view === 'control')
                    renderView();
                break;
            case 'wake':
                if (model.runtimeMode) { open('intent'); break; }
                if (model.muted) {
                    toast('mic-off', 'Microphone muted', 'Tap to unmute', 'control');
                    break;
                }
                setAgent(model.agent === 'listening' ? 'idle' : 'listening');
                break;
            case 'toggle-reminders':
                revealReminders();
                break;
            case 'reminder-accordion':
                model.reminderAccordionOpen = !model.reminderAccordionOpen;
                refreshReminderSurfaces();
                q('.cc-reminder-menu .reminder-accordion-toggle').focus({ preventScroll: true });
                break;
            case 'reminder-menu':
                model.reminderMenuOpen = !model.reminderMenuOpen;
                renderView();
                q('.cc-reminder-toggle').focus({ preventScroll: true });
                break;
            case 'sound-mute':
                if (model.spotifyMode === 'live') { if (model.volume !== null) void sendSpotify({ action: 'volume', value: model.volume === 0 ? model.spotifyLastVolume : 0 }); break; }
                if (model.spotifyMode !== 'demo') break;
                model.soundMuted = !model.soundMuted;
                receipt('audio.mute', { muted: model.soundMuted });
                break;
            case 'toast-close':
                closeToast();
                break;
            case 'toast-open':
                if (model.toast)
                    open(model.toast.target);
                break;
            case 'light-toggle':
                toggleLight(model.selectedLight);
                break;
            case 'all-off':
                if (!canControlLights()) break;
                if (model.haMode === 'live') { void sendLight(model.lights.filter(l => l.status !== 'Unavailable').map(l => ({ entityId: l.id, action: 'off' }))); break; }
                for (const l of model.lights)
                    l.on = false;
                receipt('light.group_off', { entity_ids: model.lights.map(l => l.id) });
                renderView();
                break;
            case 'temp-minus':
                adjustTemperature(model.target - 1);
                break;
            case 'temp-plus':
                adjustTemperature(model.target + 1);
                break;
            case 'timer-minus':
            case 'timer-plus': {
                const delta = b.dataset.action === 'timer-plus' ? 60 : -60;
                if (model.timer) {
                    const remaining = Math.max(1, timerRemaining() + delta);
                    model.timer.total = Math.max(model.timer.total, remaining);
                    model.timer.remaining = remaining;
                    if (model.timer.end)
                        model.timer.end = Date.now() + remaining * 1000;
                }
                else
                    model.timerDraft = Math.max(60, Math.min(86400, model.timerDraft + delta));
                renderView();
                break;
            }
            case 'timer-save':
                saveTimer();
                break;
            case 'timer-pause':
                if (model.timer.end) {
                    model.timer.remaining = timerRemaining();
                    model.timer.end = null;
                }
                else
                    model.timer.end = Date.now() + model.timer.remaining * 1000;
                receipt(model.timer.end ? 'timer.resume' : 'timer.pause', {});
                renderView();
                break;
            case 'timer-cancel':
                model.timer = null;
                receipt('timer.cancel', {});
                restoreHome();
                break;
            case 'alarm-picker-close':
                model.alarmPickerOpen = false;
                renderView();
                break;
            case 'alarm-save':
                saveAlarm();
                break;
            case 'alarm-remove':
                model.alarm = null;
                receipt('alarm.remove', {});
                restoreHome();
                break;
            case 'new-reminder':
                model.editingReminder = null;
                open('new-reminder');
                break;
            case 'reminder-save':
                saveReminder();
                break;
            case 'alert-dismiss':
                dismissAlert(false);
                break;
            case 'alert-snooze':
                dismissAlert(true);
                break;
            case 'resume-call':
                open('call');
                break;
            case 'call-mute':
                if (model.call?.kind !== 'demo') break;
                model.call.muted = !model.call.muted;
                renderView();
                break;
            case 'call-speaker':
                if (model.call?.kind !== 'demo') break;
                model.call.speaker = !model.call.speaker;
                renderView();
                break;
            case 'call-camera':
                model.cameraEnabled = !model.cameraEnabled;
                renderView();
                break;
            case 'camera-expand':
                open('call');
                break;
            case 'camera-collapse':
                restoreHome();
                break;
            case 'hangup':
                receipt(model.call.kind === 'camera' ? 'camera.close' : 'call.end', { contact: model.call.name });
                if (model.call.kind === 'camera') { model.event.simulated = false; model.event.status = 'closed'; }
                model.call = null;
                restoreHome();
                toast('phone-off', 'Call ended', '', 'contacts');
                break;
            case 'playpause':
                if (model.spotifyMode === 'live') { void sendSpotify({ action: model.playing ? 'pause' : 'play' }); break; }
                if (model.spotifyMode !== 'demo') break;
                model.playing = !model.playing;
                model.mediaLoaded = true;
                receipt(model.playing ? 'media.play' : 'media.pause', { source: 'demo' });
                if (model.view !== 'home')
                    renderView();
                break;
            case 'shuffle':
                if (model.spotifyMode === 'live') { void sendSpotify({ action: 'shuffle', value: !model.shuffle }); break; }
                if (model.spotifyMode !== 'demo') break;
                model.shuffle = !model.shuffle;
                receipt('media.shuffle', { state: model.shuffle });
                renderView();
                break;
            case 'repeat':
                if (model.spotifyMode === 'live') { void sendSpotify({ action: 'repeat', value: { off: 'context', context: 'track', track: 'off' }[model.repeat] }); break; }
                if (model.spotifyMode !== 'demo') break;
                model.repeat = { off: 'context', context: 'track', track: 'off' }[model.repeat];
                receipt('media.repeat', { state: model.repeat });
                renderView();
                break;
            case 'stop-music':
                if (model.spotifyMode === 'live') { void sendSpotify({ action: 'pause' }); break; }
                if (model.spotifyMode !== 'demo') break;
                model.playing = false;
                model.mediaLoaded = false;
                model.position = 0;
                receipt('media.stop', {});
                restoreHome();
                break;
            case 'next':
            case 'previous':
                if (model.spotifyMode === 'live') { void sendSpotify({ action: b.dataset.action }); break; }
                if (model.spotifyMode !== 'demo') break;
                model.song = (model.song + (b.dataset.action === 'next' ? 1 : 2)) % 3;
                model.position = 0;
                model.mediaLoaded = true;
                receipt('media.' + b.dataset.action, {});
                renderView();
                break;
        }
    });
    on('input', event => { stopAuto(); const input = event.target; switch (input.dataset.input) {
        case 'temperature-sensor':
            selectTemperatureSensor(input.value);
            input.blur();
            break;
        case 'brightness':
            if (canControlLights()) lightRangeEditing = true;
            setLightBrightness(model.selectedLight, Number(input.value));
            break;
        case 'seek':
            if (!canControlSpotify()) break;
            mediaEditing = true;
            model.position = Number(input.value);
            break;
        case 'volume':
            if (!canControlSpotify() || model.volume === null) break;
            mediaEditing = true;
            model.volume = Number(input.value);
            if (model.spotifyMode === 'demo') receipt('media.volume', { level: model.volume });
            break;
        case 'alarm':
            model.alarmDraft = input.value;
            break;
        case 'reminder':
            model.reminderDraft = input.value;
            break;
        case 'deadline':
            model.reminderDeadline = input.value;
            break;
    } });
    on('change', event => {
        const kind = event.target.dataset.input;
        if (kind === 'brightness') { lightRangeEditing = false; recordLightBrightness(model.selectedLight); }
        if (kind === 'seek' || kind === 'volume') {
            mediaEditing = false;
            if (model.spotifyMode === 'live') void sendSpotify({ action: kind, value: Math.round(kind === 'seek' ? model.position * 1000 : model.volume) });
            else if (kind === 'seek') receipt('media.seek', { position_ms: Math.round(model.position * 1000) });
        }
    });
    on('pointercancel', () => {
        lightRangeEditing = false; mediaEditing = false;
        if (model.haMode === 'live' && lastHaState) applyHomeAssistant(lastHaState);
        if (model.spotifyMode === 'live' && lastSpotifyState) applySpotify(lastSpotifyState);
    });
    on('wheel', () => { model.lastInteraction = Date.now(); }, { passive: true });
    on('submit', event => { event.preventDefault(); if (event.target.dataset.form === 'reminder')
        saveReminder(); });
    on('keydown', event => {
        model.lastInteraction = Date.now();
        if (event.target.matches('.thermostat')) {
            if (['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft', 'Home', 'End'].includes(event.key)) {
                event.preventDefault();
                adjustTemperature(event.key === 'Home' ? 16 : event.key === 'End' ? 30 : model.target + (['ArrowUp', 'ArrowRight'].includes(event.key) ? 1 : -1));
            }
        }
        if (event.key === 'Escape') {
            event.preventDefault();
            if (model.alert)
                dismissAlert(false);
            else if (model.view !== 'home')
                restoreHome();
            else
                closeToast();
        }
        if (event.key === 'Tab' && (model.alert || model.view !== 'home')) {
            const parents = model.alert ? [q('.alert-surface')] : [q('.task-view'), q('.call-window')].filter(Boolean);
            const nodes = parents.flatMap(parent => [...parent.querySelectorAll('button,input,select,[tabindex="0"]')]).filter(n => !n.disabled && !n.closest('[hidden]'));
            if (event.shiftKey && document.activeElement === nodes[0]) {
                nodes.at(-1).focus();
                event.preventDefault();
            }
            else if (!event.shiftKey && document.activeElement === nodes.at(-1)) {
                nodes[0].focus();
                event.preventDefault();
            }
        }
    });
    on('pointerover', event => { if (event.target.closest('.toast') && !event.relatedTarget?.closest?.('.toast') && model.toast) {
        model.toast.remaining = Math.max(0, model.toast.until - Date.now());
        model.toast.held = true;
    } });
    on('pointerout', event => { if (event.target.closest('.toast') && !event.relatedTarget?.closest?.('.toast') && model.toast) {
        model.toast.held = false;
        model.toast.until = Date.now() + Math.max(1500, model.toast.remaining);
    } });
    function tick() { if (document.hidden)
        return; const now = Date.now(), perf = performance.now(); model.now = now; if (model.alert && now >= model.alert.closesAt)
        dismissAlert(false); const idle = ['climate', 'control', 'lights', 'music', 'weather'].includes(model.view) && !model.alert && !model.alarmPickerOpen && !document.activeElement?.matches('[data-input=temperature-sensor]') && dialPointer === null && !lightDrag && !reminderOrderDrag && !reminderDrag ? now - model.lastInteraction : 0, closesAt = model.view === 'control' ? 60000 : 10000, strokeAt = closesAt - 5000; model.closeProgress = idle < strokeAt ? null : Math.min(1, (idle - strokeAt) / 5000); if (idle >= closesAt)
        restoreHome(); if (model.auto) {
        model.auto.remaining -= Math.min(perf - model.auto.last, 350);
        model.auto.last = perf;
        updateAuto();
        if (model.auto.remaining <= 0) {
            const action = model.auto.action;
            if (action === 'reminder')
                saveReminder();
            else if (action === 'timer')
                saveTimer();
            else if (action === 'alarm')
                saveAlarm();
        }
    } if (model.toast && !model.toast.held && now >= model.toast.until)
        closeToast(); if (model.call)
        drawCamera(); advancePlayback(now); checkDeadlines(); }
    function mount(element) { root = element; disposed = false; commandLifetime = new AbortController(); cameraBackground = new Image(); cameraBackground.src = '/ambient/camera-grid.svg'; const exitFullscreen = installFullscreen(root); backend = connectDummyBackend(model, dispatch, setAgent); providers = connectProviderStates({ ha: applyHomeAssistant, spotify: applySpotify, failed: providerFailed }, fetch, document); const abort = new AbortController(); for (const { event, handler, options } of handlers)
        root.addEventListener(event, handler, { ...options, signal: abort.signal }); root.addEventListener('vcm-intent', event => { if (!model.runtimeMode) dispatch(event.detail); }, { signal: abort.signal }); aura = createAuraRenderer(q('.livekit-aura'), { onFailure: message => { model.auraError = message; } }); const visibility = () => { if (model.auto)
        model.auto.last = performance.now(); if (!document.hidden)
        tick(); }; document.addEventListener('visibilitychange', visibility, { signal: abort.signal }); setAgent('idle'); tick(); const ticker = setInterval(tick, 250); return () => { disposed = true; commandLifetime.abort(); exitFullscreen(); backend.close(); providers.close(); abort.abort(); clearInterval(ticker); clearInterval(speechInterval); for (const id of timeouts)
        clearTimeout(id); timeouts.clear(); aura.destroy(); ++model.request; }; }
    return { model, mount, open, dispatch, setAgent, receipt, connectSpotify, canControlLights, canControlSpotify, canDim, liveShortcut, colors, get tracks() { return model.tracks; }, DEMO, AURA_STATES, TEMPERATURE_SENSORS, temperatureCelsius, supportsColor, duration, clockTime, nextAlarm, until, timerRemaining, drawCamera, alarmChange(value) { model.alarmDraft = value; model.lastInteraction = Date.now(); } };
}
