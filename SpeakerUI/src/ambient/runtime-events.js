/** Schema-v3 display adapter. These results never enter action dispatch. */
const routes = {
    PLAY_MUSIC: 'media.play', WEATHER: 'answer.weather', TIME: 'answer.time',
    LIGHT_ON: 'light.on', LIGHT_OFF: 'light.off', PAUSE: 'media.pause', STOP: 'media.stop', NEXT: 'media.next',
    VOLUME: 'media.volume', CALL: 'call.start', MESSAGE: 'message.compose', LIST_REMINDERS: 'reminder.list',
    TIMER: 'timer.set', ALARM: 'alarm.set', TEMPERATURE: 'climate.set_temperature',
    BRIGHTNESS: 'light.brightness', COLOR: 'light.color', CREATE_REMINDER: 'reminder.create',
};

export function describeCommand(command) {
    const slots = structuredClone(command.slots);
    let parameters = structuredClone(slots);
    if (command.intent === 'TIMER') parameters = { seconds: slots.duration_seconds };
    if (command.intent === 'TEMPERATURE') parameters = { temperature: slots.degrees, unit: slots.unit };
    if (command.intent === 'BRIGHTNESS') parameters = { light_id: slots.light_id, brightness_pct: slots.percent };
    return { intent: routes[command.intent] ?? 'unsupported', modelIntent: command.intent, slots, parameters };
}

export function applyRuntimeEvent(model, event, setAgent) {
    if(event.event==='wake_config'||event.event==='ready'){if(event.wake)model.runtimeWake=event.wake;}
    if(event.event==='notification'){model.announcement=event.text;model.runtimeNotice=event.text;setAgent('speaking');return;}
    if(event.event==='speech_error'){model.runtimeError=event.message;return;}
    if(event.event==='response'){model.announcement=event.text;setAgent('speaking');}
    if(event.event==='execution'){
        const item=(model.history||[]).find(item=>item.result.utteranceId===event.utterance_id);
        if(!item||item.execution)return;
        item.execution=event.execution;item.status=event.execution.status;
        model.announcement=event.execution.text||event.execution.detail;
        if(event.execution.command?.intent==='WEATHER'){
            model.view='weather';model.weather=event.execution.weather||null;
            model.weatherError=event.execution.weather?'':event.execution.detail;
        }
        if(event.execution.reminders)model.reminders=event.execution.reminders.map(r=>({...r,deadline:r.deadline??null,unread:!r.done,nearShown:false,dueShown:false}));
        if(model.event?.result.utteranceId===event.utterance_id)model.event={...item};
        if(event.execution.status==='local'){
            const command=event.execution.command,slots=command.slots;
            if(command.intent==='CALL'&&event.execution.camera){
                const camera=event.execution.camera;
                model.call={kind:'camera',entityId:camera.entityId,name:camera.name,started:Date.now()};
                model.cameraEnabled=true;model.view='call';
            }
            if(command.intent==='TIMER')model.timer={total:slots.duration_seconds,remaining:slots.duration_seconds,end:event.execution.deadline};
            if(command.intent==='ALARM')model.alarm={time:slots.time,deadline:event.execution.deadline};
            if(command.intent==='LIST_REMINDERS')model.view='reminders';
            if(command.intent==='TIME')model.view='home';
            if(command.intent==='CREATE_REMINDER')model.view='reminders';
        }
        return;
    }
    if (model.runtimeSession === event.session_id && event.sequence <= (model.runtimeSequence ?? -1)) return;
    model.runtimeSession = event.session_id; model.runtimeSequence = event.sequence;
    if (event.event === 'ready') { model.runtimeReady = true; model.runtimeEngine=event.engine||'onnxruntime';model.backend = model.runtimeEngine+' ready'; model.runtimePolicy = event.policy_enabled;model.runtimeConfig=event.application_config;model.runtimeUI=event.ui;model.runtimeActions=event.actions_enabled===true; }
    if (event.event === 'state' || event.event === 'ready') {
        model.runtimeState = event.state;
        model.muted = event.state === 'muted';
        const state = ({ command_ready: 'listening', recording: 'listening', processing: 'thinking', responding:'speaking' })[event.state] ?? 'idle';
        setAgent(state);
        if (event.message) model.runtimeError = event.message;
        if (['wake_listening','processing'].includes(event.state)) model.runtimeError = '';
    }
    if (event.event === 'capture_warning' || event.event === 'error' || event.event === 'wake_result') {
        model.runtimeError = event.message || event.reason;
        model.runtimeResult = event;
    }
    if (event.event !== 'command') return;
    const output = event.output;
    if (output?.schema_version !== 3 || typeof output.accepted !== 'boolean') { model.runtimeError = 'Unsupported inference result'; return; }
    const accepted = event.application ? event.application.accepted===true : output.accepted === true && output.command !== null;
    const command=event.application?.command||output.command;
    const mapped = accepted && command ? describeCommand(command) : { intent: 'rejected', parameters: {} };
    model.runtimeResult = structuredClone(event);
    model.event = { type: 'inference', simulated: false, source: 'onnx-' + event.source,
        timestamp: new Date().toISOString(), actionsEnabled: event.actions_enabled===true, status: accepted ? (event.actions_enabled?'pending':'accepted-preview') : 'rejected',
        reason: event.application?.reason||output.reason, raw: structuredClone(event),
        result: { utteranceId: event.utterance_id || `${event.session_id}-${event.sequence}`, final: true,
            ...mapped, latencyMs: event.timing?.total_ms } };
    model.history = [model.event, ...(model.history || [])].slice(0, 30);
    model.historyIndex = 0; model.receiptAt = Date.now();
    // Stop simulator auto-actions when displaying real inference.
    model.auto = null; setAgent('idle');
}

export function connectRuntime(model, setAgent) {
    let closed = false, socket, retry;
    const connect = () => {
        socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/runtime`);
        socket.onmessage = message => {
            try { applyRuntimeEvent(model, JSON.parse(message.data), setAgent); }
            catch { model.runtimeError = 'Invalid runtime event'; }
        };
        socket.onclose = () => {
            model.runtimeReady = false; model.backend = 'Runtime disconnected'; setAgent('idle');
            if (!closed) retry = setTimeout(connect, 2000);
        };
    };
    connect();
    return { close() { closed = true; clearTimeout(retry); socket?.close(); } };
}
