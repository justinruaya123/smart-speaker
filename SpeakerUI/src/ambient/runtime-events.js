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
    if(event.event==='feedback_error'){model.runtimeAudioError=event.message;return;}
    if(event.event==='execution'){
        if(!event.execution || event.execution.replayed)return false;
        const key=event.execution.id || `${event.session_id || model.runtimeSession}:${event.utterance_id}`;
        const received=model.receivedExecutions ??= {};
        const previous=received[key];
        if(previous && (previous===event.execution.status || !['pending','requested'].includes(previous)))return false;
        received[key]=event.execution.status;
        (model.executionReceipts??={})[key]=structuredClone(event.execution);
        const item=(model.history||[]).find(item=>item.result.utteranceId===event.utterance_id && (!event.session_id || item.raw?.session_id === event.session_id));
        if(!item)return true;
        if(item.execution && !['pending','requested'].includes(item.execution.status))return false;
        item.execution=event.execution;item.status=event.execution.status;
        if(model.event?.result.utteranceId===event.utterance_id)model.event={...item};
        if(event.execution.status==='local'){
            const command=event.execution.command,slots=command.slots;
            if(command.intent==='TIMER')model.timer={total:slots.duration_seconds,remaining:slots.duration_seconds,end:event.execution.deadline};
            if(command.intent==='ALARM')model.alarm={time:slots.time,deadline:event.execution.deadline};
            if(command.intent==='LIST_REMINDERS')model.view='reminders';
            if(command.intent==='TIME'){model.view='home';model.announcement=event.execution.detail;}
            if(command.intent==='CREATE_REMINDER'){model.reminderDraft=({ 'task.drink_water':'Drink water','task.study':'Study','task.exercise':'Exercise' })[slots.task_id]||slots.task_id;model.reminderDeadline='';model.editingReminder=null;model.view='new-reminder';}
        }
        return true;
    }
    if (event.event === 'command' && (event.is_final === false || event.final === false)) return;
    if (model.runtimeSession === event.session_id && event.sequence <= (model.runtimeSequence ?? -1)) return;
    if (model.runtimeSession !== event.session_id) model.tentative = null;
    model.runtimeSession = event.session_id; model.runtimeSequence = event.sequence;
    if (event.event === 'tentative') {
        if (event.is_final !== false || !event.candidate || (model.history || []).some(item => item.raw?.session_id === event.session_id && item.result.utteranceId === event.utterance_id)) return;
        model.tentative = structuredClone(event);
        return;
    }
    if (['ready', 'reset', 'command', 'error', 'capture_warning'].includes(event.event) ||
        event.event === 'state' && ['completed','muted','wake_listening','cancel','reset','capture_error'].includes(event.state)) model.tentative = null;
    if (event.event === 'ownership') model.ownership = { accepted:event.accepted===true, reason:event.reason, experimental:event.ownership_experimental===true };
    if (event.event === 'ready') {
        model.runtimeReady = event.loaded !== false; model.runtimeError = event.loaded === false ? 'Inference worker is unavailable.' : ''; model.runtimeEngine=event.engine||'onnxruntime';model.backend = model.runtimeEngine+(model.runtimeReady?' ready':' unavailable'); model.runtimePolicy = event.policy_enabled;model.runtimeConfig=event.application_config;model.runtimeUI=event.ui;model.runtimeActions=event.actions_enabled===true;
        model.runtimeIdentity={parameters:event.parameters,selected_epoch:event.selected_epoch,model_sha256:event.model_sha256,wake_phrase:event.wake_phrase,model_qualified:event.model_qualified};
    }
    if (event.event === 'state' || event.event === 'ready') {
        model.runtimeState = event.state;
        model.muted = event.state === 'muted';
        const state = ({ command_ready: 'listening', recording: 'listening', processing: 'thinking' })[event.state] ?? 'idle';
        setAgent(state);
        if (event.message) model.runtimeError = event.message;
        if (['wake_listening','processing'].includes(event.state)) model.runtimeError = '';
    }
    if (event.event === 'capture_warning' || event.event === 'error' || event.event === 'wake_result') {
        model.runtimeError = event.message || event.reason;
        model.runtimeResult = event;
        if (event.fatal === true) { model.runtimeReady=false; model.runtimeState='runtime_error'; model.runtimeActions=false; }
    }
    if (event.event !== 'command') return;
    if ((model.history || []).some(item => item.raw?.session_id === event.session_id && item.result.utteranceId === event.utterance_id)) return;
    if (event.owner_result) model.ownership = { ...event.owner_result, experimental:event.ownership_experimental===true };
    const output = event.output;
    if (output?.schema_version !== 3 || typeof output.accepted !== 'boolean') { model.runtimeError = 'Unsupported inference result'; return; }
    const accepted = event.application ? event.application.accepted===true : output.accepted === true && output.command !== null;
    const command=event.application?.command||output.command;
    const mapped = accepted && command ? describeCommand(command) : { intent: 'rejected', parameters: {} };
    model.runtimeResult = structuredClone(event);
    model.event = { type: 'inference', simulated: false, accepted, source: 'onnx-' + event.source,
        timestamp: new Date().toISOString(), actionsEnabled: event.actions_enabled===true, status: accepted ? (event.actions_enabled?'pending':'accepted-preview') : 'rejected',
        reason: event.application?.reason||output.reason, raw: structuredClone(event),
        result: { utteranceId: event.utterance_id || `${event.session_id}-${event.sequence}`, final: true,
            ...mapped, latencyMs: Number.isFinite(event.latency_ms)?event.latency_ms:event.timing?.total_ms,
            commandPosterior:event.command_probability??event.diagnostic_candidate?.class_probability,
            commandMargin:event.command_margin??event.diagnostic_candidate?.margin } };
    const earlierReceipt=model.executionReceipts?.[`${event.session_id}:${event.utterance_id}`];
    if(earlierReceipt){model.event.execution=earlierReceipt;model.event.status=earlierReceipt.status;}
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
