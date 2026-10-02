import { randomUUID } from 'node:crypto';
import WebSocket, { type RawData } from 'ws';
import { AUDIO, parseControl, parseInference, type Command, type ServerEvent } from '../../shared/protocol.js';
import type { Config } from '../config.js';
import { createMockInference, type InferenceAdapter, type InferenceFactory } from './adapter.js';
import { evaluateIntent, ResultGate } from './policy.js';

export interface ActionResult { status: 'succeeded' | 'failed' | 'unknown'; detail: string; data?: unknown }
export interface RecordingLease { owner: WebSocket | undefined }
interface Utterance {
  id: string; phase: 'recording' | 'processing'; bytes: number; queued: number;
  chain: Promise<void>; adapter: InferenceAdapter; gate: ResultGate;
  timer: ReturnType<typeof setTimeout>; finalSeen: boolean;
}

export function attachVoiceSession(socket: WebSocket, config: Config, lease: RecordingLease,
  dispatch: (command: Command) => Promise<ActionResult>, factory: InferenceFactory = createMockInference) {
  let active: Utterance | undefined;
  const seen = new Set<string>();
  let alive = true;
  function send(event: ServerEvent) {
    if (socket.readyState !== WebSocket.OPEN) return;
    if (socket.bufferedAmount > AUDIO.maxQueuedBytes) { socket.terminate(); return; }
    socket.send(JSON.stringify(event));
  }
  function finish(state: Utterance) {
    clearTimeout(state.timer); state.gate.cancel(); state.adapter.cancel();
    if (active !== state) return;
    active = undefined;
    if (lease.owner === socket) lease.owner = undefined;
    send({ type: 'state', utteranceId: state.id, state: 'idle' });
  }
  function fail(code: string, message: string) {
    send({ type: 'error', code, message, ...(active ? { utteranceId: active.id } : {}) });
    if (active) finish(active);
  }
  async function receiveResult(raw: unknown, state: Utterance) {
    if (active !== state) return;
    const result = parseInference(raw);
    if (!result) { fail('invalid_inference', 'Inference emitted an invalid result.'); return; }
    if (!state.gate.accept(result)) return;
    if (result.final && state.phase !== 'processing') { fail('early_final', 'Inference finalized before the utterance ended.'); return; }
    send({ type: 'inference', simulated: state.adapter.simulated, result });
    if (!result.final) return;
    state.finalSeen = true;
    clearTimeout(state.timer);
    const decision = evaluateIntent(result, state.adapter.simulated ? 0.9 : config.threshold, config.ha.entityIds);
    try {
      if (decision.status !== 'accepted') send({ type: 'decision', utteranceId: state.id, ...decision });
      else if (state.adapter.simulated) send({ type: 'action', utteranceId: state.id, status: 'simulated', detail: 'MOCK ONLY: no speech recognition or integration action occurred.', data: decision.command });
      else if (!config.actionsEnabled) send({ type: 'action', utteranceId: state.id, status: 'disabled', detail: 'Action dispatch is disabled.' });
      else {
        const outcome = await dispatch(decision.command);
        if (active === state) send({ type: 'action', utteranceId: state.id, ...outcome });
      }
    } catch {
      if (active === state) send({ type: 'action', utteranceId: state.id, status: 'unknown', detail: 'Action failed without a confirmed outcome; it will not be retried.' });
    } finally { if (active === state) finish(state); }
  }
  function enqueue(state: Utterance, operation: () => void | Promise<void>) {
    state.chain = state.chain.then(async () => { if (active === state) await operation(); }).catch(() => { if (active === state) fail('inference_unavailable', 'Inference failed; the utterance was cancelled.'); });
  }
  function bytes(raw: RawData): Buffer {
    if (Array.isArray(raw)) return Buffer.concat(raw);
    return Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
  }
  // Attach synchronously: ws messages can arrive before asynchronous setup completes.
  socket.on('message', (raw, binary) => {
    const data = bytes(raw);
    if (binary) {
      const state = active;
      if (!state || state.phase !== 'recording') { fail('not_recording', 'Start an utterance before sending audio.'); return; }
      if (data.length === 0 || data.length > AUDIO.frameBytes || data.length % 2 !== 0) { fail('invalid_audio', 'Expected a PCM16 frame of at most 640 bytes.'); return; }
      if (state.bytes + data.length > AUDIO.maxBytes || state.queued + data.length > AUDIO.maxQueuedBytes) { fail('audio_limit', 'Audio duration or queue limit exceeded.'); return; }
      state.bytes += data.length; state.queued += data.length;
      enqueue(state, async () => { try { await state.adapter.writeAudio(data); } finally { state.queued -= data.length; } });
      return;
    }
    let decoded: unknown;
    try { decoded = JSON.parse(data.toString()); } catch { fail('invalid_control', 'Control must be JSON.'); return; }
    const control = parseControl(decoded);
    if (!control) { fail('invalid_control', 'Invalid control or utterance ID.'); return; }
    if (control.type === 'start') {
      if (active || lease.owner) { send({ type: 'error', code: 'busy', message: 'An utterance is already active.', utteranceId: control.utteranceId }); return; }
      if (seen.has(control.utteranceId)) { send({ type: 'error', code: 'reused_id', message: 'Use a fresh utterance ID.', utteranceId: control.utteranceId }); return; }
      if (seen.size >= 1_024) { socket.close(1000, 'Session limit reached; reconnect.'); return; }
      seen.add(control.utteranceId); lease.owner = socket;
      let state: Utterance;
      const adapter = factory((result) => { void receiveResult(result, state); });
      const gate = new ResultGate(); gate.start(control.utteranceId);
      state = { id: control.utteranceId, phase: 'recording', bytes: 0, queued: 0, chain: Promise.resolve(), adapter, gate, finalSeen: false, timer: setTimeout(() => { if (active === state) fail('duration_limit', 'Maximum utterance duration is 15 seconds.'); }, AUDIO.maxDurationMs) };
      active = state;
      enqueue(state, () => adapter.start(state.id));
      send({ type: 'state', utteranceId: state.id, state: 'recording' });
      return;
    }
    const state = active;
    if (!state || state.id !== control.utteranceId) return;
    if (control.type === 'cancel') { finish(state); return; }
    if (state.phase !== 'recording') return;
    if (state.bytes === 0) { fail('empty_audio', 'No audio frames were received.'); return; }
    state.phase = 'processing'; clearTimeout(state.timer);
    state.timer = setTimeout(() => { if (active === state) fail('inference_timeout', 'No final inference result arrived within 10 seconds.'); }, 10_000);
    send({ type: 'state', utteranceId: state.id, state: 'processing' });
    enqueue(state, () => state.adapter.end());
  });
  socket.on('pong', () => { alive = true; });
  const heartbeat = setInterval(() => { if (!alive) { socket.terminate(); return; } alive = false; socket.ping(); }, 15_000);
  heartbeat.unref();
  function cleanup() { clearInterval(heartbeat); if (active) finish(active); }
  socket.on('close', cleanup);
  socket.on('error', cleanup);
  send({ type: 'ready', sessionId: randomUUID(), mode: 'mock' });
}
