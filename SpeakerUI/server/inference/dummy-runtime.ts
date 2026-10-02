import { randomUUID } from 'node:crypto';
import { isRecord } from '../../shared/protocol.js';

/** Structured fixtures only. This is an ONNX boundary substitute, not a native session or ASR. */
export const ambientFixtures = [
  { intent: 'light.brightness', parameters: { entity_id: 'light.living_room', brightness_pct: 40 } },
  { intent: 'light.list', parameters: {} },
  { intent: 'climate.set_temperature', parameters: { entity_id: 'climate.living_room', temperature: 24 } },
  { intent: 'timer.set', parameters: { seconds: 300 } },
  { intent: 'alarm.set', parameters: { time: '07:00' } },
  { intent: 'reminder.create', parameters: {} },
  { intent: 'media.play', parameters: {} },
  { intent: 'answer.weather', parameters: { location: 'Manila' } },
  { intent: 'call.start', parameters: { contact: 'Mom' } },
  { intent: 'call.start', parameters: { contact: 'Front door' } },
  { intent: 'demo.reminder_due', parameters: {} },
  { intent: 'demo.timer_due', parameters: {} },
  { intent: 'demo.alarm_due', parameters: {} },
  { intent: 'demo.reminder_near', parameters: {} },
  { intent: 'light.brightness', parameters: { entity_id: 'light.unknown', brightness_pct: 50 } },
] as const;

export function parseFixtureRequest(value: unknown): number | undefined {
  if (!isRecord(value) || Object.keys(value).length !== 1) return;
  const { fixture } = value;
  if (typeof fixture !== 'number' || !Number.isInteger(fixture) || fixture < 0 || fixture >= ambientFixtures.length) return;
  return fixture;
}

export function loadDummyRuntime() {
  let closed = false;
  const loadedAt = new Date().toISOString();
  return {
    status() { return { engine: 'dummy-onnx', loaded: !closed, loadedAt, nativeSession: false, fixtures: ambientFixtures.length } as const; },
    infer(index: number) {
      if (closed) throw new Error('Dummy inference runtime is closed.');
      const fixture = ambientFixtures[index];
      if (!fixture) throw new RangeError('Unknown inference fixture.');
      return { simulated: true, engine: 'dummy-onnx', result: { utteranceId: randomUUID(), revision: 0, final: true, confidence: 1, ...structuredClone(fixture) } };
    },
    close() { closed = true; },
  };
}
