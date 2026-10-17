import { isRecord } from './protocol.js';

export interface CanonicalCommand { intent: string; slots: Record<string, string | number> }
export const taskLabels: Record<string, string> = { 'task.drink_water': 'Drink water', 'task.study': 'Study', 'task.exercise': 'Exercise' };
const required: Record<string, string[]> = {
  PLAY_MUSIC: [], WEATHER: [], TIME: [], LIGHT_ON: ['light_id'], LIGHT_OFF: ['light_id'],
  PAUSE: [], STOP: [], NEXT: [], VOLUME: ['operation'], CALL: ['contact_id'], MESSAGE: [],
  LIST_REMINDERS: [], TIMER: ['duration_seconds'], ALARM: ['time', 'timezone'],
  TEMPERATURE: ['degrees', 'unit'], BRIGHTNESS: ['light_id', 'percent'], COLOR: ['light_id', 'color'], CREATE_REMINDER: ['task_id'],
};
const colors = new Set(['red', 'blue', 'green', 'yellow', 'white', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'warm white', 'cool white']);

/** Mirrors pcv_schema.validate_command; validates a typed command, never parses speech. */
export function canonicalCommand(value: unknown): CanonicalCommand | undefined {
  if (!isRecord(value) || Object.keys(value).length !== 2 || typeof value.intent !== 'string' || !Object.hasOwn(required, value.intent) || !isRecord(value.slots)) return;
  const { intent, slots } = value;
  const keys = required[intent]!;
  const optional = intent === 'PLAY_MUSIC' ? ['song_id', 'artist_id'] : intent === 'VOLUME' ? ['percent'] : [];
  if (keys.some(key => !Object.hasOwn(slots, key)) || Object.keys(slots).some(key => !keys.includes(key) && !optional.includes(key))) return;
  for (const [field, item] of Object.entries(slots)) {
    if (['percent', 'duration_seconds', 'degrees'].includes(field)) {
      const [low, high] = field === 'duration_seconds' ? [1, 86400] : field === 'degrees' ? [-50, 60] : [0, 100];
      if (typeof item !== 'number' || !Number.isInteger(item) || item < low! || item > high!) return;
    } else {
      if (typeof item !== 'string' || !item || item.length > 200) return;
      if (field.endsWith('_id') && (!/^[a-z]+\.[a-z0-9_.-]+$/.test(item) || !item.startsWith(field.slice(0, -3) + '.'))) return;
    }
  }
  if (intent === 'VOLUME' && (!['up', 'down', 'mute', 'set'].includes(String(slots.operation)) || (slots.operation === 'set') !== Object.hasOwn(slots, 'percent'))) return;
  if (intent === 'PLAY_MUSIC' && Object.hasOwn(slots, 'artist_id') && !Object.hasOwn(slots, 'song_id')) return;
  if (intent === 'ALARM' && (!/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(String(slots.time)) || slots.timezone !== 'Asia/Manila')) return;
  if (intent === 'TEMPERATURE' && slots.unit !== 'C') return;
  if (intent === 'COLOR' && !colors.has(String(slots.color))) return;
  return { intent, slots: { ...slots } as CanonicalCommand['slots'] };
}
