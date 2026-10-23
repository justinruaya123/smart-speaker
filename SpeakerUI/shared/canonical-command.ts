import { isRecord } from './protocol.js';

export interface CanonicalCommand { intent: string; slots: Record<string, string | number> }
/** Presentation groups for the frozen 31 complete commands, not a speech parser or new classes. */
export const irisUseCases: ReadonlyArray<{ id:string; view:string; icon:string; title:string; intents:readonly string[] }> = [
  {id:'UC-01',view:'music',icon:'music-2',title:'Music',intents:['PLAY_MUSIC']},
  {id:'UC-02',view:'weather',icon:'cloud-sun',title:'Questions',intents:['WEATHER','TIME']},
  {id:'UC-03',view:'lights',icon:'lightbulb',title:'Lights',intents:['LIGHT_ON','LIGHT_OFF']},
  {id:'UC-04',view:'dim',icon:'sliders-horizontal',title:'Dim / color',intents:['BRIGHTNESS','COLOR']},
  {id:'UC-05',view:'timer',icon:'timer',title:'Timer',intents:['TIMER']},
  {id:'UC-06',view:'alarm',icon:'alarm-clock',title:'Alarm',intents:['ALARM']},
  {id:'UC-07',view:'climate',icon:'thermometer',title:'Temperature',intents:['TEMPERATURE']},
  {id:'UC-08',view:'volume',icon:'volume-2',title:'Media',intents:['PAUSE','STOP','NEXT','VOLUME']},
  {id:'UC-09',view:'reminders',icon:'list-checks',title:'Reminders',intents:['CREATE_REMINDER','LIST_REMINDERS']},
  {id:'UC-10',view:'calls',icon:'phone',title:'Calls / messages',intents:['CALL','MESSAGE']},
];
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
