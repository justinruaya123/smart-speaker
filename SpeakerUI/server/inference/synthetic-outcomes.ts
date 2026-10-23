import type { CanonicalCommand } from '../../shared/canonical-command.js';
import type { Execution } from './action-store.js';

export type ResultSource = 'home_assistant' | 'spotify' | 'openweather' | 'local' | 'cached' | 'synthetic';
export type IrisProvenance = {
  source: ResultSource;
  synthetic: boolean;
  provider_accepted: boolean;
  device_state_confirmed: boolean;
  observed_at?: string;
  fetched_at?: string;
};

export function provenance(source: ResultSource, accepted = false, confirmed = false): IrisProvenance {
  return { source, synthetic: source === 'synthetic', provider_accepted: accepted, device_state_confirmed: confirmed };
}

/** Targets are simulated state, never observations or provider acknowledgments. */
export class SyntheticOutcomes {
  constructor(private state: Record<string, any> = {}) { this.state = structuredClone(state); }
  snapshot() { return structuredClone(this.state); }
  complete(command: CanonicalCommand, reason: string): Execution {
    const { intent, slots } = command;
    if (['LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR'].includes(intent)) {
      this.state.lights ??= {};
      const light = this.state.lights['light.hue_color_lamp_1'] ??= {};
      if (intent === 'LIGHT_ON' || intent === 'LIGHT_OFF') light.on = intent === 'LIGHT_ON';
      if (intent === 'BRIGHTNESS') { light.percent = slots.percent; light.on = Number(slots.percent) > 0; }
      if (intent === 'COLOR') { light.color = slots.color; light.on = true; }
    }
    if (['PLAY_MUSIC','PAUSE','STOP','NEXT','VOLUME'].includes(intent)) {
      const playback = this.state.playback ??= {};
      if (intent === 'PLAY_MUSIC') playback.state = 'playing';
      if (intent === 'PAUSE') playback.state = 'paused';
      if (intent === 'STOP') playback.state = 'stopped';
      if (intent === 'NEXT') playback.next_requests = (playback.next_requests || 0) + 1;
      if (intent === 'VOLUME') playback.volume_operation = { ...slots };
    }
    if (command.intent === 'TEMPERATURE') this.state.thermostat = { target: command.slots.degrees, unit: 'C' };
    const source = provenance('synthetic');
    return { status: 'synthetic', detail: `Synthetic — no device changed. ${reason}`, command,
      ...source, provenance: source, synthetic_state: this.snapshot(),
      ...(intent === 'CALL' ? { camera: { entity_id:'camera.dining_area_main_camera_tc70_mainstream',video_only:true,stream_url:null,available:false } } : {}),
      ...(intent === 'MESSAGE' ? { notification: { preset:true,delivered:false } } : {}) };
  }
}
