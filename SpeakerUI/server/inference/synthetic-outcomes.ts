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
  private state: Record<string, any> = {};
  snapshot() { return structuredClone(this.state); }
  complete(command: CanonicalCommand, reason: string): Execution {
    if (command.intent === 'TEMPERATURE') this.state.thermostat = { target: command.slots.degrees, unit: 'C' };
    const source = provenance('synthetic');
    return { status: 'synthetic', detail: `Synthetic — no device changed. ${reason}`, command,
      ...source, provenance: source, synthetic_state: this.snapshot() };
  }
}
