import { randomUUID } from 'node:crypto';
import { lightCommand, spotifyCommand, type LightCommand, type SpotifyCommand } from '../integrations/commands.js';
import { IntegrationError } from '../integrations/http.js';
import { canonicalCommand, taskLabels, type CanonicalCommand } from '../../shared/canonical-command.js';
import type { ActionBindings } from '../config.js';
import { ActionStore, type Execution } from './action-store.js';
import { SyntheticOutcomes, provenance, type ResultSource } from './synthetic-outcomes.js';
import { syntheticWeather } from '../integrations/openweather.js';

type Providers = {
  light: (command: LightCommand) => Promise<unknown>;
  lights?: (commands: LightCommand[]) => Promise<unknown>;
  spotify: (command: SpotifyCommand) => Promise<unknown>;
  song: (uri: string) => Promise<unknown>;
  volume?: () => Promise<number | undefined>;
  climate?: (entity: string, degrees: number) => Promise<unknown>;
  weather?: (entity: string) => Promise<unknown>;
  script?: (entity: string) => Promise<unknown>;
  media?: (entity: string, command: CanonicalCommand, content?: string) => Promise<unknown>;
  local?: (command: CanonicalCommand, content?: string) => Promise<Record<string, unknown>>;
  camera?: (entity: string) => Promise<Record<string, unknown>>;
  openweather?: () => Promise<Execution>;
};
const colors: Record<string, number[]> = { red: [255, 0, 0], blue: [0, 0, 255], green: [0, 255, 0], yellow: [255, 255, 0], white: [255, 255, 255], orange: [255, 165, 0], purple: [128, 0, 128], pink: [255, 192, 203], cyan: [0, 255, 255], magenta: [255, 0, 255] };

/** One dispatcher per server. Tentative/recorded events never enter dispatch. */
export class RuntimeActions {
  private synthetic: SyntheticOutcomes;
  constructor(private config: { enabled: boolean; bindings: ActionBindings; profile?: string; timezone?: string; publication?: boolean }, private providers: Providers,
    readonly store = ActionStore.memory(), private now = Date.now) { this.synthetic = new SyntheticOutcomes(store.syntheticState()); }
  syntheticState() { return this.synthetic.snapshot(); }

  async handle(event: Record<string, any>, beforeDispatch?: () => Promise<void>): Promise<Execution | undefined> {
    if (event.event !== 'command' || event.is_final === false || event.final === false || !event.utterance_id) return;
    const profile=this.config.profile || event.profile;
    return this.dispatch(`${event.session_id}:${event.utterance_id}`, event.utterance_id,
      event.source === 'microphone', event.actions_enabled !== false && profile !== 'benchmark-compatible',
      profile === 'owner-protected' && event.owner_result?.accepted !== true
        ? { accepted:false,reason:event.owner_result?.reason || 'owner_check_required' } : event.application, beforeDispatch,event.session_id);
  }

  /** Explicit same-origin touch requests use this entry point, never a client-provided inference event. */
  async touch(requestId: string, command: unknown, runtimeEnabled: boolean) {
    const id = 'touch:' + requestId;
    const result = await this.dispatch(id, requestId, true, runtimeEnabled, { accepted: true, command });
    const previous = this.store.receipt(id);
    return result ?? (previous ? { ...previous, replayed: true } : undefined);
  }

  private async dispatch(id: string, utteranceId: string, live: boolean, runtimeEnabled: boolean, decision: any, beforeDispatch?: () => Promise<void>,sessionId?: string): Promise<Execution | undefined> {
    const outcome = (status: string, detail: string, extra: Record<string, any> = {}): Execution => ({ id, status, detail, utterance_id: utteranceId,...(sessionId?{session_id:sessionId}:{}), ...extra });
    if (!live) return outcome('preview', 'Recorded WAVs and fixtures do not execute actions.');
    if (!this.config.enabled || !runtimeEnabled) return outcome('disabled', 'Actions are disabled by configuration or the recognizer.');
    if (!decision?.accepted || !decision.command) return outcome('rejected', decision?.reason || 'No accepted application command.');
    const command = canonicalCommand(decision.command);
    if (!command) return outcome('rejected', 'Invalid canonical command or slots.');
    if (!await this.store.claim(id, command)) return;
    await beforeDispatch?.();
    let result: Execution;
    const canSimulate = ['LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR','PLAY_MUSIC','PAUSE','STOP','NEXT','VOLUME','CALL','MESSAGE'].includes(command.intent);
    try {
      result = await this.execute(command, outcome);
      if (this.config.publication && canSimulate && result.status === 'not_configured') result = outcome('synthetic', '', this.synthetic.complete(command, result.detail));
    }
    catch (error) {
      result = this.config.publication && canSimulate && error instanceof IntegrationError && !error.outcomeUnknown &&
        ['disabled','unavailable','authentication','not_authorized','no_device','not_found','unsupported_capability','rate_limited','quota_exceeded'].includes(error.code)
        ? outcome('synthetic', '', this.synthetic.complete(command, error.message)) : error instanceof IntegrationError
        ? outcome(error.outcomeUnknown ? 'unknown' : ['disabled', 'invalid_target', 'unsupported_capability'].includes(error.code) ? 'not_configured' : 'failed', error.message)
        : outcome('unknown', 'Action outcome could not be confirmed; it will not be retried.');
    }
    if (this.config.publication && !result.provenance) {
      const media = ['PLAY_MUSIC','PAUSE','STOP','NEXT','VOLUME'].includes(command.intent);
      const source: ResultSource = media ? this.config.bindings.media_player?.provider || 'local' : ['TIME','TIMER','ALARM','CREATE_REMINDER','LIST_REMINDERS'].includes(command.intent) ? 'local' : 'home_assistant';
      result = { ...result, command, provenance:provenance(source,result.provider_accepted === true,result.status === 'observed') };
    }
    try { return await this.store.finish(id, result); }
    catch { return outcome('unknown', 'The action was dispatched but its receipt could not be saved. It will not be retried.', { command }); }
  }

  private async execute(command: CanonicalCommand, outcome: (status: string, detail: string, extra?: Record<string, any>) => Execution): Promise<Execution> {
    const { intent, slots } = command, bindings = this.config.bindings;
    const missing = (name: string) => outcome('not_configured', `Configure ${name} before using this command.`, { command });
    const accepted = (provider: string) => outcome('succeeded', `${provider} accepted the service request. Observed device state updates separately.`, { command, provider, provider_accepted: true, device_state_confirmed: false });
    if (['LIGHT_ON', 'LIGHT_OFF', 'BRIGHTNESS', 'COLOR'].includes(intent)) {
      const target = this.config.publication ? 'light.hue_color_lamp_1' : bindings.lights[String(slots.light_id)];
      if (!target) return missing('bindings.lights.' + slots.light_id);
      const targets = Array.isArray(target) ? target : [target];
      const commands = targets.map(entityId => {
        let action: unknown = { entityId, action: intent === 'LIGHT_ON' ? 'on' : 'off' };
        if (intent === 'BRIGHTNESS') action = { entityId, action: 'brightness', value: slots.percent };
        if (intent === 'COLOR') action = ['warm white', 'cool white'].includes(String(slots.color))
          ? { entityId, action: 'temperature', value: slots.color === 'warm white' ? 2700 : 6500 }
          : { entityId, action: 'color', rgb: colors[String(slots.color)] };
        return lightCommand(action);
      });
      if (this.providers.lights) await this.providers.lights(commands);
      else for (const item of commands) await this.providers.light(item);
      return accepted('Home Assistant');
    }
    if (intent === 'TEMPERATURE') {
      if (this.config.publication) return outcome('synthetic', '', this.synthetic.complete(command, 'Thermostat target is simulated; no climate service was called.'));
      const target = bindings.climate?.default;
      if (!target || !this.providers.climate) return missing('bindings.climate.default');
      await this.providers.climate(target, Number(slots.degrees)); return accepted('Home Assistant');
    }
    if (intent === 'WEATHER') {
      if (this.config.publication) return outcome('observed', '', { command, ...(this.providers.openweather ? await this.providers.openweather() : syntheticWeather()) });
      const target = bindings.weather?.default;
      if (!target || !this.providers.weather) return missing('bindings.weather.default');
      const weather = await this.providers.weather(target);
      return outcome('observed', 'Current weather read from the configured Home Assistant entity.', { command, weather });
    }
    if (intent === 'CALL' || intent === 'MESSAGE') {
      if (intent === 'CALL' && this.config.publication) {
        if (!this.providers.camera) return missing('the dining camera');
        const camera = await this.providers.camera('camera.dining_area_main_camera_tc70_mainstream');
        return outcome('observed', 'Home camera • video only. No telephone call or microphone audio.', { command, camera });
      }
      const target = intent === 'CALL' ? bindings.contacts?.[String(slots.contact_id)] : bindings.messages?.default;
      if (!target || !this.providers.script) return missing(intent === 'CALL' ? 'bindings.contacts.' + slots.contact_id : 'bindings.messages.default (a user-defined preset script)');
      await this.providers.script(target); return accepted('Home Assistant script');
    }
    if (['PLAY_MUSIC', 'PAUSE', 'STOP', 'NEXT', 'VOLUME'].includes(intent)) {
      const media = bindings.media_player;
      if (!media?.provider) return missing('bindings.media_player.provider');
      const content = slots.song_id ? bindings.songs[String(slots.song_id)] : undefined;
      if (slots.song_id && !content) return missing('bindings.songs.' + slots.song_id);
      if (media.provider === 'home_assistant') {
        if (!media.entity_id || !this.providers.media) return missing('bindings.media_player.entity_id');
        await this.providers.media(media.entity_id, command, content); return accepted('Home Assistant media player');
      }
      if (media.provider === 'local') {
        if (!this.providers.local) return missing('the local browser player');
        const data = await this.providers.local(command, content);
        return outcome('requested', 'Local browser playback requested; player confirmation is pending.', { command, local_media: data });
      }
      if (intent === 'STOP') return missing('a provider with a stop action; Spotify only supports pause');
      if (intent === 'PLAY_MUSIC' && content) await this.providers.song(content);
      else {
        let action: unknown = { action: ({ PLAY_MUSIC: 'play', PAUSE: 'pause', NEXT: 'next' } as Record<string, string>)[intent] };
        if (intent === 'VOLUME') {
          let value = slots.operation === 'mute' ? 0 : slots.percent;
          if (slots.operation === 'up' || slots.operation === 'down') {
            const current = await this.providers.volume?.();
            if (current === undefined) return missing('current Spotify volume');
            value = Math.max(0, Math.min(100, current + (slots.operation === 'up' ? 10 : -10)));
          }
          action = { action: 'volume', value };
        }
        await this.providers.spotify(spotifyCommand(action));
      }
      return accepted('Spotify');
    }
    if (intent === 'TIME') {
      const timezone=this.config.timezone || 'Asia/Manila';
      return outcome('local', new Intl.DateTimeFormat('en-PH', { timeStyle: 'medium', timeZone: timezone }).format(this.now()),
        { command, time: new Date(this.now()).toISOString(), timezone });
    }
    if (intent === 'LIST_REMINDERS') return outcome('local', 'Saved reminders loaded.', { command, jobs: this.store.snapshot().jobs.filter(job => job.kind === 'reminder') });
    if (intent === 'TIMER' || intent === 'ALARM' || intent === 'CREATE_REMINDER') {
      let deadline: number | null = null, label = intent === 'TIMER' ? 'Timer' : 'Alarm';
      if (intent === 'TIMER') deadline = this.now() + Number(slots.duration_seconds) * 1000;
      if (intent === 'ALARM') {
        const day = Math.floor((this.now() + 8 * 3600000) / 86400000) * 86400000 - 8 * 3600000;
        const [hours, minutes] = String(slots.time).split(':').map(Number);
        deadline = day + (hours! * 60 + minutes!) * 60000;
        if (deadline <= this.now()) deadline += 86400000;
      }
      if (intent === 'CREATE_REMINDER') {
        const task = taskLabels[String(slots.task_id)];
        if (!task) return missing('a supported finite reminder task');
        label = task;
      }
      const job = await this.store.addJob({ id: randomUUID(), kind: intent === 'TIMER' ? 'timer' : intent === 'ALARM' ? 'alarm' : 'reminder', label, deadline,
        ...(intent==='ALARM'?{timezone:String(slots.timezone)}:{}) });
      return outcome('local', intent === 'CREATE_REMINDER' ? 'Reminder saved.' : `${label} saved on the server.`, { command, job, deadline });
    }
    return outcome('unsupported', 'No action implementation for ' + intent + '.');
  }
}
