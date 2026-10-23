import { RuntimeActions } from './runtime-actions.js';
import type { Execution } from './action-store.js';

type Cue = 'intent' | 'success' | 'failure';
interface HostFeedback {
  play: (cue: Cue, utteranceId: string) => Promise<unknown>;
  publish: (execution: Execution, sessionId?: string) => void | Promise<void>;
  audioError?: (message: string) => void;
}
/** Wraps the ONE durable dispatcher; browsers cannot request host cue playback.
 * The durable claim precedes intent playback. A failed cue is consumed, not retried. */
export class ActionFeedback {
  private phases = new Set<string>();
  private active = new Map<string, { session: string; utterance: string }>();
  constructor(private actions: RuntimeActions, private host: HostFeedback) {}
  private async cue(id: string, phase: 'intent' | 'result', cue: Cue, utterance: string) {
    const key = `${id}:${phase}`;
    if (this.phases.has(key)) return;
    this.phases.add(key);
    try { await this.host.play(cue, utterance); }
    catch { this.host.audioError?.('Host feedback playback failed. The action result is unchanged; no action or audio retry was attempted.'); }
  }
  async run(event: Record<string, any>) {
    const execution = await this.actions.handle(event, async () => {
      if (event.source !== 'microphone' || event.actions_enabled !== true || event.owner_result?.accepted !== true) return;
      const id = `${event.session_id}:${event.utterance_id}`;
      this.active.set(id, { session:event.session_id,utterance:event.utterance_id });
      await this.cue(id,'intent','intent',event.utterance_id);
    });
    if (execution) {
      await this.host.publish(execution,event.session_id);
      await this.complete(execution);
    }
    return execution;
  }
  async complete(execution: Execution) {
    if (['requested','pending'].includes(execution.status) || execution.replayed) return;
    const context = this.active.get(execution.id);
    if (!context) return; // Historical receipts, reconnects and touch actions are silent.
    this.active.delete(execution.id);
    const success = ['succeeded','observed','local','cached','synthetic'].includes(execution.status);
    await this.cue(execution.id,'result',success?'success':'failure',context.utterance);
  }
}
