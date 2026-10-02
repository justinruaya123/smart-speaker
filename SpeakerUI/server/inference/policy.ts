import { Intent, MediaSource, parseInference, type Command, type InferenceResult } from '../../shared/protocol.js';

type Decision = { status: 'accepted'; command: Command } | { status: 'needs_input' | 'rejected'; reason: string };

export function evaluateIntent(raw: unknown, threshold: number | undefined, entityIds: readonly string[]): Decision {
  const result = parseInference(raw);
  if (!result) return { status: 'rejected', reason: 'Invalid inference result or confidence score.' };
  if (!result.final) return { status: 'rejected', reason: 'Partial results are provisional.' };
  if (threshold === undefined || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) return { status: 'rejected', reason: 'A validated model threshold has not been configured.' };
  if (result.confidence < threshold) return { status: 'rejected', reason: 'Confidence is below the configured threshold.' };
  const parameters = result.parameters;
  switch (result.intent) {
    case Intent.LightOn:
    case Intent.LightOff:
    case Intent.CameraSnapshot: {
      if (Array.isArray(parameters.entityId)) return { status: 'needs_input', reason: 'Select one unambiguous entity.' };
      if (typeof parameters.entityId !== 'string' || parameters.entityId.length === 0) return { status: 'needs_input', reason: 'An entityId is required.' };
      const domain = result.intent === Intent.CameraSnapshot ? 'camera.' : 'light.';
      if (!parameters.entityId.startsWith(domain) || !entityIds.includes(parameters.entityId)) return { status: 'rejected', reason: 'Entity is not allowed for this action.' };
      return { status: 'accepted', command: { intent: result.intent, entityId: parameters.entityId } };
    }
    case Intent.MediaSearch:
      if (typeof parameters.query !== 'string' || !parameters.query.trim()) return { status: 'needs_input', reason: 'A search query is required.' };
      if (parameters.query.length > 200) return { status: 'rejected', reason: 'Search query is too long.' };
      return { status: 'accepted', command: { intent: Intent.MediaSearch, query: parameters.query.trim() } };
    case Intent.MediaPlay:
    case Intent.MediaPause: {
      if (parameters.source !== MediaSource.Local && parameters.source !== MediaSource.Spotify) return { status: 'needs_input', reason: 'Select local or Spotify media.' };
      if (result.intent === Intent.MediaPause) return { status: 'accepted', command: { intent: Intent.MediaPause, source: parameters.source } };
      if (typeof parameters.id !== 'string' || !parameters.id) return { status: 'needs_input', reason: 'Select a track from search results.' };
      const validId = parameters.source === MediaSource.Local ? /^[a-f0-9]{32}$/ : /^[A-Za-z0-9]{22}$/;
      if (!validId.test(parameters.id)) return { status: 'rejected', reason: 'Invalid track ID.' };
      return { status: 'accepted', command: { intent: Intent.MediaPlay, source: parameters.source, id: parameters.id } };
    }
    default:
      return { status: 'rejected', reason: 'Unsupported intent.' };
  }
}

/** One utterance owns all revisions; finality closes it before any async dispatch. */
export class ResultGate {
  private current: { id: string; revision: number; final: boolean } | undefined;
  start(id: string) { this.current = { id, revision: -1, final: false }; }
  cancel() { this.current = undefined; }
  accept(result: InferenceResult): boolean {
    const current = this.current;
    if (!current || current.id !== result.utteranceId || current.final || result.revision <= current.revision) return false;
    current.revision = result.revision;
    current.final = result.final;
    return true;
  }
}
