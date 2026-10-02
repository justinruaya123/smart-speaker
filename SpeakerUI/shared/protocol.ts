export const AUDIO = { sampleRate: 16_000, channels: 1, frameSamples: 320, frameBytes: 640, maxQueuedBytes: 32_000, maxDurationMs: 15_000, maxBytes: 480_000 } as const;

export const enum Intent {
  LightOn = 'light.on', LightOff = 'light.off', CameraSnapshot = 'camera.snapshot',
  MediaSearch = 'media.search', MediaPlay = 'media.play', MediaPause = 'media.pause',
}
export const enum MediaSource { Local = 'local', Spotify = 'spotify' }

export interface InferenceResult {
  utteranceId: string;
  revision: number;
  final: boolean;
  intent: string;
  parameters: Record<string, unknown>;
  /** Model-documented normalized score, NOT a calibrated probability by default. */
  confidence: number;
}
export type Command =
  | { intent: Intent.LightOn | Intent.LightOff | Intent.CameraSnapshot; entityId: string }
  | { intent: Intent.MediaSearch; query: string }
  | { intent: Intent.MediaPlay; source: MediaSource; id: string }
  | { intent: Intent.MediaPause; source: MediaSource };

export interface Track { source: MediaSource; id: string; title: string; artist?: string }
export interface SearchResults { tracks: Track[]; fallbackReason?: string }
export type ClientControl = { type: 'start' | 'end' | 'cancel'; utteranceId: string };
export type ServerEvent =
  | { type: 'ready'; sessionId: string; mode: 'mock' }
  | { type: 'state'; utteranceId: string; state: 'recording' | 'processing' | 'idle' }
  | { type: 'inference'; simulated: boolean; result: InferenceResult }
  | { type: 'decision'; utteranceId: string; status: 'needs_input' | 'rejected'; reason: string }
  | { type: 'action'; utteranceId: string; status: 'simulated' | 'disabled' | 'succeeded' | 'failed' | 'unknown'; detail: string; data?: unknown }
  | { type: 'error'; code: string; message: string; utteranceId?: string };

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function parseControl(value: unknown): ClientControl | undefined {
  if (!isRecord(value) || !['start', 'end', 'cancel'].includes(String(value.type))) return;
  if (typeof value.utteranceId !== 'string' || !/^[\w-]{1,64}$/.test(value.utteranceId)) return;
  if (value.type === 'start' || value.type === 'end' || value.type === 'cancel') return { type: value.type, utteranceId: value.utteranceId };
}

export function parseInference(value: unknown): InferenceResult | undefined {
  if (!isRecord(value) || typeof value.utteranceId !== 'string' || value.utteranceId.length > 64) return;
  if (typeof value.revision !== 'number' || !Number.isSafeInteger(value.revision) || value.revision < 0) return;
  if (typeof value.final !== 'boolean' || typeof value.intent !== 'string' || value.intent.length > 64 || !isRecord(value.parameters)) return;
  if (typeof value.confidence !== 'number' || !Number.isFinite(value.confidence) || value.confidence < 0 || value.confidence > 1) return;
  return { utteranceId: value.utteranceId, revision: value.revision, final: value.final, intent: value.intent, parameters: value.parameters, confidence: value.confidence };
}
