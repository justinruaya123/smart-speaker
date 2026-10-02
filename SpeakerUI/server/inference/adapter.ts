import { AUDIO, type InferenceResult } from '../../shared/protocol.js';

/** A future service adapter owns its protocol, bounded buffering, and abort lifecycle. */
export interface InferenceAdapter {
  readonly simulated: boolean;
  start(utteranceId: string): void | Promise<void>;
  writeAudio(frame: Uint8Array): void | Promise<void>;
  end(): void | Promise<void>;
  cancel(): void;
}
export type InferenceFactory = (emit: (result: unknown) => void) => InferenceAdapter;

export function createMockInference(emit: (result: InferenceResult) => void): InferenceAdapter {
  let id: string | undefined;
  let bytes = 0;
  let partialSent = false;
  function result(final: boolean): InferenceResult {
    if (!id) throw new Error('No active mock utterance.');
    // Intentionally fixed: this does NOT recognize speech, including silence.
    return { utteranceId: id, revision: final ? 2 : 1, final, intent: 'media.search', parameters: { query: 'demo' }, confidence: 0.99 };
  }
  return {
    simulated: true,
    start(utteranceId) { id = utteranceId; bytes = 0; partialSent = false; },
    writeAudio(frame) {
      bytes += frame.byteLength;
      if (!partialSent && bytes >= AUDIO.sampleRate) { partialSent = true; emit(result(false)); }
    },
    end() { if (bytes > 0 && id) emit(result(true)); id = undefined; },
    cancel() { id = undefined; bytes = 0; },
  };
}
