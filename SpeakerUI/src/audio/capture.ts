import { AUDIO, isRecord } from '../../shared/protocol.js';

export interface Capture { flush(): Promise<void>; close(): void }
interface Callbacks { frame(buffer: ArrayBuffer): void; failure(message: string): void }

export async function startCapture(signal: AbortSignal, callbacks: Callbacks): Promise<Capture> {
  if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) throw new Error('Microphone access needs HTTPS or a loopback URL.');
  let stream: MediaStream | undefined;
  let context: AudioContext | undefined;
  let source: MediaStreamAudioSourceNode | undefined;
  let worklet: AudioWorkletNode | undefined;
  let closed = false;
  let completeFlush: (() => void) | undefined;
  let rejectFlush: ((error: Error) => void) | undefined;
  let flushTimer: ReturnType<typeof setTimeout> | undefined;
  function close() {
    closed = true;
    clearTimeout(flushTimer);
    rejectFlush?.(new Error('Capture closed.')); rejectFlush = undefined;
    signal.removeEventListener('abort', close);
    if (worklet) { worklet.port.onmessage = null; worklet.port.close(); worklet.disconnect(); }
    source?.disconnect();
    stream?.getTracks().forEach((track) => track.stop());
    if (context && context.state !== 'closed') void context.close().catch(() => {});
  }
  function checkActive() { if (closed || signal.aborted) throw new Error('Capture cancelled.'); }
  signal.addEventListener('abort', close, { once: true });
  try {
    checkActive();
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false });
    checkActive();
    context = new AudioContext({ sampleRate: AUDIO.sampleRate, latencyHint: 'interactive' });
    if (context.sampleRate !== AUDIO.sampleRate) throw new Error('This browser did not provide a 16 kHz audio context.');
    await context.audioWorklet.addModule('/audio-worklet.js'); checkActive();
    worklet = new AudioWorkletNode(context, 'pcm-capture', { channelCount: 1, channelCountMode: 'explicit', numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], processorOptions: { frameSamples: AUDIO.frameSamples, maximumFrames: AUDIO.maxQueuedBytes / AUDIO.frameBytes } });
    const node = worklet;
    node.port.onmessage = ({ data }: MessageEvent<unknown>) => {
      if (closed || !isRecord(data)) return;
      if (data.type === 'frame' && data.buffer instanceof ArrayBuffer) {
        try { callbacks.frame(data.buffer); }
        catch (error) { callbacks.failure(error instanceof Error ? error.message : 'Audio transport failed.'); }
        node.port.postMessage('ack');
      }
      if (data.type === 'overflow') callbacks.failure('Audio processing fell behind; the utterance was cancelled.');
      if (data.type === 'flushed') { clearTimeout(flushTimer); rejectFlush = undefined; completeFlush?.(); completeFlush = undefined; }
    };
    node.onprocessorerror = () => callbacks.failure('Audio processor failed.');
    for (const track of stream.getAudioTracks()) track.addEventListener('ended', () => { if (!closed) callbacks.failure('Microphone disconnected.'); });
    source = context.createMediaStreamSource(stream);
    source.connect(node); node.connect(context.destination);
    await context.resume(); checkActive();
    return {
      close,
      flush() {
        checkActive();
        return new Promise<void>((resolve, reject) => {
          completeFlush = resolve; rejectFlush = reject;
          flushTimer = setTimeout(() => { rejectFlush = undefined; reject(new Error('Audio flush timed out.')); }, 1_000);
          node.port.postMessage('flush');
        });
      },
    };
  } catch (error) { close(); throw error; }
}
