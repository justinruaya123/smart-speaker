import { PcmFramer } from './pcm-framer.js';

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.framer = new PcmFramer(options.processorOptions.frameSamples);
    this.maximumFrames = options.processorOptions.maximumFrames;
    this.pending = 0;
    this.active = true;
    this.port.onmessage = ({ data }) => {
      if (data === 'ack') this.pending = Math.max(0, this.pending - 1);
      if (data === 'flush') {
        this.active = false;
        const tail = this.framer.flush();
        if (tail) this.sendFrame(tail);
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }
  sendFrame(buffer) {
    if (this.pending >= this.maximumFrames) {
      this.active = false;
      this.port.postMessage({ type: 'overflow' });
      return;
    }
    this.pending++;
    this.port.postMessage({ type: 'frame', buffer }, [buffer]);
  }
  process(inputs) {
    if (!this.active) return true;
    const samples = inputs[0]?.[0];
    if (samples) for (const frame of this.framer.push(samples)) this.sendFrame(frame);
    // No microphone samples are copied to the output: no live monitoring/feedback.
    return true;
  }
}
registerProcessor('pcm-capture', CaptureProcessor);
