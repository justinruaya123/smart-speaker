/** Bounded PCM16 LE conversion shared by the AudioWorklet and offline tests. */
export class PcmFramer {
  constructor(frameSamples) {
    this.frameSamples = frameSamples;
    this.buffer = new ArrayBuffer(frameSamples * 2);
    this.view = new DataView(this.buffer);
    this.offset = 0;
  }
  push(samples) {
    const frames = [];
    for (const value of samples) {
      const sample = Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
      this.view.setInt16(this.offset * 2, Math.round(sample < 0 ? sample * 32768 : sample * 32767), true);
      this.offset++;
      if (this.offset === this.frameSamples) {
        frames.push(this.buffer);
        this.buffer = new ArrayBuffer(this.frameSamples * 2);
        this.view = new DataView(this.buffer);
        this.offset = 0;
      }
    }
    return frames;
  }
  flush() {
    if (this.offset === 0) return;
    const tail = this.buffer.slice(0, this.offset * 2);
    this.offset = 0;
    return tail;
  }
}
