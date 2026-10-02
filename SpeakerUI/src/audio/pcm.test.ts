import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PcmFramer } from '../../public/pcm-framer.js';

test('capture emits little-endian PCM16 with saturation and a bounded tail', () => {
  const framer = new PcmFramer(4);
  assert.equal(framer.push(new Float32Array([-2, -0.5])).length, 0);
  const frames = framer.push(new Float32Array([0.5, 2, 0.25]));
  assert.equal(frames.length, 1);
  const view = new DataView(frames[0]!);
  assert.deepEqual([0,2,4,6].map((offset) => view.getInt16(offset, true)), [-32768, -16384, 16384, 32767]);
  assert.equal(framer.flush()?.byteLength, 2);
  assert.equal(framer.flush(), undefined);
});
