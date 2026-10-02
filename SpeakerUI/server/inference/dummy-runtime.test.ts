import test from 'node:test';
import assert from 'node:assert/strict';
import { loadDummyRuntime, parseFixtureRequest } from './dummy-runtime.js';

test('dummy runtime is explicitly loaded and never claims a native ONNX session', () => {
  const runtime = loadDummyRuntime();
  assert.equal(runtime.status().loaded, true);
  assert.equal(runtime.status().nativeSession, false);
  const result = runtime.infer(3);
  assert.equal(result.simulated, true);
  assert.equal(result.result.intent, 'timer.set');
  assert.deepEqual(result.result.parameters, {seconds: 300});
  assert.notEqual(runtime.infer(3).result.utteranceId, result.result.utteranceId);
  runtime.close();
  assert.throws(() => runtime.infer(3), /closed/);
});

test('fixture requests reject arbitrary intents, noninteger IDs and extra fields', () => {
  for (const value of [null, [], {}, {fixture:-1}, {fixture:15}, {fixture:1.2}, {fixture:'3'}, {fixture:0,intent:'light.on'}]) assert.equal(parseFixtureRequest(value), undefined);
  assert.equal(parseFixtureRequest({fixture:0}), 0);
  assert.equal(parseFixtureRequest({fixture:14}), 14);
});
