import assert from 'node:assert/strict';
import { test } from 'node:test';
import { evaluateIntent, ResultGate } from './policy.js';

const entities = ['light.desk', 'light.kitchen', 'camera.entry'];
const result = (overrides = {}) => ({ utteranceId: 'u1', revision: 1, final: true, intent: 'light.on', parameters: { entityId: 'light.desk' }, confidence: 0.96, ...overrides });

test('a final confident light command resolves only an allowlisted target', () => {
  assert.deepEqual(evaluateIntent(result(), 0.9, entities), { status: 'accepted', command: { intent: 'light.on', entityId: 'light.desk' } });
  assert.equal(evaluateIntent(result({ parameters: { entityId: 'light.other' } }), 0.9, entities).status, 'rejected');
});

test('partial, low, missing, nonfinite, out-of-range scores and unknown labels cannot dispatch', () => {
  for (const change of [{ final: false }, { confidence: 0.8 }, { confidence: undefined }, { confidence: NaN }, { confidence: Infinity }, { confidence: 1.1 }, { intent: 'shell.exec' }]) {
    assert.notEqual(evaluateIntent(result(change), 0.9, entities).status, 'accepted');
  }
  assert.equal(evaluateIntent(result(), undefined, entities).status, 'rejected');
});

test('missing and ambiguous parameters ask for input; incompatible target domains reject', () => {
  assert.equal(evaluateIntent(result({ parameters: {} }), 0.9, entities).status, 'needs_input');
  assert.equal(evaluateIntent(result({ parameters: { entityId: ['light.desk', 'light.kitchen'] } }), 0.9, entities).status, 'needs_input');
  assert.equal(evaluateIntent(result({ parameters: { entityId: 'camera.entry' } }), 0.9, entities).status, 'rejected');
});

test('media uses owned IDs and bounded search queries, never external URLs', () => {
  assert.equal(evaluateIntent(result({ intent: 'media.search', parameters: { query: '  piano  ' } }), 0.9, entities).status, 'accepted');
  assert.equal(evaluateIntent(result({ intent: 'media.play', parameters: { source: 'local', id: '../../secret' } }), 0.9, entities).status, 'rejected');
});

test('result gate drops stale, duplicate, foreign, and cancelled updates', () => {
  const gate = new ResultGate();
  gate.start('u1');
  assert.equal(gate.accept(result({ final: false, revision: 2 })), true);
  assert.equal(gate.accept(result({ revision: 1 })), false);
  assert.equal(gate.accept(result({ utteranceId: 'old', revision: 3 })), false);
  assert.equal(gate.accept(result({ revision: 3 })), true);
  assert.equal(gate.accept(result({ revision: 4 })), false);
  gate.start('u2'); gate.cancel();
  assert.equal(gate.accept(result({ utteranceId: 'u2' })), false);
});
