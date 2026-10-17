import { test } from 'node:test';
import assert from 'node:assert/strict';
import { workerModule } from './python-runtime.js';

test('default runtime module remains the original resident worker', () => {
  assert.equal(workerModule({}), 'scripts.runtime.speaker_worker');
});
test('submission module is an explicit validated worker override', () => {
  assert.equal(workerModule({ SMART_SPEAKER_WORKER_MODULE:'scripts.submission.worker' }), 'scripts.submission.worker');
  for (const name of ['os', '-c', 'scripts.arbitrary', 'scripts.submission.worker;echo secret']) {
    assert.throws(() => workerModule({ SMART_SPEAKER_WORKER_MODULE:name }), /worker module/);
  }
});
