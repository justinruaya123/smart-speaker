import assert from 'node:assert/strict';
import { test } from 'node:test';
import { requireSuccess } from './http.js';

test('provider errors release unread bodies before reporting failure', async () => {
  let cancelled = false;
  const body = new ReadableStream({ cancel() { cancelled = true; } });
  const response = new Response(body, { status: 401 });
  try { await requireSuccess(response); assert.fail('Expected authentication failure'); }
  catch (error) { assert.match(String(error), /authentication/); }
  assert.equal(cancelled, true);
});
