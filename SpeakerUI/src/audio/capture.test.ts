import assert from 'node:assert/strict';
import { test } from 'node:test';
import { startCapture } from './capture.js';

function browserContext(getUserMedia: () => Promise<MediaStream>, secure = true) {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'window', { configurable: true, value: { isSecureContext: secure } });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices: { getUserMedia } } });
  return () => {
    if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow); else Reflect.deleteProperty(globalThis, 'window');
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator); else Reflect.deleteProperty(globalThis, 'navigator');
  };
}

test('permission denial and insecure contexts fail without starting audio transport', async () => {
  let frames = 0;
  let requested = 0;
  const restore = browserContext(async () => { requested++; throw new DOMException('Permission denied', 'NotAllowedError'); });
  try {
    await assert.rejects(startCapture(new AbortController().signal, { frame: () => { frames++; }, failure() {} }), /Permission denied/);
    assert.equal(requested, 1); assert.equal(frames, 0);
  } finally { restore(); }
  const restoreInsecure = browserContext(async () => { throw new Error('Must not request microphone'); }, false);
  try { await assert.rejects(startCapture(new AbortController().signal, { frame() {}, failure() {} }), /HTTPS/); }
  finally { restoreInsecure(); }
});

test('cancelling pending permission stops tracks if permission is later granted', async () => {
  let resolvePermission: (stream: MediaStream) => void = () => {};
  const permission = new Promise<MediaStream>((resolve) => { resolvePermission = resolve; });
  let stopped = 0;
  const restore = browserContext(() => permission);
  try {
    const abort = new AbortController();
    const starting = startCapture(abort.signal, { frame() { assert.fail('Cancelled capture emitted audio'); }, failure() {} });
    abort.abort();
    resolvePermission({ getTracks: () => [{ stop() { stopped++; } }] } as unknown as MediaStream);
    await assert.rejects(starting, /cancelled/);
    assert.equal(stopped, 1);
  } finally { restore(); }
});
