// End-to-end test with a local HA fixture and real multipart HTTP streams. No private cameras.
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
import { WebSocketServer } from 'ws';
import { createApp } from '../../dist-server/server/app.js';
import { readConfig } from '../../dist-server/server/config.js';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const out = process.env.TEST_OUTPUT_DIR || 'docs/testing/camera';
await mkdir(out, { recursive: true });
const port = process.env.TEST_FRONTEND_PORT || '4183', origin = `http://127.0.0.1:${port}`;
const camera = { entity_id: 'camera.front', state: 'idle', last_updated: new Date().toISOString(), attributes: { friendly_name: 'Main Camera TC70 mainStream' } };
const active = new Set(), checks = [], errors = [];
let frames = [], opened = 0, peak = 0, invalid = false, app, frontend, browser;
const ha = http.createServer((req, res) => {
  if (req.headers.authorization !== 'Bearer test-only') { res.writeHead(401); res.end(); return; }
  if (req.url === '/api/states/camera.front') { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(camera)); return; }
  if (req.url !== '/api/camera_proxy_stream/camera.front') { res.writeHead(404); res.end(); return; }
  opened++;
  if (invalid) { res.setHeader('Content-Type', 'text/html'); res.end('Private upstream detail'); return; }
  active.add(res); peak = Math.max(peak, active.size);
  res.writeHead(200, { 'Content-Type': 'multipart/x-mixed-replace; boundary=qa-frame' });
  let index = 0;
  const write = () => {
    const frame = frames[index++ % frames.length];
    res.write(`--qa-frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${frame.length}\r\n\r\n`);
    res.write(frame); res.write('\r\n');
  };
  write(); const timer = setInterval(write, 200);
  res.on('close', () => { clearInterval(timer); active.delete(res); });
});
const sockets = new WebSocketServer({ server: ha });
sockets.on('connection', ws => {
  ws.send(JSON.stringify({ type: 'auth_required' }));
  ws.on('message', bytes => {
    const message = JSON.parse(bytes.toString());
    ws.send(JSON.stringify(message.type === 'auth' ? { type: 'auth_ok' } : { type: 'result', id: message.id, success: true }));
  });
});
function state(value) {
  camera.state = value; camera.last_updated = new Date().toISOString();
  for (const ws of sockets.clients) ws.send(JSON.stringify({ type: 'event', event: { data: { new_state: camera } } }));
}
function ok(name, condition) { assert.ok(condition, name); checks.push(name); console.log('PASS', name); }
async function until(predicate) { for (let i = 0; i < 100; i++) { if (await predicate()) return; await delay(50); } assert.fail('Timed out waiting for stream lifecycle'); }
try {
  browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}), args: ['--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 800, height: 480 } });
  page.setDefaultTimeout(7000); page.on('pageerror', error => errors.push(error.message));
  frames = (await page.evaluate(() => ['#314754', '#4d6773'].map(color => {
    const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 360;
    const c = canvas.getContext('2d'); c.fillStyle = color; c.fillRect(0, 0, 640, 360);
    c.strokeStyle = '#a8c1c8'; c.strokeRect(90, 55, 460, 250);
    c.fillStyle = '#dcebef'; c.font = '24px sans-serif'; c.fillText('LOCAL CAMERA TEST', 184, 188);
    return canvas.toDataURL('image/jpeg').split(',')[1];
  }))).map(frame => Buffer.from(frame, 'base64'));
  await new Promise(resolve => ha.listen(0, '127.0.0.1', resolve));
  const cfg = readConfig({ APP_ORIGIN: origin, HA_ENABLED: 'true', HA_URL: `http://127.0.0.1:${ha.address().port}`, HA_TOKEN: 'test-only', HA_ENTITY_IDS: 'camera.front', ACTIONS_ENABLED: 'false' }, process.cwd());
  app = await createApp(cfg);
  const gateway = await app.listen({ port: 0, host: '127.0.0.1' });
  frontend = spawn(process.execPath, ['scripts/frontend.mjs'], { env: { ...process.env, FRONTEND_PORT: port, FRONTEND_HOST: '127.0.0.1', BACKEND_URL: gateway }, stdio: 'ignore' });
  await until(async () => { try { return (await fetch(origin)).ok; } catch { return false; } });

  const beforeInvalid = opened;
  ok('unselected cameras cannot open upstream streams', (await fetch(`${origin}/api/home-assistant/camera/camera.other/stream`)).status === 400 && opened === beforeInvalid);
  const leases = [new AbortController(), new AbortController()];
  for (const lease of leases) {
    const response = await fetch(`${origin}/api/home-assistant/camera/camera.front/stream`, { signal: lease.signal });
    assert.equal(response.headers.get('content-type'), 'multipart/x-mixed-replace; boundary=qa-frame');
  }
  ok('gateway bounds concurrent camera streams', (await fetch(`${origin}/api/home-assistant/camera/camera.front/stream`)).status === 429);
  for (const lease of leases) lease.abort(); await until(() => active.size === 0);
  ok('disconnect releases upstream streams through the frontend proxy', active.size === 0);

  await page.goto(origin + '/?fullscreen=0');
  await page.locator('.demo-launch').click();
  await page.waitForFunction(() => document.querySelector('[data-demo="9"] .shortcut-live'));
  await page.locator('[data-demo="9"]').click();
  const imageReady = () => page.waitForFunction(() => document.querySelector('.camera-stream')?.naturalWidth === 640);
  await imageReady(); await until(() => active.size === 1);
  ok('Camera call opens centered even with actions disabled', await page.locator('.call-centered').isVisible() && await page.locator('.call-caption strong').textContent() === camera.attributes.friendly_name);
  ok('camera view makes its video-only capability explicit', await page.locator('.call-window').textContent().then(text => text.includes('Video only')) && await page.locator('[data-action="call-mute"]').count() === 0 && await page.locator('[data-action="call-speaker"]').count() === 0);
  await page.screenshot({ path: out + '/camera-centered-800.png' });
  const beforeExpand = opened;
  await page.locator('[data-action="camera-collapse"]').click(); await imageReady();
  ok('ongoing call keeps a floating preview after minimizing', await page.locator('.call-floating').isVisible() && active.size === 1);
  await page.screenshot({ path: out + '/camera-floating-800.png' });
  await page.locator('.control-trigger').click();
  await page.locator('.control-panel [data-open="music"]').click();
  await page.locator('.play-control').waitFor();
  await page.locator('.task-view').evaluate(async panel => { await Promise.all(panel.getAnimations().map(animation => animation.finished)); });
  for (const [width, height] of [[800, 480], [1024, 600]]) {
    await page.setViewportSize({ width, height });
    ok(`floating camera leaves music controls accessible at ${width} by ${height}`, await page.locator('.call-floating').evaluate(preview => {
      const a = preview.getBoundingClientRect();
      const controls = [...document.querySelectorAll('.task-view button, .task-view input')].filter(control => control.getClientRects().length);
      return controls.length > 0 && controls.every(control => {
        const b = control.getBoundingClientRect();
        return a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom;
      });
    }));
    await page.screenshot({ path: out + `/camera-floating-music-${width}.png` });
  }
  await page.setViewportSize({ width: 800, height: 480 });
  await page.locator('[data-action="resume-call"]').click(); await imageReady();
  ok('restoring the centered call reuses its stream', opened === beforeExpand && active.size === 1);
  await page.screenshot({ path: out + '/camera-expanded-800.png' });
  const box = await page.locator('.call-centered').boundingBox();
  ok('centered camera fits 800 by 480', box.x >= 0 && box.y >= 0 && box.x + box.width <= 800 && box.y + box.height <= 480 && Math.abs(box.x + box.width / 2 - 400) < 1);
  await page.setViewportSize({ width: 1024, height: 600 }); await page.screenshot({ path: out + '/camera-expanded-1024.png' });
  await page.setViewportSize({ width: 800, height: 480 });
  await page.locator('[data-action="call-camera"]').click(); await until(() => active.size === 0);
  ok('camera off stops the stream', await page.locator('.camera-stream').count() === 0);
  await page.locator('[data-action="call-camera"]').click(); await imageReady();
  await page.evaluate(() => { Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' }); document.dispatchEvent(new Event('visibilitychange')); });
  await until(() => active.size === 0);
  ok('hidden view releases its stream', await page.locator('.camera-stream').count() === 0);
  await page.evaluate(() => { delete document.visibilityState; document.dispatchEvent(new Event('visibilitychange')); }); await imageReady();
  await page.locator('.task-view>.close-task').click();
  ok('leaving the call modal keeps the ongoing video floating', await page.locator('.call-floating').isVisible() && active.size === 1);
  await page.locator('[data-action="resume-call"]').click(); await imageReady();
  state('unavailable'); await page.waitForFunction(() => document.querySelector('.camera-status')?.textContent.includes('unavailable')); await until(() => active.size === 0);
  ok('HA unavailable state stops video without showing a demo', await page.locator('.camera-canvas').count() === 0);
  state('idle'); await imageReady();
  await page.locator('[data-action="call-camera"]').click(); await until(() => active.size === 0); invalid = true;
  await page.locator('[data-action="call-camera"]').click(); await page.getByRole('button', { name: 'Retry stream' }).waitFor();
  ok('invalid upstream data shows a recoverable error without provider details', !(await page.locator('.call-window').textContent()).includes('Private upstream'));
  await page.screenshot({ path: out + '/camera-error-800.png' });
  invalid = false; await page.getByRole('button', { name: 'Retry stream' }).click(); await imageReady();
  ok('manual retry reconnects the feed', active.size === 1);

  const cdp = await page.context().newCDPSession(page); await cdp.send('Performance.enable');
  await cdp.send('HeapProfiler.collectGarbage');
  const initial = await cdp.send('Runtime.getHeapUsage');
  const before = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
  await delay(5500); // Longer than ordinary API request timeout.
  const after = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
  ok('video continues beyond the ordinary request timeout', active.size === 1 && await page.locator('.camera-stream').evaluate(img => img.naturalWidth === 640));
  for (const response of active) response.destroy();
  await page.getByRole('button', { name: 'Retry stream' }).waitFor();
  ok('an interrupted stream stops showing its last frame as live video', await page.locator('.camera-stream').count() === 0);
  await page.getByRole('button', { name: 'Retry stream' }).click(); await imageReady();
  const beforeCycles = opened;
  for (let i = 0; i < 15; i++) {
    await page.locator('.task-view>.close-task').click();
    await page.locator('[data-action="resume-call"]').click(); await imageReady();
  }
  ok('15 minimize and restore cycles never reconnect the camera', opened === beforeCycles);
  await page.locator('[data-action="hangup"]').click(); await until(() => active.size === 0);
  await cdp.send('HeapProfiler.collectGarbage'); const final = await cdp.send('Runtime.getHeapUsage');
  ok('repeated reopen and hangup leave no streams', active.size === 0);
  await page.evaluate(() => document.querySelector('main').dispatchEvent(new CustomEvent('vcm-intent', { detail: { intent: 'call.start', parameters: { contact: 'Mom' } } })));
  await page.locator('.call-centered .camera-canvas').waitFor();
  ok('simulated calls also use a centered camera view', await page.locator('.call-caption').textContent().then(text => text.includes('Mom') && text.includes('Simulated call')));
  await page.locator('[data-action="camera-collapse"]').click();
  ok('ongoing simulated call has a floating preview', await page.locator('.call-floating .camera-canvas').isVisible());
  await page.locator('[data-action="hangup"]').click();
  ok('hangup removes the floating window', await page.locator('.call-window').count() === 0);
  ok('camera UI has no browser errors', errors.length === 0);
  await writeFile(out + '/checks.json', JSON.stringify({ checks, errors, resources: { environment: 'Desktop Chromium / SwiftShader; 640x360 JPEG fixture at 5 fps, not Pi or real-camera performance', initialHeapMB: initial.usedSize / 1e6, finalHeapMB: final.usedSize / 1e6, heapDeltaMB: (final.usedSize - initial.usedSize) / 1e6, rendererBusyPercent: (after.TaskDuration - before.TaskDuration) / (after.Timestamp - before.Timestamp) * 100, reopenCycles: 15, peakStreamsIncludingLimitTest: peak, remainingStreams: active.size } }, null, 2));
  console.log('FINISHED', checks.length);
} catch (error) {
  console.error(error);
  process.exitCode = 1;
} finally {
  await browser?.close();
  await app?.close();
  if (frontend && frontend.exitCode === null) {
    const exited = new Promise(resolve => frontend.once('exit', resolve));
    frontend.kill('SIGTERM');
    const deadline = setTimeout(() => frontend.kill('SIGKILL'), 3000);
    await exited; clearTimeout(deadline);
  }
  for (const res of active) res.destroy();
  for (const ws of sockets.clients) ws.terminate();
  sockets.close(); await new Promise(resolve => ha.close(resolve));
}
