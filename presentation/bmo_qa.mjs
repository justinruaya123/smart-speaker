/* CI browser regression for the static BMO deck, using presentation's pinned Playwright.
 * All viewer requests must resolve below a GitHub-style /smart-speaker/ prefix.
 * No inference, microphone, training, synthesis or external provider calls.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const site = resolve(root, 'docs/site');
const require = createRequire(resolve(process.env.IRIS_PLAYWRIGHT_ROOT || resolve(root, 'presentation'), 'package.json'));
const { chromium } = require('playwright');
const prefix = '/smart-speaker/';
const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.wav':'audio/wav','.csv':'text/csv'};
const server = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (!path.startsWith(prefix)) { res.writeHead(404).end(); return; }
    const file = resolve(site, path.slice(prefix.length) || 'index.html');
    if (!file.startsWith(site + sep)) { res.writeHead(403).end(); return; }
    res.writeHead(200, {'Content-Type':mime[extname(file)] || 'application/octet-stream'}).end(await readFile(file));
  } catch { res.writeHead(404).end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;
const url = origin + prefix;
const output = resolve(root, '.cache/bmo-pages-qa');
await mkdir(output, {recursive:true});
const browser = await chromium.launch({headless:true, args:['--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
const errors = [], external = [];
const report = {prefix, browser:browser.version(), slides:[], mobile:[], controls:[], errors, external};
try {
  const context = await browser.newContext({viewport:{width:1600,height:900}});
  await context.route('**/*', route => {
    if (route.request().url().startsWith(origin + prefix)) return route.continue();
    external.push(route.request().url()); return route.abort();
  });
  const page = await context.newPage();
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', e => { if (e.type() === 'error') errors.push(e.text()); });
  page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
  await page.goto(url);
  await page.waitForFunction(() => document.querySelector('#epoch-loss')?.textContent === '0.0554');
  await page.locator('#model-scene canvas').waitFor();
  assert.equal(await page.title(), 'BMO — Architecture');
  assert.equal(await page.locator('[data-slide]').count(), 4);
  assert.equal(await page.locator('img').count(), 0);
  const recordedTrace = JSON.parse(await readFile(resolve(site, 'bmo/data/trace.json'), 'utf8'));
  const logmel = recordedTrace.stages.find(item => item.id === 'logmel').thumbnail;
  await page.waitForSelector('#logmel-plot[data-source="recorded-logmel-bin-means"]');
  const display = await page.locator('#logmel-plot rect').evaluateAll(nodes => nodes.map(node => ({value:Number(node.dataset.value),row:Number(node.dataset.melRow),column:Number(node.dataset.timeColumn),y:Number(node.getAttribute('y'))})));
  assert.equal(display.length, logmel.rows * logmel.cols);
  for (const cell of display) assert.equal(cell.value, logmel.values[cell.row][cell.column], 'heatmap must preserve each measured display value');
  assert.ok(display[0].y > display.at(-1).y, 'high mel bands must be above low bands');
  assert.match(await page.locator('.model-choice').textContent(), /132\/167.*130\/167/);
  assert.equal(await page.locator('#evidence-title').textContent(), 'Measure complete commands.');

  for (const id of ['architecture','dataset','training','evidence']) {
    await page.getByRole('navigation', {name:'Slides'}).getByRole('link', {name:id, exact:false}).click();
    assert.equal(await page.locator(`[data-slide="${id}"]`).isVisible(), true);
    await page.screenshot({path:resolve(output, id + '.png')}); report.slides.push(id);
  }
  await page.getByRole('navigation', {name:'Slides'}).getByRole('link', {name:'Architecture',exact:true}).click();
  await page.locator('#expand-block').click();
  await page.getByRole('button', {name:'Play this block',exact:true}).click();
  await page.waitForFunction(() => document.querySelector('#play-model')?.getAttribute('aria-pressed') === 'false');
  assert.equal(await page.locator('#expand-block').getAttribute('aria-pressed'), 'true');
  await page.getByRole('button', {name:'06 Block 4',exact:true}).click();
  await page.getByRole('button', {name:'Next block',exact:true}).click();
  assert.equal(await page.getByRole('button', {name:'03 Block 1',exact:true}).getAttribute('aria-pressed'), 'true');
  await page.getByRole('button', {name:'Whole model',exact:true}).click();
  await page.getByRole('button', {name:'08 Heads',exact:true}).click();
  const architectureFit = await page.evaluate(() => {
    const steps = document.querySelector('.model-steps').getBoundingClientRect().top;
    const footer = document.querySelector('.deck-footer').getBoundingClientRect().top;
    return {decision:document.querySelector('#decision').getBoundingClientRect().bottom <= steps,
      explanation:document.querySelector('.architecture > .footnote').getBoundingClientRect().bottom <= footer};
  });
  assert.deepEqual(architectureFit, {decision:true,explanation:true}, 'model output and concise explanation must fit before controls');

  assert.match(await page.locator('#decision').textContent(), /BRIGHTNESS\s*50%/);
  await page.getByRole('button', {name:'Hear the sample',exact:true}).click();
  await page.waitForFunction(() => document.querySelector('#sample-audio').duration > 2.9);
  assert.equal(await page.locator('#sample-audio').evaluate(a => a.duration), 2.976);
  await page.waitForFunction(() => document.querySelector('.logmel-cursor')?.getAttribute('visibility') === 'visible');
  assert.ok(Number(await page.locator('.logmel-cursor').getAttribute('x1')) > 29, 'the spectrogram cursor must follow the actual audio');

  report.controls.push('measured log-mel values and audio cursor, recorded 3D playback, residual expansion and wrap, heads, audio');
  await page.getByRole('navigation', {name:'Slides'}).getByRole('link', {name:'Dataset',exact:true}).click();
  await page.getByRole('button', {name:'Development',exact:true}).click();
  assert.equal(await page.locator('#corpus-count').textContent(), '2,132');
  await page.getByRole('button', {name:'Synthetic test',exact:true}).click();
  assert.equal(await page.locator('#corpus-count').textContent(), '440');
  await page.locator('#all-wordings').click();
  await page.locator('#wording-search').fill('brightness');
  assert.equal(await page.locator('#wording-count').textContent(), '9 of 102 wordings');
  await page.keyboard.press('Escape');
  for (const id of ['generation','citations']) {
    await page.locator(`[data-source="${id}"]`).click();
    await page.waitForFunction(() => document.querySelector('#source-content').textContent.length > 250);
    assert.ok(await page.locator('#source-content a[href^="https://"]').count() > 1);
    await page.getByRole('button', {name:'Close details',exact:true}).click();
  }
  report.controls.push('partitions, exact wording search, generation recipe and citations');
  await page.getByRole('navigation', {name:'Slides'}).getByRole('link', {name:'Training',exact:true}).click();
  await page.locator('#epoch-slider').focus(); await page.keyboard.press('Home');
  assert.equal(await page.locator('#epoch-value').textContent(), '1');
  assert.equal(await page.locator('#epoch-phase').textContent(), 'Human-only foundation');
  await page.keyboard.press('End');
  assert.equal(await page.locator('#epoch-value').textContent(), '48');
  await page.locator('#animate-update').click();
  await page.waitForFunction(() => !document.querySelector('#animate-update').disabled);
  report.controls.push('48-epoch scrubber, phase quotas, one-update illustration');
  await page.getByRole('navigation', {name:'Slides'}).getByRole('link', {name:'Evidence',exact:true}).click();
  await page.locator('#result-mode').click();
  assert.match(await page.locator('#result-bars').textContent(), /106/);
  await page.locator('[data-source="benchmark"]').click();
  await page.waitForFunction(() => document.querySelector('#source-content').textContent.includes('71'));
  assert.match(await page.locator('#source-content').textContent(), /Pi|Raspberry/);
  await page.getByRole('button', {name:'Close details',exact:true}).click();
  for (const viewport of [{width:1280,height:720},{width:390,height:844}]) {
    await page.setViewportSize(viewport);
    for (const name of ['Architecture','Dataset','Training','Evidence']) {
      await page.getByRole('navigation', {name:'Slides'}).getByRole('link', {name,exact:true}).click();
      const dimensions = await page.evaluate(() => ({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));
      assert.ok(dimensions.scroll <= dimensions.width + 1, `${name} horizontal overflow`);
    }
    report.mobile.push(viewport);
  }
  const reduced = await browser.newContext({reducedMotion:'reduce',viewport:{width:1280,height:720}});
  const still = await reduced.newPage(); await still.goto(url + '#architecture');
  assert.equal(await still.getByRole('button', {name:'Play forward pass',exact:true}).getAttribute('aria-pressed'), 'false');
  assert.deepEqual(errors, []); assert.deepEqual(external, []);
  await import('node:fs/promises').then(fs => fs.writeFile(resolve(output, 'report.json'), JSON.stringify(report,null,2)+'\n'));
  console.log(JSON.stringify(report,null,2));
} finally { await browser.close(); await new Promise(r => server.close(r)); }
