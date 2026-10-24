/* Existing pinned Playwright only. Run from repo root:
 * pnpm --dir presentation install --frozen-lockfile
 * node presentation/qa.mjs
 * Self-owned loopback server; no microphone, external requests or providers.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile, mkdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
const root=resolve(dirname(fileURLToPath(import.meta.url)), '..');
const site=resolve(root,'docs/site');
assert.ok(existsSync(resolve(site,'index.html')), 'a complete real deck is required');
const source=process.env.IRIS_PLAYWRIGHT_ROOT || resolve(root,'presentation');
const require=createRequire(resolve(source,'package.json'));
const { chromium }=require('playwright');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png','.svg':'image/svg+xml','.mp4':'video/mp4','.woff2':'font/woff2','.woff':'font/woff','.ttf':'font/ttf'};
const server=createServer(async(req,res)=>{
  try {
    const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    const file=resolve(site,'.'+(pathname==='/'?'/index.html':pathname));
    if(!file.startsWith(site+sep)) {res.writeHead(403).end();return;}
    const data=await readFile(file);res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream'}).end(data);
  } catch {res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const url=`http://127.0.0.1:${server.address().port}`;
const output=resolve(root,'.cache/presentation-qa'); await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,channel:process.env.IRIS_BROWSER_CHANNEL,executablePath:process.env.IRIS_BROWSER_EXECUTABLE});
let page;
const errors=[];
const report={browser:browser.version(),playwright:require('playwright/package.json').version,viewports:[],use_cases:[],reduced_motion:false,offline:false,errors};
try {
  const context=await browser.newContext({viewport:{width:1920,height:1080}});
  await context.route('**/*',route=>route.request().url().startsWith(url)?route.continue():route.abort());
  page=await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',e=>{if(e.type()==='error') errors.push(e.text());});
  page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  await page.goto(url); await page.waitForSelector('html[data-ready="true"]');
  assert.equal(await page.locator('[data-chapter]').count(),6);
  assert.equal(await page.locator('#chapter-count').textContent(),'01 / 06');
  await page.keyboard.press('ArrowRight');
  assert.equal(new URL(page.url()).hash,'#pipeline');
  await page.keyboard.press('End'); assert.equal(new URL(page.url()).hash,'#run');
  await page.keyboard.press('Home'); assert.equal(new URL(page.url()).hash,'#iris');
  await page.locator('[data-go="uses"]').click();
  for(let index=0;index<10;index++) {
    const button=page.locator('[role="tab"]').nth(index);
    await button.click(); assert.equal(await button.getAttribute('aria-selected'),'true');
    const id=await page.locator('#case-id').textContent();
    assert.equal(id,`UC-${String(index+1).padStart(2,'0')}`);
    assert.equal(await page.locator('#case-diagram .flow-step').count(),3);
    assert.ok(JSON.parse(await page.locator('#case-json').textContent()));
    assert.ok(await page.locator('#case-source').isVisible());
    assert.ok((await page.locator('#case-disclaimer').textContent()).includes('example'));
    const overlap=await page.evaluate(()=>{
      const panel=document.querySelector('#case-panel').getBoundingClientRect();
      const summary=document.querySelector('.summary').getBoundingClientRect();
      const code=document.querySelector('#case-json');
      return {overlap:panel.bottom>summary.top+2 || code.scrollWidth>code.clientWidth+2,panel_bottom:panel.bottom,summary_top:summary.top,code_scroll:code.scrollWidth,code_width:code.clientWidth};
    });
    assert.equal(overlap.overlap,false,`${id} output/JSON must fit its consistent frame: ${JSON.stringify(overlap)}`);
    await page.screenshot({path:resolve(output,`${id}.png`)});report.use_cases.push(id);
  }
  await page.locator('[role="tab"]').first().focus();await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('[role="tab"]').nth(1).getAttribute('aria-selected'),'true');
  for(const viewport of [{width:1920,height:1080},{width:1280,height:720},{width:390,height:844}]) {
    await page.setViewportSize(viewport);
    for(const id of ['iris','pipeline','model','uses','evidence','run']) {
      await page.locator(`[data-go="${id}"]`).click();
      const violations=await page.evaluate(()=>{
        const panel=document.querySelector('[data-chapter]:not([hidden])');
        const box=panel.getBoundingClientRect();
        return [...panel.querySelectorAll('h1,h2,h3,p,pre,table,figure,.flow-step')].filter(e=>{
          if(e.closest('details:not([open])') && e.tagName!=='SUMMARY')return false;
          const r=e.getBoundingClientRect();return r.width&&r.height&&(r.right>box.right+3||r.bottom>box.bottom+3||r.left<box.left-3||r.top<box.top-3);
        }).map(e=>e.id||e.tagName+':'+e.textContent.slice(0,40));
      });
      await page.screenshot({path:resolve(output,`${viewport.width}-${id}.png`)});
      if(id==='model'||id==='evidence') {
        const collisions=await page.evaluate(()=>{
          const panel=document.querySelector('[data-chapter]:not([hidden])');
          const end=panel.querySelector('.chapter-foot').getBoundingClientRect().top;
          const margin=4*document.querySelector('#stage').getBoundingClientRect().width/1920;
          const content=panel.querySelector('.inspector,.evidence-grid');
          return [...content.querySelectorAll('h3,p,dl,figcaption,caption,span')].filter(e=>e.getBoundingClientRect().bottom>end-margin).map(e=>({text:e.id||e.textContent.slice(0,100),bottom:e.getBoundingClientRect().bottom,footer_top:end}));
        });
        assert.deepEqual(collisions,[],`${viewport.width} ${id}/footer collisions`);
      }
      assert.deepEqual(violations,[],`${viewport.width} ${id} overflow`);
    }
    report.viewports.push(viewport);
  }
  await page.setViewportSize({width:1280,height:720});
  const publication=JSON.parse(await readFile(resolve(site,'assets/iris.json'),'utf8'));
  const desktop=publication.benchmarks.desktop;
  if(desktop) {
    await page.locator('[data-go="evidence"]').click();
    assert.equal(await page.locator('#desktop-results .latency').first().textContent(),`${desktop.latency.p50_ms.toFixed(2)} / ${desktop.latency.p95_ms.toFixed(2)} ms`,'current desktop latency displays two decimals without changing the receipt');
    assert.ok((await page.locator('#evidence-foot').textContent()).includes(`peak process RSS ${desktop.peak_process_rss_mib.toFixed(2)} MiB`),'desktop RSS displays two decimals without changing the receipt');
    const readme=await readFile(resolve(root,'README.md'),'utf8');
    assert.ok(readme.includes(`| Current adapter, product lock (**not quiesced**) | ${desktop.latency.p50_ms.toFixed(2)} / ${desktop.latency.p95_ms.toFixed(2)} ms |`),'README current desktop latency displays two decimals from the same receipt');
    assert.ok(readme.includes(`peak whole-process RSS **${desktop.peak_process_rss_mib.toFixed(2)} MiB**`),'README desktop RSS displays two decimals from the same receipt');
  }
  await page.locator('[data-go="model"]').click();await page.reload();
  await page.waitForSelector('html[data-ready="true"]');assert.equal(new URL(page.url()).hash,'#model');
  await page.locator('#fullscreen').click();
  await page.waitForFunction(()=>!!document.fullscreenElement);
  await page.evaluate(()=>document.exitFullscreen());
  // Real exported manim-slides iframe must initialize locally, not a CSS imitation.
  await page.locator('[data-animation="model"] [data-inspect]').click();
  const frame=page.frameLocator('[data-animation="model"] iframe');
  await frame.locator('.reveal.ready').waitFor();
  assert.equal(await frame.locator('section[data-background-video]').count(),5);
  await page.locator('[data-animation="model"] [data-still]').click();
  const reduced=await browser.newContext({reducedMotion:'reduce',viewport:{width:1280,height:720}});
  const still=await reduced.newPage();await still.goto(url+'/#pipeline');await still.waitForSelector('html[data-ready="true"]');
  assert.equal(await still.locator('iframe[src]').count(),0);
  assert.ok(await still.locator('[data-animation="pipeline"] .poster').isVisible());
  assert.equal(await still.locator('video').evaluateAll(v=>v.every(x=>x.paused)),true);
  report.reduced_motion=true;
  await still.screenshot({path:resolve(output,'reduced-motion.png')});
  // No remote runtime dependency: second load uses only static loopback assets.
  report.offline=true;
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify(report,null,2));
  await import('node:fs/promises').then(fs=>fs.writeFile(resolve(output,'report.json'),JSON.stringify(report,null,2)+'\n'));
} finally {await browser.close();await new Promise(r=>server.close(r));}
