const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{const b=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{}),args:['--enable-unsafe-swiftshader']});const p=await b.newPage({viewport:{width:800,height:480},reducedMotion:'reduce',timezoneId:'Asia/Manila'}),checks=[],errors=[],records={};p.on('pageerror',e=>errors.push(e.message));const ok=(s,v)=>{assert.ok(v,s);checks.push(s);console.log('PASS',s)};
const urls={prototype:process.env.PROTOTYPE_URL||'http://127.0.0.1:8768/v2/04-ambient.html',svelte:process.env.TEST_BASE_URL||'http://127.0.0.1:3000/?fullscreen=0'};const time=new Date('2026-10-01T00:24:00Z');await p.clock.setFixedTime(time);
for(const [version,url] of Object.entries(urls)){
 await p.goto(url);await p.waitForTimeout(250);
 records[version]={};
 async function shot(name,selectors){await p.waitForTimeout(220);await p.screenshot({path:`docs/testing/ambient/${version}-${name}.png`});records[version][name]=await p.evaluate(selectors=>Object.fromEntries(selectors.map(s=>{const e=document.querySelector(s),r=e.getBoundingClientRect();return[s,{x:r.x,y:r.y,width:r.width,height:r.height,font:getComputedStyle(e).fontSize}]})),selectors);}
 async function close(){if(await p.locator('.task-view').isVisible())await p.locator('.task-view>.close-task').click();}
 async function open(view){await close();await p.locator('.control-trigger').click();await p.locator(`.control-panel [data-open=${view}]`).first().click();}
 await shot('home',['[data-clock]','.environment-line','.control-trigger','.wake-target']);
 await p.locator('.control-trigger').click();await shot('controls',['.cc-columns','.control-panel','.cc-reminder-menu','.cc-grid']);
 await open('alarm');await shot('alarm',['.cc-columns','.alarm-wheels','[data-wheel="0"]']);
 // Deliberate held drag, not merely a wheel-value click, in both outputs.
 const w=p.locator('[data-wheel="0"]'),r=await w.boundingBox();await p.mouse.move(r.x+r.width/2,r.y+110);await p.mouse.down();await p.waitForTimeout(300);await p.mouse.move(r.x+r.width/2,r.y+22,{steps:8});await p.mouse.up();await p.waitForTimeout(400);ok(`${version} held drag changes 07 to 09`,await w.getAttribute('aria-valuetext')==='09');
 await open('lights');await shot('lights',['.light-list','.light-editor','.light-row']);
 await open('climate');await shot('climate',['.thermostat','.sensor-source','.dial-number']);
 await open('music');await shot('music',['.music-layout','.music-info','.music-buttons']);
 await open('timer');await shot('timer',['.timer-adjust','.big-time']);
 await close();await p.locator('.control-trigger').click();await p.locator('[data-action=new-reminder]').click();await shot('reminder',['.reminder-form','.deadline-line']);
 await open('contacts');await shot('contacts',['.contacts','.contact-button']);
 await p.locator('[data-contact="Front door"]').click();await shot('call',['.call-person','.camera-preview']);await p.locator('[data-action=hangup]').click();
}
const parity=[];for(const [view,selects]of Object.entries(records.prototype)){for(const [s,a]of Object.entries(selects)){const z=records.svelte[view][s],delta=Math.max(...['x','y','width','height'].map(k=>Math.abs(a[k]-z[k])));parity.push({view,selector:s,maxGeometryDifferencePx:delta,fontMatches:a.font===z.font});}}
ok('principal layout geometry matches within one pixel',parity.every(r=>r.maxGeometryDifferencePx<=1&&r.fontMatches));
// Real dummy transport through the production frontend, every fixture from a clean state.
for(let i=0;i<15;i++){await p.goto(urls.svelte);await p.locator('.demo-launch').click();await p.getByText('Dummy backend ready',{exact:false}).waitFor();const response=p.waitForResponse(r=>r.url().endsWith('/api/ambient/infer'));await p.locator(`[data-demo="${i}"]`).click();ok(`fixture ${i} crosses the backend boundary`,(await response).status()===200);await p.waitForTimeout(650);if(await p.locator('.alert-surface').isVisible())await p.locator('[data-action=alert-dismiss]').click();}
await p.goto('http://127.0.0.1:3000/?diagnostics');await p.getByText('connected (mock)',{exact:false}).waitFor();ok('preserved diagnostics connect and allow scrolling',await p.locator('body').evaluate(e=>getComputedStyle(e).overflow==='auto'));
// Explicit denial path: launch tries once, first gesture retries once, Escape never loops.
await p.addInitScript(()=>{window.fullscreenAttempts=0;Object.defineProperty(document,'fullscreenEnabled',{configurable:true,value:true});Element.prototype.requestFullscreen=function(){window.fullscreenAttempts++;return Promise.reject(new DOMException('Activation required','NotAllowedError'));};});
await p.goto('http://127.0.0.1:3000/');await p.waitForTimeout(250);ok('fullscreen attempted at launch',await p.evaluate(()=>window.fullscreenAttempts===1));await p.locator('.wake-target').click();ok('fullscreen retries on first gesture',await p.evaluate(()=>window.fullscreenAttempts===2));await p.keyboard.press('Escape');await p.locator('.wake-target').click();ok('fullscreen denial leaves controls usable without repeated attempts',await p.evaluate(()=>window.fullscreenAttempts===2)&&await p.locator('main').getAttribute('data-state')==='idle');
ok('all views and fixture runs have no browser errors',errors.length===0);
fs.writeFileSync('docs/testing/ambient/parity.json',JSON.stringify({checks,parity,errors},null,2));await b.close();})().catch(e=>{console.error(e);process.exit(1)});
