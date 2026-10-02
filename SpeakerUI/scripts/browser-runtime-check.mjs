// Compiled-page checks. Contract fixtures are labelled separately from actual WAV replay.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
const { chromium }=await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : 'playwright');
const [url,bundle,output]=process.argv.slice(2);
if(!url||!bundle||!output)throw new Error('Usage: browser-runtime-check.mjs URL EXTRACTED_PACKAGE OUTPUT_DIRECTORY');
await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_EXECUTABLE_PATH?{executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH}:{})});
const errors=[];const checks=[];
try {
  for(const [width,height] of [[800,480],[1024,600]]) {
    const page=await browser.newPage({viewport:{width,height}});page.on('pageerror',e=>errors.push(e.message));
    await page.goto(url);await page.getByRole('button',{name:'Open debug panel'}).waitFor();
    await page.getByRole('button',{name:'Open debug panel'}).click();
    await page.getByRole('button',{name:'Inference',exact:true}).click();
    await page.getByLabel('Replay WAV',{exact:true}).setInputFiles(resolve(bundle,'samples/slurpsmoke-05.wav'));
    await page.waitForFunction(()=>document.querySelector('[data-runtime-result]')?.textContent?.includes('diagnostic_candidate'));
    const event=JSON.parse(await page.locator('[data-runtime-result]').innerText());
    assert.equal(event.output.schema_version,3);assert.equal(event.actions_enabled,false);
    assert.equal(await page.locator('main').getAttribute('data-view'),'demo');
    const bounds=await page.locator('.runtime-debug').boundingBox();assert.ok(bounds.x>=0 && bounds.x+bounds.width<=width+1);
    await page.screenshot({path:resolve(output,`wav-${width}x${height}.png`)});
    const readout=await page.locator('.inference-readout').boundingBox();assert.ok(readout && readout.x>=0 && readout.y+readout.height<=height);
    const metrics=await page.evaluate(()=>({resources:performance.getEntriesByType('resource').length,dom:document.querySelectorAll('*').length,heap:performance.memory?.usedJSHeapSize}));
    checks.push({viewport:[width,height],kind:'browser_resource_snapshot',metrics,passed:true});
    await page.getByRole('button',{name:'Close task',exact:true}).click();
    await page.locator('.inference-readout').waitFor({state:'visible'});
    await page.screenshot({path:resolve(output,`home-${width}x${height}.png`)});
    await page.getByRole('button',{name:'Inspect inference history',exact:true}).click();
    assert.equal(await page.locator('main').getAttribute('data-view'),'intent');
    checks.push({viewport:[width,height],kind:'real_wav',output:event.output,candidate:event.diagnostic_candidate?.candidate,passed:true});
    await page.close();
  }
  // Exercise every schema intent in the compiled Svelte UI by replacing only transport.
  // This is a contract test, not acoustic evidence and never reaches the server/actions.
  const definitions=JSON.parse(await readFile(resolve(bundle,'verbatims/definitions.json'),'utf8'));
  const commands=[...new Map(definitions.filter(r=>r.command).map(r=>[r.command.intent,r.command])).values()];
  assert.equal(commands.length,18);
  const context=await browser.newContext({viewport:{width:800,height:480}});let channel;
  await context.routeWebSocket('**/ws/runtime',ws=>{channel=ws;ws.send(JSON.stringify({event:'ready',session_id:'contract-fixtures',sequence:0,state:'awaiting_microphone',policy_enabled:false}));});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
  await page.getByRole('button',{name:'Open debug panel'}).click();
    await page.getByRole('button',{name:'Inference',exact:true}).click();
  let sequence=1;
  for(const command of commands) {
    const event={event:'command',session_id:'contract-fixtures',sequence:sequence++,utterance_id:command.intent,source:'contract-fixture',
      actions_enabled:false,output:{schema_version:3,accepted:true,command,reason:null}};
    channel.send(JSON.stringify(event));
    await page.waitForFunction(intent=>{try{return JSON.parse(document.querySelector('[data-runtime-result]').textContent).output.command.intent===intent}catch{return false}},command.intent);
    const shown=JSON.parse(await page.locator('[data-runtime-result]').innerText());assert.deepEqual(shown.output.command,command);
    assert.equal(await page.locator('main').getAttribute('data-view'),'demo');
    checks.push({kind:'typed_contract_fixture',command,passed:true,acoustic_test:false});
  }
  await page.screenshot({path:resolve(output,'intent-contract.png')});await context.close();
  assert.deepEqual(errors,[]);
  await writeFile(resolve(output,'results.json'),JSON.stringify({passed:true,checks,errors,live_microphone:false,physical_pi:false},null,2)+'\n');
  console.log(`Compiled browser: ${checks.length} checks passed (2 actual WAV layouts, 18 typed fixtures).`);
} finally {await browser.close();}
