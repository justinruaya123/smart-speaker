// Actual resident-worker replay. Optional contract fixtures are separate evidence.
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { setTimeout as pause } from 'node:timers/promises';
import { arch, platform } from 'node:os';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import assert from 'node:assert/strict';
const { chromium }=await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : 'playwright');
const [url,bundle,output,...args]=process.argv.slice(2);
if(!url||!bundle||!output)throw new Error('Usage: browser-runtime-check.mjs URL INSTALLATION_ROOT OUTPUT_DIRECTORY [--recording HASH_BOUND_REFERENCE_JSON] [--server-pid LOCAL_GATEWAY_PID] [--real-only]');
let referencePath,serverPid,realOnly=false;
for(let index=0;index<args.length;index++) {
  if(args[index]==='--recording'&&args[index+1])referencePath=resolve(args[++index]);
  else if(args[index]==='--server-pid'&&args[index+1])serverPid=Number(args[++index]);
  else if(args[index]==='--real-only')realOnly=true;
  else throw new Error('Unknown or incomplete option: '+args[index]);
}
const digest=value=>createHash('sha256').update(value).digest('hex');
const exists=async path=>{try{await access(path);return true;}catch{return false;}};
const viewports=[[800,480],[1024,600],[390,844]];
const threadKeys=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'];
let processTargets=[],clockTicks;
if(serverPid!==undefined) {
  assert.ok(Number.isSafeInteger(serverPid)&&serverPid>0,'Use the actual local gateway PID');
  assert.equal(platform(),'linux','--server-pid measurements require Linux /proc');
  assert.ok(['127.0.0.1','localhost','[::1]'].includes(new URL(url).hostname),'Process evidence must use the local gateway');
  clockTicks=Number(execFileSync('getconf',['CLK_TCK'],{encoding:'utf8',timeout:2000}).trim());
  assert.ok(clockTicks>0);
  const children=(await readFile(`/proc/${serverPid}/task/${serverPid}/children`,'utf8')).trim().split(/\s+/).filter(Boolean).map(Number);
  const workers=[];
  for(const pid of children)if((await readFile(`/proc/${pid}/cmdline`,'utf8')).includes('scripts.runtime.speaker_worker'))workers.push(pid);
  assert.equal(workers.length,1,'Gateway must own exactly one resident speaker_worker process');
  const environment=Object.fromEntries((await readFile(`/proc/${workers[0]}/environ`,'utf8')).split('\0').map(item=>{
    const index=item.indexOf('=');return [item.slice(0,index),item.slice(index+1)];
  }).filter(([key])=>threadKeys.includes(key)));
  processTargets=[{role:'node_gateway',pid:serverPid},{role:'python_worker',pid:workers[0],numeric_thread_environment:environment}];
}
async function processSnapshot(stage) {
  const processes=await Promise.all(processTargets.map(async target=>{
    const [stat,status]=await Promise.all([readFile(`/proc/${target.pid}/stat`,'utf8'),readFile(`/proc/${target.pid}/status`,'utf8')]);
    const fields=stat.slice(stat.lastIndexOf(')')+2).trim().split(/\s+/);
    const kib=name=>{const value=status.match(new RegExp('^'+name+':\\s+(\\d+) kB','m'));return value?Number(value[1])/1024:null;};
    return {...target,start_ticks:fields[19],cpu_ticks:Number(fields[11])+Number(fields[12]),rss_mib:kib('VmRSS'),lifetime_peak_rss_mib:kib('VmHWM')};
  }));
  return {stage,monotonic_ms:performance.now(),processes};
}
function processWindow(first,last,samples) {
  return {wall_ms:last.monotonic_ms-first.monotonic_ms,processes:first.processes.map(before=>{
    const after=last.processes.find(item=>item.role===before.role);
    assert.equal(after.start_ticks,before.start_ticks,'Observed process changed during measurement');
    const cpuSeconds=(after.cpu_ticks-before.cpu_ticks)/clockTicks;
    return {role:before.role,pid:before.pid,cpu_seconds:cpuSeconds,cpu_percent_one_core:100*cpuSeconds/((last.monotonic_ms-first.monotonic_ms)/1000),
      sampled_peak_rss_mib:Math.max(...samples.map(sample=>sample.processes.find(item=>item.role===before.role)?.rss_mib??0)),
      lifetime_peak_rss_mib:after.lifetime_peak_rss_mib};
  }),samples};
}
async function observeProcesses() {
  if(!processTargets.length)return;
  const idleStart=await processSnapshot('connected_idle_start');await pause(400);
  const idleEnd=await processSnapshot('connected_idle_end');
  const replayStart=await processSnapshot('replay_start'),samples=[replayStart];let pending;
  const timer=setInterval(()=>{if(!pending)pending=processSnapshot('replay').then(value=>samples.push(value)).finally(()=>{pending=undefined;});},20);
  return {finish:async()=>{clearInterval(timer);await pending;const end=await processSnapshot('replay_end');samples.push(end);
    return {idle:processWindow(idleStart,idleEnd,[idleStart,idleEnd]),replay:processWindow(replayStart,end,samples)};},
    cancel:()=>clearInterval(timer)};
}
let reference;
if(referencePath)reference=JSON.parse(await readFile(referencePath,'utf8'));
else {
  const manifest=resolve(bundle,'samples/commands.jsonl');
  const payload=await readFile(manifest,'utf8');
  const rows=payload.trim().split('\n').map(line=>JSON.parse(line));
  reference=rows.find(row=>row.expected)||rows[0];
  reference={...reference,manifest,manifest_sha256:digest(payload)};
}
assert.ok(reference?.id&&typeof reference.audio==='string'&&/^[a-f0-9]{64}$/i.test(reference.sha256),'A hash-bound recording reference is required');
assert.ok(Object.hasOwn(reference,'expected'),'Reference must explicitly declare expected command or null');
const wav=resolve(bundle,reference.audio);
assert.equal(digest(await readFile(wav)),reference.sha256.toLowerCase(),'WAV differs from permitted reference');
if(reference.manifest)assert.equal(digest(await readFile(reference.manifest)),reference.manifest_sha256,'Source manifest changed');
await mkdir(output,{recursive:true});
const errors=[];const checks=[];
const health=await (await fetch(new URL('/api/health',url))).json();
assert.equal(health.inference,'onnx');assert.equal(health.runtime.loaded,true);assert.equal(health.runtime.nativeSession,true);
assert.equal(health.runtime.profile,'benchmark-compatible');assert.equal(health.runtime.owner_gating_enabled,false);
assert.equal(health.runtime.actions_enabled,false);assert.equal(health.actionsEnabled,false);
assert.equal(health.homeAssistant,'disabled');assert.equal(health.spotify,'disabled');
if(reference.bundle) {
  const payload=await readFile(resolve(reference.bundle,'bundle.json'));
  assert.equal(digest(payload),reference.bundle_sha256,'Pinned baseline bundle changed');
  assert.equal(health.runtime.model_sha256,JSON.parse(payload).files['model.onnx'],'Worker loaded a different model');
}
const before=await (await fetch(new URL('/api/actions/state',url))).json();
const sockets=[],mutations=[],externalRequests=[],processObservations=[];
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_EXECUTABLE_PATH?{executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH}:{})});
try {
  for(const [width,height] of viewports) {
    const page=await browser.newPage({viewport:{width,height}});page.on('pageerror',e=>errors.push(e.message));
    const frames=[];
    page.on('websocket',socket=>{
      socket.on('framereceived',frame=>{
        const value=JSON.parse(String(frame.payload));frames.push({channel:new URL(socket.url()).pathname,received_at_ms:performance.now(),value});
      });
    });
    page.on('request',request=>{
      const target=new URL(request.url());
      if(['POST','PUT','PATCH','DELETE'].includes(request.method()))mutations.push({viewport:[width,height],method:request.method(),path:target.pathname,query:target.search});
      if(target.origin!==new URL(url).origin)externalRequests.push(request.url());
    });
    await page.goto(url);await page.getByRole('button',{name:'Microphone settings'}).waitFor();
    await page.getByRole('button',{name:'Microphone settings'}).click();
    const observation=await observeProcesses();
    const response=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/runtime/wav');
    const started=performance.now();
    let httpReply,httpReplyAt;
    try {
      await page.getByLabel('Replay WAV',{exact:true}).setInputFiles(wav);
      const reply=await response;assert.equal(reply.status(),200);
      httpReply=await reply.json();httpReplyAt=performance.now();
      await page.waitForFunction(()=>{try{return JSON.parse(document.querySelector('[data-runtime-result]')?.textContent).event==='command';}catch{return false;}});
      await page.locator('[data-execution-status="preview"]').waitFor();
    } catch(error) {observation?.cancel();throw error;}
    const observedWallMs=performance.now()-started;
    if(observation)processObservations.push({viewport:[width,height],...await observation.finish()});
    const event=JSON.parse(await page.locator('[data-runtime-result]').innerText());
    assert.equal(event.output.schema_version,3);assert.equal(event.actions_enabled,false);assert.equal(event.source,'wav');
    assert.equal(event.profile,'benchmark-compatible');assert.equal(event.session_id,health.runtime.session_id);
    assert.equal(event.owner_gating_enabled,false);
    const received=frames.find(frame=>frame.channel==='/ws/runtime'&&frame.value.event==='command'&&frame.value.utterance_id===event.utterance_id);
    assert.deepEqual(received?.value,event,'DOM final must be the actual worker WebSocket command');
    const execution=frames.find(frame=>frame.channel==='/ws/actions'&&frame.value.event==='execution'&&frame.value.utterance_id===event.utterance_id);
    assert.equal(execution?.value.execution.status,'preview');assert.equal(execution.value.session_id,event.session_id);
    const shownPhase=(await page.locator('.recognition .phase').textContent()).trim();
    assert.equal(shownPhase,event.application?.accepted?'Accepted · replay only':'Rejected');
    const shownReceipt=await page.getByRole('region',{name:'Execution receipt'}).innerText();
    assert.ok(shownReceipt.includes(execution.value.execution.detail));
    assert.equal(await page.locator('canvas').count(),0);
    const bounds=await page.locator('.runtime-debug').boundingBox();assert.ok(bounds.x>=0 && bounds.x+bounds.width<=width+1);
    await page.screenshot({path:resolve(output,`wav-${width}x${height}.png`)});
    const readout=await page.locator('.recognition').boundingBox();assert.ok(readout && readout.x>=0 && readout.y+readout.height<=height);
    const metrics=await page.evaluate(()=>({dom_elements:document.querySelectorAll('*').length,
      assets:[...performance.getEntriesByType('navigation'),...performance.getEntriesByType('resource')].map(entry=>({path:new URL(entry.name).pathname,
        transfer_bytes:entry.transferSize,encoded_bytes:entry.encodedBodySize,decoded_bytes:entry.decodedBodySize,duration_ms:entry.duration})),
      performance_memory_used_js_heap_bytes:performance.memory?.usedJSHeapSize??null}));
    const cdp=await page.context().newCDPSession(page);
    metrics.js_heap=await cdp.send('Runtime.getHeapUsage');metrics.dom=await cdp.send('Memory.getDOMCounters');await cdp.detach();
    checks.push({viewport:[width,height],kind:'browser_resource_snapshot',metrics,passed:true});
    await page.getByRole('button',{name:'Back to home',exact:true}).click();
    await page.locator('.recognition').waitFor({state:'visible'});
    await page.screenshot({path:resolve(output,`home-${width}x${height}.png`)});
    await page.getByRole('button',{name:'History',exact:true}).click();
    await page.getByRole('heading',{name:'History',exact:true}).waitFor();
    assert.ok((await page.locator('.history').innerText()).includes('wav'));
    checks.push({viewport:[width,height],kind:'resident_worker_wav',source_id:reference.id,
      expected:reference.expected,command_exact:isDeepStrictEqual(event.output.command,reference.expected),
      final:event,execution:execution.value,httpReply,shownPhase,shownReceipt,observed_browser_wall_ms:observedWallMs,
      path_timing_ms:{upload_to_http_reply:httpReplyAt-started,upload_to_runtime_final_ws:received.received_at_ms-started,
        upload_to_execution_receipt_ws:execution.received_at_ms-started,upload_to_rendered_receipt:observedWallMs},passed:true});
    sockets.push({viewport:[width,height],frames});
    await page.close();
  }
  // Exercise every schema intent in the compiled Svelte UI by replacing only transport.
  // This is a contract test, not acoustic evidence and never reaches the server/actions.
  const definitionsPath=resolve(bundle,'verbatims/definitions.json');
  if(!realOnly&&await exists(definitionsPath)) {
  const definitions=JSON.parse(await readFile(definitionsPath,'utf8'));
  const commands=[...new Map(definitions.filter(r=>r.command).map(r=>[r.command.intent,r.command])).values()];
  assert.equal(commands.length,18);
  const context=await browser.newContext({viewport:{width:800,height:480}});let channel;
  await context.routeWebSocket('**/ws/runtime',ws=>{channel=ws;ws.send(JSON.stringify({event:'ready',session_id:'contract-fixtures',sequence:0,state:'awaiting_microphone',policy_enabled:false}));});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
  await page.getByRole('button',{name:'Microphone settings'}).click();
  let sequence=1;
  for(const command of commands) {
    const event={event:'command',session_id:'contract-fixtures',sequence:sequence++,utterance_id:command.intent,source:'contract-fixture',
      actions_enabled:false,output:{schema_version:3,accepted:true,command,reason:null}};
    channel.send(JSON.stringify(event));
    await page.waitForFunction(intent=>{try{return JSON.parse(document.querySelector('[data-runtime-result]').textContent).output.command.intent===intent}catch{return false}},command.intent);
    const shown=JSON.parse(await page.locator('[data-runtime-result]').innerText());assert.deepEqual(shown.output.command,command);
    assert.equal(await page.locator('canvas').count(),0);
    checks.push({kind:'typed_contract_fixture',command,passed:true,acoustic_test:false});
  }
  await page.screenshot({path:resolve(output,'intent-contract.png')});await context.close();
  }
  const after=await (await fetch(new URL('/api/actions/state',url))).json();
  assert.deepEqual(after,before,'Recorded WAV replay must leave the action journal and jobs unchanged');
  assert.deepEqual(externalRequests,[]);
  assert.equal(mutations.length,viewports.length);
  assert.ok(mutations.every(request=>request.method==='POST'&&request.path==='/api/runtime/wav'&&request.query==='?wake=false'));
  assert.deepEqual(errors,[]);
  const finalHealth=await (await fetch(new URL('/api/health',url))).json();
  assert.equal(finalHealth.runtime.session_id,health.runtime.session_id,'All replays must share one resident worker');
  assert.equal(finalHealth.runtime.loaded,true);assert.equal(finalHealth.actionsEnabled,false);
  await writeFile(resolve(output,'websocket-events.json'),JSON.stringify(sockets,null,2)+'\n');
  await writeFile(resolve(output,'results.json'),JSON.stringify({schema_version:1,passed:true,utc:new Date().toISOString(),
    evidence_kind:'compiled_iris_resident_worker_recorded_wav',measurement_host:'desktop_host',host:{platform:platform(),arch:arch(),node:process.version},
    reference,health,finalHealth,action_state_before:before,action_state_after:after,mutations,externalRequests,checks,errors,
    process_resources:{available:processTargets.length>0,targets:processTargets,clock_ticks_per_second:clockTicks??null,sampling_period_ms:20,observations:processObservations,
      scope:'Actual local Node gateway and its resident speaker_worker process; Linux /proc CPU counters and RSS. '
        +'CPU resolution is one clock tick. Replay sampling is bounded to the WAV request/readout path; RSS peaks between samples can be missed. '
        +'Lifetime HWM includes earlier work. Browser CDP heap/DOM/assets are separate; no unique physical-memory total or Pi/browser RSS is inferred.'},
    real_worker_wav_n:checks.filter(check=>check.kind==='resident_worker_wav').length,
    typed_fixture_n:checks.filter(check=>check.kind==='typed_contract_fixture').length,
    source_wav_sha256:digest(await readFile(wav)),script_sha256:digest(await readFile(new URL(import.meta.url))),
    live_microphone:false,wake_tested:false,owner_check_tested:false,physical_pi:false,actions_executed:false},null,2)+'\n');
  console.log(`Compiled browser: ${viewports.length} actual resident-worker WAV layouts passed; ${checks.filter(check=>check.kind==='typed_contract_fixture').length} separate typed fixtures.`);
} finally {await browser.close();}
