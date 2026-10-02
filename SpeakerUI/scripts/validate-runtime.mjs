// Real WAV -> native worker -> gateway -> browser adapter parity. No device actions.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import WebSocket from 'ws';
import { applyRuntimeEvent } from '../src/ambient/runtime-events.js';

const [url, runtimeRoot, python, output] = process.argv.slice(2);
if (!url || !runtimeRoot || !python || !output) throw new Error('Usage: validate-runtime.mjs URL EXTRACTED_PACKAGE PYTHON OUTPUT.json');
const rows=JSON.parse(await readFile(resolve(runtimeRoot,'samples/manifest.json'),'utf8'));
const health=await (await fetch(url+'/api/health')).json();
assert.equal(health.inference,'onnx');assert.equal(health.actionsEnabled,false);assert.equal(health.runtime.nativeSession,true);
const socket=new WebSocket(url.replace(/^http/,'ws')+'/ws/runtime',{origin:url});
const events=[];socket.on('message',data=>events.push(JSON.parse(data.toString())));
await new Promise((ok,bad)=>{socket.once('open',ok);socket.once('error',bad);});
const trials=[];
try {
  for(const row of rows) {
    const wav=resolve(runtimeRoot,'samples',row.audio);
    const cli=spawnSync(python,['-m','scripts.runtime.speaker_worker','--bundle',resolve(runtimeRoot,'models/default'),'--diagnostic','--wav',wav],{cwd:runtimeRoot,encoding:'utf8',windowsHide:true});
    assert.equal(cli.status,0,cli.stderr);
    const expected=cli.stdout.trim().split(/\r?\n/).map(line=>JSON.parse(line)).find(event=>event.event==='command');
    const start=events.length, began=performance.now();
    const response=await fetch(url+'/api/runtime/wav',{method:'POST',headers:{Origin:url,'Content-Type':'audio/wav'},body:await readFile(wav)});
    assert.equal(response.status,200,await response.text());
    const deadline=performance.now()+5000;
    while(!events.slice(start).some(e=>e.event==='command') && performance.now()<deadline) await new Promise(r=>setTimeout(r,10));
    const event=events.slice(start).find(e=>e.event==='command');assert.ok(event,'Missing command event');
    assert.deepEqual(event.output,expected.output);assert.deepEqual(event.diagnostic_candidate,expected.diagnostic_candidate);
    const model={history:[]};applyRuntimeEvent(model,event,()=>{});
    assert.deepEqual(model.runtimeResult.output,event.output);assert.equal(model.event.actionsEnabled,false);
    trials.push({id:row.id,expected:row.expected,output:event.output,candidate:event.diagnostic_candidate?.candidate,
      cli_gateway_frontend_equal:true,request_to_result_ms:performance.now()-began});
  }
  const rejected=await fetch(url+'/api/runtime/wav',{method:'POST',headers:{Origin:url,'Content-Type':'audio/wav'},body:Buffer.from('invalid')});assert.equal(rejected.status,422);
  const mock=await fetch(url+'/api/ambient/infer',{method:'POST',headers:{Origin:url,'Content-Type':'application/json'},body:'{"fixture":0}'});assert.equal(mock.status,409);
  const cross=await fetch(url+'/api/runtime/mute',{method:'POST',headers:{Origin:'http://other.invalid','Content-Type':'application/json'},body:'{}'});assert.equal(cross.status,403);
  const result={scope:'Actual WAV/CPU ONNX/HTTP/WebSocket/frontend adapter parity; not live mic/Pi or qualification',trials,
    passed:true,policy_enabled:health.runtime.policy_enabled,actions_enabled:false};
  await writeFile(output,JSON.stringify(result,null,2)+'\n');console.log(`Real runtime parity: ${trials.length}/${trials.length}; actions disabled; negative requests rejected.`);
} finally {socket.close();}
