// Software boundary fixture: real dispatcher, fake providers, no network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const root=process.cwd(), compiler=process.argv[2];
assert(compiler, 'Pass the verified scratch TypeScript compiler path');
const require=createRequire(import.meta.url), ts=require(compiler);
const scratch=path.resolve(path.dirname(compiler),'../../boundary');
fs.mkdirSync(scratch,{recursive:true});
fs.writeFileSync(path.join(scratch,'package.json'),JSON.stringify({type:'module'}));
const files=['server/inference/runtime-actions.ts','server/integrations/commands.ts','server/integrations/http.ts','shared/protocol.ts'];
const hashes={};
for(const relative of files){
  const source=path.join(root,'SpeakerUI',relative), text=fs.readFileSync(source,'utf8');
  hashes[relative]=crypto.createHash('sha256').update(text).digest('hex');
  const result=ts.transpileModule(text,{fileName:relative,reportDiagnostics:true,
    compilerOptions:{module:ts.ModuleKind.ES2022,target:ts.ScriptTarget.ES2022}});
  assert(!(result.diagnostics??[]).some(d=>d.category===ts.DiagnosticCategory.Error),'Source failed isolated transpilation');
  const destination=path.join(scratch,relative.replace(/\.ts$/,'.js'));
  fs.mkdirSync(path.dirname(destination),{recursive:true});fs.writeFileSync(destination,result.outputText);
}
globalThis.fetch=async()=>{throw Error('Network forbidden in this software fixture');};
const {RuntimeActions}=await import(pathToFileURL(path.join(scratch,'server/inference/runtime-actions.js')).href);
const inventory=JSON.parse(fs.readFileSync(path.join(root,'experiments/speaker-optionb-tiny-v1/command-inventory.json'),'utf8'));
assert.equal(inventory.commands.length,32);assert.equal(inventory.verbatims.length,93);
const calls=[];
const actions=new RuntimeActions({enabled:true,bindings:{lights:{'light.all':'light.fixture'},songs:{}}},{
  light:async c=>{calls.push({provider:'light',command:c});},
  spotify:async c=>{calls.push({provider:'spotify',command:c});},
  song:async()=>{throw Error('No named songs in OptionB');},volume:async()=>50,
  weather:async()=>{calls.push({provider:'weather_fixture'});return {text:'Software weather fixture, not live weather.'};},
  temperature:async degrees=>{calls.push({provider:'thermostat_fixture',degrees});},
  camera:async()=>{calls.push({provider:'camera_fixture'});return {entityId:'camera.fixture',name:'Software fixture camera'};},
  message:async()=>{calls.push({provider:'message_fixture'});return 'Software notification fixture; no message sent.';},
});
const results=[];
try{
  for(let index=1;index<inventory.commands.length;index++){
    const command=inventory.commands[index], event={event:'command',session_id:'software-fixture',utterance_id:'class-'+index,
      source:'microphone',application:{accepted:true,command}};
    const result=await actions.handle(event);
    assert(['succeeded','local'].includes(result?.status),JSON.stringify({index,result}));
    assert.deepEqual(result.command,command);
    assert.equal(await actions.handle(event),undefined,'Duplicate event must not execute again');
    const before=calls.length;
    assert.equal((await actions.handle({...event,source:'wav',utterance_id:'wav-'+index}))?.status,'preview');
    assert.equal(calls.length,before,'WAV preview must not call a provider');
    results.push({class_index:index,command,status:result.status});
  }
  assert.equal((await actions.handle({event:'command',session_id:'software-fixture',utterance_id:'rejected',source:'microphone',application:{accepted:false}}))?.status,'rejected');
}finally{actions.close();}
for(const relative of files)assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'SpeakerUI',relative),'utf8')).digest('hex'),hashes[relative]);
const receipt={software_fixture_only:true,live_microphone_or_device_evidence:false,network_requests:0,
  classes_tested:results.length,wav_previews_tested:results.length,duplicates_rejected:results.length,
  source_hashes:hashes,node:process.version,compiler:ts.version,results,fixture_provider_calls:calls};
fs.writeFileSync(path.join(root,'experiments/speaker-optionb-tiny-v1/action-boundary-results.json'),JSON.stringify(receipt,null,2));
console.log(JSON.stringify({classes_tested:results.length,wav_previews_tested:results.length,duplicates_rejected:results.length,network_requests:0}));
