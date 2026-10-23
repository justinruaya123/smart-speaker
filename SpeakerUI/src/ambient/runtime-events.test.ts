import { test } from 'node:test';
import assert from 'node:assert/strict';
import { describeCommand, applyRuntimeEvent } from './runtime-events.js';

const cases = [
  ['PLAY_MUSIC',{song_id:'song.a',artist_id:'artist.a'},'media.play'],['WEATHER',{},'answer.weather'],['TIME',{},'answer.time'],
  ['LIGHT_ON',{light_id:'light.all'},'light.on'],['LIGHT_OFF',{light_id:'light.desk'},'light.off'],['PAUSE',{},'media.pause'],
  ['STOP',{},'media.stop'],['NEXT',{},'media.next'],['VOLUME',{operation:'set',percent:17},'media.volume'],
  ['CALL',{contact_id:'contact.mom'},'call.start'],['MESSAGE',{},'message.compose'],['LIST_REMINDERS',{},'reminder.list'],
  ['TIMER',{duration_seconds:123},'timer.set'],['ALARM',{time:'07:13',timezone:'Asia/Manila'},'alarm.set'],
  ['TEMPERATURE',{degrees:-4,unit:'C'},'climate.set_temperature'],['BRIGHTNESS',{light_id:'light.desk',percent:17},'light.brightness'],
  ['COLOR',{light_id:'light.desk',color:'purple'},'light.color'],['CREATE_REMINDER',{task_id:'task.milk'},'reminder.create'],
] as const;

for (const [intent, slots, frontend] of cases) test(`${intent} is preserved in debug without dispatching actions`,()=>{
  const command={intent,slots};const mapped=describeCommand(command);
  assert.equal(mapped.intent,frontend);
  assert.deepEqual(mapped.slots,slots);
  const model:any={history:[],request:0};let state='';
  applyRuntimeEvent(model,{event:'command',session_id:'s',sequence:1,utterance_id:'u',source:'wav',output:{schema_version:3,accepted:true,command,reason:null}},(s:string)=>state=s);
  assert.deepEqual(model.runtimeResult.output.command,command);
  assert.equal(model.event.result.intent,frontend);
  assert.equal(model.event.actionsEnabled,false);
  assert.equal(model.view,undefined);assert.equal(state,'idle');
});

test('rejection never promotes the diagnostic command; duplicate/late final ignored',()=>{
  const model:any={history:[]};const event={event:'command',session_id:'s',sequence:3,output:{schema_version:3,accepted:false,command:null,reason:'unqualified_model'},diagnostic_candidate:{candidate:{intent:'TIME',slots:{}}}};
  applyRuntimeEvent(model,event,()=>{});applyRuntimeEvent(model,event,()=>{});
  applyRuntimeEvent(model,{...event,sequence:2},()=>{});
  assert.equal(model.history.length,1);assert.equal(model.event.result.intent,'rejected');
  assert.equal(model.event.reason,'unqualified_model');assert.equal(model.runtimeResult.diagnostic_candidate.candidate.intent,'TIME');
});

test('exact parameters are adapted without clamps, timezone substitution, or invented slots',()=>{
  assert.deepEqual(describeCommand({intent:'TIMER',slots:{duration_seconds:123}}).parameters,{seconds:123});
  assert.deepEqual(describeCommand({intent:'ALARM',slots:{time:'07:13',timezone:'Asia/Manila'}}).parameters,{time:'07:13',timezone:'Asia/Manila'});
  assert.deepEqual(describeCommand({intent:'TEMPERATURE',slots:{degrees:-4,unit:'C'}}).parameters,{temperature:-4,unit:'C'});
  assert.deepEqual(describeCommand({intent:'BRIGHTNESS',slots:{light_id:'light.desk',percent:17}}).parameters,{light_id:'light.desk',brightness_pct:17});
  assert.deepEqual(describeCommand({intent:'MESSAGE',slots:{}}).parameters,{});
});

test('actual states control aura; failure never fabricates speaking',()=>{
  const model:any={};const states:string[]=[];
  for(const [sequence,state] of ['wake_listening','command_ready','processing','capture_error'].entries())
    applyRuntimeEvent(model,{event:'state',state,session_id:'s',sequence},(s:string)=>states.push(s));
  assert.deepEqual(states,['idle','listening','thinking','idle']);
});

test('a new WAV clears stale errors without substituting a result',()=>{
  const model:any={runtimeError:'Old malformed WAV',runtimeResult:null};
  applyRuntimeEvent(model,{event:'state',state:'processing',session_id:'s',sequence:1},()=>{});
  assert.equal(model.runtimeError,'');assert.equal(model.runtimeResult,null);
});


test('application acceptance is separate from qualification and local execution is applied once',()=>{
  const model:any={history:[],view:'home'};
  const command={intent:'TIMER',slots:{duration_seconds:123}};
  applyRuntimeEvent(model,{event:'command',session_id:'s',sequence:1,utterance_id:'u',source:'microphone',actions_enabled:true,
    output:{schema_version:3,accepted:false,command:null,reason:'unqualified_model'},application:{accepted:true,command}},()=>{});
  assert.equal(model.view,'home');assert.equal(model.event.status,'pending');
  const execution={event:'execution',utterance_id:'u',execution:{status:'local',command,deadline:123456}};
  applyRuntimeEvent(model,execution,()=>{});
  assert.equal(model.timer.end,123456);assert.equal(model.timer.total,123);
  model.timer.end=999;
  applyRuntimeEvent(model,execution,()=>{});assert.equal(model.timer.end,999);
});

test('tentative revisions preview only, expire on state reset and cannot overwrite a final result', () => {
  const model: any = { history: [] };
  const candidate = { intent: 'TIMER', slots: { duration_seconds: 30 } };
  applyRuntimeEvent(model, { event: 'tentative', session_id: 's', utterance_id: 'u', sequence: 1, candidate, score: .8, is_final: false }, () => {});
  assert.deepEqual(model.tentative.candidate, candidate);
  assert.equal(model.history.length, 0);
  assert.equal(model.timer, undefined);
  applyRuntimeEvent(model, { event: 'state', session_id: 's', sequence: 1.5, state: 'processing' }, () => {});
  assert.deepEqual(model.tentative.candidate, candidate);
  applyRuntimeEvent(model, { event: 'state', session_id: 's', sequence: 2, state: 'wake_listening' }, () => {});
  assert.equal(model.tentative, null);
  applyRuntimeEvent(model, { event: 'command', session_id: 's', utterance_id: 'u', sequence: 3, source: 'microphone',
    output: { schema_version: 3, accepted: true, command: candidate }, application: { accepted: true, command: candidate } }, () => {});
  applyRuntimeEvent(model, { event: 'tentative', session_id: 's', utterance_id: 'u', sequence: 4, candidate, score: .9, is_final: false }, () => {});
  assert.equal(model.tentative, null);
  assert.equal(model.history.length, 1);
});

test('a reconnect snapshot of a failed worker cannot display a ready microphone', () => {
  const model:any={runtimeReady:true};
  applyRuntimeEvent(model,{event:'ready',session_id:'s',sequence:1,loaded:false,state:'runtime_error',actions_enabled:false},()=>{});
  assert.equal(model.runtimeReady,false);
  assert.equal(model.runtimeState,'runtime_error');
});

test('atomic epoch44 diagnostics retain actual posterior, margin, identity and latency without independent-head defaults',()=>{
  const model:any={history:[]};
  const identity={parameters:258912,selected_epoch:44,model_sha256:'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce'};
  applyRuntimeEvent(model,{event:'ready',session_id:'atomic',sequence:0,...identity,wake_phrase:'hey iris',model_qualified:false,
    application_config:{mode:'joint_complete_command',min_probability:.5,min_margin:.1}},()=>{});
  assert.deepEqual(model.runtimeIdentity,{...identity,wake_phrase:'hey iris',model_qualified:false});
  const command={intent:'TEMPERATURE',slots:{degrees:22,unit:'C'}};
  applyRuntimeEvent(model,{event:'command',session_id:'atomic',sequence:1,utterance_id:'u',source:'microphone',actions_enabled:true,
    output:{schema_version:3,accepted:true,command},application:{accepted:true,command},latency_ms:7.25,
    timing:{total_ms:999},command_probability:.625,command_margin:.125},()=>{});
  assert.equal(model.event.result.latencyMs,7.25);
  assert.equal(model.event.result.commandPosterior,.625);
  assert.equal(model.event.result.commandMargin,.125);
});

test('execution transport dedupes touch and voice terminal receipts, while requested can advance to the real final result',()=>{
  const model:any={history:[]};
  const first={event:'execution',session_id:'s',utterance_id:'u',execution:{id:'s:u',status:'requested',command:{intent:'PLAY_MUSIC',slots:{}}}};
  assert.equal(applyRuntimeEvent(model,first,()=>{}),true);
  assert.equal(applyRuntimeEvent(model,first,()=>{}),false);
  const done={...first,execution:{...first.execution,status:'succeeded'}};
  assert.equal(applyRuntimeEvent(model,done,()=>{}),true);
  assert.equal(applyRuntimeEvent(model,done,()=>{}),false);
  assert.equal(applyRuntimeEvent(model,{...first,execution:{...first.execution,status:'unknown'}},()=>{}),false);
  const touch={event:'execution',execution:{id:'touch:request',status:'synthetic',command:{intent:'TEMPERATURE',slots:{degrees:22,unit:'C'}}}};
  assert.equal(applyRuntimeEvent(model,touch,()=>{}),true);
  assert.equal(applyRuntimeEvent(model,touch,()=>{}),false);
});

test('an execution delivered before its runtime frame is reconciled into history without replaying its effects',()=>{
  const model:any={history:[]};
  const command={intent:'TEMPERATURE',slots:{degrees:22,unit:'C'}};
  const execution={id:'s:u',status:'synthetic',command,provenance:{source:'synthetic',synthetic:true}};
  const receipt={event:'execution',session_id:'s',utterance_id:'u',execution};
  assert.equal(applyRuntimeEvent(model,receipt,()=>{}),true);
  applyRuntimeEvent(model,{event:'command',session_id:'s',sequence:1,utterance_id:'u',source:'microphone',actions_enabled:true,
    output:{schema_version:3,accepted:true,command},application:{accepted:true,command}},()=>{});
  assert.equal(model.event.status,'synthetic');
  assert.deepEqual(model.event.execution,execution);
  assert.equal(applyRuntimeEvent(model,receipt,()=>{}),false);
});
