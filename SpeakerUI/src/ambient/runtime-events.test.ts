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
