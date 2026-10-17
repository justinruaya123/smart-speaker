import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RuntimeActions } from './runtime-actions.js';

function event(id='u') { return {event:'command',session_id:'s',utterance_id:id,source:'microphone',application:{accepted:true,command:{intent:'LIGHT_ON',slots:{light_id:'light.kitchen'}}}}; }
test('live events dispatch exactly once across duplicate final events',async()=>{
  const calls:any[]=[];const actions=new RuntimeActions({enabled:true,bindings:{lights:{'light.kitchen':'light.actual'},songs:{}}},
    {light:async c=>{calls.push(c);},spotify:async()=>{},song:async()=>{}});
  const results=await Promise.all([actions.handle(event()),actions.handle(event())]);
  assert.equal(calls.length,1);assert.equal(results[0]?.status,'succeeded');assert.equal(results[1],undefined);
  assert.deepEqual(calls[0],{entityId:'light.actual',action:'on'});
});
test('WAV, rejection, disabled and missing bindings cannot operate providers',async()=>{
  const providers={light:async()=>assert.fail('action'),spotify:async()=>assert.fail('action'),song:async()=>assert.fail('action')};
  const actions=new RuntimeActions({enabled:true,bindings:{lights:{},songs:{}}},providers);
  assert.equal((await actions.handle({...event('wav'),source:'wav'}))?.status,'preview');
  assert.equal((await actions.handle({...event('bad'),application:{accepted:false}}))?.status,'rejected');
  assert.equal((await actions.handle(event('missing')))?.status,'not_configured');
  const disabled=new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}},providers);
  assert.equal((await disabled.handle(event()))?.status,'disabled');
});
test('local timers preserve exact seconds and server deadline',async()=>{
  const a=new RuntimeActions({enabled:true,bindings:{lights:{},songs:{}}},{light:async()=>{},spotify:async()=>{},song:async()=>{}});
  const e={...event(),application:{accepted:true,command:{intent:'TIMER',slots:{duration_seconds:123}}}};
  const result=await a.handle(e);assert.equal(result?.status,'local');assert.equal(result?.command.slots.duration_seconds,123);
  assert.ok(result!.deadline>=Date.now()+122000);
});
test('generic play music resumes Spotify without a track or playlist',async()=>{
  const calls:unknown[]=[];
  const actions=new RuntimeActions({enabled:true,bindings:{lights:{},songs:{}}},{
    light:async()=>assert.fail('unexpected light control'),
    spotify:async command=>{calls.push(command);},
    song:async()=>assert.fail('generic playback must not select a song'),
  });
  const result=await actions.handle({...event('music'),application:{accepted:true,command:{intent:'PLAY_MUSIC',slots:{}}}});
  assert.equal(result?.status,'succeeded');
  assert.deepEqual(calls,[{action:'play'}]);
  const named=await actions.handle({...event('named'),application:{accepted:true,command:{intent:'PLAY_MUSIC',slots:{song_id:'song.unconfigured'}}}});
  assert.equal(named?.status,'not_configured');
  assert.deepEqual(calls,[{action:'play'}]);
});

test('weather and time give actual answers without enabling device actions; replays stay silent',async()=>{
  let calls=0;
  const actions=new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}},{light:async()=>{},spotify:async()=>{},song:async()=>{},
    weather:async()=>{calls++;return {text:'In Manila, it is 28 degrees Celsius with clear sky.',temperature:28,location:'Manila'};}});
  const weather={...event('weather'),application:{accepted:true,command:{intent:'WEATHER',slots:{}}}};
  assert.equal((await actions.handle(weather))?.text,'In Manila, it is 28 degrees Celsius with clear sky.');
  assert.equal((await actions.handle(weather)),undefined);assert.equal(calls,1);
  assert.equal((await actions.handle({...weather,utterance_id:'replay',source:'wav'}))?.status,'preview');assert.equal(calls,1);
  const time=await actions.handle({...weather,utterance_id:'time',application:{accepted:true,command:{intent:'TIME',slots:{}}}});
  assert.match(time?.text,/time is .+Manila/);
});

test('local commands return results and reminders are actually created and listed',async()=>{
  const actions=new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}},{light:async()=>{},spotify:async()=>{},song:async()=>{}});
  const run=(intent:string,slots:Record<string,unknown>,id=intent)=>actions.handle({...event(id),application:{accepted:true,command:{intent,slots}}});
  assert.match((await run('TIMER',{duration_seconds:90}))?.text,/1 minute.*30 seconds/);
  assert.match((await run('ALARM',{time:'07:13',timezone:'Asia/Manila'}))?.text,/7:13.*Manila/);
  const created=await run('CREATE_REMINDER',{task_id:'task.study'});assert.match(created?.text,/study/i);
  const listed=await run('LIST_REMINDERS',{});assert.equal(listed?.reminders.length,1);assert.match(listed?.text,/Study/);
  assert.match((await run('CALL',{contact_id:'contact.mom'}))?.text,/not configured/i);
  assert.match((await run('MESSAGE',{}))?.text,/not configured/i);
  actions.close();
});

test('timer expiry emits one real notification; replacements and shutdown cancel pending alerts',async t=>{
  t.mock.timers.enable({apis:['setTimeout','Date']});
  const notifications:Record<string,any>[]=[];
  const actions=new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}},{light:async()=>{},spotify:async()=>{},song:async()=>{},notify:e=>notifications.push(e)});
  const timer=(id:string,seconds:number)=>({...event(id),application:{accepted:true,command:{intent:'TIMER',slots:{duration_seconds:seconds}}}});
  await actions.handle(timer('one',1));await actions.handle(timer('two',2));
  t.mock.timers.tick(1000);assert.equal(notifications.length,0);
  t.mock.timers.tick(1000);assert.equal(notifications.length,1);assert.match(notifications[0]?.text,/timer is finished/);
  await actions.handle(timer('three',1));actions.close();t.mock.timers.tick(1000);assert.equal(notifications.length,1);
});

test('configured camera and message tools use real provider results, never WAV replay',async()=>{
  let camera=0,message=0;
  const actions=new RuntimeActions({enabled:true,bindings:{lights:{},songs:{}}},{light:async()=>{},spotify:async()=>{},song:async()=>{},
    camera:async()=>{camera++;return {entityId:'camera.dining_area_main_camera_tc70_mainstream',name:'Dining camera'};},
    message:async()=>{message++;return 'Home Assistant accepted the notification to your phone.';}});
  const run=(intent:string,id=intent,source='microphone')=>actions.handle({...event(id),source,application:{accepted:true,command:{intent,slots:intent==='CALL'?{contact_id:'contact.home'}:{}}}});
  const call=await run('CALL');assert.equal(call?.camera.entityId,'camera.dining_area_main_camera_tc70_mainstream');
  assert.match((await run('MESSAGE'))?.text,/accepted the notification/);
  await run('CALL','wav','wav');assert.equal(camera,1);assert.equal(message,1);actions.close();
});

test('touch sync cancels voice alerts and completed reminders stay completed in voice lists',async t=>{
  t.mock.timers.enable({apis:['setTimeout','Date']});const alerts:any[]=[];
  const actions=new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}},{light:async()=>{},spotify:async()=>{},song:async()=>{},notify:e=>alerts.push(e)});
  await actions.handle({...event('timer'),application:{accepted:true,command:{intent:'TIMER',slots:{duration_seconds:1}}}});
  actions.syncLocal({timer:null});t.mock.timers.tick(1000);assert.equal(alerts.length,0);
  actions.syncLocal({reminders:[{id:'touch',text:'Study',done:true}]});
  const result=await actions.handle({...event('list'),application:{accepted:true,command:{intent:'LIST_REMINDERS',slots:{}}}});
  assert.match(result?.text,/no reminders/);assert.equal(result?.reminders[0].done,true);actions.close();
});
