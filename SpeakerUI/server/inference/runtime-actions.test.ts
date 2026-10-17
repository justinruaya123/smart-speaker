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
