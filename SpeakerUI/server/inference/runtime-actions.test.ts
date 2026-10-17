import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RuntimeActions } from './runtime-actions.js';
import { ActionStore } from './action-store.js';
import { IntegrationError } from '../integrations/http.js';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

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
test('worker action-disable overrides enabled server configuration for providers and local commands',async()=>{
  const actions=new RuntimeActions({enabled:true,bindings:{lights:{'light.kitchen':'light.actual'},songs:{}}},
    {light:async()=>assert.fail('action'),spotify:async()=>assert.fail('action'),song:async()=>assert.fail('action')});
  const light=await actions.handle({...event('candidate-light'),actions_enabled:false});
  assert.equal(light?.status,'disabled');
  const timer=await actions.handle({...event('candidate-timer'),actions_enabled:false,
    application:{accepted:true,command:{intent:'TIMER',slots:{duration_seconds:123}}}});
  assert.equal(timer?.status,'disabled');assert.equal(timer?.deadline,undefined);
});
test('local timers preserve exact seconds and server deadline',async()=>{
  const a=new RuntimeActions({enabled:true,bindings:{lights:{},songs:{}}},{light:async()=>{},spotify:async()=>{},song:async()=>{}});
  const e={...event(),application:{accepted:true,command:{intent:'TIMER',slots:{duration_seconds:123}}}};
  const result=await a.handle(e);assert.equal(result?.status,'local');assert.equal(result?.command.slots.duration_seconds,123);
  assert.ok(result!.deadline>=Date.now()+122000);
});

test('tentative and explicitly nonfinal command events never consume the final action ID', async () => {
  let writes = 0;
  const actions = new RuntimeActions({ enabled: true, bindings: { lights: { 'light.kitchen': 'light.actual' }, songs: {} } },
    { light: async () => { writes++; }, spotify: async () => {}, song: async () => {} });
  assert.equal(await actions.handle({ ...event(), event: 'tentative', is_final: false }), undefined);
  assert.equal(await actions.handle({ ...event(), is_final: false }), undefined);
  assert.equal(writes, 0);
  await actions.handle(event());
  assert.equal(writes, 1);
});

test('canonical slots reject extra, malformed and wrong-role IDs before providers', async () => {
  const actions = new RuntimeActions({ enabled: true, bindings: { lights: { 'light.kitchen': 'light.actual' }, songs: {} } },
    { light: async () => assert.fail('provider'), spotify: async () => {}, song: async () => {} });
  const result = await actions.handle({ ...event(), application: { accepted: true,
    command: { intent: 'LIGHT_ON', slots: { light_id: 'light.kitchen', percent: 10 } } } });
  assert.equal(result?.status, 'rejected');
});

test('HA climate, weather, finite contacts and preset messages use explicit bindings', async () => {
  const calls: any[] = [];
  const actions = new RuntimeActions({ enabled: true, bindings: { lights: {}, songs: {},
    climate: { default: 'climate.room' }, weather: { default: 'weather.home' },
    contacts: { 'contact.mom': 'script.call_mom' }, messages: { default: 'script.preset' } } } as any,
    { light: async () => {}, spotify: async () => {}, song: async () => {},
      climate: async (...args: any[]) => calls.push(['climate', ...args]),
      weather: async (...args: any[]) => { calls.push(['weather', ...args]); return { state: 'sunny', temperature: 28, unit: '°C' }; },
      script: async (...args: any[]) => calls.push(['script', ...args]) } as any);
  for (const [index, command] of [{ intent: 'TEMPERATURE', slots: { degrees: 22, unit: 'C' } },
    { intent: 'WEATHER', slots: {} }, { intent: 'CALL', slots: { contact_id: 'contact.mom' } },
    { intent: 'MESSAGE', slots: {} }].entries()) {
    const result = await actions.handle({ ...event('extra-' + index), application: { accepted: true, command } });
    assert.ok(['succeeded', 'observed'].includes(result!.status));
  }
  assert.deepEqual(calls, [['climate', 'climate.room', 22], ['weather', 'weather.home'], ['script', 'script.call_mom'], ['script', 'script.preset']]);
});

test('owner-protected voice requires a passed check and benchmark mode is inert even with a true event flag', async () => {
  let calls=0;
  const actions=new RuntimeActions({enabled:true,bindings:{lights:{'light.kitchen':'light.actual'},songs:{}}},
    {light:async()=>{calls++;},spotify:async()=>{},song:async()=>{}});
  for(const [index,reason] of ['mismatch','overlap','insufficient_speech'].entries()) {
    const result=await actions.handle({...event('reject-'+index),actions_enabled:true,profile:'owner-protected',owner_result:{accepted:false,reason}});
    assert.equal(result?.status,'rejected');assert.equal(result?.detail,reason);
  }
  assert.equal((await actions.handle({...event('benchmark'),actions_enabled:true,profile:'benchmark-compatible'}))?.status,'disabled');
  assert.equal(calls,0);
  await actions.handle({...event('owner'),profile:'owner-protected',owner_result:{accepted:true},actions_enabled:true});
  assert.equal(calls,1);
  const configured=new RuntimeActions({enabled:true,bindings:{lights:{'light.kitchen':'light.actual'},songs:{}},profile:'owner-protected'},
    {light:async()=>assert.fail('missing owner result'),spotify:async()=>{},song:async()=>{}});
  assert.equal((await configured.handle(event('missing-owner')))?.status,'rejected');
});

test('an unknown provider result remains deduplicated after the server journal is reopened', async () => {
  const directory=await mkdtemp(join(tmpdir(),'iris-dispatch-'));let calls=0;
  const cfg={enabled:true,bindings:{lights:{'light.kitchen':'light.actual'},songs:{}}};
  const providers={light:async()=>{calls++;throw new IntegrationError('unavailable','Outcome unknown',true);},spotify:async()=>{},song:async()=>{}};
  try {
    const first=new RuntimeActions(cfg,providers,await ActionStore.open(directory));
    assert.equal((await first.handle(event()))?.status,'unknown');
    const second=new RuntimeActions(cfg,providers,await ActionStore.open(directory));
    assert.equal(await second.handle(event()),undefined);assert.equal(calls,1);
  } finally {await rm(directory,{recursive:true,force:true});}
});

test('TIME follows configured local timezone while ALARM preserves its command timezone and deadline', async () => {
  const now=Date.UTC(2026,9,17,0,0);
  const actions=new RuntimeActions({enabled:true,bindings:{lights:{},songs:{}},timezone:'UTC'},
    {light:async()=>{},spotify:async()=>{},song:async()=>{}},ActionStore.memory(()=>now),()=>now);
  const clock=await actions.touch('clock',{intent:'TIME',slots:{}},true);
  assert.equal(clock?.timezone,'UTC');assert.match(clock!.detail,/12:00:00/);
  const alarm=await actions.touch('alarm',{intent:'ALARM',slots:{time:'09:00',timezone:'Asia/Manila'}},true);
  assert.equal(alarm?.deadline,Date.UTC(2026,9,17,1,0));
  assert.equal(alarm?.job.timezone,'Asia/Manila');
});
