import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate as settle } from 'node:timers/promises';
import { RuntimeActions } from './runtime-actions.js';
import { ActionFeedback } from './feedback.js';
import { IntegrationError } from '../integrations/http.js';
import { EventEmitter } from 'node:events';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { setTimeout as delay } from 'node:timers/promises';
import { createApp } from '../app.js';
import { readConfig } from '../config.js';

function event(id = 'u') { return { event: 'command', session_id: 'session', utterance_id: id, source: 'microphone', is_final: true,
  actions_enabled: true, owner_result: { accepted: true }, application: { accepted: true, command: { intent: 'LIGHT_ON', slots: { light_id: 'light.all' } } } }; }
function deferred() { let resolve!: () => void; const promise = new Promise<void>(r => { resolve = r; }); return { resolve, promise }; }
const config = { enabled: true, publication: true, profile: 'owner-protected', bindings: { lights: {}, songs: {} } };

test('host intent playback finishes before dispatch, final receipt publishes before result playback, and duplicate phases are silent', async () => {
  const order: string[] = [], intent = deferred(), provider = deferred();
  const actions = new RuntimeActions(config, { light: async () => { order.push('provider'); await provider.promise; }, spotify: async () => {}, song: async () => {} });
  const lifecycle = new ActionFeedback(actions, { play: async (cue, id) => { assert.equal(id, 'u'); order.push(cue); if (cue === 'intent') await intent.promise; },
    publish: result => { order.push('receipt:' + result.status); } });
  const pending = lifecycle.run(event()); await settle();
  assert.deepEqual(order, ['intent']);
  intent.resolve(); await settle(); assert.deepEqual(order, ['intent', 'provider']);
  provider.resolve(); await pending;
  assert.deepEqual(order, ['intent', 'provider', 'receipt:succeeded', 'success']);
  await lifecycle.run(event());
  await lifecycle.complete(actions.store.receipt('session:u')!);
  assert.deepEqual(order, ['intent', 'provider', 'receipt:succeeded', 'success']);
});

test('WAV, fixtures, tentative, rejected, invalid, owner-rejected and disabled paths never play feedback or invoke providers', async () => {
  const actions = new RuntimeActions(config, { light: async () => assert.fail('provider'), spotify: async () => {}, song: async () => {} });
  const lifecycle = new ActionFeedback(actions, { play: async () => assert.fail('audio'), publish: () => {} });
  for (const [index, patch] of [{ source: 'wav' }, { source: 'fixture' }, { event: 'tentative', is_final: false }, { is_final: false },
    { actions_enabled: false }, { owner_result: { accepted: false } }, { application: { accepted: false, command: null } },
    { application: { accepted: true, command: { intent: 'LIGHT_ON', slots: {} } } }].entries())
    await lifecycle.run({ ...event(String(index)), ...patch });
});

test('an unknown dispatch produces failure once and audio failure cannot retry or synthesize the action', async () => {
  let calls = 0;
  const order: string[] = [], errors: string[] = [];
  const actions = new RuntimeActions(config, { light: async () => { calls++; throw new IntegrationError('unavailable', 'Outcome unknown.', true); }, spotify: async () => {}, song: async () => {} });
  const lifecycle = new ActionFeedback(actions, { play: async cue => { order.push(cue); throw new Error('Output unavailable'); }, publish: result => { order.push(result.status); },
    audioError: message => { errors.push(message); } });
  await lifecycle.run(event()); await lifecycle.run(event());
  assert.deepEqual(order, ['intent', 'unknown', 'failure']); assert.equal(calls, 1); assert.equal(errors.length, 2);
  assert.deepEqual(actions.syntheticState(), {});
});

test('local requested playback waits for its real final player outcome and never beeps on history replay', async () => {
  const order: string[] = [];
  const actions = new RuntimeActions({ ...config, bindings: { lights: {}, songs: {}, media_player: { provider: 'local' as const } } },
    { light: async () => {}, spotify: async () => {}, song: async () => {}, local: async () => ({ action: 'PLAY_MUSIC', url: '/api/media/local/id' }) });
  const lifecycle = new ActionFeedback(actions, { play: async cue => { order.push(cue); }, publish: result => { order.push(result.status); } });
  await lifecycle.run({ ...event(), application: { accepted: true, command: { intent: 'PLAY_MUSIC', slots: {} } } });
  assert.deepEqual(order, ['intent', 'requested']);
  const final = await actions.store.finish('session:u', { status: 'succeeded', detail: 'Real browser player confirmation.' });
  await lifecycle.complete(final); await lifecycle.complete(final);
  assert.deepEqual(order, ['intent', 'requested', 'success']);
  await lifecycle.complete({ ...final, id: 'previous-session:u' });
  assert.deepEqual(order, ['intent', 'requested', 'success']);
});

async function until(predicate: () => boolean) {
  for (let count = 0; count < 200; count++) { if (predicate()) return; await delay(5); }
  assert.ok(predicate(), 'Expected actual gateway event did not arrive');
}
class OfflineHost extends EventEmitter {
  cues: any[] = [];
  status() { return { loaded:true,actions_enabled:true,capabilities:{feedback:true} }; }
  current() { return { event:'ready',...this.status() }; }
  async request(op: string, payload?: any) { this.cues.push([op,payload]); return { ok:true }; }
  async close() {}
}
const headers = { host:'127.0.0.1:3000',origin:'http://127.0.0.1:3000' };
function gatewayConfig(directory: string, extra: Record<string,string> = {}) {
  return readConfig({ IRIS_PUBLICATION:'true',ACTIONS_ENABLED:'true',INFERENCE_MODE:'onnx',SMART_SPEAKER_PYTHON:'offline-never-spawn',SMART_SPEAKER_ROOT:'.',
    SMART_SPEAKER_BUNDLE:'.',SMART_SPEAKER_TOKEN:'offline-fixture',DATA_DIR:directory,...extra },directory);
}

test('actual gateway awaits worker playback replies and HA completion once across two browsers and duplicate finals', async () => {
  const directory=await mkdtemp(join(tmpdir(),'iris-feedback-gateway-'));
  const host=new OfflineHost(),intent=deferred(),provider=deferred(),order:string[]=[];
  host.request=async(op,payload)=>{host.cues.push([op,payload]);order.push(payload.cue_id);if(payload.cue_id==='intent')await intent.promise;return {ok:true};};
  host.on('action',e=>order.push('receipt:'+e.execution.status));
  const socket=Object.assign(new EventEmitter(),{send(){},close(){},terminate(){},ping(){}});
  const app=await createApp(gatewayConfig(directory,{HA_ENABLED:'true',HA_URL:'http://offline.invalid',HA_TOKEN:'offline-fixture',HA_ENTITY_IDS:'light.hue_color_lamp_1'}),{
    runtimeFactory:()=>host,haSocketFactory:()=>socket,fetcher:async(url,init)=>{
      if(String(url).includes('/api/states/'))return Response.json({entity_id:'light.hue_color_lamp_1',state:'on',attributes:{}});
      assert.equal(init?.method,'POST');order.push('provider');await provider.promise;return Response.json([]);
    }});
  await app.ready();
  const clients:any[]=[],received:any[][]=[[],[]];
  try {
    for(let index=0;index<2;index++)clients.push(await app.injectWS('/ws/actions',{headers},{onInit(s){s.on('message',b=>received[index]!.push(JSON.parse(b.toString())));}}));
    host.emit('event',event());host.emit('event',event());
    await until(()=>order.length>0);assert.deepEqual(order,['intent']);
    intent.resolve();await until(()=>order.includes('provider'));assert.deepEqual(order,['intent','provider']);
    provider.resolve();await until(()=>order.includes('success'));
    assert.deepEqual(order,['intent','provider','receipt:succeeded','success']);
    assert.deepEqual(host.cues,[['feedback',{cue_id:'intent',utterance_id:'u'}],['feedback',{cue_id:'success',utterance_id:'u'}]]);
    for(const messages of received)await until(()=>messages.filter(e=>e.event==='execution'&&e.execution.status==='succeeded').length===1);
    for(const patch of [{source:'wav'},{event:'tentative',is_final:false},{owner_result:{accepted:false}},{actions_enabled:false}])host.emit('event',{...event(JSON.stringify(patch)),...patch});
    await delay(25);assert.equal(host.cues.length,2);
    assert.equal((await app.inject({method:'POST',url:'/api/runtime/feedback',headers,payload:{cue_id:'success'}})).statusCode,404);
  } finally {intent.resolve();provider.resolve();for(const client of clients)client.terminate();await app.close();await rm(directory,{recursive:true,force:true});}
});

test('actual gateway local-player confirmation publishes its final receipt before the host result cue',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'iris-feedback-player-'));
  await writeFile(join(directory,'Fixture.mp3'),'No playback: offline local-player contract fixture.');
  const host=new OfflineHost(),order:string[]=[];
  host.request=async(op,payload)=>{host.cues.push([op,payload]);order.push(payload.cue_id);return {ok:true};};
  host.on('action',e=>order.push('receipt:'+e.execution.status));
  const cfg=gatewayConfig(directory,{LOCAL_MEDIA_DIR:directory,SMART_SPEAKER_BINDINGS:JSON.stringify({media_player:{provider:'local'}})});
  const app=await createApp(cfg,{runtimeFactory:()=>host,fetcher:async()=>assert.fail('No external I/O')});
  await app.ready();
  const received:any[]=[];
  const ws=await app.injectWS('/ws/actions',{headers},{onInit(s){s.on('message',b=>received.push(JSON.parse(b.toString())));}});
  try {
    const search=(await app.inject({url:'/api/media/search?q=Fixture',headers})).json();cfg.bindings.songs['song.fixture']=search.tracks[0].id;
    host.emit('event',{...event(),application:{accepted:true,command:{intent:'PLAY_MUSIC',slots:{song_id:'song.fixture'}}}});
    await until(()=>order.includes('receipt:requested'));
    assert.deepEqual(order,['intent','receipt:requested']);
    const owner=received.find(e=>e.event==='action_state').client_id;
    const responses=await Promise.all([0,1].map(()=>app.inject({method:'POST',url:'/api/actions/local-receipt',headers,payload:{id:'session:u',client_id:owner,status:'succeeded'}})));
    assert.deepEqual(responses.map(r=>r.statusCode).sort(),[200,409]);
    assert.equal(responses.find(r=>r.statusCode===200)!.json().session_id,'session','Final local receipts preserve their source session');
    await until(()=>order.includes('success'));
    assert.deepEqual(order,['intent','receipt:requested','receipt:succeeded','success']);
    assert.equal((await app.inject({method:'POST',url:'/api/actions/local-receipt',headers,payload:{id:'session:u',client_id:owner,status:'succeeded'}})).statusCode,409);
    assert.equal(host.cues.length,2);
  }finally{ws.terminate();await app.close();await rm(directory,{recursive:true,force:true});}
});
