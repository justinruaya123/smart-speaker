import { test, type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { setImmediate as nextTurn } from 'node:timers/promises';
import { createApp } from '../app.js';
import { readConfig } from '../config.js';
import type { PythonRuntime } from './python-runtime.js';
import { RuntimeActions } from './runtime-actions.js';

function legacyEvent(intent:string, slots:Record<string,unknown> = {}, id = intent) {
  return {event:'command', session_id:'legacy', utterance_id:id, source:'microphone', actions_enabled:false,
    application:{accepted:true, command:{intent, slots}}};
}

for (const intent of ['TIME', 'WEATHER']) {
  test(`legacy actions_enabled=false retains ${intent} with device actions disabled`, async () => {
    let weatherCalls = 0;
    const actions = new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}}, {
      light:async () => assert.fail('device action'), spotify:async () => assert.fail('device action'),
      song:async () => assert.fail('device action'),
      weather:async () => { weatherCalls++; return {text:'Fixture weather answer',temperature:28}; },
    });
    try {
      const event = legacyEvent(intent);
      const result = await actions.handle(event);
      assert.equal(result?.status, intent === 'TIME' ? 'local' : 'succeeded');
      assert.match(result?.text, intent === 'TIME' ? /The time is .+ in Manila\./ : /Fixture weather answer/);
      assert.equal(await actions.handle(event), undefined);
      assert.equal(weatherCalls, intent === 'WEATHER' ? 1 : 0);
      assert.equal((await actions.handle({...event,utterance_id:'wav',source:'wav'}))?.status, 'preview');
      assert.equal(weatherCalls, intent === 'WEATHER' ? 1 : 0);
    } finally { actions.close(); }
  });
}

test('legacy actions_enabled=false schedules real local timer and alarm notifications', async (t:TestContext) => {
  t.mock.timers.enable({apis:['setTimeout','Date'],now:Date.UTC(2026,0,1)});
  const notifications:Record<string,any>[] = [];
  const actions = new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}}, {
    light:async () => assert.fail('device action'), spotify:async () => assert.fail('device action'),
    song:async () => assert.fail('device action'), notify:event => notifications.push(event),
  });
  try {
    const timer = legacyEvent('TIMER',{duration_seconds:2});
    const timerResult = await actions.handle(timer);
    assert.equal(timerResult?.status, 'local');
    assert.equal(timerResult?.deadline, Date.now()+2000);
    assert.equal(await actions.handle(timer), undefined);
    const alarmResult = await actions.handle(legacyEvent('ALARM',{time:'08:01',timezone:'Asia/Manila'}));
    assert.equal(alarmResult?.status, 'local');
    assert.equal(alarmResult?.deadline, Date.now()+60000);
    t.mock.timers.tick(2000);
    assert.deepEqual(notifications.map(event => event.kind), ['timer']);
    t.mock.timers.tick(58000);
    assert.deepEqual(notifications.map(event => event.kind), ['timer','alarm']);
  } finally { actions.close(); }
});

test('legacy actions_enabled=false creates and lists session reminders', async () => {
  const actions = new RuntimeActions({enabled:false,bindings:{lights:{},songs:{}}}, {
    light:async () => assert.fail('device action'), spotify:async () => assert.fail('device action'),
    song:async () => assert.fail('device action'),
  });
  try {
    const create = legacyEvent('CREATE_REMINDER',{task_id:'task.study'});
    const created = await actions.handle(create);
    assert.equal(created?.status, 'local');
    assert.equal(await actions.handle(create), undefined);
    const listed = await actions.handle(legacyEvent('LIST_REMINDERS'));
    assert.equal(listed?.status, 'local');
    assert.deepEqual(listed?.reminders, [{id:'CREATE_REMINDER',text:'Study',done:false}]);
  } finally { actions.close(); }
});

for (const enabled of [false, true]) {
  test(`legacy actions_enabled=false preserves configured device gate (${enabled})`, async () => {
    const calls:unknown[] = [];
    const actions = new RuntimeActions({enabled,bindings:{lights:{'light.all':'light.fixture'},songs:{}}}, {
      light:async command => { calls.push(command); }, spotify:async () => assert.fail('unexpected provider'),
      song:async () => assert.fail('unexpected provider'),
    });
    try {
      const result = await actions.handle(legacyEvent('LIGHT_ON',{light_id:'light.all'}));
      assert.equal(result?.status, enabled ? 'succeeded' : 'disabled');
      assert.deepEqual(calls, enabled ? [{entityId:'light.fixture',action:'on'}] : []);
    } finally { actions.close(); }
  });
}

test('explicit atomic preview boundary precedes every local and external action', async (t:TestContext) => {
  t.mock.timers.enable({apis:['setTimeout','Date']});
  const unexpected = async () => assert.fail('preview must not call a provider');
  const actions = new RuntimeActions({enabled:true,bindings:{lights:{'light.all':'light.fixture'},songs:{}}}, {
    light:unexpected, spotify:unexpected, song:unexpected, weather:unexpected, camera:unexpected, message:unexpected,
    temperature:unexpected, notify:() => assert.fail('preview must not schedule an alert'),
  });
  try {
    for (const [intent, slots] of [
      ['TIME',{}], ['WEATHER',{}], ['TIMER',{duration_seconds:10}],
      ['ALARM',{time:'07:00',timezone:'Asia/Manila'}], ['CREATE_REMINDER',{task_id:'task.study'}],
      ['LIST_REMINDERS',{}], ['CALL',{contact_id:'contact.home'}], ['MESSAGE',{}],
      ['LIGHT_ON',{light_id:'light.all'}], ['PLAY_MUSIC',{}], ['TEMPERATURE',{degrees:22,unit:'C'}],
    ] as const) {
      const event = {event:'command',session_id:'atomic',utterance_id:intent,source:'microphone',
        inference_contract:'audio_to_atomic_complete_optionb_command', actions_enabled:false, model_qualified:false,
        application:{accepted:true,mode:'joint_complete_command',command:{intent,slots}}};
      const result = await actions.handle(event);
      assert.equal(result?.status, 'preview', intent);
      assert.deepEqual(result?.command, {intent,slots});
      assert.equal(await actions.handle(event), undefined, 'duplicate must still be suppressed');
    }
    t.mock.timers.tick(86400000);
    const listed = await actions.handle(legacyEvent('LIST_REMINDERS',{},'after-preview'));
    assert.equal(listed?.status, 'local');
    assert.deepEqual(listed?.reminders, [], 'preview must not mutate local reminders');
  } finally { actions.close(); }
});

for (const atomic of [true, false]) {
  test(atomic
    ? 'OptionB gateway preview stays silent with inherited playback=true and preserves capture'
    : 'legacy gateway local response retains playback with inherited playback=true', async (t:TestContext) => {
    const original = process.env.SUBMISSION_RESPONSE_PLAYBACK;
    process.env.SUBMISSION_RESPONSE_PLAYBACK = 'true';
    t.after(() => {
      if (original === undefined) delete process.env.SUBMISSION_RESPONSE_PLAYBACK;
      else process.env.SUBMISSION_RESPONSE_PLAYBACK = original;
    });
    const requests:{op:string;text:string}[] = [];
    const events:Record<string,any>[] = [], actions:Record<string,any>[] = [];
    class Worker extends EventEmitter {
      capturing = true;
      state = 'listening';
      status() { return {loaded:true,state:this.state}; }
      current() { return undefined; }
      async close() {}
      async request(op:string, payload:{text:string}) {
        requests.push({op,text:payload.text});
        if (!atomic && op === 'speak') return {spoken:true};
        // OptionB rejects unsupported operations and stops its active capture.
        this.capturing = false;
        this.state = 'runtime_error';
        this.emit('event',{event:'state',state:this.state,message:'Unknown runtime operation'});
        throw new Error('Unknown runtime operation');
      }
    }
    const worker = new Worker();
    worker.on('event', event => events.push(event));
    worker.on('action', event => actions.push(event));
    let providerCalls = 0;
    const app = await createApp({
      ...readConfig({APP_ORIGIN:'http://127.0.0.1:3000',ACTIONS_ENABLED:'true'}, 'optionb-gateway-fixtures-missing'),
      runtime:{python:'unused',root:'unused',bundle:'unused',threads:1,token:'fixture'},
    }, {
      runtimeFactory:() => worker as unknown as PythonRuntime,
      fetcher:async () => { providerCalls++; throw new Error('Gateway regression must stay offline'); },
    });
    t.after(() => app.close());
    await app.ready();
    const event = {...legacyEvent('TIME'), ...(atomic ? {
      inference_contract:'audio_to_atomic_complete_optionb_command',model_qualified:false,
    } : {})};
    worker.emit('event', event);
    worker.emit('event', event);
    await nextTurn(); // Drain both gateway promise chains, including speech failures.

    const executions = actions.filter(event => event.event === 'execution');
    assert.equal(executions.length, 1, 'duplicate commands must not replay');
    const execution = executions[0]!.execution;
    assert.equal(requests.filter(request => request.op === 'speak').length, atomic ? 0 : 1);
    assert.deepEqual(requests, atomic ? [] : [{op:'speak',text:execution.text}]);
    assert.deepEqual(actions.filter(event => event.event === 'speech_error'), []);
    assert.deepEqual(events.filter(event => event.event === 'error' || event.event === 'state'), []);
    assert.equal(worker.capturing, true);
    assert.equal(worker.state, 'listening');
    assert.equal(execution.status, atomic ? 'preview' : 'local');
    assert.deepEqual(execution.command, event.application.command);
    if (!atomic) assert.match(execution.text, /The time is .+ in Manila\./);
    assert.equal(providerCalls, 0);
  });
}
