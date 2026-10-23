/** Actual pinned ONNX -> resident worker -> HTTP gateway -> UI reducer parity.
 * Run after the ordinary SpeakerUI build. No microphone or providers are used.
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID, createHash } from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import assert from 'node:assert/strict';
import {validateProof, verifyWavBytes} from './optionb-parity-inputs.mjs';
import { PythonRuntime } from '../dist-server/server/inference/python-runtime.js';
import { createApp } from '../dist-server/server/app.js';
import { readConfig } from '../dist-server/server/config.js';
import { applyRuntimeEvent } from '../src/ambient/runtime-events.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const output = process.argv[2];
if (!output) throw new Error('Pass a new receipt filename under .cache/.');
const python=process.env.OPTIONB_PYTHON || resolve(root, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
const run=promisify(execFile);
const inputProof=async()=>{
  // Reuse the actual pinned canonical-manifest and WAV/decoded-PCM verifier.
  // No graph inference, training, selection or fixture responses occur here.
  const {stdout}=await run(python,['-m','scripts.test.optionb_release_check','--preflight'],
    {cwd:root,maxBuffer:4*1024*1024,windowsHide:true});
  return JSON.parse(stdout);
};
const proof=await inputProof();
const rows=validateProof(proof);
const bundle = resolve(root, proof.bundle);
const previous = process.env.SMART_SPEAKER_WORKER_MODULE;
process.env.SMART_SPEAKER_WORKER_MODULE = 'scripts.runtime.optionb_worker';
const config = readConfig({INFERENCE_MODE:'onnx', ACTIONS_ENABLED:'false',
  SMART_SPEAKER_ROOT:root, SMART_SPEAKER_BUNDLE:bundle, SMART_SPEAKER_THREADS:'1',
  SMART_SPEAKER_PYTHON:python,
  SMART_SPEAKER_TOKEN:randomUUID(), APP_ORIGIN:'http://127.0.0.1:3000',
  DATA_DIR:resolve(root,'.cache/speaker-optionb-release-v1/ui-data'),
  LOCAL_MEDIA_DIR:resolve(root,'.cache/speaker-optionb-release-v1/no-media'),
}, root);
const runtime = new PythonRuntime(config.runtime);
const events = [];
const model = {history:[], request:0};
runtime.on('event', event => { events.push(event); applyRuntimeEvent(model, event, () => {}); });
const app = await createApp(config, {runtimeFactory:() => runtime, fetcher:async () => assert.fail('No outbound provider request is allowed')});
try {
  if (!runtime.status().loaded) await new Promise((accept, reject) => {
    const timeout = setTimeout(() => reject(new Error('Real worker readiness timed out')), 60000);
    const ready = event => {
      if (event.event === 'ready') { clearTimeout(timeout); runtime.off('event', ready); accept(); }
      if (event.event === 'error') { clearTimeout(timeout); runtime.off('event', ready); reject(new Error(event.message)); }
    };
    runtime.on('event', ready);
  });
  const health = (await app.inject({url:'/api/health',headers:{host:'127.0.0.1:3000'}})).json();
  assert.equal(health.runtime.loaded, true);
  assert.equal(health.runtime.parameters, 258912);
  assert.equal(health.runtime.selected_epoch, 44);
  assert.equal(health.runtime.actions_enabled, false);
  assert.equal(health.runtime.model_qualified, false);
  const decisions = [];
  for (const row of rows) {
    const before = events.length;
    const payload=verifyWavBytes(await readFile(row.audio),row);
    const response = await app.inject({method:'POST',url:'/api/runtime/wav',headers:{host:'127.0.0.1:3000',origin:config.origin,'content-type':'audio/wav'},payload});
    assert.equal(response.statusCode, 200, response.body);
    const commands = events.slice(before).filter(event => event.event === 'command');
    assert.equal(commands.length, 1);
    const event = commands[0];
    assert.deepEqual(event.output, row.expected_output);
    assert.deepEqual(event.application, row.expected_application);
    assert.equal(event.actions_enabled, false);
    assert.equal(event.qualified_output.accepted, false);
    assert.deepEqual(model.runtimeResult.application, event.application);
    assert.deepEqual(model.event.raw.output, event.output);
    assert.equal(model.event.actionsEnabled, false);
    if (event.output.accepted) {
      assert.deepEqual(model.event.result.slots, event.output.command.slots);
      assert.equal(model.event.result.modelIntent, event.output.command.intent);
    } else assert.equal(model.event.result.intent, 'rejected');
    decisions.push({id:row.id,output:event.output,application:event.application});
  }
  assert.equal(decisions.length, proof.expected_count);
  assert.equal(new Set(decisions.map(row => row.id)).size, proof.expected_count);
  assert.deepEqual(await inputProof(),proof,'Frozen input/reference/audio identities changed during replay');
  const receipt = {format:'optionb-native-speakerui-parity-v2',status:'verified',
    source_commit:proof.source_commit, parameters:258912, selected_epoch:44,
    protocol_sha256:proof.protocol_sha256,expected_keys_sha256:proof.expected_keys_sha256,
    independently_pinned_canonical_cohort:true,consumed_bindings_verified:proof.bindings_verified,
    original_wav_and_decoded_pcm_verified:true,
    path:'FP32 ONNX -> JSONL worker -> compiled gateway HTTP WAV route -> actual UI event reducer',
    original_human_holdout_rows:decisions.length, accepted:decisions.filter(row => row.output.accepted).length,
    rejected:decisions.filter(row => !row.output.accepted).length,
    exact_output_application_and_slot_parity:true,actions_enabled:false,qualified:false,
    decisions_sha256:createHash('sha256').update(JSON.stringify(decisions)).digest('hex'),
    physical_pi:false,live_microphone:false,browser_layout_test:false,
  };
  const target = resolve(root, output);
  assert.ok(target.startsWith(resolve(root,'.cache') + (process.platform === 'win32' ? '\\' : '/')), 'Receipts belong under .cache/');
  await mkdir(dirname(target),{recursive:true});
  await writeFile(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(receipt,null,2));
} finally {
  await app.close();
  if (previous === undefined) delete process.env.SMART_SPEAKER_WORKER_MODULE;
  else process.env.SMART_SPEAKER_WORKER_MODULE = previous;
}
