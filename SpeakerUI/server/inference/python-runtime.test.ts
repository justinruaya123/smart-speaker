import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, writeFile, chmod, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PythonRuntime } from './python-runtime.js';

const keys = ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'];

async function childEnvironment(values: Record<string,string | undefined>, threads: number) {
  const saved = Object.fromEntries(keys.map(key=>[key,process.env[key]]));
  const directory = await mkdtemp(join(tmpdir(),'iris-worker-launch-'));
  const executable = join(directory,'worker-env');
  let worker: PythonRuntime | undefined;
  try {
    for (const key of keys) {
      if(values[key]===undefined)delete process.env[key];
      else process.env[key]=values[key];
    }
    // An actual process speaks the worker protocol and reports its inherited
    // environment before any numeric library import; no model or mic is used.
    await writeFile(executable,`#!${process.execPath}\nimport { createInterface } from 'node:readline';
process.stdout.write(JSON.stringify({event:'ready',loaded:true,thread_environment:Object.fromEntries(${JSON.stringify(keys)}.map(key=>[key,process.env[key]??null])),args:process.argv.slice(2)})+'\\n');
for await (const line of createInterface({input:process.stdin}))if(JSON.parse(line).op==='shutdown')process.exit(0);
`);
    await chmod(executable,0o700);
    worker = new PythonRuntime({python:executable,root:directory,bundle:directory,threads,profile:'benchmark-compatible'});
    const ready = await new Promise<Record<string,any>>((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Environment child did not become ready')),5000);
      worker!.once('event',event=>{clearTimeout(timer);event.event==='ready'?resolve(event):reject(new Error(JSON.stringify(event)));});
    });
    for(const key of keys)assert.equal(process.env[key],values[key],'Launching a child must not change the server environment');
    return ready;
  } finally {
    await worker?.close();
    for (const key of keys) {
      if(saved[key]===undefined)delete process.env[key];
      else process.env[key]=saved[key];
    }
    await rm(directory,{recursive:true,force:true});
  }
}

// Pi/Linux execute the same shebang child used here; Windows does not execute
// shebang files directly through spawn(), so this OS-level check is skipped.
test('resident worker bounds unset frontend libraries to its configured engine threads before launch', {skip:process.platform==='win32'}, async()=>{
  const ready=await childEnvironment({OMP_NUM_THREADS:'',MKL_NUM_THREADS:'   '},3);
  assert.deepEqual(ready.thread_environment,Object.fromEntries(keys.map(key=>[key,'3'])));
  assert.deepEqual(ready.args.slice(-4),['--threads','3','--profile','benchmark-compatible']);
});

test('resident worker respects explicit nonempty numeric-library thread overrides', {skip:process.platform==='win32'}, async()=>{
  const overrides=Object.fromEntries(keys.map((key,index)=>[key,String(index+1)]));
  const ready=await childEnvironment(overrides,2);
  assert.deepEqual(ready.thread_environment,overrides);
});
