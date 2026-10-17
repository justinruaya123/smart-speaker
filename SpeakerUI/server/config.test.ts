import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readConfig } from './config.js';
import { createApp } from './app.js';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

test('action configuration normalizes missing maps and accepts explicit HA capabilities and light groups', () => {
  const bindings = { lights: { 'light.all': ['light.a', 'light.b'] }, climate: { default: 'climate.room' },
    weather: { default: 'weather.home' }, media_player: { provider: 'home_assistant', entity_id: 'media_player.room' },
    contacts: { 'contact.home': 'script.call_home' }, messages: { default: 'script.preset' } };
  const cfg = readConfig({ SMART_SPEAKER_BINDINGS: JSON.stringify(bindings),
    HA_ENTITY_IDS: 'light.a,light.b,climate.room,weather.home,media_player.room,script.call_home,script.preset' }, '/tmp');
  assert.deepEqual(cfg.bindings.lights['light.all'], ['light.a', 'light.b']);
  assert.deepEqual(cfg.bindings.songs, {});
  assert.equal((cfg.bindings as any).media_player.provider, 'home_assistant');
});

test('provider selection and every binding fail closed on malformed configuration', () => {
  for (const bindings of [{ lights: {}, songs: {}, media_player: { provider: 'auto' } },
    { lights: { 'light.all': [] } }, { contacts: { 'contact.home': 'light.desk' } },
    { climate: { default: 'climate.room' }, extra: {} }, { songs: [], lights: {} }]) {
    assert.throws(() => readConfig({ SMART_SPEAKER_BINDINGS: JSON.stringify(bindings) }, '/tmp'), /binding/i);
  }
  assert.deepEqual((readConfig({}, '/tmp').bindings as any).media_player, {});
});

test('local clock uses a validated configured IANA timezone', () => {
  assert.equal(readConfig({}, '/tmp').timezone, 'Asia/Manila');
  assert.equal(readConfig({ TIMEZONE: 'Europe/London' }, '/tmp').timezone, 'Europe/London');
  for (const zone of ['Asia/Mars', '+08:00']) assert.throws(() => readConfig({ TIMEZONE: zone }, '/tmp'), /TIMEZONE/);
});

test('untouched source and emitted release defaults start the gateway with media unconfigured', async () => {
  const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
  const directory=await mkdtemp(join(tmpdir(),'iris-fresh-defaults-'));
  const configuredPython=join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
  const python=existsSync(configuredPython)?configuredPython:process.platform==='win32'?'python':'python3';
  try {
    // Read the exact packaged template literal without fitting/loading acoustic models.
    // Each example then follows the real first-start initialization and env exporter.
    const exported=JSON.parse(execFileSync(python,['-c',`
import ast,json,sys
from pathlib import Path
from scripts.runtime.speaker_config import initialize_config,resolve_config
root=Path(sys.argv[1]);temporary=Path(sys.argv[2])
templates=[('root',(root/'config.example.toml').read_bytes()),('runtime',(root/'scripts/runtime/config.example.toml').read_bytes())]
module=ast.parse((root/'scripts/mvp_delivery.py').read_text())
emitted=[node.value.value for node in ast.walk(module) if isinstance(node,ast.Assign) and
 isinstance(node.value,ast.Constant) and isinstance(node.value.value,bytes) and
 any(isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='entries' and
 isinstance(target.slice,ast.Constant) and target.slice.value=='config.example.toml' for target in node.targets)]
assert len(emitted)==2,'Expected candidate and release templates'
templates.append(('emitted-release',emitted[-1]))
values=[]
for name,template in templates:
 folder=temporary/name;folder.mkdir();(folder/'config.example.toml').write_bytes(template)
 initialize_config(folder);cfg,env=resolve_config(folder,environ={})
 values.append({'name':name,'env':env})
print(json.dumps(values))
`,root,directory],{cwd:root,encoding:'utf8'}));
    for(const entry of exported){
      // Mock inference keeps this startup regression independent of model assets;
      // all provider bindings come from the untouched template, without an override.
      const config=readConfig({...entry.env,INFERENCE_MODE:'mock'},join(directory,entry.name));
      const app=await createApp(config);
      try{
        await app.listen({host:'127.0.0.1',port:0});
        const health=await app.inject({url:'/api/health',headers:{host:new URL(config.origin).host}});
        assert.equal(health.statusCode,200,entry.name);
        assert.equal(health.json().inference,'mock',entry.name);
        assert.deepEqual(config.bindings.media_player,{},entry.name);
        assert.equal(config.ha.enabled,false,entry.name);
      }finally{await app.close();}
    }
  }finally{await rm(directory,{recursive:true,force:true});}
});
