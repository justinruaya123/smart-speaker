// Compiled Iris + actual gateway/ActionStore; explicitly offline provider/model/audio fixtures.
// This does NOT test acoustic recognition, live microphones, speakers or physical devices.
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdir, mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { chromium } from 'playwright';
import { createApp } from '../dist-server/server/app.js';
import { readConfig } from '../dist-server/server/config.js';

const output = resolve(process.argv[2] || '../.cache/iris-publication-v1/ui');
await mkdir(output,{recursive:true});
const inventory=JSON.parse(await readFile(new URL('../../submission/models/optionb-tiny/commands.json',import.meta.url),'utf8')).slice(1);
const groups=[['Music',['PLAY_MUSIC']],['Questions',['WEATHER','TIME']],['Lights',['LIGHT_ON','LIGHT_OFF']],['Dim / color',['BRIGHTNESS','COLOR']],
  ['Timer',['TIMER']],['Alarm',['ALARM']],['Temperature',['TEMPERATURE']],['Media',['PAUSE','STOP','NEXT','VOLUME']],
  ['Reminders',['CREATE_REMINDER','LIST_REMINDERS']],['Calls / messages',['CALL','MESSAGE']]];
assert.equal(inventory.length,31);
for(const command of inventory)assert.equal(groups.filter(([,intents])=>intents.includes(command.intent)).length,1);
const local=new Set(['TIME','TIMER','ALARM','CREATE_REMINDER','LIST_REMINDERS']);
const camera='camera.dining_area_main_camera_tc70_mainstream';
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const checks=[],errors=[];
try {
  for(const [width,height] of [[800,480],[1024,600],[390,844]]) {
    const directory=await mkdtemp(join(process.env.TMPDIR||tmpdir(),'iris-publication-ui-'));
    let sequence=0,cameraAvailable=false,cameraReads=0,providerWrites=0;
    const ready={event:'ready',session_id:'offline-contract',sequence:0,loaded:true,state:'wake_listening',actions_enabled:true,profile:'owner-protected',
      owner_gating_enabled:true,ownership_experimental:true,parameters:258912,selected_epoch:44,model_qualified:false,wake_phrase:'hey iris',
      model_sha256:'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce',
      application_config:{mode:'joint_complete_command',min_probability:.5,min_margin:.1},capabilities:{feedback:true}};
    const host=Object.assign(new EventEmitter(),{status:()=>ready,current:()=>ready,request:async()=>({ok:true,devices:[]}),close:async()=>{}});
    const socket=Object.assign(new EventEmitter(),{send(){},close(){},terminate(){},ping(){}});
    const config=readConfig({IRIS_PUBLICATION:'true',ACTIONS_ENABLED:'true',INFERENCE_MODE:'onnx',SMART_SPEAKER_PYTHON:'offline-never-spawn',
      SMART_SPEAKER_ROOT:'.',SMART_SPEAKER_BUNDLE:'.',SMART_SPEAKER_TOKEN:'offline-fixture',DATA_DIR:directory,
      HA_ENABLED:'true',HA_URL:'http://offline.invalid',HA_TOKEN:'not-a-real-secret',HA_ENTITY_IDS:`light.hue_color_lamp_1,light.dining_light,${camera},sensor.offline_fixture_temperature`,
      HA_TEMPERATURE_SENSOR_ID:'sensor.offline_fixture_temperature'},process.cwd());
    const imagePage=await browser.newPage();
    const jpeg=Buffer.from((await imagePage.evaluate(()=>{const canvas=document.createElement('canvas');canvas.width=160;canvas.height=90;
      const ctx=canvas.getContext('2d');ctx.fillStyle='#14212a';ctx.fillRect(0,0,160,90);ctx.fillStyle='#9ce8d3';ctx.font='12px sans-serif';ctx.fillText('Offline camera fixture',12,48);return canvas.toDataURL('image/jpeg').split(',')[1];})),'base64');
    await imagePage.close();
    const app=await createApp(config,{staticDir:resolve('dist'),runtimeFactory:()=>host,haSocketFactory:()=>socket,fetcher:async(input,init)=>{
      const url=String(input);
      if(init?.method&&init.method!=='GET'){providerWrites++;assert.fail('No real provider mutation in browser QA');}
      if(url.endsWith('/api/states/sensor.offline_fixture_temperature'))return Response.json({entity_id:'sensor.offline_fixture_temperature',state:'29.5',last_updated:'2026-10-23T09:00:00Z',attributes:{device_class:'temperature',unit_of_measurement:'°C'}});
      if(url.includes('/api/states/light.'))return Response.json({entity_id:'light.hue_color_lamp_1',state:'unavailable',attributes:{}});
      if(url.includes('/api/camera_proxy')){
        cameraReads++;
        if(!cameraAvailable)throw new Error('Offline camera unavailable');
        if(url.includes('/api/camera_proxy_stream/'))return new Response(new ReadableStream({start(controller){
          const frame=()=>controller.enqueue(Buffer.concat([Buffer.from('--iris-fixture\r\nContent-Type: image/jpeg\r\nContent-Length: '+jpeg.length+'\r\n\r\n'),jpeg,Buffer.from('\r\n')]));
          frame();const timer=setInterval(frame,500);init.signal.addEventListener('abort',()=>{clearInterval(timer);try{controller.close();}catch{}},{once:true});
        }}),{headers:{'Content-Type':'multipart/x-mixed-replace; boundary=iris-fixture'}});
        return new Response(jpeg,{headers:{'Content-Type':'image/jpeg'}});
      }
      assert.fail('Unexpected outbound fixture URL: '+url);
    }});
    await app.listen({host:'127.0.0.1',port:0});config.port=app.server.address().port;config.origin=`http://127.0.0.1:${config.port}`;
    const context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'});
    const page=await context.newPage();
    page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
    try {
      await page.goto(config.origin);
      await page.getByRole('heading',{name:'Say “Hey Iris”'}).waitFor();
      await page.waitForFunction(()=>document.querySelectorAll('.use-cases button').length===10,{},{timeout:3000});
      for(const [title] of groups)await page.getByRole('button',{name:title,exact:true}).waitFor();
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      await page.screenshot({path:resolve(output,`home-${width}x${height}.png`)});
      const captured=new Set();
      for(const [index,command] of inventory.entries()) {
        const id=`v${width}-c${index}`;
        host.emit('event',{event:'command',session_id:ready.session_id,sequence:++sequence,utterance_id:id,is_final:true,source:'microphone',actions_enabled:true,
          owner_result:{accepted:true},output:{schema_version:3,accepted:true,command},application:{accepted:true,command},latency_ms:7.25,command_probability:.625,command_margin:.125});
        const expected=local.has(command.intent)?'local':'synthetic';
        await page.locator(`[data-execution-status="${expected}"][data-execution-id="${ready.session_id}:${id}"]`).waitFor();
        await page.locator('.recognition').getByText(/Posterior 62.5%.*Margin 12.5%/).waitFor();
        await page.locator('.recognition').getByText(/7.25 ms/).waitFor();
        const badge=page.locator('[aria-label="Execution receipt"] [data-source]');
        assert.equal(await badge.getAttribute('data-source'),local.has(command.intent)?'local':'synthetic');
        if(expected==='synthetic')await page.getByRole('status').filter({hasText:'Synthetic — no device changed'}).waitFor();
        if(command.intent==='TEMPERATURE'){
          await page.locator('[data-synthetic-target]').waitFor();
          await page.locator('[data-observed-temperature]').filter({hasText:'29.5°C'}).waitFor();
          assert.match(await page.locator('[data-synthetic-target]').innerText(),new RegExp(String(command.slots.degrees)));
        }
        if(command.intent==='CALL'){
          await page.getByText('Home camera • video only',{exact:true}).waitFor();
          assert.equal(await page.locator('img[data-home-camera]').count(),0,'Synthetic CALL must not activate the camera');
        }
        const category=groups.find(([,intents])=>intents.includes(command.intent))[0];
        if(!captured.has(category)){captured.add(category);await page.screenshot({path:resolve(output,`${String(groups.findIndex(([name])=>name===category)+1).padStart(2,'0')}-${width}x${height}.png`)});}
        const state=(await app.inject({url:'/api/actions/state',headers:{host:new URL(config.origin).host}})).json();
        for(const job of state.jobs.filter(j=>['timer','alarm'].includes(j.kind)&&!['cancelled','completed'].includes(j.status)))
          await app.inject({method:'POST',url:'/api/actions/jobs/'+job.id,headers:{host:new URL(config.origin).host,origin:config.origin},payload:{action:'cancel'}});
      }
      assert.equal(captured.size,10);
      await page.getByRole('button',{name:'Microphone settings'}).click();
      await page.getByText(/joint_complete_command.*Posterior ≥ 50%.*Margin ≥ 10%/).waitFor();
      await page.getByText(/258,912 parameters.*epoch 44.*Hey Iris/i).waitFor();
      assert.doesNotMatch(await page.locator('.runtime-debug').innerText(),/95%|90%|Each slot/);
      await page.screenshot({path:resolve(output,`diagnostics-${width}x${height}.png`)});
      const before=cameraReads;
      host.emit('event',{event:'tentative',session_id:ready.session_id,sequence:++sequence,utterance_id:'preview',is_final:false,candidate:{intent:'CALL',slots:{contact_id:'contact.home'}}});
      await page.locator('[data-recognition-phase=tentative]').waitFor();assert.equal(cameraReads,before);
      cameraAvailable=true;
      host.emit('event',{event:'command',session_id:ready.session_id,sequence:++sequence,utterance_id:`camera-${width}`,source:'microphone',actions_enabled:true,owner_result:{accepted:true},
        output:{schema_version:3,accepted:true,command:{intent:'CALL',slots:{contact_id:'contact.home'}}},application:{accepted:true,command:{intent:'CALL',slots:{contact_id:'contact.home'}}}});
      await page.locator('img[data-home-camera]').waitFor();
      await page.waitForFunction(()=>document.querySelector('img[data-home-camera]')?.naturalWidth>0);
      const frameBox=await page.locator('img[data-home-camera]').boundingBox(),panelBox=await page.locator('.panel-body').boundingBox();
      assert.ok(frameBox.y>=panelBox.y&&frameBox.y+frameBox.height<=panelBox.y+panelBox.height,'The camera frame must be visible, not clipped below the panel');
      assert.equal(await page.getByRole('status').filter({hasText:'Synthetic — no device changed'}).count(),0,'A real provider fixture result must clear an earlier synthetic toast');
      await page.screenshot({path:resolve(output,`camera-offline-fixture-${width}x${height}.png`)});
      assert.equal(await page.locator('audio[autoplay],video[autoplay]').count(),0);
      await page.getByRole('button',{name:'Close camera',exact:true}).click();
      assert.equal(await page.locator('img[data-home-camera]').count(),0);
      await page.reload();await page.getByRole('heading',{name:'Say “Hey Iris”'}).waitFor();
      await page.getByRole('button',{name:'Reminders',exact:true}).click();
      await page.locator('.job').filter({hasText:'Study'}).waitFor();
      await page.getByRole('button',{name:'Back to home'}).click();
      await page.getByRole('button',{name:'Temperature',exact:true}).click();
      await page.locator('[data-synthetic-target]').filter({hasText:'26'}).waitFor();
      await page.locator('[data-observed-temperature]').filter({hasText:'29.5°C'}).waitFor();
      assert.equal(await page.locator('[role=status]').filter({hasText:'Synthetic — no device changed'}).count(),0,'Reconnection must not replay a toast');
      assert.equal(providerWrites,0);
      checks.push({viewport:[width,height],typed_commands:inventory.length,categories:captured.size,actual_persistent_jobs:true,observed_sensor:'offline read-only fixture',camera:'admitted offline MJPEG fixture',synthetic_camera_no_stream:true,provider_mutations:providerWrites,passed:true});
    } finally {await context.close();await app.close();await rm(directory,{recursive:true,force:true});}
  }
  assert.deepEqual(errors,[]);
  await writeFile(resolve(output,'results.json'),JSON.stringify({passed:true,checks,errors,model_transport:'explicit offline fixture',providers:'offline fetch fixtures',acoustic:false,live_microphone:false,host_audio_playback:false,physical_devices:false},null,2)+'\n');
  console.log(JSON.stringify({passed:true,viewports:checks.length,commands_per_viewport:inventory.length,categories:groups.length,errors:errors.length,physical_devices:false,output}));
} finally {await browser.close();}
