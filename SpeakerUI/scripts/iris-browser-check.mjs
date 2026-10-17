// Compiled product checks with explicitly mocked inference transport. No acoustic/device claims.
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { chromium } from 'playwright';
import { createApp } from '../dist-server/server/app.js';
import { readConfig } from '../dist-server/server/config.js';

const output=resolve(process.argv[2] || '../.cache/iris-release-v1/ui');
await mkdir(output,{recursive:true});
const directory=await mkdtemp(resolve(tmpdir(),'iris-browser-'));
const config=readConfig({ACTIONS_ENABLED:'true',DATA_DIR:directory,TIMEZONE:'UTC'},process.cwd());
const app=await createApp(config,{staticDir:'dist',fetcher:async()=>{throw new Error('Browser contract checks must not contact providers');}});
await app.listen({host:'127.0.0.1',port:0});
config.port=app.server.address().port;config.origin=`http://127.0.0.1:${config.port}`;
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const errors=[],checks=[];
const commands=[['PLAY_MUSIC',{song_id:'song.dynamite'}],['WEATHER',{}],['TIME',{}],['LIGHT_ON',{light_id:'light.all'}],
  ['LIGHT_OFF',{light_id:'light.all'}],['PAUSE',{}],['STOP',{}],['NEXT',{}],['VOLUME',{operation:'set',percent:17}],
  ['CALL',{contact_id:'contact.mom'}],['MESSAGE',{}],['LIST_REMINDERS',{}],['TIMER',{duration_seconds:123}],
  ['ALARM',{time:'07:00',timezone:'Asia/Manila'}],['TEMPERATURE',{degrees:22,unit:'C'}],
  ['BRIGHTNESS',{light_id:'light.all',percent:37}],['COLOR',{light_id:'light.all',color:'blue'}],['CREATE_REMINDER',{task_id:'task.study'}]];
try {
  for(const [width,height] of [[800,480],[1024,600],[390,844]]) {
    const context=await browser.newContext({viewport:{width,height}});
    let channel,sequence=0;
    const send=value=>channel.send(JSON.stringify({session_id:'contract-fixtures',sequence:sequence++,...value}));
    await context.route('**/api/health',route=>route.fulfill({json:{inference:'onnx'}}));
    await context.route('**/api/runtime/devices',route=>route.fulfill({json:{devices:[]}}));
    await context.routeWebSocket('**/ws/runtime',socket=>{
      channel=socket;
      send({event:'ready',loaded:true,state:'wake_listening',profile:'owner-protected',owner_gating_enabled:true,ownership_experimental:true,actions_enabled:true});
    });
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    let actionRequests=0;page.on('request',request=>{if(request.method()==='POST'&&request.url().endsWith('/api/actions'))actionRequests++;});
    await page.goto(config.origin);
    await page.getByRole('heading',{name:'Say “Hey Iris”'}).waitFor();
    assert.equal(await page.locator('canvas').count(),0);
    for(const [reason,label] of [['speaker_mismatch','Speaker Mismatch'],['overlap','Overlap'],['insufficient_speech','Insufficient Speech']]) {
      send({event:'ownership',accepted:false,reason,ownership_experimental:true});
      await page.locator('.profile').filter({hasText:label}).waitFor();
      assert.match(await page.locator('.profile').innerText(),/Experimental owner voice check enabled/);
    }
    await page.getByRole('button',{name:'Microphone settings'}).click();
    assert.match(await page.locator('.profile').innerText(),/Experimental owner voice check enabled.*Insufficient Speech/);
    await page.getByRole('button',{name:'Back to home'}).click();
    send({event:'tentative',utterance_id:'preview',candidate:{intent:'TIMER',slots:{duration_seconds:30}},score:.8,is_final:false});
    await page.locator('[data-recognition-phase=tentative]').waitFor();
    send({event:'state',state:'processing'});
    await page.getByRole('heading',{name:'Understanding'}).waitFor();
    assert.match(await page.locator('[data-recognition-phase=tentative]').innerText(),/duration_seconds=30/);
    assert.equal(actionRequests,0);
    send({event:'command',utterance_id:'preview',source:'wav',actions_enabled:false,
      output:{schema_version:3,accepted:false,command:null,reason:'unqualified_model'}});
    await page.getByText('Rejected',{exact:true}).waitFor();
    assert.equal(actionRequests,0);
    send({event:'ownership',accepted:true,reason:'owner_match',ownership_experimental:true});
    for(const [intent,slots] of commands) {
      send({event:'command',utterance_id:intent,source:'contract-fixture',actions_enabled:false,
        output:{schema_version:3,accepted:true,command:{intent,slots},reason:null}});
      await page.locator('.recognition strong').filter({hasText:new RegExp('^'+intent+'$')}).waitFor();
    }
    assert.equal(actionRequests,0);
    checks.push({viewport:[width,height],kind:'18_typed_contract_fixtures_and_preview_reset',owner_reasons_visible:['speaker_mismatch','overlap','insufficient_speech'],owner_status_visible_in_settings:true,acoustic:false,provider_calls:0,passed:true});
    for(const title of ['Lights','Music','Temperature','Volume','Call','Message','Weather','Time','Reminders','Alarm']) {
      await page.getByRole('button',{name:title,exact:true}).click();
      await page.getByRole('heading',{name:title,exact:true}).waitFor();
      if(title==='Time')await page.getByText('UTC',{exact:true}).waitFor();
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      await page.getByRole('button',{name:'Back to home'}).click();
    }
    await page.screenshot({path:resolve(output,`home-${width}x${height}.png`)});
    await page.getByRole('button',{name:'Timer',exact:true}).click();
    await page.getByLabel('Duration in seconds').fill('1');
    await page.getByRole('button',{name:'Start timer',exact:true}).click();
    await page.locator('[data-execution-status=local]').waitFor();
    await page.getByRole('alertdialog',{name:'Timer'}).waitFor();
    await page.getByRole('button',{name:'Dismiss',exact:true}).click();
    await page.getByRole('alertdialog').waitFor({state:'detached'});
    await page.getByRole('button',{name:'Back to home'}).click();
    await page.getByRole('button',{name:'Reminders',exact:true}).click();
    await page.getByLabel('Reminder task',{exact:true}).selectOption('task.study');
    await page.getByRole('button',{name:'Save reminder'}).click();
    await page.locator('.job').filter({hasText:'Study'}).waitFor();
    await page.reload();
    await page.getByRole('button',{name:'Reminders',exact:true}).click();
    await page.locator('.job').filter({hasText:'Study'}).waitFor();
    await page.getByRole('button',{name:'Done',exact:true}).click();
    await page.getByRole('button',{name:'Back to home'}).click();
    const metrics=await page.evaluate(()=>({dom:document.querySelectorAll('*').length,heap:performance.memory?.usedJSHeapSize,resources:performance.getEntriesByType('resource').length}));
    checks.push({viewport:[width,height],kind:'actual_local_timer_due_dismiss_and_reminder_reload',physical_device:false,metrics,passed:true});
    await context.close();
  }
  const diagnostic=await browser.newPage({viewport:{width:800,height:480}});
  await diagnostic.goto(config.origin+'/?diagnostics');
  await diagnostic.getByText('MOCK audio transport:',{exact:true}).waitFor();
  await diagnostic.getByText(/This audio path does not recognize speech or execute actions/).waitFor();
  await diagnostic.getByRole('link',{name:'Return to Iris'}).waitFor();
  await diagnostic.close();
  checks.push({kind:'explicit_mock_diagnostics_and_separate_manual_media',acoustic:false,passed:true});
  assert.deepEqual(errors,[]);
  await writeFile(resolve(output,'results.json'),JSON.stringify({passed:true,checks,errors,inference_transport:'mocked',acoustic_test:false,live_microphone:false,physical_pi:false},null,2)+'\n');
  console.log(`Iris compiled browser: ${checks.length} checks passed (three layouts; preview/finals and actual persisted local jobs).`);
} finally {await browser.close();await app.close();await rm(directory,{recursive:true,force:true});}
