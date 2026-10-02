import {spawn} from 'node:child_process';import {once} from 'node:events';import assert from 'node:assert/strict';import {writeFile} from 'node:fs/promises';import {WebSocket} from 'ws';
const env={...process.env,HA_ENABLED:'false',SPOTIFY_ENABLED:'false',ACTIONS_ENABLED:'false',FRONTEND_HOST:'127.0.0.1',HOST:'127.0.0.1',FRONTEND_PORT:'4310',BACKEND_PORT:'4311',DISPLAY_ORIGIN:'',BACKEND_URL:'',TLS_CERT_PATH:'',TLS_KEY_PATH:''},checks=[];
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const ok=(s,v)=>{assert.ok(v,s);checks.push(s);console.log('PASS',s)};
async function available(url){try{return (await fetch(url,{signal:AbortSignal.timeout(500)})).status!==503;}catch{return false;}}
async function run(mode,extra={}){const child=spawn(process.execPath,['scripts/start.mjs',...mode],{env:{...env,...extra},stdio:['ignore','pipe','pipe']});let log='';child.stdout.on('data',d=>log+=d);child.stderr.on('data',d=>log+=d);return {child,getLog:()=>log};}
async function waitUp(job,url){for(let i=0;i<150;i++){if(await available(url))return;if(job.child.exitCode!==null)throw Error(job.getLog());await delay(100);}throw Error('Startup timeout: '+job.getLog());}
async function stop(job){const exited=once(job.child,'exit');job.child.kill('SIGTERM');await Promise.race([exited,delay(6000).then(()=>{throw Error('Shutdown timeout')})]);}
for(const mode of [[],['backend'],['frontend']]){
 const job=await run(mode);let ws;
 try{
 const front=mode[0]!=='backend',back=mode[0]!=='frontend';await waitUp(job,front?'http://127.0.0.1:4310/':'http://127.0.0.1:4311/api/health');
 ok(`start ${mode[0]||'both'} builds and starts selected side`,job.getLog().includes(front?'built in':'tsc'));
 if(back){const health=await (await fetch('http://127.0.0.1:4311/api/health')).json();ok(`start ${mode[0]||'both'} loads dummy runtime`,health.runtime.loaded&&health.runtime.nativeSession===false);}
 if(front&&back){const fixture=await fetch('http://127.0.0.1:4310/api/ambient/infer',{method:'POST',headers:{origin:'http://127.0.0.1:4310','Content-Type':'application/json'},body:'{"fixture":3}'});ok('frontend proxies validated dummy inference',fixture.status===200&&(await fixture.json()).result.intent==='timer.set');
 const rejected=await fetch('http://127.0.0.1:4310/api/ambient/infer',{method:'POST',headers:{origin:'https://other.invalid','Content-Type':'application/json'},body:'{"fixture":3}'});ok('cross-origin mutation remains rejected through proxy',rejected.status===403);
 for(const [accept,expected] of [['br','br'],['gzip','gzip'],['br;q=0,gzip;q=0.8','gzip']]){const response=await fetch('http://127.0.0.1:4310/',{headers:{'Accept-Encoding':accept}});ok(`precompressed response respects ${accept}`,response.headers.get('content-encoding')===expected);await response.arrayBuffer();}
 ws=new WebSocket('ws://127.0.0.1:4310/ws',{origin:'http://127.0.0.1:4310'});const [data]=await once(ws,'message');ok('diagnostic WebSocket works through frontend',JSON.parse(data.toString()).type==='ready');}
 if(front&&!back)ok('frontend alone stays usable with backend absent',(await fetch('http://127.0.0.1:4310/api/health')).status===503);
 }finally{await stop(job);ws?.terminate();}
 ok(`start ${mode[0]||'both'} releases both ports after shutdown`,!await available('http://127.0.0.1:4310/')&&!await available('http://127.0.0.1:4311/api/health'));
}
for(const [mode,extra] of [[['bogus'],{}],[[],{FRONTEND_PORT:'0'}],[[],{FRONTEND_PORT:'4311'}]]){const job=await run(mode,extra);const [code]=await once(job.child,'exit');ok('invalid mode/port is rejected before serving',code!==0);}
// Interrupt an actual frontend build and check the supervisor owns its child process group.
const interrupted=await run(['frontend']);for(let i=0;i<100&&!interrupted.getLog().includes('bake-assets');i++)await delay(10);await stop(interrupted);await delay(300);ok('interrupted build leaves no listening service',!await available('http://127.0.0.1:4310/'));
await writeFile('docs/testing/ambient/startup.json',JSON.stringify({checks},null,2));
