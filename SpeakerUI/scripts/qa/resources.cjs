const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),os=require('node:os'),{execFileSync}=require('node:child_process');
(async()=>{
const b=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{}),args:['--enable-unsafe-swiftshader']});
const bs=await b.newBrowserCDPSession(),base=process.env.TEST_BASE_URL||'http://127.0.0.1:3000/?fullscreen=0';
const report={date:new Date().toISOString(),environment:{platform:os.platform(),arch:os.arch(),cpu:os.cpus()[0].model,browser:await b.version(),viewport:'800x480',gpu:'SwiftShader software WebGL',limitation:'Desktop Chromium measurements, not Raspberry Pi hardware or physical touch/audio validation.'},loads:[],states:[],errors:[]};
let p,c;
for(let i=0;i<3;i++){
 const ctx=await b.newContext({viewport:{width:800,height:480},timezoneId:'Asia/Manila'});p=await ctx.newPage();c=await ctx.newCDPSession(p);await c.send('Network.enable');await c.send('Network.setCacheDisabled',{cacheDisabled:true});p.on('pageerror',e=>report.errors.push(e.message));
 await p.goto(base,{waitUntil:'networkidle'});await p.waitForTimeout(300);
 report.loads.push(await p.evaluate(()=>{const n=performance.getEntriesByType('navigation')[0],r=performance.getEntriesByType('resource');return {domContentLoadedMs:n.domContentLoadedEventEnd,loadMs:n.loadEventEnd,fcpMs:performance.getEntriesByName('first-contentful-paint')[0]?.startTime,transferBytes:n.transferSize+r.reduce((n,r)=>n+r.transferSize,0),encodedBytes:n.encodedBodySize+r.reduce((n,r)=>n+r.encodedBodySize,0),decodedBytes:n.decodedBodySize+r.reduce((n,r)=>n+r.decodedBodySize,0),requests:r.map(r=>({name:new URL(r.name).pathname,encodedBytes:r.encodedBodySize,decodedBytes:r.decodedBodySize})),domNodes:document.querySelectorAll('*').length};}));
 if(i<2)await ctx.close();
}
await c.send('Performance.enable');
async function sample(name){
 await p.waitForTimeout(800);await c.send('HeapProfiler.collectGarbage');
 const before=Object.fromEntries((await c.send('Performance.getMetrics')).metrics.map(x=>[x.name,x.value]));
 const procBefore=(await bs.send('SystemInfo.getProcessInfo')).processInfo;
 const t=performance.now();
 const frame=await p.evaluate(()=>new Promise(resolve=>{const values=[];let previous=performance.now(),start=previous;function frame(now){values.push(now-previous);previous=now;if(now-start<5000){requestAnimationFrame(frame);return;}const sorted=values.slice(1).sort((a,b)=>a-b);resolve({frames:sorted.length,medianMs:sorted[Math.floor(sorted.length*.5)],p95Ms:sorted[Math.floor(sorted.length*.95)],over50Ms:sorted.filter(x=>x>50).length});}requestAnimationFrame(frame);}));
 const elapsed=(performance.now()-t)/1000,after=Object.fromEntries((await c.send('Performance.getMetrics')).metrics.map(x=>[x.name,x.value])),procAfter=(await bs.send('SystemInfo.getProcessInfo')).processInfo;
 const cpu=procAfter.map(a=>({type:a.type,id:a.id,cpuPercent:Math.max(0,(a.cpuTime-(procBefore.find(p=>p.id===a.id)?.cpuTime??a.cpuTime))/elapsed*100)}));
 const heap=await c.send('Runtime.getHeapUsage');
 let browserResidentMiB=null;try{const rss=execFileSync('ps',['-o','rss=','-p',procAfter.map(p=>p.id).join(',')],{encoding:'utf8'});browserResidentMiB=rss.trim().split(/\s+/).reduce((n,x)=>n+Number(x),0)/1024;}catch{}
 report.states.push({name,seconds:elapsed,browserResidentMiB,rendererMainThreadBusyPercent:(after.TaskDuration-before.TaskDuration)/elapsed*100,jsHeapUsedMB:heap.usedSize/1e6,jsHeapTotalMB:heap.totalSize/1e6,domNodes:after.Nodes,framePacing:frame,cpuProcesses:cpu,totalBrowserCpuPercent:cpu.reduce((n,x)=>n+x.cpuPercent,0)});
 console.log(name,JSON.stringify(report.states.at(-1)));
}
async function intent(intent){await p.evaluate(intent=>document.querySelector('main').dispatchEvent(new CustomEvent('vcm-intent',{detail:{intent,parameters:{}}})),intent);await p.waitForTimeout(650);}
await sample('idle');await p.locator('.wake-target').click();await sample('listening');
await p.locator('.demo-launch').click();await p.locator('[data-agent=speaking]').click();await sample('speaking');
await intent('media.play');await sample('music playback');await p.locator('.control-trigger').click();await sample('control center glass');
await intent('demo.alarm_due');await sample('alarm flashing edges');await p.locator('[data-action=alert-dismiss]').click();
await p.locator('.control-trigger').click();await p.locator('[data-open=contacts]').click();await p.locator('[data-contact="Front door"]').click();await sample('simulated camera call');
await p.locator('[data-action=hangup]').click();await p.locator('.toast-close').click();
await c.send('HeapProfiler.collectGarbage');const initialHeap=await c.send('Runtime.getHeapUsage');
for(let i=0;i<30;i++){await p.locator('.control-trigger').click();await p.locator('[data-open=alarm]').click();await p.locator('.task-view>.close-task').click();}
await p.waitForTimeout(1000);await c.send('HeapProfiler.collectGarbage');const finalHeap=await c.send('Runtime.getHeapUsage');report.reopen={cycles:30,initialUsedMB:initialHeap.usedSize/1e6,finalUsedMB:finalHeap.usedSize/1e6,deltaMB:(finalHeap.usedSize-initialHeap.usedSize)/1e6};
fs.writeFileSync('docs/testing/ambient/resources.json',JSON.stringify(report,null,2));await b.close();console.log('Saved resources.json');
})().catch(e=>{console.error(e);process.exitCode=1});
