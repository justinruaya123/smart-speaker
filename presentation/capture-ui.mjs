/* Capture actual compiled c062894 Iris UI, with explicit non-actuating fixture.
 * First build from SpeakerUI: node node_modules/vite/bin/vite.js build
 *   --outDir ../.cache/presentation-ui --emptyOutDir
 * IRIS_PLAYWRIGHT_ROOT=/existing/SpeakerUI node presentation/capture-ui.mjs
 * Never starts the gateway, microphone, native worker or external providers.
 */
import {createRequire} from 'node:module';
import {createServer} from 'node:http';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname,extname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(resolve(process.env.IRIS_PLAYWRIGHT_ROOT||resolve(root,'SpeakerUI'),'package.json'));
const {chromium}=require('playwright');
const folder=resolve(root,'.cache/presentation-ui');
const state={revision:1,jobs:[],receipts:[],actions_enabled:false,timezone:'Asia/Manila',bindings:{lights:{},songs:[],climate:null,weather:null,media_player:{},contacts:[],messages:false}};
const server=createServer(async(req,res)=>{
  if(req.url.startsWith('/api/')){
    const value=req.url==='/api/health'?{inference:'disabled'}:req.url==='/api/actions/state'?state:{entities:[],status:'not_configured'};
    res.writeHead(200,{'Content-Type':'application/json'}).end(JSON.stringify(value));return;
  }
  try{const file=resolve(folder,'.'+(req.url==='/'?'/index.html':decodeURIComponent(req.url.split('?')[0])));if(!file.startsWith(folder+sep))throw Error('outside');
    const data=await readFile(file);res.writeHead(200,{'Content-Type':{'.html':'text/html','.css':'text/css','.js':'text/javascript','.svg':'image/svg+xml'}[extname(file)]||'application/octet-stream'}).end(data);
  }catch{res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,channel:process.env.IRIS_BROWSER_CHANNEL,executablePath:process.env.IRIS_BROWSER_EXECUTABLE});
try{
  const page=await browser.newPage({viewport:{width:1024,height:600},reducedMotion:'reduce'});
  await page.addInitScript(()=>{window.WebSocket=class extends EventTarget{constructor(){super();this.readyState=1;}close(){this.readyState=3;}send(){}};});
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.getByRole('heading',{name:'Microphone unavailable',exact:true}).waitFor();
  await mkdir(resolve(root,'docs/site/assets'),{recursive:true});
  const output=resolve(root,'docs/site/assets/iris-ui.png');
  await page.screenshot({path:output});
  const payload=await readFile(output),source=await readFile(resolve(root,'SpeakerUI/src/ambient/Iris.svelte'));
  const receipt={browser:browser.version(),playwright:require('playwright/package.json').version,format:'iris-ui-screenshot-v1',source_revision:'c0628940400642f204ff98e26908549b1c07e41c',source_path:'SpeakerUI/src/ambient/Iris.svelte',source_sha256:createHash('sha256').update(source).digest('hex'),screenshot_sha256:createHash('sha256').update(payload).digest('hex'),viewport:[1024,600],fixture:true,inference:'disabled',actions_enabled:false,live_microphone:false,physical_device_evidence:false,note:'Actual compiled base-main Iris UI. Health/action/entity routes are local software fixtures; no synthetic recognition score or provider result is shown.'};
  await writeFile(resolve(root,'docs/site/assets/iris-ui-receipt.json'),JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify(receipt,null,2));
}finally{await browser.close();await new Promise(r=>server.close(r));}
