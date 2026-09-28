import {Pipeline, parseWav, canonical, sha256} from './core.mjs';
import {TensorScene} from './scene.mjs';

const $ = id => document.getElementById(id);
const state = {lastDraw:0, pipeline:null, recipe:null, graph:null, weights:null, samples:[], pcm:new Float32Array(0), position:0, running:false, timer:null, generation:0, selected:'stem', rows:[], filtered:[], page:0, microphone:null, final:null};
let scene;
const descriptions = {
  mel: 'A 25 ms Hann window is transformed into frequency power, combined into 40 mel bands, and logged. These are measured features, not recognized words.',
  stem: 'The stem mixes 40 frequency channels over three causal frames. Its learned filters create 64 feature channels. The cache stores only past frames.',
  conv1: 'A width-5 temporal convolution followed by ReLU. Each output combines all 64 input channels. ReLU removes negative responses.',
  conv2: 'Dilation 2 spaces temporal taps two frames apart. The layer expands the past context without downsampling or looking ahead.',
  conv3: 'Dilation 4 expands the acoustic context. The displayed bars sample 16 of the 64 channels; the heatmap and download contain the complete tensor.',
  conv4: 'Dilation 8 can combine evidence separated by 80 ms. Cached values are real prior activations and are reset between utterances.',
  conv5: 'Dilation 16 provides wider past context. The complete encoder has a 127-frame receptive field. No future audio is used.',
  ctc: 'The linear projection emits 39 phoneme scores and a blank score for each frame. CTC sums valid alignments against complete catalog commands. These logits are not calibrated command confidence.'
};
function error(message) { $('error').textContent = message || ''; }
function guard(fn) { return async (...args) => { try { error(''); return await fn(...args); } catch(e) { pause();error(e.message);console.error(e); } }; }
function pause() {state.running=false;clearTimeout(state.timer);$('play').textContent='▶ Run';}
function download(name, content, type='application/json') {const a=document.createElement('a'),url=URL.createObjectURL(new Blob([content],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function loadModel(recipe,weights,graph) {
  pause();if(state.microphone)await stopMicrophone();const generation=++state.generation;
  $('phase').textContent='Validating…';const model=await Pipeline.create(recipe,weights,graph);
  if(generation!==state.generation)return;
  state.pipeline=model;state.recipe=recipe;state.weights=weights;state.graph=graph;state.final=null;state.position=0;
  $('parameterCount').textContent=recipe.layers.reduce((sum,l)=>sum+l.parameters,0).toLocaleString();
  $('contract').textContent=JSON.stringify({schema:recipe.schema_version,onnx_sha256:recipe.source_onnx_sha256,weights_sha256:recipe.weights_sha256,graph_hash:recipe.graph_hash,frontend:'16 kHz / 400 / 160 / 512 / 40 mel',chunk_frames:recipe.chunk_frames,acceptance_enabled:false},null,2);
  if(scene)scene.setLayers(recipe.layers);
  $('layerButtons').replaceChildren();for(const l of [{id:'mel'},...recipe.layers]){const b=document.createElement('button');b.textContent=l.id;b.dataset.layer=l.id;b.addEventListener('click',()=>select(l.id));$('layerButtons').append(b);}
  select('stem');refresh();
}
async function loadSample(name) {
  pause();if(state.microphone)await stopMicrophone();
  const response=await fetch(`assets/${encodeURIComponent(name)}`);if(!response.ok)throw new Error('Sample WAV is missing. Run export_assets.py or restore the full payload.');
  const pcm=parseWav(await response.arrayBuffer());setAudio(pcm);const example=state.samples.find(s=>s.file===name);
  $('verbatim').textContent=example?.verbatim||name.replaceAll('_',' ');
  $('sample').value=name;
}
function setAudio(pcm) {
  if(!state.pipeline)throw new Error('Load a model first');
  // The fixed 100 ms tail is part of the inspection protocol, not microphone endpointing.
  state.pcm=new Float32Array(pcm.length+1600);state.pcm.set(pcm);
  state.originalSamples=pcm.length;reset();
}
function reset() {pause();state.pipeline?.reset();state.position=0;state.final=null;refresh();}
function startNextChunk() {
  const p=state.pipeline;if(!p||p.finished)return false;if(p.current)return true;
  while(p.pendingFrames<p.r.chunk_frames&&state.position<state.pcm.length){const end=Math.min(state.position+640,state.pcm.length);p.enqueue(state.pcm.subarray(state.position,end));state.position=end;}
  return p.startChunk(state.position===state.pcm.length);
}
function stepLayer() {
  if(state.microphone)throw new Error('Stop live capture before inspection stepping');
  if(!state.pcm.length)throw new Error('Load a waveform');
  if(state.pipeline.finished)return;
  if(!startNextChunk()){finalize('push_to_talk_release');return;}
  const event=state.pipeline.stepLayer();select(event.layer);refresh();
  if(event.done&&state.position===state.pcm.length&&!state.pipeline.pendingFrames)finalize('push_to_talk_release');
}
function stepChunk() {
  if(state.microphone)throw new Error('Stop live capture before inspection stepping');
  if(state.pipeline.finished)return;
  if(!startNextChunk()){finalize('push_to_talk_release');return;}
  let event;do {event=state.pipeline.stepLayer();}while(!event.done);
  refresh();if(state.position===state.pcm.length&&!state.pipeline.pendingFrames)finalize('push_to_talk_release');
}
function finalize(reason) {if(state.pipeline.finished)return;state.final=state.pipeline.finish(reason);pause();refresh();}
async function play() {
  if(state.microphone)throw new Error('Live microphone input is already running at 1×');
  if(state.running){pause();return;}if(state.pipeline.finished)reset();
  state.running=true;$('play').textContent='❚❚ Pause';const generation=state.generation;
  const tick=()=>{if(!state.running||generation!==state.generation)return;const begin=performance.now();try {stepChunk();}catch(e){pause();error(e.message);return;}if(!state.running)return;const interval=state.recipe.chunk_frames*10/Number($('speed').value);state.timer=setTimeout(tick,Math.max(0,interval-(performance.now()-begin)));};tick();
}
function select(id) {state.selected=id;if(scene)scene.selected=id;for(const b of $('layerButtons').children)b.classList.toggle('active',b.dataset.layer===id);refreshProbe();if(scene)scene.draw();}
function plotWave() {
  const canvas=$('wave'),ctx=canvas.getContext('2d'),width=canvas.clientWidth||600;heightResize(canvas,width,90);ctx.clearRect(0,0,width,90);ctx.strokeStyle='#264653';ctx.beginPath();ctx.moveTo(0,45);ctx.lineTo(width,45);ctx.stroke();
  if(!state.pcm.length)return;ctx.strokeStyle='#60ddc6';ctx.beginPath();const step=Math.max(1,Math.ceil(state.pcm.length/width));for(let x=0;x<width;x++){let min=0,max=0;for(let i=x*step;i<Math.min((x+1)*step,state.pcm.length);i++){min=Math.min(min,state.pcm[i]);max=Math.max(max,state.pcm[i]);}ctx.moveTo(x,45-40*max);ctx.lineTo(x,45-40*min);}ctx.stroke();ctx.strokeStyle='#f8b981';const cursor=state.position/state.pcm.length*width;ctx.beginPath();ctx.moveTo(cursor,0);ctx.lineTo(cursor,90);ctx.stroke();
}
function heightResize(canvas,w,h) {const d=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);}canvas.getContext('2d').setTransform(d,0,0,d,0,0);}
function refreshProbe() {
  const p=state.pipeline,id=state.selected,l=state.recipe?.layers.find(x=>x.id===id),probe=p?.probes[id];
  $('probeTitle').textContent=id==='mel'?'Log-mel features':l?`${id.toUpperCase()} · ${l.in_channels} → ${l.out_channels}`:'Select a layer';
  $('probeDescription').textContent=descriptions[id]||'Actual intermediate output for this compatible convolution recipe.';
  $('probeStats').replaceChildren();if(l){for(const [label,value] of [['Kernel',l.kernel],['Dilation',l.dilation],['Parameters',l.parameters.toLocaleString()],['Cache',`${l.in_channels*l.history_frames*4} B`]]){const div=document.createElement('div');div.textContent=`${label}: ${value}`;$('probeStats').append(div);}}
  const canvas=$('heat'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  $('contributions').replaceChildren();
  if(!probe){$('formula').textContent='No activation available yet. Step the waveform.';return;}
  const vals=probe.data;let min=Infinity,max=-Infinity;for(const v of vals){min=Math.min(min,v);max=Math.max(max,v);}for(let c=0;c<probe.channels;c++)for(let t=0;t<probe.frames;t++){const value=vals[c*probe.frames+t],u=(value-min)/(max-min||1);ctx.fillStyle=`rgb(${Math.round(18+62*u)},${Math.round(30+185*u)},${Math.round(48+142*u)})`;ctx.fillRect(t*canvas.width/probe.frames,c*canvas.height/probe.channels,canvas.width/probe.frames+1,canvas.height/probe.channels+1);}
  const dim=document.createElement('div');dim.textContent=`Tensor [1, ${probe.channels}, ${probe.frames}] · min ${min.toFixed(3)} / max ${max.toFixed(3)}`;$('probeStats').append(dim);
  $('channel').max=probe.channels-1;$('frame').max=probe.frames-1;
  const explain=p.explain(id,Number($('channel').value)||0,Number($('frame').value)||0);
  if(!explain){$('formula').textContent='log(max(mel_filter × |FFT(Hann × PCM)|², 1e-8))';return;}
  $('formula').textContent=`${explain.activation==='relu'?'ReLU':'Linear'}(bias + Σ input × weight)\nbias = ${explain.bias.toFixed(6)}\ny[${explain.channel},${explain.time}] = ${explain.output.toFixed(6)}\nShowing 12 of ${explain.total_terms} contributions`;
  const table=document.createElement('table'),head=document.createElement('tr');for(const title of ['In','Lag','x','w','x × w']){const th=document.createElement('th');th.textContent=title;head.append(th);}table.append(head);for(const term of explain.contributions){const tr=document.createElement('tr');for(const v of [term.input_channel,term.lag_frames,term.input.toFixed(3),term.weight.toFixed(3),term.product.toFixed(3)]){const td=document.createElement('td');td.textContent=v;tr.append(td);}table.append(tr);}$('contributions').append(table);
}
function refresh() {
  const p=state.pipeline;if(!p)return;
  $('phase').textContent=state.final?'Endpoint':p.current?`Layer ${p.stage+1}/${p.layers.length}`:p.decoder.frames?'Streaming':'Ready';
  $('health').textContent=state.final?`No action · ${state.final.reason}`:'Uncalibrated catalog · provisional outputs only';
  $('clock').textContent=`${(state.position/16000).toFixed(3)} / ${(state.pcm.length/16000).toFixed(3)} s`;
  $('progressFill').style.width=`${state.pcm.length?state.position/state.pcm.length*100:0}%`;
  $('phones').textContent=p.decoder.greedy.map(i=>p.r.vocabulary[i]).join(' ')||'No phoneme evidence yet.';
  $('result').textContent=JSON.stringify(state.final||p.last||{status:'waiting',action_executed:false},null,2);
  if(p.costs.length)$('cost').textContent=`Last chunk: ${p.costs.at(-1).toFixed(2)} ms · ${p.decoder.nodes.toLocaleString()} decoder states`;
  const now=performance.now();if(state.final||(!state.running&&!state.microphone)||now-state.lastDraw>=120){state.lastDraw=now;if(scene){scene.probes=p.probes;scene.draw();}refreshProbe();plotWave();}
}
function renderRows() {
  const query=$('search').value.toLowerCase(),trial=$('trial').value,filter=$('outcome').value;
  state.filtered=state.rows.filter(r=>(!trial||`${r.trial}:${r.dataset}`===trial)&&(!query||JSON.stringify([r.verbatim,r.id,r.expected,r.raw_output]).toLowerCase().includes(query))&&(!filter||(filter==='raw_wrong'&&r.expected&&!r.raw_exact)||(filter==='raw_correct'&&r.raw_exact)||(filter==='wrong_accept'&&r.accepted&&!r.raw_exact)||r.outcome===filter));
  const pages=Math.max(1,Math.ceil(state.filtered.length/50));state.page=Math.min(state.page,pages-1);$('rows').replaceChildren();
  for(const r of state.filtered.slice(state.page*50,state.page*50+50)){const tr=document.createElement('tr');for(const v of [`${r.trial} / ${r.dataset}\n${r.voice}`,r.verbatim,JSON.stringify(r.expected),JSON.stringify(r.raw_output),`${r.accepted?'DIAGNOSTIC ACCEPT':'REJECT / CLARIFY'}\n${r.outcome}`]){const td=document.createElement('td');td.textContent=v;tr.append(td);}$('rows').append(tr);}
  $('pageInfo').textContent=`${state.filtered.length.toLocaleString()} rows · page ${state.page+1} / ${pages}`;$('prev').disabled=state.page===0;$('next').disabled=state.page>=pages-1;
}
async function loadEvidence() {try {const r=await fetch('../final_suite/data/evidence.json.gz');if(!r.ok)throw new Error('Restore generated evidence to load the exhaustive table.');const stream=r.body.pipeThrough(new DecompressionStream('gzip'));const evidence=JSON.parse(await new Response(stream).text());state.rows=evidence.rows;for(const trial of [...new Set(state.rows.map(r=>`${r.trial}:${r.dataset}`))]){const o=document.createElement('option');o.value=trial;o.textContent=trial;$('trial').append(o);}$('evidenceStatus').textContent=`${state.rows.length.toLocaleString()} evaluation rows`;renderRows();}catch(e){$('evidenceStatus').textContent=e.message;}}
async function startMicrophone() {
  pause();if(state.microphone){await stopMicrophone();return;}
  if(!navigator.mediaDevices?.getUserMedia)throw new Error('Microphone needs localhost or HTTPS and browser permission.');
  const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
  const context=new AudioContext({sampleRate:16000});
  if(context.sampleRate!==16000){stream.getTracks().forEach(t=>t.stop());await context.close();throw new Error('This browser did not provide 16 kHz AudioContext. Record offline and resample once.');}
  try {const workletURL=window.inlineCaptureWorklet?URL.createObjectURL(new Blob([window.inlineCaptureWorklet],{type:'text/javascript'})):'capture-worklet.mjs';await context.audioWorklet.addModule(workletURL);if(window.inlineCaptureWorklet)URL.revokeObjectURL(workletURL);const source=context.createMediaStreamSource(stream),node=new AudioWorkletNode(context,'vcm-capture');source.connect(node);node.connect(context.destination);state.pipeline.reset();state.final=null;state.position=0;state.pcm=new Float32Array(0);state.microphone={stream,context,source,node,pre:[],preCount:0,active:false,silence:0,pending:new Float32Array(0)};
    node.port.onmessage=e=>{try {const message=e.data;if(message.rate!==16000)throw new Error('Capture sample-rate changed');const mic=state.microphone;if(!mic)return;if(mic.nextFrame!==undefined&&message.frame!==mic.nextFrame)throw new Error('Capture sequence gap');mic.nextFrame=message.frame+message.pcm.length;if(mic.context.currentTime-message.frame/16000>.25)throw new Error('Live display fell behind capture; reset and use recorded inspection');handleMicrophone(message.pcm);}catch(err){state.pipeline.fault=err.message;finalize('capture_fault');stopMicrophone();error(err.message);}};
    await context.resume();$('mic').textContent='Stop microphone';$('micHint').textContent='Listening locally. Speak one command; stop after the endpoint.';for(const id of ['play','step','chunk','speed','load','reset'])$(id).disabled=true;
  }catch(e){stream.getTracks().forEach(t=>t.stop());await context.close();throw e;}
}
function handleMicrophone(pcm) {
  const m=state.microphone;if(!m)return;const pending=new Float32Array(m.pending.length+pcm.length);pending.set(m.pending);pending.set(pcm,m.pending.length);m.pending=pending;
  while(m.pending.length>=640){const block=m.pending.slice(0,640);m.pending=m.pending.slice(640);const rms=Math.sqrt(block.reduce((s,x)=>s+x*x,0)/block.length);
    if(!m.active){m.pre.push(block);m.preCount+=block.length;while(m.preCount>3200){m.preCount-=m.pre.shift().length;}if(rms<.012)continue;m.active=true;for(const b of m.pre){state.pipeline.enqueue(b);state.pipeline.drain();const data=new Float32Array(state.pcm.length+b.length);data.set(state.pcm);data.set(b,state.pcm.length);state.pcm=data;}m.pre=[];
    }else {state.pipeline.enqueue(block);state.pipeline.drain();const data=new Float32Array(state.pcm.length+block.length);data.set(state.pcm);data.set(block,state.pcm.length);state.pcm=data;}
    state.position=state.pcm.length;m.silence=rms<.012?m.silence+640:0;refresh();
    if(m.silence>=6400){finalize('silence');stopMicrophone();return;}
  }
}
async function stopMicrophone() {const m=state.microphone;if(!m)return;state.microphone=null;m.node.port.onmessage=null;m.source.disconnect();m.node.disconnect();m.stream.getTracks().forEach(t=>t.stop());await m.context.close();if(!state.pipeline.finished&&m.active)finalize('push_to_talk_release');$('mic').textContent='Start microphone';for(const id of ['play','step','chunk','speed','load','reset'])$(id).disabled=false;}
async function hotSwap(files) {const jsons=[];for(const file of files)if(file.name.endsWith('.json'))jsons.push({name:file.name,value:JSON.parse(await file.text())});const recipe=jsons.find(x=>x.value.schema_version==='vcm-inspector-1')?.value,graph=jsons.find(x=>Array.isArray(x.value.forms))?.value,weight=files.find(f=>f.name.endsWith('.bin'));if(!recipe||!graph||!weight)throw new Error('Select compatible recipe.json, weights.bin, and catalog.json together. An ONNX file alone has no frontend, catalog, or layer-probe contract.');const onnx=files.find(f=>f.name.endsWith('.onnx'));if(onnx&&await sha256(await onnx.arrayBuffer())!==recipe.source_onnx_sha256)throw new Error('ONNX identity does not match recipe');await loadModel(recipe,await weight.arrayBuffer(),graph);reset();}
$('load').onclick=guard(()=>loadSample($('sample').value));$('play').onclick=guard(play);$('reset').onclick=guard(reset);$('step').onclick=guard(stepLayer);$('chunk').onclick=guard(stepChunk);$('mic').onclick=guard(startMicrophone);
$('audioUpload').onchange=guard(async e=>{if(state.microphone)await stopMicrophone();const file=e.target.files[0];if(file){setAudio(parseWav(await file.arrayBuffer()));$('verbatim').textContent=file.name;}});
$('modelUpload').onchange=guard(e=>hotSwap([...e.target.files]));$('speed').oninput=()=>{$('speedLabel').textContent=`${$('speed').value}×`;};for(const id of ['channel','frame'])$(id).oninput=refreshProbe;
$('downloadTensor').onclick=()=>{const p=state.pipeline?.probes[state.selected];if(p)download(`${state.selected}-tensor.json`,JSON.stringify({...p,data:Array.from(p.data)},null,2));};
for(const id of ['search','trial','outcome'])$(id).addEventListener('input',()=>{state.page=0;renderRows();});$('prev').onclick=()=>{state.page--;renderRows();};$('next').onclick=()=>{state.page++;renderRows();};$('exportRows').onclick=()=>download('filtered-commands.json',JSON.stringify(state.filtered,null,2));
try {scene=new TensorScene($('scene'),$('layerLabels'),select);$('rendererStatus').textContent=scene.gl?'WebGL · GPU rendering':'3D canvas fallback · WebGL unavailable';}catch(e){$('rendererStatus').textContent='2D probes · WebGL unavailable';}
window.addEventListener('resize',plotWave);window.addEventListener('pagehide',()=>{if(state.microphone)state.microphone.stream.getTracks().forEach(t=>t.stop());});
window.vcmTest={state,loadSample,stepLayer,stepChunk,reset,hotSwap,runToEnd:async()=>{let n=0;while(startNextChunk()){while(state.pipeline.current)state.pipeline.stepLayer();if(++n%8===0)await new Promise(r=>setTimeout(r,0));}finalize('push_to_talk_release');return state.final;},renderer:()=>scene?.gl?'webgl':scene?'canvas3d':null,version:1};
(async()=>{try {const [r,w,g,examples]=await Promise.all([fetch('assets/recipe.json').then(r=>r.json()),fetch('assets/weights.bin').then(r=>r.arrayBuffer()),fetch('assets/catalog.json').then(r=>r.json()),fetch('assets/examples.json').then(r=>r.json())]);state.samples=examples;for(const e of examples){const o=document.createElement('option');o.value=e.file;o.textContent=e.name;$('sample').append(o);}await loadModel(r,w,g);await loadSample('song_by_artist.wav');window.vcmReady=true;}catch(e){error(e.message);$('phase').textContent='Load failed';}loadEvidence();})();
