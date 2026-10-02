/* Real, dependency-free CPU/WASM inference. Rendering is a separate concern. */
import embeddedKernel from './kernel_base64.mjs';

export const finite = values => {for (const v of values) if (!Number.isFinite(v)) throw new Error('Non-finite tensor/PCM');};
export function canonical(v) {if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`; if(v && typeof v==='object')return `{${Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+canonical(v[k])).join(',')}}`;return JSON.stringify(v);}
export async function sha256(bytes) {
  if(globalThis.crypto?.subtle){const raw=await crypto.subtle.digest('SHA-256',bytes);return [...new Uint8Array(raw)].map(v=>v.toString(16).padStart(2,'0')).join('');}
  return softwareSha256(new Uint8Array(bytes));
}
// Standard SHA-256, used only when SubtleCrypto is unavailable (e.g. DOM fixtures).
export function softwareSha256(input) {
  const K=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
  const h=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
  const bytes=new Uint8Array(Math.ceil((input.length+9)/64)*64);bytes.set(input);bytes[input.length]=128;
  const view=new DataView(bytes.buffer),bits=BigInt(input.length)*8n;view.setBigUint64(bytes.length-8,bits,false);
  const rotate=(x,n)=>(x>>>n)|(x<<(32-n)),w=new Int32Array(64);
  for(let offset=0;offset<bytes.length;offset+=64){for(let i=0;i<16;i++)w[i]=view.getInt32(offset+i*4,false);for(let i=16;i<64;i++){const x=w[i-15],y=w[i-2];w[i]=(w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>>3))+w[i-7]+(rotate(y,17)^rotate(y,19)^(y>>>10)))|0;}
    let [a,b,c,d,e,f,g,z]=h;for(let i=0;i<64;i++){const t1=(z+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+K[i]+w[i])|0,t2=((rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c)))|0;z=g;g=f;f=e;e=(d+t1)|0;d=c;c=b;b=a;a=(t1+t2)|0;}[a,b,c,d,e,f,g,z].forEach((v,i)=>h[i]=(h[i]+v)|0);
  }
  return h.map(v=>(v>>>0).toString(16).padStart(8,'0')).join('');
}
export function validateRecipe(r) {
  if(r.schema_version!=='vcm-inspector-1'||r.sample_rate!==16000||r.window!==400||r.hop!==160||r.fft!==512||r.mel_bins!==40)throw new Error('Unsupported recipe/frontend contract');
  if(![4,8].includes(r.chunk_frames)||r.maximum_seconds!==8||r.acceptance_enabled!==false||r.action_execution_enabled!==false)throw new Error('Unsafe or unsupported streaming contract');
  if(!Array.isArray(r.layers)||r.layers.length<1||r.layers.length>12)throw new Error('Invalid layers');
  if(!Array.isArray(r.vocabulary)||r.vocabulary.length<2||r.vocabulary.length>128||r.vocabulary[0]!=='<blank>'||new Set(r.vocabulary).size!==r.vocabulary.length)throw new Error('Invalid vocabulary');
  if(r.mean.length!==40||r.std.length!==40||r.std.some(v=>v<=0)||r.mel_filters.length!==40*257||r.window_coefficients.length!==400)throw new Error('Normalization/filter dimensions');
  [r.mean,r.std,r.mel_filters,r.window_coefficients].forEach(finite);
  let ci=40, offset=0;
  const ids=new Set();
  for(const l of r.layers){
    if(l.type!=='Conv1d'||l.in_channels!==ci||!Number.isInteger(l.out_channels)||l.out_channels<1||l.out_channels>256||l.stride!==1||![1,3,5,7].includes(l.kernel)||![1,2,4,8,16,32].includes(l.dilation)||!['relu','linear'].includes(l.activation))throw new Error('Unsupported convolution');
    if(ids.has(l.id)||typeof l.id!=='string')throw new Error('Duplicate layer ID');ids.add(l.id);
    if(l.history_frames!==(l.kernel-1)*l.dilation||l.weight_offset!==offset||l.bias_offset!==offset+ci*l.out_channels*l.kernel)throw new Error('Weight/cache layout mismatch');
    offset=l.bias_offset+l.out_channels;ci=l.out_channels;
  }
  if(ci!==r.vocabulary.length)throw new Error('Output vocabulary mismatch');
  return offset;
}
export function parseWav(buffer) {
  const v=new DataView(buffer), text=(o,n)=>String.fromCharCode(...new Uint8Array(buffer,o,n));
  if(v.byteLength<44||text(0,4)!=='RIFF'||text(8,4)!=='WAVE')throw new Error('Use a mono PCM16 or float32 WAV');
  let format, channels, rate, bits, block, start, bytes;
  for(let o=12;o+8<=v.byteLength;){const name=text(o,4), n=v.getUint32(o+4,true);if(o+8+n>v.byteLength)throw new Error('Truncated WAV');
    if(name==='fmt '){if(n<16)throw new Error('Short fmt');format=v.getUint16(o+8,true);channels=v.getUint16(o+10,true);rate=v.getUint32(o+12,true);block=v.getUint16(o+20,true);bits=v.getUint16(o+22,true);}
    if(name==='data'){start=o+8;bytes=n;}o+=8+n+(n&1);}
  if(channels!==1||start===undefined||!((format===1&&bits===16)||(format===3&&bits===32))||block!==bits/8||bytes%block)throw new Error('Use mono PCM16 or IEEE float32 WAV');
  if(rate!==16000)throw new Error('WAV must be 16 kHz. Convert once with prepare_wav.py; no per-chunk resampling.');
  if(bytes/block>16000*7.8)throw new Error('Maximum input duration is 7.8 seconds');
  const pcm=new Float32Array(bytes/block);
  for(let i=0;i<pcm.length;i++)pcm[i]=format===1?v.getInt16(start+i*2,true)/32768:v.getFloat32(start+i*4,true);
  finite(pcm);return pcm;
}
function fftReal(input) {
  const re=new Float64Array(512), im=new Float64Array(512);re.set(input);
  for(let i=1,j=0;i<512;i++){let bit=256;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){const a=re[i];re[i]=re[j];re[j]=a;}}
  for(let len=2;len<=512;len*=2){const angle=-2*Math.PI/len, cr=Math.cos(angle),si=Math.sin(angle);
    for(let i=0;i<512;i+=len){let wr=1,wi=0;for(let j=0;j<len/2;j++){const a=i+j,b=a+len/2,vr=re[b]*wr-im[b]*wi,vi=re[b]*wi+im[b]*wr;re[b]=re[a]-vr;im[b]=im[a]-vi;re[a]+=vr;im[a]+=vi;const next=wr*cr-wi*si;wi=wr*si+wi*cr;wr=next;}}}
  const power=new Float64Array(257);for(let k=0;k<257;k++)power[k]=re[k]*re[k]+im[k]*im[k];return power;
}
export class Frontend {
  constructor(r){this.r=r;this.buffer=new Float32Array(0);}
  push(pcm){finite(pcm);const joined=new Float32Array(this.buffer.length+pcm.length);joined.set(this.buffer);joined.set(pcm,this.buffer.length);const n=Math.max(0,1+Math.floor((joined.length-400)/160)),out=new Float32Array(40*n),windowed=new Float32Array(400);
    for(let t=0;t<n;t++){for(let k=0;k<400;k++)windowed[k]=joined[t*160+k]*this.r.window_coefficients[k];const power=fftReal(windowed);
      for(let m=0;m<40;m++){let e=0;for(let k=0;k<257;k++)e+=power[k]*this.r.mel_filters[m*257+k];out[m*n+t]=Math.log(Math.max(e,1e-8));}}
    this.buffer=joined.slice(n*160);return {data:out,frames:n,channels:40};}
}
export function appendFrames(a,na,b,nb,channels){const out=new Float32Array(channels*(na+nb));for(let c=0;c<channels;c++){out.set(a.subarray(c*na,(c+1)*na),c*(na+nb));out.set(b.subarray(c*nb,(c+1)*nb),c*(na+nb)+na);}return out;}
export function logSoftmax(x,channels,frames){const out=new Float64Array(channels*frames);for(let t=0;t<frames;t++){let max=-Infinity;for(let c=0;c<channels;c++)max=Math.max(max,x[c*frames+t]);let sum=0;for(let c=0;c<channels;c++)sum+=Math.exp(x[c*frames+t]-max);const z=max+Math.log(sum);for(let c=0;c<channels;c++)out[t*channels+c]=x[c*frames+t]-z;}return out;}
export class WasmKernel {
  static async create(){const bytes=Uint8Array.from(atob(embeddedKernel),c=>c.charCodeAt(0));const {instance}=await WebAssembly.instantiate(bytes,{env:{exp:Math.exp,log1p:Math.log1p}});return new WasmKernel(instance);}
  constructor(instance){this.instance=instance;this.memory=instance.exports.memory;this.cursor=Number(instance.exports.__heap_base.value);}
  alloc(Type,n){const size=n*Type.BYTES_PER_ELEMENT;this.cursor=(this.cursor+15)&~15;if(!Number.isSafeInteger(size)||size<0||this.cursor+size>this.memory.buffer.byteLength)throw new Error('Model/graph exceeds bounded WASM memory');const a=new Type(this.memory.buffer,this.cursor,n);this.cursor+=size;return a;}
}
export class Decoder {
  constructor(k,graph,symbols){if(!graph.forms?.length||graph.forms.length>10000)throw new Error('Catalog form limit');this.k=k;this.symbols=symbols;this.graph=graph;const parent=[0],label=[0],children=[new Map()],term=[];
    for(const form of graph.forms){if(!Array.isArray(form.target)||form.target.length>256||!form.target.length)throw new Error('Bad phone target');let n=0;for(const p of form.target){if(!Number.isInteger(p)||p<1||p>=symbols)throw new Error('Unknown/blank phone target');if(!children[n].has(p)){children[n].set(p,parent.length);parent.push(n);label.push(p);children.push(new Map());}n=children[n].get(p);}term.push(n);}
    if(parent.length>200000)throw new Error('Catalog state limit');this.nodes=parent.length;this.parents=k.alloc(Int32Array,this.nodes);this.labels=k.alloc(Int32Array,this.nodes);this.parents.set(parent);this.labels.set(label);this.term=term;
    this.pb=k.alloc(Float64Array,this.nodes);this.pn=k.alloc(Float64Array,this.nodes);this.qb=k.alloc(Float64Array,this.nodes);this.qn=k.alloc(Float64Array,this.nodes);this.totals=k.alloc(Float64Array,this.nodes);this.lp=k.alloc(Float64Array,symbols);
    this.groups=new Map();this.formGroups=[];this.commands=[];
    for(const f of graph.forms){const key=canonical(f.command);if(!this.groups.has(key)){this.groups.set(key,this.commands.length);this.commands.push(f.command);}this.formGroups.push(this.groups.get(key));}this.reset();}
  reset(){this.pb.fill(-Infinity);this.pn.fill(-Infinity);this.pb[0]=0;this.frames=0;this.greedy=[];this.last=0;}
  push(lp,frames){finite(lp);for(let t=0;t<frames;t++){const frame=lp.subarray(t*this.symbols,(t+1)*this.symbols);this.lp.set(frame);this.k.instance.exports.ctc(this.parents.byteOffset,this.labels.byteOffset,this.lp.byteOffset,this.pb.byteOffset,this.pn.byteOffset,this.qb.byteOffset,this.qn.byteOffset,this.totals.byteOffset,this.nodes,this.symbols);[this.pb,this.qb]=[this.qb,this.pb];[this.pn,this.qn]=[this.qn,this.pn];let best=0;for(let c=1;c<this.symbols;c++)if(frame[c]>frame[best])best=c;if(best!==0&&best!==this.last)this.greedy.push(best);this.last=best;this.frames++;}return this.rank();}
  rank(){const add=(a,b)=>a===-Infinity?b:b===-Infinity?a:Math.max(a,b)+Math.log1p(Math.exp(-Math.abs(a-b)));const scores=new Float64Array(this.commands.length);scores.fill(-Infinity);
    for(let i=0;i<this.term.length;i++){const node=this.term[i],group=this.formGroups[i];scores[group]=Math.max(scores[group],add(this.pb[node],this.pn[node]));}
    const order=Array.from(scores,(_,i)=>i).sort((a,b)=>scores[b]-scores[a]);const best=scores[order[0]];const ties=Number.isFinite(best)?order.filter(i=>Math.abs(scores[i]-best)<=1e-7).map(i=>this.commands[i]):[];
    return {candidate:Number.isFinite(best)?this.commands[order[0]]:null,score:Number.isFinite(best)?best:null,margin:Number.isFinite(best)&&Number.isFinite(scores[order[1]])?best-scores[order[1]]:null,ties:ties.length>1?ties:[],top:order.slice(0,3).map(i=>({command:this.commands[i],score:Number.isFinite(scores[i])?scores[i]:null})),frames:this.frames,status:'tentative',action_executed:false};}
}
export class Pipeline {
  static async create(r,weights,graph){const n=validateRecipe(r);if(weights.byteLength!==n*4||await sha256(weights)!==r.weights_sha256)throw new Error('Weight bytes/hash mismatch');if(canonical(graph.phone_inventory)!==canonical(r.vocabulary)||graph.graph_hash!==r.graph_hash)throw new Error('Catalog identity/vocabulary mismatch');if(await sha256(new TextEncoder().encode(canonical(graph)))!==r.catalog_body_sha256)throw new Error('Catalog body hash mismatch');const k=await WasmKernel.create();return new Pipeline(r,weights,graph,k);}
  constructor(r,weights,graph,k){this.r=r;this.k=k;this.graph=graph;const w=new Float32Array(weights);finite(w);this.layers=r.layers.map(l=>{const weight=k.alloc(Float32Array,l.bias_offset-l.weight_offset),bias=k.alloc(Float32Array,l.out_channels);weight.set(w.subarray(l.weight_offset,l.bias_offset));bias.set(w.subarray(l.bias_offset,l.bias_offset+l.out_channels));return {...l,weight,bias,x:k.alloc(Float32Array,l.in_channels*(l.history_frames+r.chunk_frames)),y:k.alloc(Float32Array,l.out_channels*r.chunk_frames),cache:new Float32Array(l.in_channels*l.history_frames)};});this.decoder=new Decoder(k,graph,r.vocabulary.length);this.reset();}
  reset(){this.frontend=new Frontend(this.r);this.layers.forEach(l=>l.cache.fill(0));this.decoder.reset();this.pending=new Float32Array(0);this.pendingFrames=0;this.samples=0;this.probes={};this.last=null;this.current=null;this.stage=-1;this.finished=false;this.fault=null;this.costs=[];}
  enqueue(pcm){try{if(this.finished||this.fault)throw new Error('Reset after endpoint/capture fault');if(this.samples+pcm.length>128000)throw new Error('Eight-second capture limit');const start=performance.now();const f=this.frontend.push(pcm);this.samples+=pcm.length;this.probes.waveform={data:pcm.slice(),channels:1,frames:pcm.length};if(f.frames){this.probes.mel={...f,data:f.data.slice()};const normalized=f.data.slice();for(let c=0;c<40;c++)for(let t=0;t<f.frames;t++)normalized[c*f.frames+t]=(normalized[c*f.frames+t]-this.r.mean[c])/this.r.std[c];this.pending=appendFrames(this.pending,this.pendingFrames,normalized,f.frames,40);this.pendingFrames+=f.frames;}this.frontendMs=performance.now()-start;}catch(e){this.fault=e.message;throw e;}}
  startChunk(flush=false){if(this.fault||this.finished)throw new Error('Reset after endpoint/capture fault');if(this.current)throw new Error('Finish current chunk');const T=this.r.chunk_frames;if(!this.pendingFrames||(!flush&&this.pendingFrames<T))return false;this.valid=Math.min(T,this.pendingFrames);this.current=new Float32Array(40*T);for(let c=0;c<40;c++)this.current.set(this.pending.subarray(c*this.pendingFrames,c*this.pendingFrames+this.valid),c*T);const remain=this.pendingFrames-this.valid,rest=new Float32Array(40*remain);for(let c=0;c<40;c++)rest.set(this.pending.subarray(c*this.pendingFrames+this.valid,(c+1)*this.pendingFrames),c*remain);this.pending=rest;this.pendingFrames=remain;this.stage=0;this.chunkComputeMs=0;return true;}
  stepLayer(){if(!this.current||this.fault)throw new Error('No complete chunk or capture fault');const T=this.r.chunk_frames,l=this.layers[this.stage],L=l.history_frames+T,start=performance.now();for(let c=0;c<l.in_channels;c++){l.x.set(l.cache.subarray(c*l.history_frames,(c+1)*l.history_frames),c*L);l.x.set(this.current.subarray(c*T,(c+1)*T),c*L+l.history_frames);}this.k.instance.exports.conv(l.x.byteOffset,l.weight.byteOffset,l.bias.byteOffset,l.y.byteOffset,l.in_channels,l.out_channels,l.kernel,l.dilation,T,l.activation==='relu'?1:0);finite(l.y);for(let c=0;c<l.in_channels;c++)l.cache.set(l.x.subarray(c*L+T,(c+1)*L),c*l.history_frames);this.probes[l.id]={data:l.y.slice(),channels:l.out_channels,frames:T,valid_frames:this.valid,milliseconds:performance.now()-start};this.current=l.y;this.stage++;
    if(this.stage===this.layers.length){const all=logSoftmax(this.current,l.out_channels,T);this.probes.posterior={data:Float32Array.from(all,x=>Math.exp(x)),channels:T,frames:l.out_channels};this.last=this.decoder.push(all.subarray(0,this.valid*l.out_channels),this.valid);this.last.audio_position_s=(400+160*(this.decoder.frames-1))/16000;this.current=null;this.chunkComputeMs+=performance.now()-start;this.costs.push(this.chunkComputeMs);return {done:true,layer:l.id,result:this.last};}this.chunkComputeMs+=performance.now()-start;return {done:false,layer:l.id};}
  drain(){const events=[];while(this.startChunk()){let step;do{step=this.stepLayer();}while(!step.done);events.push(step.result);}return events;}
  finish(reason='push_to_talk_release'){if(this.finished)throw new Error('Already finalized');if(!this.fault){if(this.current)while(this.current)this.stepLayer();while(this.startChunk(true))while(this.current)this.stepLayer();}this.finished=true;const why=this.fault?'capture_fault':!this.last?.candidate?'incomplete':this.last.ties.length?'ambiguous_entity':!['silence','push_to_talk_release'].includes(reason)?'capture_not_complete':'uncalibrated';return {...(this.last??{}),status:'needs_clarification',reason:why,action_executed:false,endpoint_reason:reason};}
  explain(id,channel=0,time=0){const l=this.layers.find(x=>x.id===id);if(!l)return null;const T=this.r.chunk_frames,L=l.history_frames+T;channel=Math.max(0,Math.min(l.out_channels-1,channel));time=Math.max(0,Math.min(T-1,time));const terms=[];for(let c=0;c<l.in_channels;c++)for(let j=0;j<l.kernel;j++){const x=l.x[c*L+time+j*l.dilation],w=l.weight[(channel*l.in_channels+c)*l.kernel+j];terms.push({input_channel:c,tap:j,lag_frames:l.history_frames-j*l.dilation,input:x,weight:w,product:x*w});}terms.sort((a,b)=>Math.abs(b.product)-Math.abs(a.product));return {layer:id,channel,time,bias:l.bias[channel],activation:l.activation,output:l.y[channel*T+time],contributions:terms.slice(0,12),total_terms:terms.length,cache_bytes:l.cache.byteLength};}
}
