import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Pipeline,Frontend,parseWav,sha256} from '../core.mjs';
const root=path.resolve(process.argv[2] || fileURLToPath(new URL('../../',import.meta.url)));
const r=JSON.parse(fs.readFileSync(path.join(root,'visualizer/assets/recipe.json')));
const g=JSON.parse(fs.readFileSync(path.join(root,'visualizer/assets/catalog.json')));
const read=p=>{let b=fs.readFileSync(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const examples=JSON.parse(fs.readFileSync(path.join(root,'visualizer/assets/examples.json')));
const outputs=[];
for(const ex of examples){const pcm=parseWav(read(path.join(root,'visualizer/assets',ex.file))),audio=new Float32Array(pcm.length+1600);audio.set(pcm);const p=await Pipeline.create(r,read(path.join(root,'visualizer/assets/weights.bin')),g);const f=new Frontend(r),full=f.push(audio);const out=path.join(root,'final_suite/reports/browser_parity');fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,ex.file+'.mel.f32'),Buffer.from(full.data.buffer));const lp=[];const inspect=[];
  for(let start=0;start<audio.length;start+=640){p.enqueue(audio.subarray(start,start+640));while(p.startChunk()){while(p.current){const step=p.stepLayer();if(p.decoder.frames===0)inspect.push({layer:step.layer,data:Array.from(p.probes[step.layer].data)});if(step.done)lp.push(...p.probes.ctc.data);}}}
  const before=p.decoder.frames;const final=p.finish();if(p.decoder.frames>before)lp.push(...p.probes.ctc.data);
  fs.writeFileSync(path.join(out,ex.file+'.ctc.f32'),Buffer.from(new Float32Array(lp).buffer));fs.writeFileSync(path.join(out,ex.file+'.probes.json'),JSON.stringify(inspect));
  outputs.push({file:ex.file,frames:p.decoder.frames,final,mean_chunk_ms:p.costs.reduce((a,b)=>a+b,0)/p.costs.length,max_chunk_ms:Math.max(...p.costs)});
}
fs.writeFileSync(path.join(root,'final_suite/reports/browser_math.json'),JSON.stringify(outputs,null,2));console.log(outputs.map(o=>({file:o.file,frames:o.frames,output:o.final.candidate,status:o.final.status,mean_chunk_ms:o.mean_chunk_ms})));
