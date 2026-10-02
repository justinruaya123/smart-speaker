// Emit actual WASM tensors for the portable PCV ONNX comparison.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [moduleDir,assetDir,wav,output]=process.argv.slice(2);
const {Pipeline,parseWav}=await import(pathToFileURL(path.resolve(moduleDir,'core.mjs')));
const read=name=>{const b=fs.readFileSync(name);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const recipe=JSON.parse(fs.readFileSync(path.join(assetDir,'recipe.json')));
const graph=JSON.parse(fs.readFileSync(path.join(assetDir,'catalog.json')));
const pcm=parseWav(read(wav));
const pipeline=await Pipeline.create(recipe,read(path.join(assetDir,'weights.bin')),graph);
const chunks=[];
function drain(flush=false){
  while(pipeline.startChunk(flush)){
    const features=Array.from(pipeline.current),valid=pipeline.valid;
    while(pipeline.current)pipeline.stepLayer();
    chunks.push({features,valid,logits:Array.from(pipeline.probes.ctc.data)});
  }
}
for(let start=0;start<pcm.length;start+=640){pipeline.enqueue(pcm.subarray(start,start+640));drain();}
drain(true);
fs.writeFileSync(output,JSON.stringify({chunks,final:pipeline.finish(),source_onnx_sha256:recipe.source_onnx_sha256}));
