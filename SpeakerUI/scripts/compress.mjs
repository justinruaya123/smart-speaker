import { readdir, readFile, writeFile, cp } from 'node:fs/promises';
import { gzipSync, brotliCompressSync, constants } from 'node:zlib';
import { join } from 'node:path';
async function compress(dir){for(const entry of await readdir(dir,{withFileTypes:true})){const path=join(dir,entry.name);if(entry.isDirectory())await compress(path);else if(/\.(html|js|css|svg)$/.test(entry.name)){const data=await readFile(path);await Promise.all([writeFile(path+'.gz',gzipSync(data,{level:9})),writeFile(path+'.br',brotliCompressSync(data,{params:{[constants.BROTLI_PARAM_QUALITY]:9}}))]);}}}
await compress('dist');
await cp('mockups/ambient-v6/licenses','dist/licenses',{recursive:true});
