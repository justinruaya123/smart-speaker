import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {validateProof, verifyWavBytes} from '../../scripts/optionb-parity-inputs.mjs';

const hash=(bytes:Buffer|string)=>createHash('sha256').update(bytes).digest('hex');
function proof(){
  const expected_keys=[['holdout','one'],['holdout','two']];
  return {format:'optionb-native-input-proof-v2',protocol_sha256:'a'.repeat(64),
    wav_and_decoded_pcm_verified:true,expected_count:2,expected_keys,
    expected_keys_sha256:hash(JSON.stringify(expected_keys)),
    rows:expected_keys.map(([split,id])=>({split,id,sha256:hash('actual WAV bytes'),pcm_sha256:'b'.repeat(64)}))};
}
test('native replay rejects empty, missing, duplicated or substituted canonical human rows',()=>{
  const original=proof();assert.equal(validateProof(original).length,2);
  const missing=proof();missing.rows.pop();
  const duplicate=proof();duplicate.rows[1]=duplicate.rows[0];
  const replaced=proof();replaced.rows[0]={...replaced.rows[0],id:'different'};
  const empty={...proof(),rows:[],expected_keys:[],expected_count:0};
  for(const bad of [missing,duplicate,replaced,empty])assert.throws(()=>validateProof(bad));
});
test('native replay refuses missing PCM preflight or changed source cohort proof',()=>{
  assert.throws(()=>validateProof({...proof(),wav_and_decoded_pcm_verified:false}));
  assert.throws(()=>validateProof({...proof(),expected_keys_sha256:'0'.repeat(64)}));
});
test('the exact WAV bytes sent to the gateway must match the preflight identity',()=>{
  const row=proof().rows[0];
  const original=Buffer.from('actual WAV bytes');
  assert.equal(verifyWavBytes(original,row),original);
  assert.throws(()=>verifyWavBytes(Buffer.from('changed WAV bytes'),row));
});
