/** Verification inputs only; never part of acoustic recognition or action dispatch. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const key = row => JSON.stringify([row.split,row.id]);

export function validateProof(proof) {
  assert.equal(proof.format,'optionb-native-input-proof-v2');
  assert.match(proof.protocol_sha256,/^[a-f0-9]{64}$/);
  assert.equal(proof.wav_and_decoded_pcm_verified,true);
  assert.ok(Number.isInteger(proof.expected_count) && proof.expected_count > 0,'Canonical cohort must be nonempty');
  const expected=proof.expected_keys.map(pair=>{
    assert.ok(Array.isArray(pair)&&pair.length===2&&pair.every(value=>typeof value==='string'&&value.length>0));
    return JSON.stringify(pair);
  });
  assert.equal(expected.length,proof.expected_count);
  assert.equal(new Set(expected).size,expected.length,'Duplicate canonical IDs');
  assert.equal(proof.rows.length,proof.expected_count);
  const actual=proof.rows.map(key);
  assert.equal(new Set(actual).size,actual.length,'Duplicate replay IDs');
  assert.deepEqual([...actual].sort(),[...expected].sort(),'Replay differs from pinned canonical cohort');
  const pairs=[...proof.expected_keys].sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:a[1]<b[1]?-1:a[1]>b[1]?1:0);
  assert.equal(sha(JSON.stringify(pairs)),proof.expected_keys_sha256);
  for(const row of proof.rows){
    assert.match(row.sha256,/^[a-f0-9]{64}$/);
    assert.match(row.pcm_sha256,/^[a-f0-9]{64}$/);
  }
  return proof.rows;
}

export function verifyWavBytes(bytes,row) {
  assert.equal(sha(bytes),row.sha256,'WAV bytes differ from the decoded-PCM-verified input');
  return bytes;
}
