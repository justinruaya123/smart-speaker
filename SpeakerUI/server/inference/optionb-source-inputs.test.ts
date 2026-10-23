import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as nativeInputs from '../../scripts/optionb-source-inputs.mjs';

test('native replay keeps the exact legacy preflight arguments when no protocol is supplied',()=>{
  assert.equal(typeof nativeInputs.preflightArgs,'function');
  assert.deepEqual(nativeInputs.preflightArgs(),['-m','scripts.test.optionb_release_check','--preflight']);
});
test('native replay passes the selected portable protocol as one literal verifier argument',()=>{
  const protocol='.cache/optionb-source-repro-v1/run 01/locks/release.json';
  assert.deepEqual(nativeInputs.preflightArgs(protocol),[
    '-m','scripts.test.optionb_release_check','--preflight','--protocol',protocol,
  ]);
});
