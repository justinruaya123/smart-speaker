# Preliminary PC-v1 results

**No human speech was trained or evaluated. No final model is qualified.**

The code and experiments below were executed locally. The selected disabled baseline and unrelated research were not changed.

| Experiment | Result | Evidence limit |
|---|---:|---|
| New implementation test suite | 48 passed, 0 failed | Not the entire original repository suite |
| Original proposal consistency checks | 46 passed | Catalog/schema checks, not recorded speech |
| Full phone graph | 570 positives + 200 reject competitors; 402 typed commands | Pronunciations, not acoustic coverage |
| Cross-command exact phone collisions | 0 | Reviewed dictionary spellings do not prove accent distinguishability |
| Generated noiseless phone logits | 770/770 raw exact | Input logits are constructed from known targets, not microphone audio |
| Severe generated logit noise, SD 6 | 541/770 raw exact | All 229 failures retained; not an audio-SNR or accepted-recall result |
| Padded CTC median, 250 frames, five runs | 289.508021 ms | Desktop decoder only |
| Exact shared-prefix CTC median | 81.717550 ms | Same inputs and catalog; 3.542789× faster |
| Maximum decoder score difference | 4.884981308350689e-15 | No beam pruning or phrase removal |
| CTC probability-state storage | 572,880 → 115,888 bytes | Not total decoder memory or process RSS |
| Baseline CNN gradient smoke | Loss 52.181038 → 12.338114 over 12 updates | Generated feature tensors; not real speech learning |
| CNN parameters / neural caches | 113,064 / 32,064 bytes | Actual baseline architecture instantiated |
| Native CPU chunk parity | 1.9073486328125e-5 max error | Below unchanged 2e-5 bound; ONNX not tested |

## Failed and unavailable work remains explicit

The original oneDNN CPU chunk-parity result was 2.288818359375e-5 and failed the 2e-5 bound. The native CPU backend comparison and failed first result are both retained. No threshold was relaxed.

The first multi-noise experiment was interrupted before writing a complete trial CSV. Its partial log/benchmark remains under `reports/preliminary-v1/`. Subsequent chunked runs write one CSV row immediately per case, and completed all 770 noiseless and all 770 severe-noise cases.

Direct GitHub/Hugging Face/pip downloads failed in this sandbox. A connected GitHub source archive was downloaded, but it did not contain the required original recordings or trained weights. PyTorch CPU was available; CUDA, ONNX and ONNX Runtime were not. The real 48-test result uses byte-verified baseline frontend/model/decoder modules, not a claim that every file of a fresh clone was tested.

No baseline accepted accuracy was improved or remeasured. No human acoustic precision, recall or false-accept rate is available for this successor. No target Pi latency, microphone behavior, ONNX parity, or runtime installation is established.

## Publication scope

GitHub milestone `1a4caaf3263a0106653b8bdd33b8e3f4a4ccac27` adds only the decoder, command helper, five independent decoder tests, benchmark receipt and milestone documentation (six files). The complete source delivery and checked additive patch contain the broader implementation and all local results. Applying that patch and committing the remaining files is explicitly part of the handoff, not something already claimed to have happened.

See `CODEX_FINAL_EXECUTION.md` for fixed candidate/configurations, source and split rules, remaining CV/endpoint/capture/release integration, exact commands, and stopping conditions. New human recording, successful training/calibration, fresh sealed confirmation, and physical hardware qualification remain mandatory.
