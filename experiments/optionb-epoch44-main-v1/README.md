# Additive epoch44 main integration

Base main: `2c669589229619d0e282ebda621dc9ae3002951e`.
Original source: `3c05b3beb46f06c3108089e0d8c79474670c6cb7`.

Iris remains the default. This integration publishes the exact original OptionB epoch44 graph, recovery checkpoint and metadata, plus an opt-in verifier and explicit commands; it does not merge the historical research tree into Iris.

## Verified scope

| Gate | Observed result |
| --- | --- |
| Original distribution | All 15 frozen payloads match the pinned original snapshot. |
| Original graph software load | Finite `[1, 32]` output with CPU ONNX Runtime; no acoustic-quality or training claim. |
| Retained GPU rebuild | Independent expected checkpoint/model/prediction bytes and all 48 epoch results match; 4,608 CUDA optimizer updates, selected GPU epoch 28. The main checker itself performs no training. |
| Original CPU44 equality | **Still failed.** GPU28 is not relabelled CPU44; the original comparison failure and absent success lock remain required. |
| Epoch44 software cases | **227 passed**, including rebound-evidence and CUDA-provenance regressions. |
| Existing main baseline | Unchanged main: 185 passed, 37 failed. Additive candidate: 412 passed, the same 37 failed. No changed outcomes for existing tests; 445 prior application files remain byte-identical. This is not an all-green Iris reproduction claim. |
| Independent review closure | P2 closed; six representative tests passed and all 43 contradictory/missing-provenance overlays were rejected against actual retained-fit evidence. |
| Windows checkout byte preservation | All 27 bound payload/source/evidence files present at the CRLF probe matched staged bytes; stdlib artifact verification passed. |

## Evidence

- `review-v1.json` and `review-v1-source/`: failed original review and exact superseded source snapshots. `.snapshot` files are historical data, not executable adapters.
- `review-v2.json`: independent closure tied to the final reviewed diff and verifier/test hashes.
- `verification-v2.json`: actual artifact, graph-load and retained-GPU verification commands and outputs.
- `baseline-regression-v1.json`: pre-correction Windows comparison.
- `baseline-regression-v2.json`: post-correction baseline-aware counts, source hashes and archive bindings.
- `baseline-tests-v1.xml.gz` and `software-tests-v3.xml.gz`: actual baseline/candidate JUnit output. The 37 existing Iris failures remain visible.

Published original-source evidence already records two independent full GPU builds with their own source/environment/data acquisition. This main integration does not claim a new third fit, exact GPU-to-CPU reproduction, held-out acoustic acceptance, fresh human qualification, Pi testing, live microphone/provider operation or enabled actions.

Commands and stack limits: [`docs/REPRODUCIBILITY-optionb-epoch44.md`](../../docs/REPRODUCIBILITY-optionb-epoch44.md).
