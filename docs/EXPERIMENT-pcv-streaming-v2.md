# Streaming VCM second-pass evidence

One selected architecture: **Causal Slot-TCN-96 v2**, from scratch, with an
independent small speaker tower. Two exports: benchmark baseline plus a separate
absolute-volume track; demo with named lights/music and volume. No transcript,
phone inventory, pretrained teacher, wake/VAD model or synthesized speech engine
is introduced. This is a source prototype and architecture decision, not a
trained final model.

## What changed after the user's clarification

The priority is unfamiliar human voices and connected slots; the existing
roughly 200 ms inference is acceptable according to the user. The old endpoint
model remains the accuracy reference. Its high synthetic score is not evidence
that new human voices are accepted. The new design eliminates noncausal kernels
and large complete-command catalog scoring, but lower computational cost does
not establish better recognition.

The first in-chat causal prototype used a finite-phone output. Before committing,
this second pass replaced it with finite semantic/slot heads and learned acoustic
embeddings, so the proposed runtime does not need any phone/word transcription
component. It also separates content-invariant and speaker-discriminative
objectives into independent parameter sets. Earlier uncommitted v1 probes are
not part of this release candidate.

## Measured here

See `../experiments/pcv-streaming-v2/preflight-results.json` and
`operation-count.json` in that folder. Measurements are random-weight x86-64
CPU tests, one logical CPU and one PyTorch thread; **not speech, ONNX or Pi**.

| Item | Result |
|---|---:|
| Total parameters, all heads and speaker tower | 115,825 |
| Speaker tower parameters | 12,480 |
| FP32 parameter bytes | 463,300 |
| Explicit stream-state bytes | 103,592 |
| Conv/linear MACs per audio second, all heads | 10,473,600 |
| Benchmark frozen TorchScript 40 ms chunk median / p95 | 0.580 / 0.843 ms |
| Demo frozen TorchScript 40 ms chunk median / p95 | 0.848 / 1.111 ms |
| Benchmark eager PyTorch median / p95 | 1.440 / 1.907 ms |
| Demo eager PyTorch median / p95 | 1.377 / 1.779 ms |

Each timing row uses 1,000 warm stepped calls. Small timing differences between
eager profiles are not meaningful superiority evidence; both eager paths execute
all heads. The benchmark export passes through speaker state so dead-code
elimination can remove the unused speaker computation. Actual ONNX pruning is
unverified. State/parameter bytes exclude frontend buffers, runtime allocator,
queue, decoder, Python/Node/UI, and speaker-reference buffers. Operation counts
exclude normalization, nonlinearities, pooling and all non-model work.

## Verification scope

The final verification receipt records the exact focused test count, source
hashes, dependency versions and commands. Tests cover future perturbation,
irregular audio/feature chunking, independent full/online pooling, reset and
concurrent states, long bounded-state sequences, full/streamed slot and speaker
outputs, padded batches, gradients, traced warm states and real final fragments.

Policy tests cover role-specific enrollment, duplicate exemplars, indistinguishable
IDs, unknown rejection, hash invalidation, no acoustic-gate override, immutable
wake reference, speaker takeover, short/uncertain evidence, overlapping wake,
explicit official-spliced bypass, malformed flags, mandatory ownership,
unresolved explicit target, audio sequence gaps, no partial execution, and
101 type/range cases for complete volume values. Range enumeration is software
coverage, not 101 spoken-number tests.

A numerical mixture identity test demonstrates why the mixture alone does not
uniquely specify isolated sources. Fabricated embedding tests do not measure a
learned speaker verifier's error rate. OwnerGuard is not a source separator or
an overlap detector.

Source-only test helpers were developed red/green. Initial missing v2 modules
produced 161 failures; a second targeted pass exposed seven ownership/type/gap
policy failures, which were fixed before the final run. The earlier benchmark
export test was corrected to require the intentionally pass-through speaker
state, rather than equating it to the demo's updated speaker state.

## Existing evidence and critical interpretation

The supplied 6,768-recording report, not replayed here, gives human raw exactness
498/613 versus 3,556/3,590 supplied-synthetic positives, with 251/613 human correct
commands surviving hypothetical frozen checks. Existing qualification remains
failed. The current graph/checkpoint and recorded corpus are absent from the
provided files; neither can be inferred from reports.

The official benchmark prepends independently recorded wake words to heldout
commands. Strict same-speaker verification contradicts that construction. Keep
`official_spliced` explicitly labeled and measure ownership separately on true
same/different-speaker sequences. It must never be selected by recognizing a
benchmark filename or process.

## Research-derived design, not reproduced claims

- Streaming KWS: Rybakov et al., https://arxiv.org/abs/2005.06720 . Caching
  supports incremental execution; accuracy/latency tradeoffs need measurement.
- Acoustic enrollment: Settle et al., https://arxiv.org/abs/1706.03818 and
  Yuan et al., https://arxiv.org/abs/1806.03621 . Acoustic word matching and
  context-aware embeddings motivate enrollment; no pretrained model is reused.
- Personal VAD: https://google.github.io/speaker-id/publications/PersonalVAD/ .
  Distinguishes target/non-target activity; published parameter counts do not
  include proof that our tiny speaker tower works.
- VoiceFilter-Lite: https://arxiv.org/abs/2009.04323 . Target separation is a
  distinct learned task, not something a same-speaker acceptance flag provides.
- Short-speaker duration: https://arxiv.org/abs/1810.10884 . Short references
  need duration-conditioned evaluation; its teacher is NOT adopted.
- GE2E: https://arxiv.org/abs/1710.10467 ; SpecAugment:
  https://arxiv.org/abs/1904.08779 . Borrow objectives/augmentation cautiously,
  not speech models, additional corpora, or reported accuracy.
- ONNX Runtime threading:
  https://onnxruntime.ai/docs/performance/tune-performance/threading.html .
  Measure one/two threads and spinning; do not guess target performance.
- Pi 4 official specifications:
  https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/ .
  Weight/state arithmetic is far below 4 GB; whole-system timing/RSS is pending.

## Remaining work

No acoustic training or new recorded-audio accuracy was obtained. The sandbox
cannot resolve HF/PyPI/GitHub download hosts; web access to the two supplied HF
cards also failed. ONNX and ONNX Runtime are not installed. No physical Pi,
microphone, SpeakerUI integration run, real enrolled waveform, wake qualification,
trained speaker verification, overlap detection, TTS comparison or provider
operation was performed. The source prototype can be executed and audited, but
must not be packaged as trained weights or an automatically accepting demo.

The implementation/qualification handoff is
[HANDOFF-pcv-streaming-v2.md](HANDOFF-pcv-streaming-v2.md), including its exact
Codex kickoff sentence, frozen architecture, training recipe, two artifact
contracts, owner-benchmark distinction and proposed latency/quality gates.
