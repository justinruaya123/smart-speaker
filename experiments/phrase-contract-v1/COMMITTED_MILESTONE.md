# PC-v1 decoder milestone (not a trained speech release)

This milestone contains the exact shared-prefix decoder, strict command helper,
independent decoder tests, and preliminary measurements. It does not replace the
selected disabled diagnostic model. No workflow, dataset upload, deployment,
model activation, service execution, or existing research modification is included.

The complete implementation delivery additionally contains the expanded PC-v1
specification, data preparation, training/selection, endpoint/runtime, calibration,
ONNX adapter, reporting, and detailed fixed execution handoff. Those accompanying
files must be added from the supplied source delivery before running that larger
pipeline. This milestone alone is not that entire pipeline.

Run this milestone's independent tests from the repository root:

    python -m pytest experiments/phrase-contract-v1/tests/test_decoder_parity.py -q

Preliminary results on the offline x86 host, using generated tensors only:

- Full 770-entry catalog, 250 frames, five paired decoder runs: padded median
  289.508021 ms versus exact-prefix median 81.717550 ms (3.542789x).
- Maximum score discrepancy 4.884981308350689e-15; no beam or phrase pruning.
- Decoder probability-state storage 572,880 to 115,888 bytes; excludes index
  tables, frontend, model, ONNX Runtime and process RSS.
- Noiseless generated-phone-logit suite: 770/770 exact. Severe artificial logit
  noise (standard deviation 6, one fixed seed per case): 541/770 exact; all 229
  failures retained in the complete delivery. This is NOT an acoustic SNR test.
- A 12-update artificial-feature CTC smoke test exercised the 113,064-parameter
  baseline. Loss 52.18104 to 12.33811 is plumbing evidence, not speech learning.
- Original CPU oneDNN chunk difference 2.28882e-5 failed the fixed 2e-5 limit.
  Native CPU convolution produced 1.90735e-5 and passed without changing the limit.
- No human-waveform recognition result, final ONNX, or physical Pi measurement.

Release gates remain 99% accepted precision, 90% full-command accepted recall,
1% negative false accepts, every required phrase/use case independently covered,
new sealed human confirmation, and physical hardware qualification. Never promote
an all-reject policy or the generated-feature smoke weights.
