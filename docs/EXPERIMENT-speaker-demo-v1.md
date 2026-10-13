# Speaker demo v1: verified software, pending acoustic evidence

Baseline: `codex/semifinal-handoff` at
`7d8296c96e19e1e65fc3f0cd1b82d2391df33dbc`.

## Delivered changes

Reusable single-command owner policy; a small direct keyword model;
class/speaker-balanced training and a cross-speaker contrastive objective;
a nine-run matched training driver; immutable audit/plan/lock/replay/score tools;
focused tests, numerical stress runner and execution handoff.

The existing live wake gate, application policy, installed artifacts, frontend,
full-command catalog and default engine are not modified. The speaker policy is
an experimental helper, not an enabled or acoustically validated security feature.

## Executed evidence

41 focused tests passed. These cover model padding/gradients, sampling weights,
contrastive edge cases, owner replacement, expiry, model mismatch, overlap and
short speech, session consumption, manifest/PCM leakage, trial expectation and
coverage integrity, malformed/missing predictions, and grid enumeration.

A separate numerical stress run passed 10,000 policy cases and validated a
12,000-trial synthetic-metadata plan with all missing-result denominators intact.
An 80-step toy-feature optimization smoke decreased cross-entropy from about
1.41496 to 0.00000807. This tests optimization plumbing, not voice accuracy.
The 11-class, width-64 keyword candidate has 22,027 parameters.

The environment was Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, not the user's
root environment. Tests used new experiment sources in an isolated workspace,
not a complete repository checkout. No full-repository regression claim is made.
One first combined stress invocation hit the 45-second execution limit; the
separate one-thread runs completed. No isolated timing conclusion is inferred.

## Not performed / not established

No human corpus inference, acoustic candidate fitting, speaker backend training,
voice-threshold calibration, ncnn/ONNX candidate export, full microphone integration,
physical Pi measurement or provider action was performed. No increase in deployed
accepted-command accuracy is claimed. Numerical matching embeddings do not prove
same-person recognition. Paired-crop replay does not exercise wake segmentation.
The epoch trainer and existing-runtime replay path still require full-checkout
integration runs using the actual locked environment and payloads.

A context-only patch is supplied for the root experiment index. It and the source
map registration remain Codex integration steps; the remote index was not replaced
with a partial reconstruction. No audio/weights/cache outputs belong in the commit.

## Next execution

Follow [the handoff](HANDOFF-speaker-demo-v1.md) and the frozen
[protocol](../experiments/speaker-demo-v1/protocol.json). Human/synthetic results,
raw versus accepted commands, and command crops versus full live sessions must
remain separate. Qualification targets are proposals, not measured achievements.
