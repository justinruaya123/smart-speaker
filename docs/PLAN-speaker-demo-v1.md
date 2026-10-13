# Speaker demo v1: implementation and validation plan

Goal: enable reproducible unfamiliar-speaker command experiments and conservative
wake-owner checks without installing a new model or relaxing the application gates.
Baseline: codex/semifinal-handoff, 7d8296c96e19e1e65fc3f0cd1b82d2391df33dbc.
Authorization: user requested implementation, commits where possible, and an
execution-ready large-scale Codex test handoff after the preceding design review.

## Scope and implementation sequence

1. Add a reusable, one-command speaker-session policy. Require valid embeddings,
   matching model identity, enough speech, explicit overlap evidence, owner/session
   binding, and expiry. Never let a positive speaker result rescue a VCM rejection.
   Test all failure paths with numerical fixtures before implementation.
2. Add a small, randomly initialized direct demo classifier and cross-speaker
   contrastive loss. Balance class/speaker cells. Keep unknown samples out of
   positive contrastive pairs. Test variable lengths, padding, gradients and balance.
3. Add a standalone audit / plan / replay / score CLI. Reuse the existing runtime
   and application_decision; never call an action provider. Hash and validate files,
   reject cross-split speaker/PCM overlap, preserve missing-trial denominators,
   report human/synthetic strata and per-speaker metrics separately.
4. Freeze the experiment protocol, write the execution handoff, run focused tests
   and a large numerical stress test, and commit only reviewed experiment sources.

## Boundaries

The local execution directory is an isolated source workspace, not a complete
checkout. GitHub reads pin the baseline. Direct network downloads are unavailable.
The project's .venv, model payloads, recordings, ncnn and ONNX Runtime are not
available here. Only available host Python/NumPy/PyTorch/pytest are used for tests.
No real-audio recognition, human-speaker verification, Pi measurements, new
pretraining, model installation, provider action or accuracy improvement is claimed.
No change to installed weights, 0.95 intent / 0.90 slot gates, existing frozen
reports, catalogs, frontends, or shipping defaults. Experimental code is not added
to the publication allowlist until export/parity and acoustic gates pass.

Existing committed .gitignore already covers .cache, audio and weights; no new
ignore commit is required. All mutable outputs go under .cache/speaker-demo-v1.
No full-repository test claim is permitted from this source-only workspace.
