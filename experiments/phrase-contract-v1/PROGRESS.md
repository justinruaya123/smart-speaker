# PC-v1 execution ledger

Binding input: original PC-v1 package and the user's request to perform experiments and commit implementation. Target branch: `codex/semifinal-handoff`, original commit `a788bb38e3345a3a7d9dbd0316b9c9719ee20639`.

## Rulings

- The latest instruction authorizes implementation and preliminary experiments using PC-v1 as the working catalog. It does not manufacture a consent/license receipt, approve production release, or turn absent human recordings into evidence. The historical unsigned approval file remains unchanged.
- All changes are isolated under `experiments/phrase-contract-v1/`. Original research, selected disabled bundle, root documentation, and source are not overwritten.
- Direct network downloads, pip installation, and repository cloning were unavailable. A connected GitHub source artifact was obtained; it contains neither original waveform datasets nor trained weights. Three current baseline modules were separately read and verified against their Git blob identities. This is not a complete checkout.
- The first attempted remote commit included a data-collection workflow. GitHub's branch update was blocked; that workflow did not execute. It is excluded from this delivery. Do not retry that workflow or silently change connector permissions.
- Reuse the existing causal CNN/frontend, but replace padded independent-sequence CTC with exactly equivalent shared-prefix CTC inside the opt-in experiment. The measured decoder-only improvement is supported by real timing and numerical comparison, not speech recognition accuracy.
- Initial development training is first-party consented human speech, not an inferred FSC product-use authorization. Hierarchical use-case/phrase/speaker balancing uses 80% positives and 20% real negatives. One optional ablation doubles fixed numeric/contact hard-case sampling. Neither changes the architecture or final gates.
- Model ranking uses every development positive, not 64 examples. Select by phrase success count, worst use case, macro exactness, total exactness, then deterministic seed/epoch tie breakers. Retain all epoch checkpoints and failed candidates.
- Global score/margin calibration is the first implemented policy (at most 651 combinations). Each use case and each negative class must pass independently. Calibration needs at least 50 samples per UC per negative class; final testing still requires at least 100. The former mixed negative denominator was corrected. No all-reject promotion exists.
- The observed native-vs-oneDNN FP32 difference is preserved. Native CPU convolution passes the unchanged 2e-5 chunk-parity threshold in the artificial-feature smoke test; the first oneDNN failure remains evidence. This is not ONNX parity.
- No trained deployment graph is created from random or artificial-feature smoke-test weights. Export requires a real development-selected checkpoint. No human accuracy or Pi qualification can be declared here.

## Completed work

## 2026-09-30 current-checkout continuation

- Baseline: `f13ffcaa4974eee96e8c6576af136c95b6e1d6a5`, branch `codex/semifinal-handoff`; no staged changes. Unrelated untracked research remains untouched. Existing committed ignore rules cover all planned cache/checkpoint/audio outputs; no ignore commit required.
- Ruling: execute the existing five-step handoff in the current checkout, using this existing ledger; no worktree, architecture redesign, maintenance migration or new documentation hierarchy. New code/tests/reports stay under this experiment.
- Baseline defect: `init-spec` failed on Windows CRLF output. All six regenerated files match the original bound hashes after LF conversion. Normalize only temporary generated text, then verify original hashes; existing frozen files still fail on any mutation. Regression added before fix. Baseline: 51 tests pass; contract unchanged (570/402/200).
- Interfaces: CV consumes verified raw features and recomputes fit-fold normalization; endpoint selection consumes reviewed development timestamps and feeds the shared runtime; capture consumes the same runtime; packaging must bind the selected endpoint, policy, data, replay and physical measurements.
- Remaining: (1) CV and shared compute budget, (2) development endpoint selection, (3) bounded capture, (4) exact-graph/target dependencies, (5) evidence-derived release; then original human recording and physical qualification only if real inputs are available.

Strict full-command membership; all 770 pronunciations; context-specific AM and unmute; exact shared-prefix decoder; sticky capture/endpoint faults; shared calibration decision predicate; recording-slot plan; provenance/duplicate checks; frozen feature preparation; own-model training and immutable epoch retention; full-development selection; ONNX export adapter; a single real-endpoint runtime; immutable calibration/replay/report orchestration; diagnostic-only allowlist publisher; software regression tests and retained preliminary experiments.

## Explicit incomplete validation/integration

Real human corpus acquisition and labels; pronunciation listening; acoustic training/development/calibration/confirmation; ONNX/ORT dependency installation and real parity; near-duplicate/manual rights review; cross-validation execution; endpoint-threshold development selection; live microphone capture adapter; Pi measurements; target-architecture hash-locked installation; and a qualified-release publisher that verifies all physical receipts. None has a passing flag.
