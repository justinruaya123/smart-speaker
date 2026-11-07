# BMO Pages implementation plan

Goal: publish the reviewed four-slide BMO documentation deck on GitHub Pages, with source-backed dataset generation, citations and VCM benchmarks.

Architecture: a static HTML entry point with local JavaScript modules, Three.js, recorded tensors and frozen evidence. No backend, training or live neural inference runs in the browser. Keep the previous six-chapter presentation as `reference.html`.

Constraints: preserve current research worktrees, installed artifacts, scientific recipes and acoustic acceptance receipts. Use the existing root interpreter. Keep physical Pi results as unmeasured placeholders and distinguish the published epoch-38 model from the epoch-36 RTX 4080 refit. No AI-generated images.

- [x] Isolate the implementation at main `5bb30ebb`; commit only the exact sample ignore exception first.
- [x] Integrate `docs/site/index.html` and `docs/site/bmo/` with paths that work at `/smart-speaker/` and nested prefixes.
- [x] Add source citations, generation model/provider metadata, split rules, exact command inventory and VCM benchmark details; mark unrecorded metadata explicitly.
- [x] Add static validation and regression checks for hashes, relative dependencies, data contracts and publication boundaries. Preserve the previous deck's existing validation.
- [x] Separate static documentation deployment from the unchanged acoustic qualification check in the existing Pages workflow; document the policy change.
- [x] Verify all four slides and controls through the in-app browser at a GitHub-style project prefix, run source validators, review the scoped diff, prepare the approved documentation changes for GitHub publication. Deployment success is verified from the actual workflow and public URL after merge.

Review focus: project-subpath fetches; WebGL failure and motion controls; source-specific dataset rights; cross-GPU result attribution; no implication that documentation publication qualifies the acoustic model or Pi hardware.

## Concise deck and log-mel update — 2026-11-07

Keep the existing four-slide layout and pinned epoch-38 model. Its matched DEV win (132/167 human commands versus 130/167 for the 175,139-parameter flat control; both 400/408 demo-supported) is the selection reason. The trained graph already exists on main; expose a direct download and preserve its SHA-256. Do not replace it with the unevaluated epoch-36 refit.

- [x] Shorten visible explanations into complete sentences; retain the full recipe, benchmark and citation panels.
- [x] Render the recorded log-mel thumbnail as an accessible SVG on Architecture. Identify its 16 × 48 display means over the original 40 × 296 feature grid and link playback position to the recorded audio.
- [x] Add the short DEV selection reason and model link to the generated README and deck.
- [ ] Validate the recorded-data rendering, responsive layout, immutable model bytes and project-prefix interactions; push, merge and verify Pages.
