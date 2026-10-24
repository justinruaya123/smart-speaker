# Iris final publication — rollback and safety

The final bundles are research software. A source publication or a successful package replay does not waive the failed wake acoustic gate. Keep the existing model hashes, locked thresholds and `qualified=false` status.

## Installed files

Extract releases into a new directory instead of overwriting an installation. Preserve private configuration, Home Assistant tokens, scheduler state and receipts outside release archives. To return to a previous bundle, stop only that installation's owned process, select its previous verified directory, and recheck its model/manifest hashes. Do not restore private state from an unrelated installation or automatically retry actions with unknown outcomes.

## Git source

The inspected pre-integration main baseline is `c0628940400642f204ff98e26908549b1c07e41c`. Existing inference/UI, presentation and wake-evaluation commits are integrated without rewriting their history. Frozen model bytes are additive publication artifacts, not a newly fitted model.

Use a reviewed `git revert` commit on main, never reset/force-push a shared branch. Revert dependent package/documentation changes before their runtime dependency. For a merge commit, inspect both parents with `git show --no-patch --pretty=raw COMMIT` and use `git revert -m 1 COMMIT` only when parent 1 is the intended retained main lineage. Recheck the resulting diff before committing or pushing. Do not blindly revert a whole numeric range containing merges.

The publication scopes are:

- Inference adapter/launcher, worker feedback and UI: the named source changes and tests recorded in `docs/RELEASE-iris-publication-v1.md`.
- Presentation: `presentation/`, `docs/site/`, the two-minute handout and generated README. Regenerate README through `presentation/build_readme.py` rather than changing a second independent copy.
- Experimental OpenWakeWord: `scripts/runtime/iris_openwakeword.py` and its explicit study/tests/licenses. It is uninstalled and unpromoted; removing that research component must not substitute a wake backend or change native model bytes.
- Final archives: the checked wrapper around the shared `scripts/archive_io.py` writer. Existing releases, source tags and historical evidence remain immutable.

## Verify after any rollback

Run the frozen payload verifier, affected unit tests, static publication validator, UI checks/build and extracted-package checks for the selected revision. Re-read the exact remote main and release assets after publication. Keep real capture, physical Home Assistant actions, fresh human acoustic quality and Pi performance as separately authorized operator tests. A software rollback does not repair those unmeasured or failed gates.
