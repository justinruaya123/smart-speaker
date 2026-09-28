# Final suite

Run from the repository root after restoring the supplied research payload.
`run_all.py --browser` executes software, evidence, WASM/ONNX parity and browser
component checks. `--qualify` additionally enforces release gates and currently
exits 2. Source-only GitHub CI is not acoustic or hardware qualification.

`review.py` rebuilds complete ledgers; `dataset_audit.py` audits recorded metadata;
`coverage_probe.py` regenerates the 44 synthetic full-scope diagnostics using
espeak and the retained ONNX. None trains or calibrates a model.
`refresh_notebook.py` refreshes data and executes the source-controlled notebook
into the ignored reports folder. Existing inspected sets stay regression-only.
