# Source verification and optional historical evidence

Run `python final_suite/run_all.py --qualify` from the repository root. It executes the explicit tracked Python test directories, skips unavailable optional historical inputs, and evaluates fail-closed qualification. Missing historical ZIPs are not bootstrap errors. The current diagnostic model also has its own bound verdict in `reports/semifinal-v1/release.json`; source tests alone cannot qualify it.

`review.py`, `dataset_audit.py`, `verify_math.py`, browser checks and the safe historical archive-restoration helper remain as optional historical evidence tools. They are not runtime or publication dependencies. Original reports/notebook are preserved in `docs/finalization/baseline-cd98092.zip`. The old notebook refresh writer was retired; the current `Architecture.ipynb` imports canonical modules and reads measured reports.

Seven tests requiring absent historical payload are reported as skipped. New streaming, typed catalog, replay reporting, package integrity, no-network/no-training-import and source-extraction tests live in root `tests/`.
