# VCM-only removal and Option B audit

This commit removes the independent character-transcription model, not the
constrained phoneme research model. The v12 family/finite heads and frontend
remain unchanged. The workflow applies the reviewed migration and verifies a
random-weight ONNX fixture and the actual publisher's one-graph allowlist before
committing the root notebook/config changes. Original trained v12 artifacts and
full human recordings are not available in this clone; no new v12 accuracy is
claimed. Source tests do not establish Pi performance or acceptance quality.

## Pinned external dataset

Reviewed `markandrian30/AI231`, commit
`ed245708c2573188eb769096aba76588f79b8de1`, `MEX2/OptionB`.

The README describes DGX2 counts of 150 reference speakers and 27,956 active
files. The committed manifest and complete Git tree instead agree on 18,375
WAVs, 100 speaker IDs and 96 distinct phrases. `dataset_design.txt` agrees with
the latter. Train/validation/test contain 14,677/1,856/1,842 rows and 80/10/10
speaker IDs. All 96 phrases recur in each split; it is not a phrase-disjoint test.

No duplicate paths, missing/extra manifest WAVs, cross-split speaker IDs,
clean/noisy parent groups, or identical Git blobs were found. These checks do
not establish that distinct source IDs correspond to independent human-command
recordings or exclude near-duplicate/cloned audio.

A local audit decoded and verified 1,845 pinned WAV files: every available
brightness file (1,787), plus 58 small coverage controls. All are mono, with
603 at 16 kHz and 1,242 at 12 kHz. No nonfinite PCM, duplicate decoded PCM,
clipped samples, or duration discrepancies beyond 1.1 ms were found in this
subset. Resample whole records to 16 kHz; do not reinterpret 12 kHz as 16 kHz.
This was not manual transcription adjudication.

The Option B subtree has no Python training program, training notebook, optimizer,
epoch schedule, model selection record or accuracy report. Its metadata supports
reviewing dataset design, not reconstructing an unpublished training recipe.
The README describes offline transcription-similarity filtering at 0.80 and
mentions transcription tools; this does not imply a runtime transcription model.

Brightness values are only 20, 60 and 100. There are no numeric volume settings,
custom light names or song-title/BY/artist tuples in its finite templates.
It is useful for external-domain and fixed-intent controls, not proof of
compositional values or dynamic entity recognition. Command recording/generation
provenance and reuse permissions should be clarified before redistribution.

## Bounded improvement test

The existing acoustic weights remain frozen. Three complete brightness templates
from Option B's training rows were compiled for **all** values 0 through 100,
not just the three reference values. This creates a separately hashed catalog;
no old acceptance thresholds are inherited. Matched development and held-out
results, ORT parity and runtime timing are recorded separately when completed.
The production path remains VCM-only and uncertain values still clarify.
