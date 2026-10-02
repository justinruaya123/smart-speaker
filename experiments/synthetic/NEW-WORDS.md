# New wording without adding intents or parameter classes

The appropriate next extension is query-by-example keyword enrollment. The semantic output inventory remains exactly the current fixed families and finite values. A user supplies a few recordings of a new phrase and binds them to an existing semantic ID, such as BRIGHTNESS/INCREASE/10. This adds acoustic exemplars, not another command or an artist-name recognizer.

An implementation would normalize a compact acoustic encoder's embeddings, average several enrollment examples into a prototype, and compare new recordings against those prototypes. A prototype needs both a minimum similarity and a margin over competing prototypes. Unknown/background speech must still be rejected. Multiple examples and speakers are preferable to a single recording; exact requirements must be measured.

[Rusci et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/rusci23_interspeech.html) study on-device open-set keyword customization using acoustic encoders and prototype classifiers. Their work supports this as an engineering direction, not a guarantee that this VCM's current embeddings will generalize. The present model was optimized for fixed classification and keyword supervision, not a triplet/metric objective. A follow-up should compare a metric-trained encoder against the current representation before enabling enrollment.

[Kim et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kim24r_interspeech.html) address keyword-incremental learning and calibration. This reinforces that expanding acoustic exemplars changes competition and requires recalibration; a JSON alias alone does not add an acoustic detector.

For text-only enrollment, the alternative is a phoneme-posterior keyword graph with a reviewed pronunciation. The phoneme alphabet stays fixed, while a new word is represented by a new phoneme sequence. That needs a validated phoneme encoder and unknown-word evaluation. Dictionary coverage alone is insufficient, and the previous research phonetic candidate is not silently promoted by this experiment.

The source-preference ensemble in this experiment does not implement either enrollment path. No novel-word corpus is evaluated in this phase, as requested. The design separates the feasibility of future enrollment from the current fixed-vocabulary results. Artist names, songs and named devices remain work for the later parameter/STT phase.
