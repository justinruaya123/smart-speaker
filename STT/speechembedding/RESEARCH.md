# Choosing a reader for text-enrolled music names

20 September 2026. Sources were inspected live for this study. The [measured local report](REPORT.md) separates these architectural arguments from results of the two fitted candidates.

The target is lexical identity: an utterance containing an artist or song name should retrieve that catalog entry. A semantic audio embedding or a speaker-identity embedding solves a different problem. Phonetically similar names need discrimination even when they share a topic or speaker. Full free-form transcription is not required, but a model still has to learn the sounds and their ordering.

The existing `keyword-v2` implementation has a convolutional/BiLSTM encoder and a fixed command-keyword CTC vocabulary. Its tokenizer maps unfamiliar name words to `<other>`. It therefore cannot resolve arbitrary artist identities by itself. This experiment tests keyword supervision through a fresh auxiliary classification head on the same human corpus; it does not reuse synthetic/mixed checkpoints or claim a full-pipeline comparison with `keyword-v2`.

| Approach | How a new name enters | Good fit | Limitation for this request |
| --- | --- | --- | --- |
| Fixed KWS, including a BC-ResNet classifier | New trained output and audio examples | Stable command words | Catalog text alone cannot add an acoustic class |
| Acoustic enrollment embedding | Original recordings become name prototypes | Personal aliases or unusual pronunciations | Requires recordings; whole-command context can distort similarity |
| Joint audio/pronunciation embedding | Encode reviewed pronunciation text | Fast catalog retrieval with text-only enrollment | Needs sound/text alignment learned across varied words and speakers |
| Phoneme posterior plus constrained name graph | Add pronunciation paths | Connected names and explicit acoustic sequence verification | Search cost and recognition errors remain; confidence needs recalibration |
| Retrieval followed by a small sequence verifier | Add name vectors and pronunciation patterns | Larger catalogs with confusing names | More components; must measure end-to-end cost and rejection |

Joint audio/text embeddings have a direct research basis. [He, Wang and Livescu](https://arxiv.org/abs/1611.04496) learn representations of acoustic and character sequences with a multi-view contrastive objective. That supports comparing modalities in a learned common space; it does not justify applying cosine similarity to two independently trained unrelated encoders. Our smaller convolutional pronunciation-based model is an inspired implementation, not a reproduction of their architecture or scores.

[Flexible Keyword Spotting based on Homogeneous Audio-Text Embedding](https://machinelearning.apple.com/research/flexible-keyword) addresses mismatch between audio and text representations using phoneme-based text enrollment and audio-derived phoneme representations. This is more directly relevant to typing a new artist name than a fixed-class keyword classifier. Its reported benchmark and training resources differ from ours; none of its weights or published performance are imported into this experiment.

[MM-KWS](https://www.isca-archive.org/interspeech_2024/ai24_interspeech.html) investigates text and spoken templates together, supporting the idea of optional audio enrollment. The [authors' repository](https://github.com/aizhiqi-work/MM-KWS) also describes synthesis-based data preparation. We did not copy that recipe: this version uses only original human recordings. Optional recording enrollment is evaluated separately from the text-only target.

The supplied [BC-ResNet paper](https://arxiv.org/pdf/2106.04140) is investigated in [BCRESNET.md](BCRESNET.md). Its role would be to improve the acoustic feature extractor; dynamic text enrollment is a separate architectural requirement.

SLURP-real is the most relevant source already present because it contains natural assistant requests with artist/song annotations. The [publisher repository](https://github.com/pswietojanski/slurp) documents original and synthetic recordings separately, paired microphones and speaker metadata. Its audio is CC BY-NC 4.0; the text repository is CC BY 4.0. The frozen study uses only original audio with one microphone recording per utterance and preserves source/license fields. Corpus entity annotations are expected labels and may not perfectly describe what a speaker actually said; the existing uniform consistency quarantine is retained.

The older corpus has only 166 artist and 99 song training targets. More isolated digit recordings do not fill that gap. [LibriSpeech](https://www.openslr.org/12/) offers original read speech and can increase phonetic coverage; the existing phonetic-v2 study already used a bounded expansion. General read speech is still not a replacement for connected, accented artist and playlist requests. The new embeddings intentionally reuse the exact smaller parameter-study corpus so that their comparison with its old embedding model uses the same training data.

No new dataset download was needed for this feasibility comparison. The highest-value next collection is original human music requests with new speakers, reviewed proper-name pronunciations, similar-sounding alternatives and names absent from the active catalog. Playlist names themselves need explicit coverage: artist and song results cannot be relabeled as playlist-name accuracy.

My architectural preference is pronunciation-aware candidate retrieval followed by local sequence verification, conditioned on an accepted VCM intent. Keep only candidates belonging to the actual declared playlist/library catalog and calibrate rejection for that catalog. This is an inference from the problem structure and inspected research, not evidence that an untrained design outperforms the measured phonetic reader.
