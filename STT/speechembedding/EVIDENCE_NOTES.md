# Interpreting the completed speech-embedding experiment

These observations use the frozen outputs in `.cache/speechembedding-v1/results.json`. They do not change models, thresholds or the protocol after test scoring.

The development-selected reader is `dual/text_global/expanded`; it recognizes 4/77 artists and 5/45 songs on test. The highest observed new embedding result is `dual_kws/text_local/expanded`, at 12/77 artists and 8/45 songs. The latter is an observed ablation result, **not a replacement selected using test accuracy**. Both models used one seed. Ranking them on this small test cannot establish a general architecture winner.

The central text-enrollment limitation is more severe than the aggregate name score suggests. **Both of those readers recognize 0/44 artist queries whose artist label was absent from training.** All 12 correct artist results of the keyword-assisted local reader concern labels seen during training. One other text variant (`dual_kws/text_global`) retrieves one of the 16 deliberately reserved artist queries, but its overall artist accuracy is lower, and it also fails the development quality gate. Text addition is mechanically implemented; reliable unseen-artist recognition is not demonstrated.

Pronunciation coverage explains part of the failure: only 60/77 artist references have dictionary-derived catalog representations. However, the keyword-assisted local reader still misses 48 of those 60 representable queries. Supplying the missing pronunciations alone cannot explain or fix those observed matching errors. Proper-name training diversity, the learned alignment and surrounding command context need new controlled experiments; this run does not isolate their causal contributions.

Optional audio enrollment retrieves 5/16 deliberately reserved artist queries with up to five speakers per name. Only eight of the 127 artist catalog entries actually have five available enrollment speakers. Missing and limited enrollment are reported explicitly. No audio or hybrid music configuration meets the development precision rule.

The existing phonetic-v2 wide phrase graph recognizes 36/77 artists and 24/45 songs on identical recording/reference pairs. It had a larger real-English training corpus, so that result supports it as the stronger currently measured reader, not a claim that its architecture alone causes the entire difference. Its music-name confidence gates also remain failed.

The incidental number result must not hide another failed transfer: the development-selected text reader emits 768 correct values among 825 number emissions on test, **93.1% precision**, below the 95% development target. It also falsely accepts 2/300 background number trials. This is separate from its 0/10 raw connected-number success and does not approve a numeric runtime reader.

CPU measurements are a new matched timing run: approximately 13.1 ms median for the keyword-assisted local embedding reader and 18.1 ms for the wide phonetic phrase graph on the same 48 original music recordings. These are desktop frontend-plus-reader times with preloaded PCM, excluding the default 400 ms endpoint wait and speech duration. Historical report latency used different utterance mixtures; compare the new matched table rather than mixing old and new timing numbers.

All candidate fitting/export processes completed with exit code zero. Variable-length and batch ONNX checks, 19,313 original file hashes, source preservation, runtime regression and the file-query adapter checks passed. These are software, artifact and recorded-audio checks. No live microphone, Raspberry Pi, Spotify account, custom-playlist corpus or device action was tested.
