# VCM recognition failure research handoff

The selected candidate fails the requirement to recognize previously unseen human speakers. Packaging completion did not establish microphone usability. Preserve this failed candidate and its benchmark evidence for independent analysis. The user will choose a research agent to investigate; this handoff does not start another training run.

The source baseline is commit `6aee6bbe8075141c13ab2ddbb06ef3f985615c90` in `codex/vcm-a100-finalization-20261015`; the published project branch is `codex/vcm-intent-slots-v1`. The selected checkpoint is uniform sampling, seed 89, epoch 28. This is the human-development winner of six runs of the same architecture, not an established best model across repository history.

## Acceptance requirement and scope

Recognition must generalize to unseen people without personal voice training or speaker adaptation. Do not add the user's voice to fitting, checkpoint selection, calibration, or enrollment to make this candidate pass. Keep the mandatory benchmark data and references fixed. Microphone acceptance recordings must remain evaluation-only after the candidate and its policy are frozen. Preserve the failed model and all historical outcomes when proposing any new version.

The original architecture, 0.95 intent / 0.90 slot confidence thresholds, grammar, catalog, source files, checkpoint, calibration, and packaged reports retain their frozen identities. The only additions are this brief, the numeric diagnostic receipt, and the experiment-directory entry. There is no new model inference or fitting in this review.

## Verified slot acceptance failure

The exact selected policy has `require_calibration=true` and six fitted slot calibrators. Every calibrator has a conservative mathematical probability upper bound below 0.90. A command requiring one of these calibrated slots therefore cannot pass the selected policy, regardless of its input waveform.

For constituent probabilities $x_j\in[0,1]$, the implemented score is

$$c(x)=\sigma(b+w^\top x),\qquad \sigma(z)=\frac{1}{1+e^{-z}}.$$

Monotonicity gives

$$c(x)\leq\sigma\left(b+\sum_j\max(0,w_j)\right).$$

The bound was checked against every corner of each calibrator's input cube. Correlation between constituent probabilities can only make the actual reachable maximum smaller.

| Calibrator | Probability upper bound | Correct calibration proposals / rows |
| --- | ---: | ---: |
| operation/acoustic_finite_role/3 | 0.329050 | 32/147 |
| light_id/qualified_generic_request/2 | 0.675899 | 156/242 |
| light_id/acoustic_enrollment/3 | 0.294398 | 21/103 |
| time/recognized_clock_with_whole_hour_rule/8 | 0.666439 | 2/4 |
| color/acoustic_finite_role/3 | 0.713579 | 45/67 |
| percent/acoustic_finite_role/3 | 0.378909 | 30/124 |

Missing calibration groups also reject because calibration is required. The recorded observed missing cells include contact enrollment and the timezone constant. This establishes infeasibility of the frozen acceptance policy for the affected slots. It does not establish that logistic regression is implemented incorrectly or that boosting scores/lowering thresholds would produce a reliable recognizer.

Sources: [slot score and decoder](../scripts/runtime/vcm_intent_slots_decode.py), exact coefficient vectors and file hashes in the [diagnostic receipt](../experiments/vcm-mixed-final-v2/post-handoff-diagnostics-20261015.json).

## Verified supervision mismatch

The actual fitting manifest contains 1,895 human recordings and 1,896 synthetic recordings. None of the human rows has an explicit role-state label. Explicit role-state supervision is supplied by synthetic data from two training voice IDs. The decoder requires confident absent/present decisions for all 15 roles.

Human recordings still supply intent and some finite-value targets through their command references; they are not wholly unlabeled. However, the mixed trainer masks entity targets whenever the corresponding role is not explicitly `present`, so these human rows also supply zero observed entity targets after that mask. The selected run records zero reviewed acoustic spans and zero supervised alarm-minute values.

| Split | Kind | Rows | Declared speaker or voice IDs | Rows with role labels | Entity targets after training mask |
| --- | --- | ---: | ---: | ---: | ---: |
| train | human | 1895 | 182 | 0 | 0 |
| train | synthetic | 1896 | 2 | 1374 | 295 |
| dev | human | 256 | 36 | 0 | 0 |
| dev | synthetic | 1037 | 1 | 758 | 156 |
| calibration | human | 210 | 24 | 0 | 0 |
| calibration | synthetic | 1037 | 1 | 758 | 156 |

The JSON receipt records every view hash and per-head finite-target counts. Speaker IDs are declared source identities; physical cross-source aliases remain unresolved. Sources: [target construction](../scripts/train/vcm_intent_slots_train.py), [mixed fitting and entity masking](../scripts/train/vcm_intent_slots_mixed_train.py).

## Acoustic errors and rejection stages

The saved fresh-human predictions identify the correct intent on 43/93 supported recordings. Eight supported recordings have both the correct intent and intent confidence at least 0.95; none produces an accepted exact command. First rejection reasons on this cohort are 72 low-intent-confidence decisions, 13 OOV decisions, and eight invalid-argument/slot-confidence decisions.

On the original human test cohort, intent argmax agrees with the reference on 352/636 supported recordings, while only 25/636 final commands are correct and accepted. The underlying acoustic predictions are therefore weak as well as the frozen slot policy being infeasible. The relative causal contribution of training labels, acoustic generalization, and score fitting has not been isolated by controlled ablations.

Diagnostic condition counts in the receipt are independently evaluated conditions, not a sequential funnel. Saved diagnostic proposals bypassing confidence gates remain diagnostic outputs and never replace the frozen benchmark scores.

One original synthetic supported example, `Send my message`, was accepted as `WEATHER` instead of `MESSAGE` at intent confidence 0.9615731914557104. Its trial ID is retained in the receipt. Zero negative false accepts do not eliminate wrong accepts on supported commands.

## Frozen benchmark results

Exact supported-command accuracy requires the correct intent, exactly the expected typed slots, and an accepted final command. Negative and ambiguous references are not included in that supported-command denominator. A zero negative denominator means no negative observations, not a measured zero false-accept rate.

| Cohort | Kind | Exact accepted / supported | Exact supported accuracy | Negative false accepts / negatives |
| --- | --- | ---: | ---: | ---: |
| fresh_human106 | all | 0/93 | 0.000% | 0/13 |
| fresh_human106 | human | 0/93 | 0.000% | 0/13 |
| mandatory_defaults202 | all | 5/186 | 2.688% | 0/16 |
| mandatory_defaults202 | human | 2/86 | 2.326% | 0/10 |
| mandatory_defaults202 | synthetic | 3/100 | 3.000% | 0/6 |
| original_holdout196 | all | 8/183 | 4.372% | 0/11 |
| original_holdout196 | human | 5/83 | 6.024% | 0/11 |
| original_holdout196 | synthetic | 3/100 | 3.000% | 0/0 |
| original_test4418 | all | 44/4004 | 1.099% | 0/229 |
| original_test4418 | human | 25/636 | 3.931% | 0/229 |
| original_test4418 | synthetic | 19/3368 | 0.564% | 0/0 |

The default-202, original-test and original-holdout results are historical regression. The fresh-human cohort contains 106 total trials, of which 93 are supported and 13 negative. It does not cover all supported intent classes. On the original test, commands with supplied slots score 0/3,046; the corresponding default-202 result is 0/138. These are CUDA offline audio replays, not a measured resident microphone service or physical Pi benchmark.

The original-test accepted-command precision is 44/45 (97.78%) despite only 44/4,004 supported commands being correct and accepted. Fresh-human precision is undefined because no command is accepted. The original test preserves two overlong ambiguous capture faults and 185 ambiguous references separately.

## Training and selection record

Each of six runs completed 30 epochs at 96 optimizer updates per epoch, batch size 64, on one A100. This is 2,880 updates per run and 17,280 across the grid. The selected epoch-28 checkpoint contains weights after 2,688 updates; 2,880 describes its completed parent run. Fitting uses random initialization, AdamW at learning rate 0.001, weight decay 0.0001, and gradient norm clipping at 1.0. No CTC or whole-command contrastive term is enabled.

Selection maximizes human-development macro-person exact accepted recall, then minimizes accepted errors, then maximizes human micro recall, then prefers the earlier epoch. The fixed arm/seed order breaks remaining ties. Selection uses a declared uncalibrated development policy; final benchmark results use the separately fitted frozen policy. Synthetic development scores do not choose the winner.

| Sampling | Seed | Selected epoch | Human macro recall | Accepted errors | Human micro recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 53 | 30 | 14.816% | 12 | 13.990% |
| uniform | 71 | 28 | 18.710% | 12 | 18.653% |
| uniform | 89 | 28 | 28.655% | 5 | 28.497% |
| balanced | 53 | 26 | 2.913% | 0 | 2.591% |
| balanced | 71 | 24 | 5.729% | 3 | 4.145% |
| balanced | 89 | 24 | 7.083% | 1 | 4.145% |

The selected human-development result is 55/193 exact supported commands (28.50%) and 55/60 accepted precision (91.67%). These scores already fail the intended usability goal. Passing training execution, export parity, report generation and package checks did not change that result.

## Architecture and equations

The selected IntentSlotsVCM contains 280,818 parameters. Its path is mono 16 kHz PCM, frozen train-only normalization of 40 log-mel bins, a stride-two convolution, eight depthwise temporal residual blocks, 16 semantic queries, parallel output heads, acoustic entity memory, and the typed calibrated decoder. It uses complete utterances and symmetric temporal convolutions. There is no ASR, CTC, LLM, causal model or KV cache in the model.

For a valid clip of $N$ samples, the frontend produces $T=1+\lfloor(N-400)/160\rfloor$ frames. It uses a periodic Hann window of 400 samples (25 ms), 160-sample hop (10 ms), a 512-point FFT, and 40 triangular HTK mel filters spanning 20 to 7,600 Hz:

$$S_{t,k}=\sum_{n=0}^{399}x[160t+n]w_n e^{-2\pi i kn/512},\qquad w_n=0.5-0.5\cos(2\pi n/400),$$

$$X_{t,m}=\frac{\log\!\left(\max\left(\sum_k M_{k,m}|S_{t,k}|^2,10^{-8}\right)\right)-\mu_m}{\sigma_m}.$$

The frozen $\mu_m,\sigma_m$ are fitted on training frames. They are not per-utterance normalization. Input features have shape $B\times40\times T$; padded frames are masked. The stem is Conv1d(40,128,kernel=5,stride=2,padding=2), giving $L=\lceil T/2\rceil$ frames. There is no extra stem activation in the source.

For dilations $d=(1,2,4,8,1,2,4,8)$, each width-128 residual block computes

$$H' = m\odot\left[H+\operatorname{Dropout}_{0.1}\left(\operatorname{PWConv}\left(\operatorname{ReLU}\left(\operatorname{LayerNorm}_{channels}(\operatorname{DWConv}_{5,d}(m\odot H))\right)\right)\right)\right].$$

Layer normalization acts across 128 channels at each frame; the pointwise convolution is 128 to 128. Dropout is disabled in evaluation. The convolution stack's receptive field is $5+8\sum d=245$ original mel frames, corresponding to 2.465 seconds of waveform support. Global pooling subsequently uses every valid frame of the complete utterance.

For encoded frames $h_t\in\mathbb R^{128}$, a shared key projection and 16 learned 64-dimensional queries compute

$$k_t=W_kh_t+b_k,\qquad a_{t,r}=\frac{k_t^\top q_r}{8},\qquad \alpha_{t,r}=\operatorname{softmax}_{t}(a_{t,r}),\qquad u_r=\sum_t\alpha_{t,r}h_t.$$

The divisor is $\sqrt{64}=8$. Attention excludes padded frames. Query zero produces intent and validity logits; each of the 15 remaining queries supplies one semantic role.

| Output | Shape per clip | Computation |
| --- | --- | --- |
| Intent | 19 | $W_Iu_0+b_I$; 18 supported intent classes plus OOV |
| Request validity | 4 | $W_Vu_0+b_V$; valid, missing, conflicting, unsupported |
| Role states | 15 by 4 | One linear head per role; absent, present, conflicting, unsupported |
| Finite values | 415 total logits | Ten role-specific linear heads |
| Entity embeddings | 5 by 64 | $e_r=(W_ru_r+b_r)/\max(\lVert W_ru_r+b_r\rVert_2,10^{-8})$ |

Finite-head class counts are brightness percent 101, volume operation 4, volume percent 101, timer unit 3, alarm hour 12, alarm minute 60, alarm meridiem 2, temperature degrees 111, color 12, and timer amount 9. The timer amounts are the frozen finite set {1,5,10,30,60,61,300,600,3660}. The entity roles are light, song, artist, contact and task.

Entity matching uses normalized examples, not text similarity or an average prototype: $s_c=\max_{j\in E_c}e^\top e_j$. It checks the best score and best-minus-second-best margin against a bound policy, then applies entity calibration when available. The selected enrollment contains 11 names and 33 synthetic examples; independent acoustic listening and physical alias checks remain incomplete.

The joint training loss is

$$\mathcal L=\mathcal L_{intent}+\mathcal L_{role}+\mathcal L_{validity}+\frac{\sum_r n_r\mathcal L_{value,r}}{\sum_r n_r}+0.1\mathcal L_{entity}+0.1\mathcal L_{span}.$$

Cross-entropies average only observed targets; unknown targets use the ignored label -100. Entity positives have the same role/entity and different declared speaker IDs. Each entity-pair term is $1-e_a^\top e_b+\max(0,\max_c e_a^\top e_c-e_a^\top e_b+0.2)$ when negatives exist; the hinge term is absent otherwise. Reviewed span loss is $-\log\sum_{t\in[start,end)}\alpha_{t,r}$, averaged over reviewed spans. The selected fitting run has zero such spans.

Final intent probabilities are $\operatorname{softmax}(z/2.8029388409427383)$. The fixed decoder requires non-OOV intent confidence at least 0.95, valid request confidence at least 0.90, applicable confident role states, calibrated supplied-slot confidence at least 0.90, and valid typed arguments/catalog/bindings. It emits a command or rejection, with device actions disabled.

Sources: [model and loss](../scripts/models/vcm_intent_slots.py), [frontend](../vcm/frontend.py), [decoder](../scripts/runtime/vcm_intent_slots_decode.py), [entity matching](../scripts/runtime/vcm_intent_slots_entities.py).

## Parameter and computation accounting

The following multiply-accumulate counts are analytic counts for a single clip. They exclude FFT, bias additions, layer normalization, activations, softmax, embedding normalization, entity memory and typed decoding; they are not measured device latency.

| Component | Parameters | MACs |
| --- | ---: | ---: |
| Stem | 25,728 | $25,600L$ |
| Eight residual blocks | 140,288 | $136,192L$ |
| Key projection | 8,256 | $8,192L$ |
| Learned queries and query scoring | 1,024 | $1,024L$ |
| Semantic pooling | 0 | $2,048L$ |
| Intent head | 2,451 | 2,432 |
| Validity head | 516 | 512 |
| Role-state heads | 7,740 | 7,680 |
| Finite-value heads | 53,535 | 53,120 |
| Entity heads | 41,280 | 40,960 |
| Total | 280,818 | $173,056L+104,704$ |

| Audio seconds | Mel frames | Encoded frames | Neural MACs |
| --- | ---: | ---: | ---: |
| 1 | 98 | 49 | 8,584,448 |
| 2 | 198 | 99 | 17,237,248 |
| 3 | 298 | 149 | 25,890,048 |

The dense mel projection additionally uses $257\times40\times T=10,280T$ MACs, apart from the FFT and frontend scalar operations. FP32 parameter storage alone is 1,123,272 bytes (1.071 MiB); the actual ONNX graph is 1,148,441 bytes. These are not peak-memory measurements. The original-test A100 offline replay records processing p50/p95 of 43.47/46.32 ms; fresh-human replay records 44.48/48.96 ms. Physical Pi timing and resident-service behavior remain unmeasured.

## Evidence access and review questions

The [numeric receipt](../experiments/vcm-mixed-final-v2/post-handoff-diagnostics-20261015.json) contains exact coefficient vectors, bounds, all four cohort summaries, per-intent results, independent diagnostic conditions, raw finite-value reference agreement, supervision counts, source paths/hashes, selection details and computation accounting. The [frozen comprehensive report](EXPERIMENT-vcm-mixed-final-v2.md), [model card](../final/MODEL_CARD.md), [verification receipt](../final/final-verification.json), and [missed verbatims](../final/missed-verbatims.csv) preserve the original results.

Full raw trial records and preserved evidence are available in the local 86,039,578-byte Pi handoff ZIP at `final/vcm-pi-handoff-20261015-01.zip`, SHA-256 `7e86166167560cce4724d2a569e614a6bb2ddaaa7803d452871ceb10e9351dd5`. The ZIP remains an ignored transfer artifact; a reviewer using only GitHub needs this ZIP or access to the preserved local worktree for a complete raw-prediction audit. Committed summaries and coefficient vectors permit review of the recorded arithmetic without that payload.

The next useful research steps are to verify the target masking and human/synthetic supervision boundary; examine how calibration examples and correctness labels produce the infeasible acceptance region; assess the difference between the development selection policy and the final calibrated policy; and propose controlled training/development ablations. Determine whether the shared acoustic encoder and finite heads generalize across held-out human speakers before attributing the problem only to confidence fitting. Preserve this candidate's results and keep the unchanged benchmark outside fitting and development decisions.

No one-factor cause, fix, or claim of recognition for arbitrary unseen people has been established. Independent listening, missing intent coverage, physical identity aliases, ncnn parity, resident inference, and physical Pi qualification remain open. This brief records failure evidence for review and makes no model-readiness claim.
