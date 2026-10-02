"""Render one research report from the frozen experiment evidence."""
from collections import Counter
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, read_jsonl


def load(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def pct(value):
    return f'{100*value:.1f}%'


def table(headers, rows):
    def clean(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(clean(v) for v in row)+' |' for row in rows])


def main():
    char = load('runs/scratch_ctc/evaluation.json')
    num = load('runs/numeric_ctc/evaluation.json')
    charconfig = load('runs/scratch_ctc/config.json')
    numconfig = load('runs/numeric_ctc/config.json')
    completed = load('runs/scratch_ctc/completed.json')
    data = load('evidence/data_audit.json')
    dedup = load('evidence/deduplication.json')
    acquisition = load('evidence/acquisition_final.json')
    validation = load('evidence/final_validation.json')
    checks = load('evidence/protocol_checks.json')
    isolation = load('evidence/pipeline_isolation.json')
    calibration = load('runs/scratch_ctc/catalog_calibration.json')['selected']
    resources = load('dataset_catalog.json')
    fixed = char['catalog']['fixed']['all']
    dynamic = char['catalog']['dynamic']['all']
    npositive = sum(char['catalog']['off'][k]['n'] for k in ['known','newly_added','out_of_catalog'])
    refs = dynamic['correct_entities']+dynamic['missed_entities']
    new_control = checks['text_only_oracle_transcript_control']['dynamic']
    new_speech = validation['new_music_entity_occurrences_all_request_categories']['dynamic']
    volume_num = validation['finite_word_model_on_full_natural_volume_commands']
    volume_char = char['numeric']['natural_volume_commands']

    source_rows = []
    for source in ['SLURP-real','FluentSpeechCommands','SnipsSmartLights','LibriSpeech','FSDD']:
        counts = [data['source_split'].get(source+'/'+s,0) for s in ['train','dev','test']]
        source_rows.append([source,*counts,sum(counts)])
    number_rows = [[name, values['n'], values['correct_accepted'], pct(values['raw_exact_value_accuracy'])]
                   for name,values in [('Human isolated numerals',num['human_isolated']),
                                       ('Constructed word sequences',num['constructed_sequences'])]]
    number_rows += [[s, num['by_source'][s]['n'],num['by_source'][s]['correct_accepted'],
                     pct(num['by_source'][s]['raw_exact_value_accuracy'])]
                    for s in ['FSDD','MLEndSpokenNumerals','MSWC-English']]
    catalog_rows = [[mode,char['catalog'][mode]['all']['correct_entities'],char['catalog'][mode]['all']['wrong_entities'],
                     pct(char['catalog'][mode]['all']['precision']),pct(char['catalog'][mode]['all']['recall']),
                     char['catalog'][mode]['negative']['accepted_requests']]
                    for mode in ['off','fixed','dynamic']]
    text_rows = [[mode,v['correct_entities'],v['wrong_entities'],pct(v['precision']),pct(v['recall']),
                  f"{v['new_catalog_entities_correct']}/{v['new_catalog_entities_total']}"]
                 for mode,v in checks['text_only_oracle_transcript_control'].items()]
    asr_rows = [[s,m['n'],pct(m['wer']),pct(m['cer']),pct(m['exact'])] for s,m in char['test_by_source'].items()]
    surveyed = table(['Resource / primary source','Fit and limitations','Access / use terms','Actual status'],
        [[f"[{r['name']}]({r['url']})",r['relevance'],r['terms'],r['status']] for r in resources])
    numeric_test = read_jsonl(ROOT/'runs/numeric_ctc/predictions.jsonl')
    main_manifest = read_jsonl(ROOT/'data/manifest.jsonl')
    numeric_manifest = read_jsonl(ROOT/'data/numeric_manifest.jsonl')
    unique = {r['audio_sha256'] for r in main_manifest+numeric_manifest}
    fresh_shards = [r for r in acquisition['completed_download_receipts'] if r['path'].endswith('.parquet')]

    # Scientific plots show test support explicitly rather than implying 101 natural values.
    figdir = ROOT/'figures'
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2,1,figsize=(12,6.7),sharex=True,layout='constrained')
    natural_x = [v for v in range(101) if num['per_value'][str(v)]['human_n']]
    natural_y = [100*num['per_value'][str(v)]['human_correct']/num['per_value'][str(v)]['human_n'] for v in natural_x]
    axes[0].scatter(natural_x,natural_y,s=35,color='#176b79',zorder=3)
    axes[0].set_title('Held-out human isolated numerals: 29 values; unmarked values have no natural test audio',loc='left',fontsize=11)
    composed_y = [10*num['per_value'][str(v)]['composed_correct'] for v in range(101)]
    axes[1].plot(range(101),composed_y,color='#9a4c16',linewidth=1)
    axes[1].scatter(range(101),composed_y,s=12,color='#9a4c16')
    axes[1].set_title('Constructed sequences: 10 per value, using concatenated held-out human word features',loc='left',fontsize=11)
    for ax in axes:
        ax.set_ylim(-5,105);ax.set_xlim(-1,101);ax.set_ylabel('Exact value (%)');ax.grid(alpha=.2)
    axes[1].set_xlabel('Requested value (0–100)')
    fig.savefig(figdir/'numeric_per_value.png',dpi=160)
    plt.close(fig)

    hchar = load('runs/scratch_ctc/history.json')
    hnum = load('runs/numeric_ctc/history.json')
    fig,axes = plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    hc = [r for r in hchar if r['stage']=='commands']
    hn = [r for r in hnum if r['stage']=='compositions']
    axes[0].plot([r['epoch'] for r in hc],[100*r['dev']['cer'] for r in hc],color='#176b79')
    axes[0].set_title('Character model: command-stage development CER',fontsize=10)
    axes[0].set_ylabel('Character error (%)')
    axes[1].plot([r['epoch'] for r in hn],[100*r['dev']['raw_exact_value_accuracy'] for r in hn],color='#9a4c16')
    axes[1].set_title('Numeric model: development exact value accuracy',fontsize=10)
    axes[1].set_ylabel('Exact value (%)')
    for ax in axes:
        ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.savefig(figdir/'learning_curves.png',dpi=160)
    plt.close(fig)

    observations = read_jsonl(ROOT/'runs/scratch_ctc/catalog_predictions.jsonl')
    errors = [r for r in observations if r['split']=='test' and r['references'] and r['category']=='newly_added'][:4]
    examples = table(['Reference request','Decoded speech','Updated-catalog output'],
                     [[r['reference'],r['hypothesis'],json.dumps(r['accepted_dynamic'],ensure_ascii=False)] for r in errors])
    timings = table(['Entries','Median dictionary rebuild (ms)'],
                    [[k,f"{v['in_memory_index_build_ms_median']:.3f}"] for k,v in checks['catalog_index_build'].items()])
    partial_gib = acquisition['partial_bytes']/1024**3
    report = f'''# STT research: spoken music requests and volume values 0–100

Research and experiments completed 19 September 2026. Everything produced for this investigation is inside `STT/`. The existing speaker pipeline was not changed or integrated. The scope is English speech describing songs/artists and numeric commands; recognizing a recording of a song, humming, or lyrics is a different task.

## Findings

**A dynamic song corpus is technically feasible without retraining the acoustic model, but this scratch-trained baseline does not establish reliable spoken-song recognition.** The fixed catalog recovered {fixed['correct_entities']}/{refs} reference entities; the updated catalog recovered {dynamic['correct_entities']}/{refs}. In the separate control using correct text instead of recognized speech, adding catalog names recovered {new_control['new_catalog_entities_correct']}/{new_control['new_catalog_entities_total']} newly introduced entity occurrences. The update mechanism therefore works at the text/catalog layer; the measured speech result remains inadequate.

**A small spoken-word vocabulary can represent every integer from 0 through 100, but representation coverage is not recognition accuracy.** The independent numeric model achieved {pct(num['human_isolated']['raw_exact_value_accuracy'])} exact-value accuracy on {num['human_isolated']['n']:,} held-out human isolated numeral recordings and {pct(num['constructed_sequences']['raw_exact_value_accuracy'])} on 1,010 constructed sequences. Natural human test audio covers 29 elementary values; 72 compound values have only constructed sequence evidence. The numeric model recovered {volume_num['correct_accepted']}/{volume_num['n']} values from available held-out natural full volume commands; the character model recovered {volume_char['correct']}/{volume_char['n']}. This small sample cannot estimate general volume-control reliability.

Two approximately 2.08-million-parameter models were actually trained from random initialization. This is an exploratory result from one seed, short training schedules, and limited data. It is not evidence that catalog biasing or scratch STT is inherently infeasible; it shows that these particular models/data do not meet the experiment's screening targets.

## What a vocabulary can and cannot teach

A **text vocabulary** supplies possible spellings, names, aliases, and permitted numeric forms. It can update a decoder's language model, contextual phrase list, pronunciation lexicon, or downstream catalog matcher. It does not itself supply evidence of what words sound like. Acoustic training needs labeled speech, or a separate self-supervised audio objective with substantial unlabelled speech. The character model here first learns from spoken digits and then from complete commands; no pretrained external weights are used.

A **finite word output vocabulary** is a useful fit for numerical parameters: zero–nineteen, twenty/thirty/forty/fifty/sixty/seventy/eighty/ninety, and hundred give 29 lexical items. The model has 31 outputs including CTC blank and an unknown class. A strict grammar maps permitted sequences to integers and rejects invalid/out-of-range text. The parser also accepts text variants such as `one hundred`, but the numeric acoustic vocabulary does not include every optional word, such as `a` or `and`.

A growing collection of song names is a different vocabulary problem. Treating each complete song title as a classifier class requires new classes and usually new labeled audio when songs are added. Character/subword recognition with a replaceable contextual catalog supports new combinations without changing the output layer. Unfamiliar pronunciations, homophones, duplicate titles, non-English names, and recognition errors can still defeat catalog matching. Established contextual ASR work supports this direction; the simple matcher tested here is not a reproduction of a full contextual decoder. [Contextual speech recognition study](https://www.isca-archive.org/interspeech_2019/zhao19d_interspeech.html).

Updating the **training corpus** means adding recordings/transcripts, then fitting and reevaluating a versioned model. Updating the **inference catalog** means replacing names/aliases and rebuilding an index while keeping model weights fixed. Only the latter was simulated for the dynamic-feature comparison. No Spotify API calls, playlist reader, folder watcher, or user filepath ingestion were implemented.

## Dataset investigation and acquisition

The survey covers **{len(resources)} resources**, including related dataset variants, text-only corpora, and supporting metadata/noise resources. This is a broad search, not a claim of exhaustive coverage or {len(resources)} independent speech datasets. Primary publisher/repository links, modality, domain fit, access, and actual acquisition status are recorded below and in [dataset_catalog.json](dataset_catalog.json).

The most useful next human-speech sources are SLURP for full music requests, MLEnd/MSWC for numeric words, and the Sonos SmartSpeaker evaluation set for artist requests. Sonos' training ontology includes volume and PlayMusic labels, but the released human SmartSpeaker recordings described by the publisher are test recordings of artist requests; its volume training text is not a ready-made full 0–100 volume audio corpus. [Sonos dataset description](https://github.com/sonos/spoken-language-understanding-research-datasets).

### Data actually pulled

- **FSDD:** complete pinned archive and all 3,000 human digit WAVs, freshly downloaded.
- **MLEnd:** 1,711 selected human numeral WAVs, freshly downloaded, covering 29 values and 151 speakers before cleanup. Two byte-identical files with conflicting value labels were quarantined; they were development examples, not test examples.
- **MSWC:** verified English split metadata plus **{acquisition['mswc_completed_decodable_files']:,}** complete, decodable clips: 1,000 each for eight, hundred, ten, play, and song. The frozen numeric experiment uses **900** clips, 450 each for eight and hundred, because that manifest was frozen before later downloads finished. The remaining completed clips are available locally but not credited as training data.
- **SLURP:** pinned annotations/metadata downloaded; **{len(fresh_shards)} complete audio parquet shard** downloaded and checksum-verified after the training manifest was frozen. The trained SLURP audio consists of isolated byte copies of the already available public dataset, not this later shard.
- **SpokeN-100:** README/source information only. No synthetic speech was used in either headline experiment.
- Selected existing **FSC, SmartLights, LibriSpeech, and SLURP** recordings were copied into `STT/data/audio/`. Their source files were only read; existing model weights/features were not reused.

Large archive transfers repeatedly timed out or ended early. Download jobs were stopped after preserving completed files. Approximately **{partial_gib:.2f} GiB** of resumable `.part` files remain under `STT/data/raw/`; these are incomplete and excluded from training. The Google Speech Commands archive did not complete. No gated account registration, access application, purchase, or contact with dataset owners occurred. [Acquisition receipts and inventory](evidence/acquisition_final.json), [MSWC per-clip inventory](data/mswc_downloaded_inventory.jsonl).

The completed files used for training remain in STT, together with hashes and frozen manifests. Catalog names are a controlled SLURP-based fixture, not data pulled from a Spotify account. Dataset access/licensing notes below describe research reuse constraints, not permission to redistribute all downloaded data or trained artifacts commercially.

### Survey

{surveyed}

## Training data and leakage controls

The character experiment uses **{data['rows']:,} unique recordings**, about **{data['hours']:.2f} hours** across all partitions. Counts below are after duration/alignment filtering and byte-level duplicate removal. These are custom experimental subsets, not published benchmark test scores.

{table(['Source','Train','Development','Test','Total'],source_rows)}

The numeric manifest contains **{len(numeric_manifest):,} unique recordings**, including human nonnumeric rejection examples. It reuses some recordings from the character experiment, so adding the two manifest sizes would double-count audio. Across both frozen manifests there are **{len(unique):,} unique audio hashes**. Numeric split counts and the 29-word vocabulary are in [numeric configuration](runs/numeric_ctc/config.json).

Speakers are assigned deterministically by a hash into train/development/test; FSDD instead explicitly holds out Theo for development and Yweweler for test. Speaker identifiers and paired-recording groups do not cross partitions. One microphone is selected per SLURP take, preferring the headset recording; SmartLights uses close recordings. Source dataset partitions are not silently assumed speaker-disjoint. Identity checks use publisher IDs, not recognition of real-world individuals across unrelated corpora.

The final audit removed **{dedup['manifest']['removed_duplicate_rows']:,} repeated character-manifest rows** and **{dedup['numeric_manifest']['removed_duplicate_rows']} repeated numeric-manifest rows**. It also quarantined the two conflicting MLEnd files. No identical audio hashes crossed splits. Both final models were restarted from random weights after this cleanup. Earlier diagnostic runs are archived under names ending `_unmasked` or `_resampled`; none supplies the final tables. [Deduplication evidence](evidence/deduplication.json).

Features are 80-channel log-mel spectra from mono 16 kHz audio, a 25 ms window and 10 ms hop, with per-utterance normalization. Recordings shorter than 0.12 s, longer than 14 s, or too short for their CTC targets are excluded. Source prompt text is normalized to English letters, spaces and apostrophes; it is not independently re-transcribed here. MLEnd targets are numeric category labels mapped to canonical words, not guaranteed verbatim transcripts.

Synthetic composition concatenates **features of human word recordings within the same split**. It supplies 5,050 training, 505 development and 1,010 test sequences for the numeric model. It does not recreate natural coarticulation, background conditions, command boundaries, or a complete spoken volume request. Some examples reuse the same word recordings, so their nominal counts are not independent speakers or independent speech events.

## Models investigated and the two models trained

| Model family | Fit for this investigation | Work performed |
| --- | --- | --- |
| CNN + bidirectional GRU + CTC | Small local character/word sequence model; explicit finite vocabulary or compositional text output | Implemented and trained twice independently from random weights: character and numeric models |
| [QuartzNet](https://arxiv.org/abs/1910.10261) | Separable convolutions offer an efficient CTC acoustic model for a larger training experiment | Literature investigation only |
| [Conformer](https://arxiv.org/abs/2005.08100) | Convolution plus attention can model richer speech context; requires a larger training study | Literature investigation only |
| [wav2vec 2.0](https://arxiv.org/abs/2006.11477) | Learns acoustic representations from unlabelled audio before supervised adaptation; a text word list is insufficient for that objective | Literature investigation only; no SSL pretraining or pretrained weights used |
| [Vosk/Kaldi grammar adaptation](https://alphacephei.com/vosk/lm) | Runtime grammar/language-model adaptation is relevant to dynamic catalogs; vocabulary/pronunciation support depends on the model | Documentation investigation only; no Vosk model loaded |

Both implemented models have two temporal convolutions (192 channels, first stride 2), three bidirectional GRU layers (192 hidden units per direction), layer normalization and a CTC output layer. They require an utterance and are **not streaming models**. CTC permits sequence supervision without manually supplied word timestamps. [Original CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

| Final run | Character STT | Finite-word numeric STT |
| --- | --- | --- |
| Parameters | {charconfig['parameters']:,} | {numconfig['parameters']:,} |
| Initialization | Random, seed 20260919 | Independent random initialization, same seed |
| Curriculum | {charconfig['args']['word_epochs']} spoken-word epochs, then {charconfig['args']['command_epochs']} command epochs | 12 word/rejection epochs, then 24 composition epochs |
| Selection | Minimum development character error | Development exact value accuracy minus a false-accept penalty |
| Selected final-stage epoch | {char['selected_epoch']} | {num['selected_epoch']} |
| Optimizer updates | {completed['optimizer_steps']:,} | {num['optimizer_steps']:,} |
| Recorded training time | {completed['seconds']/60:.2f} minutes | {num['seconds']/60:.2f} minutes |
| L2 parameter change from initialization | {completed['parameter_l2_change_from_random_initialization']:.3f} | {num['parameter_l2_change_from_initial']:.3f} |

Training used AdamW, cosine learning-rate schedules, gradient clipping and time/frequency masking. The host GPU was an RTX 4080 SUPER; download/preparation time is excluded above and concurrent host activity affects timing. Approximately 8.3 MB of float32 parameters per model does not account for activations or runtime memory. There is no Pi, embedded-device, live-microphone or Philippine-English acceptance measurement.

![Development learning curves](figures/learning_curves.png)

No test recording was used for gradient updates, epoch selection or threshold tuning. Test annotations were intentionally used only to construct the explicitly labeled catalog-membership simulation. Initial-weight digests, manifest hashes, configurations, checkpoints and completed-run evidence are retained in each run directory.

## Dynamic catalog: feature off, fixed corpus, updated corpus

The same frozen acoustic checkpoint is used in all three conditions:

1. **Off:** greedy character transcription followed by a simple music-request parser; no catalog correction.
2. **Fixed:** a {char['catalog_sizes']['fixed']}-entry catalog of training song/artist names.
3. **Dynamic:** that catalog plus {char['catalog_sizes']['additions']} additional names, for {char['catalog_sizes']['dynamic']} entries, without fitting or changing model weights.

The additions are a deterministic half of unseen development/test entity names. This is an **optimistic controlled simulation of user-supplied catalog membership**, because held-out names are explicitly made available. Full reference sentences and audio are not supplied to the matcher. Other unseen names remain absent. It is not an independently collected playlist test.

The matcher parses only the decoded text, shortlists eight similar catalog names, and reranks replacement transcripts using the frozen model's CTC scores. A single threshold/margin selected on dynamic-catalog development examples is reused for both catalog conditions. It is a simple post-transcription correction experiment, not beam-search contextual biasing. Missing command words can prevent the parser from proposing a slot at all; aggressive name substitution can produce a wrong song.

Development selected score threshold **{char['catalog_threshold']}** and margin **{char['catalog_margin']}**, yielding {calibration['correct_entities']} correct and {calibration['wrong_entities']} wrong entity predictions, with {pct(calibration['recall'])} recall. The screening target was at least 90% accepted-entity precision with at least five correct entities, chosen for this experiment. **Selection status: {char['calibration_reason']}** Such a small accepted set cannot establish reliable calibration on unseen speakers.

The held-out comparison contains **{npositive} requests with {refs} annotated song/artist entities**, plus **{char['catalog']['off']['negative']['n']} sampled other utterances without those entities**. Precision is correct emitted entities divided by all emitted entities; recall is correct emitted entities divided by reference entities. An exact title/artist string and entity type are required. The table's negative false accepts are utterances, not entity counts.

{table(['Condition','Correct entities','Wrong entities','Entity precision','Entity recall','Negative utterances accepted'],catalog_rows)}

Newly introduced entity occurrences across all test request categories: **{new_speech['correct']}/{new_speech['new_entity_occurrences']}** correctly recovered by the updated catalog. Out-of-catalog requests remain in the evaluation, and incorrect substitutions count as errors. Aggregate frame accuracy is deliberately not the headline because correctly rejecting many negative requests would conceal very poor music recall.

Examples from the held-out newly-added-catalog request group:

{examples}

### Text-only control

To test catalog updates independently of acoustics, the parser is also given the **correct reference transcript**. Catalog conditions accept only exact text matches in this control; this is a different, simpler matching rule than the speech experiment and must not be presented as STT accuracy.

{table(['Condition','Correct entities','Wrong entities','Precision','Recall','New entity occurrences correct'],text_rows)}

The updated catalog binds {new_control['new_catalog_entities_correct']} newly introduced occurrences that the fixed catalog cannot bind. The off condition can emit a raw name that is absent from the catalog; it has not resolved a canonical catalog item. Parser limitations remain even with perfect text. The fixture contains independent song/artist entities, not verified title–artist pairs or Spotify track IDs, so neither table measures actual track selection or playback success.

Rebuilding a plain in-memory dictionary was also measured:

{timings}

These are dictionary construction microbenchmarks only. They do **not** measure acoustic inference, phonetic indexing, CTC candidate search over 10,000 songs, file/playlist loading, or end-to-end latency. The matcher scans names and has not been benchmarked for a production-sized catalog. The acoustic checkpoint hash remains unchanged across catalog conditions. [Protocol checks](evidence/protocol_checks.json).

## Numbers: all 101 values represented, limited natural-speech evidence

The numeric model's held-out results are:

{table(['Test set / source','Examples','Exact values correct','Exact-value accuracy'],number_rows)}

The source rows are components of the isolated-human total, not additional independent tests. MSWC's high result covers only eight and hundred and must not be described as 0–100 accuracy. FSDD's test comprises one held-out speaker. MLEnd broadens value/speaker coverage but its canonical category labels do not verify the exact spoken wording.

The development search for a **95% accepted-value precision** threshold {'succeeded' if num['development_precision_gate_met'] else 'failed'}. The reported threshold is {num['threshold']}; {'it is a development-selected screening threshold.' if num['development_precision_gate_met'] else 'zero is a diagnostic fallback, not a recommended control threshold.'} At this setting the model falsely accepts **{num['natural_non_numeric']['ood_false_accepts']}/{num['natural_non_numeric']['n']}** natural nonnumeric test requests. Nonnumeric evaluation does not include a comprehensive silence/noise/out-of-range speech suite.

All 101 text values pass grammar round trips, including explicit 100 support. Invalid negative, fractional, ambiguous, and greater-than-100 **text** cases are rejected by the protocol checks. This is parser correctness, not proof that out-of-range **audio** will be rejected after a recognition error.

![Exact numeric value accuracy and test coverage](figures/numeric_per_value.png)

The numeric model recovered **{volume_num['correct_accepted']}/{volume_num['n']}** values when applied to the available full natural volume requests. The character model recovered **{volume_char['correct']}/{volume_char['n']}**. Those requests cover {', '.join(map(str,volume_char['values_covered']))}; they do not establish all-value reliability. A parameter-only numeric recognizer would need a verified parameter audio boundary; no word timings were invented from SLURP text spans. [Full-command numeric predictions](runs/numeric_ctc/natural_volume_predictions.jsonl).

The per-value results for **every value 0–100**, including missing natural coverage, are preserved in [numeric evaluation](runs/numeric_ctc/evaluation.json). A useful next test would collect naturally spoken full commands and verified parameter slices for all 101 values across held-out speakers, microphones and noise conditions. Constructed concatenations can support training/debugging, but they cannot replace that acceptance test.

## General transcription diagnostics

{table(['Source','Held-out utterances','Word error rate','Character error rate','Exact transcript'],asr_rows)}

The character model's pooled WER is {pct(char['test_transcription']['wer'])}, but task decisions must use music entity and numeric-value results above. Simple recurring FSC phrases are much easier than rare names or audiobook sentences. Poor SLURP/name transcription explains why dynamic catalog additions do not reliably produce correct spoken entities. The word-pretraining curriculum was executed, but its causal benefit was not isolated with a separate no-word-pretraining ablation.

## Feasibility decision and next evidence needed

**Proceed as an isolated research direction; do not promote these checkpoints into the current pipeline.** Dynamic catalog updates are a suitable design for changing song names and do not inherently require acoustic retraining. This experiment demonstrates the mechanics and exposes the current acoustic/parser bottleneck. A larger speaker-disjoint music-command corpus, pronunciation-aware or contextual decoding, and canonical title/artist ambiguity handling should be tested before drawing a stronger conclusion.

For 0–100, retain a compositional finite grammar and explicit rejection, but train/evaluate on natural connected number speech and real command contexts. Prefer exact value precision/recall, all-value coverage, unknown rejection and held-out speakers over aggregate word accuracy. The screening targets used here are research choices, not a claimed production safety standard.

Adding many more general-ASR hours may improve acoustics, but a vocabulary list alone cannot solve pronunciation, coarticulation, or domain mismatch. The most valuable missing evidence is natural spoken song/artist requests and complete volume commands—not additional duplicate recordings or text-only catalog names.

## Reproduction and artifacts

The existing `.venv` was used as a read-only runtime. No dependency installation, root manifest, root lockfile, deployment bundle or pipeline configuration was changed. Recorded versions: Python {validation['environment']['python']}, PyTorch {validation['environment']['torch']}, NumPy {validation['environment']['numpy']}, SciPy {validation['environment']['scipy']}, SoundFile {validation['environment']['soundfile']}, einops {validation['environment']['einops']}.

From the repository root in PowerShell, the frozen-manifest experiments can be rerun with:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\\.venv\\Scripts\\python.exe STT\\train.py
.\\.venv\\Scripts\\python.exe STT\\numeric_experiment.py
.\\.venv\\Scripts\\python.exe STT\\evaluate.py
.\\.venv\\Scripts\\python.exe STT\\protocol_checks.py
.\\.venv\\Scripts\\python.exe STT\\final_audit.py
.\\.venv\\Scripts\\python.exe STT\\build_report.py
```

These commands overwrite the corresponding STT run outputs. The present report is reproducible from the frozen manifests and downloaded audio; GPU executions need not be bit-for-bit identical. Do not rerun acquisition/preparation when reproducing these exact results: later downloads would change the available corpus. `deduplicate_frozen.py` is an audited one-time migration of the inherited manifest, not a regular training step. Acquisition scripts are retained separately for future new versions; they are not Spotify/file catalog integrations.

| Artifact | Location |
| --- | --- |
| Dataset survey | [dataset_catalog.json](dataset_catalog.json) |
| Character corpus and audio hashes | [data/manifest.jsonl](data/manifest.jsonl) |
| Numeric corpus and audio hashes | [data/numeric_manifest.jsonl](data/numeric_manifest.jsonl) |
| Frozen catalog fixtures and protocol | [catalogs/protocol.json](catalogs/protocol.json) |
| Character checkpoint | [runs/scratch_ctc/commands_best.pt](runs/scratch_ctc/commands_best.pt) |
| Numeric checkpoint | [runs/numeric_ctc/compositions_best.pt](runs/numeric_ctc/compositions_best.pt) |
| Detailed speech/catalog predictions | [runs/scratch_ctc/catalog_predictions.jsonl](runs/scratch_ctc/catalog_predictions.jsonl) |
| Numeric predictions, confidence and composition components | [runs/numeric_ctc/predictions.jsonl](runs/numeric_ctc/predictions.jsonl) |
| Final checks and environment | [evidence/final_validation.json](evidence/final_validation.json) |
| Acquisition status / hashes | [evidence/acquisition_final.json](evidence/acquisition_final.json) |
| Isolation verification | [evidence/pipeline_isolation.json](evidence/pipeline_isolation.json) |

Character checkpoint SHA-256: `{char['checkpoint_sha256']}`. Numeric checkpoint SHA-256: `{num['checkpoint_sha256']}`. Manifest hashes are recorded in both configurations and [final validation](evidence/final_validation.json).

Final isolation audit: **{isolation['before_files']:,} existing pipeline files compared, {len(isolation['changed'])} changed, {len(isolation['added_outside_STT'])} added and {len(isolation['removed_outside_STT'])} removed outside STT** within the audited scope. {isolation['hashed_files']} non-audio files were content-hashed; existing audio files were compared by byte count and modification time. Git internals, the environment/cache, and STT itself are excluded from that snapshot. Pre-existing uncommitted work was preserved. The trained models, data, scripts, report and figures stay in STT.
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({'report':str(ROOT/'REPORT.md'),'survey_resources':len(resources),'words':len(report.split()),
                     'unique_audio_hashes':len(unique),'numeric_test_rows':len(numeric_test)},indent=2))


if __name__ == '__main__':
    main()
