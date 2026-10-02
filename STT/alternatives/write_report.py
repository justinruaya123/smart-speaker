"""Render the study report from final audited results; never select a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.cache/parameter-study'
HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def pct(value):return '—' if value is None else f'{value*100:.1f}%'
def count(item):return f"{item['correct']}/{item['positives']} ({pct(item['recall'])})"


def main():
    report=read(OUT/'results.json');audit=read(ROOT/'data/parameter-study/audit.json')
    assert report['protocol']['manifest_sha256']==audit['manifest_sha256'],'Do not publish superseded scores'
    for m in report['models'].values():assert m['training']['manifest_sha256']==audit['manifest_sha256']
    rows=[json.loads(l) for l in (ROOT/'data/parameter-study/manifest.jsonl').read_text().splitlines()]
    catalog=read(ROOT/'data/parameter-study/catalogs.json');runtime=read(OUT/'runtime-validation.json')
    capacity=read(OUT/'vcm-capacity.json');scaling=read(OUT/'catalog-scaling.json')
    names={'fixed':'Fixed KWS / direct slot head','prototype':'Audio-enrollment embedding','phoneme':'Phoneme CTC + keyword matching','character':'Compact character CTC STT'}
    def selected(name):return report['models'][name]['variants']['expanded_5shot' if name=='prototype' else 'expanded']
    hours={s:sum(r['duration'] for r in rows if r['split']==s)/3600 for s in ('train','enroll','dev','test')}
    natural=[r for r in rows if r['split']=='test' and r['kind']=='natural' and any(t['role']=='number' for t in r['targets'])]
    values=sorted({int(t['value']) for r in natural for t in r['targets'] if t['role']=='number'})
    lines=['# Human-only speech parameter alternatives', '', '19 September 2026. Final audited feasibility experiment.', '',
      '**Recommendation: retain VCM for intent and stable finite controls; treat expanding names as a separate parameter-reading problem. None of these four small, from-scratch candidates establishes reliable natural numeric commands or arbitrary music/device names.** The code now retires the legacy convolutional ASR model. The installed experimental word-STT brightness checkpoint remains unchanged; the four candidates are isolated and were not promoted.', '',
      'A vocabulary list is a label specification, not acoustic training data. A fixed classifier needs recordings and new output weights for new classes. Audio enrollment can add a name from real recordings; a phonetic or character reader can use an updated text catalog without changing its acoustic weights, but only if it already recognizes the relevant speech well.', '',
      '## Four approaches actually trained and tested', '',
      '| Approach | Parameters | FP32 ONNX bytes | Selected epoch / epochs run | CPU p50 / p95 |',
      '| --- | ---: | ---: | ---: | ---: |']
    for name,m in report['models'].items():
        tr=m['training'];tm=m['latency']
        lines.append(f"| {names[name]} | {tr['parameters']:,} | {tr['graph_bytes']:,} | {tr['epoch']} / {tr['epochs_run']} | {tm['p50_ms']:.2f} / {tm['p95_ms']:.2f} ms |")
    lines += ['', 'Fixed KWS uses a temporal depthwise residual encoder and a fixed softmax label set. Audio enrollment uses a separately initialized copy of that encoder with classification plus supervised contrastive loss, then normalized mean prototypes. Phoneme and character models use separate width-64, two-layer bidirectional GRU CTC networks with convolutional inputs. The phonetic reader uses CMUdict pronunciations, greedy CTC and exact substring edit-distance matching; the character reader applies the same matching idea to decoded text. These are small feasibility implementations, not reproductions of published SOTA scores.', '',
      'All acoustic weights started randomly, using one seed per approach (31–34). No legacy weights, pretrained acoustic encoder, cloud service, LLM, TTS or joined-word audio was used. The CTC models received three isolated-word warmup epochs, then the mixed human corpus. Maximum fitting was 28 epochs, with development selection and patience 9. This limited-data pilot is not a hyperparameter sweep or general-English pretraining.', '',
      '## Exact parameter recognition on original human recordings', '',
      'These are **raw value/name matches before confidence gating**, including abstentions in the denominator. The slot role is supplied from annotations, so this is a conditional parameter test. It does not establish automatic artist-versus-title selection, word boundaries, complete-command correctness or end-to-end VCM accuracy. Room/place entities are proxies for Home Assistant names, not the user’s actual device inventory.', '',
      '| Approach | Isolated number words | Connected numeric commands | Artists | Songs | Room/place names |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        g=selected(name)['groups']
        lines.append('| '+names[name]+' | '+' | '.join(count(g[k]['raw']) for k in ('isolated_number','natural_number','artist','song','device'))+' |')
    lines += ['',f"Only **{len(natural)} connected numeric commands** survive the frozen test protocol, covering values {', '.join(map(str,values))}. There are many more isolated number words. These results cannot establish all 101 values, optional-percent robustness, coarticulation or microphone reliability. Counting successful text parsing of `brightness 50%` would not fill that acoustic gap.", '',
      'Per-role thresholds were chosen on development data for at least 95% precision and 10 emissions. A role that could not meet that rule was disabled with an unreachable threshold. The rule is an operating-point heuristic, not a statistical precision guarantee; the test checks whether it transfers.', '',
      '| Approach | Accepted isolated numbers / all | Connected numeric values accepted correctly | Artist/song/device roles passing development gate | All accepted parameters: correct / emitted |',
      '| --- | ---: | ---: | --- | ---: |']
    for name in names:
        v=selected(name);g=v['groups'];active=[role for role in ('artist','song','device') if v['policies'][role]['passed']]
        lines.append(f"| {names[name]} | {count(g['isolated_number']['gated'])} | {count(g['natural_number']['gated'])} | {', '.join(active) or 'None'} | {v['gated']['correct']}/{v['gated']['emitted']} ({pct(v['gated']['precision'])}) |")
    lines += ['', 'The pooled accepted-parameter column is dominated by number/context words and must not be interpreted as name accuracy. The 300 original background recordings are queried separately for each of five roles, producing 1,500 negative role trials, not 1,500 independent recordings.', '',
      '| Approach | Raw / gated false number matches on 300 backgrounds | Raw / gated false artist matches on 300 backgrounds |',
      '| --- | ---: | ---: |']
    for name in names:
        g=selected(name)['groups'];a=g['negative_number'];b=g['negative_artist']
        lines.append(f"| {names[name]} | {a['raw']['false_accepts']} / {a['gated']['false_accepts']} | {b['raw']['false_accepts']} / {b['gated']['false_accepts']} |")
    lines += ['', 'The high raw false-match rates of nearest-prototype and fuzzy catalog matching show why “choose the nearest name” is insufficient. Gating suppresses errors but also rejects useful speech. No failed gate was relaxed to improve the reported recall.', '',
      '## Updating a corpus without retraining', '',
      f"The fixed fixture contains {len(catalog['fixed']['artist'])} artists, {len(catalog['fixed']['song'])} songs and {len(catalog['fixed']['device'])} room/place labels. The expanded fixture contains {len(catalog['expanded']['artist'])}, {len(catalog['expanded']['song'])} and {len(catalog['expanded']['device'])}, respectively. Both have the 0–100 text grammar. These lists were declared from the dataset ontology before fitting, not customized to each query. No Spotify/playlist/filepath integration was implemented.", '',
      f"{len(catalog['novel'])} entity labels were reserved using training counts and a deterministic hash. All training utterances containing those phrases or carrying those entity labels were withheld from gradient updates; their original audio could be used only for enrollment. Enrollment and test speakers are disjoint. Zero reserved labels overlap the final acoustic training classes.", '',
      '| Method | Raw correct reserved-name queries without update | Raw correct after update | Correct accepted after update |',
      '| --- | ---: | ---: | ---: |']
    for name in names:
        variants=report['models'][name]['variants']
        for shots in ([1,3,5] if name=='prototype' else [0]):
            suffix=f'_{shots}shot' if shots else '';a=variants['fixed'+suffix]['novel_names'];b=variants['expanded'+suffix]['novel_names']
            label=names[name]+(f' (up to {shots} enrollment speakers)' if shots else '')
            lines.append(f"| {label} | {count(a['raw'])} | {count(b['raw'])} | {count(b['gated'])} |")
    novel_detail=selected('prototype')['novel_names']['by_role']
    lines += ['',f"The reserved-name test consists of {novel_detail['artist']['raw']['positives']} artist, {novel_detail['song']['raw']['positives']} song and {novel_detail['device']['raw']['positives']} room/place query occurrences. It is too small for a general music-name claim, especially for songs. Enrollment uses up to the requested number of distinct available speakers; some labels have fewer. Whole original commands are embedded because no verified name-only audio boundaries are available.", '',
      'The fixed classifier’s acoustic class set cannot expand by editing JSON. The audio-enrollment mechanism works without acoustic retraining but requires recordings. Phoneme/character catalog changes also leave the acoustic graph unchanged. Those mechanisms are technically feasible; their recognition quality here is insufficient.', '',
      f"The phonetic catalog has {len(selected('phoneme')['catalog_OOV'])} unpronounceable-by-this-dictionary entries in the expanded fixture. Proper names, spelling variants, tokenization and foreign names need explicit pronunciations or a separately tested G2P model. Dictionary presence still does not imply that the acoustic model recognizes a name. Homophone/tied matches abstain.", '',
      '## Does VCM suffice?', '',
      'For stable intent families and a finite vocabulary, **yes in principle**. The current VCM already has 103 amount outputs: NONE, OTHER and every integer 0–100. However, all 101 numeric value thresholds are 1.1, so its current policy disables them. More text labels will not fix that. Proper connected human training and per-value calibration could make a finite numeric head useful without general transcription.', '',
      f"For names, its target head has {capacity['parameter_heads']['target']} entries including NONE/OTHER, and its media head is only NONE/OTHER/GENERIC. A newly added artist or device ID is not a new acoustic output. Keep VCM for intent, and augment the parameter stage for changing catalogs.", '',
      'Recommended architecture:', '', '```text',
      'PCM → existing VAD → VCM accepts intent',
      '    → fixed finite heads for supported stable controls',
      '    → otherwise an approved parameter reader + versioned catalog/grammar',
      '       → typed percent/delta or entity ID, calibrated confidence, or abstention',
      '    → shared validation → command JSON', '```', '',
      'A future music grammar must resolve artist/title roles and multiple slots from the utterance rather than receiving the annotation role used by this study. Percentages need the explicit 0–100 grammar, distinction between absolute and relative changes, and rejection of conflicting values. A reader must never change VCM intent or execute device actions.', '',
      'My next architecture choice is **small phonetic or learned audio/text KWS with a command grammar**, after expanding real phrase-level training. For user-enrolled aliases, keep audio-template/embedding enrollment as a separate option. A fifth researched approach, [OHF Speech-to-Phrase](https://github.com/OHF-voice/speech-to-phrase), shows how command templates and device names can form a constrained grammar; its downloadable models were not trained or installed here. See [the dataset/model survey](DATASETS_AND_MODELS.md) for DS-CNN, TC-ResNet, BC-ResNet, MatchboxNet, few-shot KWS and audio/text alternatives.', '',
      '## Dataset, preprocessing and audit', '',
      f"{audit['rows']:,} original human recordings were collated. Training uses {audit['by_split']['train']:,} ({hours['train']:.2f} hours), development {audit['by_split']['dev']:,} ({hours['dev']:.2f} hours), test {audit['by_split']['test']:,} ({hours['test']:.2f} hours), and held-out enrollment {audit['by_split']['enroll']:,} ({hours['enroll']:.2f} hours). Not every recording carries a studied parameter. The feature protocol accepts 0.1–8 seconds, converts to mono 16 kHz and uses per-utterance 80-bin CMVN. There is no audio augmentation or synthetic composition.", '',
      '| Human source | Train | Enrollment | Dev | Test |', '| --- | ---: | ---: | ---: | ---: |']
    for source,counts in audit['source_split'].items():lines.append('| '+source+' | '+' | '.join(str(counts.get(k,0)) for k in ('train','enroll','dev','test'))+' |')
    lines += ['', 'The corpus retains all chosen parameter examples and a deterministic bounded sample of other speech. It is heavily weighted toward isolated numeric words; only a small portion is general English or individual music names. Root calibration recordings were excluded. Speaker/recording hashes were audited, paired microphones were reduced to one recording and conflicting duplicates were quarantined. MLEnd targets retain their canonical-category caveat. Public data and matching metadata can still contain errors; these are not manually reviewed ground-truth command labels.', '',
      'A first pilot exposed a source inconsistency: a birthday reminder had unrelated music entity annotations. A uniform punctuation-insensitive entity/transcript consistency rule quarantined three records across all splits. Reserved-name exclusion now also checks explicit entity labels. All four models were retrained from random weights with unchanged seeds and recipes. The superseded pilot is retained in `.cache/parameter-study/superseded-v1-annotation-audit/`. Final test scoring is a rerun after this data-quality repair, **not a newly collected blind test**; no test error was used to choose hyperparameters or confidence thresholds.', '',
      'Initial GPU evaluation also failed a tight confidence parity check against CPU. Final development calibration and test scoring use the actual CPU ONNX engine, with 96 matched single-versus-batched task checks per model. Variable-length Torch/ONNX and padding parity are recorded in each training JSON. Model/threshold/catalog choices are persisted before final test scoring. This avoids reporting CUDA confidence estimates as deployed CPU behavior.', '',
      '## Recognition cost and expanding-list cost', '',
      'Main timing uses the same 96 original held-out waveforms for each candidate: one warmup pass, three measured passes, CPUExecutionProvider, two intra-op threads and one inter-op thread. It includes the frontend, acoustic inference, decoding and catalog matching; excludes file reads, speech duration, capture and the default 400 ms VAD endpoint wait. The model graph itself takes less time than the complete recognition path. These are Windows workstation measurements, not Raspberry Pi or live-microphone results.', '',
      '| Reader | 100 text names p50 / p95 | 1,000 names | 5,000 names |', '| --- | ---: | ---: | ---: |']
    for name,sizes in scaling['models'].items():
        lines.append('| '+names[name]+' | '+' | '.join(f"{sizes[str(n)]['p50_ms']:.2f} / {sizes[str(n)]['p95_ms']:.2f} ms" for n in (100,1000,5000))+' |')
    lines += ['', 'Scaling is a separate timing-only test on eight original artist-query recordings with deterministic generated **text** distractors. It generates no audio and establishes no large-catalog accuracy. The simple matching implementation scans candidates; a large playlist needs a better index/grammar or learned retrieval and new false-accept calibration. An audio-enrollment catalog cannot acquire new acoustic prototypes from text alone.', '',
      'The retired model had 1,321,757 parameters and a 5,319,980-byte graph; the new candidate graphs are substantially smaller. Its old latency numbers used a different, constructed-command test set, so they are not a valid matched speed comparison against this human-only test. It was not restored for this study. Smaller graphs alone are not evidence of reliable or faster end-to-end commands.', '',
      '## Runtime changes and verification', '',
      'Removed the retired model implementation, continuation trainer, runtime loader, legacy switches/custom-media fallback, active checkpoint/graph and ASR/dual-model release ZIPs. The transcription dataset and audio remain. New brightness preparation is human-only under `data/stt-human/`; the old constructed dataset/report are historical. `training.update_artifacts=false` prevents automatic replacement by new experiments. The installed VCM and word-STT graph hashes are unchanged.', '',
      f"Runtime component, VAD/capture simulation, intent ownership, retirement and brightness grammar checks passed, including 404 numeric parser forms. Full existing validation audited {runtime['validation']['data_audit']['audio_files_verified']:,} original recordings and replayed {runtime['validation']['examples']:,} held-out examples. Its quality gate remains **failed**: the current full pipeline emitted {runtime['validation']['pipeline']['commands']['correct_complete']}/{runtime['validation']['pipeline']['commands']['emitted']} correct complete finite commands, with {runtime['validation']['pipeline']['commands']['finite_references']} finite references. These are existing-system replay results, separate from the new parameter pilot. No live mic, physical Pi, Spotify or Home Assistant test was performed.", '',
      '## Reproduce and extend', '', '```powershell',
      '.\\.venv\\Scripts\\python.exe scripts/parameter_study.py prepare',
      '.\\.venv\\Scripts\\python.exe scripts/parameter_study.py train --epochs 28',
      '.\\.venv\\Scripts\\python.exe scripts/parameter_study.py evaluate',
      '.\\.venv\\Scripts\\python.exe scripts/parameter_study.py scale',
      '.\\.venv\\Scripts\\python.exe STT/alternatives/write_report.py',
      '.\\.venv\\Scripts\\python.exe train.py --tests-only --no-setup',
      '.\\.venv\\Scripts\\python.exe publish.py', '```', '',
      'Use the source map and contracts in [AGENTS.md](../../AGENTS.md). `--models fixed prototype phoneme character` selects training candidates. Frozen manifests/catalogs are in `data/parameter-study/`; checkpoints, graphs, per-query predictions, per-value results, training histories, selection locks and timings are in `.cache/parameter-study/`. Exact provenance is recorded in `audit.json`, the pronunciation receipt and `reproducibility.json`.', '',
      'The next data priority is a larger, speaker-stratified connected-command benchmark using more available natural commands, followed by real recordings for missing 0–100 values and confusable/out-of-range negatives. This study’s deterministic split placed most SmartLights examples in training, so its small numeric holdout must not be confused with the full public corpus. MSWC/LibriPhrase and a larger real English phonetic pool can then strengthen representations. Add independent human names in natural commands and real enrollment examples for device aliases. Synthetic speech for rare names/numbers is a later, separately labeled roadmap experiment; it must keep the real-human test untouched. The [survey](DATASETS_AND_MODELS.md) distinguishes locally used datasets, surveyed sources, access-gated data and excluded synthetic corpora.', '',
      'No candidate was promoted on these results. The best approach depends on the contract: finite KWS for stable controls, audio enrollment when recordings can be supplied, and phonetic/audio-text matching plus a grammar when text catalogs must expand. The present bare datasets do not yet demonstrate the desired natural-command reliability.', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import sys,platform,torch,onnxruntime
    files=[ROOT/'scripts/modules.json',*list((ROOT/'scripts').rglob('*.py'))]
    provenance={'manifest_sha256':audit['manifest_sha256'],'catalog_sha256':hashlib.sha256((ROOT/'data/parameter-study/catalogs.json').read_bytes()).hexdigest(),
                'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'onnxruntime':onnxruntime.__version__,'training_GPU':torch.cuda.get_device_name(0)},
                'current_reproducible_source_hashes':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'selection':read(OUT/'selection-before-test.json'),'VCM_capacity':capacity}
    (OUT/'reproducibility.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(HERE/'REPORT.md')


if __name__=='__main__':main()
