"""Write a report only from completed, locked evidence; no fitting or promotion."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'.cache/phonetic-v2'
DATA = ROOT/'data/phonetic-v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ratio(metric):
    n = metric['positives']
    return f"{metric['correct']}/{n} ({100*metric['correct']/n:.1f}%)" if n else '0/0 (no coverage)'


def report():
    evidence = read(OUT/'results.json')
    audit = read(DATA/'audit.json')
    checks = read(OUT/'checks.json')
    scaling = read(OUT/'catalog-scaling.json')
    lock = read(OUT/'selection-before-test.json')
    acquisition = read(ROOT/'STT/phonetic-v2/acquisition.json')
    if lock != evidence['protocol'] or lock['manifest_sha256'] != sha(DATA/'manifest.jsonl'):
        raise ValueError('Results do not match the frozen selection and manifest')
    before = read(OUT/'preserved-before.json')
    preserved = {path: sha(ROOT/path) == value for path, value in before.items()}
    if not all(preserved.values()):
        raise ValueError('A protected current/historical artifact changed')
    training = {name: read(OUT/name/'training.json') for name in ('small', 'wide')}
    candidates = evidence['candidates']
    reserved_trials = [row for row in read(OUT/'wide_graph_expanded-test.json') if row['novel']]
    reserved_counts = {role: sum(row['role'] == role for row in reserved_trials)
                       for role in ('artist', 'song', 'device')}
    reserved_correct = {role: sum(row['role'] == role and row['prediction'] == row['reference']
                                 for row in reserved_trials)
                        for role in reserved_counts}
    reserved_distinct = len({(row['role'], row['reference']) for row in reserved_trials})
    names = {'small_greedy_expanded': 'Width 64 + greedy matching', 'small_graph_expanded': 'Width 64 + CTC graph',
             'wide_greedy_expanded': 'Width 96 + greedy matching', 'wide_graph_expanded': 'Width 96 + CTC graph'}
    lines = ['# Phonetic keyword reader and constrained grammar', '',
        '19 September 2026. Follow-up to the four-approach human-speech study.', '',
        '**Implemented and measured in isolation. The shipping VCM/STT graphs, default, configuration and prior results are unchanged.** '
        'This report separates recognition quality from software checks. New reader outputs are conditional on an accepted intent and slot role; '
        'they are not complete-command or full-pipeline accuracy.', '',
        f"**The best observed raw absolute-brightness result is {ratio(candidates['wide_graph_expanded']['raw']['absolute_brightness'])}, "
        f"at {candidates['wide_graph_expanded']['brightness_timing']['p50_ms']:.2f} ms median CPU time.** "
        f"The same acoustic model with greedy matching gets {ratio(candidates['wide_greedy_expanded']['raw']['absolute_brightness'])} "
        f"at {candidates['wide_greedy_expanded']['brightness_timing']['p50_ms']:.2f} ms. "
        'No combination passes the connected-number or name development gate, so the gated readers abstain on those roles. No candidate is ready to replace the installed reader.', '',
        f"Text-catalog expansion improves the larger graph reader's reserved-name retrieval from "
        f"{ratio(candidates['wide_graph_fixed']['raw']['reserved_names'])} to {ratio(candidates['wide_graph_expanded']['raw']['reserved_names'])} without acoustic retraining. "
        'None is accepted after confidence gating. This demonstrates the update mechanism, while leaving recognition reliability unresolved.', '',
        '## What was implemented', '',
        'A small phoneme CTC acoustic model feeds a shared-prefix Viterbi phrase graph. Brightness uses a bounded sentence grammar with 0–100 values and optional percent. '
        'Artists, songs and room-name proxies use keyword/filler phone graphs. The graph retains frame scores, unlike the earlier greedy-phone/edit matcher. '
        'A versioned catalog provides stable entity IDs, aliases and optional explicit pronunciations. Adding names recompiles that graph without altering acoustic weights; '
        'a changed catalog invalidates the adapter\'s calibration. Entries without a usable pronunciation are excluded. Tied homophones, unsupported intent/role pairs and disabled confidence gates abstain.', '',
        'The typed adapter preserves VCM intent and returns a percent or entity ID; it executes no action. Only absolute brightness has a connected numeric grammar here. '
        'This is an independent phonetic speech-recognition implementation, not a claim that renaming ASR eliminates speech-recognition work. '
        'The retired model was not restored. No Spotify, filepath source connector or Home Assistant integration was implemented.', '',
        '## Real data and split', '',
        f"The publisher-listed LibriSpeech train-clean-100 archive was downloaded and verified: {acquisition['bytes']:,} bytes, MD5 `{acquisition['md5']}`, SHA-256 `{acquisition['sha256']}`. "
        '[LibriSpeech](https://www.openslr.org/12/) provides the human recordings and [publisher checksums](https://www.openslr.org/resources/12/md5sum.txt). '
        'This run uses a bounded selection of its original utterances, not the entire official benchmark.', '',
        f"The frozen manifest has **{audit['rows']:,} original recordings**; no generated or joined audio. "
        f"The manifest assigns {audit['hours']['train']:.2f} hours to training, {audit['hours']['dev']:.2f} to development and {audit['hours']['test']:.2f} to test. "
        'General English is capped at 30 additional training hours and one hour each for development/test. Per-utterance 80-bin CMVN features are float32, '
        'with mono conversion/resampling; training English may last up to 20 seconds, while command audio retains the eight-second protocol.', '',
        '| Source | Train | Enroll | Dev | Test |', '| --- | ---: | ---: | ---: | ---: |']
    for source, splits in audit['sources'].items():
        lines.append('| '+source+' | '+' | '.join(str(splits.get(s, 0)) for s in ('train', 'enroll', 'dev', 'test'))+' |')
    lines += ['', f"Annotated speaker-ID intersections: {len(audit['cross_split_speakers'])}; reserved-name training overlap: {audit['reserved_name_training_overlap']}; "
        f"previous test recordings preserved: {audit['legacy_test_preserved']:,}. All original file hashes are checked before reuse.", '',
        'The supplemental SmartLights holdout selects two well-covered speakers using training-side transcript counts and a fixed hash, one for development and one for test. '
        'These speakers were previously seen by historical models, so **only new v2 models may be evaluated on the supplemental holdout**. '
        'The old ten-command subset is retained and reported separately for a fair historical comparison. This is a protocol revision using existing recordings, not newly collected blind speech.', '',
        '| Split | Connected numeric examples | Distinct values | Values covered |', '| --- | ---: | ---: | --- |']
    for split, values in audit['numeric_commands'].items():
        lines.append(f"| {split} | {sum(values.values())} | {len(values)} | {', '.join(sorted(values, key=int))} |")
    lines += ['', 'These counts do not establish 101-value natural-speech coverage. MLEnd remains canonical word labeling; room/place annotations are proxies for Home Assistant names. '
        'Source transcriptions and entities remain publisher/programmatic labels, not newly human-adjudicated truth. '
        '[Sonos SmartLights](https://github.com/sonos/spoken-language-understanding-research-datasets) has research-only source terms. '
        '[LibriPhrase](https://github.com/gusrud1103/LibriPhrase) was reviewed again; the published test package was kept out of training.', '',
        '## Random-initialization training', '',
        '| Model | Parameters | ONNX bytes | Best epoch / run | Development phone error: word / command / general |',
        '| --- | ---: | ---: | ---: | --- |']
    for name, r in training.items():
        errors=r['development_error']
        lines.append(f"| {name} | {r['parameters']:,} | {r['graph_bytes']:,} | {r['epoch']} / {r['epochs_run']} | "+' / '.join(f"{100*errors[g]:.1f}%" for g in ('word','command','general'))+' |')
    lines += ['', 'Both use two GRU layers, a two-epoch word warmup, maximum 36 epochs, fourfold sampling of natural parameter commands, '
        'and mean development phone error across three groups for checkpoint selection. No pretrained acoustic weights are loaded. '
        'Transcripts without a complete dictionary pronunciation, or too many target phones for their frame count, are excluded from CTC fitting; '
        f"the two runs each used {training['small']['training_examples']:,} eligible training recordings. "
        'Held-out name-enrollment rows are not used by these text-catalog models. '
        'The width comparison uses one distinct seed per model, so it is not a multi-seed estimate of a pure capacity effect.', '',
        '## Exact matches before confidence gating', '',
        '| Candidate | Isolated numbers | Connected numbers | Artists | Songs | Room/place names |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, name in names.items():
        r=candidates[key]['raw']
        lines.append('| '+name+' | '+' | '.join(ratio(r[g]) for g in ('isolated_number','natural_number','artist','song','device'))+' |')
    lines += ['', 'The connected-number pool contains 52 text-valid absolute brightness requests and eight other numeric utterances '
        '(volume, relative changes or unsupported intent/grammar). The table above preserves the broad conditional number task; '
        'it does not mean the adapter supports all those operations. The brightness-specific view is:', '',
        '| Candidate | Absolute brightness value: raw | Correct after the connected-number gate |', '| --- | ---: | ---: |']
    for key, name in names.items():
        r=candidates[key]
        lines.append(f"| {name} | {ratio(r['raw']['absolute_brightness'])} | {ratio(r['gated']['absolute_brightness'])} |")
    lines += ['', '## Confidence-gated performance', '',
        'Each group/catalog/decoder has its own development threshold: at least 95% observed precision and ten emissions, otherwise disabled. '
        'Connected numbers have a separate gate from isolated numbers. These development rules are not a statistical guarantee. All choices were locked before test inference.', '',
        '| Candidate | Isolated numbers accepted correctly | Connected numbers accepted correctly | Artist / song / room accepted correctly | Raw / gated false number matches on 300 backgrounds |',
        '| --- | ---: | ---: | --- | ---: |']
    for key, name in names.items():
        r=candidates[key]['gated']
        lines.append(f"| {name} | {ratio(r['isolated_number'])} | {ratio(r['natural_number'])} | "+' / '.join(ratio(r[g]) for g in ('artist','song','device'))+f" | {candidates[key]['raw']['negative_natural_number']['false_accepts']} / {r['negative_natural_number']['false_accepts']} |")
    lines += ['', '| Candidate | Accepted isolated-number trials: correct / emitted | Test precision | Background false accepts / 300 |', '| --- | ---: | ---: | ---: |']
    for key, name in names.items():
        r=candidates[key]['gated']['isolated_number']
        negative=candidates[key]['gated']['negative_isolated_number']
        emitted=r['emitted']+negative['emitted']
        precision=f"{100*r['correct']/emitted:.1f}%" if emitted else 'No emissions'
        lines.append(f"| {name} | {r['correct']}/{emitted} | {precision} | {negative['false_accepts']}/300 |")
    lines += ['', 'Only the width-96 greedy decoder maintains at least 95% observed precision on accepted isolated-number test clips. '
        'The graph decoders improve raw number recognition but their isolated-number calibration does not transfer at that target. '
        'This isolated-word result does not establish connected brightness reliability.']
    lines += ['', 'Negatives are the same original recordings reused as separate role trials; they are not independent speech samples for each role. '
        'Both graph readers select an allowed numeric candidate on all 300 background queries before gating. The current score does not separate correct natural commands from those forced matches well enough. '
        'A stronger explicit unknown/background alternative and better connected-command evidence are needed; merely expanding the allowed grammar does not solve rejection. '
        'Full per-value, per-intent and unknown-role results and every prediction are in `.cache/phonetic-v2/`.', '',
        '## Catalog expansion without acoustic retraining', '',
        '| Candidate | Fixed catalog: reserved names raw | Expanded: reserved names raw | Expanded: gated | Expanded using stale fixed thresholds |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key, name in names.items():
        expanded=candidates[key]
        fixed=candidates[key.replace('_expanded','_fixed')]
        lines.append(f"| {name} | {ratio(fixed['raw']['reserved_names'])} | {ratio(expanded['raw']['reserved_names'])} | {ratio(expanded['gated']['reserved_names'])} | {ratio(expanded['fixed_thresholds_after_update']['reserved_names'])} |")
    lines += ['', 'Every fixed/expanded pair uses exactly the same acoustic graph. The stale-threshold ablation is reported for diagnosis; the adapter refuses to treat it as calibrated. '
        'Correct raw retrieval alone does not justify adding a reader to the pipeline. The small reserved-name sample still limits conclusions about arbitrary playlists.', '',
        f"The {len(reserved_trials)} reserved-name trials cover {reserved_distinct} distinct role/name pairs: "
        f"{reserved_counts['artist']} artist recordings, {reserved_counts['song']} song recording and {reserved_counts['device']} room/place recordings. "
        f"The width-96 graph reader retrieves {reserved_correct['artist']}/{reserved_counts['artist']} artists, "
        f"{reserved_correct['song']}/{reserved_counts['song']} songs and {reserved_correct['device']}/{reserved_counts['device']} rooms correctly before gating. "
        'One held-out song recording is insufficient to establish general playlist-title recognition.', '',
        '| Candidate | Previously cataloged names: raw before / after expansion | Gated before / after |',
        '| --- | --- | --- |']
    for key, name in names.items():
        fixed=candidates[key.replace('_expanded','_fixed')]
        expanded=candidates[key]
        lines.append(f"| {name} | {ratio(fixed['raw']['known_catalog_names'])} / {ratio(expanded['raw']['known_catalog_names'])} | {ratio(fixed['gated']['known_catalog_names'])} / {ratio(expanded['gated']['known_catalog_names'])} |")
    lines += ['',
        '## Preserved legacy numeric comparison', '',
        'The earlier phonetic reader correctly matched **1/10** connected numeric commands before gating and **0/10** after gating. '
        'On those same ten recordings, the new candidates give:', '',
        '| Candidate | Raw | Gated |', '| --- | ---: | ---: |']
    for key, name in names.items():
        r=candidates[key]
        lines.append(f"| {name} | {ratio(r['raw']['legacy_positive']['natural_number'])} | {ratio(r['gated']['legacy_positive']['natural_number'])} |")
    lines += ['', 'The historical model was not tested on the supplemental holdout, where its previous training would invalidate the comparison.', '',
        '## CPU recognition cost', '',
        '| Candidate | Mixed commands p50 / p95 | Absolute brightness p50 / p95 | Catalog compilation |', '| --- | ---: | ---: | ---: |']
    for key, name in names.items():
        t=candidates[key]['timing']
        b=candidates[key]['brightness_timing']
        compile_ms=lock['candidates'][key]['catalog_compile_ms']
        lines.append(f"| {name} | {t['p50_ms']:.2f} / {t['p95_ms']:.2f} ms | {b['p50_ms']:.2f} / {b['p95_ms']:.2f} ms | {compile_ms:.1f} ms |")
    lines += ['', f"Warm timing uses {len(evidence['timing_audio_ids'])} identical preloaded natural-command recordings, including all 52 absolute brightness requests, "
        'one warmup and three measured passes, CPU ONNX with two intra-op/one inter-op threads and one BLAS thread. '
        'It includes frontend, graph inference and catalog/grammar decoding; excludes disk, utterance duration, cold construction and the 400 ms endpoint wait. '
        'This mixed-role timing is separate from the previous pilot\'s mostly-word timing, so those earlier latencies are not a matched comparison. '
        'These are workstation results, not physical Pi or live-microphone measurements.', '',
        f"Model session creation took {lock['candidates']['small_graph_expanded']['session_load_ms']:.2f} ms for the smaller model and "
        f"{lock['candidates']['wide_graph_expanded']['session_load_ms']:.2f} ms for the larger model. "
        'The catalog column measures object construction: the graph reader reuses an already loaded dictionary, while the unchanged greedy constructor rereads it. '
        'Those setup figures are implementation costs, not a controlled comparison of total cold startup.', '',
        '| Candidate | 100 names p50 / p95 | 1,000 names | 5,000 names |', '| --- | ---: | ---: | ---: |']
    for key, name in names.items():
        rows=scaling['models'][key.replace('_expanded','')]
        lines.append('| '+name+' | '+' | '.join(f"{r['p50_ms']:.2f} / {r['p95_ms']:.2f} ms" for r in rows)+' |')
    lines += ['', 'This separate scaling experiment uses eight original artist recordings and deterministic dictionary-word text distractors, '
        'with no generated audio and no large-catalog accuracy claim. Different audio from the main timing table means the two tables should not be directly compared. '
        'Construction time, phone-graph state counts and dictionary exclusions are retained in `catalog-scaling.json`.', '',
        '## Verification, limitations and promotion', '',
        f"The decoder passed {checks['exhaustive_ctc_cases']} exhaustive full-CTC cases and {checks['exhaustive_keyword_filler_cases']} keyword/filler cases; "
        f"the parser passed {checks['numeric_text_forms']} numeric forms and rejected {checks['malformed_text_rejected']} malformed forms. "
        'Catalog tampering/version changes, disabled gates, homophone abstention and VCM intent ownership are checked. '
        'Torch/ONNX and padded/single parity include 20-second inputs; development CPU task parity is recorded per model.', '',
        'Both 36-epoch CUDA fitting runs completed and wrote checkpoints, export metadata and parity results, then their Windows processes exited with native code 3221226505. '
        'A small GRU reproducer isolates the issue to the cuDNN path on this build: disabling cuDNN makes that reproducer exit normally. '
        'The selected checkpoints were independently re-exported in a fresh CPU process with exit code zero, and separately compared with PyTorch CPU outputs. '
        'The workflow now separates export into a CPU worker and provides an explicit `export` command. An optional `--no-cudnn` fitting path is available, '
        'but a full 36-epoch run using that slower path was not performed. Original exit statuses and diagnostics remain in the evidence folder.', '',
        'An initial development-only CPU batching check exceeded a strict absolute 1e-4 bound by 2.64e-5 on one recording, at logit magnitude 17.3. '
        'There were no frame argmax changes. Final comparison uses absolute tolerance 1e-4 plus relative tolerance 1e-5 and separately requires identical decoded values and gated decisions. '
        'The recognition precision target was not relaxed; this numerical correction occurred before any test scoring.', '',
        '**No candidate was installed.** The shipping checkpoint and its existing experimental quality limitations remain. '
        'A constrained grammar improves neither missing training coverage nor unknown rejection by itself; it can force unsupported audio toward an allowed phrase. '
        'The next acceptance decision must use natural-command precision/recall and per-value/name coverage, not text parser tests or pooled isolated words. '
        'Connected volume parsing, automatic artist/title role extraction, verified word timing, actual device names and Philippine-English coverage are not established here.', '',
        'Synthetic speech remains a future experiment only. First add missing real connected values and difficult unknown/multiple-number recordings, '
        'then evaluate any synthetic augmentation against a separate untouched human test. Wider open-vocabulary models and learned audio/text retrieval remain alternatives if this phonetic approach cannot pass the required operating points.', '',
        '## Reproduce', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe STT/phonetic-v2/acquire.py',
        '.\\.venv\\Scripts\\python.exe scripts/phonetic.py prepare',
        '.\\.venv\\Scripts\\python.exe scripts/phonetic.py train --models small wide --epochs 36',
        '.\\.venv\\Scripts\\python.exe scripts/phonetic.py export --models small wide',
        '.\\.venv\\Scripts\\python.exe scripts/phonetic.py check',
        '.\\.venv\\Scripts\\python.exe scripts/phonetic.py evaluate',
        '.\\.venv\\Scripts\\python.exe scripts/phonetic.py scale',
        '.\\.venv\\Scripts\\python.exe STT/phonetic-v2/write_report.py', '```', '',
        'See [PROTOCOL.md](PROTOCOL.md) for the declared design, [AGENTS.md](../../AGENTS.md) for the architecture contract, '
        'and [the previous study](../alternatives/REPORT.md) for the earlier four approaches. Generated data manifests live under `data/phonetic-v2/`; '
        'weights, graphs, features, thresholds and predictions live under `.cache/phonetic-v2/`. Research modules are excluded from shipping.', '']
    (ROOT/'STT/phonetic-v2/REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT/'reproducibility.json').write_text(json.dumps(dict(preserved=preserved, results_sha256=sha(OUT/'results.json'),
        report_sha256=sha(ROOT/'STT/phonetic-v2/REPORT.md'), manifest_sha256=audit['manifest_sha256']), indent=2)+'\n')
    print(str(ROOT/'STT/phonetic-v2/REPORT.md'))


if __name__ == '__main__':
    report()
