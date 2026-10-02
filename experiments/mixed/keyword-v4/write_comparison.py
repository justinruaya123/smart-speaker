"""Render frozen result artifacts; does not perform fitting or inference."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
def read(path):return json.loads((ROOT/path).read_text())
def pct(value):return 'n/a' if value is None else f'{100*value:.2f}%'
def write(path,lines):(ROOT/path).write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    base='experiments/mixed/keyword-v4/'
    mixed=read(base+'results.json');synthetic=read('experiments/synthetic/keyword-v4/results.json')
    old=read('experiments/mixed/results.json');control=read('experiments/mixed/human-only-results.json')
    replay=read(base+'endpoint-results.json');diag=read(base+'post-test-diagnostics.json')
    selection=read(base+'selection-before-test.json');checks=read(base+'checks.json')
    lines=['# Fixed-keyword VCM: completed comparison','',
        '**Human intent recognition and gated command recall improved, but the all-intent and numeric quality targets still failed.** The selected mixed candidate is an experimental temporal CNN intent model followed by an ordered fixed-word reader. Brightness, color and absolute volume remain disabled by development gates. It is not installed as the default.','',
        'The requested Kaggle corpus was acquired and incorporated into expanded-data candidates. Speech-to-Intent-Micro was studied at a pinned commit and a small independent Conv2D baseline was trained. See [the source study](SOURCE_STUDY.md), [the mixed report with a table for every intent](REPORT.md), and [the synthetic report](../../synthetic/keyword-v4/REPORT.md).','',
        '## Human comparison on original, presegmented recordings','',
        '| Experiment | Raw intent accuracy | Exact accepted / valid | Exact recall | Output precision | Negative false accepts |',
        '|---|---:|---:|---:|---:|---:|']
    for name,obj in [('Prior mixed v1',old),('Prior human-only control',control)]:
        v=obj['tests']['human_test']['overall'];lines.append(f"| {name} | {pct(v['raw_family_accuracy'])} | {v['correct_accepted']}/{v['positive_n']} | {pct(v['exact_command_recall'])} | {pct(v['accepted_precision'])} | {v['false_accepts']}/{v['negative_n']} |")
    for name,obj in [('Mixed keyword v4',mixed),('Synthetic keyword v4',synthetic)]:
        v=obj['tests']['human']['gated']['overall'];lines.append(f"| {name} | {pct(v['family_accuracy'])} | {v['correct_accepted']}/{v['positive_n']} | {pct(v['exact_recall'])} | {pct(v['precision'])} | {v['false_accepts']}/{v['negative_n']} |")
    lines+=['',
        'The presegmented mixed result improves raw intent accuracy from 65.45% to 78.20%, and exact accepted recall from 1.19% to 37.96%. It also exceeds the earlier human-only control (15.15% recall at 89% precision). These experiments change data, architecture and calibration together; this is not an attribution of the gain to FSC alone.','',
        '**The raw command result did not improve:** before confidence gates, the earlier mixed model had 54.47% exact recall; the new ordered reader has 44.60% (524/1,175). It improves precision before gates from 67.30% to 85.90%. Better intent classification and the replacement of the earlier failed coordinate calibration both contribute to the accepted-result improvement. Ordered cue validation still loses correct commands when cues are missing.','',
        '## Every intent, including the actual endpoint path','',
        '| Intent | Valid human commands | Raw intent recall | Original-file exact recall | Original-file precision | VAD exact recall | VAD output precision |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for family,v in mixed['tests']['human']['gated']['per_intent'].items():
        e=replay['tests']['human']['vad_endpoint_runtime']['per_intent'][family]
        lines.append(f"| {family} | {v['positive_n']} | {pct(v['family_recall'])} | {pct(v['command_recall'])} | {pct(v['command_precision'])} | {pct(e['exact_command_recall'])} | {pct(e['output_precision'])} |")
    lines+=['','Raw intent recall uses all examples bearing that family label, including unresolved requests; command recall uses only valid references. Output precision groups by the emitted family and includes cross-family mistakes. UNKNOWN has no valid commands: 476/478 are rejected in the original-file path and 478/478 through VAD.','',
        'The mixed VAD path emits 379 outputs, 363 correct, for 30.89% recall and 95.78% precision. It falsely accepts 3/628 negative or unresolved recordings (0.48%). Time-query recall is zero and volume precision is 82.14%; aggregate precision does not qualify every intent for use. The policy was calibrated on original development utterances, not VAD-transformed development. That deployment mismatch is now measured and must be addressed in a new development-selected version, not by tuning this test.','',
        'The original-file runtime agrees with the batched evaluation on every acceptance/command decision for both historical test sets. VAD loses 123 previously correct human commands and gains 40. Among lost commands, no-speech rejection is not the main explanation: changed family/confidence and missing keyword/readiness/command evidence account for the losses. See [post-test diagnostics](post-test-diagnostics.json). No thresholds were changed after these observations.','',
        '## Numeric and directional coverage','',
        '| Operation | Human valid references | Correct before gates | Correct accepted | Accepted output precision |',
        '|---|---:|---:|---:|---:|']
    for operation,v in mixed['tests']['human']['gated']['operations'].items():
        raw=mixed['tests']['human']['before_confidence_gates']['operations'][operation]
        lines.append(f"| {operation} | {v['positive_n']} | {raw['correct']} | {v['correct']} | {pct(v['precision'])} |")
    lines+=['',
        'There are only 179 original human absolute-brightness training commands covering 13 distinct values, and 31 absolute-volume commands covering 11 values. The existing synthetic training data covers all 101 values for each family, and isolated human number words add acoustic supervision. Neither proves connected human-number recognition. Absolute volume has only three development positives, below the ten-accept minimum for enabling an operation. There is no human acoustic evidence here supporting all 0–100 values.','',
        'The grammar passes 404 textual cases (101 values × two numeric families × optional percent), plus conflict/order tests. Those are software checks. The new mixed acoustic system gets only 3/75 absolute numeric test commands correct before gates, compared with 2/75 previously; both accept zero after calibration. The requested reliable “set brightness to 100 percent” behavior has not been achieved.','',
        '## Ordered-cue case study','',
        'The path takes an external wake-end sample, endpoints following speech, selects an intent, then reads ordered cues only after acceptance. `SET -> BRIGHTNESS -> TO -> FIFTY -> PERCENT` represents an intended cue sequence, not a successful measured example. The finite reader distinguishes TO from BY, bounds values to 0–100, treats bare UP/DOWN as +/-10 percentage points, and rejects misplaced/conflicting cues.','',
        'A real development recording (`smartlights/2623/far`, “Put the brightness in the shed at fifty.”) illustrates the remaining failure. The classifier identifies BRIGHTNESS. Offline diagnostics decode brightness/to/twenty and infer 20 instead of 50. The frozen runtime rejects at the disabled brightness intent gate before invoking the reader. Another recording of the same phrase lacks brightness context in the decoded cues. Preserving order cannot repair missing or wrong acoustic words. See [ordered examples](ordered-cue-examples.json) and [development diagnostics](development-diagnostics.json).','',
        'In another development phrase (“turn up the lights in here please”), the intent model selects POWER_OFF while the reader detects turn/up/lights. The context check rejects the conflict; it does not change the accepted family to rescue the command. VCM ownership is preserved.','',
        '## Both synthetic and mixed experiments','',
        '| Model | Test speech | Raw intent accuracy | Exact recall before gates | Accepted / correct | Gated exact recall | Gated precision |',
        '|---|---|---:|---:|---:|---:|---:|']
    for mode,obj in [('mixed',mixed),('synthetic',synthetic)]:
        for source,v in obj['tests'].items():
            a=v['gated']['overall'];b=v['before_confidence_gates']['overall']
            lines.append(f"| {mode} | {source} | {pct(a['family_accuracy'])} | {pct(b['exact_recall'])} | {a['accepted']}/{a['correct_accepted']} | {pct(a['exact_recall'])} | {pct(a['precision'])} |")
    lines+=['',
        'The synthetic-only model still fails transfer to human speech and all its human-development gates remain disabled. The selected mixed intent model is the earlier 1D CNN; the newly added FSC recordings train the selected keyword reader and the evaluated shared/Micro-inspired intent candidates. The earlier CNN already used the pre-existing FSC subset. This distinction matters when interpreting the data addition.','',
        '## Failure cascade for each valid human intent','',
        '| Intent | Valid | Correct family | Pass intent gate | Valid reader + ready | Exact before final gate | Exact accepted |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for family,value in diag['cascade'].items():
        if family=='UNKNOWN':continue
        s=value['stages'];counts=[s.get(k,0) for k in ('valid_reference','family_correct','intent_gate_passed','valid_parameters_and_ready','exact_before_command_gate','exact_accepted')]
        lines.append('| '+family+' | '+' | '.join(map(str,counts))+' |')
    timing=checks['cpu_timing']
    lines+=['','## Implementation, validation and limits','',
        '- Canonical implementation: `scripts/models/keyword_vcm.py`, `keyword_intent.py`, `scripts/train/keyword_data.py`, `keyword_training.py`, `scripts/runtime/keyword_runtime.py`, and `scripts/test/keyword_eval.py`. Registered as research-only in `scripts/modules.json`.',
        '- The selected mixed bundle loads 1,243,830 parameters across two FP32 graphs totaling 5,029,167 bytes. It is a CPU ONNX experimental command path, not the repository’s 29 KB INT8 model.',
        f"- Identical 60 preloaded development recordings, two CPU threads: p50 {timing['p50_ms']:.2f} ms, p95 {timing['p95_ms']:.2f} ms; session loading {timing['session_load_ms']:.2f} ms. Timing includes frontend and conditional reader/validation. It excludes disk, speech duration, VAD and the 400 ms endpoint wait; many rejected intents skip the reader.",
        '- Passed variable-length/batch/padding ONNX parity, ordered numeric/conflict tests, rejection-before-reader checks, all 35,687 manifest-record hashes, speaker separation, root runtime/parser tests, and preservation of installed artifacts and prior result JSONs.',
        '- Replayed all 1,803 human and 672 synthetic historical test recordings through both frozen models using both original-file and VAD paths. A separate software fixture verifies that an externally supplied 500 ms wake boundary excludes preceding samples and preserves following speech. No wake detector, live microphone, device action or physical Pi was tested.',
        '- All 48 keyword epochs and exports completed. Both training launchers returned code 1 without an independently captured native Python exit code. Separate CPU exports returned 0 and graph checks passed. The 40-epoch Micro candidate returned 0. See `execution-receipt.json`; do not infer a specific CUDA fault from the launcher code.',
        '- Human labels are inherited silver annotations. TTS intended transcripts and requested Filipino accents are not acoustically adjudicated. The prior test has already been used historically, so this is a comparison rather than an untouched confirmation.',
        '- No online vocabulary expansion was tested. Adding a word to JSON does not add an acoustic output class. Names, artists, new-corpus tests and automatic STT fallback remain outside this phase. Installed VCM/STT defaults are unchanged.','',
        '## Run the isolated pipeline','',
        '```powershell',
        '.venv/Scripts/python.exe scripts/keywords.py predict --mode mixed --audio path/to/command.wav',
        '.venv/Scripts/python.exe scripts/keywords.py predict --mode mixed --audio path/to/recording.wav --wake-end-ms 900 --vad',
        '.venv/Scripts/python.exe scripts/keywords.py list-mics',
        '.venv/Scripts/python.exe scripts/keywords.py listen --mode mixed --mic 2 --vad-debug',
        '```','',
        'The microphone command must be armed by the assistant; it is not a wake listener. Outputs are typed JSON or abstentions and never execute an action. Use `--mode synthetic` for the corresponding frozen synthetic control.','',
        'Analysis commands are `scripts/keywords.py check --mode both` and `scripts/keywords.py report --mode both`; the comparison is rendered by `experiments/mixed/keyword-v4/write_comparison.py`. Fitting, calibration and evaluation refuse to overwrite completed/frozen evidence. New training or endpoint calibration requires a new study version. Saved checkpoint histories record the greedy-CTC selection used during fitting; the final finite-number decoder was added before acceptance calibration and test.','',
        'The next justified experiment is to collect/adjudicate connected human brightness, direction and numeric commands from independent speakers, then fit and calibrate through the same endpoint path. A controlled same-data comparison should separate the effect of the intent architecture, ordered reader, FSC expansion and calibration. These test outcomes must not choose the next thresholds.']
    write(base+'COMPARISON.md',lines)
    for mode,obj in [('mixed',mixed),('synthetic',synthetic)]:
        folder=f'experiments/{mode}/keyword-v4/';r=read(folder+'endpoint-results.json')
        ep=[f'# {mode.title()} frozen endpoint replay','',
            'Original historical files with wake boundary zero, using the unchanged frozen policy. There are no annotated wake words in these files. This supplemental evaluation was specified after the original-file results were seen and does not select a model or threshold.','',
            '| Speech | Accepted outputs | Correct outputs | Exact / valid commands | Exact recall | Output precision | Negative false accepts |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for source,value in r['tests'].items():
            v=value['vad_endpoint_runtime']['overall'];ep.append(f"| {source} | {v['accepted_outputs']} | {v['correct_outputs']} | {v['exact_commands']}/{v['positive_n']} | {pct(v['exact_command_recall'])} | {pct(v['output_precision'])} | {v['negative_false_accepts']}/{v['negative_n']} |")
        for family in obj['tests']['human']['gated']['per_intent']:
            ep+=['',f'## {family}','','| Speech | Valid commands | Exact accepted | Exact recall | Output precision |','|---|---:|---:|---:|---:|']
            for source,value in r['tests'].items():
                v=value['vad_endpoint_runtime']['per_intent'][family];ep.append(f"| {source} | {v['positive_n']} | {v['exact_commands']} | {pct(v['exact_command_recall'])} | {pct(v['output_precision'])} |")
        ep+=['','All original-file single-recording decisions matched the batched evaluator. Endpoint exact correctness requires one correct command and no extra accepted outputs. UNKNOWN recall is undefined; see the raw JSON for false accepts.','',
             'The software wake-boundary fixture passed; it does not establish wake-word recognition. Neither live microphone nor Pi results are claimed. See [the full comparison](../../mixed/keyword-v4/COMPARISON.md).']
        write(folder+'ENDPOINTS.md',ep)
        v=obj['tests']['human']['gated']['overall']
        outcome=[f'# Latest {mode} study outcome','',f"Human original-file exact recall: **{pct(v['exact_recall'])}**, accepted output precision: **{pct(v['precision'])}**. The all-intent and 0–100 acoustic quality target is not met.",'',
            '[Per-intent original-file results](REPORT.md) · [Per-intent endpoint results](ENDPOINTS.md) · [Comparison and source study](../../mixed/keyword-v4/COMPARISON.md)','',
            'This research version preserves the prior experiment and installed defaults. Models, checkpoints and predictions remain under `.cache/`; frozen manifests are under `data/keyword-v4/`.']
        write(folder+'OUTCOME.md',outcome)
        write(f'experiments/{mode}/LATEST.md',[f'# Latest {mode} experiment','',
              '[Read the keyword-v4 outcome](keyword-v4/OUTCOME.md). The original top-level report is preserved as historical v1 evidence.'])


if __name__=='__main__':main()
