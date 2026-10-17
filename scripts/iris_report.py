"""Render the release card/report from frozen acoustic and software receipts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=Path('experiments/iris-release-v1')
HF='https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/evidence/'


def read(root,name,optional=False):
    p=root/HERE/name
    if optional and not p.exists():return {}
    return json.loads(p.read_text())


def fraction(correct,n):
    return f'{correct}/{n} ({100*correct/n:.1f}%)' if n else '0/0 (unmeasured)'


def markdown_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join('---' for _ in headers)+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def render(root=ROOT):
    root=Path(root)
    recognition=read(root,'recognition-results.json')
    wake=read(root,'wake-results.json');owner=read(root,'owner-results.json')
    owner_export=read(root,'owner-export.json',True)
    software=read(root,'software-verification.json',True)
    resources=read(root,'pipeline-resources.json',True)
    choice=recognition['promotion'];results=recognition['release_results'];official=recognition['release_official']
    baseline=recognition['baseline'];candidate=recognition['results']
    selected=read(root,'development-selection.json')
    width=choice['width'];model_name=choice['selected']+' '+width
    def score(m):
        x=m['accepted']['command_level'];return fraction(round(x['accuracy']*x['n']),x['n'])
    rows=[]
    for title,records,official_record in [('V3 baseline',baseline['project'],baseline['official']),
           ('Diversity candidate',candidate,recognition['official']),('Selected release',results,official)]:
        human=records['test']['strata']['human']
        personal=(baseline['personal'] if title=='V3 baseline' else records['personal-test'])['overall']
        rows.append([title,fraction(personal['correct_accepts'],personal['positive_n']),
            personal['wrong_accepted_positives'],f"{personal['unknown_accepts']}/{personal['negative_n']}",
            fraction(human['correct_accepts'],human['positive_n']),f"{human['unknown_accepts']}/{human['negative_n']}",score(official_record)])
    comparison=markdown_table(['Model','Personal correct accepts','Personal wrong accepts','Personal negative accepts','Human correct accepts','Human UNKNOWN accepts','196-row accepted exact'],rows)
    voice=results['voice-test']['overall'];human=results['test']['strata']['human'];personal=results['personal-test']['overall']
    candidate_voice=candidate['voice-test']['overall']
    raw=official['raw_diagnostic']['command_level'];accepted=official['accepted']['command_level']
    guards='; '.join(f"{k}: {'pass' if v else 'FAIL'}" for k,v in choice['guards'].items())
    test_wake=wake['test']
    owner_domains=[('Original-human proxies',owner['test']),
        *sorted(owner.get('test_domains',{}).items()),
        ('Synthetic wake proxies',owner.get('synthetic_wake_command_proxy',{}))]
    def owner_error(metrics,key):
        value=metrics.get(key,{})
        return fraction(value.get('errors',0),value.get('denominator',0))
    owner_table=markdown_table(['Speaker trial population','Same-speaker rejection','Impostor acceptance',
        'Speaker-switch acceptance','Overlap missed','Clean speech flagged overlap'],[
        [name,*[owner_error(metrics,key) for key in ('same_speaker_FRR','impostor_FAR','switch_FAR',
            'overlap_miss_rate','clean_false_overlap')]] for name,metrics in owner_domains])
    owner_failed=', '.join(key for key,value in owner.get('gates',{}).items() if not value)
    session=owner_export.get('runtime_policy_proxies',{}).get('original_human_short_speech_proxy',{})
    session_note=('Actual ONNX plus the full session policy on short original-human proxies rejected '
        f"{owner_error(session,'same_speaker_FRR')} same-speaker trials, accepted "
        f"{owner_error(session,'impostor_FAR')} impostors and {owner_error(session,'switch_FAR')} switches, "
        f"and accepted {owner_error(session,'overlap_FAR')} constructed overlaps. "
        'These results differ from raw embedding/head metrics because speech-duration, continuity and '
        'overlap requirements also reject. The wake reference requires at least 0.8 seconds of detected '
        'voiced speech; short Hey Iris phrases can fail before command recognition. Two older synthetic '
        'wake references both failed this minimum. These are proxy tests, not original human wake pairs.') if session else ''
    wake_duration=read(root,'owner-wake-duration-diagnostic.json',True)
    if wake_duration:
        duration=wake_duration['all']
        session_note+=(' On the new four-voice held-out set, only '
            f"{duration['arm_accepted']}/{duration['references']} full standalone synthetic wake clips armed the owner gate; "
            f"{duration['rejection_reasons'].get('insufficient_speech',0)} were too short. "
            'This is a substantial usability limitation. Standalone full clips are not live detector windows, '
            'so no end-to-end human wake acceptance is inferred.')
    cost='Resource measurement is pending; no Pi cost is inferred.'
    if resources:
        timing=resources['measurements'];sizes=resources['sizes']
        cost=markdown_table(['Recorded host cost (one CPU thread)','Value'],[
            ['All packaged graph/metadata variants',f"{sizes['all_packaged_model_asset_bytes']/1048576:.2f} MiB"],
            ['Five model sessions: combined load',f"{resources['total_session_load']['wall_ms']:.1f} ms"],
            ['Python process peak RSS',f"{resources['memory']['peak_process_rss_mib']:.1f} MiB"],
            ['Final VCM call p50 / p95',f"{timing['component_latency_ms']['vcm_final']['wall_ms']['p50']:.1f} / {timing['component_latency_ms']['vcm_final']['wall_ms']['p95']:.1f} ms"],
            ['Wake, previews, final VCM, VAD and speaker checks: serial compute budget p50 / p95',f"{timing['complete_compute_budget_ms']['wall_ms']['p50']:.1f} / {timing['complete_compute_budget_ms']['wall_ms']['p95']:.1f} ms"],
        ])+'\nEight fixed synthetic wake/command pairs were profiled. The complete serial budget includes every half-second prefix, endpoint-silence computation and continuity checks; it is not live user latency. Audio elapsed time, endpoint waiting, file decoding, worker IPC, Node/browser and provider calls are excluded. Recorded host/settings and individual component timings remain in the resource receipt. UI/process integration costs are reported separately; physical Pi cost is unmeasured.'
    evidence='\n'.join(f'- [{label}]({HF+name})' for label,name in [
        ('Command outcomes, per-intent/value/role/source/speaker denominators','recognition-results.json'),
        ('Promotion decision and failed guards','release-choice.json'),
        ('Weight/policy/catalog lock before test','selection-before-test.json'),
        ('Matched training commands, seeds and optimizer snapshots','training-execution.json'),
        ('Preserved pre-fit source and bounded compatibility checks','reproduction-prefit-compatibility.json'),
        ('Actual Torch/ONNX/ncnn command decision parity','decision-parity.json'),
        ('Independent speaker metrics','owner-results.json'),('Runtime owner/session proxy results','owner-export.json'),
        ('Wake recall/duplicates/negative exposure','wake-results.json'),('Frozen schedules and input hashes','protocol.json'),
        ('Verified software checks','software-verification.json'),('Full runtime resource accounting','pipeline-resources.json'),
        ('Compiled product and resident-worker integration','ui-final-integration.json'),
        ('Extracted-package offline replay under both engines','package-replay.json'),
        ('Independent empty-cache reconstruction','reproduction-fresh-clone-check.json'),
        ('Independent LibriSpeech download and preparation','reproduction-owner-fresh-preparation.json')])
    card=f'''# Iris finite voice-command model

Selected model: **{model_name}**. Parameters: **{selected['parameters']:,}**. The release {'promotes the new diversity candidate' if choice['promoted'] else 'retains the predeclared V3 baseline because the new candidate failed promotion'}. This is an experimental software release; strict acoustic qualification remains failed/unresolved.

Iris maps a whole mono 16 kHz utterance directly to one of 328 finite outputs (UNKNOWN included). The reference keeps 31 baseline intent-and-slot outcomes, all 93 wordings and requested personal commands. Untrained outputs are blocked. Configuration binds recognized IDs to supplied entities; it cannot teach a new spoken name. General questions, unrestricted reminder text and arbitrary new names/values are unsupported.

## Measured command-only results

{comparison}
Fresh designed-voice holdout: **{fraction(voice['correct_accepts'],voice['positive_n'])}** correct accepts, {voice['wrong_accepted_positives']} wrong accepted positives and {voice['unknown_accepts']}/{voice['negative_n']} negative accepts. These are designed synthetic voices, not verified human demographic coverage. The unchanged personal cohort has **zero original-human positive recordings**.

The unpromoted diversity candidate reached **{fraction(candidate_voice['correct_accepts'],candidate_voice['positive_n'])}** on the same fresh voice holdout, with {candidate_voice['wrong_accepted_positives']} wrong accepted positives and {candidate_voice['unknown_accepts']}/{candidate_voice['negative_n']} negative accepts. Its improved positive acceptance does not waive the failed human UNKNOWN guard.

The unchanged 196-row scorer gives raw exact **{fraction(round(raw['accuracy']*raw['n']),raw['n'])}**, accepted exact **{fraction(round(accepted['accuracy']*accepted['n']),accepted['n'])}**. Raw argmax is diagnostic and does not execute. Human regression precision is {100*human['precision']:.1f}% over {human['accepted']} accepts; its 185 ambiguous references are excluded from accuracy and retained in evidence. Promotion guards: {guards}. Test results only veto promotion; thresholds were not retuned after test.

## Architecture and training

Microphone → Hey Iris → command capture/Silero VAD → direct VCM plus independent speaker check → typed validation → configured executor → UI receipt. Recognition has no ASR/STT, phonetic/transcription decoder, LLM, remote model service or automatic model fallback. Offline synthesis is solely a training-data process and never ships as recognition.

The VCM uses a 40-bin log-mel frontend, stride-two convolution, six contextual blocks (four-head attention, temporal depthwise convolution and feedforward layers), masked mean pooling and one atomic command classifier. Compared widths are 128 (969,928 parameters) and 192 (2,118,280). The candidate recipe chosen on development is {selected['arm']}, seed {selected['seed']}. It continues our own pinned V2 project encoder/head; no external pretrained command encoder is used. If the candidate fails promotion, the selected runtime instead retains the unchanged same-width V2 weights with the predeclared V3 acceptance policy.

Four matched fits compare CE and CE + 0.1 cross-speaker contrastive loss (temperature 0.1) at both widths, seed 53. Both objectives repeat at the development-selected width, seed 67. The frozen bound is 16 epochs × 96 updates, batch 32/microbatch eight, AdamW 1e-4, cosine decay, weight decay .001, smoothing .02, clip norm five and declared early stopping. Batches balance human/synthetic source, command and speaker, include UNKNOWN examples, ±6 dB training-only gain and bounded time/frequency masks. Development chooses accepted human macro-speaker recall, then human/synthetic correct counts and compute cost. Development/calibration fit empirical class acceptance thresholds; these scores are not guarantees of correctness.

The independent random-init speaker encoder uses six temporal depthwise blocks, masked mean/std pooling, a 128-D embedding and overlap head. It compares one wake speaker to one following command, rejects mismatch/overlap/missing/short evidence and resets. It has **no persistent user enrollment** and does not separate overlapping speech. Command embeddings are never used as identity. Its data comprise 9,429 original-human recordings (24.264 hours) plus separately labelled constructed mixtures and synthetic calibration/test proxies. Original human wake-to-command and real captured overlap are unmeasured.

{owner_table}
Failed or unmeasured speaker gates: **{owner_failed}**. In particular, the 5% impostor-acceptance and 20% overlap-miss proxy limits fail. This experimental gate is not identity authentication. Thresholds remain locked despite those failures.

{session_note}

The own Hey Iris network has 82,114 parameters. The expanded and previous wake candidates are compared on calibration only. Selected: {wake['selected']}. Test: `{json.dumps(test_wake,sort_keys=True)}`. Zero-event false-wake upper 95% bound, when applicable: {wake.get('zero_event_false_wakes_per_hour_upper_95')} per hour. No original-human wake positives exist; synthetic recall and short negative exposure do not establish live wake reliability.

## Use, actions and reproducibility

Download the verified Pi ZIP and hashes from this repository. Extract to a new folder; run `python3 start.py --list-mics`, configure your Home Assistant entities/scripts and private token, then `python3 start.py --mode serve --mic INDEX --benchmarks skip`. First dependency setup needs internet; acoustic inference is local thereafter. WAV replay and benchmarks always remain non-executing. Tentative intent/slots never execute. Only a final accepted command with speaker evidence and valid configured bindings can execute. Receipts distinguish service acceptance from observed device state. Jobs and execution IDs persist atomically; unknown outcomes never retry automatically.

From source: `python train.py` verifies/prepares inputs, performs the frozen fits, calibrates/exports/evaluates, builds the UI, packages and writes reports. `python train.py --skip-train` verifies published project artifacts and reconstructs the delivery; `--publish` uploads verified results. Project initializers are an immutable shortcut. `--from-scratch` declares a separate random-init study and does not reproduce identical historical V2 weights. Exact historical executable lineage/protocols remain in the immutable research archive; the current comparison's optimizer/RNG snapshots and input hashes are preserved. Full random-init 48-epoch reconstruction is not claimed measured by a tiny resume test.

The public [source snapshot](https://huggingface.co/justinruaya123/ai231-smart-speaker/tree/main/reproduction/source) includes the default reconstruction receipts. Download `reproduction/source/**` with the Hugging Face client, enter that directory and run `python train.py --skip-train`. The workflow resolves inputs at their published immutable revisions; this machine's cache paths are not required.

Original human data, supplied synthetic data, designed voices and ambiguous/unsupported cases are reported separately. Restricted primary/supplemental recordings are referenced at original immutable upstream revisions, not newly uploaded. LibriSpeech is CC BY 4.0; Qwen synthesis model source is Apache 2.0; other component/data terms remain component-specific and research-limited. See the dataset card and NOTICES for restrictions.

Physical Pi latency/RAM/thermals, fresh original-human wake/command qualification, live microphone quality and observed real Home Assistant device outcomes remain **unmeasured operator acceptance steps**. The software-verification receipt lists the executed checks and results. Desktop measurements do not establish Pi performance.

## Resource measurements

{cost}

## Evidence and methods

{evidence}

The training-only loss adapts [Khosla et al., Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html); masks adapt [Park et al., SpecAugment](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html). These are adaptations, not reproductions of their paper scores. [LibriSpeech](https://www.openslr.org/12/) supplies audio and speaker labels only. The [Qwen model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) describes the offline generator; requested voice characteristics are not independently verified.
'''
    (root/'MODEL_CARD.md').write_text(card)
    fits=markdown_table(['Width','Objective','Seed','Selected epoch','Human macro-speaker accepted recall','Human correct','Synthetic correct','Training seconds'],[
        [r['model'],r['arm'],r['seed'],r['epoch'],f"{r['key'][0]:.4f}",r['key'][1],r['key'][2],f"{r['training_seconds']:.1f}"] for r in selected['all_runs']])
    per_intent=markdown_table(['Intent','Human raw','Human correct accepts','Human wrong accepts','Synthetic correct accepts'],[
        [intent,fraction(v['human']['raw_correct'],v['human']['positive_n']),fraction(v['human']['correct_accepts'],v['human']['positive_n']),v['human']['wrong_accepted_positives'],fraction(v['synthetic']['correct_accepts'],v['synthetic']['positive_n'])]
        for intent,v in sorted(results['test']['per_intent'].items())])
    report='''# Iris release v1: final experiment and product evidence

This new version addresses ME-01–08 and UC-01–10. The hypothesis was improved accepted exact commands across held-out voices within frozen development limits. Data, loss, schedule, catalog and selection were frozen before fitting. Existing test populations, recordings, protocols and earlier results were preserved. The isolated release branch archived historical research before named cleanup; original worktrees and shared recordings remain untouched.

## Matched fits and decision

'''+fits+'\n'+comparison+'\nPromotion decision: `'+json.dumps(choice,sort_keys=True)+'`.\n\n## Selected per-intent regression\n\n'+per_intent+'''
All slot values/roles, sources, requested accent conditions and individual speakers/voice IDs retain denominators in recognition-results.json. Full-command correctness, per-role correctness, raw predictions and accepted decisions remain distinct. Missing acoustic coverage is unsupported; entity configuration is not new training.

## Architecture, method, datasets and limits

'''+card+'''
## Implementation and verification findings

An independent review found and reproduced stale unrelated package selection, catch-up preview bursts and an inherited indefinite-capture edge. Failing regressions preceded fixes: current release path/hash matching, monotonic 2 Hz preview scheduling, and bounded capture deadline/reset. Owner/VCM acceptance is an intersection; a later application policy cannot rescue an ownership rejection. Plain WAV previews explicitly declare the action-free benchmark profile.

Final review also reproduced truncated test-cache reuse, alternate-release verification against the wrong model paths, and parity source-root misattribution. Replay caches now publish hash-bound completion receipts with complete ordered sample IDs; the official wrapper also binds its unchanged scorer's variation/slot CSV. Unsupported alternate reconstruction paths and shadow source roots reject. Exact preserved-source AST checks permit completed-fit reuse across the evaluation-only patch and redundant metadata-keyword correction while rejecting changes to loss, ranking, data or acceptance policy. The training execution receipt preserves the scratch wrapper's final-log collision and recovered metadata serialization failure; all six fits completed without rerunning or altering their weights.

The final delivery audit reproduced a fresh-start configuration failure: the generated example selected Home Assistant media playback without its required entity ID. Source and package examples now leave that provider unconfigured. A regression runs the actual first-start configuration exporter and gateway listeners with untouched templates. The final packaged UI check uses those defaults without an empty-bindings override; supplied entities remain an operator configuration step. Earlier package/proof bytes are preserved separately.

The independent source reconstruction also exposed a build-order defect: the Python package-inventory test required compiled UI files before the workflow had built them. UI installation, tests, checks and compilation now precede Python package tests. After a successful cold UI build, the retained copied run `fresh-orderfix` exposed an omitted root `benchmark.py` launcher in the reconstruction inventory. A trace of all nine application target/benchmark modes and an independent import/UI/bootstrap audit found no further required source omissions. Regressions start without compiled UI output, verify the build boundary and check every application source read against the published inventory. The supported publisher also includes card-linked late evidence and reconstruction copies as explicit hash-verified plan items; missing linked evidence rejects before upload, without binding a proof into the release it measures. Failed copied runs, previously completed acoustic evaluation/parity results and independently acquired data remain preserved.

A second GPU experiment was attempted under the five-second guard and stopped before GPU allocation when GPU4 rose by 20 utilization points. The primary synthesis continued on GPU5; subsequent fits used one GPU sequentially. Other users' jobs were preserved. Software/UI/API fixtures establish behavior, not fresh human or device quality. No threshold changes follow held-out test failures.

Source and package reconstruction use pinned inputs, hashes, source/environment receipts and exact optimizer/RNG checkpoints. The separate random-init route has a genuine interrupted/resumed CPU fit check, but its full 48-epoch study was not run in this release. Historical initializer lineage is retained on the archive; published initializers are the verified shortcut for the bounded matched experiment.

See [dataset audits](DATASETS.md), [Pi handoff and operator acceptance](HANDOFF-iris-rpi-v1.md), [archive lifecycle/recovery](DOCS.md), and [current machine receipts](../experiments/iris-release-v1/). The actual final graphs, package and publication revisions are pinned by the reproduction/publication receipts. No physical Pi, live microphone or external HA device test is claimed.
'''
    (root/'docs/EXPERIMENT-iris-release-v1.md').write_text(report)
    return {'model':model_name,'promoted':choice['promoted'],'strict_qualification':False,
            'documents':['MODEL_CARD.md','docs/EXPERIMENT-iris-release-v1.md']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(render(p.parse_args().root)))

if __name__=='__main__':main()
