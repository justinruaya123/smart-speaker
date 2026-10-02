"""V2 entry point: shared recognition stages and mandatory wake-gated file input."""
import argparse
import json
from pathlib import Path
import revision as rev
import corpus
from version_context import bind


def status():
    plan=rev.read_rows(rev.HERE/'generation-plan.jsonl')+rev.read_rows(rev.HERE/'wake-generation-plan.jsonl')
    result={}
    for engine in ('qwen','parler','dia'):
        wanted=[r for r in plan if r['engine']==engine]
        old=rev.OLD_CACHE/f'{engine}-manifest.jsonl';new=rev.CACHE/f'generated-{engine}.jsonl'
        present={r['recording_id'] for r in rev.read_rows(old)} if old.exists() and engine!='dia' else set()
        if new.exists():present|={r['recording_id'] for r in rev.read_rows(new)}
        n=sum(r['recording_id'] in present for r in wanted)
        result[engine]=dict(generated=n,planned=len(wanted),remaining=len(wanted)-n)
    result.update(trained_candidate=(rev.CACHE/'training/selection.json').exists(),
        command_test_complete=(rev.CACHE/'final-test/results.json').exists(),wake_test_complete=(rev.CACHE/'wake-final-test/results.json').exists(),
        controller_transition_recorded=(rev.CACHE/'controller-transition-complete.json').exists())
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','status','generate','assemble','prepare','train',
        'export','parity','calibrate','evaluate','wake-dev','wake-calibration','wake-evaluate','report','recognize'])
    parser.add_argument('--engine',choices=['qwen','parler','dia']);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--wav',type=Path);args=parser.parse_args();cache=rev.CACHE
    if args.stage=='freeze':
        result=rev.freeze()
        from final_report import write_verbatim_guide
        result['verbatim_guide']=write_verbatim_guide()
    elif args.stage=='status':result=status()
    elif args.stage=='generate':
        if not args.engine:parser.error('--engine is required')
        result=corpus.generate(args.engine)
    elif args.stage=='assemble':result=corpus.assemble()
    else:
        rev.validate()
        with bind():
            import fitting,finishing,synthbench
            if args.stage=='prepare':result=fitting.prepare()
            elif args.stage=='train':result=fitting.train(cache/'training')
            elif args.stage=='export':result=finishing.export(cache/'training',cache/'bundle-uncalibrated')
            elif args.stage=='parity':result=finishing.waveform_parity(cache/'training',cache/'bundle-uncalibrated',cache/'waveform-parity')
            elif args.stage=='calibrate':result=finishing.calibrate(cache/'bundle-uncalibrated',cache/'calibration',cache/'waveform-parity/results.json')
            elif args.stage=='evaluate':result=synthbench.evaluate(cache/'calibration/bundle','test',cache/'final-test')
            elif args.stage in ('wake-dev','wake-calibration','wake-evaluate'):
                from wake_benchmark import evaluate
                split={'wake-dev':'dev','wake-calibration':'calibration','wake-evaluate':'test'}[args.stage]
                folder='wake-final-test' if split=='test' else f'wake-{split}'
                result=evaluate(cache/'calibration/bundle',cache/folder,split=split)
            elif args.stage=='report':
                from final_report import report
                result=report(cache/'report')
            else:
                if not args.bundle or not args.wav:parser.error('recognize requires --bundle and --wav')
                import soundfile as sf
                import numpy as np
                from wake_gate import from_bundle,replay_pcm
                gate=from_bundle(args.bundle,rev.ROOT/'artifacts');pcm,rate=sf.read(args.wav,dtype='float32')
                if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Requires finite mono 16 kHz WAV')
                events=replay_pcm(gate,pcm)
                result=dict(events=events,final=next((e for e in reversed(events) if e.get('stage')=='command'),gate.eof()),
                    wake_required=True,audio_sha256=rev.file_hash(args.wav),warmup_silence_seconds=2,timeout_flush_seconds=6)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
