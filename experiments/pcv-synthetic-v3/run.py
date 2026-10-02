"""Thin stages using the original model/trainer/exporter under a v3 context."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_pcv_v3_workflow_cli',HERE/'workflow.py')
wf=importlib.util.module_from_spec(spec);spec.loader.exec_module(wf)
STAGES=('freeze','audit','prepare','train','export','parity','calibrate','wake-dev','wake-calibration','evaluate','wake-evaluate','report')


def stage(name):
    cache=wf.CACHE
    if name=='freeze':return wf.freeze()
    if name=='audit':
        result=wf.audit_all_splits();wf.immutable_json(cache/'assembly-audit.json',result);return result
    wf.validate()
    with wf.bind():
        import fitting,finishing,synthbench,wake_benchmark,final_report
        bundle=cache/'calibration/bundle'
        if name=='prepare':return fitting.prepare()
        if name=='train':
            wf.validate_budget(initial=True)
            return fitting.train(cache/'training')
        if name=='export':return finishing.export(cache/'training',cache/'bundle-uncalibrated')
        if name=='parity':return finishing.waveform_parity(cache/'training',cache/'bundle-uncalibrated',cache/'waveform-parity')
        if name=='calibrate':return finishing.calibrate(cache/'bundle-uncalibrated',cache/'calibration',cache/'waveform-parity/results.json')
        if name=='evaluate':
            synthbench.load_runtime(bundle);wf.reserve_ancestors('command',bundle,cache/'final-test')
            return synthbench.evaluate(bundle,'test',cache/'final-test')
        if name in ('wake-dev','wake-calibration','wake-evaluate'):
            split={'wake-dev':'dev','wake-calibration':'calibration','wake-evaluate':'test'}[name]
            output=cache/('wake-final-test' if split=='test' else f'wake-{split}')
            if split=='test':synthbench.load_runtime(bundle);wf.reserve_ancestors('wake',bundle,output)
            return wake_benchmark.evaluate(bundle,output,split=split)
        if name=='report':return final_report.report(cache/'report')
        raise ValueError('Unknown stage')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=(*STAGES,'status'));args=parser.parse_args()
    if args.stage=='status':
        result=dict(version='pcv-synthetic-v3',generation_complete=(wf.CACHE/'generation-complete.json').exists(),
            frozen=(HERE/'protocol.json').exists(),audited=(wf.CACHE/'assembly-audit.json').exists(),
            selected=(wf.CACHE/'training/selection.json').exists(),
            jobs=[json.loads(p.read_text()) for p in (wf.CACHE/'jobs').glob('*/status.json')])
    else:result=stage(args.stage)
    print(json.dumps(result,indent=2),flush=True)
