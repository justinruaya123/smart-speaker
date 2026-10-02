"""Child-study stages; parent synthesis and failed preparation are preserved."""
import argparse
import json
from pathlib import Path
import workflow as repair


def reserve_parent_test(kind, bundle, output):
    """Also consume the parent's identical sealed set before child inference."""
    import os
    from revision import file_hash, identity
    parent = repair.rev.CACHE
    manifests = {e:file_hash(parent/f'{e}-manifest.jsonl') for e in repair.ENGINES}
    if kind == 'command':
        dataset = dict(plan=file_hash(repair.PARENT/'generation-plan.jsonl'), manifests=manifests)
        folder = 'test-access'
    else:
        dataset = dict(plan=file_hash(repair.PARENT/'wake-generation-plan.jsonl'),
            references=file_hash(repair.PARENT/'wake-verbatims.jsonl'),
            wake_manifests={e:file_hash(parent/f'wake-{e}-manifest.jsonl') for e in repair.ENGINES},
            command_manifests=manifests)
        folder = 'wake-test-access'
    destination = parent/folder/f'{identity(dataset)}.json'
    destination.parent.mkdir(exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(dict(dataset=dataset, child_protocol_sha256=file_hash(repair.HERE/'protocol.json'),
            bundle=str(bundle), output=str(output), state='RESERVED_PRESERVE_ON_FAILURE'), stream)
        stream.flush(); os.fsync(stream.fileno())
    return destination


def main():
    stages = ('freeze','status','generate-repair','assemble','prepare','train','export','parity',
        'calibrate','wake-dev','wake-calibration','evaluate','wake-evaluate','report')
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=stages)
    args = parser.parse_args(); cache = repair.CACHE
    if args.stage == 'freeze': result = repair.freeze()
    elif args.stage == 'status':
        result = dict(version='pcv-synthetic-v2-repair1',
            replacement_generated=(cache/'repair-generation.json').exists(),
            assembly_complete=(cache/'assembly-audit.json').exists(),
            prepared=(cache/'prepared/freeze.json').exists(),
            trained_candidate=(cache/'training/selection.json').exists(),
            command_test_complete=(cache/'final-test/results.json').exists(),
            wake_test_complete=(cache/'wake-final-test/results.json').exists())
    elif args.stage == 'generate-repair': result = repair.generate()
    elif args.stage == 'assemble': result = repair.assemble()
    else:
        repair.validate()
        with repair.bind():
            import fitting, finishing, synthbench, wake_benchmark, final_report
            bundle = cache/'calibration/bundle'
            if args.stage == 'prepare': result = fitting.prepare()
            elif args.stage == 'train': result = fitting.train(cache/'training')
            elif args.stage == 'export': result = finishing.export(cache/'training',cache/'bundle-uncalibrated')
            elif args.stage == 'parity': result = finishing.waveform_parity(cache/'training',cache/'bundle-uncalibrated',cache/'waveform-parity')
            elif args.stage == 'calibrate': result = finishing.calibrate(cache/'bundle-uncalibrated',cache/'calibration',cache/'waveform-parity/results.json')
            elif args.stage == 'evaluate':
                synthbench.load_runtime(bundle)
                reserve_parent_test('command',bundle,cache/'final-test')
                result = synthbench.evaluate(bundle,'test',cache/'final-test')
            elif args.stage in ('wake-dev','wake-calibration','wake-evaluate'):
                split = {'wake-dev':'dev','wake-calibration':'calibration','wake-evaluate':'test'}[args.stage]
                output = cache/('wake-final-test' if split=='test' else f'wake-{split}')
                if split == 'test':
                    synthbench.load_runtime(bundle); reserve_parent_test('wake',bundle,output)
                result = wake_benchmark.evaluate(bundle,output,split=split)
            else: result = final_report.report(cache/'report')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
