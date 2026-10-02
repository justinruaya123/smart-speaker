"""Build, report and package the measured synthetic VCM without changing training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
import hashlib
import json
import shutil
import zipfile
from pcvcm.contract import canonical,file_hash,identity,write_json
from scripts.runtime.pcv_deploy.bundle import ASSETS,WAKE_FILES,EVIDENCE,verify_bundle


def report(study_cache,output):
    cache=Path(study_cache);output=Path(output)
    if output.exists():raise FileExistsError('Delivery reports are immutable')
    protocol=json.loads((ROOT/'experiments/pcv-delivery-v1/protocol.json').read_text())
    demo_path=ROOT/'experiments/pcv-delivery-v1/demo.json'
    if file_hash(demo_path)!=protocol['demo_sha256']:raise ValueError('Frozen demo changed')
    demo=json.loads(demo_path.read_text())['commands']
    source_results=json.loads((cache/'final-test/results.json').read_text())
    predictions=cache/'final-test/predictions.jsonl'
    if not source_results['complete'] or file_hash(predictions)!=source_results['predictions_sha256']:raise ValueError('Incomplete or changed study result')
    rows=[json.loads(line) for line in predictions.read_text().splitlines()]
    recordings={}
    for engine in ('qwen','parler','dia'):
        for line in (cache/f'{engine}-manifest.jsonl').read_text(encoding='utf-8').splitlines():
            item=json.loads(line);recordings[item['recording_id']]=item
    for row in rows:
        recording=recordings[row['recording_id']]
        if recording['audio_sha256']!=row['audio_sha256']:raise ValueError('Prediction audio identity differs from manifest')
        row['audio']=str((cache/recording['audio']).resolve())
    refs={r['id']:r for r in [json.loads(line) for line in (ROOT/'experiments/pcv-synthetic-v2/verbatims.jsonl').read_text().splitlines()]}
    demo_rows=[];mistakes=[]
    for reference in demo:
        trials=[r for r in rows if r['verbatim_id']==reference['id']]
        correct=[r for r in trials if r['output']['accepted'] and canonical(r['output']['command'])==canonical(reference['expected_command'])]
        coverage={engine:len({r['voice_group'] for r in trials if r['engine']==engine}) for engine in ('qwen','parler','dia')}
        demo_rows.append(dict(reference=reference,trials=len(trials),correct=len(correct),coverage=coverage,
            passed=len(trials)>0 and len(correct)==len(trials) and coverage['qwen']>=2 and coverage['parler']>=2,outputs=trials))
    for row in rows:
        expected=row['expected_command'];actual=row['output'].get('command');candidate=row.get('candidate')
        correct=(row['output']['accepted'] and canonical(actual)==canonical(expected)) if expected is not None else not row['output']['accepted']
        if not correct or row.get('capture_fault') or not row.get('endpoint_detected',True):
            mistakes.append(dict(verbatim=refs[row['verbatim_id']]['text'],audio=row['audio'],parsed_intent=(candidate or {}).get('intent'),
                expected=expected,raw_candidate=candidate,accepted_output=actual,reason=row['output'].get('reason'),record=row))
    result=dict(binding=source_results['binding'],commands=len(demo_rows),use_cases=sorted({r['reference']['use_case'] for r in demo_rows}),
        demo_gates_passed=all(r['passed'] for r in demo_rows) and len(demo_rows)==protocol['demo_commands'],
        protocol_sha256=file_hash(ROOT/'experiments/pcv-delivery-v1/protocol.json'),predictions_sha256=file_hash(predictions),
        synthetic_only=True,physical_pi_measured=False,human_qualified=False,commands_with_results=demo_rows)
    output.mkdir(parents=True)
    write_json(output/'demo-results.json',result)
    (output/'mistakes.jsonl').write_text(''.join(canonical(r)+'\n' for r in mistakes),encoding='utf-8')
    write_json(output/'summary.json',dict(binding=result['binding'],demo_gates_passed=result['demo_gates_passed'],
        mistakes=len(mistakes),mistakes_sha256=file_hash(output/'mistakes.jsonl'),demo_results_sha256=file_hash(output/'demo-results.json')))
    return dict(demo_commands=len(demo_rows),demo_gates_passed=result['demo_gates_passed'],mistakes=len(mistakes))


def portable_bundle(source,output,*,study_cache,delivery_report,diagnostic=False):
    source=Path(source);output=Path(output);cache=Path(study_cache)
    if output.exists():raise FileExistsError('Portable bundles are immutable')
    from scripts.pcv_study_context import bind_study
    with bind_study(source) as selected:
        if cache.resolve()!=selected.CACHE.resolve():raise ValueError('Study cache belongs to another candidate version')
        import synthbench
        runtime=synthbench.load_runtime(source)
    original=json.loads((source/'synthetic-bundle.json').read_text())
    if diagnostic:
        if runtime.policy['enabled']:raise ValueError('Diagnostic conversion cannot disable an enabled candidate silently')
    else:
        final=json.loads((cache/'report/finalization.json').read_text())
        demo=json.loads((Path(delivery_report)/'demo-results.json').read_text())
        if not final['synthetic_qualified'] or not demo['demo_gates_passed'] or final['binding']!=runtime.binding or demo['binding']!=runtime.binding:
            raise ValueError('Broad, wake and declared demo acceptance gates must pass before publication')
        if file_hash(cache/'final-test/predictions.jsonl')!=demo['predictions_sha256']:raise ValueError('Demo evidence changed')
    output.mkdir(parents=True);(output/'wake').mkdir()
    for name in ('model.onnx','catalog.json','normalization.json','endpoint.json'):shutil.copyfile(source/name,output/name)
    for name in WAKE_FILES:shutil.copyfile(ROOT/'artifacts'/name,output/'wake'/name)
    write_json(output/'settings.json',dict(sample_rate=16000,threads=1,wake_threshold=.5,release_ms=320,
        release_timeout_seconds=3.,command_timeout_seconds=5.))
    files={name:file_hash(output/name) for name in ASSETS};binding=identity(files)
    write_json(output/'policy.json',dict(binding=binding,enabled=runtime.policy['enabled'],thresholds=runtime.policy['thresholds'],
        source_binding=runtime.binding,source_policy_sha256=file_hash(source/'policy.json')))
    write_json(output/'deployment.json',dict(schema_version=1,files=files,binding=binding,policy_sha256=file_hash(output/'policy.json'),
        status='DIAGNOSTIC_UNQUALIFIED' if diagnostic else 'SYNTHETIC_VALIDATED',source_binding=original['binding'],
        source_manifest_sha256=file_hash(source/'synthetic-bundle.json'),human_qualified=False,physical_pi_measured=False))
    if not diagnostic:
        for name,path in [('command-results.json',cache/'final-test/results.json'),('wake-results.json',cache/'wake-final-test/results.json'),
                          ('demo-results.json',Path(delivery_report)/'demo-results.json'),('source-manifest.json',source/'synthetic-bundle.json'),
                          ('source-protocol.json',source/'protocol.json'),('calibration-selection.json',cache/'calibration/selection.json'),
                          ('source-identity.json',source/'source-identity.json'),
                          ('source-policy.json',source/'policy.json'),('source-finalization.json',cache/'report/finalization.json')]:shutil.copyfile(path,output/name)
        manifest=json.loads((output/'deployment.json').read_text())
        manifest['evidence_files']={name:file_hash(output/name) for name in EVIDENCE}
        write_json(output/'deployment.json',manifest)
    verify_bundle(output,allow_diagnostic=diagnostic)
    return dict(path=str(output),binding=binding,synthetic_only=True)


def package(bundle,output,*,diagnostic=False,portable_parity=None,sample_manifest=None,
            source_receipt=None,source_card=None,qualification_receipt=None,extra_files=None):
    bundle=Path(bundle);output=Path(output)
    if (bundle/'recipe.json').exists():
        recipe=json.loads((bundle/'recipe.json').read_text(encoding='utf-8'))
        if recipe.get('inference_mode')=='complete_utterance':
            from scripts.pcv_final_delivery import package as endpoint_package
            return endpoint_package(bundle,output,diagnostic=diagnostic,portable_parity=portable_parity,
                sample_manifest=sample_manifest,source_receipt=source_receipt,source_card=source_card,
                qualification_receipt=qualification_receipt,extra_files=extra_files)
    if any(value is not None for value in (sample_manifest,source_receipt,source_card,qualification_receipt,extra_files)):
        raise ValueError('Endpoint sample/evidence options require a complete_utterance recipe')
    if output.exists():raise FileExistsError('ZIP outputs are immutable')
    content=verify_bundle(bundle,allow_diagnostic=diagnostic)
    from pcvcm.dependencies import verify_lock
    lock=ROOT/'experiments/phrase-contract-v1/configs/runtime-cp312-aarch64'
    wheels=ROOT/'.cache/phrase-contract-v1/dependencies/cp312-aarch64'
    dependencies=verify_lock(lock,wheels)
    entries={}
    for name in ('__init__.py','contract.py','runtime.py','decision.py','decoder.py','endpoint.py'):
        entries['pcvcm/'+name]=(ROOT/'experiments/phrase-contract-v1/pcvcm'/name).read_bytes()
    entries['vcm/__init__.py']=b'"""Streaming frontend only; no training dependencies."""\n'
    entries['vcm/frontend.py']=(ROOT/'vcm/frontend.py').read_bytes()
    entries['wake_gate.py']=(ROOT/'experiments/pcv-synthetic-v2/wake_gate.py').read_bytes()
    for name in ('scripts/__init__.py','scripts/runtime/__init__.py'):entries[name]=b''
    entries['scripts/runtime/wake_models.py']=(ROOT/'scripts/runtime/wake_models.py').read_bytes()
    if not content['diagnostic']:
        protocol=json.loads((bundle/'source-protocol.json').read_text())
        for name,payload in entries.items():
            original=('experiments/phrase-contract-v1/'+name if name.startswith('pcvcm/') else
                      'experiments/pcv-synthetic-v2/wake_gate.py' if name=='wake_gate.py' else name)
            if name in ('vcm/__init__.py','scripts/__init__.py','scripts/runtime/__init__.py'):continue
            expected=protocol['shared_sources'].get(original.replace('/','\\'),protocol['shared_sources'].get(original))
            # Child studies inherit the identical parent wake implementation.
            if name=='wake_gate.py':
                sources=json.loads((bundle/'source-identity.json').read_text())
                expected=sources.get(original.replace('/','\\'),sources.get(original,expected))
            if not expected or hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Recognition source differs from measured run: '+original)
    for path in (ROOT/'scripts/runtime/pcv_deploy').glob('*.py'):entries['scripts/runtime/pcv_deploy/'+path.name]=path.read_bytes()
    if not content['diagnostic']:
        from scripts.pcv_portable_parity import verify_receipt
        verify_receipt(portable_parity,content)
        entries['portable-parity.json']=Path(portable_parity).read_bytes()
    for name in (*ASSETS,'policy.json','deployment.json'):
        entries['models/default/'+name]=(bundle/name).read_bytes()
    for name in EVIDENCE:
        if (bundle/name).exists():entries['models/default/'+name]=(bundle/name).read_bytes()
    for name in ('requirements.txt','THIRD_PARTY_NOTICES.txt','dependencies.json'):entries[name]=(lock/name).read_bytes()
    for wheel in dependencies['wheels'].values():entries['wheels/'+wheel['filename']]=(wheels/wheel['filename']).read_bytes()
    for name in ('openwakeword-LICENSE.txt','silero-LICENSE.txt','pretrained-NOTICE.txt'):entries['licenses/'+name]=(ROOT/'artifacts'/name).read_bytes()
    entries['licenses/CMUDICT-LICENSE.txt']=(ROOT/'experiments/phrase-contract-v1/configs/CMUDICT-LICENSE.txt').read_bytes()
    for name in ('PCV-MODEL.md','PCV-TOOL-SERVER.md','PCV-DEMO-COMMANDS.md'):
        if not content['diagnostic'] and not (ROOT/'docs'/name).exists():raise ValueError('Missing required delivery documentation: '+name)
        if (ROOT/'docs'/name).exists():entries['docs/'+name]=(ROOT/'docs'/name).read_bytes()
    diagnostic_flag=' --diagnostic' if content['diagnostic'] else ''
    entries['verify_package.py']=b'''import hashlib,json
from pathlib import Path,PurePosixPath
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for name,expected in manifest['files'].items():
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\\\' in name or ':' in name:raise ValueError('Unsafe package path')
    target=(root/name).resolve()
    if not target.is_relative_to(root):raise ValueError('Package path escapes root')
    with target.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected:raise ValueError('Package integrity mismatch: '+name)
print('Package identity verified')
'''
    entries['setup.sh']=('''#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
test "$(uname -m)" = aarch64 || { echo 'Requires 64-bit ARM64 Raspberry Pi OS.' >&2; exit 1; }
command -v python3.12 >/dev/null || { echo 'Install CPython 3.12 and OS libsndfile1/libportaudio2 first; see docs.' >&2; exit 1; }
python3.12 verify_package.py
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheels --only-binary=:all: --require-hashes -r requirements.txt
.venv/bin/python -m scripts.runtime.pcv_deploy check --bundle models/default'''+diagnostic_flag+'\n').encode()
    entries['serve.sh']=('''#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
.venv/bin/python verify_package.py
exec .venv/bin/python -m scripts.runtime.pcv_deploy serve --bundle models/default'''+diagnostic_flag+' "$@"\n').encode()
    entries['check.py']=b'from scripts.runtime.pcv_deploy.__main__ import main\nmain()\n'
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    entries['MANIFEST.json']=(canonical(dict(binding=content['binding'],files=hashes,synthetic_only=True,
        diagnostic=content['diagnostic'],human_qualified=False,physical_pi_measured=False))+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(entries.items()):
            from scripts.runtime.pcv_deploy.bundle import below
            below(output.parent,name)
            info=zipfile.ZipInfo(name,(2026,10,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if name.endswith('.sh') else 0o100644)<<16
            archive.writestr(info,data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
    receipt=dict(path=str(output),sha256=file_hash(output),bytes=output.stat().st_size,binding=content['binding'],
        diagnostic=content['diagnostic'],target=dependencies['target'],physical_pi_measured=False)
    write_json(output.with_suffix('.receipt.json'),receipt);return receipt


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['report','catalog','assemble-bundle','package','export'])
    parser.add_argument('--bundle',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--study-cache',type=Path,default=ROOT/'.cache/pcv-synthetic-v2')
    parser.add_argument('--delivery-report',type=Path,default=ROOT/'.cache/pcv-delivery-v1/report')
    parser.add_argument('--csv',type=Path);parser.add_argument('--lexicon',type=Path)
    parser.add_argument('--base-catalog',type=Path)
    parser.add_argument('--portable-parity',type=Path)
    parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--training',type=Path)
    args=parser.parse_args()
    if args.stage=='report':result=report(args.study_cache,args.output)
    elif args.stage=='catalog':
        from scripts.runtime.pcv_deploy.catalog import compile_csv,merge_inventory
        if not args.csv or not args.lexicon:parser.error('catalog requires --csv and --lexicon')
        if args.output.exists():raise FileExistsError('Catalog outputs are immutable')
        result=compile_csv(args.csv,json.loads(args.lexicon.read_text(encoding='utf-8'))['words'])
        if args.base_catalog:result=merge_inventory(json.loads(args.base_catalog.read_text(encoding='utf-8')),result)
        write_json(args.output,result)
    elif args.stage=='export':
        if not args.training:parser.error('export requires --training')
        from scripts.pcv_study_context import bind_study
        with bind_study(args.training):
            import finishing
            result=finishing.export(args.training,args.output)
    elif args.stage=='assemble-bundle':
        if not args.bundle:parser.error('--bundle required')
        result=portable_bundle(args.bundle,args.output,study_cache=args.study_cache,delivery_report=args.delivery_report,diagnostic=args.diagnostic)
    else:
        if not args.bundle:parser.error('--bundle required')
        result=package(args.bundle,args.output,diagnostic=args.diagnostic,portable_parity=args.portable_parity)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
