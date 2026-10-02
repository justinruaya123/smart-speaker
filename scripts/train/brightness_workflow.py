"""Small orchestration surface for preparing, comparing and installing STT."""
import copy
import json
from pathlib import Path
import shutil
from .brightness_data import prepare, fetch_context, sha, write_json


def install(root,config,candidate=None):
    root=Path(root);out=root/'.cache/stt-human';candidate=Path(candidate or out/'candidate')
    report=json.loads((out/'benchmark.json').read_text())
    selection=json.loads((out/'selection_locked_before_test.json').read_text())
    metadata=json.loads((candidate/'bundle.json').read_text())
    if metadata['stt']['training'].get('limited'):
        raise ValueError('Limited debug candidates cannot replace artifacts')
    current=json.loads((root/'artifacts/bundle.json').read_text())
    for name in ('vcm.fp32.onnx',):
        if sha(candidate/name)!=sha(root/'artifacts'/name):raise ValueError('Frozen baseline changed; rebuild comparison')
    if report['manifest_sha256']!=sha(root/'data/stt-human/brightness.jsonl'):raise ValueError('Dataset changed after benchmark')
    for name in ('stt',):
        if report['models'][name]['sha256']!=sha(candidate/f'{name}.fp32.onnx') or selection['model_sha256'][name]!=report['models'][name]['sha256']:
            raise ValueError('Model changed after selection/benchmark')
        if metadata['brightness_parameters'][name]!=report['policies'][name] or selection['policies'][name]!=report['policies'][name]:
            raise ValueError('Acceptance policy changed after selection/benchmark')
    for name in ('stt.pt','stt.fp32.onnx'):
        if metadata['files'][name]['sha256']!=sha(candidate/name):raise ValueError('Candidate integrity failed: '+name)
    if not report['policies']['stt']['development_gate_passed']:
        raise ValueError('STT development precision gate failed; retain candidate without replacing artifacts')
    from .validation import run_checks
    checks_config=copy.deepcopy(config);checks_config['parameters']={'enabled':True,'backend':'stt'}
    checks_config['tests']={'vad':True,'vcm':True,'stt':True,'pipeline':True}
    checks=run_checks(checks_config,candidate)
    backup=out/'previous-artifacts';backup.mkdir(exist_ok=True)
    for name in ('bundle.json','manifest.json','stt.pt','stt.fp32.onnx','brightness-validation.json'):
        path=root/'artifacts'/name
        if path.exists():shutil.copy2(path,backup/name)
    for name in ('stt.pt','stt.fp32.onnx'):
        shutil.copy2(candidate/name,root/'artifacts'/name)
        current['files'][name]=metadata['files'][name]
    current['stt']=metadata['stt'];current['brightness_parameters']=metadata['brightness_parameters']
    current['runtime_components']=list(dict.fromkeys(current.get('runtime_components',['vcm'])+['stt']))
    current['brightness_parameters']['experimental']=True
    write_json(root/'artifacts/bundle.json',current)
    write_json(root/'artifacts/brightness-validation.json',{'benchmark':report,'checks':checks,
        'promotion':'experimental default requested by user; development precision gate passed; natural coverage remains limited'})
    # Imported lazily: only the workstation facade owns artifact manifests.
    import train
    train.refresh_artifact_manifest(root)
    return {'artifacts_updated':True,'backend':'stt','checks':checks,'report':str(out/'benchmark.json')}


def run(root,config,action='all',extra_manifest=None,refresh=False):
    root=Path(root)
    if action=='fetch':
        fetch_context(root);return {'downloaded':True}
    if action=='prepare':
        rows=prepare(root,extra_manifest,refresh);return {'rows':len(rows)}
    if action in {'train','all'}:
        from .brightness_training import fit_brightness
        training=fit_brightness(root,config,extra_manifest=extra_manifest,refresh=refresh)
        if action=='train':return training
    if action in {'benchmark','all'}:
        from .brightness_benchmark import benchmark
        comparison=benchmark(root,config)
        if action=='benchmark':return {'report':str(root/'.cache/stt-human/benchmark.json')}
    if action=='install' or action=='all' and config['training']['update_artifacts']:
        return install(root,config)
    return {'artifacts_updated':False,'report':str(root/'.cache/stt-human/benchmark.json')}
