"""Endpoint-candidate adapter for the existing PCV publisher (no installed writes)."""
from pathlib import Path,PurePosixPath
import hashlib
import io
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
from scripts.runtime.pcv_final_runtime import BINDINGS,Runtime,verify_bundle,sha,read_json
from scripts.runtime.pcv_schema import canonical,normalize_text,output as typed_output


def source_path(name):
    """Receipts written on Windows remain readable on POSIX workstations."""
    return ROOT/Path(str(name).replace('\\','/'))


BOOTSTRAP = '''"""Offline setup, identity check and Pi replay/guided benchmark."""
import argparse,hashlib,json,os,platform,subprocess,sys
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        pure=PurePosixPath(name);path=(ROOT/name).resolve()
        if pure.is_absolute() or '..' in pure.parts or ':' in name or chr(92) in name or not path.is_relative_to(ROOT):
            raise ValueError('Unsafe package path')
        with path.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest: raise ValueError('Package identity mismatch: '+name)
if __name__=='__main__':
    verify()
    args=sys.argv[1:]
    if '--verify-only' in args:
        print('All packaged file identities verified.');sys.exit(0)
    if '--setup' in args:
        args.remove('--setup')
        if platform.machine() not in ('aarch64','arm64') or platform.system()!='Linux' or sys.version_info[:2]!=(3,12):
            raise SystemExit('Offline wheels require Linux ARM64 CPython 3.12 (glibc >=2.28). Run python3.12 pi_test.py --setup ...')
        venv=ROOT/'.venv'
        if not (venv/'bin/python').exists():subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
        python=venv/'bin/python'
        subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(ROOT/'wheels'),
            '--only-binary=:all:','--require-hashes','-r',str(ROOT/'requirements.txt')],check=True)
        subprocess.run([str(python),str(Path(__file__).resolve()),*args],check=True);sys.exit(0)
    if '--list-mics' in args:
        import sounddevice
        print(sounddevice.query_devices());sys.exit(0)
    from scripts.runtime.pcv_pi_benchmark import main
    sys.argv=[sys.argv[0],'--root',str(ROOT),*args]
    main()
'''


def smoke_samples(definitions,*,sample_manifest=None,source_receipt=None,source_card=None):
    """Extract original bytes from a hash-pinned shard; no audio conversion."""
    import pyarrow.parquet as pq
    import soundfile as sf
    folder=ROOT/'experiments/pcv-updated-voice-v1'
    provided=[sample_manifest,source_receipt,source_card]
    if any(p is not None for p in provided) and not all(p is not None for p in provided):
        raise ValueError('Provide sample_manifest, source_receipt and source_card together')
    if sample_manifest is None:
        freeze=read_json(folder/'data-freeze.json')
        manifest=source_path(freeze['manifests']['test']['path'])
        if sha(manifest)!=freeze['manifests']['test']['sha256']:raise ValueError('Smoke manifest changed')
        source_receipt=folder/'acquisition-complete.json';card=folder/'source-card.txt'
        protocol=read_json(folder/'acquisition-protocol.json')
        if sha(card)!=protocol['bindings'][str(card.relative_to(ROOT))]:raise ValueError('Source terms changed')
    else:manifest=Path(sample_manifest);card=Path(source_card)
    acquisition=read_json(source_receipt)
    protocols=[Path(source_receipt).parent/name for name in ('protocol.json','acquisition-protocol.json')]
    protocol_path=next((p for p in protocols if p.exists() and sha(p)==acquisition.get('protocol_sha256')),None)
    if protocol_path is None:raise ValueError('Acquisition protocol identity missing or changed')
    source_protocol=read_json(protocol_path)
    card_hashes=[digest for name,digest in source_protocol.get('bindings',{}).items() if source_path(name).resolve()==card.resolve()]
    if card_hashes!=[sha(card)]:raise ValueError('Source card is not bound to acquisition protocol')
    candidates=[json.loads(line) for line in manifest.read_text().splitlines()]
    selected=[];seen=set()
    # One original human example per supported intent, plus two explicit negatives.
    for row in candidates:
        if row['source']!='SLURP' or row['synthetic'] or row['reference_status'] not in ('supported','unsupported'):continue
        if row.get('split') not in ('test','holdout') or row.get('published_split') not in ('test','holdout'):continue
        key=row['expected']['intent'] if row['expected'] else 'negative'
        cap=2 if key=='negative' else 1
        if sum(r['expected'] is None if key=='negative' else bool(r['expected'] and r['expected']['intent']==key) for r in selected)>=cap:continue
        if row['audio_sha256'] in seen:continue
        selected.append(row);seen.add(row['audio_sha256'])
        if len(selected)>=20:break
    entries={};rows=[]
    definitions_by_text={normalize_text(d.get('spoken_text',d['text'])):d for d in definitions}
    for parquet_name in sorted({r['parquet'] for r in selected}):
        parquet=source_path(parquet_name)
        relative='data/'+parquet.name
        receipt=next((p for p in acquisition['files'] if source_path(p['path']).resolve()==parquet.resolve()),None)
        if receipt is None or sha(parquet)!=receipt['sha256']:raise ValueError('Pinned sample shard identity mismatch')
        indexes={r['parquet_row']:r for r in selected if r['parquet']==parquet_name}
        offset=0
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128,columns=['audio']):
            for i,record in enumerate(batch.to_pylist(),offset):
                if i not in indexes:continue
                reference=indexes[i];data=record['audio']['bytes']
                digest=hashlib.sha256(data).hexdigest()
                if digest!=reference['audio_sha256']:raise ValueError('Sample bytes changed')
                waveform,rate=sf.read(io.BytesIO(data),dtype='float32')
                if rate!=16000 or waveform.ndim!=1:raise ValueError('Unexpected sample format')
                name=f'slurpsmoke-{len(rows)+1:02d}.wav';entries['samples/'+name]=data
                definition=definitions_by_text.get(normalize_text(reference['text']))
                phrase_id=definition['id'] if definition and canonical(definition['command'])==canonical(reference['expected']) else None
                rows.append({'id':reference['id'],'audio':name,'sha256':digest,'expected':reference['expected'],
                    'expected_output':typed_output(reference['expected']) if reference['expected'] is not None else None,
                    'phrase_id':phrase_id,'spoken_text':reference['text'],'source':'SLURP','speaker':reference['speaker'],
                    'license':'CC BY 4.0','published_split':reference['published_split'],'synthetic':False,
                    'repository':acquisition['repository'],'revision':acquisition['revision'],
                    'parquet':relative,'parquet_row':i,'original_file':reference['file'],'modifications':'none; original WAV bytes'})
            offset+=batch.num_rows
    entries['samples/manifest.json']=(json.dumps(rows,indent=2)+'\n').encode()
    entries['licenses/HF-SOURCE-CARD.txt']=card.read_bytes()
    entries['licenses/HF-SOURCE-RECEIPT.json']=(json.dumps({'repository':acquisition['repository'],
        'revision':acquisition['revision'],'manifest_sha256':sha(manifest),'source_card_sha256':sha(card),
        'acquisition_receipt_sha256':sha(source_receipt),
        'selection':'first per-intent CC BY SLURP human test rows; up to two negatives; previously evaluated smoke only',
        'sample_count':len(rows)},indent=2)+'\n').encode()
    entries['licenses/SLURP-ATTRIBUTION.txt']=b'''SLURP: A Spoken Language Understanding Resource Package.
Authors: Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, Verena Rieser.
Original project: https://github.com/pswietojanski/slurp
License: Creative Commons Attribution 4.0 International.
https://creativecommons.org/licenses/by/4.0/
These bounded smoke samples were distributed through the pinned HF source listed
in HF-SOURCE-RECEIPT.json. Original WAV bytes are retained without modification.
Each original recording identifier, speaker and expected finite output is listed
in samples/manifest.json. No class-restricted group recordings are included.
'''
    return entries,rows


def handoff(recipe,policy,definitions,sample_count):
    model=recipe['model_config']
    return f'''# PCV Pi test package and model workings

This is a finite Voice Command Model. Acceptance enabled: **{policy['enabled']}**.
When the frozen policy is disabled it rejects speech with `unqualified_model`. `diagnostic_candidate`
is observable for investigation and must never trigger a device action.
Model quality, hardware speed, and software integrity are separate outcomes.

## Install and test

Target: Raspberry Pi 4/5, 64-bit Linux ARM64, glibc >= 2.28, CPython 3.12.
The archive includes hash-locked ARM64 wheels. Install CPython 3.12, its venv
support, libsndfile1 and libportaudio2 through your OS before offline setup.
On a copied and extracted archive run:

```sh
python3.12 pi_test.py --setup --output results/replay-01
.venv/bin/python pi_test.py --list-mics
.venv/bin/python pi_test.py --guided --mic 0 --case B1-001 --output results/live-01
```

Replace mic 0 with the actual listed input. Omit `--case` to walk all {len(definitions)}
definitions; repeat it to select several. Guided tests read the prompt and wait
for Enter. Say **Hey Jarvis**, pause until `command_ready`, then speak the command.
`s` skips a prompt and `q` ends the session. `--no-wake` provides fixed-duration
command-only recording for explicit capture diagnosis. Results directories must
be new. A desktop run labels physical Pi evidence false. Live sessions can record
failures, and the report does not claim the requested words were human verified.
The script writes metrics.json, REPORT.md and trials.jsonl suitable for slide input.
Use `--manifest path/to/manifest.json` for additional mono 16 kHz WAV recordings;
each record requires id, audio (relative path), sha256, expected typed command or
null, and optional phrase_id. Missing recordings are untested, never passes.

## Frontend contract

```sh
.venv/bin/python -m scripts.runtime.pcv_final_runtime --bundle models/default --diagnostic --wav command.wav
.venv/bin/python -m scripts.runtime.pcv_final_runtime --bundle models/default --diagnostic --mic 0
```

Standard output is JSONL. Command events contain `output` (the authoritative
schema-v3 accepted/command/reason result), a separate `diagnostic_candidate`,
model SHA-256, inference timing and audio duration. Wake events carry stage/status.
No action integrations exist. `output.schema.json` is the typed output contract.
Python callers may load Runtime and call recognize(float32_mono_pcm, 16000).
The runtime rejects wrong rates, stereo, NaNs, out-of-range PCM, silence and clips
over 15 seconds. It does not silently resample or convert a bad capture.

Consumer example for a disabled-policy command event (only the decision fields
are shown; this illustrates the format and is not a measured recognition):

```json
{{"event":"command","output":{{"schema_version":3,"accepted":false,"command":null,"reason":"unqualified_model"}},"diagnostic_candidate":{{"candidate":{{"intent":"TIME","slots":{{}}}}}}}}
```

An application reads `event.output.accepted` first. It may use
`event.output.command` only when accepted is true; diagnostic candidates are
display/debug data. Other event types update the wake/capture status indicator.

## Model computation

Mono 16 kHz PCM -> fixed 25 ms HTK log-mel windows every 10 ms -> 40 bins with
frozen global normalization -> CPU ONNX contextual encoder (width {model['width']},
{model['layers']} layers, {model['heads']} heads, feedforward {model['feedforward']}) ->
three raw heads: ctc [1,T,40], intent [1,T,19], target [1,T,2]. The input is
features [1,40,N], and T=ceil(N/2). There is no recurrent state. Complete utterances
are encoded after endpointing; this model does not support causal frame inference.
The frozen PrefixCTC computes full likelihoods only for compiled finite commands.
Intent/target probabilities and the calibration policy validate the selected
command. Acoustic cue values are never decoded into unrestricted words or text.
No PyTorch, training implementation, teacher, ASR/transcriber or cloud inference
dependency is included. CPU ONNX Runtime is the only model engine.

Wake uses pinned Hey Jarvis (mel, embedding and wake head); Silero VAD supplies
speech/endpoint probabilities. Its 320 ms wake-release pause and 400 ms command
silence endpoint are distinct from inference latency. All wake/VAD costs count
towards deployed bytes. Their provenance and license notices are in licenses/
and models/default/wake/wake-models.json. No physical Pi or live-mic test occurred
while this archive was built.

## Finite scope and replacement

All {len(definitions)} approved definitions and exact machine expectations are in
verbatims/definitions.json and VERBATIMS.md. A definition is not a recording or
an acoustic pass. The catalog additionally contains measured finite aliases;
catalog.json is the complete acoustic competition set. Custom names are bounded
IDs in custom-slots.json. Arbitrary names, dictation, free-form reminders/messages
and open-ended search are unsupported. Recognizing WEATHER does not fetch weather.
Adding a name requires a new compiled catalog, audited pronunciations/enrollment,
calibration and unknown-name tests. Editing display text cannot supply evidence.

To replace a model, use the workstation publisher on a newly selected bundle
containing model.onnx, recipe.json, normalization.json, catalog.json,
custom-slots.json, output.schema.json and policy.json. All binding hashes must
match, output heads/stride and frontend must match, and a changed state interface
requires a new adapter. Package a new directory and ZIP; never edit a graph in
place. Disabled-policy candidates require explicit diagnostic publication.
Enabling acceptance requires a matching qualification receipt and hashed results
with passing gates, not a policy toggle. Bundled evidence records that decision.

## Measurement limits

{sample_count} bundled CC BY SLURP recordings are a bounded previously evaluated
smoke subset. They do not establish all-phrase or held-out recognition quality.
Guided recordings remain user-operated and are not silently performed at build
time. Per-phrase zero denominators are untested. Report exact commands, precision,
recall, false accepts and slot denominators separately from diagnostic candidates.
Model timings include frontend, encoder and finite scorer but exclude utterance
duration, disk reads and wake/endpoint waiting. The report includes p50/p95, RTF,
thread/provider settings, process CPU, peak RSS when available and device identity.
'''


def qualification_files(path,binding,policy_sha256):
    if path is None:raise ValueError('Enabled publication requires qualification_receipt')
    path=Path(path);receipt=read_json(path)
    if receipt.get('qualified') is not True or receipt.get('test_used_for_selection') is not False or receipt.get('model_binding')!=binding or receipt.get('policy_sha256')!=policy_sha256:
        raise ValueError('Qualification receipt does not pass or match model binding')
    results_path=path.parent/receipt['results_path']
    if sha(results_path)!=receipt.get('results_sha256'):raise ValueError('Qualification results identity mismatch')
    results=read_json(results_path)
    if results.get('model_binding')!=binding or results.get('qualified') is not True or results.get('policy_sha256')!=policy_sha256:
        raise ValueError('Qualification result gates failed or binding changed')
    return {'evidence/qualification-receipt.json':path.read_bytes(),'evidence/qualification-results.json':results_path.read_bytes()}


def add_extra_files(entries,extra_files):
    for name,path in (extra_files or {}).items():
        pure=PurePosixPath(name)
        if pure.is_absolute() or len(pure.parts)!=2 or pure.parts[0] not in {'docs','slides','evidence'} or '..' in pure.parts or ':' in name or '\\' in name:
            raise ValueError('Unsafe or disallowed extra archive path: '+name)
        if name in entries:raise ValueError('Extra file collides with packaged source: '+name)
        if Path(path).suffix.lower() not in {'.md','.txt','.json','.jsonl','.pdf','.pptx','.html'}:
            raise ValueError('Only presentation, report and evidence extras are supported')
        entries[name]=Path(path).read_bytes()


def write_archive(entries,output,metadata):
    """One checked ZIP writer shared by endpoint and direct-command packages."""
    from scripts.runtime.speaker_package import safe_name
    output=Path(output)
    if output.exists():raise FileExistsError('ZIP outputs are immutable')
    for name in entries:safe_name(name)
    payload=dict(entries)
    payload['MANIFEST.json']=(json.dumps({**metadata,'files':{n:hashlib.sha256(b).hexdigest() for n,b in entries.items()}},indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(2026,10,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;archive.writestr(info,data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
    return {'path':str(output),'sha256':sha(output),'bytes':output.stat().st_size}


def package(bundle,output,*,diagnostic=False,portable_parity=None,sample_manifest=None,
            source_receipt=None,source_card=None,qualification_receipt=None,extra_files=None,
            app_target=None,benchmarks='skip',benchmark_manifest=None,ncnn_folder=None,wake_folder=None,vad_folder=None):
    bundle=Path(bundle);output=Path(output)
    if (bundle/'bundle.json').exists() and read_json(bundle/'bundle.json').get('format')=='mvp-static-v1':
        if not diagnostic:raise ValueError('Direct MVP is an unqualified candidate')
        from scripts.mvp_delivery import package_candidate
        return package_candidate(bundle,output,wake_folder=wake_folder,vad_folder=vad_folder,extra_files=extra_files)
    if output.exists():raise FileExistsError('ZIP outputs are immutable')
    recipe,policy,norm,catalog=verify_bundle(bundle,diagnostic=diagnostic)
    qualification=qualification_files(qualification_receipt,recipe['binding'],sha(bundle/'policy.json')) if policy['enabled'] else {}
    runtime=Runtime(bundle,diagnostic=diagnostic)
    from pcvcm.dependencies import verify_lock
    lock=ROOT/'experiments/phrase-contract-v1/configs/runtime-cp312-aarch64'
    wheels=ROOT/'.cache/phrase-contract-v1/dependencies/cp312-aarch64'
    dependencies=verify_lock(lock,wheels) if app_target is None else {'target':'uv-locked-cross-platform'}
    entries=dict(qualification)
    for name in ('scripts/__init__.py','scripts/runtime/__init__.py','vcm/__init__.py','pcvcm/__init__.py'):
        entries[name]=b''
    for name in ('vcm/frontend.py','vcm/catalog.py','scripts/runtime/pcv_schema.py',
                 'scripts/runtime/pcv_final_runtime.py','scripts/runtime/speaker_confidence.py','scripts/runtime/pcv_pi_benchmark.py','scripts/runtime/wake_models.py'):
        entries[name]=(ROOT/name).read_bytes()
    for name in ('contract.py','decoder.py'):
        entries['pcvcm/'+name]=(ROOT/'experiments/phrase-contract-v1/pcvcm'/name).read_bytes()
    entries['wake_gate.py']=(ROOT/'experiments/pcv-synthetic-v2/wake_gate.py').read_bytes()
    assets=set(BINDINGS.values())|{'recipe.json','policy.json'}
    for name in assets:entries['models/default/'+name]=(bundle/name).read_bytes()
    if ncnn_folder is not None:
        folder=Path(ncnn_folder);record=read_json(folder/'engine.json')
        if record['onnx_sha256']!=sha(bundle/'model.onnx') or not record.get('parity',{}).get('passed'):raise ValueError('ncnn conversion is not verified for this bundle')
        for name in ('model.ncnn.param','model.ncnn.bin','position-frequency.json','engine.json'):
            if name!='engine.json' and record['files'].get(name)!=sha(folder/name):raise ValueError('ncnn export changed')
            entries['models/default/ncnn/'+name]=(folder/name).read_bytes()
    wake_source=bundle/'wake' if (bundle/'wake').is_dir() else ROOT/'artifacts'
    for name in ('wake-melspectrogram.onnx','wake-embedding.onnx','wake-hey-jarvis.onnx','silero-vad.onnx','wake-models.json'):
        entries['models/default/wake/'+name]=(wake_source/name).read_bytes()
    from scripts.runtime.wake_models import session
    for name in ('wake-melspectrogram.onnx','wake-embedding.onnx','wake-hey-jarvis.onnx','silero-vad.onnx'):session(wake_source,name)
    deployed={name.removeprefix('models/default/'):hashlib.sha256(data).hexdigest() for name,data in entries.items() if name.startswith('models/default/')}
    entries['models/default/deployment.json']=(canonical({'files':deployed,'status':'QUALIFIED' if policy['enabled'] else 'DIAGNOSTIC_UNQUALIFIED'})+'\n').encode()
    definitions=read_json(ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json')
    entries['verbatims/definitions.json']=(json.dumps(definitions,indent=2)+'\n').encode()
    definitions=[{**row,'expected_output':typed_output(row['command'],None if row['command'] is not None else row.get('reason','OOV'))} for row in definitions]
    entries['verbatims/definitions.json']=(json.dumps(definitions,indent=2)+'\n').encode()
    lines=['# Spoken phrases and expected finite commands','','Definitions only; missing recordings are untested. These are desired typed outputs, not promises from a disabled or failing model. Negative expected null means rejection.','']
    for row in definitions:
        lines.extend([f"## {row['id']} ({row['tier']})",'',row.get('spoken_text',row['text']),'',
                      '```json',canonical(row['expected_output']),'```',''])
    entries['VERBATIMS.md']=('\n'.join(lines)+'\n').encode()
    rows=[]
    if app_target!='inference':
        samples,rows=smoke_samples(definitions,sample_manifest=sample_manifest,source_receipt=source_receipt,source_card=source_card);entries.update(samples)
    else:
        for name in ('verbatims/definitions.json','VERBATIMS.md','scripts/runtime/pcv_pi_benchmark.py'):entries.pop(name,None)
    entries['HANDOFF.md']=handoff(recipe,policy,definitions,len(rows)).encode()
    entries['pi_test.py']=BOOTSTRAP.encode()
    if app_target is None:
        for name in ('requirements.txt','THIRD_PARTY_NOTICES.txt','dependencies.json'):entries[name]=(lock/name).read_bytes()
        for wheel in dependencies['wheels'].values():entries['wheels/'+wheel['filename']]=(wheels/wheel['filename']).read_bytes()
    for name in ('openwakeword-LICENSE.txt','silero-LICENSE.txt','pretrained-NOTICE.txt'):entries['licenses/'+name]=(ROOT/'artifacts'/name).read_bytes()
    entries['licenses/CMUDICT-LICENSE.txt']=(ROOT/'experiments/phrase-contract-v1/configs/CMUDICT-LICENSE.txt').read_bytes()
    add_extra_files(entries,extra_files)
    if benchmark_manifest is not None and app_target!='inference':
        from scripts.pcv_benchmark_package import benchmark_files
        entries.update(benchmark_files(benchmark_manifest))
    application={}
    if app_target is not None:
        from scripts.speaker_publish import application_files,runtime_identity
        entries.update(application_files(app_target,benchmarks))
        # Legacy ARM-only setup instructions are not shipped in the uv application.
        entries.pop('HANDOFF.md',None);entries.pop('pi_test.py',None)
        application={'target':app_target,'runtime_id':runtime_identity(entries),'startup_action':'install-model',
                     'bundle_id':sha(bundle/'recipe.json'),'output_schema_version':3,'runtime_api_version':2}
    # Explicit allowlist above excludes all trainers, feature caches and teacher artifacts.
    metadata={'format':'smart-speaker-v1' if app_target else 'pcv-endpoint-pi-v1',**application,
        'model_binding':recipe['binding'],'diagnostic':diagnostic,'acceptance_enabled':policy['enabled'],
        'physical_pi_measured':False,'live_microphone_measured':False}
    write_archive(entries,output,metadata)
    receipt={'path':str(output),'sha256':sha(output),'bytes':output.stat().st_size,'model_binding':recipe['binding'],
             'diagnostic':diagnostic,'acceptance_enabled':policy['enabled'],'sample_count':len(rows),'defined_phrases':len(definitions),
             'catalog_entries':len(catalog),'runtime_load_ms':runtime.load_ms,'target':dependencies['target'],
             'physical_pi_measured':False,'live_microphone_measured':False}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt
