"""Self-contained bundle verification and the unchanged finite CPU recognizer."""
import json
import math
from pathlib import Path,PurePosixPath
import numpy as np
from pcvcm.contract import file_hash,identity
from pcvcm.runtime import StreamEngine,OnnxEncoder
from pcvcm.endpoint import EndpointConfig
from .catalog import CatalogContract

STATE_SHAPES=[(1,40,2),(1,64,4),(1,64,8),(1,64,16),(1,64,32),(1,64,64)]
WAKE_FILES=('wake-melspectrogram.onnx','wake-embedding.onnx','wake-hey-jarvis.onnx','silero-vad.onnx','wake-models.json')
ASSETS=('model.onnx','catalog.json','normalization.json','endpoint.json','settings.json',*[f'wake/{n}' for n in WAKE_FILES])
EVIDENCE=('source-manifest.json','source-policy.json','source-protocol.json','source-identity.json','calibration-selection.json',
          'source-finalization.json','command-results.json','wake-results.json','demo-results.json')


def read_json(path):
    def invalid(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def below(root,name):
    if not isinstance(name,str):raise ValueError('Invalid bundle path')
    pure=PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name:raise ValueError('Unsafe bundle path')
    result=(Path(root)/name).resolve()
    if not result.is_relative_to(Path(root).resolve()):raise ValueError('Bundle path escapes root')
    return result


def verify_bundle(root,*,allow_diagnostic=False):
    root=Path(root);manifest=read_json(root/'deployment.json')
    if manifest.get('schema_version')!=1 or set(manifest.get('files',{}))!=set(ASSETS):raise ValueError('Unsupported deployment manifest')
    for name,digest in manifest['files'].items():
        path=below(root,name)
        if not path.is_file() or path.stat().st_size>128*1024*1024 or file_hash(path)!=digest:raise ValueError('Bundle asset identity changed: '+name)
    if (root/'model.onnx').stat().st_size>=1024*1024:raise ValueError('VCM graph exceeds 1 MiB')
    binding=identity(manifest['files'])
    if binding!=manifest.get('binding') or file_hash(root/'policy.json')!=manifest.get('policy_sha256'):
        raise ValueError('Deployment binding changed')
    policy=read_json(root/'policy.json')
    if policy.get('binding')!=binding or type(policy.get('enabled')) is not bool:raise ValueError('Stale policy binding')
    status=manifest.get('status')
    if status=='DIAGNOSTIC_UNQUALIFIED':
        if not allow_diagnostic or policy['enabled']:raise ValueError('Diagnostic mode requires explicit opt-in and disabled acceptance')
    elif status!='SYNTHETIC_VALIDATED' or not policy['enabled']:raise ValueError('No validated acceptance policy')
    if status=='SYNTHETIC_VALIDATED':
        if set(manifest.get('evidence_files',{}))!=set(EVIDENCE):raise ValueError('Missing bound acceptance evidence')
        for name,digest in manifest['evidence_files'].items():
            if file_hash(root/name)!=digest:raise ValueError('Acceptance evidence changed')
        source=read_json(root/'source-manifest.json');source_policy=read_json(root/'source-policy.json')
        source_binding=identity(source['files'])
        if source_binding!=source['binding'] or file_hash(root/'source-policy.json')!=source['policy_sha256'] or source_policy['binding']!=source_binding:
            raise ValueError('Source candidate binding changed')
        if any(source['files'][n]!=manifest['files'][n] for n in ('model.onnx','catalog.json','normalization.json','endpoint.json')):
            raise ValueError('Published assets differ from measured candidate')
        protocol=read_json(root/'source-protocol.json')
        if file_hash(root/'source-protocol.json')!=source['files']['protocol.json']:raise ValueError('Measured protocol changed')
        if file_hash(root/'source-identity.json')!=source['files']['source-identity.json']:
            raise ValueError('Measured source identity changed')
        for name in WAKE_FILES:
            if manifest['files']['wake/'+name]!=protocol['wake_assets']['artifacts\\'+name]:raise ValueError('Wake assets differ from measured candidate')
        measured_settings=dict(sample_rate=16000,threads=1,wake_threshold=protocol['wake_threshold'],
            release_ms=protocol['wake_release_ms'],release_timeout_seconds=protocol['wake_release_timeout_seconds'],
            command_timeout_seconds=protocol['wake_command_timeout_seconds'])
        from pcvcm.contract import canonical
        if canonical(read_json(root/'settings.json'))!=canonical(measured_settings):raise ValueError('Measured wake profile changed')
        calibration=read_json(root/'calibration-selection.json')
        if calibration.get('test_accessed') is not False or canonical(calibration.get('policy'))!=canonical(source_policy):
            raise ValueError('Published policy differs from pre-test selection')
        if source_policy.get('enabled') is not True or canonical(source_policy.get('thresholds'))!=canonical(policy.get('thresholds')):
            raise ValueError('Published policy differs from calibration')
        final,command,wake,demo=[read_json(root/n) for n in ('source-finalization.json','command-results.json','wake-results.json','demo-results.json')]
        if any(r.get('binding')!=source_binding for r in (final,command,wake,demo)):
            raise ValueError('Evidence candidate mismatch')
        if (not final.get('synthetic_qualified') or not command.get('complete') or not command.get('synthetic_gates_passed')
            or not wake.get('complete') or not wake.get('wake_gates_passed') or not demo.get('demo_gates_passed')
            or file_hash(root/'command-results.json')!=final['command_results_sha256']
            or file_hash(root/'wake-results.json')!=final['wake_results_sha256']
            or command['predictions_sha256']!=demo['predictions_sha256']):raise ValueError('Acceptance gates failed or changed')
    for limits in policy.get('thresholds',{}).values():
        if not isinstance(limits,list) or len(limits)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in limits) or limits[1]<0:
            raise ValueError('Invalid acceptance thresholds')
    settings=read_json(root/'settings.json')
    if settings!={'sample_rate':16000,'threads':1,'wake_threshold':.5,'release_ms':320,'release_timeout_seconds':3.,'command_timeout_seconds':5.}:
        raise ValueError('Unsupported deployment/wake profile')
    catalog=read_json(root/'catalog.json');entries=catalog.get('entries',[])
    if not 1<=len(entries)<=4096:raise ValueError('Catalog entry budget exceeded')
    ids=set();prefixes=set();phones_seen={}
    from pcvcm.contract import canonical
    for row in entries:
        if not isinstance(row.get('id'),str) or row['id'] in ids:raise ValueError('Invalid catalog IDs')
        ids.add(row['id']);phones=row.get('phones')
        if not isinstance(phones,list) or not 1<=len(phones)<=256 or any(type(p) is not int or not 0<p<40 for p in phones):
            raise ValueError('Invalid phone inventory')
        key=tuple(phones);command=canonical(row.get('command'))
        if key in phones_seen and phones_seen[key]!=command:raise ValueError('Conflicting acoustic targets')
        phones_seen[key]=command
        if row.get('use_case') not in {f'UC-{i:02}' for i in range(1,11)}:raise ValueError('Unknown use case')
        for i in range(1,len(key)+1):prefixes.add(key[:i])
        if len(prefixes)+1>60000:raise ValueError('Catalog node budget exceeded')
    return dict(root=root,binding=binding,catalog=catalog,policy=policy,settings=settings,
        normalization=read_json(root/'normalization.json'),endpoint=EndpointConfig(**read_json(root/'endpoint.json')),
        diagnostic=status=='DIAGNOSTIC_UNQUALIFIED',node_count=len(prefixes)+1,manifest=manifest)


def cpu_session(path):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def verify_recognition_source(content):
    """A hot-loaded model must have been measured with this recognizer code."""
    if content['diagnostic']:return
    import importlib
    root=content['root'];protocol=read_json(root/'source-protocol.json');sources=read_json(root/'source-identity.json')
    names={'pcvcm':'experiments/phrase-contract-v1/pcvcm/__init__.py',
        **{'pcvcm.'+n:'experiments/phrase-contract-v1/pcvcm/'+n+'.py' for n in ('contract','runtime','decision','decoder','endpoint')},
        'vcm.frontend':'vcm/frontend.py','scripts.runtime.wake_models':'scripts/runtime/wake_models.py',
        'wake_gate':'experiments/pcv-synthetic-v2/wake_gate.py'}
    for module,original in names.items():
        key=original.replace('/','\\')
        expected=sources.get(key) or protocol['shared_sources'].get(key)
        actual=importlib.import_module(module)
        if not expected or file_hash(Path(actual.__file__))!=expected:
            raise ValueError('Executing recognition source differs from measured bundle: '+module)


def load_bundle(root,*,allow_diagnostic=False):
    content=verify_bundle(root,allow_diagnostic=allow_diagnostic);root=content['root']
    verify_recognition_source(content)
    from scripts.runtime.wake_models import HeyJarvis,SileroProbability
    # Reuse the exact frozen feature/state implementations with strict session
    # scheduling for all four activation graphs as well as the VCM graph.
    class CpuWake(HeyJarvis):
        def __init__(self):
            self.mel=cpu_session(root/'wake/wake-melspectrogram.onnx')
            self.embedding=cpu_session(root/'wake/wake-embedding.onnx')
            self.head=cpu_session(root/'wake/wake-hey-jarvis.onnx');self.reset()
    class CpuVoice(SileroProbability):
        def __init__(self):self.model=cpu_session(root/'wake/silero-vad.onnx');self.reset()
    from wake_gate import WakeGate
    # Keep the shared encoder's cache reset and forward computation; initialize
    # its session here so both ORT spinning settings are explicitly disabled.
    encoder=OnnxEncoder.__new__(OnnxEncoder)
    encoder.session=cpu_session(root/'model.onnx');encoder.shapes=STATE_SHAPES
    if ([x.name for x in encoder.session.get_inputs()]!=['features',*[f'state_{i}' for i in range(6)]] or
        [x.name for x in encoder.session.get_outputs()]!=['ctc',*[f'next_state_{i}' for i in range(6)]]):
        raise ValueError('Unexpected ONNX streaming interface')
    encoder.reset()
    if encoder.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPU-only inference required')
    probe=encoder(np.zeros((40,4),dtype=np.float32))
    if probe.shape!=(4,40):raise ValueError('Invalid ONNX CTC dimensions')
    encoder.reset()
    command=StreamEngine(CatalogContract(content['catalog']),content['catalog'],encoder,content['normalization'],
        content['endpoint'],content['policy'],content['binding'])
    gate=WakeGate(command,CpuWake(),CpuVoice())
    return {**content,'gate':gate}
