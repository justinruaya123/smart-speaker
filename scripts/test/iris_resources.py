"""Bounded, action-free CPU cost profile of the actual local recognition models.

Run ``python -m scripts.test.iris_resources --help``. This is resource evidence
on the recorded host, not live latency, identity authentication or speech/Pi
qualification. No training framework, microphone, worker or provider is started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from scripts.runtime.iris_pi_benchmark import digest, hardware, memory_usage, quantiles, read_pcm, telemetry

SAMPLE_RATE = 16000
SAMPLES = 8
PREFIX_SAMPLES = 8000
WARMUP_SAMPLES = 32000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def measured(call):
    cpu, wall = time.process_time(), time.perf_counter()
    result = call()
    return result, {'wall_ms': (time.perf_counter()-wall)*1000,
                    'cpu_ms': (time.process_time()-cpu)*1000}


def summary(values):
    return {key: quantiles([item[key] for item in values]) for key in ('wall_ms', 'cpu_ms')}


def total(values):
    return {key: sum(item[key] for item in values) for key in ('wall_ms', 'cpu_ms')}


def relative(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path.is_relative_to(root):
        return path.relative_to(root).as_posix()
    # Worktrees deliberately mount the one shared cache outside their root.
    # Inventory receives resolved model paths, so lexical path handling alone
    # cannot preserve frozen repository-relative manifest/model bindings.
    cache = (root/'.cache').resolve()
    if not root.is_relative_to(cache) and path.is_relative_to(cache):
        return (Path('.cache')/path.relative_to(cache)).as_posix()
    return str(path)


def inventory(bundle, wake, owner, vad_folder, root):
    """Verify all shipped graph variants, including inactive engine duplicates."""
    paths, files = {}, []
    def add(component, path, kind='metadata', expected=None, size=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected or size is not None and path.stat().st_size != size:
            raise ValueError('Changed model binding: ' + str(path))
        paths[f'{component}:{relative(path, root)}'] = path
        files.append(dict(component=component, path=relative(path, root), kind=kind,
                          bytes=path.stat().st_size, sha256=actual))
        return read(path) if path.suffix == '.json' else None
    vcm = add('vcm', bundle/'bundle.json')
    for name, expected in vcm['files'].items():
        path = (bundle/name).resolve()
        if not path.is_relative_to(bundle): raise ValueError('Unsafe VCM asset path')
        add('vcm', path, 'graph' if name.endswith('.onnx') else 'metadata', expected)
    native = add('vcm', bundle/'ncnn/engine.json')
    if native['onnx_sha256'] != vcm['files']['model.onnx'] or not native['parity']['passed']:
        raise ValueError('Unverified packaged VCM ncnn conversion')
    for name in ('model.ncnn.param', 'model.ncnn.bin', 'position-frequency.json'):
        add('vcm', bundle/'ncnn'/name, 'support' if name.endswith('.json') else 'graph', native['files'][name])
    wake_meta = add('wake', wake/'iris.json')
    for name in ('iris.onnx', 'iris.ncnn.param', 'iris.ncnn.bin'):
        add('wake', wake/name, 'graph', wake_meta['files'][name])
    owner_meta = add('owner', owner/'owner.json')
    owner_graph = (owner/owner_meta['model']['path']).resolve()
    if not owner_graph.is_relative_to(owner): raise ValueError('Unsafe owner asset path')
    add('owner', owner_graph, 'graph', owner_meta['model']['sha256'], owner_meta['model'].get('bytes'))
    vad_meta = add('silero', vad_folder/'wake-models.json')
    silero = vad_meta['files']['silero-vad.onnx']
    add('silero', vad_folder/'silero-vad.onnx', 'graph', silero['sha256'], silero['bytes'])
    sizes = dict(files=files, graph_bytes=sum(row['bytes'] for row in files if row['kind']=='graph'),
        metadata_and_support_bytes=sum(row['bytes'] for row in files if row['kind']!='graph'),
        all_packaged_model_asset_bytes=sum(row['bytes'] for row in files),
        by_component={name:sum(row['bytes'] for row in files if row['component']==name)
                      for name in ('vcm','wake','owner','silero')},
        scope='Runtime graph variants plus required model metadata/support; includes ONNX and ncnn copies. '
              'Silero graph is packaged once and loaded into two sessions. Excludes training checkpoints, '
              'Python/Node libraries, UI, licenses, archive compression and operating-system memory.')
    def count(value):
        return value if type(value) is int and value > 0 else None
    counts = dict(vcm=count(vcm.get('parameters')), wake=count(wake_meta.get('parameters')),
                  owner=count(owner_meta['model'].get('runtime_parameters')))
    parameters = dict(**counts, known_own_runtime_total=sum(value for value in counts.values() if value is not None),
        metadata_count_complete=all(value is not None for value in counts.values()),
        silero={'trainable_parameters':None,'status':'unknown; pretrained graph trainable-parameter inventory not audited'},
        complete_pipeline_trainable_parameters=None,
        scope='Declared VCM/wake parameter counts and owner runtime_parameters; no ONNX initializer-size proxy. '
              'Owner training identity classifier is excluded from its runtime count.')
    return sizes, parameters, paths


def inputs(root):
    """Select frozen first eight own synthetic positives, never human payloads."""
    from scripts.iris_release import resolve_audio
    from scripts.runtime.pcv_schema import validate_command
    here = root/'experiments/iris-release-v1'
    protocol_path = here/'protocol.json'
    protocol = read(protocol_path)
    personal = root/'.cache/iris-release-v1/inputs/base-views/personal-test.jsonl'
    generated = root/'.cache/iris-release-v1/diversity/generated.jsonl'
    complete_path = here/'generation-complete.json'
    complete = read(complete_path)
    if protocol['input_bindings'].get(relative(personal,root)) != digest(personal):
        raise ValueError('Changed pinned personal command manifest')
    if complete['manifest_sha256'] != digest(generated) or complete['protocol_sha256'] != digest(protocol_path):
        raise ValueError('Complete pinned synthetic generation is required before resource profiling')
    license_path = root/'data/sources/qwen-synthetic-completion.source.json'
    if read(license_path).get('license') != 'Apache-2.0':
        raise ValueError('Missing declared license for own synthetic resource WAVs')
    bound = {'protocol':protocol_path,'generation_complete':complete_path,'personal_manifest':personal,
             'generated_manifest':generated,'synthetic_license':license_path}
    groups = {}
    for kind, manifest in [('commands',personal),('wakes',generated)]:
        selected=[];seen=set()
        for line in manifest.read_text(encoding='utf8').splitlines():
            if not line.strip(): continue
            row=json.loads(line)
            eligible=(row.get('expected') is not None and row.get('role')=='command') if kind=='commands' else (
                row.get('wake_positive') is True and row.get('split')=='voice-test' and row.get('role')=='wake')
            if not eligible: continue
            if row.get('synthetic') is not True or not str(row.get('source','')).startswith('Qwen3-original-designed-'):
                continue
            if row.get('sha256') in seen: continue
            if kind=='commands': validate_command(row['expected'])
            path=resolve_audio(row,root=root)
            pcm=read_pcm(path,row['sha256'])
            selected.append((row,pcm));seen.add(row['sha256'])
            bound[kind+':'+str(row['id'])]=path
            if len(selected)==SAMPLES: break
        if len(selected)!=SAMPLES:
            raise ValueError(f'Requires {SAMPLES} distinct licensed own synthetic {kind} in the pinned manifest')
        groups[kind]=selected
    return groups, bound


def software_versions():
    result={}
    for package in ('numpy','scipy','soundfile','onnxruntime','ncnn','einops'):
        try: result[package]=version(package)
        except PackageNotFoundError: result[package]='unavailable'
    return result


def source_bindings(root):
    """Bind the executing checkout's code and the input root's configuration.

    --root can select separate model/data inputs. It does not import or execute
    a shadow implementation there. A copied standalone harness has its own ROOT.
    Source binding keys stay portable; observed roots below are provenance only.
    """
    names=['scripts/test/iris_resources.py','scripts/runtime/iris_pi_benchmark.py','scripts/iris_release.py',
           'scripts/runtime/mvp_static.py','scripts/runtime/mvp_wake.py','scripts/runtime/wake_models.py',
           'scripts/runtime/wake_gate.py','scripts/runtime/iris_owner.py','scripts/runtime/speaker_session.py',
           'scripts/runtime/speaker_worker.py','scripts/runtime/speaker_config.py','scripts/runtime/pcv_schema.py',
           'scripts/runtime/audio.py','vcm/frontend.py']
    result={name:ROOT/name for name in names}
    for name in ('config.toml','deployment.json','.env'):
        if (root/name).is_file(): result['configuration:'+name]=root/name
    return result


def load_components(bundle,wake,owner,vad_folder,threads,engine):
    from scripts.runtime.mvp_static import Runtime
    from scripts.runtime.mvp_wake import HeyIris
    from scripts.runtime.wake_models import SileroProbability
    from scripts.runtime.iris_owner import OwnerBackend
    objects={};loads={};memory=[{'stage':'before_load',**memory_usage()}]
    constructors={'vcm':lambda:Runtime(bundle,threads=threads,engine=engine,mode='demo',profile='benchmark-compatible'),
                  'wake':lambda:HeyIris(wake,engine=engine),'owner':lambda:OwnerBackend(owner,threads=threads),
                  'wake_vad':lambda:SileroProbability(vad_folder),'endpoint_vad':lambda:SileroProbability(vad_folder)}
    for name,constructor in constructors.items():
        objects[name],loads[name]=measured(constructor)
        memory.append({'stage':'after_'+name,**memory_usage()})
    return objects,loads,memory


def stream_clip(wake,voice,pcm,components):
    """Run original 512-sample CPU calls; no sleeping or fabricated scores."""
    wake.reset();voice.reset()
    def phase(values):
        padded=np.pad(values,(0,(-len(values))%512))
        timings=[];hits=0;speech=0
        for offset in range(0,len(padded),512):
            frame=padded[offset:offset+512]
            def step():
                nonlocal hits,speech
                probability,timing=measured(lambda:voice(frame));components['silero_wake_frame'].append(timing)
                score,timing=measured(lambda:wake.push(frame));components['wake_push_frame'].append(timing)
                hits+=int(score==1.);speech+=int(probability>=.5)
            _,timing=measured(step);timings.append(timing)
        return dict(**total(timings),frames=len(timings),source_samples=len(values),delivered_samples=len(padded),
                    zero_alignment_samples=len(padded)-len(values),hits=hits,speech_frames=speech)
    warmup=phase(np.zeros(WARMUP_SAMPLES,dtype='float32'))
    active=phase(pcm)
    active['wall_rtf']=active['wall_ms']/1000/(len(pcm)/SAMPLE_RATE)
    active['cpu_rtf']=active['cpu_ms']/1000/(len(pcm)/SAMPLE_RATE)
    return warmup,active


def continuity(owner,pcm):
    """Instrument actual backend calls without copying its segmentation policy."""
    evidence=owner.evidence;calls=[]
    previous=owner.__dict__.get('evidence')
    def observed(values):
        result,timing=measured(lambda:evidence(values));calls.append(timing)
        return result
    owner.evidence=observed
    try: result,timing=measured(lambda:owner.continuity(pcm))
    finally:
        if previous is None: del owner.evidence
        else: owner.evidence=previous
    return result,timing,calls


def measurements(objects,groups):
    from scripts.runtime.speaker_session import SpeakerSession,gate_application
    from scripts.runtime.wake_models import SileroVAD
    from scripts.runtime.pcv_schema import canonical
    runtime,wake,owner=objects['vcm'],objects['wake'],objects['owner']
    components={key:[] for key in ('wake_graph_with_frontend','wake_push_frame','silero_wake_frame',
        'wake_stream','wake_warmup','vcm_final','vcm_preview','owner_wake_reference','owner_command_whole',
        'owner_continuity','owner_continuity_window','command_vad','speaker_arm','speaker_finish','application_gate')}
    original_score=wake.score
    def timed_score(values):
        result,timing=measured(lambda:original_score(values));components['wake_graph_with_frontend'].append(timing)
        return result
    wake.score=timed_score
    wakes=[];references=[]
    for row,pcm in groups['wakes']:
        warmup,active=stream_clip(wake,objects['wake_vad'],pcm,components)
        evidence,timing=measured(lambda:owner.evidence(pcm));references.append((evidence,timing))
        for key,value in [('wake_warmup',warmup),('wake_stream',active),('owner_wake_reference',timing)]:components[key].append(value)
        wakes.append(dict(id=row['id'],sha256=row['sha256'],audio_seconds=len(pcm)/SAMPLE_RATE,
                          warmup=warmup,active=active,owner_reference=timing))
    endpoint=SileroVAD(None,{'silence_ms':400,'max_utterance_seconds':15,'preroll_ms':128,'speech_threshold':.5},
                       probability=objects['endpoint_vad'])
    commands=[]
    for index,(row,pcm) in enumerate(groups['commands']):
        stages={};previews=[]
        for end in range(PREFIX_SAMPLES,len(pcm),PREFIX_SAMPLES):
            candidate,timing=measured(lambda:runtime.recognize(pcm[:end],sample_rate=SAMPLE_RATE))
            components['vcm_preview'].append(timing)
            diagnostic=candidate.get('diagnostic_candidate') or {}
            previews.append(dict(samples=end,raw_candidate=diagnostic.get('candidate'),
                score=diagnostic.get('class_probability'),output=candidate['output'],timing=timing))
        event,stages['vcm_final']=measured(lambda:runtime.recognize(pcm,sample_rate=SAMPLE_RATE))
        endpoints,stages['command_vad']=measured(lambda:endpoint.segment(pcm))
        whole,stages['owner_command_whole']=measured(lambda:owner.evidence(pcm))
        windows,stages['owner_continuity'],window_times=continuity(owner,pcm)
        components['owner_continuity_window'].extend(window_times)
        reference,reference_time=references[index%len(references)]
        stages['owner_wake_reference']=reference_time
        session=SpeakerSession(owner.policy);token='resource-'+str(index)
        arm,stages['speaker_arm']=measured(lambda:session.arm(token,reference,now=0.))
        verdict,stages['speaker_finish']=measured(lambda:session.finish(token,whole,now=.1,windows=windows))
        output,stages['application_gate']=measured(lambda:gate_application(event['output'],verdict))
        for key,value in stages.items():
            if key!='owner_wake_reference':components[key].append(value)
        stages['vcm_frontend']={'wall_ms':event.get('frontend_ms'),'cpu_ms':None}
        stages['vcm_engine_decision_and_wrapper']={'wall_ms':stages['vcm_final']['wall_ms']-event['frontend_ms'], 'cpu_ms':None}
        # Aggregate complete measured spans, never add nested window/frontend
        # timings a second time. Preview calls are a serial max-cadence budget.
        budget=total([value for key,value in stages.items() if key not in
            ('vcm_frontend','vcm_engine_decision_and_wrapper')]+[item['timing'] for item in previews])
        complete=total([budget,wakes[index%len(wakes)]['active']])
        seconds=(len(pcm)/SAMPLE_RATE)+wakes[index%len(wakes)]['audio_seconds']
        complete.update(wall_rtf=complete['wall_ms']/1000/seconds,cpu_rtf=complete['cpu_ms']/1000/seconds)
        commands.append(dict(id=row['id'],sha256=row['sha256'],audio_seconds=len(pcm)/SAMPLE_RATE,
            timing=stages,preview_calls=len(previews),preview_candidates=previews,
            preview_candidate_changes=sum(canonical(a['raw_candidate'])!=canonical(b['raw_candidate']) for a,b in zip(previews,previews[1:])),
            final_raw_candidate=(event.get('diagnostic_candidate') or {}).get('candidate'),output=output,
            owner_continuity_calls=len(window_times),owner_continuity_complete=windows is not None,
            owner_arm={'accepted':arm.accepted,'reason':arm.reason},owner_final={'accepted':verdict.accepted,'reason':verdict.reason},
            paired_wake_id=wakes[index%len(wakes)]['id'],command_vad_endpoints=len(endpoints),
            compute_budget=budget,complete_compute_budget=complete))
    return dict(command_n=len(commands),wake_n=len(wakes),commands=commands,wakes=wakes,
        component_latency_ms={key:summary(values) for key,values in components.items()},
        command_frontend_wall_ms=quantiles([row['timing']['vcm_frontend']['wall_ms'] for row in commands]),
        command_engine_decision_wrapper_wall_ms=quantiles([row['timing']['vcm_engine_decision_and_wrapper']['wall_ms'] for row in commands]),
        command_compute_budget_ms=summary([row['compute_budget'] for row in commands]),
        complete_compute_budget_ms=summary([row['complete_compute_budget'] for row in commands]),
        complete_compute_rtf={key:quantiles([row['complete_compute_budget'][key] for row in commands]) for key in ('wall_rtf','cpu_rtf')})


def profile(*,root,bundle,wake,owner,vad_folder,output,threads=1,engine='ncnn'):
    if type(threads) is not int or not 1<=threads<=8 or engine not in ('ncnn','onnxruntime'):
        raise ValueError('Use threads 1..8 and engine ncnn or onnxruntime')
    root,bundle,wake,owner,vad_folder,output=map(lambda value:Path(value).resolve(),(root,bundle,wake,owner,vad_folder,output))
    if any(output.is_relative_to(folder) for folder in (bundle,wake,owner,vad_folder)):
        raise ValueError('Resource output must be outside immutable model folders')
    sizes,parameters,paths=inventory(bundle,wake,owner,vad_folder,root)
    groups,input_paths=inputs(root);paths.update(input_paths);paths.update(source_bindings(root))
    versions=software_versions()
    bindings=dict(files={name:{'sha256':digest(path),'bytes':path.stat().st_size} for name,path in sorted(paths.items())},
        engine=engine,threads=threads,sample_rate=SAMPLE_RATE,samples_per_component=SAMPLES,
        prefix_cadence_samples=PREFIX_SAMPLES,warmup_source_samples=WARMUP_SAMPLES,
        frontend_thread_environment={key:os.environ.get(key) for key in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},
        software_versions=versions)
    if output.exists():
        previous=read(output)
        if previous.get('bindings')!=bindings:
            raise ValueError('Preserved resource receipt bindings changed; use a new --output for new measurements: '+str(output))
        return previous
    host=hardware();before=telemetry(pi=host['physical_pi_measured'])
    objects,loads,memory=load_components(bundle,wake,owner,vad_folder,threads,engine)
    result=measurements(objects,groups)
    memory.append({'stage':'after_measurement',**memory_usage()})
    if any(digest(path)!=bindings['files'][name]['sha256'] for name,path in paths.items()):
        raise ValueError('Resource input/source/model binding changed during profiling; no receipt was written')
    receipt=dict(schema_version=1,utc=datetime.now(timezone.utc).isoformat(),evidence_kind='bounded_cpu_component_resource_profile',
        hardware=host,physical_pi_measured=host['physical_pi_measured'],live_microphone_measured=False,
        human_audio_measured=False,actions_executed=False,acoustic_quality_approved=False,ownership_experimental=True,
        bindings=bindings,parameters=parameters,sizes=sizes,measurements=result,session_load=loads,
        source_context={'module_root':str(ROOT),'input_root':str(root),
            'scope':'Observed paths for this measurement only, not replay dependencies. Code hashes bind the '
                    'executing module checkout; --root selects input/configuration files. Fresh execution resolves '
                    'its current module root and explicitly configured .cache mount.'},
        total_session_load=total(list(loads.values())),
        memory={'samples':memory,'resident_sessions':{'vcm':1,'wake':1,'owner':1,'silero':2},
            'peak_process_rss_mib':max((row['peak_rss_mib'] for row in memory if row['peak_rss_mib'] is not None),default=None),
            'scope':'Whole current Python process including libraries, all five sessions, retained PCM and result storage; '
                    'lifetime peak, not model-only RSS. Python worker IPC, Node, browser and OS costs excluded.'},
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        telemetry_before=before,telemetry_after=telemetry(pi=host['physical_pi_measured']),software_versions=versions,
        thread_scope={'vcm':threads,'owner':threads,'wake':1,'silero_each':1,'vulkan':False},
        timing_scope='Actual serial CPU calls without sleeping. Complete budget sums one final VCM, every pre-final '
            '8000-sample prefix, command VAD including computed silence tail, wake reference evidence, command evidence, '
            'all actual continuity passes, SpeakerSession arm/finish/application intersection, and active wake+VAD streaming. '
            'Wake startup warmup is measured separately. Nested frontend/window timings are not added twice.',
        timing_excludes=['audio file I/O/decoding/hash checks','utterance elapsed time','endpoint waiting elapsed time',
                         'microphone/capture scheduling','worker IPC','providers/network/actions','Node/browser/UI'],
        interpretation='Preview max-cadence serial compute budget, not live concurrency or user latency. RTF divides '
            'compute by original paired command+wake duration; command endpoint silence and 512 alignment compute are '
            'included but add no credited speech duration. Warmup is 32000 zeros plus explicit final-frame alignment. '
            'First model calls are included. Synthetic command/wake pairs are chosen by fixed manifest order, not verified '
            'same-speaker ownership pairs; verdicts are diagnostic, not acoustic accuracy or authentication evidence.',
        input_policy='First eight distinct hash-bound own Qwen3 synthetic positive personal commands and first eight '
                     'generated voice-test positive wakes; Apache-2.0 generation source receipt. No mixing/resampling/truncation.')
    payload=json.dumps(receipt,indent=2,allow_nan=False)+'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as stream:stream.write(payload)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    for name in ('bundle','wake','owner','vad-folder','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--engine',choices=['ncnn','onnxruntime'],default='ncnn')
    args=parser.parse_args(argv);root=args.root.resolve();cache=root/'.cache/iris-release-v1'
    output=args.output or root/'experiments/iris-release-v1/pipeline-resources.json'
    existed=output.exists()
    try:
        result=profile(root=root,bundle=args.bundle or cache/'diversity/release-export',wake=args.wake or cache/'wake/export',
            owner=args.owner or cache/'owner/export',vad_folder=args.vad_folder or cache/'inputs/vad',
            output=output,threads=args.threads,engine=args.engine)
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))
    print(json.dumps({'output':str(output),'reused':existed,'actions_executed':False,
                      'measurement_host':result.get('hardware',{}).get('measurement_host')},allow_nan=False))
    return result


if __name__=='__main__':main()
