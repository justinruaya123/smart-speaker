"""Direct-command assets for the existing publisher and resident application."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.runtime.mvp_static import Runtime,digest

ROOT=Path(__file__).resolve().parents[1]


def package_candidate(bundle,output,*,wake_folder,vad_folder=None,extra_files=None):
    from scripts.pcv_final_delivery import write_archive
    from scripts.speaker_publish import application_files,runtime_identity
    from scripts.runtime.mvp_wake import HeyIris
    if wake_folder is None:raise ValueError('A trained, locked Hey Iris export is required')
    bundle=Path(bundle);wake_folder=Path(wake_folder)
    runtime=Runtime(bundle,engine='ncnn');HeyIris(wake_folder,engine='ncnn')
    entries=application_files('inference','skip')
    for name in ('scripts/__init__.py','scripts/runtime/__init__.py','vcm/__init__.py'):entries[name]=b''
    for name in ('scripts/runtime/mvp_static.py','scripts/runtime/mvp_wake.py','scripts/runtime/pcv_schema.py',
                 'scripts/runtime/wake_models.py','scripts/runtime/pcv_benchmark_adapter.py','vcm/frontend.py','vcm/catalog.py'):
        path=ROOT/name
        if path.exists():entries[name]=path.read_bytes()
    entries['wake_gate.py']=(ROOT/'experiments/pcv-synthetic-v2/wake_gate.py').read_bytes()
    for name in ('bundle.json',*runtime.meta['files']):entries['models/default/'+name]=(bundle/name).read_bytes()
    for name in ('engine.json','model.ncnn.param','model.ncnn.bin','position-frequency.json'):
        entries['models/default/ncnn/'+name]=(bundle/'ncnn'/name).read_bytes()
    iris=json.loads((wake_folder/'iris.json').read_text())
    for name in ('iris.json',*iris['files']):entries['models/default/wake/'+name]=(wake_folder/name).read_bytes()
    source=ROOT/'artifacts';vad=json.loads((source/'wake-models.json').read_text())
    vad_source=Path(vad_folder) if vad_folder is not None else source
    record=vad['files']['silero-vad.onnx']
    if digest(vad_source/'silero-vad.onnx')!=record['sha256']:raise ValueError('VAD source identity mismatch')
    entries['models/default/wake/silero-vad.onnx']=(vad_source/'silero-vad.onnx').read_bytes()
    entries['models/default/wake/wake-models.json']=(json.dumps({'files':{'silero-vad.onnx':record}},indent=2)+'\n').encode()
    for name in ('silero-LICENSE.txt',):entries['licenses/'+name]=(source/name).read_bytes()
    entries['config.example.toml']=b'''[inference]\nmode = "demo"\nengine = "ncnn"\nthreads = 1\n\n[actions]\nenabled = false\n\n[ui]\ndebug = true\nreadout = false\n'''
    entries['deployment.json']=(json.dumps(dict(target='inference',benchmarks='skip',runtime_api_version=2,
        actions_enabled=False,default_mode='serve',profile='benchmark-compatible',qualified=False),indent=2)+'\n').encode()
    entries['START-HERE.txt']=b'''Hey Iris finite-command candidate (unqualified; actions disabled)

Extract into a new directory; preserve your existing installation.
Python 3.11-3.13; Linux ARM64 on Pi4/5. First dependency setup needs internet.
Recognition uses local ncnn (command + Hey Iris), local Silero VAD and typed JSON.
No ASR/STT, phonetic scorer, LLM, cloud inference, trainer or TTS model ships.
The ORT graphs are local parity/reference assets. All their bytes count.

python3 start.py --list-mics
python3 start.py --mode serve --mic INDEX --benchmarks skip

Choose the actual listed microphone. Say Hey Iris, pause until command_ready,
then speak one command. One wake permits one command and resets. config.toml
defaults to action-free demo mode; acceptance is empirical, not qualification.

Command-only or wake+command WAV replay (no device actions):
python3 start.py --wav command.wav
python3 start.py --wav wake-and-command.wav --wake-wav

After setup, the resident JSONL worker can also run directly:
PCV_BENCHMARK_LOG="$HOME/vcm_benchmark/iris.log" .venv/bin/python -u -m scripts.runtime.speaker_worker --bundle models/default --engine ncnn --profile benchmark-compatible

The unchanged official benchmark runs from benchmarks/official on the tester's
laptop, using its original instructions. Its fixed holdout/audio dependencies
are separate inputs, not shipped training or evaluation audio. The physical
Pi loudspeaker/microphone/no-wake trials and telemetry remain NOT MEASURED.

Owner-protected mode is unavailable: no verified human wake/command ownership
backend or overlap evidence exists. --profile owner-protected fails closed.
Do not treat synthetic wake recall or a CLI replay as human/Pi qualification.
'''
    from scripts.runtime.pcv_schema import baseline_cases
    entries['docs/BASELINE-93.json']=(json.dumps(baseline_cases(),indent=2)+'\n').encode()
    bench=ROOT/'.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191'
    for path in bench.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.csv','.md','.txt'):
            entries['benchmarks/official/'+path.relative_to(bench).as_posix()]=path.read_bytes()
    for name,data in (extra_files or {}).items():
        if name in entries:raise ValueError('Candidate extra collides: '+name)
        entries[name]=data
    metadata=dict(format='smart-speaker-v1',target='inference',startup_action='candidate-only',
        runtime_id=runtime_identity(entries),bundle_id=digest(bundle/'bundle.json'),
        model_binding=runtime.recipe['binding'],output_schema_version=3,runtime_api_version=2,
        diagnostic=True,acceptance_enabled=True,qualified=False,actions_enabled=False,
        profile='benchmark-compatible',owner_gating_enabled=False,physical_pi_measured=False,live_microphone_measured=False)
    receipt=write_archive(entries,output,metadata)
    receipt.update(model_sha256=runtime.recipe['binding']['model_sha256'],parameters=runtime.meta['parameters'],
        qualified=False,profile='benchmark-compatible',actions_enabled=False,files=len(entries))
    Path(output).with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    from scripts.pcv_final_delivery import package
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--wake',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vad-folder',type=Path,help='Existing Silero asset directory; hash must match the pinned artifact receipt')
    args=parser.parse_args();print(json.dumps(package(args.bundle,args.output,diagnostic=True,wake_folder=args.wake,vad_folder=args.vad_folder)),flush=True)


if __name__=='__main__':main()
