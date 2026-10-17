"""Direct-command assets for the existing publisher and resident application."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.runtime.mvp_static import Runtime,digest

ROOT=Path(__file__).resolve().parents[1]


def _package(bundle,output,*,wake_folder,vad_folder=None,extra_files=None,owner_folder=None,release=False):
    from scripts.archive_io import write_archive
    from scripts.speaker_publish import application_files,runtime_identity
    from scripts.runtime.mvp_wake import HeyIris
    if wake_folder is None:raise ValueError('A trained, locked Hey Iris export is required')
    bundle=Path(bundle);wake_folder=Path(wake_folder)
    runtime=Runtime(bundle,engine='ncnn');HeyIris(wake_folder,engine='ncnn')
    entries=application_files('full' if release else 'inference','skip')
    for name in ('scripts/__init__.py','scripts/runtime/__init__.py','vcm/__init__.py'):entries[name]=b''
    for name in ('scripts/runtime/mvp_static.py','scripts/runtime/mvp_wake.py','scripts/runtime/pcv_schema.py',
                 'scripts/runtime/wake_models.py','scripts/runtime/wake_gate.py','scripts/runtime/pcv_benchmark_adapter.py','vcm/frontend.py'):
        path=ROOT/name
        if path.exists():entries[name]=path.read_bytes()
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
    for name,data in (extra_files or {}).items():
        if name in entries:raise ValueError('Candidate extra collides: '+name)
        entries[name]=data
    metadata=dict(format='smart-speaker-v1',target='inference',startup_action='candidate-only',
        runtime_id=runtime_identity(entries),bundle_id=digest(bundle/'bundle.json'),
        model_binding=runtime.recipe['binding'],output_schema_version=3,runtime_api_version=2,
        diagnostic=True,acceptance_enabled=True,qualified=False,actions_enabled=False,
        profile='benchmark-compatible',owner_gating_enabled=False,physical_pi_measured=False,live_microphone_measured=False)
    if release:
        if owner_folder is None:raise ValueError('The release requires the independent owner backend')
        from scripts.runtime.iris_owner import OwnerBackend
        owner_folder=Path(owner_folder);owner=OwnerBackend(owner_folder,threads=1)
        owner_meta=json.loads((owner_folder/'owner.json').read_text())
        for name in ('owner.json',*owner_meta['files']):
            entries['models/default/owner/'+name]=(owner_folder/name).read_bytes()
        for name in ('scripts/runtime/iris_owner.py','scripts/runtime/speaker_session.py','scripts/runtime/audio.py'):
            entries[name]=(ROOT/name).read_bytes()
        entries['deployment.json']=(json.dumps(dict(target='inference',benchmarks='skip',runtime_api_version=2,
            actions_enabled=True,default_mode='serve',profile='owner-protected',qualified=False),indent=2)+'\n').encode()
        entries['config.example.toml']=b'''[inference]
mode = "demo"
engine = "ncnn"
threads = 1
profile = "owner-protected"

[actions]
enabled = true

[ui]
debug = false
readout = false

[providers]
ha_enabled = false
ha_url = ""
ha_entity_ids = ""
local_media_dir = "media"
data_dir = "state"
timezone = "Asia/Manila"

[bindings.lights]
# "light.ceiling" = "light.your_ceiling"
# "light.all" = ["light.your_ceiling", "light.your_kitchen"]
[bindings.songs]
# "song.enemy" = "media-source://media_source/local/enemy.mp3"
[bindings.climate]
# default = "climate.your_room"
[bindings.weather]
# default = "weather.home"
[bindings.media_player]
# Leave media unconfigured until you select a provider and its required binding.
# provider = "home_assistant" # Explicit alternatives: local, spotify.
# entity_id = "media_player.your_speaker"
[bindings.contacts]
# "contact.home" = "script.call_home"
# "contact.mom" = "script.call_mom"
[bindings.messages]
# default = "script.send_preset_message"
'''
        entries['START-HERE.txt']=b'''Iris: local finite-command smart speaker

Extract into a new directory. Use Raspberry Pi OS 64-bit on Pi4/5.
First dependency setup needs internet; recognition then runs locally.

python3 start.py --list-mics
python3 start.py --mode serve --mic INDEX --benchmarks skip

Choose the actual microphone index. Say Hey Iris, wait for command-ready,
then speak one command. The experimental owner check compares this wake
speaker with the command, rejects overlap/mismatch, then resets.
The UI displays tentative intent/slots; only final accepted commands execute.
Supply your HA URL/token, allowed entities and finite bindings in config.toml
and private .env. Unconfigured targets do not execute or invent replacements.

Command-only replay and benchmarks never execute actions:
python3 start.py --profile benchmark-compatible --wav command.wav

See docs/HANDOFF-iris-rpi-v1.md and MODEL_CARD.md for configuration, measured
results, operator tests and limitations. Physical Pi performance and original
human wake-to-command qualification remain unmeasured in this release.
'''
        # Bounded original-designed synthetic diagnostic samples only. Restricted
        # human corpora remain external, pinned training/benchmark inputs.
        from scripts.iris_release import resolve_audio
        sample_rows=[]
        sample_source=ROOT/'.cache/iris-release-v1/inputs/base-views/personal-test.jsonl'
        for line in sample_source.read_text().splitlines():
            row=json.loads(line)
            if not row['synthetic']:raise ValueError('Pi smoke sample must be our permitted synthetic recording')
            audio=resolve_audio(row,root=ROOT)
            if digest(audio)!=row['sha256']:raise ValueError('Changed diagnostic sample')
            name='audio/'+hashlib.sha256(row['id'].encode()).hexdigest()[:24]+'.wav'
            entries['samples/'+name]=audio.read_bytes()
            sample_rows.append(dict(id=row['id'],audio=name,sha256=row['sha256'],
                expected=runtime.catalog[row['class_index']]['command'] if row['class_index'] else None,
                synthetic=True,source=row['source']))
        entries['samples/commands.jsonl']=''.join(json.dumps(r)+'\n' for r in sample_rows).encode()
        metadata.update(startup_action='install-model',profile='owner-protected',
            owner_gating_enabled=True,ownership_experimental=True,actions_enabled=True,
            owner_model_sha256=owner.model_id if hasattr(owner,'model_id') else owner_meta.get('model_id'),
            runtime_id=runtime_identity(entries))
    receipt=write_archive(entries,output,metadata)
    receipt.update(model_sha256=runtime.recipe['binding']['model_sha256'],parameters=runtime.meta['parameters'],
        qualified=False,profile=metadata['profile'],actions_enabled=release,files=len(entries))
    Path(output).with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def package_candidate(bundle,output,*,wake_folder,vad_folder=None,extra_files=None):
    return _package(bundle,output,wake_folder=wake_folder,vad_folder=vad_folder,extra_files=extra_files)


def package_release(bundle,output,*,wake_folder,owner_folder,vad_folder=None,extra_files=None):
    return _package(bundle,output,wake_folder=wake_folder,owner_folder=owner_folder,
        vad_folder=vad_folder,extra_files=extra_files,release=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--wake',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vad-folder',type=Path,help='Existing Silero asset directory; hash must match the pinned artifact receipt')
    args=parser.parse_args();print(json.dumps(package_candidate(args.bundle,args.output,wake_folder=args.wake,vad_folder=args.vad_folder)),flush=True)


if __name__=='__main__':main()
