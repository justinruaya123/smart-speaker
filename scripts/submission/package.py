"""Package the selected graph and prebuilt UI using the shared runtime allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile

from .common import ROOT, read, runtime_entries, safe_name, save, sha, write_archive


MODEL_PREFIX = 'submission/models/full/'
PARAMETERS = 2066831
GRAPH_SHA256 = '3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e'


START_HERE = '''2M smart-speaker inference release

Primary VCM: 2,066,831 parameters, control mixed epoch 32, INT8 ONNX Runtime CPU.
Demo gates are fixed .95 intent / .90 required slot. Strict qualification failed;
actions and providers are disabled by default.
Recognition returns finite commands or rejection. The GUI supports explicitly
configured Home Assistant and generic Spotify play/resume; CLI stays action-free.

On Windows: python inference.py
On Raspberry Pi: python3 inference.py

Choose CLI or UI, then actual microphone and speaker indexes. CLI is silent.
UI serves localhost and speaks actual tool results using local OS text-to-speech.
Say "Hey Iris" or the "Hey Jarvis" fallback, wait for command-ready, then speak.
In SpeakerUI open Debug > Inference and use the Wake word selector for Iris with
Jarvis fallback (default), experimental Iris only, or Jarvis only. Selection
resets capture; fallback accepts either actual phrase. If Iris cannot load,
fallback mode explicitly reports Jarvis active. Iris was trained separately:
82,114 wake parameters, 12/16 synthetic positive test streams, no real-human
positive wake qualification. It does not change the selected 2M command model.

For GUI integrations, copy SpeakerUI/.env.example to a private .env in the ZIP
root and set HA_ENABLED / HA_URL / HA_TOKEN or Spotify credentials and flags.
Set ACTIONS_ENABLED=true explicitly for device controls. The GUI defaults to:
HA_ENTITY_IDS=light.hue_color_lamp_1,light.dining_light,camera.dining_area_main_camera_tc70_mainstream
Spotify voice scope is generic "play music" on the active/configured Connect
device. No song/artist search or playlist selection is supplied by this release.
Use http://127.0.0.1:3000/auth/spotify/callback for the default GUI port.
Finite acoustic light labels require explicit [bindings.lights] in config.toml;
the entity allowlist alone does not assign acoustic labels to physical lamps.
Keep .env and tokens private. Benchmark and reproduction never control devices.

Python 3.12/3.13 and a 64-bit OS are required. First setup uses the minimal uv lock.
On Pi OS, install CPython venv support, libportaudio2 and libsndfile1 through your OS.
First UI startup downloads pinned checksummed Node 24.13 only if absent. The UI
and production JavaScript dependencies are already built: no pnpm or Pi compilation.
There is no training corpus, training checkpoint, trainer, test suite or neural TTS model.
macOS uses built-in say; Windows uses System.Speech. On Linux/Pi install espeak-ng.
Copy SpeakerUI/.env.example to root .env and set OPENWEATHER_ENABLED=true and
OPENWEATHER_API_KEY for live weather. OPENWEATHER_LAT/LON default to Manila.
Weather needs internet; voice recognition, time, timers, alarms and reminders are local.
Time and alarms use Asia/Manila. One timer and one alarm are active at a time;
they and reminders last for this running session. Keep the server running for alerts.
For thermostat control set HA_THERMOSTAT_ENTITY and include that climate.* entity
in HA_ENTITY_IDS; the device must report Celsius and an allowed temperature range.
General spoken lights default to light.hue_color_lamp_1; config.toml bindings may
override that mapping. Calls open camera.dining_area_main_camera_tc70_mainstream
(video only); override with HA_CAMERA_ENTITY. Targets must be allowlisted.
Messages call notify.send_message on HA_NOTIFY_ENTITY (default notify.jus_phone),
with HA_NOTIFY_TITLE=Iris and HA_NOTIFY_MESSAGE=This works. Configure these in .env.
The model has no dictated-message content slot. Phone notification acceptance is
not proof of delivery. On macOS the same message is also submitted as a local
desktop notification; OS notification permission/settings control its display.
For a remote MacBook, set HA_MACBOOK_NOTIFY_ENTITY to its notify.* entity.
On python.org macOS Python, run /Applications/Python 3.13/Install Certificates.command
if setup reports CERTIFICATE_VERIFY_FAILED. Certificate verification stays enabled.
Recognition and reply playback run locally after setup. No physical Pi result is
implied by this archive; evaluate on the actual device separately.

Silent checks (no capture, playback or browser):
  python inference.py --check
  python inference.py --mode cli --wav /path/to/mono-16khz.wav
  python inference.py --mode ui --no-audio --no-browser

SUBMISSION-MANIFEST.json hashes every file. Release receipt hashes the complete ZIP.
Keep third-party notices. Project code licensing does not grant recording/model
rights. Wake assets retain their separate CC BY-NC-SA / MIT terms. Fixed replies
are synthetic recognition acknowledgments; they do not claim an action succeeded.
'''


def validate_entries(entries):
    """Verify payload identity and exclude research/private files before writing."""
    forbidden_parts = {'experiments', '.cache', '.venv', '.git',
                       'checkpoints', 'training', 'tests', '__pycache__'}
    for name in entries:
        parts = Path(name).parts
        if (not safe_name(name) or forbidden_parts.intersection(parts) or parts[0] in ('data','datasets')
                or name.endswith(('.pt','.pth','.ckpt','.pyc','.safetensors'))
                or '/test_' in name or name in ('train.py','reproduce.py','benchmark.py')
                or name.startswith('scripts/train/') or '/responses.py' in name
                or any((p == '.env' or p.startswith('.env.')) and p != '.env.example' for p in parts)
                or (name.endswith('.wav') and not name.startswith('submission/responses/audio/'))):
            raise ValueError('Not an inference-only member: '+name)
    project = tomllib.loads(entries['pyproject.toml'].decode('utf-8'))
    requirements = json.dumps(project).lower()
    lock = entries['uv.lock'].decode('utf-8').lower()
    if any(value in requirements or f'name = "{value}"' in lock
           for value in ('torch','qwen','transformers','training','reporting','matplotlib')):
        raise ValueError('Inference archive contains a training dependency')
    prefix = MODEL_PREFIX
    deployment = json.loads(entries[prefix+'deployment.json'])
    for name, expected in deployment['files'].items():
        path = prefix+name
        if path not in entries or hashlib.sha256(entries[path]).hexdigest() != expected:
            raise ValueError('Packaged model identity mismatch: '+path)
    recipe = json.loads(entries[prefix+'recipe.json'])
    if (recipe.get('parameters',PARAMETERS) != PARAMETERS
            or recipe['binding']['model_sha256'] != GRAPH_SHA256
            or GRAPH_SHA256 != hashlib.sha256(entries[prefix+'model.onnx']).hexdigest()):
        raise ValueError('Selected 2M model identity mismatch')
    if any(name.startswith('submission/models/') and not name.startswith(prefix) for name in entries):
        raise ValueError('Inference-only release must contain only the selected 2M model')
    iris='submission/wake/hey-iris/'
    if iris+'iris.onnx' in entries:
        meta=json.loads(entries[iris+'iris.json'])
        actual=hashlib.sha256(entries[iris+'iris.onnx']).hexdigest()
        if actual!='4cc32fcb2cf20e7bf1310dcbe1d4f4ad0a9ad99d35575d0e31e3c5f5a4db95ac' or meta['files']['iris.onnx']!=actual:
            raise ValueError('Hey Iris model identity mismatch')
    return {name:hashlib.sha256(data).hexdigest() for name,data in sorted(entries.items())}


def _tree(entries, base, destination, *, ancestors=()):
    """Dereference only the explicit production tree; preserve portable npm paths."""
    base = Path(base)
    resolved = base.resolve(strict=True)
    if resolved in ancestors:
        return  # A package dependency cycle resolves through an existing ancestor.
    for file in sorted(base.iterdir()):
        if (file.name in ('__pycache__','.bin','test','tests','__tests__','spec','fixtures','integration',
                          'benchmark','benchmarks','example','examples','docs','.github')
                or file.name in ('test.js','test.mjs','bench.js','benchmark.js','example.js','example.mjs')
                or file.name.startswith(('test-','bench-','benchmark-','example-'))
                or file.suffix == '.pyc' or '.test.' in file.name or '.spec.' in file.name or '.tst.' in file.name):
            continue
        name = destination+'/'+file.name
        if file.is_dir():
            _tree(entries,file,name,ancestors=(*ancestors,resolved))
        elif file.is_file():
            entries[name] = file.read_bytes()


def gather_entries(*, ui_root=None, node_modules=None):
    entries = runtime_entries()
    if 'LICENSE' not in entries:
        raise ValueError('Include the authorized project code license before release')
    scope = ROOT/'docs/SUBMISSION-LICENSES.md'
    entries['submission/licenses/PROJECT-SCOPE-NOTICE.txt'] = (
        b'Source: docs/SUBMISSION-LICENSES.md in the reviewed source checkout.\n\n'+scope.read_bytes())
    sources = ('inference.py','scripts/submission/inference.py','scripts/submission/worker.py',
               'scripts/submission/response_runtime.py','scripts/submission/speech.py','scripts/submission/wake.py','scripts/runtime/speaker_worker.py',
               'scripts/runtime/speaker_config.py','scripts/runtime/speaker_tools.py')
    for name in sources:
        entries[name] = (ROOT/name).read_bytes()
    iris=ROOT/'submission/wake/hey-iris'
    for name in ('iris.json','iris.onnx'):
        entries['submission/wake/hey-iris/'+name]=(iris/name).read_bytes()
    ui = Path(ui_root or ROOT/'SpeakerUI')
    for name in ('package.json','pnpm-lock.yaml','UPSTREAM.json','.env.example'):
        entries['SpeakerUI/'+name] = (ui/name).read_bytes()
    for name in ('dist/index.html','dist-server/server/main.js'):
        if not (ui/name).is_file():
            raise ValueError('Build the UI on the workstation before packaging: '+name)
    for folder in ('dist','dist-server'):
        _tree(entries,ui/folder,'SpeakerUI/'+folder)
    production = Path(node_modules or ROOT/'.cache/submission/ui-production-hoisted/node_modules')
    if not production.is_dir():
        raise ValueError('Missing locked production JavaScript dependencies: '+str(production))
    # pnpm's physical store is copied through logical package paths. Node requires
    # regular files after ZIP extraction; host symlinks never cross into the ZIP.
    for file in sorted(production.iterdir()):
        if file.name.startswith('.'):
            continue
        if file.is_dir():
            _tree(entries,file,'SpeakerUI/node_modules/'+file.name)
    required = ('fastify','@fastify/static','@fastify/websocket','ws')
    for name in required:
        if 'SpeakerUI/node_modules/'+name+'/package.json' not in entries:
            raise ValueError('Missing production dependency: '+name)
    for path in ui.glob('*LICENSE*'):
        entries['submission/licenses/SpeakerUI-'+path.name] = path.read_bytes()
    # Acknowledgment generation is separate research. Only its checked WAV outputs
    # and metadata enter the inference archive; no synthesizer source or weights.
    response_manifest = ROOT/'submission/responses/manifest.json'
    responses = read(response_manifest)
    from scripts.submission.response_runtime import validate_manifest
    validate_manifest(responses)
    entries['submission/responses/manifest.json'] = response_manifest.read_bytes()
    for row in responses['responses']:
        relative = 'submission/responses/'+row['path']
        source = ROOT/relative
        if not source.is_file(): source = ROOT/'.cache/submission/responses'/row['path']
        if source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
            raise ValueError('Changed acknowledgment audio: '+row['id'])
        entries[relative] = source.read_bytes()
    entries['START-HERE.txt'] = START_HERE.encode('utf-8')
    validate_entries(entries)
    return entries


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Repeated archive path')
        manifest = json.loads(archive.read('SUBMISSION-MANIFEST.json'))
        if set(names) != {*manifest['files'], 'SUBMISSION-MANIFEST.json'}:
            raise ValueError('Inference archive inventory mismatch')
        entries = {name:archive.read(name) for name in manifest['files']}
        actual = validate_entries(entries)
        if actual != manifest['files']:
            raise ValueError('Inference archive checksum mismatch')
        if archive.testzip() is not None:
            raise ValueError('Inference archive CRC failed')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/final-submission/smart-speaker-inference-2m-final.zip')
    parser.add_argument('--ui-root',type=Path)
    parser.add_argument('--node-modules',type=Path)
    parser.add_argument('--verify-only',type=Path)
    parser.add_argument('--no-update-inventory',action='store_true',help='Keep the checked-in historical inventory unchanged during source/CI builds')
    args = parser.parse_args(argv)
    if args.verify_only:
        manifest = verify_archive(args.verify_only)
        print(f"Verified {len(manifest['files'])} inference-only files; physical Pi tests remain pending.")
        return
    entries = gather_entries(ui_root=args.ui_root,node_modules=args.node_modules)
    inventory = validate_entries(entries)
    recipe = json.loads(entries[MODEL_PREFIX+'recipe.json'])
    receipt = write_archive(args.output,entries,purpose='inference-only',
                            model_binding=recipe['binding'],parameters=PARAMETERS,
                            qualified=False,actions_enabled=False,physical_pi_measured=False,
                            live_microphone_measured=False,ui_prebuilt=True)
    verify_archive(args.output)
    if args.output.is_relative_to(ROOT): receipt['path'] = args.output.relative_to(ROOT).as_posix()
    receipt.update(model_binding=recipe['binding'],parameters=PARAMETERS,files=inventory,
                   qualified=False,actions_enabled=False,physical_pi_measured=False,ui_prebuilt=True,
                   production_dependencies='Locked JavaScript sources, portable regular files; no build tools',
                   entrypoint='python inference.py',distribution='Local reviewed artifact; no public release URL claimed')
    save(args.output.with_suffix('.receipt.json'),receipt)
    inventory_path = ROOT/'submission/release/inference.json'
    # This generated inventory describes the latest reviewed build; ZIPs remain immutable.
    if not args.no_update_inventory:
        inventory_path.parent.mkdir(parents=True,exist_ok=True)
        inventory_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'files'},indent=2))


if __name__ == '__main__':
    main()
