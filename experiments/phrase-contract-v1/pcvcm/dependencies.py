"""Target-specific wheel closure, immutable hashes and bundled license notices."""
from email.parser import BytesParser
import json
from pathlib import Path
import zipfile
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from .contract import file_hash, write_json

TARGET = {'python': '3.12', 'implementation': 'cp', 'machine': 'aarch64',
          'system': 'Linux', 'minimum_glibc': '2.28'}
REQUIRED = {'numpy', 'onnxruntime', 'einops', 'soundfile', 'sounddevice'}
PLATFORMS = ['manylinux_2_28_aarch64', 'manylinux_2_27_aarch64',
             'manylinux2014_aarch64', 'manylinux_2_17_aarch64']


def inspect_wheels(directory, required):
    tags = set(cpython_tags((3, 12), abis=['cp312'], platforms=PLATFORMS))
    tags.update(compatible_tags((3, 12), interpreter='cp312', platforms=PLATFORMS))
    environment = {**default_environment(), 'python_version': '3.12',
        'python_full_version': '3.12.13', 'platform_machine': 'aarch64',
        'platform_system': 'Linux', 'sys_platform': 'linux', 'os_name': 'posix',
        'implementation_name': 'cpython', 'platform_python_implementation': 'CPython', 'extra': ''}
    records, notices = {}, []
    for path in sorted(Path(directory).glob('*.whl')):
        name, version, _, wheel_tags = parse_wheel_filename(path.name)
        if not tags.intersection(wheel_tags):
            raise ValueError(f'Wheel target mismatch: {path.name}')
        if name in records or name in {'torch', 'onnx', 'onnxruntime-gpu'}:
            raise ValueError(f'Unexpected/duplicate runtime wheel: {name}')
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('Corrupt wheel')
            metadata_paths = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata_paths) != 1:
                raise ValueError('Ambiguous wheel metadata')
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            if canonicalize_name(metadata['Name']) != name or metadata['Version'] != str(version):
                raise ValueError('Wheel metadata identity mismatch')
            if '3.12.13' not in SpecifierSet(metadata.get('Requires-Python', '')):
                raise ValueError('Python target unsupported')
            licenses = [n for n in archive.namelist() if
                        any(word in Path(n).name.lower() for word in ('license', 'copying', 'notice')) and not n.endswith('/')]
            if not licenses and name == 'flatbuffers' and str(version) == '25.12.19':
                external = Path(__file__).resolve().parents[1] / 'configs/flatbuffers-25.12.19-LICENSE.txt'
                source = json.loads(external.with_suffix('.source.json').read_text())
                if file_hash(external) != source['sha256']:
                    raise ValueError('Pinned upstream license changed')
                notices.append(source['url'] + '\n' + external.read_text())
            elif not licenses:
                raise ValueError(f'Missing license notices: {name}')
            notices.append(f'\n=== {name} {version} ===\n')
            for n in licenses:
                notices.append(n + '\n' + archive.read(n).decode('utf-8', errors='replace'))
            records[name] = {'version': str(version), 'filename': path.name,
                'sha256': file_hash(path), 'bytes': path.stat().st_size,
                'requires_dist': metadata.get_all('Requires-Dist', [])}
    if not set(required) <= records.keys():
        raise ValueError('Missing required runtime dependency')
    for name, record in records.items():
        for text in record['requires_dist']:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            dependency = records.get(canonicalize_name(requirement.name))
            if not dependency or dependency['version'] not in requirement.specifier:
                raise ValueError(f'Unresolved dependency {name}: {text}')
    return records, '\n'.join(notices)


def lock_wheels(directory, output, required=REQUIRED):
    records, notices = inspect_wheels(directory, required)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    text = ''.join(f'{name}=={r["version"]} --hash=sha256:{r["sha256"]}\n'
                   for name, r in sorted(records.items()))
    (output / 'requirements.txt').write_text(text, encoding='utf-8', newline='\n')
    (output / 'THIRD_PARTY_NOTICES.txt').write_text(notices, encoding='utf-8', newline='\n')
    receipt = {'target': TARGET, 'required': sorted(required), 'wheels': records,
               'requirements_sha256': file_hash(output / 'requirements.txt'),
               'notices_sha256': file_hash(output / 'THIRD_PARTY_NOTICES.txt'),
               'system_dependencies': ['libportaudio2', 'libsndfile1'],
               'physical_installation': 'NOT_TESTED', 'qualified': False}
    write_json(output / 'dependencies.json', receipt)
    return receipt


def verify_lock(lock, wheels):
    lock = Path(lock)
    receipt = json.loads((lock / 'dependencies.json').read_text())
    if receipt['target'] != TARGET:
        raise ValueError('Dependency target changed')
    for name, key in [('requirements.txt', 'requirements_sha256'), ('THIRD_PARTY_NOTICES.txt', 'notices_sha256')]:
        if file_hash(lock / name) != receipt[key]:
            raise ValueError('Dependency lock hash mismatch')
    for r in receipt['wheels'].values():
        path = Path(wheels) / r['filename']
        if Path(r['filename']).name != r['filename'] or file_hash(path) != r['sha256']:
            raise ValueError('Wheel hash mismatch')
    records, _ = inspect_wheels(wheels, receipt['required'])
    if records != receipt['wheels']:
        raise ValueError('Wheel closure changed')
    return receipt
