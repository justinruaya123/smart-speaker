"""Stdlib release validator for the prebaked Iris site; no scientific dependencies.

Validates local HTML/CSS dependencies, exact finite data, real render hashes,
README diagrams/citation order and Git-tracked model bytes. --remote additionally
reads back the exact public GitHub commit's ONNX anonymously; no publish action.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.iris_publication_data import build_publication, MODEL_HASHES
from presentation.build_readme import render as render_readme


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.chapters = 0
        self.videos = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'data-chapter' in attrs:
            self.chapters += 1
        if 'data-background-video' in attrs:
            self.videos += 1
        for key in ('src', 'href', 'poster', 'data-src', 'data-background-video'):
            if attrs.get(key):
                resource = key != 'href' or tag in {'link', 'script'}
                self.links.append((attrs[key], resource))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_target(site, source, reference, resource):
    reference = reference.replace('\\', '/')
    parts = urlsplit(reference)
    if parts.scheme in ('data', 'mailto') or not parts.path:
        return None
    if parts.scheme or parts.netloc:
        if resource:
            raise ValueError(f'external viewer dependency: {reference}')
        if parts.scheme != 'https':
            raise ValueError('external navigation must use HTTPS')
        return None
    target = (site / unquote(parts.path).lstrip('/') if parts.path.startswith('/') else source.parent / unquote(parts.path)).resolve()
    if not target.is_relative_to(site.resolve()):
        raise ValueError(f'viewer dependency escapes site: {reference}')
    return target


def inventory(site):
    return {p.relative_to(site).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(site.rglob('*')) if p.is_file() and p.name != 'asset-manifest.json'}


def validate_site(root=ROOT, require_integration=False):
    site = root / 'docs/site'
    data = json.loads((site / 'assets/iris.json').read_text(encoding='utf-8'))
    if data != build_publication(root):
        raise ValueError('publication JSON is stale or not backed by frozen inputs')
    missing = []
    count = 0
    main = None
    for source in sorted(site.rglob('*')):
        if source.is_symlink():
            raise ValueError('site cannot publish a symlink')
        if not source.is_file():
            continue
        count += 1
        if source.suffix.lower() not in {'.html', '.css'}:
            continue
        body = source.read_text(encoding='utf-8')
        if source.suffix == '.html':
            parser = References()
            parser.feed(body)
            refs = parser.links
            if source.name == 'index.html':
                main = parser
            elif source.stem in ('pipeline', 'model') and parser.videos != 5:
                raise ValueError('real five-stage Manim Slides export missing')
        else:
            refs = [(value.strip(' \"\''), True) for value in re.findall(r'url\(([^)]+)\)', body)]
        for reference, resource in refs:
            target = local_target(site, source, reference, resource)
            if target is not None and not target.is_file():
                missing.append(f'{source.relative_to(site)} → {reference}')
    if missing:
        raise ValueError(f'missing static assets: {missing}')
    if main is None or main.chapters != 6:
        raise ValueError('exactly six chapters required')
    receipt = json.loads((site / 'animations/render-receipt.json').read_text(encoding='utf-8'))
    if receipt['resolution'] != [1920, 1080] or receipt['fps'] != 30 or receipt['renderer'] != 'cairo':
        raise ValueError('production render receipt must be 1080p30 CPU Cairo')
    if receipt['source_sha256'] != sha(root / 'presentation/iris_scenes.py'):
        raise ValueError('Manim source changed without rendering')
    for name, record in receipt['assets'].items():
        video = site / 'animations' / record['video']
        if sha(video) != record['sha256'] or video.stat().st_size != record['video_bytes']:
            raise ValueError(f'actual rendered movie mismatch: {name}')
        if video.read_bytes()[4:8] != b'ftyp' or record['duration_seconds'] < 5:
            raise ValueError('not an actual rendered MP4')
        poster = (site / 'animations' / record['poster']).read_bytes()
        if poster[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', poster[16:24]) != (1920, 1080):
            raise ValueError('production poster must be actual 1080p PNG')
    assets = inventory(site)
    declared = json.loads((site / 'asset-manifest.json').read_text(encoding='utf-8'))
    if declared != {'format': 'iris-static-assets-v1', 'files': assets}:
        raise ValueError('static asset hash manifest is stale')
    readme = (root / 'README.md').read_text(encoding='utf-8')
    if readme != render_readme(data):
        raise ValueError('README is not generated from the same ten-case storyboard')
    diagrams = readme.count('```mermaid\n')
    if not readme.startswith('# Iris\n') or diagrams != 12:
        raise ValueError('README requires two architecture and ten use-case diagrams')
    if re.findall(r'^## (.+)$', readme, re.M)[-1] != 'Citation':
        raise ValueError('Citation must be the final README heading')
    for relative in ('docs/MODEL-CARD-iris-epoch44.md', 'docs/IRIS-GETTING-STARTED.md', 'docs/IRIS-CITATIONS.md'):
        if not (root / relative).is_file():
            raise ValueError(f'missing publication document: {relative}')
    for name, expected in MODEL_HASHES.items():
        relative = 'submission/models/optionb-tiny/' + name
        committed = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:' + relative])
        if committed.startswith(b'version https://git-lfs.github.com/spec') or hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError('frozen payload is absent, substituted or an LFS pointer in Git')
    pending = [p for p in ('inference.py', 'docs/RELEASE-iris-publication-v1.md', 'docs/REVERT-iris-publication-v1.md') if not (root / p).is_file()]
    if require_integration and pending:
        raise ValueError(f'release-owner integration paths missing: {pending}')
    return {'chapters': main.chapters, 'use_cases': len(data['use_cases']), 'readme_diagrams': diagrams,
            'assets_checked': count, 'missing_assets': missing, 'pi': data['benchmarks']['pi'],
            'integration_paths_missing': pending, 'frozen_git_payloads': len(MODEL_HASHES)}


def release_gate(root=ROOT):
    data = build_publication(root)
    verdict = data['wake']['verdict']
    if verdict.get('release_blocker') is not False or verdict.get('synthetic_acceptance_pass') is not True:
        baseline = data['wake']['baseline']
        candidate = data['wake']['companion_test']
        raise ValueError(f"wake acceptance gate failed: native {baseline['detected']}/{baseline['positive_n']}; candidate {candidate['detected']}/{candidate['positive_n']} with {candidate['false_wakes']}/{candidate['negative_n']} false wakes. Static/software QA is not an acoustic release pass.")
    return {'acoustic_release_gate': 'passed declared synthetic gate only', 'human_live_qualified': False}


def remote_model(root=ROOT, revision=None):
    revision = revision or subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('remote readback requires exact Git commit')
    url = f'https://raw.githubusercontent.com/justinruaya123/smart-speaker/{revision}/submission/models/optionb-tiny/model.onnx'
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    actual = hashlib.sha256(payload).hexdigest()
    if len(payload) != 1082940 or payload.startswith(b'version https://git-lfs') or actual != MODEL_HASHES['model.onnx']:
        raise ValueError('anonymous GitHub ONNX readback mismatch')
    return {'revision': revision, 'bytes': len(payload), 'sha256': actual, 'anonymous': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    parser.add_argument('--require-integration', action='store_true')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--release-gate', action='store_true', help='Independent acoustic gate, not static QA')
    parser.add_argument('--revision', help='Exact published Git commit for anonymous model readback')
    args = parser.parse_args()
    if args.release_gate:
        try:
            print(json.dumps(release_gate(ROOT), indent=2))
        except ValueError as failure:
            raise SystemExit(str(failure)) from failure
        return
    if args.write_manifest:
        target = ROOT / 'docs/site/asset-manifest.json'
        target.write_text(json.dumps({'format': 'iris-static-assets-v1', 'files': inventory(target.parent)}, indent=2) + '\n', encoding='utf-8', newline='\n')
    result = validate_site(ROOT, args.require_integration)
    if args.remote:
        result['github_model_readback'] = remote_model(ROOT, args.revision)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
