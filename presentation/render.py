"""Actually render, export and probe Iris Manim Slides; isolated environment only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = {'IrisPipeline': 'pipeline', 'IrisModel': 'model'}


def run(args, cwd):
    print('+', ' '.join(map(str, args)), flush=True)
    temporary = cwd / 'temporary'
    temporary.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **{key: str(temporary) for key in ('TMPDIR', 'TEMP', 'TMP')}}
    subprocess.run([str(a) for a in args], cwd=cwd, check=True, env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', choices=('draft', 'production'), default='draft')
    args = parser.parse_args()
    production = args.quality == 'production'
    width, height, fps = (1920, 1080, 30) if production else (854, 480, 15)
    work = ROOT / '.cache/presentation-render' / args.quality
    work.mkdir(parents=True, exist_ok=True)
    executable = Path(sys.executable).parent / ('manim-slides.exe' if sys.platform == 'win32' else 'manim-slides')
    if not executable.is_file():
        raise SystemExit('Run using the isolated presentation Python with manim-slides installed')
    run([executable, 'render', '--renderer', 'cairo', '-r', f'{width},{height}', '--fps', fps,
         '--progress_bar', 'none', '--media_dir', work / 'media', ROOT / 'presentation/iris_scenes.py', *SCENES], work)
    destination = ROOT / 'docs/site/animations' if production else work / 'preview'
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {'format': 'iris-manim-render-v1', 'renderer': 'cairo', 'quality': args.quality,
               'versions': {p: importlib.metadata.version(p) for p in ('manim', 'manim-slides', 'manimpango')},
               'source_sha256': hashlib.sha256((ROOT / 'presentation/iris_scenes.py').read_bytes()).hexdigest(),
               'resolution': [width, height], 'fps': fps, 'assets': {}}
    for scene, name in SCENES.items():
        clips = [p for p in (work / 'media/videos').rglob(f'{scene}.mp4') if 'partial_movie_files' not in str(p)]
        if len(clips) != 1:
            raise ValueError(f'expected one actual scene movie, got {clips}')
        movie = destination / f'{name}.mp4'
        shutil.copyfile(clips[0], movie)
        run([executable, 'convert', '--offline', '--folder', work / 'slides', '-ccontrols=true',
             '-cprogress=true', '-cslide_number=true', '-cbackground_color=#090d11',
             f'-ctitle=Iris · {name}', scene, destination / f'{name}.html'], work)
        exported = destination / f'{name}.html'
        html = exported.read_text(encoding='utf-8')
        html = re.sub(r'((?:src|href)=\")([^\"]+)', lambda match: match[1] + match[2].replace(chr(92), '/'), html)
        html = html.replace('<html>', '<html lang="en">')
        html = html.replace('width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no', 'width=device-width, initial-scale=1.0')
        exported.write_text(html, encoding='utf-8', newline='\n')
        poster = destination / f'{name}-poster.png'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-sseof', '-0.10', '-i', movie,
             '-frames:v', '1', poster], work)
        details = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                                                       '-of', 'json', str(movie)], text=True))
        stream = next(s for s in details['streams'] if s['codec_type'] == 'video')
        if [stream['width'], stream['height']] != [width, height] or stream['r_frame_rate'] != f'{fps}/1':
            raise ValueError('actual render resolution/fps mismatch')
        if float(details['format']['duration']) < 5:
            raise ValueError('actual movie is too short')
        receipt['assets'][name] = {'video': movie.name, 'poster': poster.name, 'html': f'{name}.html',
                                   'duration_seconds': float(details['format']['duration']),
                                   'video_bytes': movie.stat().st_size, 'codec': stream['codec_name'],
                                   'sha256': hashlib.sha256(movie.read_bytes()).hexdigest()}
    (destination / 'render-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
