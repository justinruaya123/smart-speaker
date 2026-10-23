"""Build seven pre-baked Iris feedback pairs (network only at build time).

Speech: named non-cloned en-US-AriaNeural through isolated edge-tts==7.3.1.
Playback never imports this builder, TTS, uv, ffmpeg or an MP3 decoder.
Outputs are new-only; generation is staged in task scratch, not in scientific envs.
"""
from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT=Path(__file__).resolve().parents[1]
RATE=16000
VOICE='en-US-AriaNeural'
EDGE_VERSION='7.3.1'
TEXTS={'wake_yes':'Yes','wake_hmm':'Hmm','wake_yeah':'Yeah','wake_here':'Here'}
TONES={'intent':((880,80),),'success':((660,80),(990,80)),'failure':((440,100),(220,100))}


def tone_pcm(cue_id):
    if cue_id not in TONES:
        raise ValueError('Unknown finite tone cue')
    result=array('h')
    fade=round(RATE*.01)
    for part,(frequency,milliseconds) in enumerate(TONES[cue_id]):
        if part:result.extend([0]*round(RATE*.03))
        count=round(RATE*milliseconds/1000)
        for i in range(count):
            edge=min(i,count-1-i,fade-1)/(fade-1)
            envelope=.5-.5*math.cos(math.pi*edge)
            result.append(round(.10*32767*envelope*math.sin(2*math.pi*frequency*i/RATE)))
    return result


def write_pcm(path,values):
    pcm=array('h',values)
    if sys.byteorder!='little':pcm.byteswap()
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(RATE);stream.writeframes(pcm.tobytes())


def file_record(path):
    return {'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def wave_record(path):
    with wave.open(str(path),'rb') as stream:
        if stream.getnchannels()!=1 or stream.getframerate()!=RATE or stream.getsampwidth()!=2:
            raise ValueError('Feedback must decode to mono16kHz PCM16')
        frames=stream.getnframes();payload=stream.readframes(frames)
    pcm=array('h');pcm.frombytes(payload)
    if sys.byteorder!='little':pcm.byteswap()
    peak=max(abs(v) for v in pcm)/32768
    if not .05<=frames/RATE<=2.5 or not 0<peak<.7:
        raise ValueError('Feedback duration/peak outside predeclared bounds')
    return dict(samples=frames,duration_seconds=frames/RATE,peak=peak,
                pcm_sha256=hashlib.sha256(payload).hexdigest(),pcm_format='s16le')


def build_feedback(output,*,voice=VOICE,scratch=None):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing feedback; select a new output: '+str(output))
    if voice!=VOICE:
        raise ValueError('This release fixes the named non-cloned voice '+VOICE)
    ffmpeg=shutil.which('ffmpeg');uv=shutil.which('uv')
    if not ffmpeg or not uv:
        raise RuntimeError('Build only: ffmpeg with libmp3lame and uv are required')
    command=[uv,'run','--isolated','--no-project','--python','3.13','--with','edge-tts=='+EDGE_VERSION,'python','-m','edge_tts']
    version=subprocess.run(command+['--version'],check=True,capture_output=True,text=True,timeout=180).stdout.strip()
    if EDGE_VERSION not in version:
        raise ValueError('Unexpected isolated TTS generator version: '+version)
    voices=subprocess.run(command+['--list-voices'],check=True,capture_output=True,text=True,timeout=180).stdout
    if not any(line.split() and line.split()[0]==voice for line in voices.splitlines()):
        raise ValueError('Named TTS voice is not available: '+voice)
    ffmpeg_version=subprocess.run([ffmpeg,'-version'],check=True,capture_output=True,text=True,timeout=10).stdout.splitlines()[0]
    scratch=Path(scratch or os.environ['TMPDIR']).resolve();scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='iris-feedback-build-',dir=scratch) as temporary:
        stage=Path(temporary);finished=stage/'finished';finished.mkdir()
        manifest=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),sample_rate=RATE,
            channels=1,playback_format='predecoded_pcm16_wav',
            build={'source':'scripts/build_iris_feedback.py','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'ffmpeg_version':ffmpeg_version,'speech_service':'Microsoft Edge online TTS (build time only)',
                   'edge_tts_version':EDGE_VERSION,'voice_available_at_build':voice},cues={})
        for cue in (*TEXTS,*TONES):
            source=stage/(cue+('.mp3' if cue in TEXTS else '.wav'))
            if cue in TEXTS:
                subprocess.run(command+['--voice',voice,'--text',TEXTS[cue],'--rate=-10%','--volume=-20%',
                    '--write-media',str(source)],check=True,capture_output=True,text=True,timeout=180)
                generator={'name':'edge-tts','version':EDGE_VERSION,'rate':'-10%','volume':'-20%','non_cloned_voice':True}
            else:
                write_pcm(source,tone_pcm(cue))
                generator={'name':'iris_tone','version':'1','tones_hz_ms':[list(p) for p in TONES[cue]],
                           'gap_ms':30 if len(TONES[cue])>1 else 0,'fade_ms':10,'amplitude':.10}
            mp3=finished/(cue+'.mp3');wav=finished/(cue+'.wav')
            options=['-af','volume=0.5'] if cue in TEXTS else []
            subprocess.run([ffmpeg,'-nostdin','-v','error','-i',str(source),'-map_metadata','-1',
                '-ac','1','-ar',str(RATE),*options,'-c:a','libmp3lame','-b:a','64k',str(mp3)],check=True,capture_output=True,timeout=30)
            subprocess.run([ffmpeg,'-nostdin','-v','error','-i',str(mp3),'-map_metadata','-1',
                '-ac','1','-ar',str(RATE),'-c:a','pcm_s16le',str(wav)],check=True,capture_output=True,timeout=30)
            record=dict(generator=generator,mp3=file_record(mp3),wav=file_record(wav),**wave_record(wav))
            if cue in TEXTS:record.update(text=TEXTS[cue],voice=voice)
            manifest['cues'][cue]=record
        if len({r['pcm_sha256'] for r in manifest['cues'].values()})!=7:
            raise ValueError('Feedback waveforms must be distinct')
        (finished/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
        output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(finished,output,dirs_exist_ok=False)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'submission/feedback')
    parser.add_argument('--scratch',type=Path,help='Task-owned staging directory, default TMPDIR')
    args=parser.parse_args(argv)
    result=build_feedback(args.output,scratch=args.scratch)
    print(json.dumps({'output':str(args.output),'cues':list(result['cues']),
                      'manifest_sha256':hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
