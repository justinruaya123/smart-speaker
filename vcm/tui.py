"""Inspect tentative and final commands. Preview only: never executes actions."""
import argparse
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np


def read_wav(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    pcm,rate=sf.read(path,dtype='float32')
    if pcm.ndim!=1 or not np.isfinite(pcm).all(): raise ValueError('Expected finite mono WAV')
    if not 8000<=rate<=96000: raise ValueError('Supported WAV sample rate range is 8–96 kHz')
    if rate!=16000:
        factor=gcd(rate,16000); pcm=resample_poly(pcm,16000//factor,rate//factor).astype('float32')
    return pcm


def inspect_wav(runtime,path,speed=0.,step=False):
    pcm=read_wav(path); runtime.reset()
    for pos in range(0,len(pcm),640):
        end=pos+640>=len(pcm)
        result=runtime.feed(pcm[pos:pos+640],pos,endpoint=end)
        if pos%8000==0 or end:
            print(json.dumps({'position_seconds':min(pos+640,len(pcm))/16000,'state':'final' if end else 'tentative',**result}),flush=True)
        if step: input('Enter for next 40 ms: ')
        elif speed>0: time.sleep(.04/speed)
    if not len(pcm): print(json.dumps(runtime.predict(pcm)))


def inspect_microphone(runtime,device):
    import sounddevice as sd
    # Native 16 kHz only: never independently resample callback blocks.
    sd.check_input_settings(device=device,channels=1,dtype='float32',samplerate=16000)
    q=queue.Queue(maxsize=32); stop=threading.Event(); overflow=threading.Event(); sequence=0
    def callback(indata,frames,_timing,status):
        nonlocal sequence
        item=(indata[:,0].copy(),sequence,bool(status)); sequence+=frames
        try: q.put_nowait(item)
        except queue.Full: overflow.set()
    input('Press Enter to begin push-to-talk (30 second limit). ')
    def finish(): input('Press Enter to finish. '); stop.set()
    threading.Thread(target=finish,daemon=True).start()
    runtime.reset(); started=time.monotonic()
    with sd.InputStream(device=device,samplerate=16000,channels=1,dtype='float32',blocksize=640,callback=callback):
        while not stop.is_set() and time.monotonic()-started<30:
            try: pcm,pos,status=q.get(timeout=.2)
            except queue.Empty: continue
            result=runtime.feed(pcm,pos,overflow=status or overflow.is_set())
            print(json.dumps({'state':'tentative','queue_depth':q.qsize(),**result}),flush=True)
    # Drain every captured block after stopping the stream; no missing tail accepted.
    while not q.empty():
        pcm,pos,status=q.get_nowait(); runtime.feed(pcm,pos,overflow=status or overflow.is_set())
    print(json.dumps({'state':'final',**runtime.feed(np.empty(0,dtype='float32'),sequence,endpoint=True,overflow=overflow.is_set())}))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--bundle',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True); source.add_argument('--wav',type=Path); source.add_argument('--mic',type=int)
    p.add_argument('--speed',type=float,default=0.,help='WAV inspection speed; zero runs unpaced')
    p.add_argument('--step',action='store_true',help='Step through WAV only')
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    if args.threads<1 or args.speed<0: p.error('Threads must be positive and speed nonnegative')
    if args.mic is not None and (args.step or args.speed): p.error('Live microphone runs at 1x without stepping')
    from .runtime import Runtime
    runtime=Runtime(args.bundle,args.threads)
    if args.wav: inspect_wav(runtime,args.wav,args.speed,args.step)
    else: inspect_microphone(runtime,args.mic)


if __name__=='__main__': main()
