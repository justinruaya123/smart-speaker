"""CPU-only archive entry point; set thread limits before importing NumPy/ORT."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'


def main():
    import argparse
    import asyncio
    import json
    import platform
    from pathlib import Path
    import time
    import numpy as np
    from .bundle import load_bundle
    from .stream import StreamManager
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check','serve','wav','benchmark'])
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--wav',type=Path)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--releases',type=Path);parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args();started=time.perf_counter()
    snapshot=load_bundle(args.bundle,allow_diagnostic=args.diagnostic);load_seconds=time.perf_counter()-started
    if args.mode=='serve':
        from .server import serve
        asyncio.run(serve(StreamManager(snapshot),args.releases or args.bundle.parent,args.host,args.port,
            token=os.environ.get('PCV_SERVER_TOKEN'),allow_diagnostic=args.diagnostic));return
    result=dict(binding=snapshot['binding'],provider='CPUExecutionProvider',threads=1,load_seconds=load_seconds,
        diagnostic=snapshot['diagnostic'],platform=platform.platform(),machine=platform.machine(),
        catalog_entries=len(snapshot['catalog']['entries']),catalog_nodes=snapshot['node_count'],
        model_bytes=(args.bundle/'model.onnx').stat().st_size,
        activation_bytes=sum(p.stat().st_size for p in (args.bundle/'wake').glob('*.onnx')))
    if args.mode!='check':
        if args.wav is None:parser.error('--wav is required')
        import soundfile as sf
        from pcvcm.contract import file_hash
        from wake_gate import replay_pcm
        pcm,rate=sf.read(args.wav,dtype='float32')
        if rate!=16000 or pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('Finite mono 16 kHz WAV required')
        if not 1<=args.repeats<=100:raise ValueError('Benchmark repeats must be 1-100')
        times=[];events=[]
        for _ in range(args.repeats if args.mode=='benchmark' else 1):
            began=time.perf_counter();events=replay_pcm(snapshot['gate'],pcm);times.append(time.perf_counter()-began)
        result.update(events=events,audio_sha256=file_hash(args.wav),source_seconds=len(pcm)/16000,warmup_silence_seconds=2,timeout_flush_seconds=6,
            repeats=len(times),processing_seconds_p50=float(np.quantile(times,.5)),processing_seconds_p95=float(np.quantile(times,.95)),
            compute_rtf_p95=float(np.quantile(times,.95)/(len(pcm)/16000)),file_io_in_timing=False,
            physical_pi_measurement=platform.system()=='Linux' and platform.machine()=='aarch64')
        # Machine architecture alone is not proof of a Raspberry Pi model.
        result['physical_pi_measurement']=False
        model_path=Path('/proc/device-tree/model')
        if model_path.exists():
            result['hardware_model']=model_path.read_bytes().rstrip(b'\0').decode(errors='replace')
            result['physical_pi_measurement']='Raspberry Pi' in result['hardware_model']
        try:
            import resource
            result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:result['peak_rss_kib']=None
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
