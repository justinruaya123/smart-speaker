"""Read-only segmentation diagnostic, never a test-record selection filter."""
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pyarrow.parquet as pq
import soundfile as sf
from scripts.runtime.wake_models import SileroVAD
from scripts.train.pcv_updated_data import HERE,CACHE
from scripts.train.pcv_public_data import sha,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_qualification import verify_bindings


def main():
    settings={'preroll_ms':200,'silence_ms':400,'max_utterance_seconds':12.,'speech_threshold':.5}
    audit=json.loads((HERE/'audio-audit.json').read_text());verify_bindings(ROOT,{audit['manifest']:audit['manifest_sha256']})
    files=[Path(__file__),ROOT/'scripts/runtime/wake_models.py',ROOT/'artifacts/silero-vad.onnx',HERE/'audio-audit.json']
    immutable(HERE/'boundary-protocol.json',{'settings':settings,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'purpose':'Check human command clip segmentation without ASR or changing official split/waveform membership',
        'scope':'all published human train, test and holdout command clips; no numeral or synthetic pool',
        'no_training_or_test_filtering':True,'thresholds_selected_from_test':False,'human_listening':False})
    rows=read_rows(ROOT/audit['manifest']);wanted={r['id']:r for r in rows if not r['is_synthetic'] and r['published_split']!='numerals'}
    vad=SileroVAD(ROOT/'artifacts',settings);records=[];began=time.perf_counter()
    for file in json.loads((HERE/'acquisition-complete.json').read_text())['files']:
        path=ROOT/file['path'];split=path.name.split('-')[0]
        if split=='numerals':continue
        index=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['audio']):
            for value in batch.to_pylist():
                identity=f'{split}/{path.stem}/{index:06d}';index+=1
                if identity not in wanted:continue
                row=wanted[identity];pcm,rate=sf.read(io.BytesIO(value['audio']['bytes']),dtype='float32')
                if rate!=16000:raise ValueError('Unexpected sampling rate')
                segments=vad.segment(pcm)
                records.append({'id':identity,'source':row['source'],'published_split':split,'count':len(segments),
                    'segment_seconds':[len(p)/rate for p,_ in segments], 'timed_out':any(t for _,t in segments),
                    'annotation':row['transcript_source'],'publisher_command':row['command']})
            if len(records)%250<100:print('BOUNDARY_AUDIT',len(records),round(time.perf_counter()-began,1),flush=True)
    counts={split:dict(Counter(str(r['count']) for r in records if r['published_split']==split)) for split in ('train','test','holdout')}
    immutable(HERE/'boundary-audit.json',{'counts_by_published_split':counts,'rows':len(records),'records':records,
        'seconds':time.perf_counter()-began,'protocol_sha256':sha(HERE/'boundary-protocol.json'),
        'waveforms_or_test_membership_changed':False,'independent_human_segmentation_verified':False,
        'interpretation':'Zero/multiple segments are review flags, not proof of incorrect speech or grounds for removing hard test cases'})
    print('BOUNDARIES_COMPLETE',json.dumps(counts),flush=True)


if __name__=='__main__':main()
