"""Independent finite-word STT frontend and CPU ONNX decoding; no PyTorch."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import onnxruntime as ort
from einops import rearrange, reduce


@lru_cache(maxsize=1)
def melbank():
    low, high = 2595*np.log10(1+np.array([20,7600])/700)
    edges = 700*(10**(np.linspace(low,high,82)/2595)-1)
    frequencies = np.fft.rfftfreq(400, 1/16000)
    bank = np.zeros((201,80),dtype=np.float32)
    for i in range(80):
        bank[:,i] = np.maximum(0,np.minimum((frequencies-edges[i])/(edges[i+1]-edges[i]),
                                           (edges[i+2]-frequencies)/(edges[i+2]-edges[i+1])))
    return bank


def stt_features(pcm):
    pcm = np.asarray(pcm,dtype=np.float32)
    if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError('STT expects finite nonempty mono 16 kHz PCM')
    if len(pcm)<400:
        pcm = np.pad(pcm,(0,400-len(pcm)))
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(pcm,(200,200)),400)[::160]
    power = np.abs(np.fft.rfft(windows*np.hanning(400).astype(np.float32),axis=-1))**2
    mel = np.log(np.maximum(power@melbank(),1e-7)).astype(np.float32)
    mean = reduce(mel,'time mel -> 1 mel','mean')
    variance = reduce((mel-mean)**2,'time mel -> 1 mel','mean')
    return (mel-mean)/np.sqrt(variance+1e-5)


def decode_words(logits, vocabulary):
    logits = np.asarray(logits,dtype=np.float64)
    shifted = logits-logits.max(axis=-1,keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1,keepdims=True)
    ids = probabilities.argmax(axis=-1)
    words, confidence = [], []
    previous = -1
    for i,index in enumerate(ids):
        if index and index != previous:
            words.append(vocabulary[index]); confidence.append(float(probabilities[i,index]))
        previous = index
    return {'text':' '.join(words),'confidence':min(confidence) if confidence else 0.}


class AcousticSTT:
    def __init__(self, bundle, threads=2, variant='fp32'):
        bundle = Path(bundle)
        self.metadata = json.loads((bundle/'bundle.json').read_text(encoding='utf-8'))['stt']
        self.vocabulary = self.metadata['vocabulary']
        if self.vocabulary[:2] != ['<blank>','<unknown>'] or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError('Invalid STT word vocabulary')
        options = ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
        self.session = ort.InferenceSession(str(bundle/f'stt.{variant}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        if self.session.get_outputs()[0].shape[-1] != len(self.vocabulary):
            raise ValueError('STT graph/vocabulary mismatch')

    def recognize(self, pcm):
        mel = stt_features(pcm)
        logits,lengths = self.session.run(None,{'features':rearrange(mel,'time mel -> 1 time mel'),
                                               'lengths':np.array([len(mel)],dtype=np.int64)})
        if not np.isfinite(logits).all():
            raise ValueError('STT returned nonfinite logits')
        return decode_words(logits[0,:int(lengths[0])],self.vocabulary)

    def transcribe(self, pcm):
        return self.recognize(pcm)['text']
