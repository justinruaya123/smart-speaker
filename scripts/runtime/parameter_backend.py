"""One interchangeable parameter-reader contract. VCM owns the intent."""
import json
import re
from pathlib import Path
from .routing import parse_parameters


def parse_brightness(text, catalog):
    """Only explicit absolute brightness, with the existing target/number checks."""
    if '<unknown>' in text or not re.search(r'\b(brightness|lights?|lamps?|lighting)\b',text.lower()):
        return {'status':'needs_parameters','reason':'brightness_phrase_required'}
    if re.search(r'\b(volume|music|songs?|temperature)\b',text.lower()):
        return {'status':'needs_clarification','reason':'conflicting_parameter_domain'}
    result = parse_parameters('BRIGHTNESS',text,catalog)
    if result.get('status') == 'accepted' and result['command']['action'] != 'SET_BRIGHTNESS':
        return {'status':'needs_parameters','reason':'parameter_backend_supports_absolute_brightness_only'}
    return result


class ParameterBackend:
    def __init__(self,bundle,backend='stt',threads=2,variant='fp32'):
        if backend != 'stt':
            raise ValueError('Only stt is installed; experimental alternatives require separate evaluation')
        self.bundle=Path(bundle);self.name=backend;self.threads=threads;self.variant=variant
        self.metadata=json.loads((self.bundle/'bundle.json').read_text(encoding='utf-8'))
        self.engine=None
        if not (self.bundle/f'{backend}.{variant}.onnx').is_file():
            raise FileNotFoundError(f'{backend} is not in this bundle; republish with --backend {backend}')

    def load(self):
        if self.engine is None:
            from .stt_runtime import AcousticSTT
            self.engine=AcousticSTT(self.bundle,self.threads,self.variant)
        return self.engine

    def recognize(self,pcm):
        return self.load().recognize(pcm)

    def resolve_brightness(self,pcm):
        recognized=self.recognize(pcm)
        policy=self.metadata.get('brightness_parameters',{}).get(self.name,{})
        threshold=policy.get('threshold',1.1)
        result=parse_brightness(recognized['text'],self.metadata.get('catalog',{}))
        if result.get('status')=='accepted' and recognized['confidence']<threshold:
            result={'status':'needs_clarification','reason':'low_parameter_confidence'}
        return {**result,'parameter_source':self.name,'parameter_transcript':recognized['text'],
                'parameter_confidence':recognized['confidence'],'parameter_threshold':threshold,
                'stt_invoked':True,'parameter_invoked':True,
                'parameter_audio':{'start_sample_16k':0,'end_sample_16k':len(pcm),
                    'policy':'whole_utterance_no_verified_boundary','timing_verified':False}}
