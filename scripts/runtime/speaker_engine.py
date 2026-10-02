"""Explicit CPU engines. No engine fallback and no training dependencies."""
import hashlib
import json
from pathlib import Path
import numpy as np
from einops import rearrange, repeat


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class NCNNSession:
    def __init__(self,root,threads):
        import ncnn
        self.ncnn=ncnn;root=Path(root);folder=root/'ncnn'
        receipt=json.loads((folder/'engine.json').read_text())
        if receipt['onnx_sha256']!=digest(root/'model.onnx'):raise ValueError('ncnn source ONNX identity mismatch')
        for name in ('model.ncnn.param','model.ncnn.bin','position-frequency.json'):
            if receipt['files'].get(name)!=digest(folder/name):raise ValueError('ncnn file identity mismatch: '+name)
        self.frequency=np.asarray(json.loads((folder/'position-frequency.json').read_text()),dtype='float32')
        self.net=ncnn.Net();self.net.opt.num_threads=threads;self.net.opt.use_vulkan_compute=False
        self.net.opt.use_fp16_packed=False;self.net.opt.use_fp16_storage=False;self.net.opt.use_fp16_arithmetic=False
        self.net.opt.use_bf16_packed=False;self.net.opt.use_bf16_storage=False
        self.net.opt.use_packing_layout=False
        if self.net.load_param(str(folder/'model.ncnn.param'))!=0 or self.net.load_model(str(folder/'model.ncnn.bin'))!=0:
            raise RuntimeError('ncnn model load failed; select onnxruntime explicitly or rebuild the ncnn package')

    def get_providers(self):return ['ncnnCPU']

    def run(self,outputs,feed):
        x=feed['features'];frames=(x.shape[2]+1)//2
        angles=rearrange(np.arange(frames,dtype='float32'),'t -> t 1')*rearrange(self.frequency,'d -> 1 d')
        position=rearrange([np.sin(angles),np.cos(angles)],'two t d -> t (d two)')
        values=[]
        # Keep NumPy storage and Mat owners alive until every output is copied.
        # The Python binding does not retain temporary input-buffer owners.
        buffers=[np.ascontiguousarray(rearrange(x,'1 c t -> c t')),np.ascontiguousarray(position)]
        inputs=[self.ncnn.Mat(value).clone() for value in buffers]
        with self.net.create_extractor() as ex:
            for name,mat in zip(('in0','in1'),inputs):
                if ex.input(name,mat)!=0:raise RuntimeError('ncnn input failed: '+name)
            for i,width in enumerate((40,19,2)):
                code,mat=ex.extract(f'out{i}')
                if code!=0:raise RuntimeError(f'ncnn output {i} failed ({code})')
                value=np.array(mat,copy=True)
                if i==0:
                    if value.shape!=(frames,width):raise ValueError(f'ncnn CTC shape mismatch: {value.shape}')
                    value=rearrange(value,'t c -> 1 t c')
                else:
                    value=repeat(rearrange(value,'... -> (...)'),'c -> 1 t c',t=frames)
                values.append(value)
        return values
