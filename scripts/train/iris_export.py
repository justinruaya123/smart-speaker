"""Canonical FP32 ONNX/ncnn export of the selected finite VCM."""
from pathlib import Path
import numpy as np
import torch
from scripts.models.mvp_static import ExportStatic
from scripts.train.iris_diversity import write,digest

def export_onnx(model, path):
    import onnx
    import onnxruntime as ort
    wrapper=ExportStatic(model.cpu().eval()).eval()
    generator=torch.Generator().manual_seed(53)
    sample=torch.randn(1,40,37,generator=generator)
    torch.onnx.export(wrapper,(sample,),str(path),input_names=['features'],output_names=['commands'],
                      dynamic_axes={'features':{2:'frames'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    errors=[];frames=[1,3,37,128,231]
    for n in frames:
        x=torch.randn(1,40,n,generator=generator)
        with torch.no_grad(): expected=wrapper(x).numpy()
        actual=session.run(None,{'features':x.numpy()})[0]
        if actual.shape != expected.shape: raise ValueError('Export output shape changed')
        errors.append(float(np.abs(actual-expected).max()))
    receipt=dict(passed=max(errors)<2e-5,max_absolute_error=max(errors),frames=frames,
                 tolerance=2e-5,bytes=Path(path).stat().st_size,sha256=digest(path))
    if not receipt['passed']: raise ValueError('ONNX numerical parity failed: '+str(receipt))
    return receipt


class NCNNStatic(torch.nn.Module):
    """The same batch-one endpoint encoder; ncnn supplies explicit positions."""
    def __init__(self,model):
        super().__init__();self.model=model;self.attention=torch.nn.ModuleList()
        for block in model.blocks:
            width=block.qkv.in_features
            attention=torch.nn.MultiheadAttention(width,block.heads,batch_first=True,dropout=0.)
            attention.in_proj_weight.data.copy_(block.qkv.weight)
            attention.in_proj_bias.data.copy_(block.qkv.bias)
            attention.out_proj.load_state_dict(block.attention_output.state_dict())
            self.attention.append(attention)

    def forward(self,features,position):
        x=self.model.stem(features).transpose(1,2)+position
        for block,attention in zip(self.model.blocks,self.attention):
            normalized=block.attention_norm(x)
            attended,_=attention(normalized,normalized,normalized,need_weights=False)
            x=x+attended
            x=x+block.local(block.local_norm(x).transpose(1,2)).transpose(1,2)
            x=x+block.ff(block.ff_norm(x))
        return self.model.classifier(self.model.norm(x).mean(1))


def export_ncnn(model, onnx, folder):
    import ncnn
    import pnnx
    folder=Path(folder);folder.mkdir(exist_ok=False)
    model=model.cpu().eval();wrapper=NCNNStatic(model).eval()
    generator=torch.Generator().manual_seed(53)
    def inputs(n):
        f=torch.randn(1,40,n,generator=generator)
        angle=torch.arange((n+1)//2,dtype=torch.float32)[:,None]*model.position_frequency[None]
        position=torch.stack((angle.sin(),angle.cos()),dim=-1).flatten(-2)[None]
        return f,position
    rewrite=[]
    for n in (1,37,128,231):
        f,pos=inputs(n)
        with torch.no_grad():rewrite.append(float((wrapper(f,pos)-model(f,torch.tensor([n]))).abs().max()))
    if max(rewrite)>2e-5:raise ValueError('ncnn endpoint rewrite changed logits')
    torch.backends.mha.set_fastpath_enabled(False)
    pnnx.export(wrapper,str(folder/'model.pt'),inputs=inputs(128),inputs2=inputs(231),fp16=False)
    write(folder/'position-frequency.json',model.position_frequency.tolist())
    net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=False
    net.opt.use_fp16_packed=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
    net.opt.use_bf16_packed=False;net.opt.use_bf16_storage=False;net.opt.use_packing_layout=False
    if net.load_param(str(folder/'model.ncnn.param')) or net.load_model(str(folder/'model.ncnn.bin')):
        raise ValueError('ncnn export did not load')
    errors=[]
    for n in (1,3,37,128,231,511):
        f,pos=inputs(n)
        with torch.no_grad():expected=model(f,torch.tensor([n])).numpy().reshape(-1)
        buffers=[np.ascontiguousarray(f[0].numpy()),np.ascontiguousarray(pos[0].numpy())]
        mats=[ncnn.Mat(x).clone() for x in buffers]
        with net.create_extractor() as ex:
            if ex.input('in0',mats[0]) or ex.input('in1',mats[1]):raise ValueError('ncnn input failed')
            code,value=ex.extract('out0')
            if code:raise ValueError('ncnn extraction failed')
            actual=np.array(value,copy=True).reshape(-1)
        errors.append(float(np.abs(actual-expected).max()))
    receipt=dict(onnx_sha256=digest(onnx),rewrite_errors=rewrite,
        files={f:digest(folder/f) for f in ('model.ncnn.param','model.ncnn.bin','position-frequency.json')},
        parity=dict(passed=max(errors)<2e-5,max_absolute_error=max(errors),errors=errors,tolerance=2e-5))
    write(folder/'engine.json',receipt)
    if not receipt['parity']['passed']:raise ValueError('ncnn numerical parity failed: '+str(receipt))
    return receipt
