"""A wider endpoint student using the frozen trainer and compact INT8 export."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train import pcv_context_training as base
from scripts.train.pcv_public_data import sha
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_qualification import verify_bindings

PREVIOUS=ROOT/'experiments/pcv-updated-context-v1'
HERE=ROOT/'experiments/pcv-updated-context-int8-v1'
CACHE=ROOT/'.cache/pcv-updated-context-int8-v1'


def freeze(rows,excluded):
    parent=json.loads((PREVIOUS/'protocol.json').read_text());verify_bindings(ROOT,parent['bindings'])
    verify_bindings(ROOT,parent['installed_artifact_hashes'])
    path=HERE/'protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text());verify_bindings(ROOT,protocol['bindings']);return protocol
    if [r['id'] for r in rows]!=parent['fit_ids'] or excluded!=parent['ctc_frame_exclusions']:
        raise ValueError('Wider comparison changed fitting membership')
    protocol=copy.deepcopy(parent)
    protocol.update({'experiment':'pcv-updated-context-int8-v1','seed':903,
        'hypothesis':'A 935485-parameter endpoint student improves complete-command development recognition while dynamic INT8 export retains the two-MiB graph budget.',
        'causal_limitation':'This comparison changes width and random seed. Do not claim an isolated capacity effect.',
        'architecture':{'width':128,'layers':6,'heads':4,'feedforward':256,'dropout':.1},
        'initialization':'random weights; same 24 human-only plus 40 mixed teacher epochs as the 80-channel candidate',
        'development_architecture_baseline':json.loads((PREVIOUS/'selection-before-calibration.json').read_text())['selected'],
        'quantization':{'format':'dynamic QOperator','weight_type':'QInt8','per_channel':True,
            'operators':['MatMul'],'constant_weight_only':True,'reduce_range':False,
            'preprocess':'ONNX shape inference; skip graph optimization and symbolic inference',
            'maximum_graph_bytes':2*1024*1024,
            'development_fidelity_gate':{'maximum_human_exact_loss':1,'maximum_synthetic_exact_loss':1,
                'maximum_extra_human_negative_errors':1},
            'note':'Compression fidelity is separate from unchanged acceptance-quality gates. Pi speed remains unmeasured.'}})
    additions=['scripts/train/pcv_context_int8.py','experiments/pcv-updated-context-int8-v1/model-preflight.json',
        'experiments/pcv-updated-context-v1/protocol.json','experiments/pcv-updated-context-v1/selection-before-calibration.json']
    protocol['bindings'].update({p:sha(ROOT/p) for p in additions})
    immutable(path,protocol);return protocol


def export_quantized(model,directory):
    import onnx
    from onnxruntime.quantization import quantize_dynamic,QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from scripts.models.pcv_contextual import export_onnx
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    original=directory/'model.fp32.onnx';prepared=directory/'model.preprocessed.onnx';quantized=directory/'model.int8.onnx'
    if any(p.exists() for p in (original,prepared,quantized)):raise FileExistsError('Preserve existing export')
    floating=export_onnx(model,original)
    if floating['max_absolute_error']>=floating['tolerance']:raise ValueError('FP32 numerical export failed')
    quant_pre_process(original,prepared,skip_optimization=True,skip_symbolic_shape=True,skip_onnx_shape=False)
    quantize_dynamic(prepared,quantized,op_types_to_quantize=['MatMul'],weight_type=QuantType.QInt8,
        per_channel=True,reduce_range=False,extra_options={'MatMulConstBOnly':True})
    onnx.checker.check_model(str(quantized))
    return {'fp32':floating,'fp32_numerical_parity_pass':True,
        'int8_bytes':quantized.stat().st_size,'size_pass':quantized.stat().st_size<2*1024*1024,
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in (original,prepared,quantized)},
        'accuracy_fidelity':'requires locked development comparison','physical_pi':False}


def run():
    base.HERE=HERE;base.CACHE=CACHE;base.freeze=freeze
    base.run()


if __name__=='__main__':run()
