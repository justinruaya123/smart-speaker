"""Record one frozen Baseline A forward pass for the local BMO explanation.

No fitting, downloads, dependency changes or saved model modifications.
Only this script's directory receives outputs. Run with the existing Python -B.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import einops
from einops import rearrange, reduce
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
REPO = Path(r"C:\Users\SCL-Jus\Documents\Grad\smart-speaker\.cache\fresh-repro-20261107-03\repo")
SOURCE = REPO / "submission/shared-demo-a/source"
sys.path.insert(0, str(SOURCE))
from vcm.frontend import features
from scripts.runtime.demo_slots_contract import ACTIONS, decode_scores
import vcm.frontend as frontend_module
import scripts.runtime.demo_slots_contract as contract_module

assert Path(frontend_module.__file__).resolve() == SOURCE / "vcm/frontend.py"
assert Path(contract_module.__file__).resolve() == SOURCE / "scripts/runtime/demo_slots_contract.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def round_values(values):
    return np.round(np.asarray(values, dtype=np.float64), 6).tolist()


def matrix(values):
    """Display one batch item with channels on rows and time on columns."""
    if values.ndim == 3:
        return rearrange(values, "1 channels time -> channels time")
    if values.ndim == 2:
        return rearrange(values, "channels time -> channels time")
    return rearrange(values, "channels -> channels 1")


def thumbnail(values, vector=False):
    grid = rearrange(values, "1 channels -> channels 1") if vector else matrix(values)
    channels, frames = grid.shape
    rows, cols = min(16, channels), min(48, frames)
    channel_edges = [i * channels // rows for i in range(rows + 1)]
    time_edges = [i * frames // cols for i in range(cols + 1)]
    cells = np.asarray([
        [float(reduce(grid[channel_edges[c]:channel_edges[c+1], time_edges[t]:time_edges[t+1]],
                      "channels time ->", "mean")) for t in range(cols)]
        for c in range(rows)
    ], dtype=np.float32)
    return dict(rows=rows, cols=cols, values=round_values(cells),
                channel_edges=channel_edges, time_edges=time_edges,
                aggregation="Arithmetic mean of each contiguous channel/time bin; no padding, interpolation or invented values",
                layout="channel_bin,time_bin", min=float(reduce(cells, "c t ->", "min")),
                max=float(reduce(cells, "c t ->", "max")))


def numeric_record(identifier, label, values, operation, symbolic_shape, **extra):
    value = np.asarray(values)
    mean = float(reduce(value, "... ->", "mean"))
    vector = value.ndim == 2 and value.shape[0] == 1
    record = dict(id=identifier, label=label, shape=list(value.shape), symbolic_shape=symbolic_shape,
                  dtype=str(value.dtype), operation=operation, parameters=0,
                  stats=dict(min=float(reduce(value, "... ->", "min")),
                             max=float(reduce(value, "... ->", "max")), mean=mean,
                             std=float(np.sqrt(reduce((value-mean)**2, "... ->", "mean")))),
                  thumbnail=thumbnail(value, vector=vector), **extra)
    if vector:
        record["values"] = round_values(rearrange(value, "1 channels -> channels"))
    return record


def write_json(name, value):
    path = OUT / name
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def main():
    for name in ("trace.json", "provenance.json", "sample.wav"):
        if (OUT / name).exists():
            raise FileExistsError("Preserve existing demo output: " + name)
    dataset = REPO / ".cache/hf-main/dataset/shared-demo-a-v1"
    with gzip.open(dataset / "test.jsonl.gz", "rt", encoding="utf-8") as stream:
        row = next(json.loads(line) for line in stream
                   if 'demo-v1-test-en-US-ChristopherNeural-0140' in line)
    assert row["id"] == "demo-v1-test-en-US-ChristopherNeural-0140"
    source_audio = REPO / ".cache/baseline-a-delete-repeat-03/local-synthesis" / (row["sha256"] + ".wav")
    assert digest(source_audio) == row["sha256"]
    pcm, rate = sf.read(source_audio, dtype="float32", always_2d=False)
    assert rate == 16000 and pcm.ndim == 1 and len(pcm) == 47616
    pcm_hash = hashlib.sha256(f"pcm:f32le:{rate}:{len(pcm)}\0".encode() + np.asarray(pcm, dtype="<f4").tobytes()).hexdigest()
    assert pcm_hash == row["pcm_sha256"]

    bundle = REPO / "submission/models/shared-demo-a"
    recipe = json.loads((bundle / "recipe.json").read_text())
    graph = bundle / "model.onnx"
    graph_hash = digest(graph)
    assert graph_hash == recipe["binding"]["model.onnx"]
    assert recipe["selected_epoch"] == 38 and recipe["parameters"] == 175653
    normalizer = json.loads((bundle / "normalization.json").read_text())
    assert digest(bundle / "normalization.json") == recipe["binding"]["normalization.json"]
    raw_features = features(pcm)
    normalized = features(pcm, normalizer["mean"], normalizer["std"])
    assert normalized.shape == (40, 296)
    feed = dict(features=rearrange(normalized, "mel time -> 1 mel time"),
                lengths=np.asarray([normalized.shape[1]], dtype=np.int64))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    original_session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    original_logits = original_session.run(None, feed)[0]

    # Existing output tensors become additional graph outputs in memory only.
    # Operators, learned tensors and original model bytes remain untouched.
    capture = {"/Slice_output_0": [1, 40, 148],
               "/stem/conv/Conv_output_0": [1, 128, 148],
               "/stem_norm/Transpose_1_output_0": [1, 128, 148],
               "/Mul_output_0": [1, 128, 148]}
    for block in range(4):
        for sublayer in range(2):
            capture[f"/blocks.{block}/depthwise.{sublayer}/conv/Conv_output_0"] = [1, 128, 148]
            capture[f"/blocks.{block}/pointwise.{sublayer}/Conv_output_0"] = [1, 128, 148]
            capture[f"/blocks.{block}/norms.{sublayer}/Transpose_1_output_0"] = [1, 128, 148]
            capture[f"/blocks.{block}/Mul{'_1' if sublayer else ''}_output_0"] = [1, 128, 148]
        capture[f"/blocks.{block}/Add_output_0"] = [1, 128, 148]
        capture[f"/blocks.{block}/Mul_2_output_0"] = [1, 128, 148]
    capture.update({"/Div_1_output_0": [1, 128], "/ReduceMax_output_0": [1, 128],
                    "/Concat_5_output_0": [1, 256], "/head/Gemm_output_0": [1, 29],
                    "/percent_head/Gemm_output_0": [1, 4], "/alarm_head/Gemm_output_0": [1, 4]})
    model = onnx.load(graph)
    original_nodes = [node.SerializeToString() for node in model.graph.node]
    original_initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
    all_outputs = {name for node in model.graph.node for name in node.output}
    assert set(capture).issubset(all_outputs)
    for name, shape in capture.items():
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    onnx.checker.check_model(model)
    trace_options = ort.SessionOptions()
    trace_options.intra_op_num_threads = 1
    trace_options.inter_op_num_threads = 1
    trace_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    trace_session = ort.InferenceSession(model.SerializeToString(), sess_options=trace_options,
                                         providers=["CPUExecutionProvider"])
    output_names = [output.name for output in trace_session.get_outputs()]
    outputs = dict(zip(output_names, trace_session.run(output_names, feed)))
    assert [node.SerializeToString() for node in model.graph.node] == original_nodes
    assert [tensor.SerializeToString() for tensor in model.graph.initializer] == original_initializers
    max_difference = float(reduce(np.abs(original_logits-outputs["logits"]), "b scores ->", "max"))
    assert np.allclose(original_logits, outputs["logits"], rtol=1e-5, atol=1e-5), max_difference
    scores = rearrange(outputs["logits"], "1 scores -> scores")
    decoded = decode_scores(scores, "heads")
    assert decoded["class_index"] == row["class_index"] == 6
    predictions_path = REPO / ".cache/published-a-replay-delete-repeat-03/predictions.json"
    prior = next(p for p in json.loads(predictions_path.read_text()) if p["id"] == row["id"])
    assert decoded["class_index"] == prior["predicted_class"]
    assert decoded["command"] == prior["predicted"] == row["demo_expected"]

    stages = [numeric_record("logmel", "Log-mel features", raw_features,
                             "Periodic Hann 400 samples; hop 160; FFT 512; 40 HTK mel bins, natural log with floor 1e-8", "40 × T"),
              numeric_record("normalize", "Frozen normalization", feed["features"],
                             "Per-mel-bin frozen mean and standard deviation from 1,889 equally weighted human TRAIN utterances", "B × 40 × T"),
              numeric_record("decimate", "Keep every second frame", outputs["/Slice_output_0"],
                             "features[..., ::2]; length becomes ceil(T/2); no learned strided convolution", "B × 40 × ceil(T/2)",
                             source_output="/Slice_output_0")]
    stem_children = [
        numeric_record("stem.conv", "Causal stem convolution", outputs["/stem/conv/Conv_output_0"],
                       "Conv1d 40→128, kernel 5, dilation 1, stride 1, left padding 4, no bias", "B × 128 × T′", source_output="/stem/conv/Conv_output_0"),
        numeric_record("stem.norm", "Framewise LayerNorm", outputs["/stem_norm/Transpose_1_output_0"],
                       "Normalize 128 channels independently at each time step; learned scale and bias", "B × 128 × T′", source_output="/stem_norm/Transpose_1_output_0")]
    stem_children[0]["parameters"] = 25600
    stem_children[1]["parameters"] = 256
    stem = numeric_record("stem", "40→128 causal stem", outputs["/Mul_output_0"],
                          "Causal kernel-5 convolution → framewise LayerNorm → SiLU", "B × 128 × T′",
                          source_output="/Mul_output_0", children=stem_children)
    stem["parameters"] = 25856
    stages.append(stem)
    for block, dilation in enumerate((1, 2, 4, 8)):
        children = []
        for sublayer in range(2):
            base = f"block{block+1}.layer{sublayer+1}"
            specs = [
                ("depthwise", "Depthwise convolution", f"/blocks.{block}/depthwise.{sublayer}/conv/Conv_output_0",
                 f"128 independent kernel-7 channels, groups 128, dilation {dilation}, left padding {6*dilation}, stride 1, no bias", 896),
                ("pointwise", "Pointwise channel mixing", f"/blocks.{block}/pointwise.{sublayer}/Conv_output_0",
                 "Conv1d 128→128, kernel 1, stride 1, no bias", 16384),
                ("norm", "Framewise LayerNorm", f"/blocks.{block}/norms.{sublayer}/Transpose_1_output_0",
                 "Normalize 128 channels at each time step, learned scale and bias", 256),
                ("silu", "SiLU activation", f"/blocks.{block}/Mul{'_1' if sublayer else ''}_output_0",
                 "x × sigmoid(x); training dropout p=0.1 is inactive during this inference", 0)]
            for suffix, label, tensor, operation, parameters in specs:
                child = numeric_record(base+"."+suffix, label, outputs[tensor], operation, "B × 128 × T′", source_output=tensor)
                child["parameters"] = parameters
                children.append(child)
        add_tensor = f"/blocks.{block}/Add_output_0"
        children.append(numeric_record(f"block{block+1}.residual", "Residual addition", outputs[add_tensor],
                                       "Second transformed branch plus unchanged block input", "B × 128 × T′", source_output=add_tensor))
        tensor = f"/blocks.{block}/Mul_2_output_0"
        record = numeric_record(f"block{block+1}", f"Residual block {block+1}", outputs[tensor],
                                "Two DW7 → PW1 → LayerNorm → SiLU stages; add block input, then multiply sum by 1/sqrt(2)",
                                "B × 128 × T′", dilation=dilation, residual_scale=2**-0.5, source_output=tensor,
                                residual_input_stage="stem" if block == 0 else f"block{block}", children=children)
        record["parameters"] = 35072
        stages.append(record)
    pool = numeric_record("pool", "Masked mean + maximum", outputs["/Concat_5_output_0"],
                          "Mean over valid frames (128 values), maximum over valid frames (128 values), concatenate in that order", "B × 256",
                          source_output="/Concat_5_output_0", children=[
                              numeric_record("pool.mean", "Temporal mean", outputs["/Div_1_output_0"], "Mask padded frames then divide by valid length", "B × 128", source_output="/Div_1_output_0"),
                              numeric_record("pool.max", "Temporal maximum", outputs["/ReduceMax_output_0"], "Mask invalid frames with -1e4 before temporal maximum", "B × 128", source_output="/ReduceMax_output_0")])
    stages.append(pool)
    head_specs = {
        "action": ("/head/Gemm_output_0", ["UNKNOWN" if a is None else json.dumps(a["command"], separators=(",", ":")) for a in ACTIONS], 7453),
        "percent": ("/percent_head/Gemm_output_0", ["missing / not applicable", "0%", "50%", "100%"], 1028),
        "alarm": ("/alarm_head/Gemm_output_0", ["missing / not applicable", "06:00", "12:30", "21:45"], 1028)}
    heads = {}
    head_children = []
    for name, (tensor, labels, parameters) in head_specs.items():
        values = rearrange(outputs[tensor], "1 scores -> scores")
        heads[name] = dict(labels=labels, logits=values.tolist(), argmax=int(values.argmax()),
                           shape=list(outputs[tensor].shape), source_output=tensor,
                           required_for_this_command=name in ("action", "percent"))
        child = numeric_record("heads."+name, name.capitalize()+" head", outputs[tensor],
                               f"Linear 256→{len(labels)} with bias; independent argmax", f"B × {len(labels)}", source_output=tensor)
        child["parameters"] = parameters
        head_children.append(child)
    head_stage = numeric_record("heads", "Three parallel linear heads", outputs["logits"],
                                "Concatenate action 29, percentage 4 and alarm 4 logits; not a joint calibrated probability", "B × 37",
                                source_output="logits", children=head_children)
    head_stage["parameters"] = 9509
    stages.append(head_stage)
    stages.append(dict(id="decode", label="Finite command decoder", shape=None, symbolic_shape="Typed command or rejection",
                       parameters=0, operation="Independent head argmax. Action selects required slot; UNKNOWN action or missing required slot rejects. Irrelevant slot-head outputs are ignored.",
                       output=decoded["command"]))
    parameter_total = sum(stage["parameters"] for stage in stages)
    assert parameter_total == 175653
    assert digest(graph) == graph_hash

    waveform_edges = [i*len(pcm)//256 for i in range(257)]
    waveform_bins = [pcm[waveform_edges[i]:waveform_edges[i+1]] for i in range(256)]
    sample = dict(id=row["id"], text=row["text"], audio_file="sample.wav", duration_seconds=len(pcm)/rate,
                  samples=len(pcm), sample_rate_hz=rate, channels=1, synthetic=True, voice=row["voice"],
                  split="test", previously_evaluated=True, sha256=row["sha256"], pcm_sha256=pcm_hash,
                  waveform=dict(sample_edges=waveform_edges,
                                minimum=[float(reduce(v, "samples ->", "min")) for v in waveform_bins],
                                maximum=[float(reduce(v, "samples ->", "max")) for v in waveform_bins],
                                aggregation="Per-bin min/max envelope from the original float32 decoded PCM"))
    verification = dict(audio_file_hash_matches=True, decoded_pcm_hash_matches=True,
                        original_graph_sha256_unchanged=True, graph_nodes_and_initializers_unchanged=True,
                        onnx_intermediate_outputs_added_in_memory_only=len(capture),
                        original_vs_trace_max_abs_logit_difference=max_difference,
                        original_vs_trace_allclose=True, allclose_atol=1e-5, allclose_rtol=1e-5,
                        final_decision_matches_preserved_replay=True, final_decision_matches_reference=True,
                        parameters_sum=parameter_total, onnxruntime=ort.__version__, onnx=onnx.__version__,
                        numpy=np.__version__, einops=einops.__version__, execution_provider="CPUExecutionProvider",
                        threads=1, instrumented_graph_optimizations="disabled", unmodified_graph_optimizations="default",
                        inference_timing_measured=False, training_updates=0)
    trace = dict(schema_version=1, evidence_kind="recorded_forward_pass_demo", sample=sample,
                 model=dict(name="SharedDemoSlotVCM-v1", sha256=graph_hash, selected_epoch=38,
                            parameters=175653, dtype="float32", qualified=False, actions_enabled=False),
                 features=dict(shape=list(normalized.shape), layout="mel,time", values=round_values(normalized),
                               serialization="Actual float32 normalized features rounded to 6 decimal places for JSON display",
                               float32_sha256=hashlib.sha256(normalized.tobytes()).hexdigest()),
                 stages=stages, heads=heads,
                 logits37=scores.tolist(), original_graph_logits37=rearrange(original_logits, "1 scores -> scores").tolist(),
                 decision=dict(class_index=decoded["class_index"], command=decoded["command"],
                               reason=None, head_argmax=decoded["heads"], actions_enabled=False,
                               model_qualified=False, prior_receipt_match=True),
                 verification=verification,
                 display_notes=[
                     "Measured values from one preserved synthetic recording and the frozen published epoch-38 graph. This is a demo replay, not a new evaluation cohort.",
                     "Feature matrices and activation thumbnails are real values; JSON display matrices are rounded to 6 decimal places.",
                     "Thumbnail rows combine contiguous channels; columns combine contiguous time frames. Arithmetic mean can cancel positive and negative activations. Edges identify every exact bin.",
                     "Color scales should use each thumbnail's documented min/max; do not interpret independent per-stage color scales as comparable magnitudes.",
                     "The playback/step animation is explanatory and does not represent measured operation latency or live streaming inference.",
                     "LayerNorm works across channels independently at each time step. Training dropout is inactive here.",
                     "All three heads execute; only action and percentage are used for this brightness command. Alarm output is ignored.",
                     "Logits are uncalibrated scores, not confidence percentages. Device actions and model qualification remain disabled."])
    write_json("trace.json", trace)
    shutil.copyfile(source_audio, OUT / "sample.wav")
    provenance = dict(schema_version=1, evidence_kind="recorded_forward_pass_demo", generated_on="2026-11-07",
                      sample={k:v for k,v in sample.items() if k != "waveform"}, source_audio_path=str(source_audio),
                      source=dict(repository="justinruaya123/ai231-smart-speaker", prefix="shared-demo-a-v1",
                                  publication_revision="13d388c31985be7f42af56b46570281728ecc054",
                                  acquisition_revision="39d5032248becc8a6efdd80738a4a4ef59f6d969",
                                  manifest_path="shared-demo-a-v1/test.jsonl.gz", manifest_row_id=row["id"],
                                  archive="shared-demo-a-v1/authored-synthesis.tar", archive_member=row["location"]["member"],
                                  provider="Microsoft Edge Speech", voice=row["voice"], authored_prompt=row["text"],
                                  synthetic=True, original_human=False, split="test", previously_evaluated=True),
                      license=dict(metadata_license="other", license_name="source-specific-research-terms",
                                   summary="Source-specific research/education dataset. Microsoft voice/output rights are separate from the edge-tts software license; no MIT, CC0 or Apache recording license is asserted.",
                                   card_url="https://huggingface.co/datasets/justinruaya123/ai231-smart-speaker/blob/13d388c31985be7f42af56b46570281728ecc054/shared-demo-a-v1/README.md",
                                   publication_authorization="This task assembles a local demo only; no external publication performed."),
                      model=dict(repository="justinruaya123/smart-speaker", commit="4efe5425901b96b1d80d990a95270ee7b1ddb58a",
                                 path="submission/models/shared-demo-a/model.onnx", sha256=graph_hash, selected_epoch=38,
                                 normalization_sha256=recipe["binding"]["normalization.json"]),
                      prior_prediction=dict(receipt_path=str(predictions_path), receipt_sha256=digest(predictions_path), row=prior),
                      source_modules=[dict(path=str(SOURCE/rel),sha256=digest(SOURCE/rel)) for rel in
                                      ["vcm/frontend.py", "scripts/runtime/demo_slots_contract.py", "scripts/models/demo_slots_v1.py", "scripts/models/vcm.py"]],
                      files={"sample.wav":dict(bytes=(OUT/"sample.wav").stat().st_size,sha256=digest(OUT/"sample.wav")),
                             "trace.json":dict(bytes=(OUT/"trace.json").stat().st_size,sha256=digest(OUT/"trace.json")),
                             "extract_trace.py":dict(bytes=Path(__file__).stat().st_size,sha256=digest(Path(__file__)))},
                      verification=verification)
    write_json("provenance.json", provenance)
    assert digest(OUT/"sample.wav") == row["sha256"]
    assert digest(graph) == graph_hash
    print(json.dumps(dict(files={p.name:p.stat().st_size for p in OUT.iterdir() if p.is_file()},
                          verification=verification, head_argmax=decoded["heads"], decision=decoded["command"]),indent=2))


if __name__ == "__main__":
    main()
