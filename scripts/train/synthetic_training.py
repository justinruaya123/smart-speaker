"""From-scratch source experts; select weights using development only."""
from collections import Counter
import json
import math
import random
import re
import time
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange
from .synthetic_data import ROOT, DATA, CACHE, SYN, MIX, HEADS, WORDS, rows, dump, digest, prepare_features
from .synthetic_vcm import FixedVCM


def records(source):
    path = DATA / ("manifest.jsonl" if source == "synthetic" else "human.jsonl")
    if source == "synthetic":
        if len(rows(path)) != len(rows(DATA / "planned.jsonl")): raise ValueError("Complete TTS generation before fitting")
    return rows(path)


def batch(data, features, indices, device):
    lengths = [features[i].shape[1] for i in indices]
    x = np.zeros((len(indices), 40, max(lengths)), dtype=np.float32)
    for j, i in enumerate(indices): x[j, :, :lengths[j]] = features[i]
    target = {name: torch.tensor([values.index(data[i]["labels"][name]) for i in indices], device=device) for name, values in HEADS.items()}
    keyword = np.zeros((len(indices), len(WORDS)), dtype=np.float32)
    for j, i in enumerate(indices):
        tokens = set(re.findall("[a-z]+", data[i]["transcript"].lower()))
        for k, word in enumerate(WORDS): keyword[j, k] = word in tokens
    return torch.from_numpy(x).to(device), torch.tensor(lengths, device=device), target, torch.from_numpy(keyword).to(device)


def losses(outputs, targets, keywords):
    result = F.cross_entropy(outputs[0], targets["intent"]) + .6*F.cross_entropy(outputs[1], targets["ready"])
    ready = targets["ready"] == 1
    for name in ("operation", "amount", "color"):
        relevant = ready & ((targets["intent"] == 3) | (targets["intent"] == 9)) if name != "color" else ready & (targets["intent"] == 4)
        if relevant.any():
            result = result + F.cross_entropy(outputs[list(HEADS).index(name)][relevant], targets[name][relevant])
    return result + .25*F.binary_cross_entropy_with_logits(outputs[-1], keywords)


def exact(raw, data):
    guessed = {k: np.argmax(v, axis=1) for k,v in raw.items()}
    matches = []; by_intent = {c: [] for c in HEADS["intent"]}
    for i,r in enumerate(data):
        y=r["labels"]; c=y["intent"]
        correct = guessed["intent"][i] == HEADS["intent"].index(c) and guessed["ready"][i] == HEADS["ready"].index(y["ready"])
        if y["ready"] == "READY":
            required = ("operation", "amount") if c in ("BRIGHTNESS", "VOLUME") else ("color",) if c == "COLOR_SET" else ()
            correct = correct and all(guessed[k][i] == HEADS[k].index(y[k]) for k in required)
        matches.append(bool(correct)); by_intent[c].append(bool(correct))
    return {"exact": float(np.mean(matches)), "macro_exact": float(np.mean([np.mean(v) for v in by_intent.values() if v])),
            "per_intent": {k: {"n":len(v), "exact":float(np.mean(v)) if v else None} for k,v in by_intent.items()}}


@torch.inference_mode()
def predict(model, data, feature_list=None, device="cuda", batch_size=48):
    model.eval()
    feature_list = feature_list if feature_list is not None else prepare_features(data)
    collected = {k: [] for k in HEADS}
    for offset in range(0,len(data),batch_size):
        idx=list(range(offset,min(offset+batch_size,len(data))))
        x,lengths,_,_=batch(data,feature_list,idx,device)
        out=model(x,lengths)
        for name, values in zip(HEADS,out): collected[name].append(values.float().softmax(-1).cpu().numpy())
    from einops import pack
    return {k:pack(v,"* c")[0] for k,v in collected.items()}


def fit(source, epochs=32):
    torch.set_num_threads(4); torch.manual_seed(17); np.random.seed(17); random.seed(17)
    torch.backends.cuda.matmul.allow_tf32 = True
    out=CACHE/source;out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        saved=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
        receipt=json.loads((out/"complete.json").read_text())
        manifest=DATA/("manifest.jsonl" if source=="synthetic" else "human.jsonl")
        if saved["protocol_sha256"]!=digest(DATA/"protocol.json") or saved["manifest_sha256"]!=digest(manifest):
            raise ValueError("Completed model belongs to a different frozen dataset/protocol")
        if receipt["checkpoint_sha256"]!=digest(out/"best.pt") or receipt["onnx_sha256"]!=digest(out/"model.onnx"):
            raise ValueError("Completed model artifact changed")
        print("Existing completed source model verified",source,flush=True);return
    protocol=json.loads((DATA/"protocol.json").read_text())
    if epochs!=protocol["epochs"]: raise ValueError("Epoch recipe is frozen; change requires a new protocol")
    data=records(source)
    training=[r for r in data if r["split"]=="train"]
    development=[r for r in data if r["split"]=="dev"]
    print("FEATURES",source,len(training),len(development),flush=True)
    train_features=prepare_features(training);dev_features=prepare_features(development)
    model=FixedVCM().to("cuda")
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00003)
    counts=Counter(r["labels"]["intent"] for r in training)
    weights=torch.tensor([1/math.sqrt(counts[r["labels"]["intent"]]) for r in training],dtype=torch.double)
    history=[];best=-1.;start=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0.;steps=0
        indices=torch.multinomial(weights,len(training),replacement=True).tolist()
        for offset in range(0,len(indices),48):
            ids=indices[offset:offset+48]
            x,lengths,target,keyword=batch(training,train_features,ids,"cuda")
            optimizer.zero_grad(set_to_none=True)
            output=model(x,lengths)
            loss=losses(output,target,keyword)
            if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach());steps+=1
        scheduler.step()
        raw=predict(model,development,dev_features)
        metrics=exact(raw,development)
        entry={"epoch":epoch+1,"loss":total/steps,"development":metrics,"seconds":time.perf_counter()-start}
        history.append(entry);dump(out/"history.json",history)
        if metrics["macro_exact"]>best:
            best=metrics["macro_exact"]
            torch.save({"state":model.state_dict(),"epoch":epoch+1,"width":128,"source":source,"protocol_sha256":digest(DATA/"protocol.json"),
                        "manifest_sha256":digest(DATA/("manifest.jsonl" if source=="synthetic" else "human.jsonl")),"development":metrics},out/"best.pt")
        print("EPOCH",source,epoch+1,"loss",round(total/steps,4),"dev_macro_exact",round(metrics["macro_exact"],4),"seconds",round(time.perf_counter()-start),flush=True)
    checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["state"]);model=model.cpu().eval()
    x=torch.randn(2,40,240);lengths=torch.tensor([240,173])
    torch.onnx.export(model,(x,lengths),str(out/"model.onnx"),input_names=["features","lengths"],output_names=[*HEADS,"keywords"],
        dynamic_axes={"features":{0:"batch",2:"frames"},"lengths":{0:"batch"},**{k:{0:"batch"} for k in [*HEADS,"keywords"]}},opset_version=17,dynamo=False)
    receipt={"source":source,"epochs_completed":epochs,"selected_epoch":checkpoint["epoch"],"development":checkpoint["development"],
        "train_n":len(training),"dev_n":len(development),"random_initialization":True,"test_used_for_selection":False,
        "parameters":sum(p.numel() for p in model.parameters()),"checkpoint_sha256":digest(out/"best.pt"),"onnx_sha256":digest(out/"model.onnx"),
        "onnx_bytes":(out/"model.onnx").stat().st_size,"elapsed_seconds":time.perf_counter()-start,"native_exit_status":"see process receipt"}
    dump(out/"complete.json",receipt);dump((SYN if source=="synthetic" else MIX)/(source+"-training.json"),receipt)
    dump((SYN if source=="synthetic" else MIX)/(source+"-history.json"),history)
    print(json.dumps(receipt),flush=True)
