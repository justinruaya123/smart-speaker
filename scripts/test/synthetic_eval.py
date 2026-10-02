"""Development-only source preference/acceptance calibration and locked test."""
from collections import Counter
from datetime import datetime, timezone
import copy
import json
import time
import numpy as np
import onnxruntime as ort
from einops import rearrange, pack
from .audio import features, load_audio
from .synthetic_data import ROOT, DATA, CACHE, SYN, MIX, HEADS, WORDS, rows, dump, digest, label, catalog, prepare_features


def session(source):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(CACHE/source/"model.onnx"),options,providers=["CPUExecutionProvider"])


def softmax(x):
    x=x-np.max(x,axis=-1,keepdims=True);p=np.exp(x);return p/p.sum(axis=-1,keepdims=True)


def inference(source,data,split_name):
    path=CACHE/source/(split_name+"-probabilities.npz")
    receipt=path.with_suffix(".json")
    signature={"ids":[r["id"] for r in data],"model_sha256":digest(CACHE/source/"model.onnx")}
    if path.exists() and receipt.exists() and json.loads(receipt.read_text())==signature:
        with np.load(path) as value:return {k:value[k] for k in HEADS}
    engine=session(source);all_features=prepare_features(data);out={k:[] for k in HEADS}
    for start in range(0,len(data),32):
        chunk=all_features[start:start+32];lengths=np.array([x.shape[1] for x in chunk],dtype=np.int64)
        batch=np.zeros((len(chunk),40,int(max(lengths))),dtype=np.float32)
        for i,x in enumerate(chunk):batch[i,:,:x.shape[1]]=x
        values=engine.run(None,{"features":batch,"lengths":lengths})
        for k,v in zip(HEADS,values):out[k].append(softmax(v))
    out={k:pack(v,"* c")[0] for k,v in out.items()}
    np.savez_compressed(path,**out);dump(receipt,signature);return out


def standard_policy(alpha):
    return {c:{"real_weight":alpha,"log_odds_bias":0.,"intent_threshold":.5,"slot_threshold":.5,"margin":0.} for c in HEADS["intent"]}


def decode(synthetic,human,policy):
    n=len(synthetic["intent"]);columns=[]
    for j,c in enumerate(HEADS["intent"]):
        p=policy[c];a=p["real_weight"]
        v=np.clip((1-a)*synthetic["intent"][:,j]+a*human["intent"][:,j],1e-7,1-1e-7)
        columns.append(1/(1+np.exp(-(np.log(v)-np.log1p(-v)+p["log_odds_bias"]))))
    scores=rearrange(columns,"c n -> n c")
    normalized=scores/scores.sum(-1,keepdims=True)
    intent=scores.argmax(-1);sorted_scores=np.sort(normalized,axis=-1)
    winner_score=np.max(normalized,axis=-1);margin=sorted_scores[:,-1]-sorted_scores[:,-2]
    guessed={"intent":intent};confidence={"intent":winner_score}
    active=np.array([policy[HEADS["intent"][j]]["real_weight"] for j in intent])
    for k in ("ready","operation","amount","color"):
        a=rearrange(active,"n -> n 1");p=(1-a)*synthetic[k]+a*human[k]
        guessed[k]=p.argmax(-1);confidence[k]=p.max(-1)
    thresholds=np.array([policy[HEADS["intent"][j]]["intent_threshold"] for j in intent])
    slot_thresholds=np.array([policy[HEADS["intent"][j]]["slot_threshold"] for j in intent])
    margins=np.array([policy[HEADS["intent"][j]]["margin"] for j in intent])
    accepted=(intent!=0)&(winner_score>=thresholds)&(margin>=margins)&(guessed["ready"]==1)&(confidence["ready"]>=slot_thresholds)
    for k in ("operation","amount","color"):
        relevant=np.isin(intent,[3,9]) if k!="color" else intent==4
        accepted&=(~relevant)|((guessed[k]!=0)&(confidence[k]>=slot_thresholds))
    # Zero is a valid absolute value, but the existing typed contract rejects
    # a zero relative delta. Mirror that in calibration/scoring as well.
    accepted&=~(np.isin(intent,[3,9])&np.isin(guessed["operation"],[2,3])&(guessed["amount"]==1))
    for j,c in enumerate(HEADS["intent"]):
        for operation in policy[c].get("disabled_operations",[]):
            accepted&=~((intent==j)&(guessed["operation"]==HEADS["operation"].index(operation)))
    return guessed,confidence,accepted


def matches(guessed,data):
    truth={k:np.array([HEADS[k].index(r["labels"][k]) for r in data]) for k in HEADS}
    correct=(guessed["intent"]==truth["intent"])&(truth["ready"]==1)&(guessed["ready"]==1)
    for k in ("operation","amount","color"):
        relevant=np.isin(truth["intent"],[3,9]) if k!="color" else truth["intent"]==4
        correct&=(~relevant)|(guessed[k]==truth[k])
    return correct,truth


def score(synthetic,human,policy,data):
    guessed,confidence,accepted=decode(synthetic,human,policy);correct,truth=matches(guessed,data)
    positive=truth["ready"]==1;negative=~positive
    def metric(mask):
        emit=accepted&mask;pos=positive&mask;neg=negative&mask
        return {"n":int(mask.sum()),"positive_n":int(pos.sum()),"accepted":int(emit.sum()),"correct_accepted":int((emit&correct).sum()),
            "accepted_precision":float(correct[emit].mean()) if emit.any() else None,
            "exact_command_recall":float((emit&correct).sum()/pos.sum()) if pos.any() else None,
            "acceptance_rate":float(emit.sum()/mask.sum()) if mask.any() else None,
            "negative_n":int(neg.sum()),"false_accepts":int((emit&negative).sum()),
            "negative_false_accept_rate":float((emit&negative).sum()/neg.sum()) if neg.any() else None,
            "raw_family_accuracy":float((guessed["intent"][mask]==truth["intent"][mask]).mean()) if mask.any() else None}
    report={"overall":metric(np.ones(len(data),dtype=bool)),"per_intent":{c:metric(truth["intent"]==i) for i,c in enumerate(HEADS["intent"])},
            "per_emitted_intent":{c:metric(guessed["intent"]==i) for i,c in enumerate(HEADS["intent"])},
            "number_commands":metric(np.isin(truth["intent"],[3,9])&(truth["operation"]==1)),
            "relative_commands":metric(np.isin(truth["intent"],[3,9])&np.isin(truth["operation"],[2,3])),
            "per_source":{s:metric(np.array([r["source"]==s for r in data])) for s in sorted({r["source"] for r in data})},
            "per_requested_accent":{s:metric(np.array([r.get("accent_requested")==s for r in data])) for s in sorted({r["accent_requested"] for r in data if "accent_requested" in r})}}
    report["number_values"]={str(v):metric(np.isin(truth["intent"],[3,9])&(truth["operation"]==1)&(truth["amount"]==v+1)) for v in range(101)}
    return report,(guessed,confidence,accepted,correct)


def command_json(values):
    """Only fixed typed values; no transcript, name resolution or actions."""
    from .routing import validate_arguments
    family=values["intent"]
    if family=="UNKNOWN" or values["ready"]!="READY":return {"status":"rejected","reason":"unaccepted_fixed_command"}
    args={};action=None
    if family in ("POWER_ON","POWER_OFF","BRIGHTNESS","COLOR_SET"):args["target"]={"id":"example.main_lights"}
    if family in ("POWER_ON","POWER_OFF"):args["on"]=family=="POWER_ON"
    if family in ("BRIGHTNESS","VOLUME"):
        if values["amount"]=="NONE" or values["operation"]=="NONE":return {"status":"rejected","reason":"missing_finite_parameter"}
        amount=int(values["amount"])
        if values["operation"]=="SET":args["percent"]=amount;action="SET_"+family
        else:args["delta_percentage_points"]=amount*(1 if values["operation"]=="INCREASE" else -1);action="ADJUST_"+family
    if family=="COLOR_SET":
        if values["color"]=="NONE":return {"status":"rejected","reason":"missing_color"}
        args["color"]=values["color"]
    return validate_arguments(family,args,catalog(),action=action)


def predict_file(path,mode):
    lock=json.loads((SYN/"selection-before-test.json").read_text())
    if digest(DATA/"protocol.json")!=lock["protocol_sha256"]:raise ValueError("Protocol compatibility changed")
    protocol=json.loads((DATA/"protocol.json").read_text())
    if HEADS!=protocol["heads"]:raise ValueError("Fixed vocabulary compatibility changed")
    for name,expected in lock["runtime_source_sha256"].items():
        if digest(ROOT/name)!=expected:raise ValueError("Frontend/decoder source compatibility changed: "+name)
    pcm,_=load_audio(path);a=features(pcm)
    inputs={"features":rearrange(a,"c t -> 1 c t"),"lengths":np.array([a.shape[1]],dtype=np.int64)}
    probabilities={}
    for source in (["synthetic"] if mode=="synthetic" else ["synthetic","human"]):
        if digest(CACHE/source/"model.onnx")!=lock["models"][source]:raise ValueError("Model compatibility changed")
        probabilities[source]={k:softmax(v) for k,v in zip(HEADS,session(source).run(None,inputs))}
    policy=lock["calibration"][mode]["policy"]
    guessed,confidence,accepted=decode(probabilities["synthetic"],probabilities.get("human",probabilities["synthetic"]),policy)
    values={k:HEADS[k][int(v[0])] for k,v in guessed.items()}
    result=command_json(values) if accepted[0] else {"status":"rejected","reason":"confidence_or_finite_parameter_gate"}
    result.update({"family":values["intent"],"confidence":{k:float(v[0]) for k,v in confidence.items()},"experimental":True,"device_actions_executed":False})
    print(json.dumps(result,indent=2));return result


def calibrate(synthetic,human,data,mode):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    policy=standard_policy(0. if mode=="synthetic" else 1. if mode=="human" else .5);studies={}
    # A two-pass coordinate search calibrates competing family scores on dev only.
    for pass_no in range(2):
        for j,c in enumerate(HEADS["intent"][1:],1):
            base=copy.deepcopy(policy)
            def objective(trial):
                candidate=copy.deepcopy(base)
                candidate[c]={"real_weight":0. if mode=="synthetic" else 1. if mode=="human" else trial.suggest_float("real_weight",0.,1.),
                    "log_odds_bias":trial.suggest_float("log_odds_bias",-1.5,1.5),
                    "intent_threshold":trial.suggest_float("intent_threshold",.15,.95),
                    "slot_threshold":trial.suggest_float("slot_threshold",.15,.95),
                    "margin":trial.suggest_float("margin",0.,.25)}
                guessed,_,accepted=decode(synthetic,human,candidate);correct,truth=matches(guessed,data)
                chosen=accepted&(guessed["intent"]==j);n=int(chosen.sum());tp=int((chosen&correct).sum())
                neg=truth["ready"]==0;fa=int((chosen&neg).sum())
                precision=tp/n if n else 0.;far=fa/max(1,int(neg.sum()))
                feasible=n>=10 and precision>=.95 and far<=.01
                trial.set_user_attr("accepted",n);trial.set_user_attr("precision",precision);trial.set_user_attr("negative_far",far)
                trial.set_user_attr("policy",candidate[c]);trial.set_user_attr("feasible",feasible)
                if feasible:return 1.+tp/max(1,int((truth["intent"]==j).sum()))
                return -max(0.,.95-precision)-10*max(0.,far-.01)-max(0,10-n)/10
            study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=17+31*pass_no+j))
            study.optimize(objective,n_trials=100)
            best=study.best_trial
            policy[c]=best.user_attrs["policy"]
            studies[f"pass{pass_no}/{c}"]={"best_value":best.value,"best_trial":best.number,"attributes":best.user_attrs,
                "trials":[{"number":t.number,"value":t.value,"params":t.params,"attributes":t.user_attrs} for t in study.trials]}
    # Recheck final competition; unsupported intents abstain, without relaxing targets.
    guessed,_,accepted=decode(synthetic,human,policy);correct,truth=matches(guessed,data)
    gates={}
    for j,c in enumerate(HEADS["intent"][1:],1):
        emitted=accepted&(guessed["intent"]==j);n=int(emitted.sum());precision=float(correct[emitted].mean()) if n else 0.
        far=float((emitted&(truth["ready"]==0)).sum()/max(1,int((truth["ready"]==0).sum())))
        passed=n>=10 and precision>=.95 and far<=.01
        gates[c]={"passed":passed,"accepted":n,"precision":precision,"negative_far":far}
        if not passed:policy[c]["intent_threshold"]=1.1
    folder=SYN if mode=="synthetic" else MIX
    dump(folder/("human-only-optuna-development.json" if mode=="human" else "optuna-development.json"),studies)
    operation_gates={}
    guessed,_,accepted=decode(synthetic,human,policy);correct,truth=matches(guessed,data)
    for c in ("BRIGHTNESS","VOLUME"):
        j=HEADS["intent"].index(c)
        for operation in ("SET","INCREASE","DECREASE"):
            emitted=accepted&(guessed["intent"]==j)&(guessed["operation"]==HEADS["operation"].index(operation))
            n=int(emitted.sum());precision=float(correct[emitted].mean()) if n else 0.
            passed=n>=10 and precision>=.95
            operation_gates[c+"/"+operation]={"passed":passed,"accepted":n,"precision":precision,
                "available_dev_positives":int(((truth["intent"]==j)&(truth["operation"]==HEADS["operation"].index(operation))&(truth["ready"]==1)).sum())}
            if not passed:policy[c].setdefault("disabled_operations",[]).append(operation)
    # The per-family budgets above do not imply a global 1% budget. Enforce it
    # on the complete system using development only, then recheck all groups.
    candidates=[]
    for shift in np.linspace(0.,1.,21):
        candidate=copy.deepcopy(policy)
        for p in candidate.values():p["intent_threshold"]+=float(shift)
        for _ in range(16):
            guessed,_,accepted=decode(synthetic,human,candidate);correct,truth=matches(guessed,data)
            changed=False
            for j,c in enumerate(HEADS["intent"][1:],1):
                mask=accepted&(guessed["intent"]==j)
                if mask.any() and (mask.sum()<10 or correct[mask].mean()<.95):
                    candidate[c]["intent_threshold"]=1.1;changed=True
                if c in ("BRIGHTNESS","VOLUME"):
                    for operation in ("SET","INCREASE","DECREASE"):
                        m=mask&(guessed["operation"]==HEADS["operation"].index(operation))
                        if m.any() and (m.sum()<10 or correct[m].mean()<.95):
                            candidate[c].setdefault("disabled_operations",[]).append(operation);changed=True
            if not changed:break
        guessed,_,accepted=decode(synthetic,human,candidate);correct,truth=matches(guessed,data)
        neg=truth["ready"]==0;far=float((accepted&neg).sum()/max(1,int(neg.sum())))
        if far<=.01:candidates.append((int((accepted&correct).sum()),-float(shift),candidate,far))
    _,neg_shift,policy,global_far=max(candidates,key=lambda x:(x[0],x[1]))
    guessed,_,accepted=decode(synthetic,human,policy);correct,truth=matches(guessed,data)
    for j,c in enumerate(HEADS["intent"][1:],1):
        mask=accepted&(guessed["intent"]==j)
        gates[c]["final_accepted"]=int(mask.sum());gates[c]["final_precision"]=float(correct[mask].mean()) if mask.any() else None
        gates[c]["passed"]=bool(mask.sum()>=10 and correct[mask].mean()>=.95) if mask.any() else False
    for key,gate in operation_gates.items():
        c,operation=key.split("/");mask=accepted&(guessed["intent"]==HEADS["intent"].index(c))&(guessed["operation"]==HEADS["operation"].index(operation))
        gate["final_accepted"]=int(mask.sum());gate["final_precision"]=float(correct[mask].mean()) if mask.any() else None
        gate["passed"]=bool(mask.sum()>=10 and correct[mask].mean()>=.95) if mask.any() else False
    return {"policy":policy,"development_gates":gates,"operation_gates":operation_gates,"selection_split":"human_dev", "trials":1800,
            "global_negative_gate":{"passed":global_far<=.01,"false_accept_rate":global_far,"threshold_shift":-neg_shift,"development_only":True},
            "targets":{"precision":.95,"negative_false_accept_rate":.01,"minimum_accepts":10},"test_used":False}


def baseline():
    path=MIX/"human-only-selection-before-test.json"
    signature={"model_sha256":digest(CACHE/"human/model.onnx"),"manifest_sha256":digest(DATA/"human.jsonl"),
               "selection_specification_sha256":digest(SYN/"selection-specification.json")}
    if path.exists():
        result=json.loads(path.read_text())
        if result["signature"]!=signature:raise ValueError("Human control changed; preserve the old protocol")
        return result
    data=[r for r in rows(DATA/"human.jsonl") if r["split"]=="dev"]
    h=inference("human",data,"human-dev")
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"signature":signature,"calibration":calibrate(h,h,data,"human")}
    dump(path,result)
    print("Human-only control selected on development; no test scoring",flush=True)
    return result


def check():
    import torch
    from .synthetic_vcm import FixedVCM
    torch.set_num_threads(2)
    protocol=json.loads((DATA/"protocol.json").read_text());human=rows(DATA/"human.jsonl");synthetic=rows(DATA/"manifest.jsonl")
    assert protocol["heads"]==HEADS and protocol["keyword_vocabulary"]==WORDS
    result={"manifest_checks":{},"onnx_parity":{},"latency":{}}
    for filename,expected in protocol["manifest_sha256"].items():
        assert digest(DATA/filename)==expected,filename
    for name,data in [("human",human),("synthetic",synthetic)]:
        sets={s:{r["speaker_id"] for r in data if r["split"]==s} for s in ("train","dev","test")}
        assert not (sets["train"]&sets["dev"] or sets["train"]&sets["test"] or sets["dev"]&sets["test"])
        hashes={s:{r["audio_sha256"] for r in data if r["split"]==s} for s in sets}
        assert not (hashes["train"]&hashes["test"] or hashes["train"]&hashes["dev"] or hashes["dev"]&hashes["test"])
        for r in data:
            assert digest(ROOT/r["audio"])==r["audio_sha256"]
            for span in r["keyword_spans"]:assert r["transcript"][span["start"]:span["end"]]==span["text"]
        result["manifest_checks"][name]={"n":len(data),"speakers_disjoint":True,"hashes_disjoint":True,"audio_hashes_verified":True,"keyword_spans_verified":True}
    # Text parser verification is explicitly separate from spoken accuracy.
    for family in ("BRIGHTNESS","VOLUME"):
        for value in range(101):
            for suffix in (""," percent"):
                assert label(family,f"set {family.lower()} to {value}{suffix}")["amount"]==str(value)
        assert label(family,f"{family.lower()} up")["amount"]=="10"
        assert label(family,f"{family.lower()} down")["operation"]=="DECREASE"
        for text in ("minus five","one hundred and one","fifty point five","twenty thirty"):
            assert label(family,f"set {family.lower()} to {text}")["ready"]=="REJECT"
    result["parser_checks"]={"absolute_text_cases":404,"spoken_cases":0,"relative_and_invalid_cases_passed":True}
    routing_cases=0
    for family in ("BRIGHTNESS","VOLUME"):
        for operation in HEADS["operation"]:
            for amount in HEADS["amount"]:
                values={"intent":family,"ready":"READY","operation":operation,"amount":amount,"color":"NONE"}
                p={k:np.ones((1,len(v)))*.00001 for k,v in HEADS.items()}
                for k,v in values.items():p[k][0,HEADS[k].index(v)]=.999
                assert bool(decode(p,p,standard_policy(0.))[2][0])==(command_json(values)["status"]=="accepted")
                routing_cases+=1
    result["finite_decoder_checks"]={"numeric_head_combinations":routing_cases,"typed_validator_parity":True,"spoken_cases":0}
    engines={}
    for source in ("synthetic","human"):
        saved=torch.load(CACHE/source/"best.pt",map_location="cpu",weights_only=True)
        model=FixedVCM().eval();model.load_state_dict(saved["state"])
        started=time.perf_counter();engine=session(source);load_ms=(time.perf_counter()-started)*1000;engines[source]=engine
        errors=[]
        for b,t in ((1,37),(1,400),(3,223)):
            x=torch.randn(b,40,t);lengths=torch.tensor([max(1,t-31*i) for i in range(b)])
            with torch.inference_mode():expected=model(x,lengths)
            actual=engine.run(None,{"features":x.numpy(),"lengths":lengths.numpy()})
            for a,e in zip(actual,expected):np.testing.assert_allclose(a,e.numpy(),atol=2e-4,rtol=2e-4);errors.append(float(np.max(np.abs(a-e.numpy()))))
        result["onnx_parity"][source]={"max_abs_error":max(errors),"variable_length_and_batch":True}
        pcm=[load_audio(ROOT/r["audio"])[0] for r in human if r["split"]=="dev"][:60]
        for x in pcm[:5]:a=features(x);engine.run(None,{"features":rearrange(a,"c t -> 1 c t"),"lengths":np.array([a.shape[1]],dtype=np.int64)})
        times=[]
        for x in pcm:
            started=time.perf_counter();a=features(x);raw=engine.run(None,{"features":rearrange(a,"c t -> 1 c t"),"lengths":np.array([a.shape[1]],dtype=np.int64)})
            p={k:softmax(v) for k,v in zip(HEADS,raw)};decode(p,p,standard_policy(.5));times.append((time.perf_counter()-started)*1000)
        result["latency"][source]={"n":len(times),"p50_ms":float(np.percentile(times,50)),"p95_ms":float(np.percentile(times,95)),
            "session_load_ms":load_ms,"onnx_threads":2,"includes":"frontend, ONNX, finite decoding", "excludes":"disk, speech duration, energy VAD, 400 ms endpoint wait", "device":"desktop CPU"}
    times=[]
    for x in pcm:
        started=time.perf_counter();a=features(x);inputs={"features":rearrange(a,"c t -> 1 c t"),"lengths":np.array([a.shape[1]],dtype=np.int64)}
        probs={s:{k:softmax(v) for k,v in zip(HEADS,engine.run(None,inputs))} for s,engine in engines.items()}
        guess,_,accept=decode(probs["synthetic"],probs["human"],standard_policy(.5))
        if accept[0]:command_json({k:HEADS[k][int(v[0])] for k,v in guess.items()})
        times.append((time.perf_counter()-started)*1000)
    result["latency"]["mixed"]={"n":len(times),"p50_ms":float(np.percentile(times,50)),"p95_ms":float(np.percentile(times,95)),
        "session_load_ms":sum(result["latency"][s]["session_load_ms"] for s in engines),"onnx_threads_per_session":2,
        "includes":"one frontend, two sequential ONNX experts, probability blend, finite decoding and typed validation",
        "excludes":"disk, speech duration, energy VAD, 400 ms endpoint wait", "device":"desktop CPU", "identical_preloaded_pcm":True}
    for folder in (SYN,MIX):dump(folder/"checks.json",result)
    print(json.dumps(result),flush=True);return result


def evaluate():
    for folder in (SYN,MIX):
        if (folder/"results.json").exists():raise ValueError("Test already scored. Preserve evidence and create a new study version to rerun.")
    human=rows(DATA/"human.jsonl");synth=rows(DATA/"manifest.jsonl")
    development=[r for r in human if r["split"]=="dev"]
    s=inference("synthetic",development,"human-dev");h=inference("human",development,"human-dev")
    calibrations={mode:calibrate(s,h,development,mode) for mode in ("synthetic","mixed")}
    control=baseline()
    lock={"created_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":digest(DATA/"protocol.json"),"human_manifest_sha256":digest(DATA/"human.jsonl"),"tts_manifest_sha256":digest(DATA/"manifest.jsonl"),
        "models":{mode:digest(CACHE/mode/"model.onnx") for mode in ("synthetic","human")},"calibration":calibrations,"test_scored_before_receipt":False,
        "implementation_sha256":{str(p.relative_to(ROOT)):digest(p) for folder in ("models","train","test") for p in (ROOT/"scripts"/folder).glob("synthetic_*.py")},
        "selection_specification_sha256":digest(SYN/"selection-specification.json"),
        "source_metadata_sha256":digest(DATA/"source-metadata.json"),
        "runtime_source_sha256":{f:digest(ROOT/f) for f in ("scripts/runtime/audio.py","scripts/runtime/routing.py","scripts/runtime/levels.py")},
        "human_only_control_selection_sha256":digest(MIX/"human-only-selection-before-test.json")}
    for folder in (SYN,MIX):dump(folder/"selection-before-test.json",lock)
    groups={"human_test":[r for r in human if r["split"]=="test"],"synthetic_test":[r for r in synth if r["split"]=="test"]}
    outcomes={mode:{"development":score(s,h,calibrations[mode]["policy"],development)[0],"calibration":calibrations[mode],"tests":{}} for mode in calibrations}
    control_results={"calibration":control["calibration"],"tests":{}}
    for name,data in groups.items():
        sp=inference("synthetic",data,name);hp=inference("human",data,name)
        control_results["tests"][name]=score(hp,hp,control["calibration"]["policy"],data)[0]
        for mode in outcomes:
            metrics,detail=score(sp,hp,calibrations[mode]["policy"],data);outcomes[mode]["tests"][name]=metrics
            raw_policy=copy.deepcopy(calibrations[mode]["policy"])
            for p in raw_policy.values():
                p.update(intent_threshold=0.,slot_threshold=0.,margin=0.);p.pop("disabled_operations",None)
            outcomes[mode].setdefault("before_confidence_gates",{})[name]=score(sp,hp,raw_policy,data)[0]
            guessed,conf,accepted,correct=detail
            predictions=[{"id":r["id"],"labels":r["labels"],"guessed":{k:HEADS[k][int(v[i])] for k,v in guessed.items()},
                          "confidence":{k:float(v[i]) for k,v in conf.items()},"accepted":bool(accepted[i]),"correct":bool(correct[i]),"source":r["source"]} for i,r in enumerate(data)]
            dump((SYN if mode=="synthetic" else MIX)/(name+"-predictions.json"),predictions)
    for mode,outcome in outcomes.items():
        outcome.update({"all_intents_passed_development_gate":all(v["passed"] for v in outcome["calibration"]["development_gates"].values()),
            "all_numeric_operations_passed_development_gate":all(v["passed"] for v in outcome["calibration"]["operation_gates"].values()),
            "new_words_tested":False,"live_microphone_tested":False,"physical_pi_tested":False,"accent_verified":False,
            "labels":"Source annotations and programmatic silver references; TTS labels are intended text.","installed_checkpoint_modified":False})
        dump((SYN if mode=="synthetic" else MIX)/"results.json",outcome)
    dump(MIX/"human-only-results.json",control_results)
    report()
    print(json.dumps({m:{n:v["overall"] for n,v in o["tests"].items()} for m,o in outcomes.items()}),flush=True)


def report():
    for mode,folder in (("synthetic",SYN),("mixed",MIX)):
        result=json.loads((folder/"results.json").read_text());data=json.loads((DATA/"protocol.json").read_text())
        lines=[f"# {mode.title()} fixed-command VCM experiment", "", "Research-only results; installed VCM/STT and previous human-only studies are preserved.", "",
            "Synthetic acoustic expert uses only generated training audio. Mixed uses separately trained human and synthetic experts with per-intent source preference; it is a calibrated ensemble, not a jointly fitted single model.","",
            "Weights were selected on source-specific development. Both systems' acceptance calibration uses human development; no final test labels were used by Optuna.","",
            f"TTS utterances: {sum(data['synthetic_counts'].values())}; human recordings: {sum(data['human_counts'].values())}. Qwen3-TTS 1.7B VoiceDesign creates original anchor voices; the 1.7B Base model produces whole English utterances. Filipino accents are requested by prompt and remain unverified.","",
            "| Test | N | Accepted | Correct accepted | Precision | Exact command recall | Negative false accepts |", "|---|---:|---:|---:|---:|---:|---:|"]
        def pct(x):return "n/a" if x is None else f"{100*x:.2f}%"
        for name,v in result["tests"].items():
            a=v["overall"];lines.append(f"| {name} | {a['n']} | {a['accepted']} | {a['correct_accepted']} | {pct(a['accepted_precision'])} | {pct(a['exact_command_recall'])} | {a['false_accepts']}/{a['negative_n']} |")
        lines += ["","## Human test by intent","","| Intent | N | Accepted | Precision | Exact recall | Dev gate |","|---|---:|---:|---:|---:|---|"]
        for c,v in result["tests"]["human_test"]["per_intent"].items():
            gate=result["calibration"]["development_gates"].get(c,{}).get("passed")
            lines.append(f"| {c} | {v['n']} | {v['accepted']} | {pct(v['accepted_precision'])} | {pct(v['exact_command_recall'])} | {gate} |")
        lines += ["","## Confidence and source preference","","`real_weight=0` uses the synthetic expert; `1` uses the human expert. Values between them blend acoustic probabilities. `log_odds_bias` changes the family prior. Intent, slot and margin thresholds must pass before emitting a command. Parameter heads cannot change the selected family.","",
            "Optuna ran 1,800 development trials per system. Targets were at least 95% accepted precision, at most 1% negative false acceptance and at least ten development accepts per intent. An intent without support is disabled, not promoted by reducing the gate.","","```json",json.dumps(result["calibration"]["policy"],indent=2),"```","",
            "## Coverage and limits","","All nine supported fixed intent families and UNKNOWN are in the manifest. Amount classes contain every integer 0-100. Bare UP/DOWN means ten percentage points. Per-value test counts and numeric/relative strata are in results.json; unobserved human values are not proven by parser coverage.","",
            "Known-corpus original human speech and held-out synthetic voices are separate tests. Names, artists, songs and new-word corpora are excluded. Silver labels are not human adjudication. No Filipino-human accent benchmark, live microphone, endpointed full-pipeline run or Raspberry Pi measurement is claimed.","",
            "## New words within fixed intents","","A fixed softmax cannot learn acoustic words from a JSON alias alone. The next supported research path is labeled audio enrollment into an embedding prototype bank, with each prototype bound to an existing intent/finite value. New wording changes exemplars, not output classes. Version the enrollment set, require multiple independent examples, and recalibrate the margin/unknown gate on development before activation. A phonetic keyword graph is another option using reviewed pronunciations and a validated phoneme encoder. Neither path is enabled or claimed validated by this fixed-corpus experiment.","",
            "## Reproduce","","Run from the root environment:","```powershell",".venv/Scripts/python.exe experiments/synthetic/acquire_tts.py",".venv/Scripts/python.exe scripts/synthetic.py prepare",".venv/Scripts/python.exe scripts/synthetic.py generate --batch-size 16",".venv/Scripts/python.exe scripts/synthetic.py train",".venv/Scripts/python.exe scripts/synthetic.py check",".venv/Scripts/python.exe scripts/synthetic.py evaluate","```","",
            "Reports and selection receipts are here; generated recordings are in data/audio/tts, manifests in data/tts, and model/feature artifacts in .cache. Test evaluation refuses to overwrite an existing result.","",
            "Sources: [Qwen models](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), [official inference API](https://github.com/QwenLM/Qwen3-TTS)."]
        checks=json.loads((folder/"checks.json").read_text()) if (folder/"checks.json").exists() else None
        if checks:
            timing=checks["latency"][mode]
            lines += ["","## Desktop CPU timing","",f"Identical preloaded development PCM, two ONNX threads: p50 {timing['p50_ms']:.2f} ms, p95 {timing['p95_ms']:.2f} ms. Session loading: {timing['session_load_ms']:.2f} ms.","",
                "These measurements include the frontend and finite decoder; mixed also includes both acoustic experts and probability blending. Disk I/O, speech duration, energy VAD and the 400 ms endpoint wait are excluded. They do not establish Raspberry Pi or microphone performance."]
        lines += ["","## Before confidence gates","","| Test | Head-based precision | Exact command recall | Accepted |","|---|---:|---:|---:|"]
        for name,metrics in result["before_confidence_gates"].items():
            v=metrics["overall"];lines.append(f"| {name} | {pct(v['accepted_precision'])} | {pct(v['exact_command_recall'])} | {v['accepted']} |")
        lines += ["","The row above retains the learned readiness decision and finite-value validity but removes confidence, margin and disabled-operation gates. It is a diagnostic result, not an alternative calibrated operating point.","",
            "## Numeric operation support gates","","```json",json.dumps(result["calibration"]["operation_gates"],indent=2),"```"]
        lines += ["","## Held-out synthetic voices by requested accent","","Accent names below describe generator prompts. They are not verified accent labels or a Filipino-human benchmark.","",
                  "| Requested accent | N | Accepted | Precision | Exact recall |","|---|---:|---:|---:|---:|"]
        for name,m in result["tests"]["synthetic_test"]["per_requested_accent"].items():
            lines.append(f"| {name} | {m['n']} | {m['accepted']} | {pct(m['accepted_precision'])} | {pct(m['exact_command_recall'])} |")
        model_names=["synthetic"] if mode=="synthetic" else ["synthetic","human"]
        training=[json.loads((CACHE/name/"complete.json").read_text()) for name in model_names]
        lines += ["","## Model size","",f"{sum(t['parameters'] for t in training):,} trainable parameters across {len(training)} acoustic model(s); {sum(t['onnx_bytes'] for t in training):,} ONNX bytes. TTS checkpoints and PyTorch are not part of this CPU inference path."]
        if mode=="mixed" and (MIX/"human-only-results.json").exists():
            control=json.loads((MIX/"human-only-results.json").read_text())
            lines += ["","## Human-only control","","This control uses the same human expert, development budgets and acceptance targets, with real_weight fixed to one. It measures the contribution of generated speech to the ensemble.","",
                "| Test | Accepted | Precision | Exact command recall | Negative false accepts |","|---|---:|---:|---:|---:|"]
            for name,metrics in control["tests"].items():
                m=metrics["overall"];lines.append(f"| {name} | {m['accepted']} | {pct(m['accepted_precision'])} | {pct(m['exact_command_recall'])} | {m['false_accepts']}/{m['negative_n']} |")
        (folder/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def main(command):
    {"check":check,"baseline":baseline,"evaluate":evaluate,"report":report}[command]()
