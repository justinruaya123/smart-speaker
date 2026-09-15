"""Held-out human-audio evaluation. Final test scores never select a checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from einops import rearrange, reduce
import torch

from .audio import load_audio, features
from .data import ROOT, read_jsonl, write_json, write_jsonl, stable_number, normalize_text
from .export import normalized, session
from .inference import Recognizer
from .routing import PARAMETER_COMMANDS, parameter_slice, parse_parameters
from .train import validate_asr, edit_distance, warm_cache
from .vcm import VCM


def probabilities(logits):
    x = logits.astype(np.float64)
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def first_trigger(trace, threshold, labels, stability=3, minimum=0.5):
    previous, stable = None, 0
    for event in trace:
        index, confidence, timestamp = event
        stable = (stable + 1 if index == previous else 1) if confidence >= threshold else 0
        previous = index
        if stable >= stability and labels[index] != "UNKNOWN" and timestamp >= minimum:
            return {"vcm_command": labels[index], "confidence": confidence, "audio_time_s": timestamp}
    return None


@torch.inference_mode()
def prefix_trace(model, x):
    logits = model.prefix_logits(torch.from_numpy(x).to("cuda"))[0].cpu().numpy()
    ends = list(range(3, x.shape[-1], 4))
    if not ends or ends[-1] != x.shape[-1] - 1:
        ends.append(x.shape[-1] - 1)
    probs = probabilities(logits[ends])
    return [(int(p.argmax()), float(p.max()), (frame * 0.01 + 0.025)) for p, frame in zip(probs, ends)]


def score(outputs, labels, threshold=0):
    lookup = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    accepted = correct_accepted = false_accepts = positive = positive_correct = unknown = 0
    for item in outputs:
        gold, pred = item["gold"], item["predicted"]
        confusion[lookup[gold], lookup[pred]] += 1
        yes = pred != "UNKNOWN" and item["confidence"] >= threshold
        positive += gold != "UNKNOWN"
        unknown += gold == "UNKNOWN"
        accepted += yes
        correct_accepted += yes and pred == gold
        positive_correct += yes and pred == gold and gold != "UNKNOWN"
        false_accepts += yes and gold == "UNKNOWN"
    tp = np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(labels)), where=denominator != 0)
    per_class = {label: {"samples": int(confusion[i].sum()), "correct": int(tp[i]), "recall": float(tp[i] / max(1, confusion[i].sum())), "f1": float(f1[i])} for i, label in enumerate(labels)}
    return {"samples": len(outputs), "accuracy": float(tp.sum() / len(outputs)), "macro_f1": float(f1[confusion.sum(1) > 0].mean()), "per_class": per_class, "confusion": confusion.tolist(), "threshold": threshold, "accepted": int(accepted), "correct_accepted": int(correct_accepted), "acceptance_coverage": accepted / len(outputs), "accepted_precision": correct_accepted / max(1, accepted), "positive_recall_after_rejection": positive_correct / max(1, positive), "unknown_samples": int(unknown), "unknown_false_accepts": int(false_accepts), "unknown_false_accept_rate": false_accepts / max(1, unknown)}


def grouped_accuracy_interval(outputs, draws=1000):
    groups = defaultdict(list)
    for item in outputs:
        groups[item["paired_group"]].append(item["predicted"] == item["gold"])
    values = list(groups.values())
    successes = np.asarray([sum(g) for g in values])
    totals = np.asarray([len(g) for g in values])
    rng = np.random.default_rng(1731)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(values), len(values))
        estimates.append(successes[indices].sum() / totals[indices].sum())
    return {"method": "paired-recording cluster bootstrap; does not eliminate shared-speaker dependence", "groups": len(values), "draws": draws, "lower": float(np.quantile(estimates, 0.025)), "upper": float(np.quantile(estimates, 0.975))}


def early_metrics(outputs, threshold, labels):
    triggered = correct = wrong = negative_triggered = positive = unknown = 0
    fractions, times = [], []
    for item in outputs:
        event = first_trigger(item["trace"], threshold, labels)
        is_positive = item["gold"] != "UNKNOWN"
        positive += is_positive
        unknown += not is_positive
        if event:
            triggered += 1
            right = event["vcm_command"] == item["gold"]
            correct += right
            wrong += not right
            negative_triggered += not is_positive
            if right:
                times.append(event["audio_time_s"])
                fractions.append(event["audio_time_s"] / item["duration"])
    return {"samples": len(outputs), "threshold": threshold, "triggered": int(triggered), "correct": int(correct), "wrong": int(wrong), "precision": correct / max(1, triggered), "positive_trigger_recall": correct / max(1, positive), "wrong_trigger_rate_all": wrong / len(outputs), "unknown_trigger_rate": negative_triggered / max(1, unknown), "median_correct_trigger_s": float(np.median(times)) if times else None, "median_fraction_of_recording": float(np.median(fractions)) if fractions else None, "boundary_accuracy": None, "boundary_note": "No verified timing labels; these are command prediction times, not parameter-onset measurements."}


def run_vcm(bundle):
    metadata = json.loads((bundle / "bundle.json").read_text())
    ck = torch.load(ROOT / metadata["vcm"]["checkpoint"], map_location="cpu", weights_only=False)
    model = VCM(**ck["model_config"])
    model.load_state_dict(ck["model"])
    model.eval().to("cuda")
    labels = ck["labels"]
    stats = ck["normalization"]
    runtimes = {variant: session(bundle / f"vcm.{variant}.onnx") for variant in ("fp32", "int8")}
    all_rows = read_jsonl(ROOT / "data/manifests/vcm.jsonl")
    results = {}
    for split in ("calibration", "test"):
        rows = [r for r in all_rows if r["split"] == split]
        outputs = {variant: [] for variant in runtimes}
        started = time.monotonic()
        for index, row in enumerate(rows):
            x = normalized(row, stats)
            feed = {"features": x, "lengths": np.array([x.shape[-1]], dtype=np.int64)}
            trace = prefix_trace(model, x)
            common = {"id": row["id"], "gold": row["command"], "transcript": row["transcript"], "audio": row["audio"], "duration": row["duration"], "paired_group": row["paired_group"], "speaker_id": row["speaker_id"], "microphone": "headset" if "headset" in row["audio"] else "distant"}
            for variant, runtime in runtimes.items():
                p = probabilities(runtime.run(None, feed)[0][0])
                outputs[variant].append({**common, "predicted": labels[int(p.argmax())], "confidence": float(p.max()), **({"trace": trace} if variant == "fp32" else {})})
            if (index + 1) % 400 == 0:
                print(f"VCM {split} {index + 1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if split == "calibration":
            thresholds = [float(t) for t in np.arange(0.5, 1.0, 0.01)] + [0.995, 0.999, 1.0]
            table = [score(outputs["fp32"], labels, t) for t in thresholds]
            feasible = [r for r in table if r["unknown_false_accept_rate"] <= 0.01 and r["accepted_precision"] >= 0.95 and r["accepted"] > 0]
            chosen = max(feasible, key=lambda r: r["correct_accepted"]) if feasible else table[-1]
            early_table = [early_metrics(outputs["fp32"], t, labels) for t in thresholds if t >= 0.8]
            early_feasible = [r for r in early_table if r["unknown_trigger_rate"] <= 0.01 and r["wrong_trigger_rate_all"] <= 0.01 and r["triggered"] > 0]
            early = max(early_feasible, key=lambda r: r["correct"]) if early_feasible else early_table[-1]
            metadata["routing"].update(acceptance_threshold=chosen["threshold"], trigger_threshold=early["threshold"])
            write_json(bundle / "bundle.json", metadata)
            write_json(ROOT / "reports/threshold_calibration.json", {"partition": "calibration only", "final_rule": "maximize correctly accepted commands subject to unknown false accepts <=1% and accepted precision >=95%; reject-all if infeasible", "early_rule": "maximize correct triggers subject to <=1% unknown triggers and <=1% wrong triggers over all calibration audio; disable if infeasible", "selected_final": chosen, "selected_early": early, "final_table": [{k: v for k, v in r.items() if k not in ("confusion", "per_class")} for r in table], "early_table": early_table})
        threshold = metadata["routing"]["acceptance_threshold"]
        results[split] = {variant: score(items, labels, threshold) for variant, items in outputs.items()}
        results[split]["early_fp32"] = early_metrics(outputs["fp32"], metadata["routing"]["trigger_threshold"], labels)
        for variant, items in outputs.items():
            write_jsonl(ROOT / f"reports/predictions_vcm_{split}_{variant}.jsonl", items)
        if split == "test":
            results[split]["fp32"]["accuracy_interval"] = grouped_accuracy_interval(outputs["fp32"])
            results[split]["quantization_agreement"] = sum(a["predicted"] == b["predicted"] for a, b in zip(outputs["fp32"], outputs["int8"])) / len(rows)
            groups = {}
            for microphone in ("headset", "distant"):
                groups[microphone] = score([o for o in outputs["fp32"] if o["microphone"] == microphone], labels, threshold)
            for name, codes in (("native_english_metadata", ("FE", "ME")), ("other_native_language_metadata", ("FO", "MO")), ("unknown_language_metadata", ("UNK",))):
                subset = [o for o in outputs["fp32"] if o["speaker_id"].split("/")[-1].split("-")[0] in codes]
                if subset:
                    groups[name] = score(subset, labels, threshold)
            results[split]["subgroups"] = groups
    results["labels"] = labels
    results["test_protocol"] = "Untouched curated SLURP-real test cohort, after development-only model selection and calibration-only thresholds. Source partitions share speakers; microphone pairs crossing partitions were quarantined."
    write_json(ROOT / "reports/vcm_metrics.json", results)
    # Predeclared voice suite: up to six independent recordings per class, hash order.
    seen, selected, counts = set(), [], Counter()
    for row in sorted([r for r in all_rows if r["split"] == "test"], key=lambda r: stable_number(r["id"])):
        if row["paired_group"] not in seen and counts[row["command"]] < 6:
            selected.append(row)
            seen.add(row["paired_group"])
            counts[row["command"]] += 1
    write_jsonl(ROOT / "data/manifests/voice_test_suite.jsonl", selected)
    run_noise(bundle, all_rows)
    print("VCM TEST " + json.dumps({k: v for k, v in results["test"]["fp32"].items() if k not in ("confusion", "per_class")}), flush=True)


def run_noise(bundle, all_rows):
    recognizer = Recognizer(bundle)
    rows, counts = [], Counter()
    for row in sorted([r for r in all_rows if r["split"] == "test"], key=lambda r: stable_number(r["id"])):
        if counts[row["command"]] < 20:
            rows.append(row)
            counts[row["command"]] += 1
    reports = {}
    for snr in (None, 20, 10):
        outputs = []
        for row in rows:
            pcm, _ = load_audio(ROOT / row["audio"])
            if snr is not None:
                rng = np.random.default_rng(stable_number(row["id"]) % 2**32)
                noise = rng.normal(size=len(pcm)).astype(np.float32)
                noise *= np.sqrt(reduce(pcm * pcm, "t ->", "mean") / max(1e-10, reduce(noise * noise, "t ->", "mean"))) / 10 ** (snr / 20)
                pcm = pcm + noise
            p = recognizer.classify(pcm)
            outputs.append({"gold": row["command"], "predicted": recognizer.labels[int(p.argmax())], "confidence": float(p.max())})
        reports["original" if snr is None else f"white_noise_{snr}db"] = score(outputs, recognizer.labels, recognizer.metadata["routing"]["acceptance_threshold"])
    negatives = []
    rng = np.random.default_rng(55)
    for index in range(30):
        pcm = np.zeros(16000 * (1 + index % 4), dtype=np.float32) if index < 10 else rng.normal(0, 0.005 + index * 0.002, 32000).astype(np.float32)
        p = recognizer.classify(pcm)
        negatives.append({"gold": "UNKNOWN", "predicted": recognizer.labels[int(p.argmax())], "confidence": float(p.max())})
    reports["silence_and_noise_only"] = score(negatives, recognizer.labels, recognizer.metadata["routing"]["acceptance_threshold"])
    reports["interpretation"] = "Simulated additive white noise and silence; source distant-microphone results are reported separately. These are not Raspberry Pi or real-room noise measurements."
    write_json(ROOT / "reports/noise_metrics.json", reports)


def run_asr(bundle):
    from .asr import ParameterASR
    metadata = json.loads((bundle / "bundle.json").read_text())
    ck = torch.load(ROOT / metadata["asr"]["checkpoint"], map_location="cpu", weights_only=False)
    model = ParameterASR(**ck["model_config"])
    model.load_state_dict(ck["model"])
    model.to("cuda").eval()
    rows = read_jsonl(ROOT / "data/manifests/asr.jsonl")
    results = {}
    for source in ("LibriSpeech", "SLURP-real"):
        subset = [r for r in rows if r["split"] == "test" and r["source"] == source]
        warm_cache(subset, memory_mib=4096)
        metrics, predictions = validate_asr(model, subset, ck["normalization"], "cuda", batch_size=32)
        results[source] = metrics
        write_jsonl(ROOT / f"reports/predictions_asr_{source}.jsonl", predictions)
        print(f"ASR TEST {source} {json.dumps(metrics)}", flush=True)
    results["decoding"] = "greedy character CTC; no language model, cloud API, or VCM checkpoint"
    write_json(ROOT / "reports/asr_metrics.json", results)
    run_voice_suite(bundle)
    run_joint(bundle)


def expected_entities(row):
    tokens = {t["id"]: t["surface"] for t in row.get("tokens", [])}
    return [{"type": entity["type"], "text": normalize_text(" ".join(tokens[i] for i in entity["span"] if i in tokens))} for entity in row.get("entities", [])]


def run_voice_suite(bundle):
    rows = read_jsonl(ROOT / "data/manifests/voice_test_suite.jsonl")
    outputs = []
    recognizers = {variant: Recognizer(bundle, asr=True, variant=variant) for variant in ("fp32", "int8")}
    # One configurable example catalog, not inferred from each test transcript.
    catalog = json.loads((ROOT / "configs/catalog.example.json").read_text())
    for recognizer in recognizers.values():
        recognizer.catalog = catalog
    for index, row in enumerate(rows):
        pcm, _ = load_audio(ROOT / row["audio"])
        item = {"id": row["id"], "audio": row["audio"], "reference": row["transcript"], "gold": row["command"], "duration": row["duration"], "expected_entities": expected_entities(row)}
        item["reference_parser_result"] = parse_parameters(row["command"], row["transcript"], catalog)
        for variant, recognizer in recognizers.items():
            result = recognizer.predict(pcm)
            full_text = recognizer.transcribe(pcm)
            reference = normalize_text(row["transcript"])
            item[variant] = {"result": result, "full_audio_asr": full_text, "full_audio_word_errors": edit_distance(reference.split(), full_text.split()), "reference_words": len(reference.split())}
            item[variant]["oracle_intent_full_audio_parser"] = parse_parameters(row["command"], full_text, catalog)
            tail_text = result.get("parameter_transcript", "")
            entities = [e for e in item["expected_entities"] if e["text"]]
            item[variant]["entity_text_retained_in_tail"] = sum(e["text"] in normalize_text(tail_text) for e in entities)
            item[variant]["entity_count"] = len(entities)
        outputs.append(item)
        if (index + 1) % 10 == 0:
            print(f"VOICE SUITE {index + 1}/{len(rows)}", flush=True)
    write_jsonl(ROOT / "reports/voice_test_results.jsonl", outputs)
    results = {"suite_rule": "up to six paired-recording-distinct test clips per class, deterministic ID hash order", "samples": len(outputs), "catalog": "configs/catalog.example.json; explicit example bindings, no physical device actions", "boundary_note": "Entity text retention is an end-to-end diagnostic. It confounds timing, ASR, and annotations, and is not a gold-tail WER or onset accuracy measurement."}
    for variant in recognizers:
        result_rows = [o[variant] for o in outputs]
        count = sum(o["entity_count"] for o in result_rows)
        correct_intent = sum(o[variant]["result"].get("vcm_command") == o["gold"] for o in outputs)
        statuses = Counter(o["result"]["status"] for o in result_rows)
        eligible = [o for o in outputs if o["reference_parser_result"].get("status") == "accepted"]
        frame_exact = sum(o[variant]["result"].get("status") == "accepted" and o[variant]["result"].get("command") == o["reference_parser_result"]["command"] for o in eligible)
        oracle_exact = sum(o[variant]["oracle_intent_full_audio_parser"].get("status") == "accepted" and o[variant]["oracle_intent_full_audio_parser"].get("command") == o["reference_parser_result"]["command"] for o in eligible)
        stages = Counter()
        for item in outputs:
            actual = item[variant]["result"]
            reference = item["reference_parser_result"]
            if item["gold"] == "UNKNOWN":
                stage = "unsupported_rejected" if actual["status"] == "rejected" else "unsupported_not_rejected"
            elif actual.get("vcm_command") != item["gold"]:
                stage = "wrong_raw_intent"
            elif actual["status"] == "rejected":
                stage = actual.get("reason", "intent_rejected")
            elif actual["status"] == "needs_parameters":
                stage = actual.get("reason", "parameter_audio_unavailable")
            elif actual["status"] == "needs_clarification":
                stage = "parameter_parse_or_target_failed"
            elif reference.get("status") != "accepted":
                stage = "reference_parser_not_scored"
            elif actual.get("command") == reference.get("command"):
                stage = "exact_reference_parser_frame"
            else:
                stage = "wrong_reference_parser_arguments"
            stages[stage] += 1
        results[variant] = {"statuses": dict(statuses), "stage_counts": dict(stages), "correct_vcm_label": correct_intent, "full_audio_wer": sum(o["full_audio_word_errors"] for o in result_rows) / max(1, sum(o["reference_words"] for o in result_rows)), "entity_count": count, "tail_entity_text_recall": sum(o["entity_text_retained_in_tail"] for o in result_rows) / max(1, count), "tail_asr_invocations": sum("parameter_transcript" in o["result"] for o in result_rows), "parser_reference_eligible": len(eligible), "parser_reference_exact_frames": frame_exact, "parser_reference_exact_rate": frame_exact / max(1, len(eligible)), "oracle_intent_full_audio_exact_frames": oracle_exact, "oracle_control_note": "Diagnostic control supplies the gold command only to the parser after transcribing whole audio. No gold text/intent is injected into the measured production pipeline."}
    write_json(ROOT / "reports/voice_suite_metrics.json", results)


def run_joint(bundle):
    """Replay the complete VCM test cohort through the actual optional-ASR path."""
    rows = [r for r in read_jsonl(ROOT / "data/manifests/vcm.jsonl") if r["split"] == "test"]
    catalog = json.loads((ROOT / "configs/catalog.example.json").read_text())
    recognizers = {variant: Recognizer(bundle, asr=True, variant=variant) for variant in ("fp32", "int8")}
    for recognizer in recognizers.values():
        recognizer.catalog = catalog
    outputs = []
    for index, row in enumerate(rows):
        pcm, _ = load_audio(ROOT / row["audio"])
        # Gold text/command enter this scoring reference only, never predict().
        reference = parse_parameters(row["command"], row["transcript"], catalog)
        outputs.append({"id": row["id"], "gold": row["command"], "transcript": row["transcript"], "reference_parser_result": reference,
                        **{variant: recognizer.predict(pcm) for variant, recognizer in recognizers.items()}})
        if (index + 1) % 500 == 0:
            print(f"JOINT VOICE TEST {index+1}/{len(rows)}", flush=True)
    summary = {"samples": len(rows), "catalog": "configs/catalog.example.json, fixed across all examples", "scoring_note": "Exact-frame reference is the deterministic parser applied to curated gold intent and human transcript. It is a silver reference, not manually adjudicated gold slots. Known examples whose reference cannot be resolved by the fixed catalog are reported as unscored. All recordings pass through actual CPU ONNX predict(), with ASR only on the predicted tail when routing permits it."}
    eligible = [o for o in outputs if o["gold"] != "UNKNOWN" and o["reference_parser_result"]["status"] == "accepted"]
    for variant in recognizers:
        exact = sum(o[variant]["status"] == "accepted" and o[variant].get("command") == o["reference_parser_result"]["command"] for o in eligible)
        scored_accepted = [o for o in eligible if o[variant]["status"] == "accepted"]
        summary[variant] = {"statuses": dict(Counter(o[variant]["status"] for o in outputs)), "reasons": dict(Counter(o[variant].get("reason", "none") for o in outputs)), "supported_samples": sum(o["gold"] != "UNKNOWN" for o in outputs), "reference_parser_eligible": len(eligible), "supported_unscored": sum(o["gold"] != "UNKNOWN" for o in outputs) - len(eligible), "exact_frames": int(exact), "exact_frame_rate": exact / max(1, len(eligible)), "scored_accepted_frames": len(scored_accepted), "accepted_frame_precision": exact / max(1, len(scored_accepted)), "unknown_samples": sum(o["gold"] == "UNKNOWN" for o in outputs), "accepted_frames_on_unknown": sum(o["gold"] == "UNKNOWN" and o[variant]["status"] == "accepted" for o in outputs), "tail_asr_invocations": sum("parameter_transcript" in o[variant] for o in outputs)}
    write_json(ROOT / "reports/joint_metrics.json", summary)
    write_jsonl(ROOT / "reports/predictions_joint.jsonl", outputs)
    print("JOINT SUMMARY " + json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["vcm", "asr", "voice-suite"])
    parser.add_argument("--bundle", default="artifacts/tinyvcm")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    bundle = ROOT / args.bundle
    {"vcm": run_vcm, "asr": run_asr, "voice-suite": run_voice_suite}[args.task](bundle)


if __name__ == "__main__":
    main()
