"""Pure receipt, provenance and accounting checks; no model execution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from final import assemble_card as card
from scripts.test import vcm_mixed_benchmark as benchmark


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def reference(identifier, command, status="supported", kind="human", text="Play music"):
    return {"id": identifier, "kind": kind, "speaker_id": "speaker-" + identifier,
            "reference_status": status, "command": command, "transcript": text}


def prediction(identifier, command=None, *, fault=None):
    if fault:
        return {"id": identifier, "fault": fault}
    return {"id": identifier, "final_application": {
        "accepted": command is not None, "command": command,
        "reason": "accepted" if command is not None else "low_intent"}}


def observations():
    supported = {"intent": "PLAY_MUSIC", "slots": {}}
    slotted = {"intent": "VOLUME", "slots": {"operation": "set", "percent": 25}}
    trials = [reference("good", supported), reference("wrong", slotted, text="Set volume to 25"),
              reference("negative", None, "unsupported", text="Hello there"),
              reference("fault", supported), reference("missing", supported),
              reference("ambiguous", None, "ambiguous", text="Blue lights please")]
    predictions = [prediction("good", supported),
                   prediction("wrong", {"intent": "VOLUME", "slots": {"operation": "set", "percent": 26}}),
                   prediction("negative"), prediction("fault", fault="capture failure"),
                   prediction("ambiguous")]
    records = [benchmark.record_result(row, found)
               for row, found in benchmark._pairs(trials, predictions)]
    metrics = benchmark.summarize(trials, predictions)
    return records, metrics


def test_actual_accounting_keeps_faults_missing_and_ambiguous_denominators():
    records, metrics = observations()
    card.validate_metrics(metrics, records, expected_n=6)
    assert metrics["all"]["faults"] == 1
    assert metrics["all"]["missing"] == 1
    assert metrics["all"]["ambiguous_n"] == 1
    assert metrics["all"]["complete_command_accuracy"] == 0.25
    assert metrics["all"]["joint_supplied_slots"]["accuracy"] == 0.25
    assert metrics["all"]["every_slot"]["percent"]["correct"] == 0


@pytest.mark.parametrize("field", ["correct", "supported_n", "negative_n", "faults", "missing",
                                 "complete_command_accuracy", "strict_accuracy"])
def test_invented_or_wrong_denominator_metric_is_rejected(field):
    records, metrics = observations()
    metrics["all"][field] += 1
    with pytest.raises(ValueError, match="accounting"):
        card.validate_metrics(metrics, records, expected_n=6)


def test_incorrect_extra_slot_credit_is_rejected():
    records, metrics = observations()
    metrics["all"]["joint_supplied_slots"]["correct"] += 1
    with pytest.raises(ValueError, match="accounting"):
        card.validate_metrics(metrics, records, expected_n=6)


@pytest.mark.parametrize("group", ["intents", "tracks", "people", "macro_person_recall",
                                   "worst_person_recall", "people_cluster_recall_interval"])
def test_invented_subgroup_or_person_statistics_are_rejected(group):
    records, metrics = observations()
    if group in ("intents", "tracks", "people"):
        first = next(iter(metrics[group].values()))
        first["supported_n"] = 999
        first["complete_command_accuracy"] = 1.0
    else:
        metrics[group] = {"invented": 1.0}
    with pytest.raises(ValueError, match="accounting"):
        card.validate_metrics(metrics, records, expected_n=6)


@pytest.mark.parametrize("mutation", ["outcome", "verbatim", "duplicate", "source_id"])
def test_trial_provenance_cannot_be_rewritten(mutation):
    records, metrics = observations()
    if mutation == "duplicate":
        records[-1] = deepcopy(records[0])
    elif mutation == "source_id":
        records[0]["source_row"]["id"] = "changed"
    else:
        records[1][mutation] = "rewritten"
    with pytest.raises(ValueError, match="trial|verbatim|outcome"):
        card.validate_metrics(metrics, records, expected_n=6)


def test_grouped_misses_preserve_unicode_punctuation_and_verbatim_spaces():
    records, _ = observations()
    original = deepcopy(records)
    records[1]["verbatim"] = "  Set volume—25 | please\nnow  "
    groups = card.group_misses({"fresh_human106": records, "original_test4418": records})
    entry = next(row for row in groups if row["verbatim"] == records[1]["verbatim"])
    assert entry["count"] == 2
    assert entry["cohorts"] == {"fresh_human106": 1, "original_test4418": 1}
    assert sum(row["count"] for row in groups) == 6
    assert not any(row["outcome"] == "ambiguous_reference" for row in groups)
    assert original[1]["source_row"] == records[1]["source_row"]


def tts_receipts():
    voices = list(card.FINAL_TTS_VOICES)
    audit = {"plan_sha256": "a" * 64, "manifest_sha256": "b" * 64,
             "attempted": 18, "generated": 16, "generation_faults": 2,
             "per_voice": {voice: {"attempted": 3, "generated": 3, "generation_fault": 0}
                           for voice in voices},
             "independent_listening_performed": False, "VCM_results_accessed": False,
             "supplemental_qualified_lock": None, "TTS_accuracy": None,
             "every_slot_correctness": None, "fit_selection_calibration_enrollment_use": False,
             "takes_regenerated": False, "fault_types": {"technical failure": 2}}
    audit["per_voice"][voices[-1]].update(generated=1, generation_fault=2)
    accounting = {"plan_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                  "case_references": 24, "unique_audio": 18,
                  "per_voice": {voice: {"references": 4, "supported": 3, "negative": 1,
                                        "generation_fault_references": 0} for voice in voices},
                  "VCM_results_accessed": False, "inference_metrics": None,
                  "listening_review_completed": False,
                  "generation_fault_counts_are_not_recognition_errors": True}
    accounting["per_voice"][voices[-1]]["generation_fault_references"] = 3
    return audit, accounting


def test_tts_generation_faults_never_become_recognition_errors_or_accuracy():
    audit, accounting = tts_receipts()
    observed = card.tts_summary(audit, accounting)
    assert observed["attempted"] == 18 and observed["generated"] == 16
    assert observed["generation_faults"] == 2
    assert observed["generation_fault_references"] == 3
    assert observed["case_references"] == 24
    for row in observed["per_voice"].values():
        assert row["complete_command_accuracy"] is None
        assert row["every_slot_accuracy"] is None
        assert row["false_accept_rate"] is None
        assert row["faults"] is None and row["missing"] is None
    assert observed["final_tts_used_for_fit_selection_calibration_enrollment"] is False


@pytest.mark.parametrize("change", ["take_count", "reference_count", "voice", "manifest", "fit_use", "scores"])
def test_tts_accounting_or_unsealed_outcomes_fail(change):
    audit, accounting = tts_receipts()
    if change == "take_count":
        audit["generated"] += 1
    elif change == "reference_count":
        accounting["case_references"] += 1
    elif change == "voice":
        del audit["per_voice"]["af_heart"]
    elif change == "manifest":
        accounting["manifest_sha256"] = "c" * 64
    elif change == "fit_use":
        audit["fit_selection_calibration_enrollment_use"] = True
    else:
        audit["TTS_accuracy"] = 1.0
    with pytest.raises(ValueError, match="TTS"):
        card.tts_summary(audit, accounting)


def test_finite_unique_json_and_changed_hashes_are_enforced(tmp_path):
    file = tmp_path / "evidence.json"
    binding = write_json(file, {"actual": 2})
    store = card.Evidence()
    assert store.json(binding) == {"actual": 2}
    file.write_text('{"actual":3}')
    with pytest.raises(ValueError, match="changed"):
        store.verify_unchanged()
    for payload in ('{"actual":NaN}', '{"actual":1,"actual":2}'):
        file.write_text(payload)
        fresh = card.Evidence()
        with pytest.raises(ValueError, match="Nonfinite|Duplicate"):
            fresh.json({"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})


def test_evidence_archive_copies_original_bytes_without_serialization(tmp_path):
    file = tmp_path / "original.jsonl"
    payload = b'{ "verbatim" : "  original  " }\r\n'
    file.write_bytes(payload)
    store = card.Evidence()
    reference = store.capture(file)
    output = tmp_path / "derived"
    store.write(output)
    assert (output / reference["path"]).read_bytes() == payload
    assert file.read_bytes() == payload


def test_new_directory_requirement_preserves_existing_report(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "card.json").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        card.assemble("unused", "unused", "unused", "unused", "unused", "unused", output)
    assert (output / "card.json").read_bytes() == b"preserve"


def test_duration_receipt_is_bound_to_exact_views_and_preserves_kind_counts():
    rows = [reference("human", None, "unsupported"),
            reference("tts", None, "unsupported", kind="synthetic")]
    for i, row in enumerate(rows):
        row.update(samples=(i + 1) * 16000, sample_rate=16000)
    views = {"train": rows}
    inputs = {"views": {"train": {"sha256": "a" * 64}}}
    receipt = {"inputs_sha256": "b" * 64, "splits": {"train": {
        "view_sha256": "a" * 64, "utterances": 2,
        "by_kind": {kind: {"utterances": 1, "samples": row["samples"],
                           "hours": row["samples"] / 16000 / 3600,
                           "declared_speaker_ids": 1} for kind, row in zip(("human", "synthetic"), rows)}}}}
    card.verify_duration(receipt, inputs, views, inputs_sha256="b" * 64)
    receipt["splits"]["train"]["by_kind"]["human"]["utterances"] = 2
    with pytest.raises(ValueError, match="duration"):
        card.verify_duration(receipt, inputs, views, inputs_sha256="b" * 64)


@pytest.fixture
def normalization_fixture(tmp_path):
    training = tmp_path / "training"
    inputs_path = tmp_path / "inputs.json"
    write_json(inputs_path, {"frozen": "inputs"})
    normalization_path = training / "features/normalization.json"
    write_json(normalization_path, {"frames": 123, "fit_split": "train"})
    features = write_json(training / "features/features.json", {"frozen": "feature receipt"})
    context = {"training": training, "inputs_path": inputs_path,
               "inputs": {"views": {"train": {"sha256": "a" * 64}}},
               "normalization_path": normalization_path, "normalization": {"frames": 123}}
    receipt = {"status": "exact_train_only_normalization_verified", "fit_split": "train",
               "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
               "training_manifest_sha256": "a" * 64,
               "features_receipt_sha256": features["sha256"],
               "normalization_sha256": hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
               "verified_training_feature_arrays": 2, "training_frames": 123,
               "mean_max_error": 0.0, "std_max_error": 0.0,
               "development_calibration_enrollment_evaluation_final_TTS_frames_used": False,
               "actual_neural_inference_performed": False}
    return context, receipt, [{"id": "first"}, {"id": "second"}]


def test_normalization_recheck_binds_actual_training_features(normalization_fixture):
    context, receipt, rows = normalization_fixture
    card.verify_normalization_recheck(context, receipt, rows)


@pytest.mark.parametrize("field,value", [
    ("inputs_sha256", "b" * 64), ("training_manifest_sha256", "b" * 64),
    ("features_receipt_sha256", "b" * 64), ("normalization_sha256", "b" * 64),
    ("verified_training_feature_arrays", 1), ("training_frames", 124),
    ("mean_max_error", 1e-12), ("std_max_error", False), ("fit_split", "dev"),
    ("development_calibration_enrollment_evaluation_final_TTS_frames_used", True),
    ("actual_neural_inference_performed", True),
])
def test_unbound_or_contaminated_normalization_recheck_fails(normalization_fixture, field, value):
    context, receipt, rows = normalization_fixture
    receipt[field] = value
    with pytest.raises(ValueError, match="normalization recheck"):
        card.verify_normalization_recheck(context, receipt, rows)


def test_gates_cannot_infer_qualification_from_offline_gpu_accuracy():
    audit, accounting = tts_receipts()
    tts = card.tts_summary(audit, accounting)
    gates = card.qualification_gates(
        preparation={"reviewed_acoustic_spans": 0, "acoustic_listening_verified": 0,
                     "missing_intents": {"fresh_human": ["TIME", "WEATHER"]}},
        calibration={"missing_slot_calibration": ["percent/acoustic/1"]}, tts=tts,
        export={"cuda_parity_passed": True, "ncnn_status": "blocked_cpu_hold"})
    assert gates["accuracy_ready"] is False
    assert gates["cuda_pytorch_onnx_numeric_parity"]["status"] == "passed"
    assert gates["ncnn_parity"]["status"] == "blocked"
    assert gates["resident_service"]["status"] == "blocked"
    assert gates["reviewed_acoustic_supervision"]["status"] == "failed"
    assert gates["fresh_human_scope"]["status"] == "failed"
    assert gates["final_tts_generation_qa"]["status"] == "failed"
    assert gates["final_tts_listening_and_lock"]["status"] == "blocked"
    assert gates["final_tts_replay"]["status"] == "blocked"
    assert gates["owner_paired_recordings"]["status"] == "blocked"


def test_untrained_value_head_remains_a_failed_gate():
    audit, accounting = tts_receipts()
    gates = card.qualification_gates(preparation={}, calibration={},
        tts=card.tts_summary(audit, accounting), export={},
        training={"zero_supervision_value_heads": ["alarm_minute"]})
    assert gates["value_head_supervision"] == {
        "status": "failed", "reason": "Actual finite-value loss denominators must be nonzero",
        "zero_supervision_heads": ["alarm_minute"]}
    assert gates["physical_person_alias_verification"]["status"] == "blocked"
    assert gates["accuracy_ready"] is False


def test_waveform_gate_is_separate_from_ncnn_resident_and_label_accuracy():
    audit, accounting = tts_receipts()
    gates = card.qualification_gates(preparation={}, calibration={},
        tts=card.tts_summary(audit, accounting),
        export={"waveform_parity": {"parity_passed": True, "total": {"written_rows": 4922,
                 "matching_capture_faults": 2, "head_max_errors": {name: 1e-6 for name in card.HEADS}}}},
        training={"zero_supervision_value_heads": []})
    assert gates["cuda_pytorch_onnx_waveform_parity"]["status"] == "passed"
    assert gates["ncnn_parity"]["status"] == "blocked"
    assert gates["resident_service"]["status"] == "blocked"
    assert gates["accuracy_ready"] is False


def test_missing_or_failed_waveform_receipt_cannot_become_a_pass():
    audit, accounting = tts_receipts()
    for wave, expected in ((None, "blocked"), ({"parity_passed": False, "total": {}}, "failed")):
        gates = card.qualification_gates(preparation={}, calibration={},
            tts=card.tts_summary(audit, accounting), export={"waveform_parity": wave})
        assert gates["cuda_pytorch_onnx_waveform_parity"]["status"] == expected


def test_waveform_card_bridge_preserves_every_artifact_and_exact_first_pass(tmp_path, monkeypatch):
    from scripts.test import vcm_mixed_wave_parity_cuda as wave
    receipt = write_json(tmp_path / "wave.json", {"software_only": "receipt"})
    original = write_json(tmp_path / "benchmark-receipt.json", {"software_only": "original first pass"})
    artifact = write_json(tmp_path / "preserved.json", {"software_only": "raw evidence"})
    selected = {"checkpoint_sha256": "a" * 64}
    records = {"fresh_human106": [{"id": "original-id"}]}
    called = {}
    def verify(binding, **kwargs):
        called.update(kwargs)
        assert binding == receipt
        return {"files": {"raw": artifact}, "parity_passed": False}
    monkeypatch.setattr(wave, "verify_waveform_receipt", verify)
    monkeypatch.setattr(wave, "compact_summary", lambda _: {"parity_passed": False, "placement_evidence": {}})
    evidence = card.Evidence()
    result = card.waveform_section(receipt, {"selected": selected, "contract": {"contract_hash": "c" * 64}},
        {"recipe_sha256": "b" * 64}, records, original["path"], evidence)
    assert called == {"selected": selected, "recipe_sha256": "b" * 64, "contract_hash": "c" * 64,
                      "first_pass_receipt": original, "expected_ids": {"fresh_human106": ["original-id"]}}
    assert result["parity_passed"] is False and result["receipt"]["sha256"] == receipt["sha256"]
    assert str(Path(artifact["path"]).resolve()) in evidence.sources
    assert card.waveform_section(None, {}, {}, {}, "unused", evidence) is None


@pytest.fixture
def official_outputs_fixture(tmp_path):
    source = card.ROOT / ".cache/vcm-mixed-final-v2/official-benchmark" / benchmark.OFFICIAL_REVISION
    if not source.is_dir():
        pytest.skip("Pinned scorer must already be present; no test downloads")
    spec = {"root": str(source), "revision": benchmark.OFFICIAL_REVISION,
            "snapshot": benchmark.binding(source / "snapshot-receipt.json")}
    trials = [reference("timer", {"intent": "TIMER", "slots": {"duration_seconds": 60}}),
              reference("negative", None, "unsupported"),
              reference("fault", {"intent": "TIME", "slots": {}})]
    for row, intent, slot, variation in zip(trials, ("TIMER", "OUT_OF_SCOPE", "TIME"),
                                          ("1 minute", "", ""),
                                          ("Set a timer for 1 minute", "", "What time is it")):
        row.update(publisher_intent=intent, publisher_slot=slot, publisher_variation=variation,
                   publisher_transcript="Unchanged original " + row["id"])
    predictions = [prediction("timer", trials[0]["command"]), prediction("negative"),
                   prediction("fault", fault="software-only capture fault")]
    for pred in predictions[:2]:
        pred.update(timing={"total_ms": 1.0}, audio_seconds=1.0)  # Scorer fixture only.
    report, official_records = benchmark.official_scores(trials, predictions, benchmark.load_official(spec))
    logs = [row["log_record"] for row in official_records if row["log_record"] is not None]
    destination = tmp_path / "mandatory_defaults202"
    files = {"mandatory_defaults202/official-metrics.json": write_json(destination / "official-metrics.json", report)}
    for name, rows in (("official-records.jsonl", official_records), ("benchmark-log.jsonl", logs)):
        path = destination / name
        path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        files["mandatory_defaults202/" + name] = benchmark.binding(path)
    records = [benchmark.record_result(row, pred) for row, pred in zip(trials, predictions)]
    metrics = {"official_metrics": files["mandatory_defaults202/official-metrics.json"]}
    return metrics, records, files, spec, report, official_records, logs


def test_official_scores_are_recomputed_from_original_predictions(official_outputs_fixture):
    metrics, records, files, spec, report, _, _ = official_outputs_fixture
    actual = card.verify_official_outputs(metrics, records, files=files, official_spec=spec, evidence=card.Evidence())
    assert actual == report and actual["command"]["accuracy"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("change", ["accuracy", "parsed_reference", "adapter_log", "report_binding"])
def test_invented_official_accuracy_or_logs_fail(official_outputs_fixture, tmp_path, change):
    metrics, records, files, spec, report, official_records, logs = official_outputs_fixture
    if change == "accuracy":
        report["command"]["accuracy"] = 1.0
        key = "mandatory_defaults202/official-metrics.json"
        files[key] = write_json(Path(files[key]["path"]), report)
        metrics["official_metrics"] = files[key]
    elif change == "report_binding":
        metrics["official_metrics"] = write_json(tmp_path / "unbound-replacement.json", report)
    else:
        name, rows = (("official-records.jsonl", official_records) if change == "parsed_reference"
                      else ("benchmark-log.jsonl", logs))
        if change == "parsed_reference":
            rows[0]["publisher_slot"] = "changed reference"
        else:
            rows[0]["intent"] = "TIME"
        key = "mandatory_defaults202/" + name
        path = Path(files[key]["path"])
        path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        files[key] = benchmark.binding(path)
    with pytest.raises(ValueError, match="Official|official"):
        card.verify_official_outputs(metrics, records, files=files, official_spec=spec, evidence=card.Evidence())


def test_architecture_uses_actual_contract_field_names_and_counts():
    contract = {"roles": ["role" + str(i) for i in range(15)],
                "entity_roles": ["entity" + str(i) for i in range(5)],
                "intent_order": ["intent" + str(i) for i in range(19)],
                "role_states": ["absent", "present", "conflicting", "unsupported"],
                "request_validity_states": ["valid", "unsupported", "ambiguous", "incomplete"],
                "finite_values": {"color": ["red", "blue", "green"]}}
    result = card.architecture_summary(contract, {"total": 280818}, {"bins": 40})
    assert result["validity_states"] == contract["request_validity_states"]
    assert result["semantic_queries"] == 16 and result["intents"] == 19
    assert result["causal_model"] is False and result["ASR"] is False
    assert result["CTC"] is False and result["KV_cache"] is False
    assert result["finite_value_heads"] == {"color": 3}
    contract["roles"].pop()
    with pytest.raises(ValueError, match="architecture"):
        card.architecture_summary(contract, {"total": 280818}, {"bins": 40})


@pytest.fixture
def export_fixture(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    project, bundle = tmp_path / "project", tmp_path / "bundle"
    bundle.mkdir()
    contract = {"contract_hash": "a" * 64, "value_offsets": {"color": [0, 3]},
                "catalog_entries": [{"id": "light.kitchen"}], "custom": {}}
    source_hashes = {}
    for name in sorted(card.EXPORT_SOURCES):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"software test source; not an executable model\n")
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (bundle / "reference.pt").write_bytes(b"software-only checkpoint bytes")
    graph = helper.make_graph([helper.make_node("Identity", ["features"], ["output"], name="identity")],
                              "software-only-schema-fixture",
                              [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 40, "frames"])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 40, "frames"])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), bundle / "model.onnx")
    write_json(bundle / "contract.json", contract)
    write_json(bundle / "catalog.json", {"entries": contract["catalog_entries"], "custom": contract["custom"]})
    for name in ("normalization.json", "entities.json", "policy.json"):
        write_json(bundle / name, {"software_test_evidence": name})
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in bundle.iterdir()}
    selected = {"checkpoint_sha256": files["reference.pt"], "arm": "uniform", "seed": 53, "epoch": 1}
    recipe = {"format": "vcm-intent-slots-v1", "checkpoint_status": "mixed_silver_unqualified_candidate",
              "checkpoint_source_sha256": files["reference.pt"], "encoder_id": files["reference.pt"],
              "contract_hash": contract["contract_hash"], "head_order": list(card.HEADS),
              "value_offsets": contract["value_offsets"], "parameters": {"total": 280818},
              "acoustic_qualified": False, "files": files, "source_bindings": source_hashes,
              "graph_bytes": (bundle / "model.onnx").stat().st_size}
    write_json(bundle / "recipe.json", recipe)
    postfit_file = tmp_path / "postfit-receipt.json"
    postfit_binding = write_json(postfit_file, {"software_test_evidence": True})
    postfit = {"receipt_path": postfit_file, "policy": {"calibration_id": "c" * 64},
               "files": {key: {"sha256": files[name]} for key, name in (
                   ("normalization", "normalization.json"), ("entities", "entities.json"),
                   ("calibration", "policy.json"))}}
    probes = [{"frames": frames, "max_error": 1e-6,
               "head_errors": {name: 1e-6 for name in card.HEADS}, "decision_equal": True,
               "pytorch_decision": {"accepted": False, "command": None, "reason": "low_intent"},
               "onnx_decision": {"accepted": False, "command": None, "reason": "low_intent"}}
              for frames in card.PROBE_FRAMES]
    receipt = {"status": "unqualified_cuda_onnx_export", "qualification": False, "selected": selected,
               "calibration_id": "c" * 64, "postfit_receipt": postfit_binding, "parity_tolerance": 2e-5,
               "parity_inputs": "fixed_numeric_features_only", "dynamic_frames": True, "opset": 17,
               **{key: False for key in ("CPU_neural_inference_performed", "ncnn_invoked",
                                        "reference_labels_used", "test_accessed", "final_tts_accessed")},
               "files": {path.name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in bundle.iterdir()},
               "numerical": {"parameters": {"total": 280818},
                             "checkpoint_status": "mixed_silver_unqualified_candidate",
                             "gpu": {"device": "cuda", "visible_device_count": 1},
                             "ort": {"actual_providers": ["CUDAExecutionProvider"],
                                     "whole_session_fallback_disabled": True,
                                     "input_device": "cuda", "output_device": "cuda"},
                             "rewrite_parity": deepcopy(probes), "onnx_parity": deepcopy(probes)},
               "ncnn_status": "blocked_cpu_hold"}
    receipt["source_bindings"] = {name: {"path": str(project / name), "sha256": digest}
                                   for name, digest in source_hashes.items()}
    proof = bundle / "ort-proof"
    proof.mkdir()
    optimized = proof / "model.optimized.onnx"
    optimized.write_bytes((bundle / "model.onnx").read_bytes())
    events = []
    for index, _ in enumerate(card.PROBE_FRAMES):
        start = index * 100
        events += [{"cat": "Session", "name": "model_run", "ph": "X", "ts": start, "dur": 90},
                   {"cat": "Node", "name": "identity_kernel_time", "ph": "X", "ts": start + 1,
                    "dur": 5, "args": {"op_name": "Identity", "provider": "CUDAExecutionProvider"}}]
    raw_profile = proof / "raw-profile.json"
    write_json(raw_profile, events)
    options = {"CUDAExecutionProvider": {"device_id": "0", "use_tf32": "0"}}
    placement = card.export_cuda.validate_placement(optimized, raw_profile, frames=card.PROBE_FRAMES)
    placement.update(original_graph=benchmark.binding(bundle / "model.onnx"), provider_options=options)
    report = write_json(proof / "ort-node-placement.json", placement)
    placement["report"] = report
    receipt["numerical"]["placement"] = placement
    receipt["numerical"]["ort"]["provider_options"] = options
    receipt["placement_evidence"] = {key: placement[key] for key in (
        "original_graph", "optimized_graph", "profile", "report")}
    write_json(bundle / "cuda-export-receipt.json", receipt)
    return project, bundle, {"selected": selected, "contract": contract}, postfit, recipe, receipt


def test_exact_checkpoint_and_cuda_probe_receipts_bind_card_candidate(export_fixture):
    project, bundle, context, postfit, _, _ = export_fixture
    result = card.verify_bundle(bundle, context, postfit, card.Evidence(), root=project)
    assert result["cuda_parity_passed"] is True
    assert result["ncnn_status"] == "blocked_cpu_hold"
    assert result["recipe_sha256"] == hashlib.sha256((bundle / "recipe.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["missing_source", "rebound_source", "missing_profile", "cpu_node"])
def test_card_requires_complete_sources_and_actual_gpu_placement(export_fixture, change):
    project, bundle, context, postfit, _, receipt = export_fixture
    if change == "missing_source":
        del receipt["source_bindings"]["scripts/test/vcm_mixed_export_cuda.py"]
    elif change == "rebound_source":
        receipt["source_bindings"]["scripts/test/vcm_mixed_export_cuda.py"] = next(
            item for name, item in receipt["source_bindings"].items()
            if name != "scripts/test/vcm_mixed_export_cuda.py")
    elif change == "missing_profile":
        del receipt["numerical"]["placement"]["profile"]
    else:
        placement = receipt["numerical"]["placement"]
        raw = Path(placement["profile"]["path"])
        profile = json.loads(raw.read_bytes())
        profile[1]["args"]["provider"] = "CPUExecutionProvider"
        placement["profile"] = write_json(raw, profile)
    write_json(bundle / "cuda-export-receipt.json", receipt)
    with pytest.raises(ValueError):
        card.verify_bundle(bundle, context, postfit, card.Evidence(), root=project)


@pytest.mark.parametrize("change", ["fixture_status", "reference_sha", "source_bytes", "frame",
                                     "error", "decision", "cpu", "postfit_binding"])
def test_stale_fixture_or_unproved_gpu_export_is_rejected(export_fixture, change):
    project, bundle, context, postfit, recipe, receipt = export_fixture
    if change == "fixture_status":
        recipe["checkpoint_status"] = "software_fixture"
    elif change == "reference_sha":
        recipe["files"]["reference.pt"] = "e" * 64
    elif change == "source_bytes":
        (project / card.RUNTIME_SOURCES[0]).write_bytes(b"changed")
    elif change == "frame":
        receipt["numerical"]["onnx_parity"].pop()
    elif change == "error":
        receipt["numerical"]["onnx_parity"][0]["head_errors"]["intent"] = 1.0
    elif change == "decision":
        receipt["numerical"]["onnx_parity"][0]["onnx_decision"]["accepted"] = True
    elif change == "cpu":
        receipt["CPU_neural_inference_performed"] = True
    else:
        receipt["postfit_receipt"]["sha256"] = "e" * 64
    updated = write_json(bundle / "recipe.json", recipe)
    receipt["files"]["recipe.json"] = updated
    write_json(bundle / "cuda-export-receipt.json", receipt)
    with pytest.raises(ValueError):
        card.verify_bundle(bundle, context, postfit, card.Evidence(), root=project)


def test_rehashed_catalog_change_cannot_replace_the_settled_catalog(export_fixture):
    project, bundle, context, postfit, recipe, receipt = export_fixture
    changed = write_json(bundle / "catalog.json", {"entries": [], "custom": {}})
    recipe["files"]["catalog.json"] = changed["sha256"]
    receipt["files"]["catalog.json"] = changed
    receipt["files"]["recipe.json"] = write_json(bundle / "recipe.json", recipe)
    write_json(bundle / "cuda-export-receipt.json", receipt)
    with pytest.raises(ValueError, match="catalog"):
        card.verify_bundle(bundle, context, postfit, card.Evidence(), root=project)


def test_default_audit_dimensions_keep_counts_without_becoming_false_hash_fields():
    raw = {"current_default_rows": 202, "human_rows": 96, "synthetic_rows": 106,
           "fit_separation_ready": True, "reference_preservation_ready": True,
           "original_overlap_typed_reference_matches": 140,
           "default_vs_fit": {"train": {"sha256": {"rows": 0, "ids": []},
                                       "pcm_sha256": {"rows": 0, "ids": []}}},
           "fresh_vs_fit": {},
           "current_default_vs_original196_regression": {
               "pcm_sha256": {"rows": 140, "ids": ["held-140"]}},
           "qualification_limits": ["No acoustic alias review"]}
    original = deepcopy(raw)
    summary = card.default_audit_summary(raw, {"path": "audit.json", "sha256": "d" * 64})
    baseline = {"format": "vcm-model-card-v2", "bindings": {
        name: "e" * 64 for name in ("checkpoint_sha256", "recipe_sha256", "contract_hash")},
        "data": {"default_benchmark_audit": summary}, "pi": {"status": "deferred", "results": None}}
    card.render_card.validate_card(baseline)
    assert summary["comparisons"][0]["dimension"] == "sha256"
    assert summary["comparisons"][0]["rows"] == 0
    assert summary["original_overlap_typed_reference_matches"] == 140
    assert raw == original


def test_calibration_counts_and_source_rows_cannot_relabel_synthetic_as_human():
    human = reference("human-cal", None, "unsupported")
    synthetic = reference("tts-cal", None, "unsupported", kind="synthetic")
    synthetic.update(voice_id="am_echo", supervision_status="reference_derived_silver")
    views = {"train": [], "dev": [], "enrollment": [], "calibration": [human, synthetic]}
    policy = {"calibration_counts": {"human": 1, "synthetic": 1, "reference_derived_silver": 1},
              "calibration_manifest_rows": 2, "calibration_people": [human["speaker_id"]],
              "calibration_voices": ["am_echo"], "qualification": False,
              "thresholds": {"intent": .95, "slot": .90}}
    receipt = {"calibration_rows": 2, "reference_derived_silver_rows": 1}
    observations = [{"id": row["id"], "source_row": deepcopy(row)} for row in views["calibration"]]
    card.verify_calibration_provenance(views, policy, receipt, observations)
    observations[1]["source_row"]["kind"] = "human"
    with pytest.raises(ValueError, match="calibration"):
        card.verify_calibration_provenance(views, policy, receipt, observations)


def test_final_six_voices_cannot_enter_any_fitting_or_enrollment_view():
    views = {"train": [reference("final-take", None, "unsupported", kind="synthetic")],
             "dev": [], "enrollment": [], "calibration": []}
    views["train"][0]["voice_id"] = "af_heart"
    with pytest.raises(ValueError, match="final TTS"):
        card.verify_calibration_provenance(views, {}, {}, [])


def test_every_full_trial_artifact_needs_an_original_receipt_binding(tmp_path):
    files = {}
    for name in card.required_evaluation_files():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"software-only evidence bytes\n")
        files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {"files": files}
    card.verify_evaluation_files(receipt, tmp_path, card.Evidence())
    missing = "fresh_human106/trial-results.jsonl"
    del files[missing]
    with pytest.raises(ValueError, match="artifact"):
        card.verify_evaluation_files(receipt, tmp_path, card.Evidence())


def test_any_changed_preserved_trial_file_fails_its_original_receipt(tmp_path):
    files = {}
    for name in card.required_evaluation_files():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"software-only evidence bytes\n")
        files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "mandatory_defaults202/misses.jsonl").write_bytes(b"rewritten")
    with pytest.raises(ValueError, match="changed"):
        card.verify_evaluation_files({"files": files}, tmp_path, card.Evidence())
