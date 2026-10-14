"""Array, PCM-boundary and immutable accounting tests; no neural forward."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.test import vcm_mixed_export_cuda as gpu
from scripts.test import vcm_mixed_wave_parity_cuda as wave


@pytest.fixture
def context():
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases, extended_cases
    from scripts.runtime.vcm_intent_slots_contract import build_slot_contract
    cases = baseline_cases() + extended_cases()
    contract = build_slot_contract(cases, cases, DEFAULT_CUSTOM)
    return {"contract": contract, "memory": None,
            "catalog": {"entries": contract["catalog_entries"], "custom": contract["custom"]},
            "policy": {"require_calibration": False, "entities": {}}}


def heads(contract):
    sizes = [19, len(contract["roles"]) * 4, 4,
             max(value[1] for value in contract["value_offsets"].values()), 320]
    return [np.zeros((1, n), np.float32) for n in sizes]


def first_pass(reference, context, identifier="one"):
    result = gpu.compare_probe(reference, reference, context)
    return {"id": identifier, "final_application": result["pytorch_decision"]}


def source_row(identifier="one"):
    return {"id": identifier, "audio": "software-only.wav", "sha256": "a" * 64,
            "pcm_sha256": "b" * 64, "samples": 1600, "sample_rate": 16000,
            "kind": "human", "transcript": "Do not feed this text to inference",
            "command": {"intent": "PLAY_MUSIC", "slots": {}}, "reference_status": "supported"}


def test_all_named_head_errors_and_original_first_pass_decision_are_kept(context):
    reference = heads(context["contract"])
    actual = deepcopy(reference)
    actual[2][0, 0] = 1e-6
    result = wave.compare_wave_heads(reference, actual, context, first_pass(reference, context))
    assert result["status"] == "passed"
    assert set(result["head_errors"]) == set(gpu.HEAD_NAMES)
    assert result["head_errors"]["validity"] == pytest.approx(1e-6)
    assert result["numeric_pass"] is True
    assert result["decision_equal"] is True and result["first_pass_equal"] is True
    assert all(len(value) == 64 for value in result["pytorch_head_sha256"].values())


def test_numeric_mismatch_preserves_actual_errors_and_decisions(context):
    reference = heads(context["contract"])
    actual = deepcopy(reference)
    actual[0][0, 0] = 3e-5
    result = wave.compare_wave_heads(reference, actual, context, first_pass(reference, context))
    assert result["status"] == "numeric_mismatch"
    assert result["numeric_pass"] is False
    assert result["head_errors"]["intent"] > 2e-5
    assert "pytorch_decision" in result and "onnx_decision" in result


def test_tolerated_numeric_difference_cannot_hide_threshold_decision_flip(context):
    reference = heads(context["contract"])
    reference[0][0, context["contract"]["intent_order"].index("TIME")] = np.log(18 * .95 / .05) + 2e-6
    states = reference[1].reshape(-1, 4)
    states[:] = -20
    states[:, 0] = 20
    reference[2][0] = [20, -20, -20, -20]
    actual = deepcopy(reference)
    actual[0][0, context["contract"]["intent_order"].index("TIME")] -= 4e-6
    result = wave.compare_wave_heads(reference, actual, context, first_pass(reference, context))
    assert result["numeric_pass"] is True
    assert result["decision_equal"] is False
    assert result["status"] == "decision_mismatch"


@pytest.mark.parametrize("damage", ["source_decision", "source_fault", "source_missing"])
def test_replay_cannot_replace_a_failed_or_different_original_first_pass(context, damage):
    reference = heads(context["contract"])
    original = first_pass(reference, context)
    if damage == "source_decision":
        original["final_application"]["reason"] = "changed before parity"
    elif damage == "source_fault":
        original = {"id": "one", "fault": "original neural failure"}
    else:
        original = None
    result = wave.compare_wave_heads(reference, reference, context, original)
    assert result["status"] == {"source_decision": "first_pass_mismatch", "source_fault": "first_pass_fault",
                                "source_missing": "first_pass_missing"}[damage]
    assert result["first_pass_equal"] is False


@pytest.mark.parametrize("damage", ["nan", "head_missing", "wrong_dtype", "wrong_shape"])
def test_invalid_heads_are_faults_and_never_numeric_successes(context, damage):
    reference = heads(context["contract"])
    actual = deepcopy(reference)
    if damage == "nan": actual[1][0, 0] = np.nan
    elif damage == "head_missing": actual.pop()
    elif damage == "wrong_dtype": actual[0] = actual[0].astype(np.float64)
    else: actual[4] = actual[4].reshape(5, 64)
    result = wave.compare_wave_heads(reference, actual, context, first_pass(reference, context))
    assert result["status"] == "execution_fault"
    assert result["numeric_pass"] is False and result["fault"]


def test_pcm_and_frozen_norm_are_shared_and_labels_never_enter_inference(context):
    pcm = np.full(1600, .01, np.float32)
    norm = {"mean": [1.0] * 40, "std": [2.0] * 40}
    from vcm.frontend import features
    expected = features(pcm, norm["mean"], norm["std"])
    reference = heads(context["contract"])
    calls = []
    def infer(*args):
        assert len(args) == 1 and isinstance(args[0], np.ndarray)
        calls.append(args[0].copy())
        return reference, deepcopy(reference)
    result = wave.run_trial(source_row(), first_pass(reference, context), ".", norm, context,
                            infer, read_audio=lambda row, root: pcm)
    assert result["status"] == "passed"
    assert len(calls) == 1 and np.array_equal(calls[0], expected)
    assert result["source_row"]["transcript"] == source_row()["transcript"]


@pytest.mark.parametrize("samples", [399, 128001])
def test_invalid_runtime_capture_is_preserved_without_neural_calls(context, samples):
    pcm = np.zeros(samples, np.float32)
    from scripts.runtime.vcm_intent_slots_runtime import validate_pcm
    with pytest.raises(ValueError) as error:
        validate_pcm(pcm, 16000)
    fault = "ValueError: " + str(error.value)
    result = wave.run_trial(source_row(), {"id": "one", "fault": fault}, ".",
                            {"mean": [0.] * 40, "std": [1.] * 40}, context,
                            lambda x: pytest.fail("invalid PCM must not run a neural model"),
                            read_audio=lambda row, root: pcm)
    assert result["status"] == "matching_capture_fault"
    assert result["matching_capture_fault"] is True
    assert result["final_decision_agreement"] is False
    assert result["ort_called"] is False


def valid_record(context, identifier="valid"):
    reference = heads(context["contract"])
    return wave.run_trial(source_row(identifier), first_pass(reference, context, identifier), ".",
                          {"mean": [0.] * 40, "std": [1.] * 40}, context,
                          lambda x: (reference, deepcopy(reference)),
                          read_audio=lambda row, root: np.zeros(1600, np.float32))


def matching_fault_record(context, identifier="invalid"):
    from scripts.runtime.vcm_intent_slots_runtime import validate_pcm
    pcm = np.zeros(399, np.float32)
    with pytest.raises(ValueError) as error:
        validate_pcm(pcm, 16000)
    return wave.run_trial(source_row(identifier), {"id": identifier, "fault": "ValueError: " + str(error.value)}, ".",
                          {"mean": [0.] * 40, "std": [1.] * 40}, context,
                          lambda x: pytest.fail("invalid capture must not infer"), read_audio=lambda row, root: pcm)


def test_matching_capture_faults_are_separate_from_decision_agreements_and_accuracy(context):
    records = [valid_record(context), matching_fault_record(context)]
    summary = wave.summarize_records(records, ["valid", "invalid"])
    assert summary["trials"] == 2 and summary["passed_valid"] == 1
    assert summary["matching_capture_faults"] == 1 and summary["faults"] == 1
    assert summary["final_decision_agreements"] == 1
    assert summary["parity_failures"] == 0 and summary["missing"] == 0
    assert summary["parity_passed"] is True
    assert "accuracy" not in summary


def test_missing_rows_fail_full_accounting_and_keep_undefined_errors():
    report = wave.summarize_records([], ["missing"])
    assert report["missing"] == 1 and report["parity_passed"] is False
    assert report["head_max_errors"] == {name: None for name in gpu.HEAD_NAMES}
    assert report["final_decision_agreements"] == 0


def test_duplicate_or_foreign_ids_are_never_cherry_picked():
    for rows in ([{"id": "one"}, {"id": "one"}], [{"id": "foreign"}]):
        with pytest.raises(ValueError, match="trial"):
            wave.summarize_records(rows, ["one"])


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_parity_lock_is_exclusive_and_hash_binds_inputs_before_calls(tmp_path):
    source = tmp_path / "original-evaluation.json"
    binding = dump(source, {"software_test": "preserved first pass"})
    output = tmp_path / "parity-lock.json"
    lock = {"format": "vcm-mixed-cuda-wave-parity-lock-v1", "files": {"first_pass": binding},
            "audio": {}, "neural_calls": 0, "first_pass_accuracy_recomputed": False,
            "weights_fitted": False, "calibration_fitted": False, "enrollment_rebuilt": False,
            "final_tts_accessed": False, "thresholds": {"intent": .95, "slot": .90}}
    frozen = wave.freeze_parity_lock(lock, output)
    assert wave.verify_parity_lock(frozen) == lock
    with pytest.raises(FileExistsError): wave.freeze_parity_lock(lock, output)
    source.write_bytes(b"changed results")
    with pytest.raises(ValueError, match="binding|changed"):
        wave.verify_parity_lock(frozen)


def test_no_cpu_neural_claim_without_actual_operator_placement_proof():
    assert wave.cpu_neural_claim({"status": "blocked", "CPU_neural_inference_performed": None}) is None
    assert wave.cpu_neural_claim({"status": "blocked", "CPU_neural_inference_performed": True}) is True
    assert wave.cpu_neural_claim({"status": "verified_cuda_neural_placement", "CPU_neural_inference_performed": False}) is False
    with pytest.raises(ValueError, match="placement"):
        wave.cpu_neural_claim({"status": "blocked", "CPU_neural_inference_performed": False})


def test_new_directory_requirement_keeps_existing_parity_evidence(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "trial-results.jsonl").write_bytes(b"preserve first failed parity")
    with pytest.raises(FileExistsError):
        wave.run("unused", "unused", "unused", "unused", "unused", output)
    assert (output / "trial-results.jsonl").read_bytes() == b"preserve first failed parity"


@pytest.mark.parametrize("name", list(wave.benchmark.COHORTS))
def test_every_waveform_replay_is_regression_with_original_first_pass_provenance(name):
    rule = wave.benchmark.COHORTS[name]
    assert wave.replay_provenance(rule) == {
        "fresh": False, "regression": True,
        "original_evaluation_fresh": not rule["regression"],
        "original_evaluation_regression": rule["regression"],
    }


@pytest.mark.parametrize("mutation", [
    "numeric_flag", "numeric_error", "max_error", "decision_flag", "decision_body",
    "first_pass_flag", "first_pass_body", "capture", "ort_call", "frames", "feature_hash",
    "head_hash", "agreement_flag", "missing_fault", "unknown_status",
])
def test_passed_status_cannot_hide_contradictory_actual_evidence(context, mutation):
    row = valid_record(context)
    original_source, original_prediction = deepcopy(row["source_row"]), deepcopy(row["first_pass_prediction"])
    if mutation == "numeric_flag": row["numeric_pass"] = False
    elif mutation == "numeric_error": row["head_errors"]["intent"] = 1.0
    elif mutation == "max_error": row["max_error"] = 1.0
    elif mutation == "decision_flag": row["decision_equal"] = False
    elif mutation == "decision_body": row["onnx_decision"]["reason"] = "changed actual decision"
    elif mutation == "first_pass_flag": row["first_pass_equal"] = False
    elif mutation == "first_pass_body": row["first_pass_prediction"]["final_application"]["reason"] = "changed original decision"
    elif mutation == "capture": row["valid_capture"] = False
    elif mutation == "ort_call": row["ort_called"] = False
    elif mutation == "frames": row["frames"] = 0
    elif mutation == "feature_hash": row["features_sha256"] = "unknown"
    elif mutation == "head_hash": row["pytorch_head_sha256"]["intent"] = "unknown"
    elif mutation == "agreement_flag": row["final_decision_agreement"] = False
    elif mutation == "missing_fault": row.update(status="execution_fault")
    else: row["status"] = "claimed_passed"
    with pytest.raises(ValueError, match="[Ww]aveform|original|evidence"):
        wave.validate_trial(row, source_row=original_source, first_pass_prediction=original_prediction)


@pytest.mark.parametrize("mutation", ["transcript", "command", "publisher", "first_pass_fault", "source_id"])
def test_rebound_trial_cannot_rewrite_exact_original_reference_or_first_pass(context, mutation):
    row = valid_record(context)
    original_source, original_prediction = deepcopy(row["source_row"]), deepcopy(row["first_pass_prediction"])
    if mutation == "transcript": row["source_row"]["transcript"] = "rewritten reference"
    elif mutation == "command": row["source_row"]["command"] = {"intent": "TIME", "slots": {}}
    elif mutation == "publisher": row["source_row"]["publisher_transcript"] = "invented publisher form"
    elif mutation == "first_pass_fault": row["first_pass_prediction"] = {"id": row["id"], "fault": "replaced first pass"}
    else: row["source_row"]["id"] = "different source"
    with pytest.raises(ValueError, match="original"):
        wave.validate_trial(row, source_row=original_source, first_pass_prediction=original_prediction)


def test_matching_fault_requires_exact_original_shared_boundary_fault(context):
    row = matching_fault_record(context)
    wave.validate_trial(row, source_row=row["source_row"], first_pass_prediction=row["first_pass_prediction"])
    for name, value in (("fault", "new fault"), ("ort_called", True), ("valid_capture", True),
                        ("onnx_capture_fault", "new fault"), ("capture_stage", "file_or_pcm_integrity")):
        changed = {**row, name: value}
        with pytest.raises(ValueError, match="[Ww]aveform|evidence"):
            wave.validate_trial(changed)


def test_summary_checks_row_evidence_instead_of_trusting_passed_text(context):
    row = valid_record(context)
    row.update(numeric_pass=False, decision_equal=False, first_pass_equal=False, ort_called=False)
    row["head_errors"] = {name: 1.0 for name in gpu.HEAD_NAMES}
    with pytest.raises(ValueError, match="[Ww]aveform|evidence"):
        wave.summarize_records([row], [row["id"]])


def test_decision_agreement_is_label_independent_even_when_numeric_parity_fails(context):
    reference = heads(context["contract"])
    actual = deepcopy(reference)
    actual[0][0, 0] = .01
    result = wave.compare_wave_heads(reference, actual, context, first_pass(reference, context))
    assert result["status"] == "numeric_mismatch"
    assert result["final_decision_agreement"] is True
    assert "accuracy" not in result


def dump_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    return wave.benchmark.binding(path)


@pytest.fixture
def receipt_fixture(tmp_path, monkeypatch):
    """Full denominators with software-only capture faults and mocked preparation."""
    original, replay = tmp_path / "original", tmp_path / "replay"
    original.mkdir()
    replay.mkdir()
    selected = {"checkpoint_sha256": "a" * 64, "checkpoint": str(tmp_path / "selected.pt")}
    bindings = {"checkpoint_sha256": selected["checkpoint_sha256"], "recipe_sha256": "b" * 64,
                "contract_hash": "c" * 64, "normalization_sha256": "d" * 64}
    original_files, output_files, cohorts, data, predictions, rows, summaries, expected_ids = {}, {}, {}, {}, {}, {}, {}, {}
    selection = dump(original / "training" / "selection.json", {"software_only": True})
    inputs = dump(original / "inputs.json", {"software_only": True})
    audio = dump(original / "software-audio.json", {"not_audio": "no inference or PCM decoding in this fixture"})
    for name, rule in wave.benchmark.COHORTS.items():
        expected_ids[name] = [f"software-{index:05}" for index in range(rule["n"])]
        data[name] = [source_row(identifier) for identifier in expected_ids[name]]
        predictions[name] = {identifier: {"id": identifier, "fault": "ValueError: immutable software capture fault"}
                             for identifier in expected_ids[name]}
        view = dump_lines(original / name / "view.jsonl", data[name])
        trial = dump_lines(original / name / "trial-results.jsonl", [
            {"id": row["id"], "source_row": row, "prediction": predictions[name][row["id"]]} for row in data[name]])
        original_files["view/" + name] = view
        original_files["first_pass/" + name + "/trial-results.jsonl"] = trial
        cohorts[name] = {"view": view, "audio_root": str(original), "rows": rule["n"],
                         "regression": rule["regression"], "source_receipts": [audio]}
        rows[name] = [{"id": row["id"], "source_row": row, "first_pass_prediction": predictions[name][row["id"]],
                       "cohort": name, **wave.replay_provenance(rule), "status": "matching_capture_fault",
                       "head_errors": None, "max_error": None, "frames": None, "valid_capture": False,
                       "ort_called": False, "numeric_pass": False, "decision_equal": False, "first_pass_equal": False,
                       "final_decision_agreement": False, "matching_capture_fault": True,
                       "fault": predictions[name][row["id"]]["fault"],
                       "pytorch_capture_fault": predictions[name][row["id"]]["fault"],
                       "onnx_capture_fault": predictions[name][row["id"]]["fault"],
                       "capture_stage": "shared_unchanged_runtime_pcm_boundary"} for row in data[name]]
        summaries[name] = {**wave.summarize_records(rows[name], expected_ids[name]), **wave.replay_provenance(rule)}
        output_files["parity_output/" + name + "/trial-results.jsonl"] = dump_lines(replay / name / "trial-results.jsonl", rows[name])
        output_files["parity_output/" + name + "/summary.json"] = dump(replay / name / "summary.json", summaries[name])
    selected_lock = dump(original / "selection-before-test.json", {"files": {"selection": selection, "inputs": inputs}})
    first = dump(original / "benchmark-receipt.json", {"selected": selected, "selection_lock": selected_lock,
                 "postfit_receipt": dump(original / "postfit.json", {"software_only": True})})
    original_files.update(first_pass_receipt=first, first_pass_selection_lock=selected_lock)
    lock = {"format": "vcm-mixed-cuda-wave-parity-lock-v1", "files": original_files,
            "audio": {"software": audio}, "cohorts": cohorts, "bindings": bindings, "selected": selected,
            "thresholds": wave.THRESHOLDS, "neural_calls": 0, "first_pass_accuracy_recomputed": False,
            "weights_fitted": False, "calibration_fitted": False, "enrollment_rebuilt": False,
            "final_tts_accessed": False, "first_pass_receipt": first, "first_pass_selection_lock": selected_lock,
            "bundle": str(tmp_path / "bundle"), "mode": "Complete PCM-to-head/final-decision CUDA parity only"}
    lock_binding = dump(replay / "parity-lock.json", lock)
    output_files["parity_lock"] = lock_binding
    output_files["parity_output/parity-lock.json"] = lock_binding
    all_rows = [{**row, "trial_identity": name + "/" + row["id"]} for name in wave.benchmark.COHORTS for row in rows[name]]
    total = wave.summarize_records(all_rows, [name + "/" + identifier for name in wave.benchmark.COHORTS
                                            for identifier in expected_ids[name]], identity_field="trial_identity")
    receipt = {"format": "vcm-mixed-cuda-waveform-parity-v1", "status": "failed_cuda_waveform_parity",
               "parity_passed": False, "qualification": False, "selected": selected, "bindings": bindings,
               "thresholds": wave.THRESHOLDS, "tolerance": wave.TOLERANCE, "first_pass_receipt": first,
               "first_pass_selection_lock": selected_lock, "parity_lock": lock_binding, "cohorts": summaries,
               "total": total, "neural_frame_sequence": [], "gpu": None, "ort": None,
               "placement": {"status": "blocked_node_placement", "CPU_neural_inference_performed": None,
                             "reason": "software-only fixture has no neural placement"},
               "CPU_neural_inference_performed": None, "files": {**original_files, **output_files},
               "first_pass_accuracy_recomputed": False, "first_pass_results_preserved": True,
               "weights_fitted": False, "calibration_fitted": False, "enrollment_rebuilt": False,
               "reference_labels_used_for_inference": False, "final_tts_accessed": False,
               "ncnn_evidence": False, "resident_service_evidence": False, "physical_pi_evidence": False}
    expected_lock = deepcopy(lock)
    expected_data, expected_predictions = deepcopy(data), deepcopy(predictions)
    def prepare(training_arg, inputs_arg, postfit_arg, evaluation_arg, bundle_arg):
        assert Path(training_arg) == Path(selection["path"]).parent
        assert Path(inputs_arg) == Path(inputs["path"])
        assert Path(postfit_arg) == Path(wave.read_json(first["path"])["postfit_receipt"]["path"])
        assert Path(evaluation_arg) == Path(first["path"])
        if Path(bundle_arg) != Path(expected_lock["bundle"]):
            raise ValueError("Changed selected bundle source")
        return None, None, None, None, deepcopy(expected_data), deepcopy(expected_predictions), deepcopy(expected_lock)
    monkeypatch.setattr(wave, "_prepare", prepare)
    return {"path": replay / "wave-parity-receipt.json", "receipt": receipt, "lock": lock,
            "rows": rows, "selected": selected, "first": first, "expected_ids": expected_ids,
            "verify_args": {"selected": selected, "recipe_sha256": bindings["recipe_sha256"],
                "contract_hash": bindings["contract_hash"], "first_pass_receipt": first, "expected_ids": expected_ids}}


def test_complete_bound_fault_receipt_preserves_all_denominators_and_blocks_pass(receipt_fixture):
    fixture = receipt_fixture
    result = wave.verify_waveform_receipt(dump(fixture["path"], fixture["receipt"]), **fixture["verify_args"])
    assert result["total"]["trials"] == result["total"]["written_rows"] == 4922
    assert result["total"]["matching_capture_faults"] == 4922
    assert result["total"]["compared_valid"] == result["total"]["final_decision_agreements"] == 0
    assert result["parity_passed"] is False
    assert result["CPU_neural_inference_performed"] is None


@pytest.mark.parametrize("mutation", ["transcript", "command", "first_pass_prediction", "passed_status", "fresh_accounting"])
def test_consumer_rejects_rehashed_rewritten_trial_evidence(receipt_fixture, mutation):
    fixture = receipt_fixture
    name = "fresh_human106"
    row = fixture["rows"][name][0]
    if mutation == "transcript": row["source_row"]["transcript"] = "changed source verbatim"
    elif mutation == "command": row["source_row"]["command"] = {"intent": "TIME", "slots": {}}
    elif mutation == "first_pass_prediction": row["first_pass_prediction"] = {"id": row["id"], "fault": "changed first-pass fault"}
    elif mutation == "fresh_accounting": row.update(fresh=True, regression=False)
    else: row.update(status="passed", numeric_pass=False, head_errors={name: 1.0 for name in gpu.HEAD_NAMES},
                     max_error=1.0, valid_capture=True, ort_called=False)
    item = fixture["receipt"]["files"]["parity_output/" + name + "/trial-results.jsonl"]
    fixture["receipt"]["files"]["parity_output/" + name + "/trial-results.jsonl"] = dump_lines(Path(item["path"]), fixture["rows"][name])
    with pytest.raises(ValueError, match="[Ww]aveform|original|accounting|provenance"):
        wave.verify_waveform_receipt(dump(fixture["path"], fixture["receipt"]), **fixture["verify_args"])


@pytest.mark.parametrize("mutation", ["files", "audio", "cohorts", "selected", "bundle", "first_selection"])
def test_consumer_rejects_rehashed_changed_pre_call_source_maps(receipt_fixture, mutation):
    fixture = receipt_fixture
    lock = fixture["lock"]
    if mutation in ("files", "audio", "cohorts"): lock[mutation] = {}
    elif mutation == "selected": lock["selected"] = {"checkpoint_sha256": "e" * 64}
    elif mutation == "bundle": lock["bundle"] += "-changed"
    else: lock["first_pass_selection_lock"] = fixture["first"]
    item = dump(Path(fixture["receipt"]["parity_lock"]["path"]), lock)
    fixture["receipt"].update(parity_lock=item)
    fixture["receipt"]["files"].update(parity_lock=item)
    fixture["receipt"]["files"]["parity_output/parity-lock.json"] = item
    with pytest.raises(ValueError, match="[Ww]aveform|original|source|selected|lock"):
        wave.verify_waveform_receipt(dump(fixture["path"], fixture["receipt"]), **fixture["verify_args"])


def test_consumer_requires_portable_binding_for_every_immutable_input(receipt_fixture):
    fixture = receipt_fixture
    del fixture["receipt"]["files"]["view/fresh_human106"]
    with pytest.raises(ValueError, match="artifact|binding|source"):
        wave.verify_waveform_receipt(dump(fixture["path"], fixture["receipt"]), **fixture["verify_args"])


def test_compact_summary_keeps_counts_and_provider_bindings_without_raw_node_maps():
    receipt = {"format": "vcm-mixed-cuda-waveform-parity-v1", "status": "passed_cuda_waveform_parity",
               "parity_passed": True, "qualification": False, "selected": {"checkpoint_sha256": "a" * 64},
               "bindings": {"recipe_sha256": "b" * 64}, "cohorts": {"software": {"written_rows": 4922}},
               "total": {"written_rows": 4922, "head_max_errors": {name: 0.0 for name in gpu.HEAD_NAMES}},
               "thresholds": wave.THRESHOLDS, "tolerance": wave.TOLERANCE, "CPU_neural_inference_performed": False,
               "first_pass_receipt": {"path": "first.json", "sha256": "c" * 64}, "parity_lock": {"path": "lock.json", "sha256": "d" * 64},
               "gpu": {"device": "cuda"}, "ort": {"actual_providers": ["CUDAExecutionProvider"], "provider_options": {}},
               "placement": {"status": "verified_cuda_neural_placement", "CPU_neural_inference_performed": False,
                   "original_graph": {"path": "graph.onnx", "sha256": "e" * 64},
                   "optimized_graph": {"path": "optimized.onnx", "sha256": "f" * 64},
                   "profile": {"path": "profile.json", "sha256": "1" * 64}, "report": {"path": "placement.json", "sha256": "2" * 64},
                   "probes": [{"frames": 10, "node_providers": {"a": "CUDAExecutionProvider"}}] * 4922,
                   "graph_description": {"large": "raw graph proof"}, "cuda_nodes": ["a"], "cpu_metadata_nodes": []},
               "neural_frame_sequence": [10] * 4922, "files": {"all": "raw artifact map"}}
    summary = wave.compact_summary(receipt)
    assert summary["total"] == receipt["total"] and summary["cohorts"] == receipt["cohorts"]
    assert summary["placement_evidence"] == {key: receipt["placement"][key] for key in ("original_graph", "optimized_graph", "profile", "report")}
    assert summary["placement"]["profiled_calls"] == 4922
    assert "probes" not in summary["placement"] and "graph_description" not in summary["placement"]
    assert "files" not in summary and "neural_frame_sequence" not in summary
    assert "accuracy" not in summary and "raw graph proof" not in json.dumps(summary)


@pytest.mark.parametrize("which_decode", [1, 2])
def test_legitimate_decoder_fault_preserves_actual_arrays_and_partial_decisions(context, monkeypatch, which_decode):
    reference = heads(context["contract"])
    original = first_pass(reference, context)
    actual_decision = deepcopy(original["final_application"])
    calls = []
    def decode(arrays, decoder):
        calls.append(arrays)
        if len(calls) == which_decode:
            raise ValueError("software decoder fault")
        return deepcopy(actual_decision)
    monkeypatch.setattr(wave, "decision_from_heads", decode)
    result = wave.run_trial(source_row(), original, ".", {"mean": [0.] * 40, "std": [1.] * 40}, context,
                            lambda x: (reference, deepcopy(reference)), read_audio=lambda row, root: np.zeros(1600, np.float32))
    assert result["status"] == "execution_fault"
    assert result["numeric_pass"] is True and result["max_error"] == 0.0
    assert set(result["head_errors"]) == set(gpu.HEAD_NAMES)
    assert result["final_decision_agreement"] is False
    if which_decode == 2:
        assert result["pytorch_decision"] == actual_decision
        assert result["first_pass_equal"] is True
    summary = wave.summarize_records([result], ["one"])
    assert summary["execution_faults"] == summary["parity_failures"] == 1
    assert summary["parity_passed"] is False


def test_failed_cuda_setup_writes_every_trial_without_neural_calls(context, tmp_path, monkeypatch):
    reference = heads(context["contract"])
    normalization = {"mean": [0.] * 40, "std": [1.] * 40}
    from scripts.runtime.vcm_intent_slots_runtime import validate_pcm
    with pytest.raises(ValueError) as error:
        validate_pcm(np.zeros(399, np.float32), 16000)
    fault = "ValueError: " + str(error.value)
    selected = {"checkpoint_sha256": "a" * 64}
    data = {name: [source_row("valid"), source_row("invalid")] for name in wave.benchmark.COHORTS}
    predictions = {name: {"valid": first_pass(reference, context, "valid"), "invalid": {"id": "invalid", "fault": fault}}
                   for name in wave.benchmark.COHORTS}
    source = dump(tmp_path / "bound-source.json", {"software_only": True})
    lock = {"format": "vcm-mixed-cuda-wave-parity-lock-v1", "files": {"software": source}, "audio": {},
            "cohorts": {name: {"audio_root": "."} for name in wave.benchmark.COHORTS},
            "bindings": {"checkpoint_sha256": "a" * 64, "recipe_sha256": "b" * 64, "contract_hash": "c" * 64},
            "selected": selected, "thresholds": wave.THRESHOLDS, "neural_calls": 0,
            "first_pass_accuracy_recomputed": False, "weights_fitted": False, "calibration_fitted": False,
            "enrollment_rebuilt": False, "final_tts_accessed": False,
            "first_pass_receipt": source, "first_pass_selection_lock": source}
    prepared = {"contract": context["contract"], "selected": selected, "normalization": normalization}
    fitted = {"memory": context["memory"], "policy": context["policy"]}
    monkeypatch.setattr(wave, "_prepare", lambda *args: (prepared, fitted, None, lock, data, predictions, lock))
    def setup(*args):
        raise RuntimeError("software CUDA setup failure, no session/model created")
    monkeypatch.setattr(wave, "CUDAReplay", setup)
    original_run_trial = wave.run_trial
    def trial(row, *args):
        return original_run_trial(row, *args, read_audio=lambda row, root: np.zeros(399 if row["id"] == "invalid" else 1600, np.float32))
    monkeypatch.setattr(wave, "run_trial", trial)
    receipt = wave.run("unused", "unused", "unused", "unused", "unused", tmp_path / "new-output")
    assert receipt["status"] == "failed_cuda_waveform_parity"
    assert receipt["total"]["written_rows"] == receipt["total"]["trials"] == 8
    assert receipt["total"]["execution_faults"] == receipt["total"]["matching_capture_faults"] == 4
    assert receipt["total"]["missing"] == receipt["total"]["ort_calls"] == 0
    assert receipt["CPU_neural_inference_performed"] is None
    for name in wave.benchmark.COHORTS:
        rows = wave.read_jsonl(tmp_path / "new-output" / name / "trial-results.jsonl")
        assert all(row["fresh"] is False and row["regression"] is True for row in rows)
        assert [row["source_row"] for row in rows] == data[name]
