"""Numerical/accounting tests; no synthesized or human speech claims."""
import copy
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import final.tts_sanity as tts
from final.tts_sanity import (
    asset_paths,
    merge_cases,
    review_complete,
    score_cases,
    to_pcm16,
    trial_id,
)


def contract():
    command = {"intent": "BRIGHTNESS", "slots": {"percent": 20}}
    return {
        "cases": [{"id": "B1-001", "text": "Brightness 20 percent",
                   "spoken_text": "Brightness twenty percent",
                   "command": command, "tier": "baseline"}],
        "catalog_entries": [
            {"id": "B1-001", "text": "brightness twenty percent",
             "spoken_text": "Brightness twenty percent",
             "command": command, "tier": "baseline"},
            {"id": "N1", "text": "turn on the mystery light",
             "command": None, "reason": "invalid_argument"},
        ],
    }


def test_merge_preserves_exact_text_and_every_source_case():
    cases = merge_cases(contract())
    assert len(cases) == 3
    assert cases[0]["spoken_text"] == "Brightness twenty percent"
    assert len({row["case_id"] for row in cases}) == 3
    assert len({row["spoken_text"] for row in cases}) == 2


def test_conflicting_labels_for_same_audio_fail():
    data = contract()
    data["catalog_entries"][0]["command"] = {"intent": "TIME", "slots": {}}
    with pytest.raises(ValueError, match="Conflicting"):
        merge_cases(data)


def test_conflicting_rejection_references_fail_before_synthesis():
    data = contract()
    data["cases"].append({**data["catalog_entries"][-1], "id": "N2", "reason": "OOV"})
    with pytest.raises(ValueError, match="Conflicting"):
        merge_cases(data)


def test_duplicate_source_case_id_fails():
    data = contract()
    data["cases"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        merge_cases(data)


def test_missing_catalog_is_not_full_coverage():
    data = contract()
    data["catalog_entries"] = []
    with pytest.raises(ValueError):
        merge_cases(data)


def trials():
    return [{**case, "voice": "af_heart",
             "audio_id": trial_id("af_heart", case["spoken_text"])}
            for case in merge_cases(contract())]


def positive_result(percent=20):
    return {"audio_id": trials()[0]["audio_id"], "final_application": {
        "accepted": True, "command": {
            "intent": "BRIGHTNESS", "slots": {"percent": percent}},
        "reason": None}}


def negative_result(reason="invalid_argument"):
    return {"audio_id": trials()[-1]["audio_id"], "final_application": {
        "accepted": False, "command": None, "reason": reason}}


def test_exact_all_slots_and_duplicate_audio_denominators():
    result = score_cases(trials(), [positive_result(), negative_result()])
    assert result["unique_audio"]["trials"] == 2
    assert result["case_references"] == 3
    assert result["pass"] is True
    bad = score_cases(trials(), [positive_result(60), negative_result()])
    assert bad["unique_audio"]["recall"] == 0
    assert bad["pass"] is False


@pytest.mark.parametrize("negative", [None, {"fault": "read_error"}])
def test_fault_or_missing_is_not_a_correct_negative(negative):
    results = [positive_result()]
    if negative is not None:
        results.append({"audio_id": trials()[-1]["audio_id"], **negative})
    metrics = score_cases(trials(), results)
    assert metrics["unique_audio"]["correct_rejections"] == 0
    assert metrics["pass"] is False


def test_wrong_rejection_reason_fails_strict_sanity():
    outputs = [positive_result(), negative_result("OOV")]
    metrics = score_cases(trials(), outputs)
    assert metrics["unique_audio"]["false_accepts"] == 0
    assert metrics["pass"] is False


def test_rejected_candidate_cannot_be_promoted():
    value = positive_result()
    value["diagnostic_candidate"] = copy.deepcopy(value["final_application"])
    value["final_application"] = {"accepted": False, "command": None,
                                 "reason": "OOV"}
    result = score_cases(trials(), [value, negative_result()])
    assert result["unique_audio"]["recall"] == 0
    assert result["unique_audio"]["precision"] is None


def test_duplicate_or_unexpected_outputs_fail():
    with pytest.raises(ValueError):
        score_cases(trials(), [positive_result(), positive_result()])
    with pytest.raises(ValueError):
        score_cases(trials(), [{"audio_id": "unexpected", "fault": "x"}])


def test_cross_voice_scores_are_not_pooled_human_evidence():
    ts = trials()
    extra = [{**r, "voice": "am_fenrir", "audio_id": trial_id(
        "am_fenrir", r["spoken_text"])} for r in ts]
    result = score_cases(ts + extra, [positive_result(), negative_result()])
    assert result["per_voice"]["af_heart"]["recall"] == 1
    assert result["per_voice"]["am_fenrir"]["recall"] == 0
    assert result["human_generalization_established"] is False
    assert result["pass"] is False


def reviewed():
    generated = [{"audio_id": "a", "sha256": "abc", "status": "generated"}]
    review = {"blind_to_vcm": True, "items": [{
        "audio_id": "a", "sha256": "abc", "heard_exact": True,
        "quality": 4, "reviewer": "reviewer-01", "reviewed_at": "2026-10-15"}]}
    return generated, review


def test_quality_review_is_required_and_hash_bound():
    generated, review = reviewed()
    review_complete(generated, review)
    review["items"][0]["sha256"] = "changed"
    with pytest.raises(ValueError):
        review_complete(generated, review)


@pytest.mark.parametrize("field,value", [
    ("heard_exact", False), ("quality", 3), ("reviewer", ""),
    ("reviewed_at", ""), ("quality", True),
])
def test_inadequate_qa_blocks_qualification(field, value):
    generated, review = reviewed()
    review["items"][0][field] = value
    with pytest.raises(ValueError):
        review_complete(generated, review)


def test_missing_duplicate_or_unblinded_reviews_fail():
    generated, review = reviewed()
    for changed in [{"blind_to_vcm": False, "items": review["items"]},
                    {"blind_to_vcm": True, "items": []},
                    {"blind_to_vcm": True, "items": review["items"] * 2}]:
        with pytest.raises(ValueError):
            review_complete(generated, changed)


def test_resampling_is_real_not_a_rate_header_change():
    wave = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 0.2
    result = to_pcm16(wave, 24000)
    assert result.shape == (16000,)
    assert np.isfinite(result).all()
    peak = np.argmax(abs(np.fft.rfft(result)))
    assert peak == 440


@pytest.mark.parametrize("wave", [np.zeros(24000), np.array([np.nan]),
                                  np.ones((1000, 2)), np.ones(24000) * 1.1])
def test_invalid_audio_is_not_silently_repaired(wave):
    with pytest.raises(ValueError):
        to_pcm16(wave, 24000)


def test_wrong_native_rate_is_not_silently_assumed():
    with pytest.raises(ValueError):
        to_pcm16(np.ones(1000) * 0.1, 16000)


def test_asset_hashes_are_checked(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"wrong")
    spec = {"files": {"model.onnx": "0" * 64}}
    with pytest.raises(ValueError, match="hash"):
        asset_paths(tmp_path, spec)


def test_per_intent_and_per_case_denominators_are_reported():
    result = score_cases(trials(), [positive_result(), negative_result()])
    assert result["per_intent"]["BRIGHTNESS"]["supported"] == 1
    assert result["per_case"]["cases/B1-001"]["supported"] == 1


def test_malformed_accepted_slots_count_as_fault():
    result = positive_result()
    result["final_application"]["command"]["slots"] = []
    metrics = score_cases(trials(), [result, negative_result()])
    assert metrics["unique_audio"]["faults"] == 1
    assert metrics["pass"] is False


def grid_spec():
    return {
        "experiment_id": "grid-fixture-v2", "speeds": [0.9, 1.0, 1.1],
        "voices": [{"id": "af_heart", "lang": "en-us", "preset_sex": "female"},
                   {"id": "am_puck", "lang": "en-us", "preset_sex": "male"}],
    }


def test_waveform_identity_includes_speed_and_keeps_legacy_ids():
    legacy = hashlib.sha256(b'["af_heart","Time"]').hexdigest()
    assert trial_id("af_heart", "Time") == legacy
    identifiers = {trial_id("af_heart", "Time", speed=value)
                   for value in (0.9, 1.0, 1.1)}
    assert len(identifiers) == 3
    assert legacy not in identifiers
    assert trial_id("af_heart", "Time", speed=1) == trial_id(
        "af_heart", "Time", speed=1.0)


def full_contract():
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases
    from scripts.runtime.vcm_intent_slots_contract import build_slot_contract

    cases = baseline_cases() + [{
        "id": "negative-fixture", "text": "turn on the mystery light",
        "command": None, "reason": "invalid_argument", "tier": "extension"}]
    catalog = copy.deepcopy(baseline_cases()) + [{
        "id": "catalog-extra", "text": "What is the current time please?",
        "command": {"intent": "TIME", "slots": {}}, "reason": None}]
    return build_slot_contract(cases, catalog, DEFAULT_CUSTOM)


def planned_fixture(tmp_path, monkeypatch, protocol=None):
    monkeypatch.setattr(tts, "ROOT", tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(full_contract()), encoding="utf-8")
    output = tmp_path / ".cache" / "plan.json"
    kwargs = {} if protocol is None else {"protocol_path": protocol}
    tts.make_plan(contract_path, output, **kwargs)
    return tts.read_json(output), full_contract()


def test_default_plan_preserves_every_reference_at_all_six_voices_three_speeds(
        tmp_path, monkeypatch):
    plan, original = planned_fixture(tmp_path, monkeypatch)
    expected_grid = {(voice, speed) for voice in (
        "af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck")
        for speed in (0.9, 1.0, 1.1)}
    assert plan["optionb_trials"] == 1674
    assert len(plan["trials"]) == 188 * 18
    for case in merge_cases(original):
        actual = [row for row in plan["trials"] if row["case_id"] == case["case_id"]]
        assert {(row["voice"], row["speed"]) for row in actual} == expected_grid
        assert all({key: row[key] for key in case} == case for row in actual)
    assert len({row["audio_id"] for row in plan["audio"]}) == 95 * 18
    assert all(row["speed"] in (0.9, 1.0, 1.1) for row in plan["audio"])
    for field in ("fit_allowed", "selection_allowed", "calibration_allowed",
                  "enrollment_allowed"):
        assert plan[field] is False


def test_explicit_original_protocol_retains_legacy_voice_grid(tmp_path, monkeypatch):
    original = Path(tts.__file__).with_name("tts-protocol.json")
    plan, _ = planned_fixture(tmp_path, monkeypatch, original)
    assert plan["optionb_trials"] == 558
    assert {row["voice"] for row in plan["trials"]} == {
        "af_heart", "af_bella", "bf_emma", "am_fenrir", "am_michael", "bm_george"}
    assert all(row["audio_id"] == trial_id(row["voice"], row["spoken_text"])
               for row in plan["trials"])


@pytest.mark.parametrize("damage", ["drop_trial", "duplicate_trial", "drop_audio",
                                     "changed_reference", "changed_speed"])
def test_incomplete_or_changed_grid_is_rejected(tmp_path, monkeypatch, damage):
    plan, original = planned_fixture(tmp_path, monkeypatch)
    spec = tts.bound_json(plan["protocol"])
    if damage == "drop_trial":
        plan["trials"].pop()
    elif damage == "duplicate_trial":
        plan["trials"].append(copy.deepcopy(plan["trials"][0]))
    elif damage == "drop_audio":
        plan["audio"].pop()
    elif damage == "changed_reference":
        plan["trials"][0]["command"] = {"intent": "TIME", "slots": {}}
    else:
        plan["audio"][0]["speed"] = 0.85
    with pytest.raises(ValueError, match="coverage|reference|grid"):
        tts.validate_plan(plan, original, spec)


@pytest.mark.parametrize("field,value", [
    ("speeds", [0.9, 0.9]), ("speeds", []), ("speeds", [True]),
    ("speeds", [float("inf")]), ("speeds", [0]),
    ("voices", []), ("voices", [{"id": "af_heart"}, {"id": "af_heart"}]),
])
def test_invalid_grid_axes_cannot_create_a_qualifying_plan(
        tmp_path, monkeypatch, field, value):
    spec = grid_spec()
    spec[field] = value
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="voice|speed|grid"):
        planned_fixture(tmp_path, monkeypatch, protocol)


def generation_fixture(tmp_path, monkeypatch, synthesize, *, available=None,
                       session_providers=None):
    """Replace only external TTS inference; preserve real filesystem/audio QA."""
    monkeypatch.setattr(tts, "ROOT", tmp_path)
    spec = {**grid_spec(), "kokoro_onnx_version": "0.5.0", "threads": 1, "seed": 53}
    assets = tmp_path / ".cache" / "assets"
    assets.mkdir(parents=True)
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        (assets / name).write_bytes(name.encode())
    spec["files"] = {path.name: tts.sha(path) for path in assets.iterdir()}
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(spec), encoding="utf-8")
    source = tmp_path / "contract.json"
    source.write_text(json.dumps(contract()), encoding="utf-8")
    audio, refs = tts.expand_grid(merge_cases(contract()), spec)
    plan = {"protocol": tts.binding(protocol), "contract": tts.binding(source),
            "source_counts": {"cases": 1, "catalog_entries": 2},
            "optionb_trials": 6, "audio": audio, "trials": refs}
    plan_path = tmp_path / ".cache" / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / ".cache" / "generation"
    available = ["CPUExecutionProvider"] if available is None else available

    def session_factory(model, options, *, providers):
        active = ([p for p in providers if p in available]
                  if session_providers is None else session_providers)
        return SimpleNamespace(get_providers=lambda: active, disable_fallback=lambda: None)

    fake_ort = SimpleNamespace(SessionOptions=SimpleNamespace, set_seed=lambda seed: None,
                               get_available_providers=lambda: available,
                               preload_dlls=lambda: None, InferenceSession=session_factory)

    class FixtureKokoro:
        voices = ["af_heart", "am_puck"]
        tokenizer = SimpleNamespace(phonemize=lambda text, lang: "fixture")

        @staticmethod
        def create(text, *, voice, lang, speed, trim):
            assert trim is False
            return synthesize(speed)

        @classmethod
        def from_session(cls, session, voices):
            return cls()

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", SimpleNamespace(Kokoro=FixtureKokoro))
    monkeypatch.setattr(tts.importlib.metadata, "version", lambda name: "0.5.0")
    return plan_path, assets, output


def test_generation_uses_each_declared_speed_and_records_identity(tmp_path, monkeypatch):
    def synthesis(speed):
        size = {0.9: 2400, 1.0: 4800, 1.1: 7200}[speed]
        return np.sin(np.arange(size) * 0.1).astype(np.float32) * 0.2, 24000

    plan, assets, output = generation_fixture(tmp_path, monkeypatch, synthesis)
    summary = tts.generate(plan, assets, output)
    assert summary["generation_faults"] == 0
    rows = tts.read_json(output / "manifest.json")["records"]
    assert len(rows) == 12
    assert {(row["voice"], row["speed"]) for row in rows} == {
        ("af_heart", 0.9), ("af_heart", 1.0), ("af_heart", 1.1),
        ("am_puck", 0.9), ("am_puck", 1.0), ("am_puck", 1.1)}
    for row in rows:
        assert row["samples"] == {0.9: 1600, 1.0: 3200, 1.1: 4800}[row["speed"]]
        assert row["native_sample_rate"] == 24000
        assert tts.sha(row["native"]["path"]) == row["native"]["sha256"]
        assert Path(row["native_array"]["path"]).is_file()
        assert row["spoken_text"] in {
            "Brightness twenty percent", "turn on the mystery light"}
    review = tts.read_json(output / "review-template.json")
    assert all(row["heard_exact"] is None and row["speed"] in (0.9, 1.0, 1.1)
               for row in review["items"])
    with pytest.raises(ValueError):
        review_complete(rows, review)


@pytest.mark.parametrize("wave,rate", [
    (np.zeros(2400, dtype=np.float32), 24000),
    (np.array([np.nan], dtype=np.float32), 24000),
    (np.ones(2400, dtype=np.float32) * 1.1, 24000),
    (np.ones((2400, 2), dtype=np.float32) * 0.1, 24000),
    (np.ones(192001, dtype=np.float32) * 0.1, 24000),
    (np.ones(597, dtype=np.float32) * 0.1, 24000),
    (np.ones(2400, dtype=np.float32) * 0.1, 16000),
    (np.ones((2, 3, 4), dtype=np.float32) * 0.1, 24000),
])
def test_native_synthesis_is_preserved_even_when_capture_qa_fails(
        tmp_path, monkeypatch, wave, rate):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (wave, rate))
    summary = tts.generate(plan, assets, output)
    assert summary["pass"] is False
    rows = tts.read_json(output / "manifest.json")["records"]
    assert len(rows) == 12
    assert summary["generation_faults"] == 12
    for row in rows:
        assert row["status"] == "generation_fault"
        stored = row["native_array"]
        assert tts.sha(stored["path"]) == stored["sha256"]
        np.testing.assert_equal(np.load(stored["path"], allow_pickle=False), wave)
        if wave.ndim in (1, 2):
            assert Path(row["native"]["path"]).is_file()


def test_generation_cannot_cherry_pick_a_subset_of_the_grid(tmp_path, monkeypatch):
    plan_path, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000))
    plan = tts.read_json(plan_path)
    plan["audio"].pop()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage|grid"):
        tts.generate(plan_path, assets, output)
    assert not output.exists()


def test_explicit_cuda_generation_records_the_actual_provider_and_gpu_pin(
        tmp_path, monkeypatch):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    tts.generate(plan, assets, output, provider="cuda")
    manifest = tts.read_json(output / "manifest.json")
    assert manifest["execution_provider"] == "CUDAExecutionProvider"
    assert manifest["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert manifest["cuda_visible_devices"] == "0"
    assert len(manifest["records"]) == 12


@pytest.mark.parametrize("available,active", [
    (["CPUExecutionProvider"], None),
    (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
])
def test_requested_cuda_cannot_silently_generate_on_cpu(
        tmp_path, monkeypatch, available, active):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000),
        available=available, session_providers=active)
    with pytest.raises(ValueError, match="CUDA"):
        tts.generate(plan, assets, output, provider="cuda")
    assert not output.exists()


def test_joint_supplied_slot_correctness_requires_every_slot_in_one_command():
    expected = {"intent": "BRIGHTNESS", "slots": {"percent": 20, "light_id": "light.kitchen"}}
    trial = {**trials()[0], "command": expected}
    output = {"audio_id": trial["audio_id"], "final_application": {
        "accepted": True, "command": {"intent": "BRIGHTNESS", "slots": {
            "percent": 20, "light_id": "light.bedroom"}}, "reason": None}}
    metrics = score_cases([trial], [output])["unique_audio"]
    assert metrics["supplied_slots"]["percent"]["correct_accepted"] == 1
    assert metrics["supplied_slots"]["light_id"]["correct_accepted"] == 0
    assert metrics["joint_supplied_slots"] == {"n": 1, "correct_accepted": 0, "accuracy": 0.0}
    assert metrics["complete_command_recall"] == 0.0
    assert metrics["wrong_accepted_positive"] == 1


def test_extra_typed_slot_fails_exact_command_even_when_supplied_slots_match():
    output = positive_result()
    output["final_application"]["command"]["slots"]["light_id"] = "light.kitchen"
    metrics = score_cases([trials()[0]], [output])["unique_audio"]
    assert metrics["joint_supplied_slots"]["correct_accepted"] == 1
    assert metrics["correct_positive"] == 0
    assert metrics["strict_accuracy"] == 0


def test_empty_slot_commands_are_outside_the_joint_slot_denominator():
    row = {**trials()[0], "command": {"intent": "TIME", "slots": {}}}
    output = {"audio_id": row["audio_id"], "final_application": {
        "accepted": True, "command": {"intent": "TIME", "slots": {}}, "reason": None}}
    metrics = score_cases([row], [output])["unique_audio"]
    assert metrics["joint_supplied_slots"] == {"n": 0, "correct_accepted": 0, "accuracy": None}
    assert metrics["complete_command_recall"] == 1.0


def test_fault_missing_and_wrong_negative_rejections_keep_separate_denominators():
    rows = [trials()[0]]
    for name in ("negative-fault", "negative-missing", "negative-wrong", "negative-accepted"):
        rows.append({**trials()[-1], "audio_id": name, "case_id": name})
    outputs = [positive_result(), {"audio_id": "negative-fault", "fault": "read_error"},
               {"audio_id": "negative-wrong", "final_application": {
                   "accepted": False, "command": None, "reason": "OOV"}},
               {"audio_id": "negative-accepted", "final_application": {
                   "accepted": True, "command": {"intent": "TIME", "slots": {}},
                   "reason": None}}]
    metrics = score_cases(rows, outputs)["unique_audio"]
    assert metrics["negative"] == 4
    assert metrics["negative_faults"] == 1
    assert metrics["negative_missing"] == 1
    assert metrics["incorrect_rejections"] == 1
    assert metrics["false_accepts"] == 1
    assert metrics["strict_accuracy"] == 0.2


@pytest.mark.parametrize("app", [[], None, {"accepted": True, "command": {
    "slots": {}}, "reason": None}, {"accepted": False, "command": None, "reason": True},
    {"accepted": True, "command": {"intent": "TIME", "slots": {}}, "reason": "OOV"}])
def test_malformed_application_decisions_are_faults(app):
    metrics = score_cases([trials()[-1]], [{"audio_id": trials()[-1]["audio_id"],
                                          "final_application": app}])
    assert metrics["unique_audio"]["faults"] == 1
    assert metrics["unique_audio"]["correct_rejections"] == 0


def test_metrics_separate_every_voice_and_speed_combination():
    audio, refs = tts.expand_grid(merge_cases(contract()), grid_spec())
    results = [{"audio_id": row["audio_id"], "final_application": {
        "accepted": True, "command": {"intent": "BRIGHTNESS", "slots": {"percent": 20}},
        "reason": None}} for row in audio
        if row["id"] == "af_heart" and row["speed"] == 0.9
        and row["text"] == "Brightness twenty percent"]
    result = score_cases(refs, results)
    assert result["case_references"] == 18
    assert result["unique_audio"]["trials"] == 12
    assert result["per_speed"]["0.9"]["trials"] == 4
    assert result["per_voice_speed"]["af_heart"]["0.9"]["recall"] == 1.0
    assert result["per_voice_speed"]["af_heart"]["1.0"]["recall"] == 0.0
    assert result["per_voice_speed"]["am_puck"]["0.9"]["missing"] == 2
    assert result["case_reference_metrics"]["strict_correct"] == 2
    assert result["pass"] is False


def test_missed_cases_export_exact_verbatims_and_all_shared_references():
    result = score_cases(trials(), [positive_result(60), negative_result()])
    misses = result["missed_cases"]
    assert [row["case_id"] for row in misses] == ["cases/B1-001", "catalog_entries/B1-001"]
    assert all(row["spoken_text"] == "Brightness twenty percent" for row in misses)
    assert all(row["outcome"] == "wrong_command" and row["speed"] == 1.0 for row in misses)
    assert all(row["expected_command"]["slots"]["percent"] == 20 for row in misses)
    assert all(row["final_application"]["command"]["slots"]["percent"] == 60
               for row in misses)


def test_conflicting_shared_waveform_references_cannot_be_silently_pooled():
    rows = trials()
    rows[1]["command"] = {"intent": "TIME", "slots": {}}
    with pytest.raises(ValueError, match="Conflicting"):
        score_cases(rows, [positive_result(), negative_result()])


def test_exported_missed_verbatims_preserve_faults_and_reference_text(tmp_path):
    metrics = score_cases(trials(), [negative_result()])
    path = tmp_path / "missed-verbatims.json"
    tts.export_missed_verbatims(path, metrics)
    exported = tts.read_json(path)
    assert exported["case_references"] == 2
    assert exported["unique_audio"] == 1
    assert [row["spoken_text"] for row in exported["items"]] == [
        "Brightness twenty percent", "Brightness twenty percent"]
    assert all(row["outcome"] == "missing" for row in exported["items"])
    with pytest.raises(FileExistsError):
        tts.export_missed_verbatims(path, metrics)


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_worker_count_is_bounded_before_generation(tmp_path, monkeypatch, workers):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000))
    with pytest.raises(ValueError, match="worker"):
        tts.generate(plan, assets, output, workers=workers)
    assert not output.exists()


def simulated_spawn_pool(monkeypatch, *, fail_before=(), lose_result=()):
    """Replace external processes; run real worker synthesis/persistence code."""
    process = {"pid": 1000}
    monkeypatch.setattr(tts.os, "getpid", lambda: process["pid"])

    class SpawnPool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            if mp_context.get_start_method() != "spawn":
                raise ValueError("Independent spawn context required")
            self.contexts = []
            for index in range(max_workers):
                process["pid"] = 1000 + index
                initializer(*initargs)
                self.contexts.append(tts._WORKER_RUNTIME)
            self.jobs = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, item):
            index = self.jobs
            self.jobs += 1
            tts._WORKER_RUNTIME = self.contexts[index % len(self.contexts)]
            future = Future()
            try:
                if index in fail_before:
                    raise RuntimeError("fixture process failed before synthesis")
                result = function(item)
                if index in lose_result:
                    raise RuntimeError("fixture result transport failed after persistence")
                future.set_result(result)
            except Exception as error:
                future.set_exception(error)
            return future

    monkeypatch.setattr(tts, "ProcessPoolExecutor", SpawnPool)
    monkeypatch.setattr(tts, "as_completed", lambda futures: iter(reversed(list(futures))))


def test_spawned_workers_keep_the_complete_grid_and_canonical_parent_records(
        tmp_path, monkeypatch):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000))
    simulated_spawn_pool(monkeypatch)
    summary = tts.generate(plan, assets, output, workers=3)
    manifest = tts.read_json(output / "manifest.json")
    expected = sorted(row["audio_id"] for row in tts.read_json(plan)["audio"])
    assert [row["audio_id"] for row in manifest["records"]] == expected
    assert summary["generated"] == 12 and summary["generation_faults"] == 0
    assert manifest["requested_workers"] == 3
    assert {row["process_id"] for row in manifest["worker_receipts"]} == {1000, 1001, 1002}
    assert all(row["start_method"] == "spawn" for row in manifest["worker_receipts"])
    assert all(row["seed"] == 53 for row in manifest["worker_receipts"])
    assert all(row["providers"] == ["CPUExecutionProvider"]
               for row in manifest["worker_receipts"])
    assert all(row["status"] == "generated" and "native_array" in row
               for row in manifest["records"])
    assert manifest["human_reviewed"] is False
    assert len(tts.read_json(output / "review-template.json")["items"]) == 12


def test_worker_failures_are_not_retried_or_dropped(tmp_path, monkeypatch):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000))
    simulated_spawn_pool(monkeypatch, fail_before={0, 4})
    summary = tts.generate(plan, assets, output, workers=3)
    records = tts.read_json(output / "manifest.json")["records"]
    assert summary["pass"] is False and summary["generation_faults"] == 2
    assert len(records) == 12
    assert sum(row["status"] == "generated" for row in records) == 10
    assert sum(row.get("failure_stage") == "worker_execution" for row in records) == 2
    assert len(list((output / "native").glob("*.npy"))) == 10


def test_persisted_worker_result_is_recovered_without_a_second_take(tmp_path, monkeypatch):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000))
    simulated_spawn_pool(monkeypatch, lose_result={0})
    summary = tts.generate(plan, assets, output, workers=2)
    manifest = tts.read_json(output / "manifest.json")
    assert summary["generation_faults"] == 0
    assert len(manifest["records"]) == 12
    assert len(list((output / "native").glob("*.npy"))) == 12
    assert len(manifest["worker_transport_faults"]) == 1
    assert manifest["worker_transport_faults"][0]["recovered_persisted_record"] is True


def test_cuda_workers_inherit_the_one_visible_gpu_pin(tmp_path, monkeypatch):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    simulated_spawn_pool(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture-one")
    tts.generate(plan, assets, output, provider="cuda", workers=2)
    manifest = tts.read_json(output / "manifest.json")
    assert {row["cuda_visible_devices"] for row in manifest["worker_receipts"]} == {"GPU-fixture-one"}
    assert all(row["execution_provider"] == "CUDAExecutionProvider"
               for row in manifest["worker_receipts"])


@pytest.mark.parametrize("gpu_pin", [None, "", "0,1", "-1"])
def test_cuda_pool_requires_one_explicit_visible_gpu(tmp_path, monkeypatch, gpu_pin):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if gpu_pin is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_pin)
    with pytest.raises(ValueError, match="GPU|CUDA"):
        tts.generate(plan, assets, output, provider="cuda", workers=2)
    assert not output.exists()


def test_gpu_distribution_version_is_recorded_without_the_cpu_distribution(
        tmp_path, monkeypatch):
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1, 24000))

    def version(name):
        if name == "onnxruntime":
            raise tts.importlib.metadata.PackageNotFoundError(name)
        return "1.23.2" if name == "onnxruntime-gpu" else "0.5.0"

    monkeypatch.setattr(tts.importlib.metadata, "version", version)
    tts.generate(plan, assets, output)
    manifest = tts.read_json(output / "manifest.json")
    assert manifest["versions"]["onnxruntime-gpu"] == "1.23.2"
    assert "onnxruntime" not in manifest["versions"]


def test_cuda_provider_fallback_is_disabled_before_any_take(tmp_path, monkeypatch):
    fallback = {"enabled": True}
    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, lambda speed: (np.ones(2400) * 0.1,
                                              16000 if fallback["enabled"] else 24000),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fake_ort = sys.modules["onnxruntime"]
    original = fake_ort.InferenceSession

    def session_factory(*args, **kwargs):
        session = original(*args, **kwargs)
        session.disable_fallback = lambda: fallback.update(enabled=False)
        return session

    fake_ort.InferenceSession = session_factory
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    summary = tts.generate(plan, assets, output, provider="cuda")
    assert summary["generated"] == 12 and summary["pass"] is True
    assert all(row["native_sample_rate"] == 24000
               for row in tts.read_json(output / "manifest.json")["records"])


def test_runtime_provider_loss_preserves_the_take_and_blocks_remaining_cpu_takes(
        tmp_path, monkeypatch):
    state = {"lost": False}

    def synthesis(speed):
        state["lost"] = True
        return np.ones(2400) * 0.1, 24000

    plan, assets, output = generation_fixture(
        tmp_path, monkeypatch, synthesis,
        available=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fake_ort = sys.modules["onnxruntime"]
    original = fake_ort.InferenceSession

    def session_factory(*args, **kwargs):
        session = original(*args, **kwargs)
        startup = session.get_providers()
        session.get_providers = lambda: ["CPUExecutionProvider"] if state["lost"] else startup
        return session

    fake_ort.InferenceSession = session_factory
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    summary = tts.generate(plan, assets, output, provider="cuda")
    assert summary["generation_faults"] == 12 and summary["pass"] is False
    manifest = tts.read_json(output / "manifest.json")
    assert len(manifest["records"]) == 12
    assert len(list((output / "native").glob("*.npy"))) == 1
    assert manifest["execution_provider"] is None
