"""Offline evaluator evidence/scoring checks; no neural inference is executed."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def benchmark():
    name = "scripts.test.vcm_mixed_benchmark"
    assert importlib.util.find_spec(name) is not None, "Missing bounded CUDA benchmark evaluator"
    return importlib.import_module(name)


def trial(identifier, command, *, kind="human", **extra):
    return {"id": identifier, "kind": kind, "speaker_id": "software-fixture/" + identifier,
            "command": deepcopy(command), "transcript": "Preserved verbatim " + identifier,
            "reference_status": "supported" if command is not None else "unsupported", **extra}


def prediction(identifier, command, *, accepted=True):
    return {"id": identifier, "final_application": {
        "accepted": accepted, "command": deepcopy(command) if accepted else None}}


TIME = {"intent": "TIME", "slots": {}}
BRIGHTNESS = {"intent": "BRIGHTNESS", "slots": {"light_id": "light.all", "percent": 20}}


def test_joint_slots_requires_matching_intent_exact_types_and_no_extra_fields():
    commands = [BRIGHTNESS, {"intent": "TIME", "slots": {"light_id": "light.all", "percent": 20}},
                {"intent": "BRIGHTNESS", "slots": {"light_id": "light.all", "percent": 20.0}},
                {"intent": "BRIGHTNESS", "slots": {"light_id": "light.all", "percent": 20, "extra": "invented"}},
                {"intent": "BRIGHTNESS", "slots": {"light_id": "light.all"}}, TIME,
                {"intent": "TIME", "slots": {"extra": "invented"}}]
    rows = [trial(str(i), BRIGHTNESS if i < 5 else TIME) for i in range(7)]
    result = benchmark().summarize(rows, [prediction(str(i), c) for i, c in enumerate(commands)])
    assert result["all"]["complete_command_correct"] == 2
    assert result["all"]["joint_supplied_slots"] == {
        "n": 7, "correct": 2, "accuracy": 2 / 7,
        "with_supplied_slots_n": 5, "with_supplied_slots_correct": 1,
        "with_supplied_slots_accuracy": 1 / 5, "extra_slot_commands": 2}
    assert result["all"]["every_slot"]["percent"] == {"n": 5, "correct": 2, "accuracy": 2 / 5}


def test_positive_faults_missing_and_negative_faults_stay_in_denominators():
    rows = [trial("correct", TIME), trial("fault", BRIGHTNESS), trial("missing", BRIGHTNESS),
            trial("negative", None), trial("false-accept", None), trial("negative-fault", None),
            trial("ambiguous", None, reference_status="ambiguous")]
    predictions = [prediction("correct", TIME), {"id": "fault", "fault": "software fixture fault"},
                   prediction("negative", None, accepted=False), prediction("false-accept", TIME),
                   {"id": "negative-fault", "fault": "software fixture fault"}]
    result = benchmark().summarize(rows, predictions)["all"]
    assert (result["supported_n"], result["negative_n"], result["ambiguous_n"]) == (3, 3, 1)
    assert result["complete_command_accuracy"] == 1 / 3
    assert result["strict_accuracy"] == 2 / 6
    assert result["joint_supplied_slots"]["n"] == 3
    assert result["joint_supplied_slots"]["with_supplied_slots_n"] == 2
    assert result["every_slot"]["percent"] == {"n": 2, "correct": 0, "accuracy": 0.0}
    assert (result["false_accepts"], result["correct_rejections"], result["faults"], result["missing"]) == (1, 1, 2, 2)


def test_synthetic_success_cannot_improve_human_results_or_fresh_accounting():
    rows = [trial("human", TIME), trial("synthetic", TIME, kind="synthetic")]
    result = benchmark().summarize(rows, [prediction("human", None, accepted=False),
                                         prediction("synthetic", TIME)])
    assert result["kinds"]["human"]["complete_command_accuracy"] == 0
    assert result["kinds"]["synthetic"]["complete_command_accuracy"] == 1
    assert result["macro_person_recall"] == 0
    assert result["qualification"] is False
    assert result["physical_pi_evidence"] is False
    assert result["resident_service_evidence"] is False


def test_misses_preserve_verbatim_references_predictions_and_all_faults():
    rows = [trial("miss", BRIGHTNESS, publisher_metadata={"transcript": "Exact publisher words"}),
            trial("negative", None), trial("fault", TIME), trial("missing", TIME)]
    predictions = [prediction("miss", None, accepted=False), prediction("negative", TIME),
                   {"id": "fault", "fault": "Unreadable fixture audio"}]
    before = deepcopy((rows, predictions))
    misses = benchmark().misses(rows, predictions)
    assert [r["outcome"] for r in misses] == ["false_reject", "false_accept", "fault", "missing"]
    assert [r["verbatim"] for r in misses] == [r["transcript"] for r in rows]
    assert misses[0]["source_row"] == rows[0]
    assert misses[0]["source_row"]["publisher_metadata"]["transcript"] == "Exact publisher words"
    assert misses[-1]["prediction"] is None
    assert (rows, predictions) == before


@pytest.mark.parametrize("damage", ["duplicate_trial", "duplicate_prediction", "unexpected_prediction"])
def test_duplicate_or_unexpected_outputs_cannot_inflate_results(damage):
    rows, predictions = [trial("x", TIME)], [prediction("x", TIME)]
    if damage == "duplicate_trial":
        rows += deepcopy(rows)
    elif damage == "duplicate_prediction":
        predictions += deepcopy(predictions)
    else:
        predictions.append(prediction("other", TIME))
    with pytest.raises(ValueError):
        benchmark().summarize(rows, predictions)


def audio_fixture(tmp_path):
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest
    from scripts.train.vcm_intent_slots_data import sha

    path = tmp_path / "software-fixture.wav"
    sf.write(path, np.linspace(-0.2, 0.2, 1600, dtype=np.float32), 16000, subtype="PCM_16")
    pcm, _ = sf.read(path, dtype="float32")
    return {"id": "software-fixture", "audio": path.name, "sha256": sha(path),
            "pcm_sha256": pcm_digest(pcm), "sample_rate": 16000, "samples": len(pcm), "channels": 1}


def test_audio_preflight_binds_actual_bytes_decoded_pcm_and_length(tmp_path):
    row = audio_fixture(tmp_path)
    result = benchmark().audit_audio(row, tmp_path)
    assert result["path"] == str((tmp_path / row["audio"]).resolve())
    assert result["sha256"] == row["sha256"] and result["pcm_sha256"] == row["pcm_sha256"]
    assert result["samples"] == 1600 and result["sample_rate"] == 16000


@pytest.mark.parametrize("damage", ["file", "pcm", "samples", "channels", "rate"])
def test_audio_preflight_rejects_changed_or_misdeclared_payloads(tmp_path, damage):
    row = audio_fixture(tmp_path)
    if damage == "file":
        (tmp_path / row["audio"]).write_bytes(b"Changed fixture bytes")
    else:
        row[{"pcm": "pcm_sha256", "rate": "sample_rate"}.get(damage, damage)] = (
            "0" * 64 if damage == "pcm" else 42)
    with pytest.raises(ValueError):
        benchmark().audit_audio(row, tmp_path)


@pytest.mark.parametrize("pcm", [np.zeros(399), np.zeros(128001), np.full(400, np.nan),
                                  np.full(400, np.inf), np.full(400, 2.0), np.zeros((400, 1))],
                         ids=["short", "long", "nan", "infinity", "amplitude", "stereo"])
def test_invalid_runtime_capture_reaches_neither_gpu_nor_neural_boundary(monkeypatch, pcm):
    import torch
    from scripts.test import vcm_mixed_finalize
    from vcm import frontend
    calls = {"gpu": 0, "neural": 0, "frontend": 0}

    def unexpected(stage):
        def invoked(*args, **kwargs):
            calls[stage] += 1
            raise AssertionError("Invalid PCM reached " + stage)
        return invoked

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected("gpu"))
    monkeypatch.setattr(vcm_mixed_finalize, "infer_features", unexpected("neural"))
    monkeypatch.setattr(frontend, "features", unexpected("frontend"))
    with pytest.raises(ValueError, match="finite mono 16 kHz PCM, 25 ms..8 s"):
        benchmark().predict_audio(object(), pcm, {}, {})
    assert calls == {"gpu": 0, "neural": 0, "frontend": 0}


@pytest.mark.parametrize("samples", [400, 128000])
def test_existing_runtime_capture_endpoints_are_unchanged_without_executing_gpu(monkeypatch, samples):
    import torch
    calls = []

    class StopSoftwarePipeline(Exception):
        pass

    def stop_before_gpu():
        calls.append("accepted runtime PCM")
        raise StopSoftwarePipeline

    monkeypatch.setattr(torch.cuda, "synchronize", stop_before_gpu)
    with pytest.raises(StopSoftwarePipeline):
        benchmark().predict_audio(object(), np.zeros(samples), {}, {})
    assert calls == ["accepted runtime PCM"]


def test_fresh_people_pcm_and_recording_groups_cannot_overlap_exposed_or_fit_data():
    module = benchmark()
    reserved = [{"id": "seen", "sha256": "a" * 64, "pcm_sha256": "b" * 64,
                 "speaker_id": "source/seen", "pair_id": "pair/seen", "recording_group": "group/seen"}]
    clean = {"id": "fresh", "sha256": "c" * 64, "pcm_sha256": "d" * 64,
             "speaker_id": "source/fresh", "pair_id": "pair/fresh", "recording_group": "group/fresh"}
    assert module.check_separation([clean], reserved, fresh=True) == {"checked_rows": 1, "overlaps": []}
    for field in ("sha256", "pcm_sha256", "speaker_id", "pair_id", "recording_group"):
        with pytest.raises(ValueError, match="overlap"):
            module.check_separation([{**clean, field: reserved[0][field]}], reserved, fresh=True)
    # Exposed regression may reuse regression PCM, but never fitting PCM/people/groups.
    with pytest.raises(ValueError, match="overlap"):
        module.check_separation([{**clean, "pcm_sha256": "b" * 64}], reserved, fresh=False)


def test_explicit_aliases_raw_ids_and_pair_group_equivalence_cannot_bypass_separation():
    module = benchmark()
    reserved = [{"id": "seen", "speaker_id": "source/seen", "publisher_metadata": {"speaker_id": "raw-seen"},
                 "pair_id": "same-recording-family"}]
    clean = {"id": "new", "speaker_id": "different-source/new", "pair_id": "different-pair"}
    for extra in ({"person_aliases": ["source/seen"]}, {"resolved_speaker_aliases": "raw-seen"},
                  {"raw_speaker_id": "raw-seen"}, {"recording_group": "same-recording-family"}):
        with pytest.raises(ValueError, match="overlap"):
            module.check_separation([{**clean, **extra}], reserved, fresh=True)


def test_selection_lock_is_exclusive_and_changed_bound_bytes_fail(tmp_path):
    module = benchmark()
    files = {}
    for key in module.REQUIRED_LOCK_FILES:
        path = tmp_path / (key + ".software-fixture")
        path.write_text("Immutable software fixture " + key)
        files[key] = module.binding(path)
    evidence = {"files": files, "cohorts": {}, "audio": {}, "sources": {},
                "selected": {"checkpoint_sha256": files["checkpoint"]["sha256"]},
                "calibration_id": "a" * 64, "memory_id": "b" * 64}
    path = tmp_path / "selection-before-test.json"
    lock = module.freeze_selection(evidence, path)
    locked_sha = module.binding(path)["sha256"]
    assert lock["evaluation_neural_calls"] == 0
    assert lock["thresholds"] == {"intent": 0.95, "slot": 0.90}
    assert module.verify_selection(path) == lock
    assert module.verify_selection(path, expected_sha256=locked_sha) == lock
    with pytest.raises(FileExistsError):
        module.freeze_selection(evidence, path)
    Path(files["checkpoint"]["path"]).write_text("Changed software fixture")
    with pytest.raises(ValueError, match="binding"):
        module.verify_selection(path)


def test_replacing_selection_lock_cannot_rebind_the_running_evaluator(tmp_path):
    module = benchmark()
    path = dump(tmp_path / "software-selection-lock.json", {"software_fixture": "original lock"})
    frozen = module.binding(path)["sha256"]
    dump(path, {"software_fixture": "replacement lock"})
    with pytest.raises(ValueError, match="lock"):
        module.verify_selection(path, expected_sha256=frozen)


def official_spec():
    module = benchmark()
    root = ROOT / ".cache/vcm-mixed-final-v2/official-benchmark" / module.OFFICIAL_REVISION
    if not root.is_dir():
        pytest.skip("Pinned official scorer is unavailable; no download is permitted in pure tests")
    return {"root": str(root), "revision": module.OFFICIAL_REVISION,
            "snapshot": module.binding(root / "snapshot-receipt.json")}


def test_actual_pinned_line_parser_and_metrics_score_adapter_output_separately():
    module = benchmark()
    official = module.load_official(official_spec())
    rows = [trial("timer", {"intent": "TIMER", "slots": {"duration_seconds": 60}},
                  publisher_intent="TIMER", publisher_slot="1 minute", publisher_variation="Set a timer for 1 minute",
                  publisher_transcript="Original publisher transcript"),
            trial("negative", None, publisher_intent="OUT_OF_SCOPE", publisher_slot="",
                  publisher_variation="", publisher_transcript="Original negative transcript"),
            trial("fault", TIME, publisher_intent="TIME", publisher_slot="",
                  publisher_variation="What time is it", publisher_transcript="Original missing transcript")]
    predictions = [dict(prediction("timer", rows[0]["command"]), timing={"total_ms": 12.5}, audio_seconds=1.0),
                   dict(prediction("negative", None, accepted=False), timing={"total_ms": 8.0}, audio_seconds=1.0),
                   {"id": "fault", "fault": "software fixture audio fault"}]
    before = deepcopy(rows)
    report, records = module.official_scores(rows, predictions, official)
    assert report["command"]["n"] == 3 and report["command"]["accuracy"] == pytest.approx(2 / 3)
    assert report["intent"]["accuracy"] == pytest.approx(2 / 3)
    assert report["conditional_slots"]["TIMER"]["exact_rate"] == 1
    assert report["parsed_commands"] == 2 and report["faults"] == 1 and report["missing"] == 0
    assert records[0]["log_record"] == {"intent": "TIMER", "slot": "60 seconds", "infer_ms": 12.5, "audio_ms": 1000.0}
    assert records[0]["parsed"]["slot"] == "60 seconds"
    assert records[0]["publisher_transcript"] == "Original publisher transcript"
    assert records[-1]["log_record"] == {"event": "capture_fault", "reason": "capture_fault"}
    assert records[-1]["parsed"] is None
    assert rows == before


def test_official_missing_negative_is_not_rewritten_as_successful_typed_rejection():
    module = benchmark()
    official = module.load_official(official_spec())
    row = trial("negative", None, publisher_intent="OUT_OF_SCOPE", publisher_slot="",
                publisher_variation="", publisher_transcript="Preserved negative")
    report, records = module.official_scores([row], [], official)
    # Official reject includes no response, so its 1/1 is explicitly distinguished
    # from strict typed scoring, which retains this trial as a missing failure.
    assert report["command"]["accuracy"] == 1
    assert report["missing"] == 1
    assert report["successful_typed_rejections"] == 0
    assert module.summarize([row], [])["all"]["strict_accuracy"] == 0
    assert records[0]["outcome"] == "missing"


def test_official_loader_oos_rules_preserve_raw_publisher_fields_and_do_not_change_typed_truth():
    module = benchmark()
    official = module.load_official(official_spec())
    row = trial("publisher-oos", None, publisher_intent="PLAY_MUSIC", publisher_slot="raw slot",
                publisher_variation="Raw publisher variation", publisher_transcript="Exact untouched words",
                publisher_metadata={"command": "PLAY_MUSIC", "slot_value": "raw slot",
                    "variation": "Raw publisher variation", "transcript": "Exact untouched words", "out_of_scope": 1},
                official_intent="OUT_OF_SCOPE", official_slot="", official_variation="",
                official_transcript="Exact untouched words")
    pred = dict(prediction(row["id"], None, accepted=False), timing={"total_ms": 2.0}, audio_seconds=1.0)
    report, records = module.official_scores([row], [pred], official)
    assert report["intent"]["n_out_of_scope"] == 1
    assert report["command"]["accuracy"] == 1
    assert records[0]["publisher_intent"] == "PLAY_MUSIC"
    assert records[0]["official_reference"]["intent"] == "OUT_OF_SCOPE"
    assert records[0]["official_reference"]["slot"] == ""
    assert row["command"] is None


def test_changed_official_loader_reference_fields_fail_instead_of_becoming_new_truth():
    module = benchmark()
    official = module.load_official(official_spec())
    row = trial("bad-loader-reference", TIME, publisher_intent="TIME", publisher_slot="",
                publisher_variation="What time is it", publisher_transcript="Exact words",
                publisher_metadata={"command": "TIME", "slot_value": "", "variation": "What time is it",
                                    "transcript": "Exact words", "out_of_scope": 0},
                official_intent="PLAY_MUSIC", official_slot="", official_variation="What time is it",
                official_transcript="Exact words")
    with pytest.raises(ValueError, match="loader"):
        module.official_scores([row], [], official)


def test_official_source_mutation_is_rejected_before_import(tmp_path):
    import shutil
    module = benchmark()
    spec = official_spec()
    destination = tmp_path / "software-fixture-official-package"
    for name in module.OFFICIAL_HASHES:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(spec["root"]) / name, path)
    shutil.copyfile(spec["snapshot"]["path"], destination / "snapshot-receipt.json")
    (destination / "vcmbench/pi.py").write_text("raise RuntimeError('Must never import changed code')")
    spec.update(root=str(destination), snapshot=module.binding(destination / "snapshot-receipt.json"))
    with pytest.raises(ValueError, match="official"):
        module.load_official(spec)


def test_undefined_official_metrics_are_null_without_changing_finite_values():
    value = {"a": float("nan"), "b": [float("inf"), 0.0], "c": ("x", 3)}
    assert benchmark().json_finite(value) == {"a": None, "b": [None, 0.0], "c": ["x", 3]}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def software_postfit(tmp_path):
    """Bound text/JSON software fixtures, never loadable acoustic model weights."""
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases, extended_cases
    from scripts.runtime.vcm_intent_slots_contract import build_slot_contract, digest
    from scripts.test.vcm_mixed_finalize import SOURCE_FILES
    module = benchmark()
    contract = build_slot_contract(baseline_cases(), extended_cases(), DEFAULT_CUSTOM)
    training = tmp_path / "software-training"
    postfit = tmp_path / "software-postfit"
    selected = {"checkpoint": str(training / "software-checkpoint.pt"), "checkpoint_sha256": ""}
    training.mkdir()
    Path(selected["checkpoint"]).write_bytes(b"Software fixture only; these are not model weights")
    selected["checkpoint_sha256"] = module.binding(selected["checkpoint"])["sha256"]
    norm = {"mean": [0.0] * 40, "std": [1.0] * 40, "fit_split": "train", "frames": 5}
    norm_path = dump(training / "features/normalization.json", norm)
    dump(postfit / "normalization.json", norm)
    contract_path = dump(tmp_path / "software-contract.json", contract)
    protocol_path = dump(tmp_path / "software-protocol.json", {"software_fixture": True})
    manifest = tmp_path / "software-calibration.jsonl"
    manifest.write_text("", encoding="utf-8")
    inputs = {"protocol": module.binding(protocol_path), "contract": module.binding(contract_path),
              "views": {"calibration": module.binding(manifest)}}
    inputs_path = dump(tmp_path / "software-inputs.json", inputs)
    context = {"training": training, "inputs_path": inputs_path, "inputs": inputs,
               "protocol_path": protocol_path, "contract": contract, "normalization": norm,
               "normalization_path": norm_path, "selected": selected}
    dump(training / "selection-before-calibration.json", {"software_fixture": True})
    dump(training / "grid.json", {"software_fixture": True})
    encoder = {"encoder": selected["checkpoint_sha256"], "frontend": module.binding(ROOT / "vcm/frontend.py")["sha256"],
               "normalization": module.binding(norm_path)["sha256"]}
    memory = {"version": 1, "encoder_binding": encoder, "catalog_hash": contract["catalog_hash"], "examples": []}
    memory["memory_id"] = digest(memory)
    memory_path = dump(postfit / "entities.json", memory)
    raw_policy = {"entities": {}, "require_calibration": False, "contract_hash": contract["contract_hash"]}
    raw_path = dump(postfit / "raw-entity-policy.json", raw_policy)
    observation_path = dump(postfit / "observations.json", [])
    policy = {"require_calibration": True, "contract_hash": contract["contract_hash"], "entities": {},
              "thresholds": {"intent": .95, "slot": .90}, "qualification": False,
              "binding": {"checkpoint": encoder["encoder"], "normalization": encoder["normalization"],
                          "frontend": encoder["frontend"], "contract_hash": contract["contract_hash"],
                          "protocol": inputs["protocol"]["sha256"], "manifest": inputs["views"]["calibration"]["sha256"],
                          "memory": module.binding(memory_path)["sha256"],
                          "raw_entity_policy": module.binding(raw_path)["sha256"], "entities": {}}}
    policy["calibration_id"] = digest(policy)
    policy_path = dump(postfit / "calibration/calibration.json", policy)
    paths = {"selection": training / "selection-before-calibration.json", "grid": training / "grid.json",
             "inputs": inputs_path, "protocol": protocol_path, "contract": contract_path,
             "checkpoint": Path(selected["checkpoint"]), "normalization": postfit / "normalization.json",
             "frontend": ROOT / "vcm/frontend.py", "entities": memory_path, "raw_entity_policy": raw_path,
             "observations": observation_path, "calibration": policy_path}
    receipt = {"selected": selected, "completed_grid_candidates": 6, "grid_updates": 17280,
               "thresholds": {"intent": .95, "slot": .90}, "weights_fitted": False, "test_accessed": False,
               "evaluation_views_read": False, "final_tts_accessed": False, "qualification": False,
               "missing_calibration_intents": [], "calibration_id": policy["calibration_id"],
               "files": {name: module.binding(path) for name, path in paths.items()},
               "source_bindings": {name: module.binding(ROOT / name) for name in SOURCE_FILES},
               "training_runs": {"software-run-" + str(i): {"identity": module.binding(inputs_path)} for i in range(6)}}
    receipt_path = dump(postfit / "postfit-receipt.json", receipt)
    return context, postfit, receipt, receipt_path


def test_postfit_uses_existing_exact_selected_memory_calibration_and_normalization(software_postfit):
    context, path, receipt, _ = software_postfit
    result = benchmark().verify_postfit(context, path)
    assert result["policy"]["calibration_id"] == receipt["calibration_id"]
    assert result["memory"]["encoder_binding"]["encoder"] == context["selected"]["checkpoint_sha256"]
    assert result["receipt"]["weights_fitted"] is False


@pytest.mark.parametrize("damage", ["norm", "checkpoint", "memory", "calibration", "thresholds", "postfit_test_access"])
def test_stale_selected_or_postfit_binding_cannot_authorize_evaluation(software_postfit, damage):
    context, path, receipt, receipt_path = software_postfit
    if damage == "postfit_test_access":
        receipt["test_accessed"] = True
        dump(receipt_path, receipt)
    elif damage == "thresholds":
        policy_path = Path(receipt["files"]["calibration"]["path"])
        policy = json.loads(policy_path.read_text())
        policy["thresholds"]["intent"] = 0.94
        dump(policy_path, policy)
        receipt["files"]["calibration"] = benchmark().binding(policy_path)
        dump(receipt_path, receipt)
    else:
        key = {"norm": "normalization", "memory": "entities"}.get(damage, damage)
        Path(receipt["files"][key]["path"]).write_text("Changed software fixture bytes")
    with pytest.raises(ValueError):
        benchmark().verify_postfit(context, path)


@pytest.fixture
def software_evaluation(tmp_path, monkeypatch):
    """Small complete cohorts with real PCM; no model or inference fixtures."""
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest
    module = benchmark()
    rules = deepcopy(module.COHORTS)
    for name, rule in rules.items():
        rule["n"] = 10 if name == "fresh_human106" else 1
        rule.pop("sha256", None)
    monkeypatch.setattr(module, "COHORTS", rules)
    monkeypatch.setattr(module, "verify_official", lambda spec: {})
    protocol = {"evaluation": {"fresh_human_source": {"revision": "software-source-revision"}}}
    context = {"inputs": {"views": {}}, "protocol": protocol}
    data, cohorts = {}, {}
    count = 0
    for name, rule in rules.items():
        data[name] = []
        for index in range(rule["n"]):
            count += 1
            audio = tmp_path / f"software-audio-{count}.wav"
            sf.write(audio, np.linspace(-0.1, 0.1, 1600, dtype=np.float32) + count * 0.001,
                     16000, subtype="PCM_16")
            pcm, _ = sf.read(audio, dtype="float32")
            row = trial(f"{name}/{index}", TIME, audio=str(audio), sha256=module.binding(audio)["sha256"],
                        pcm_sha256=pcm_digest(pcm), samples=len(pcm), sample_rate=16000, channels=1,
                        split=rule["split"], role="command", metric_eligible=True,
                        source="FluentSpeechCommands" if name == "fresh_human106" else "software-fixture",
                        source_revision="software-source-revision", recording_group=f"software-group/{count}",
                        pair_id=f"software-pair/{count}", fresh_person_against_all_original_views=name == "fresh_human106")
            if name == "mandatory_defaults202":
                row.update(fresh=False, regression=True, publisher_intent="TIME", publisher_slot="",
                           publisher_variation="What time is it", publisher_transcript="Exact software publisher words",
                           publisher_metadata={"command": "TIME", "slot_value": "", "variation": "What time is it",
                                               "transcript": "Exact software publisher words", "out_of_scope": 0})
            data[name].append(row)
        path = tmp_path / (name + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in data[name]))
        receipt = dump(tmp_path / (name + "-source.json"), {"software_fixture": True})
        cohorts[name] = {"view": module.binding(path), "audio_root": str(tmp_path),
                         "source_receipts": [module.binding(receipt)]}
        if "input_view" in rule:
            context["inputs"]["views"][rule["input_view"]] = cohorts[name]["view"]
    for split in ("train", "dev", "calibration", "enrollment"):
        path = tmp_path / (split + ".jsonl")
        path.write_text("", encoding="utf-8")
        context["inputs"]["views"][split] = module.binding(path)
    spec = {"format": "vcm-mixed-offline-evaluation-v2", "cohorts": cohorts,
            "official": {"snapshot": module.binding(receipt)}}
    return context, spec, data


def rewrite_view(context, spec, name, data):
    path = Path(spec["cohorts"][name]["view"]["path"])
    path.write_text("".join(json.dumps(row) + "\n" for row in data))
    spec["cohorts"][name]["view"] = benchmark().binding(path)
    key = benchmark().COHORTS[name].get("input_view")
    if key:
        context["inputs"]["views"][key] = spec["cohorts"][name]["view"]


def test_complete_replay_retains_original_ambiguous_rows_without_reclassifying_them(software_evaluation):
    context, spec, data = software_evaluation
    name = "original_test4418"
    data[name][0].update(command=None, reference_status="ambiguous", metric_eligible=False)
    rewrite_view(context, spec, name, data[name])
    audited = benchmark().prepare_evaluation(context, spec)
    assert audited["data"][name][0]["reference_status"] == "ambiguous"
    assert audited["data"][name][0]["command"] is None
    assert audited["cohorts"][name]["regression"] is True
    assert len(audited["audio"]) == 13


def test_default_replacement_rows_never_count_as_fresh_people(software_evaluation):
    context, spec, data = software_evaluation
    data["mandatory_defaults202"][0]["fresh"] = True
    rewrite_view(context, spec, "mandatory_defaults202", data["mandatory_defaults202"])
    with pytest.raises(ValueError, match="regression"):
        benchmark().prepare_evaluation(context, spec)


def test_fresh_declared_people_cannot_reuse_any_exposed_regression_identity(software_evaluation):
    context, spec, data = software_evaluation
    data["fresh_human106"][0]["speaker_id"] = data["original_test4418"][0]["speaker_id"]
    rewrite_view(context, spec, "fresh_human106", data["fresh_human106"])
    with pytest.raises(ValueError, match="overlap"):
        benchmark().prepare_evaluation(context, spec)


def test_incomplete_cohort_and_absent_metadata_block_before_any_cuda_load(software_evaluation):
    context, spec, data = software_evaluation
    spec["cohorts"]["mandatory_defaults202"]["source_receipts"] = []
    with pytest.raises(ValueError, match="metadata"):
        benchmark().prepare_evaluation(context, spec)
    spec["cohorts"].pop("mandatory_defaults202")
    with pytest.raises(ValueError, match="four"):
        benchmark().prepare_evaluation(context, spec)


@pytest.mark.parametrize("damage", ["non_object", "cohort_list", "cohort_non_object",
                                    "missing_audio_root", "invalid_audio_root", "receipt_non_object"])
def test_malformed_evaluation_spec_is_rejected_as_a_data_gate(software_evaluation, damage):
    context, spec, _ = software_evaluation
    if damage == "non_object":
        spec = []
    elif damage == "cohort_list":
        spec["cohorts"] = list(spec["cohorts"])
    else:
        entry = spec["cohorts"]["fresh_human106"]
        if damage == "cohort_non_object":
            spec["cohorts"]["fresh_human106"] = []
        elif damage == "missing_audio_root":
            entry.pop("audio_root")
        elif damage == "invalid_audio_root":
            entry["audio_root"] = None
        else:
            entry["source_receipts"] = [None]
    with pytest.raises(ValueError, match="spec|cohort|audio_root|binding"):
        benchmark().prepare_evaluation(context, spec)
