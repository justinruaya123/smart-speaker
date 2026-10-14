"""Protect component isolation, leakage exclusions and bounded model selection."""
import copy
import importlib
import csv
import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest
import torch
import soundfile as sf

from scripts.models.vcm_intent_slots import IntentSlotsVCM
from scripts.test.test_vcm_intent_slots_contract import contract
from scripts.test.speaker_demo_suite import pcm_digest
from scripts.train.vcm_intent_slots_data import sha


def module():
    try:
        return importlib.import_module("scripts.train.vcm_intent_slots_numeral")
    except ModuleNotFoundError as error:
        pytest.fail(f"Missing isolated numeral experiment: {error}")


def example(person, value, take=0):
    name = f"{person:02}{value:03}{take:02}"
    row = {"id": name, "kind": "human", "published_split": "numerals",
           "reference_status": "auxiliary", "qa_eligible": True,
           "audio_flags": [], "command": None, "split": "train",
           "sample_rate": 16000, "channels": 1, "samples": 4000,
           "source_revision": "revision", "license": "research",
           "speaker_id": f"MLEnd_numerals/{person}", "pair_id": name,
           "sha256": name.ljust(64, "a"), "pcm_sha256": name.ljust(64, "b"),
           "publisher_metadata": {"source": "MLEnd_numerals",
               "speaker_id": str(person), "numerals": str(value),
               "file": f"audio/numerals_000_MLEnd_numerals_{name}.wav"}}
    annotation = {"Public filename": name, "Numeral": str(value),
                  "Speaker": str(person), "Intonation": "neutral"}
    return row, annotation


def corpus():
    pairs = [example(p, v, take) for p in range(10) for v in (3, 20)
             for take in range(3)]
    return [p[0] for p in pairs], {p[1]["Public filename"]: p[1] for p in pairs}


def test_cap_is_deterministic_and_people_pcm_pairs_are_disjoint():
    m = module()
    data, annotations = corpus()
    original = copy.deepcopy(data)
    views, audit = m.partition(data, [], annotations)
    other, _ = m.partition(list(reversed(data)), [], annotations)
    assert views == other and data == original
    assert [len(views[s]) for s in ("train", "dev", "test")] == [14, 2, 4]
    assert audit["exclusions"]["repeated_take_cap"] == 40
    assert max(Counter((r["speaker_id"], r["numeric_value"])
                       for rs in views.values() for r in rs).values()) == 1
    for key in ("speaker_id", "pcm_sha256", "pair_id", "sha256"):
        sets = [{r[key] for r in views[s]} for s in ("train", "dev", "test")]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_reserved_people_and_pcm_and_conflicting_duplicates_are_excluded():
    m = module()
    data, annotations = corpus()
    data[6]["pcm_sha256"] = data[3]["pcm_sha256"]  # values 3 and 20 conflict
    reserved = [{"speaker_id": "MLEnd_numerals/8"},
                {"pcm_sha256": data[-1]["pcm_sha256"]}]
    views, audit = m.partition(data, reserved, annotations)
    retained = [r for rs in views.values() for r in rs]
    assert all(r["speaker_id"] != "MLEnd_numerals/8" for r in retained)
    assert all(r["pcm_sha256"] not in (data[3]["pcm_sha256"], data[-1]["pcm_sha256"])
               for r in retained)
    assert audit["exclusions"]["conflicting_identity_labels"] == 2
    assert audit["exclusions"]["reserved_identity"] == 7


def test_original_metadata_conflicts_and_out_of_range_are_not_relabelled():
    m = module()
    data, annotations = corpus()
    extra, note = example(0, 1000)
    data.append(extra)
    annotations[note["Public filename"]] = note
    annotations[data[0]["id"]]["Speaker"] = "999"
    data[1]["kind"] = "synthetic"
    views, audit = m.partition(data, [], annotations)
    assert audit["exclusions"]["original_metadata_conflict"] == 1
    assert audit["exclusions"]["not_original_eligible_human"] == 1
    assert audit["exclusions"]["outside_numeric_head"] == 1
    assert {r["numeric_value"] for rs in views.values() for r in rs} == {3, 20}


def test_duplicate_ids_block_and_training_requires_each_observed_value():
    m = module()
    data, annotations = corpus()
    with pytest.raises(ValueError, match="duplicate.*id"):
        m.partition(data + [data[0]], [], annotations)
    views, _ = m.partition(data, [], annotations)
    views["dev"] = [r for r in views["dev"] if r["numeric_value"] == 3]
    with pytest.raises(ValueError, match="coverage"):
        m.require_coverage(views)


def test_duration_format_is_pruned_before_takes_are_selected():
    m = module()
    data, annotations = corpus()
    data[0]["samples"] = 128001
    data[1]["sample_rate"] = 8000
    _, audit = m.partition(data, [], annotations)
    assert audit["exclusions"]["not_original_eligible_human"] == 2


def test_only_numeric_loss_updates_observed_query_and_encoder():
    m = module()
    torch.manual_seed(53)
    c = contract()
    net = IntentSlotsVCM(c)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    optimizer = m.numeric_optimizer(net)
    for _ in range(3):
        optimizer.zero_grad()
        out = net(torch.randn(2, 40, 9), torch.tensor([9, 7]))
        loss = m.numeric_loss(out, [3, 20], c)
        loss.backward()
        optimizer.step()
    active = c["roles"].index("brightness_percent") + 1
    for name, weight in net.state_dict().items():
        if name.startswith(("intent.", "validity.", "states.", "entities.")) or (
                name.startswith("values.") and not name.startswith("values.brightness_percent.")):
            assert torch.equal(weight, before[name]), name
    inactive = [n for n in range(len(c["roles"]) + 1) if n != active]
    assert torch.equal(net.queries[inactive], before["queries"][inactive])
    assert not torch.equal(net.queries[active], before["queries"][active])
    assert not torch.equal(net.stem.weight, before["stem.weight"])
    assert len(c["finite_values"]["brightness_percent"]) == 101


def test_balanced_numeric_sampler_preserves_people_and_value_mass():
    m = module()
    data = [{"numeric_value": 3, "speaker_id": "a"}] * 100
    data += [{"numeric_value": 3, "speaker_id": "b"},
             {"numeric_value": 20, "speaker_id": "c"}]
    draws = m.sample_numeric(data, 53, 20000)
    assert np.array_equal(draws, m.sample_numeric(data, 53, 20000))
    masses = Counter(data[i]["speaker_id"] for i in draws)
    assert 4700 < masses["a"] < 5300 and 4700 < masses["b"] < 5300
    assert 9700 < masses["c"] < 10300


def test_early_stopping_keeps_prior_best_and_never_extends_budget():
    m = module()
    stop = m.DevelopmentStop(max_checks=4, patience=2)
    assert stop.observe(.5, 2.) == (True, False)
    assert stop.observe(.5, 2.1) == (False, False)
    assert stop.observe(.4, 1.) == (False, True)
    assert stop.best_check == 1
    full = m.DevelopmentStop(max_checks=4, patience=2)
    for n in range(4):
        improved, stopped = full.observe(n / 10, 4. - n)
        assert improved and stopped == (n == 3)
    with pytest.raises(ValueError, match="stopped"):
        full.observe(.9, .1)


def test_component_lock_cannot_ignore_changed_graph_or_incomplete_bindings(tmp_path):
    m = module()
    source = tmp_path / "original"
    source.write_text("immutable")
    with pytest.raises(ValueError, match="binding"):
        m.lock_component({"checkpoint": source}, tmp_path / "bad.json")
    names = m.COMPONENT_BINDINGS
    lock = tmp_path / "selection-before-component-test.json"
    m.lock_component({n: source for n in names}, lock)
    result = m.verify_component(lock)
    assert result["qualification"] is False and result["calibration"] is None
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        m.verify_component(lock)


def test_real_audio_prepare_fit_lock_and_component_test_do_not_make_commands(tmp_path):
    m = module()
    data, annotations = corpus()
    data = [r for r in data if r["id"].endswith("00")]
    for i, r in enumerate(data):
        pcm = (np.sin(np.arange(4000) * (.07 + .003 * i)) * .1).astype(np.float32)
        audio = tmp_path / (r["id"] + ".wav")
        sf.write(audio, pcm, 16000, subtype="FLOAT")
        r.update(audio=str(audio), sha256=sha(audio), pcm_sha256=pcm_digest(pcm))
    aux = tmp_path / "auxiliary.jsonl"
    aux.write_text("".join(json.dumps(r) + "\n" for r in data))
    attrs = tmp_path / "attributes.csv"
    with attrs.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(annotations.values()))))
        writer.writeheader()
        writer.writerows(annotations.values())
    cpath = tmp_path / "contract.json"
    cpath.write_text(json.dumps(contract()))
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    inputs = tmp_path / "inputs.json"
    empty = tmp_path / "reserved-empty.jsonl"
    empty.write_text("")
    inputs.write_text(json.dumps({"audio_root": str(tmp_path), "views": {
        s: {"path": str(empty), "sha256": sha(empty)}
        for s in ("train", "dev", "calibration", "test", "holdout")},
        "contract": {"path": str(cpath), "sha256": sha(cpath)},
        "catalog": {"path": str(catalog), "sha256": sha(catalog)}}))
    norm = tmp_path / "normalization.json"
    norm.write_text(json.dumps({"fit_split": "train", "mean": [0.] * 40, "std": [1.] * 40}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"experiment_id": "vcm-intent-slots-numeral-aux-v1",
        "purpose": "numeral_category_component_only", "seed": 53,
        "max_checks": 1, "steps_per_check": 1, "batch_size": 4, "patience": 2,
        "sources": {"metadata_sha256": sha(attrs)},
        "update_artifacts": False, "provider_actions_allowed": False,
        "installation_allowed": False, "calibration": None}))
    prepared, fitted = tmp_path / "prepared", tmp_path / "fitted"
    m.prepare(inputs, aux, attrs, norm, protocol, prepared)
    with pytest.raises(FileExistsError):
        m.prepare(inputs, aux, attrs, norm, protocol, prepared)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.score_run(fitted, tmp_path / "premature-test")
    result = m.fit(prepared, fitted)
    assert result["updates"] == 1 and result["qualification"] is False
    assert (fitted / "selection-before-component-test.json").is_file()
    scored = m.score_run(fitted, tmp_path / "scored")
    assert scored["selected"]["n"] == 4
    assert scored["qualification"] is False and scored["actions_enabled"] is False
    assert "command" not in scored["selected"] and scored["calibration"] is None
    cli = subprocess.run([sys.executable, "-m", "scripts.train.vcm_intent_slots_numeral",
        "score", "--run", str(fitted), "--output", str(tmp_path / "scored-cli")],
        capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert json.loads((tmp_path / "scored-cli/metrics.json").read_text())["selected"]["n"] == 4
    first = next(iter((prepared / "features").glob("*.npy")))
    first.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="feature.*changed"):
        m.score_run(fitted, tmp_path / "changed-feature-test")


@pytest.mark.parametrize("bad_input", ["missing_parent_partition", "substituted_original_csv"])
def test_preparation_enforces_parent_partitions_and_original_csv_pin(tmp_path, bad_input):
    m = module()
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    attrs = tmp_path / "attributes.csv"
    attrs.write_text("Public filename,Numeral,Intonation,Speaker\n")
    cpath = tmp_path / "contract.json"
    cpath.write_text(json.dumps(contract()))
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps({"fit_split": "train", "mean": [0.] * 40, "std": [1.] * 40}))
    views = {s: {"path": str(empty), "sha256": sha(empty)}
             for s in ("train", "dev", "calibration", "test", "holdout")}
    if bad_input == "missing_parent_partition":
        del views["holdout"]
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"audio_root": str(tmp_path), "views": views,
        "catalog": {"path": str(cpath), "sha256": sha(cpath)},
        "contract": {"path": str(cpath), "sha256": sha(cpath)}}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"experiment_id": "vcm-intent-slots-numeral-aux-v1",
        "purpose": "numeral_category_component_only", "seed": 53,
        "max_checks": 1, "steps_per_check": 1, "batch_size": 4, "patience": 2,
        "update_artifacts": False, "provider_actions_allowed": False,
        "installation_allowed": False, "calibration": None,
        "sources": {"metadata_sha256": "f" * 64 if bad_input == "substituted_original_csv" else sha(attrs)}}))
    message = "parent.*partition" if bad_input == "missing_parent_partition" else "metadata.*pin"
    with pytest.raises(ValueError, match=message):
        m.prepare(inputs, empty, attrs, norm, protocol, tmp_path / "invalid-preparation")
    assert not (tmp_path / "invalid-preparation").exists()
