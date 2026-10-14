"""Genuine paired-human owner replay; official mismatches remain a separate track."""
import importlib
from pathlib import Path

from scripts.runtime.speaker_evidence import OwnerApplication
from scripts.runtime.speaker_session import SpeakerPolicy
from scripts.runtime.vcm_intent_slots_contract import digest
from scripts.test.speaker_demo_suite import read_audio
from scripts.test.vcm_intent_slots_eval import summarize_trials
from scripts.train.vcm_intent_slots_data import sha, write_new


def replay_owner(pairs, runtime, audio_root, output):
    binding = pairs["backend"]
    for path, expected in binding["files"].items():
        if sha(path) != expected:
            raise ValueError("Owner backend binding mismatch")
    if binding["model_id"] != digest(binding["files"]):
        raise ValueError("Owner identity must bind all weight/frontend files")
    if pairs.get("calibration_people_disjoint") is not True:
        raise ValueError("Disjoint human ownership calibration required")
    module, factory = binding["factory"].split(":")
    backend = getattr(importlib.import_module(module), factory)(binding)
    policy = SpeakerPolicy(**pairs["policy"])
    receipt = pairs.get("calibration_receipt")
    if not receipt or receipt.get("status") != "human_calibration_frozen":
        raise ValueError("Missing immutable human ownership calibration receipt")
    if any(v == runtime.recipe.get("encoder_id") for v in binding["files"].values()):
        raise ValueError("Command-content encoder cannot serve as owner identity")
    owner = OwnerApplication(backend, policy, binding["model_id"], pairs["calibration_id"],
                             command_encoder_id=runtime.recipe["encoder_id"],
                             calibration_receipt=receipt)
    command_trials, owner_trials, command_predictions, owner_predictions = [], [], [], []
    for trial in pairs["trials"]:
        if trial["track"] not in ("official_owner_enabled_mismatched_pairs_retained",
                                  "original_paired_human_owner_sessions"):
            raise ValueError("Explicit owner evidence track required")
        wake_row, command_row = trial["wake"], trial["command"]
        if wake_row["kind"] != "human" or command_row["kind"] != "human":
            raise ValueError("Original human wake/command pairs required")
        if not trial.get("verified_identity_labels") or not trial.get("verified_wake_span"):
            raise ValueError("Verified pair labels and accepted wake span required")
        command_trials.append({**command_row, "id": trial["id"], "tier": trial["track"]})
        owner_trials.append({**command_row, "id": trial["id"], "tier": trial["track"],
                             "command": command_row["command"] if trial["same_person"] else None})
        try:
            wake, command = read_audio(wake_row, Path(audio_root)), read_audio(command_row, Path(audio_root))
            start, end = trial["wake_span"]
            if not 0 <= start < end <= len(wake):
                raise ValueError("Invalid accepted wake span")
            app = runtime.recognize(command)["final_application"]
            arm = owner.arm(wake[start:end], trial["id"], now=0, sequence=1, start_sample=start)
            final = owner.finish(app, command, trial["id"], now=trial["command_elapsed_seconds"],
                                 sequence=2, start_sample=trial["command_start_sample"])
            command_predictions.append({"id": trial["id"], "final_application": app})
            owner_predictions.append({"id": trial["id"], "final_application": final, "arm": arm})
        except Exception as error:
            owner.session.reset()
            owner.active = None
            fault = {"id": trial["id"], "fault": type(error).__name__+": "+str(error)}
            command_predictions.append(fault)
            owner_predictions.append(fault)
    result = {"command_only": summarize_trials(command_trials, command_predictions),
              "owner_aware": summarize_trials(owner_trials, owner_predictions),
              "overlap": "reject, never extract",
              "session_dependence": "Repeated person/reference pairs are dependent",
              "actions_enabled": False}
    write_new(Path(output)/"owner-metrics.json", result)
    return result
