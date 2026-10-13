"""Large numerical stress checks. No recording is a human speech sample."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import platform

import numpy as np

from scripts.runtime.speaker_session import (
    SpeakerEvidence, SpeakerPolicy, SpeakerSession,
)
from scripts.test.speaker_demo_suite import (
    audit, plan, summarize, validate_trials, write_new,
)


def run(cases: int = 10000, model_steps: int = 80) -> dict:
    if cases < 6 or model_steps < 1:
        raise ValueError("Need at least six policy cases and one model step")
    rng = np.random.default_rng(53)
    checks = Counter()
    for index in range(cases):
        reference = rng.normal(size=64)
        reference /= np.linalg.norm(reference)
        other = rng.normal(size=64)
        other -= np.dot(other, reference) * reference
        other /= np.linalg.norm(other)
        evidence = SpeakerEvidence(reference, "a" * 64, 1.2, False)
        lock = SpeakerSession(
            SpeakerPolicy(threshold=0.8, require_windows=False)
        )
        assert lock.arm("owner", evidence, now=0).accepted
        case = index % 6
        command = evidence
        now, token = 1.0, "owner"
        if case == 1:
            command = SpeakerEvidence(other, "a" * 64, 1.2, False)
        elif case == 2:
            command = SpeakerEvidence(reference, "a" * 64, 1.2, True)
        elif case == 3:
            now = 9.0
        elif case == 4:
            token = "intruder"
        elif case == 5:
            command = SpeakerEvidence(reference, "a" * 64, 0.1, False)
        verdict = lock.finish(token, command, now=now)
        assert verdict.accepted == (case == 0), (index, verdict)
        assert lock.finish("owner", evidence, now=10).reason == "no_owner"
        checks[verdict.reason or "matching_vectors"] += 1
    rows = []
    for speaker in range(100):
        for number in range(12):
            identity = f"numerical-{speaker}-{number}"
            role = "wake" if number < 2 else "command"
            rows.append({
                "id": identity, "audio": identity + ".wav",
                "sha256": hashlib.sha256(identity.encode()).hexdigest(),
                "pcm_sha256": hashlib.sha256(
                    (identity + "pcm").encode()
                ).hexdigest(),
                "speaker_id": f"numerical-{speaker}",
                "source": "numerical-fixture-not-speech",
                "license": "software-fixture", "kind": "synthetic",
                "split": "test", "role": role,
                "command": {"intent": "STOP", "slots": {}}
                if role == "command" else None,
            })
    audited = audit(rows)
    trials = plan(rows, split="test", seed=53, impostors=10)
    validate_trials(rows, trials)
    assert len(trials) == 12000
    scored = summarize(trials, [])
    assert scored["overall"]["missing_predictions"] == 12000
    assert scored["overall"]["recall"] == 0
    assert not scored["overall"]["complete"]

    import torch
    from scripts.models.speaker_demo import DemoKeywordNet
    from scripts.train.speaker_demo_train import optimization_step

    torch.manual_seed(53)
    torch.set_num_threads(1)
    model = DemoKeywordNet(classes=4, width=16)
    features = torch.randn(12, 40, 32) * 0.05
    labels = torch.arange(12) % 4
    for index, label in enumerate(labels):
        features[index, int(label) * 8:int(label) * 8 + 8] += 2.0
    batch = (features, torch.full((12,), 32), labels, torch.arange(12))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    with torch.no_grad():
        initial = float(torch.nn.functional.cross_entropy(
            model(features, batch[1])[0], labels
        ))
    for _ in range(model_steps):
        optimization_step(model, batch, optimizer, weight=0.1)
    with torch.no_grad():
        final = float(torch.nn.functional.cross_entropy(
            model(features, batch[1])[0], labels
        ))
    assert final < initial
    candidate = DemoKeywordNet(classes=11, width=64)
    return {
        "evidence_type": "numerical_and_software_only",
        "policy_trials": cases, "policy_failures": 0,
        "policy_cases": dict(checks),
        "planned_fixture_trials": len(trials),
        "missing_prediction_denominator_preserved": 12000,
        "manifest_fixture_recordings": audited["recordings"],
        "model_smoke_steps": model_steps,
        "model_smoke_initial_cross_entropy": initial,
        "model_smoke_final_cross_entropy": final,
        "candidate_11_class_width64_parameters": sum(
            parameter.numel() for parameter in candidate.parameters()
        ),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "physical_pi": False,
        "human_audio_evaluated": 0, "speaker_accuracy_established": False,
        "deployed_acceptance_improved": "not measured",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--model-steps", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = (Path.cwd() / ".cache/speaker-demo-v1").resolve()
    if not args.output.resolve().is_relative_to(cache):
        parser.error("Use .cache/speaker-demo-v1 for output")
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    write_new(args.output, run(args.cases, args.model_steps))


if __name__ == "__main__":
    main()
