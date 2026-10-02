"""Shared bundle contracts for runtime, service preflight and integrity checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .routing import ACTION


def runtime_components(metadata: dict[str, Any]) -> list[str]:
    """Honor explicit capabilities; retain the historical bundle convention."""
    vcm = metadata.get("vcm", {})
    streaming = vcm.get("streaming_supported", not bool(metadata.get("acoustic_slots")))
    if type(streaming) is not bool:
        raise ValueError("streaming_supported must be a boolean")
    if metadata.get("acoustic_slots") and streaming:
        raise ValueError("Acoustic slot bundles do not support streaming")
    components = metadata.get("runtime_components")
    if components is None:
        return ["vcm", *(["vcm_stream"] if streaming else [])]
    if (not isinstance(components, list)
            or not all(isinstance(item, str) for item in components)
            or "vcm" not in components or len(set(components)) != len(components)
            or set(components) - {"vcm", "vcm_stream", "stt"}):
        raise ValueError("Invalid runtime_components; VCM is mandatory")
    if ("vcm_stream" in components) != streaming:
        raise ValueError("Streaming capability must match runtime_components")
    return components


def validate_labels(metadata: dict[str, Any]) -> list[str]:
    labels = metadata.get("vcm", {}).get("labels")
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(label, str) for label in labels)
            or len(set(labels)) != len(labels) or "UNKNOWN" not in labels
            or set(labels) - {"UNKNOWN", *ACTION}):
        raise ValueError("Invalid explicit VCM label vocabulary")
    slots = metadata.get("acoustic_slots")
    if slots:
        from .slot_contract import LABELS
        expected = slots.get("intent_labels", LABELS)
        if labels != expected:
            raise ValueError("Acoustic slot intent labels/order do not match VCM labels")
        vocab = slots.get("vocabulary", {})
        if not vocab or any(not values or len(values) != len(set(values))
                            for values in vocab.values()):
            raise ValueError("Invalid acoustic head vocabulary")
    for policy in [metadata.get("routing", {}),
                   *metadata.get("routing", {}).get("variants", {}).values()]:
        if set(policy.get("acceptance_thresholds", {})) - set(labels):
            raise ValueError("Acceptance policy contains labels outside the VCM vocabulary")
    return labels


def load_runtime_contract(bundle: str | Path, variant: str = "fp32") -> dict[str, Any]:
    """Validate the selected VCM path without requiring optional parameter artifacts."""
    if variant not in {"fp32", "int8"}:
        raise ValueError("variant must be fp32 or int8")
    bundle = Path(bundle)
    path = bundle / "bundle.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}; select a bundle with --bundle")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("vcm"), dict):
        raise ValueError("Bundle metadata must contain a VCM object")
    if type(metadata.get("schema_version", 1)) is not int or metadata.get("schema_version", 1) != 1:
        raise ValueError("Unsupported bundle schema_version")
    components = runtime_components(metadata)
    validate_labels(metadata)
    for component in components:
        if component in {"stt"}:
            continue
        path = bundle / f"{component}.{variant}.onnx"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}; select a bundle with --bundle")
    return metadata


def validate_outputs(session: Any, metadata: dict[str, Any]) -> None:
    """Check named graph heads against the declared vocabularies, never by size alone."""
    outputs = {output.name: output.shape for output in session.get_outputs()}
    expected = {"logits": len(metadata["vcm"]["labels"])}
    expected.update({f"slot_{name}": len(values) for name, values in
                     metadata.get("acoustic_slots", {}).get("vocabulary", {}).items()})
    for name, size in expected.items():
        if name not in outputs or len(outputs[name]) != 2 or outputs[name][-1] != size:
            raise ValueError(f"ONNX output {name} does not match its declared vocabulary")
    # New exports can bind exact label order into ONNX; legacy exports lack this.
    declared = session.get_modelmeta().custom_metadata_map.get("vcm_labels")
    if declared is not None and json.loads(declared) != metadata["vcm"]["labels"]:
        raise ValueError("ONNX label order does not match bundle metadata")
    slot_vocab = session.get_modelmeta().custom_metadata_map.get("slot_vocabulary")
    if slot_vocab is not None and json.loads(slot_vocab) != metadata.get("acoustic_slots", {}).get("vocabulary"):
        raise ValueError("ONNX acoustic head vocabulary does not match bundle metadata")
