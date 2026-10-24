"""Presentation roles/scopes; no model fitting, capture or providers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'docs/assets/optionb-tiny-presentation/presentation-data.json'


def test_openwakeword_is_an_inference_stage_not_a_command_benchmark():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    assert 'inference' in data, 'OpenWakeWord inference role and status are missing'
    wake = data['inference']['openwakeword']
    assert data['inference']['publication_branch'] == 'main'
    assert wake['role'] == 'wake_detection'
    assert wake['runtime_module'] == 'scripts/runtime/iris_openwakeword.py'
    assert wake['integration_performed'] is False
    assert wake['promoted'] is False and wake['qualified'] is False
    findings = ROOT / 'experiments/iris-openwakeword-publication-v1/findings.json'
    assert hashlib.sha256(findings.read_bytes()).hexdigest() == wake['findings_sha256']
    scope = data['benchmark_scope']
    assert scope['stage'] == 'frozen_epoch44_command_only'
    assert {'wake', 'VAD/capture', 'owner', 'UI', 'providers'} <= set(scope['excludes'])
    assert 'OpenWakeWord' in data['slides'][0]['script']
    assert 'not installed' in data['slides'][0]['script']


def test_existing_frozen_model_and_two_minute_budget_are_preserved():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    graph = ROOT / 'submission/models/optionb-tiny/model.onnx'
    assert hashlib.sha256(graph.read_bytes()).hexdigest() == data['model']['bindings']['model.onnx']
    assert data['model']['epoch'] == 44 and data['model']['parameters'] == 258912
    assert data['model']['policy']['min_probability'] == .5
    assert data['model']['policy']['min_margin'] == .1
    assert len(data['slides']) == 2
    assert sum(s['target_seconds'] for s in data['slides']) == 120
    assert all(len(s['script'].split()) == s['script_word_count'] for s in data['slides'])


def test_replayed_originals_are_not_all_called_accuracy_scored():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    scored = sum(c['selected']['rows'] for c in data['cohorts'].values())
    ambiguous = sum(c['ambiguous_unscored'] for c in data['cohorts'].values())
    assert data.get('scored_original_recordings') == scored, 'Missing eligible accuracy denominator'
    assert data.get('ambiguous_original_recordings') == ambiguous
    assert data['original_recordings'] == scored + ambiguous
