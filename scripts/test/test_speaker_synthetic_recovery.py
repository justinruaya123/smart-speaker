"""An interrupted failed attempt must be journaled without accepting its audio."""
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.test.speaker_demo_suite import file_digest, read_json, write_new
from scripts.test.test_speaker_synthetic_data import prompts, voices
from scripts.train import speaker_synthetic_data as generation


def adapter():
    name = 'scripts.train.speaker_synthetic_recovery'
    assert importlib.util.find_spec(name) is not None, 'interruption recovery adapter is missing'
    return importlib.import_module(name)


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(generation, 'ROOT', tmp_path)
    plan = generation.build_plan(prompts(), voices())
    write_new(tmp_path / 'plan.json', plan)
    protocol = tmp_path / 'protocol.json'
    write_new(protocol, {'experiment_id': 'recovery-fixture', 'generation_plan': 'plan.json',
                        'generation': {'batch_size': 2, 'attempts': 3},
                        'bindings': {'plan.json': file_digest(tmp_path / 'plan.json')}})
    cache = tmp_path / 'generated'
    failed = cache / 'failed/train-1-0000-0-message.npy'
    failed.parent.mkdir(parents=True)
    with failed.open('xb') as stream:
        np.save(stream, np.zeros(24000, dtype=np.float32), allow_pickle=False)
    return protocol, cache, failed, plan


def recover(inputs, **kwargs):
    protocol, cache, _, _ = inputs
    return adapter().recover_attempt(protocol, cache, 'train-1', 0, 0, **kwargs)


def test_orphan_failure_becomes_skippable_attempt_without_accepting_audio(interrupted):
    _, cache, failed, plan = interrupted
    before = failed.read_bytes()
    result = recover(interrupted)
    receipt = read_json(cache / 'attempts/train-1-0000-0.json')
    assert receipt['batch_ids'] == [r['id'] for r in plan[:2]]
    assert receipt['seed'] == int(generation._hash(['recovery-fixture', 'train-1', 0, 0])[:8], 16)
    assert {r['id'] for r in receipt['failures']} == set(receipt['batch_ids'])
    assert receipt['seconds'] is None
    assert result['status'] == 'interrupted_attempt_journaled'
    assert result['accepted_rows_added'] == 0
    assert failed.read_bytes() == before
    assert not (cache / 'generated.jsonl').exists()


def test_recovery_is_idempotent(interrupted):
    _, cache, _, _ = interrupted
    result = recover(interrupted)
    path = cache / 'attempts/train-1-0000-0.json'
    before = path.read_bytes()
    assert recover(interrupted) == result
    assert path.read_bytes() == before


def test_interruption_between_recovery_journal_and_attempt_receipt_is_recoverable(interrupted, monkeypatch):
    m = adapter()
    actual = m.write_new

    def interrupt(path, value, **kwargs):
        if path.parent.name == 'attempts':
            raise InterruptedError('simulated shutdown before attempt receipt')
        return actual(path, value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(m, 'write_new', interrupt)
        with pytest.raises(InterruptedError):
            recover(interrupted)
    result = recover(interrupted)
    assert result['status'] == 'interrupted_attempt_journaled'


def test_changed_preserved_failure_is_rejected(interrupted):
    recover(interrupted)
    interrupted[2].write_bytes(b'changed failure payload')
    with pytest.raises(ValueError, match='identity mismatch'):
        recover(interrupted)


def test_recovery_cannot_invent_evidence_for_an_unstarted_attempt(interrupted):
    interrupted[2].unlink()
    with pytest.raises(ValueError, match='preserved.*evidence'):
        recover(interrupted)


def test_original_attempt_receipt_is_never_replaced(interrupted):
    path = interrupted[1] / 'attempts/train-1-0000-0.json'
    write_new(path, {'original': 'must remain'})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        recover(interrupted)
    assert path.read_bytes() == before


def test_recovery_cannot_skip_an_unjournaled_earlier_attempt(interrupted):
    protocol, cache, failed, _ = interrupted
    next_failure = failed.with_name('train-1-0000-1-message.npy')
    next_failure.write_bytes(failed.read_bytes())
    with pytest.raises(ValueError, match='earlier attempt'):
        adapter().recover_attempt(protocol, cache, 'train-1', 0, 1)


def test_complete_generation_is_not_repaired(interrupted):
    write_new(interrupted[1] / 'generation-complete.json', {'complete': True})
    with pytest.raises(ValueError, match='complete'):
        recover(interrupted)


def test_changed_recovery_attempt_receipt_is_rejected(interrupted):
    recover(interrupted)
    path = interrupted[1] / 'attempts/train-1-0000-0.json'
    value = read_json(path)
    value['failures'] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='recovery receipt'):
        recover(interrupted)


def test_frozen_generator_resumes_after_recovery_without_repeating_failed_attempt(interrupted, monkeypatch):
    protocol, cache, failed, _ = interrupted
    cfg = read_json(protocol)
    cfg['generation'].update(cuda_memory_fraction=.45, max_new_tokens=192, temperature=.75, top_p=.95)
    protocol.write_text(json.dumps(cfg))
    here = protocol.parent / 'experiment'
    write_new(here / 'voices.json', voices())
    write_new(here / 'generation-plan.json', generation.build_plan(prompts(), voices()))
    write_new(here / 'catalog.json', {'entries': [dict(r, phones=[1, 2]) for r in prompts()]})
    write_new(protocol.parent / 'data/sources/qwen-synthetic-completion.source.json', {'files': []})
    write_new(here / 'protocol.json', cfg)
    anchors = []
    for i, voice in enumerate(voices()):
        native = cache / 'anchors' / (voice['id'] + '.wav')
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_bytes(voice['id'].encode())
        anchors.append(dict(voice, pcm_sha256=f'{i + 1:064x}', native_audio=str(native),
                            native_sha256=file_digest(native)))
    write_new(cache / 'anchors.json', anchors)
    recover(interrupted)
    before = failed.read_bytes()
    seeds = []

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def create_voice_clone_prompt(self, **kwargs):
            return [object()]

        def generate_voice_clone(self, **kwargs):
            return [np.zeros(24000) for _ in kwargs['text']], 24000

    cuda = SimpleNamespace(is_available=lambda: True, set_per_process_memory_fraction=lambda _: None,
                           empty_cache=lambda: None, get_device_name=lambda _: 'mock GPU')
    torch = SimpleNamespace(cuda=cuda, set_num_threads=lambda _: None, manual_seed=seeds.append,
                            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())),
                            bfloat16=object(), __version__='mock')

    def store(waveform, rate, *, native, audio, phone_count):
        digest = generation._hash(str(native))
        return dict(audio=str(audio), native_audio=str(native), sha256=digest, pcm_sha256=digest)

    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'qwen_tts', SimpleNamespace(Qwen3TTSModel=Model))
    monkeypatch.setattr(generation, 'HERE', here)
    monkeypatch.setattr(generation, 'CACHE', cache)
    monkeypatch.setattr(generation, 'freeze', lambda: cfg)
    monkeypatch.setattr(generation, 'read_audio', lambda *args: np.zeros(24000))
    monkeypatch.setattr(generation, '_store_waveform', store)
    result = generation.generate()
    assert result['complete'] is True and result['present'] == 6
    assert seeds[0] == int(generation._hash(['recovery-fixture', 'train-1', 0, 1])[:8], 16)
    assert failed.read_bytes() == before
