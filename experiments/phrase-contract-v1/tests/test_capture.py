import threading
import numpy as np


class RuntimeProbe:
    """Capture transport probe; never acoustic evidence."""
    def __init__(self):
        self.calls = []; self.finalized = False
        self.endpoint = type('Status', (), {'fault': None})()

    def reset(self):
        self.calls.clear(); self.finalized = False

    def feed(self, pcm, sequence, *, overflow=False):
        self.calls.append((pcm.copy(), sequence, overflow, threading.get_ident()))
        if overflow:
            self.endpoint.fault = 'capture_overflow'
        return {'accepted': False, 'command': None, 'reason': 'capture_fault' if overflow else 'tentative'}

    def eof(self):
        return {'accepted': False, 'command': None, 'reason': 'incomplete'}


def test_callback_only_queues_owned_pcm_consumer_calls_same_runtime():
    from pcvcm.capture import CaptureAdapter
    runtime = RuntimeProbe(); adapter = CaptureAdapter(runtime, capacity=4)
    pcm = np.full((640, 1), .2, dtype='float32')
    adapter.callback(pcm, 640, None, None)
    adapter.callback(pcm, 640, None, None)
    assert runtime.calls == []
    pcm[:] = 0
    adapter.start(); adapter.stop()
    assert [c[1] for c in runtime.calls] == [0, 640]
    assert all(c[3] != threading.get_ident() for c in runtime.calls)
    assert np.all(runtime.calls[0][0] == np.float32(.2))
    assert adapter.result['reason'] == 'incomplete'


def test_full_queue_fault_survives_drop_and_never_accepts():
    from pcvcm.capture import CaptureAdapter
    runtime = RuntimeProbe(); adapter = CaptureAdapter(runtime, capacity=1)
    pcm = np.zeros((640, 1), dtype='float32')
    adapter.callback(pcm, 640, None, None)
    adapter.callback(pcm, 640, None, None)
    assert adapter.next_sample == 1280
    adapter.start(); adapter.stop()
    assert adapter.result['accepted'] is False
    assert adapter.result['reason'] == 'capture_fault'
    assert adapter.metrics['queue_overruns'] == 1
    assert runtime.calls[-1][2] is True


def test_portaudio_status_and_callback_shape_fault_are_sticky():
    from pcvcm.capture import CaptureAdapter
    for bad, status in [(np.zeros((640, 2), dtype='float32'), None),
                        (np.zeros((640, 1), dtype='float32'), 'input overflow')]:
        runtime = RuntimeProbe(); adapter = CaptureAdapter(runtime)
        adapter.callback(bad, 640, None, status)
        adapter.start(); adapter.stop()
        assert adapter.result['reason'] == 'capture_fault'


def test_fault_arriving_during_inference_invalidates_pending_acceptance():
    from pcvcm.capture import CaptureAdapter
    entered, release = threading.Event(), threading.Event()
    runtime = RuntimeProbe()
    def feed(pcm, sequence, *, overflow=False):
        if not overflow:
            entered.set(); release.wait(2)
            runtime.finalized = True
            return {'accepted': True, 'command': {'intent': 'TEST'}, 'reason': 'accepted'}
        return {'accepted': False, 'command': None, 'reason': 'capture_fault'}
    runtime.feed = feed
    adapter = CaptureAdapter(runtime, capacity=1); adapter.start()
    pcm = np.zeros((640, 1), dtype='float32')
    adapter.callback(pcm, 640, None, None)
    assert entered.wait(2)
    adapter.callback(pcm, 640, None, 'dropout')
    release.set(); adapter.stop()
    assert adapter.result['accepted'] is False
