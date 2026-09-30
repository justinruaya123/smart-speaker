"""Bounded push-to-talk PCM transport; recognition runs only on its consumer."""
from queue import Queue, Empty, Full
import threading
import numpy as np


class CaptureAdapter:
    def __init__(self, runtime, capacity=32, block_samples=640, on_result=None):
        if not 1 <= capacity <= 64 or block_samples != 640:
            raise ValueError('Capture uses bounded 40 ms PCM blocks')
        self.runtime, self.on_result = runtime, on_result
        self.block_samples = block_samples
        self.queue = Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.stopping, self.done = threading.Event(), threading.Event()
        self.next_sample, self.fault, self.result = 0, None, None
        self.metrics = {'queue_peak': 0, 'queue_overruns': 0, 'callback_faults': 0}
        self.thread = threading.Thread(target=self._consume, name='pcv1-consumer', daemon=True)

    def callback(self, indata, frames, time_info, status):
        # No frontend/model/decoder/UI calls here. PortAudio owns indata: copy it.
        with self.lock:
            if self.done.is_set() or self.stopping.is_set():
                return
            sequence = self.next_sample
            self.next_sample += frames
            if status or frames != self.block_samples or indata.shape != (frames, 1):
                self.fault = 'capture_dropout'
                self.metrics['callback_faults'] += 1
                return
            try:
                self.queue.put_nowait((sequence, indata[:, 0].copy()))
                self.metrics['queue_peak'] = max(self.metrics['queue_peak'], self.queue.qsize())
            except Full:
                self.metrics['queue_overruns'] += 1
                self.fault = 'capture_overflow'

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Consumer failed to stop; capture is unqualified')
        return self.result

    def _consume(self):
        sequence = 0
        self.runtime.reset()
        try:
            while True:
                with self.lock:
                    fault = self.fault
                if fault:
                    self.runtime.feed(np.empty(0, dtype='float32'), sequence, overflow=True)
                    self.result = {'schema_version': 2, 'accepted': False,
                                   'command': None, 'reason': 'capture_fault'}
                    break
                try:
                    start, pcm = self.queue.get(timeout=.02)
                except Empty:
                    if self.stopping.is_set():
                        self.result = self.runtime.eof()
                        break
                    continue
                result = self.runtime.feed(pcm, start)
                sequence = start + len(pcm)
                with self.lock:
                    if self.fault:
                        continue
                    if self.runtime.finalized or self.runtime.endpoint.fault:
                        self.result = result
                        self.done.set()  # Close this utterance before publishing.
                        break
            if self.on_result:
                self.on_result(self.result)
        except Exception as exc:
            self.fault = f'consumer_error: {type(exc).__name__}: {exc}'
            self.result = {'schema_version': 2, 'accepted': False,
                           'command': None, 'reason': 'capture_fault'}
        finally:
            self.done.set()


def microphone(runtime, device, on_result=None):
    """One explicitly armed utterance, bounded by the runtime's 12 s endpoint."""
    import sounddevice as sd
    adapter = CaptureAdapter(runtime, on_result=on_result)
    adapter.start()
    try:
        with sd.InputStream(device=device, samplerate=16000, channels=1,
                            dtype='float32', blocksize=640, callback=adapter.callback):
            if not adapter.done.wait(timeout=14):
                with adapter.lock:
                    adapter.fault = 'capture_timeout'
    finally:
        adapter.stop()
    return {'decision': adapter.result, 'capture': adapter.metrics, 'fault': adapter.fault}
