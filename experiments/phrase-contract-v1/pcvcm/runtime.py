"""One streaming recognition path. No transcript or external action interface."""
from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from .contract import Contract
from .decoder import PrefixCTC, rank_commands
from .endpoint import Endpoint, EndpointConfig
from .policy import decide


class StreamEngine:
    """Frontend, exact decoder, real endpoint, and final typed decision.

    The encoder injection exists for explicit software tests. Production uses
    Runtime.from_bundle only; missing weights never select a test encoder.
    """

    def __init__(self, contract: Contract, catalog: dict,
                 encoder: Callable, normalization: dict,
                 endpoint: EndpointConfig, policy: dict, binding: str):
        from vcm.frontend import StreamingFrontend

        self.contract = contract
        self.entries = catalog['entries']
        for entry in self.entries:
            if entry['command'] is not None:
                contract.validate(entry['command'])
        self.encoder = encoder
        self.frontend = StreamingFrontend(normalization['mean'],
                                          normalization['std'])
        self.decoder = PrefixCTC([r['phones'] for r in self.entries])
        self.endpoint = Endpoint(endpoint)
        self.policy = policy
        self.binding = binding
        self.reset()

    def reset(self) -> None:
        self.frontend.reset()
        self.decoder.reset()
        self.endpoint.reset()
        if hasattr(self.encoder, 'reset'):
            self.encoder.reset()
        self.sequence = 0
        self.finalized = False
        self.candidate = {'command': None, 'score': None, 'margin': None}
        self.timing = dict(frontend_ms=0., encoder_ms=0., decoder_ms=0.,
                           endpoint_ms=0., gate_ms=0.)
        self.diagnostic = {}

    def feed(self, pcm: np.ndarray, sequence: int, *,
             overflow: bool = False) -> dict:
        if self.finalized:
            raise ValueError('Reset required after final endpoint')
        x = np.asarray(pcm, dtype=np.float32)
        started = time.perf_counter()
        status = self.endpoint.feed(x, sequence, overflow=overflow)
        self.timing['endpoint_ms'] += (time.perf_counter() - started) * 1000
        if status['fault']:
            self.diagnostic = {**self.candidate, 'endpoint_detected': False,
                               'capture_fault': status['fault'], **self.timing}
            return decide(self.candidate, self.policy, self.binding,
                          endpoint=False, fault=status['fault'])
        self.sequence += len(x)
        if status['ended']:
            x = x[:max(0, status['end_sample'] - sequence)]
        started = time.perf_counter()
        features = self.frontend.feed(x)
        self.timing['frontend_ms'] += (time.perf_counter() - started) * 1000
        for start in range(0, features.shape[1], 4):
            started = time.perf_counter()
            logits = self.encoder(features[:, start:start + 4])
            self.timing['encoder_ms'] += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            scores = self.decoder.feed(logits)
            self.timing['decoder_ms'] += (time.perf_counter() - started) * 1000
            self.candidate = rank_commands(scores, self.entries)
        self.finalized = status['ended']
        started = time.perf_counter()
        result = decide(self.candidate, self.policy, self.binding,
                        endpoint=self.finalized)
        self.timing['gate_ms'] += (time.perf_counter() - started) * 1000
        self.diagnostic = {**self.candidate, **self.timing,
                           'endpoint_detected': self.finalized,
                           'end_sample': status['end_sample'],
                           'capture_fault': None}
        return result

    def eof(self) -> dict:
        """An absent real endpoint is a positive-case failure, not acceptance."""
        if self.finalized:
            return decide(self.candidate, self.policy, self.binding,
                          endpoint=True)
        return {'schema_version': 2, 'accepted': False, 'command': None,
                'reason': 'capture_fault' if self.endpoint.fault else 'incomplete'}


class OnnxEncoder:
    """A single ONNX Runtime CPU session and six explicitly reset caches."""

    def __init__(self, path, state_shapes: list, threads: int = 1):
        import onnxruntime as ort

        if threads not in (1, 2, 4):
            raise ValueError('Use a predeclared 1/2/4-thread profile')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(
            str(path), options, providers=['CPUExecutionProvider'])
        expected = ['features'] + [f'state_{i}' for i in range(6)]
        if [x.name for x in self.session.get_inputs()] != expected:
            raise ValueError('Unexpected ONNX inputs')
        outputs = ['ctc'] + [f'next_state_{i}' for i in range(6)]
        if [x.name for x in self.session.get_outputs()] != outputs:
            raise ValueError('Unexpected ONNX outputs')
        self.shapes = state_shapes
        self.reset()

    def reset(self) -> None:
        self.states = {f'state_{i}': np.zeros(s, dtype=np.float32)
                       for i, s in enumerate(self.shapes)}

    def __call__(self, features: np.ndarray) -> np.ndarray:
        values = self.session.run(None, {'features': features[None],
                                         **self.states})
        if len(values) != 7 or any(not np.isfinite(x).all() for x in values):
            raise ValueError('Invalid ONNX outputs')
        for value, shape in zip(values[1:], self.shapes):
            if list(value.shape) != list(shape):
                raise ValueError('Invalid returned cache dimensions')
        self.states = {f'state_{i}': v for i, v in enumerate(values[1:])}
        return values[0][0]


class Runtime(StreamEngine):
    @classmethod
    def from_bundle(cls, path, contract: Contract, threads: int = 1):
        from .bundle import verify_bundle

        content = verify_bundle(path, contract)
        encoder = OnnxEncoder(content['model_path'], content['state_shapes'],
                              threads=threads)
        return cls(contract, content['catalog'], encoder,
                   content['normalization'], EndpointConfig(**content['endpoint']),
                   content['policy'], content['binding'])
