import { connectRuntime } from './runtime-events.js';
/** Simulator is available only when the gateway explicitly reports mock mode. */
export function connectDummyBackend(model, dispatch, setAgent) {
    let runtime;
    const controller = new AbortController();
    const signal = () => AbortSignal.any([controller.signal, AbortSignal.timeout(3000)]);
    void fetch('/api/health', { signal: signal() }).then(response => { if (!response.ok)
        throw new Error('Unavailable'); return response.json(); }).then(health => { if (controller.signal.aborted)
        return;
        if (health.inference === 'onnx') { model.runtimeMode = true; runtime = connectRuntime(model, setAgent); model.backend = 'Connecting to CPU ONNX'; }
        else model.backend = health.runtime?.loaded && health.runtime.engine === 'dummy-onnx' ? 'Dummy backend ready' : 'Backend unavailable';
    }).catch(() => { if (!controller.signal.aborted) model.backend = 'Backend unavailable'; });
    return {
        async infer(index, fallback) {
            if (controller.signal.aborted)
                return;
            if (model.runtimeMode || model.backend !== 'Dummy backend ready') return;
            try {
                const response = await fetch('/api/ambient/infer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fixture: index }), signal: signal() });
                if (!response.ok)
                    throw new Error('Inference unavailable');
                const payload = await response.json();
                if (payload.simulated !== true || payload.engine !== 'dummy-onnx' || typeof payload.result?.intent !== 'string')
                    throw new Error('Unexpected fixture response');
                if (!controller.signal.aborted)
                    dispatch(payload.result);
            }
            catch {
                if (!controller.signal.aborted) {
                    model.backend = 'Backend unavailable';
                }
            }
        },
        close() { controller.abort(); runtime?.close(); },
    };
}
