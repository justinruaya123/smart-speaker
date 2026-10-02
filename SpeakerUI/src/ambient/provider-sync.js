/** Independent readers prevent a slow provider from delaying the other. */
export function connectProviderStates(handlers, fetcher, page) {
    let closed = false;
    const readers = [
        { provider: 'ha', path: '/api/home-assistant/entities', interval: 1500, pending: null, timer: null, held: false, refresh: false, fastReads: 0 },
        { provider: 'spotify', path: '/api/spotify/state', interval: 5000, pending: null, timer: null, held: false, refresh: false, fastReads: 0 },
    ];
    async function poll(reader) {
        if (closed || page.hidden || reader.pending || reader.held) return;
        reader.refresh = false;
        const controller = new AbortController();
        reader.pending = controller;
        try {
            const response = await fetcher(reader.path, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(8000)]) });
            if (!response.ok) throw new Error('Provider state unavailable');
            const value = await response.json();
            if (!closed && !controller.signal.aborted) handlers[reader.provider](value);
        } catch {
            if (!closed && !controller.signal.aborted) handlers.failed(reader.provider);
        } finally {
            reader.pending = null;
            if (!closed && !page.hidden && !reader.held) {
                const interval = reader.refresh ? 0 : reader.fastReads > 0 ? 500 : reader.interval;
                if (!reader.refresh && reader.fastReads > 0) reader.fastReads--;
                reader.timer = setTimeout(() => poll(reader), interval);
            }
        }
    }
    function visibility() {
        for (const reader of readers) {
            clearTimeout(reader.timer);
            if (page.hidden) { reader.fastReads = 0; reader.pending?.abort(); }
            else void poll(reader);
        }
    }
    page.addEventListener('visibilitychange', visibility);
    for (const reader of readers) void poll(reader);
    return { hold(provider) {
        const reader = readers.find(reader => reader.provider === provider);
        if (!reader) throw new Error('Unknown provider');
        reader.held = true;
        clearTimeout(reader.timer);
        reader.pending?.abort();
        return () => {
            reader.held = false;
            reader.refresh = true;
            reader.fastReads = provider === 'spotify' ? 6 : 0;
            void poll(reader);
        };
    }, close() {
        closed = true;
        page.removeEventListener('visibilitychange', visibility);
        for (const reader of readers) { clearTimeout(reader.timer); reader.pending?.abort(); }
    } };
}
