/** Fullscreen may require activation. Retry once on the first intentional tap, never trap Escape. */
export function installFullscreen(element) {
    if (new URLSearchParams(location.search).get('fullscreen') === '0' || !document.fullscreenEnabled)
        return () => { };
    let disposed = false;
    async function request() { if (disposed || document.fullscreenElement)
        return; try {
        await element.requestFullscreen({ navigationUI: 'hide' });
        element.dataset.fullscreen = 'active';
    }
    catch {
        if (!disposed)
            element.dataset.fullscreen = 'gesture-required';
    } }
    function activate() { void request(); }
    void request();
    element.addEventListener('pointerdown', activate, { once: true, capture: true });
    return () => { disposed = true; element.removeEventListener('pointerdown', activate, true); };
}
