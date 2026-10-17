/** Browser playback is elected by the server. Historical snapshots never call handle. */
export function createLocalPlayer(audio) {
    const seen = new Set();
    return {
        async handle(execution, owner) {
            if (!owner || execution?.status !== 'requested' || !execution.local_media || !execution.id || seen.has(execution.id)) return;
            seen.add(execution.id);
            const data = execution.local_media, slots = data.slots || {};
            try {
                if (data.action === 'PLAY_MUSIC' || data.action === 'NEXT') {
                    if (!/^\/api\/media\/local\/[a-f0-9]{32}$/.test(data.url)) throw new Error('Invalid local track');
                    const current = audio.getAttribute ? audio.getAttribute('src') : audio.src;
                    if (current !== data.url) audio.src = data.url;
                    await audio.play();
                } else if (data.action === 'PAUSE' || data.action === 'STOP') {
                    audio.pause();
                    if (data.action === 'STOP') audio.currentTime = 0;
                } else if (data.action === 'VOLUME') {
                    if (slots.operation === 'mute') audio.muted = true;
                    else {
                        const volume = slots.operation === 'set' ? slots.percent / 100 : audio.volume + (slots.operation === 'up' ? .1 : -.1);
                        if (!Number.isFinite(volume)) throw new Error('Invalid volume');
                        audio.volume = Math.max(0, Math.min(1, volume)); audio.muted = false;
                    }
                } else throw new Error('Unsupported local action');
                return { id: execution.id, status: 'succeeded' };
            } catch { return { id: execution.id, status: 'failed' }; }
        },
        close() { audio.pause?.(); audio.removeAttribute?.('src'); audio.load?.(); },
    };
}
