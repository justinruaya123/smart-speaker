<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { AUDIO, isRecord, MediaSource, type Track } from '../../shared/protocol.js';
  import { startCapture, type Capture } from '../audio/capture.js';

  let connection = $state('disconnected');
  let phase = $state<'idle' | 'starting' | 'recording' | 'processing'>('idle');
  let events = $state<string[]>([]);
  let query = $state('');
  let tracks = $state<Track[]>([]);
  let searchMessage = $state('');
  let searching = $state(false);
  let spotifyEnabled = $state(false);
  let source = $state<string | undefined>();
  let player: HTMLAudioElement | undefined;
  let socket: WebSocket | undefined;
  let disposed = false;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let retryCount = 0;
  let operation: { id: string; abort: AbortController; capture?: Capture; sending: boolean; ending: boolean; timer?: ReturnType<typeof setTimeout> } | undefined;

  function log(value: unknown) { events = [JSON.stringify(value).slice(0, 2_000), ...events].slice(0, 50); }
  function attachPlayer(element: HTMLAudioElement) {
    player = element;
    return () => { element.pause(); if (player === element) player = undefined; };
  }
  function control(type: 'start' | 'end' | 'cancel', id: string) {
    if (socket?.readyState !== WebSocket.OPEN) throw new Error('Gateway is disconnected.');
    socket.send(JSON.stringify({ type, utteranceId: id }));
  }
  function cleanup() {
    const current = operation; operation = undefined;
    if (current) { clearTimeout(current.timer); current.abort.abort(); current.capture?.close(); }
    phase = 'idle';
  }
  function cancel(message = 'Capture cancelled.') {
    if (operation && socket?.readyState === WebSocket.OPEN) control('cancel', operation.id);
    cleanup(); log({ type: 'capture', message });
  }
  function connect() {
    if (disposed) return;
    connection = 'connecting';
    const url = new URL('/ws', location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const current = new WebSocket(url); socket = current;
    current.onmessage = ({ data }: MessageEvent<unknown>) => {
      if (socket !== current || typeof data !== 'string') return;
      let event: unknown;
      try { event = JSON.parse(data); } catch { log({ error: 'Invalid server event.' }); return; }
      if (!isRecord(event) || typeof event.type !== 'string') return;
      log(event);
      if (event.type === 'ready') { connection = 'connected (mock)'; retryCount = 0; }
      if (event.utteranceId !== operation?.id) return;
      if (event.type === 'state' && event.state === 'recording' && operation) {
        phase = 'recording'; operation.sending = true;
        clearTimeout(operation.timer);
        operation.timer = setTimeout(() => { void end(); }, AUDIO.maxDurationMs - 500);
      }
      if (event.type === 'state' && event.state === 'processing') phase = 'processing';
      if ((event.type === 'state' && event.state === 'idle') || event.type === 'error') cleanup();
    };
    current.onclose = () => {
      if (socket !== current) return;
      cleanup(); connection = 'disconnected';
      if (!disposed) retry = setTimeout(connect, Math.min(10_000, 500 * 2 ** Math.min(retryCount++, 5)));
    };
    current.onerror = () => { connection = 'connection error'; current.close(); };
  }
  async function start() {
    if (phase !== 'idle' || socket?.readyState !== WebSocket.OPEN) return;
    player?.pause();
    phase = 'starting';
    const current: NonNullable<typeof operation> = { id: crypto.randomUUID(), abort: new AbortController(), sending: false, ending: false };
    operation = current;
    try {
      const capture = await startCapture(current.abort.signal, {
        frame(buffer) {
          if (operation !== current || !current.sending) return;
          if (socket?.readyState !== WebSocket.OPEN || socket.bufferedAmount + buffer.byteLength > AUDIO.maxQueuedBytes) { cancel('Audio transport queue exceeded one second.'); return; }
          socket.send(buffer);
        },
        failure(message) { if (operation === current) cancel(message); },
      });
      if (operation !== current) { capture.close(); return; }
      current.capture = capture;
      control('start', current.id);
      current.timer = setTimeout(() => { if (operation === current) cancel('Gateway did not acknowledge capture.'); }, 3_000);
    } catch (error) {
      if (operation !== current) return;
      cleanup(); log({ type: 'capture_error', message: error instanceof Error ? error.message : 'Microphone permission or setup failed.' });
    }
  }
  async function end() {
    const current = operation;
    if (!current?.capture || !current.sending || current.ending) return;
    current.ending = true; phase = 'processing'; clearTimeout(current.timer);
    try {
      await current.capture.flush();
      if (operation !== current) return;
      current.sending = false; current.capture.close(); control('end', current.id);
    } catch (error) { if (operation === current) cancel(error instanceof Error ? error.message : 'Capture could not finish.'); }
  }
  function readTracks(value: unknown): Track[] {
    if (!Array.isArray(value)) return [];
    const results: Track[] = [];
    for (const item of value.slice(0,5)) {
      if (!isRecord(item) || (item.source !== MediaSource.Local && item.source !== MediaSource.Spotify) || typeof item.id !== 'string' || typeof item.title !== 'string') continue;
      const pattern = item.source === MediaSource.Local ? /^[a-f0-9]{32}$/ : /^[A-Za-z0-9]{22}$/;
      if (!pattern.test(item.id)) continue;
      results.push({ source: item.source, id: item.id, title: item.title, ...(typeof item.artist === 'string' ? { artist: item.artist } : {}) });
    }
    return results;
  }
  async function search(event: SubmitEvent) {
    event.preventDefault();
    if (!query.trim() || searching) return;
    searching = true;
    try {
      const response = await fetch(`/api/media/search?q=${encodeURIComponent(query.trim())}`);
      const result: unknown = await response.json();
      if (!response.ok || !isRecord(result)) throw new Error('Media search failed.');
      tracks = readTracks(result.tracks);
      searchMessage = typeof result.fallbackReason === 'string' ? result.fallbackReason : 'Spotify results';
      if (!tracks.length) searchMessage += ' No matches.';
    } catch (error) { searchMessage = error instanceof Error ? error.message : 'Search failed.'; }
    finally { searching = false; }
  }
  async function play(track: Track) {
    if (phase !== 'idle') return;
    if (track.source === MediaSource.Local) {
      const url = `/api/media/local/${track.id}`;
      source = url;
      if (player) { player.src = url; try { await player.play(); } catch { log({ message: 'Use the audio controls to start playback.' }); } }
      return;
    }
    try {
      const response = await fetch('/api/media/play', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: track.source, id: track.id }) });
      const result: unknown = await response.json(); log(result);
      if (isRecord(result) && isRecord(result.data)) tracks = readTracks(result.data.tracks);
    } catch { log({ error: 'Playback request failed; outcome is unknown. It was not retried.' }); }
  }
  async function authorizeSpotify() {
    try {
      const response = await fetch('/auth/spotify', { method: 'POST' });
      const result: unknown = await response.json();
      if (!response.ok || !isRecord(result) || typeof result.url !== 'string' || !result.url.startsWith('https://accounts.spotify.com/authorize?')) { log(result); return; }
      location.assign(result.url);
    } catch { log({ error: 'Spotify authorization could not start.' }); }
  }
  onMount(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'auto';
    connect();
    void fetch('/api/health').then((response) => response.json()).then((health: unknown) => { log(health); spotifyEnabled = isRecord(health) && health.spotify === 'configured'; }).catch(() => log({ error: 'Health endpoint unavailable.' }));
    const hidden = () => { if (document.hidden && operation) cancel('Page hidden; capture cancelled.'); };
    document.addEventListener('visibilitychange', hidden);
    return () => { document.body.style.overflow = previousOverflow; disposed = true; clearTimeout(retry); cleanup(); socket?.close(); player?.pause(); document.removeEventListener('visibilitychange', hidden); };
  });
</script>

<main>
  <h1>Audio diagnostics</h1>
  <p><strong>MOCK audio transport:</strong> every nonempty recording returns a fixed “demo” media search. This audio path does not recognize speech or execute actions. Use Iris microphone settings for native worker diagnostics.</p>
  <p>Media search and playback below are separate manual controls for your configured providers. <a href="/">Return to Iris</a>.</p>
  <p aria-live="polite">Connection: {connection}. Capture: {phase}.</p>
  <button onclick={start} disabled={phase !== 'idle' || connection !== 'connected (mock)'}>Start microphone</button>
  <button onclick={end} disabled={phase !== 'recording'}>Stop and process</button>
  <button onclick={() => cancel()} disabled={phase === 'idle'}>Cancel</button>
  <p>Mono PCM16, 16 kHz. Recording stops before the 15-second limit. Audio is not saved.</p>

  <form onsubmit={search}>
    <label for="search">Media query </label><input id="search" bind:value={query} maxlength="200" />
    <button disabled={searching}>{searching ? 'Searching…' : 'Search media'}</button>
  </form>
  <p aria-live="polite">{searchMessage}</p>
  <ul>{#each tracks as track (`${track.source}:${track.id}`)}<li><button onclick={() => play(track)} disabled={phase !== 'idle'}>Play {track.title}</button> ({track.source}){track.artist ? ` — ${track.artist}` : ''}</li>{/each}</ul>
  <audio {@attach attachPlayer} src={source} controls aria-label="Local MP3 playback" onplay={(event) => { if (phase !== 'idle') event.currentTarget.pause(); }}></audio>
  {#if spotifyEnabled}<p><button onclick={authorizeSpotify}>Connect Spotify</button></p>{/if}
  <p>Recent events (maximum 50): <button onclick={() => { events = []; }}>Clear</button></p>
  <pre aria-label="Diagnostic events">{events.join('\n\n')}</pre>
</main>
