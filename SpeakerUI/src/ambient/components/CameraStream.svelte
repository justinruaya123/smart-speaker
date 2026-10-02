<script>
  import { onMount } from 'svelte';

  let { entityId, name, available } = $props();
  let visible = $state(false);
  let failed = $state(false);

  onMount(() => {
    const updateVisibility = () => { visible = document.visibilityState === 'visible'; };
    updateVisibility();
    document.addEventListener('visibilitychange', updateVisibility);
    return () => document.removeEventListener('visibilitychange', updateVisibility);
  });

  function stream(node) {
    const lifetime = new AbortController(), session = crypto.randomUUID();
    const path = `/api/home-assistant/camera/${encodeURIComponent(entityId)}`;
    let timer;
    async function checkConnection() {
      try {
        const response = await fetch(`${path}/stream-status/${session}`, { signal: AbortSignal.any([lifetime.signal, AbortSignal.timeout(5000)]) });
        if (!response.ok || (await response.json()).active !== true) throw new Error('Stream ended');
        if (!lifetime.signal.aborted) timer = setTimeout(checkConnection, 1500);
      } catch { if (!lifetime.signal.aborted) failed = true; }
    }
    // Image load/error events alone do not report an MJPEG disconnection reliably.
    const loaded = () => { timer = setTimeout(checkConnection, 1500); };
    node.addEventListener('load', loaded, { once: true });
    node.src = `${path}/stream?session=${session}`;
    // Removing a streaming image from the DOM alone can leave its request open.
    return () => { lifetime.abort(); clearTimeout(timer); node.removeEventListener('load', loaded); node.removeAttribute('src'); };
  }
</script>

{#if !available}
  <div class="camera-status" role="status">Camera unavailable</div>
{:else if failed}
  <div class="camera-status" role="status"><span>Camera unavailable</span><button class="button-secondary" onclick={() => { failed = false; }}>Retry stream</button></div>
{:else if visible}
  <img class="camera-stream" alt={`${name} live video`} {@attach stream} onerror={() => { failed = true; }} />
{:else}
  <div class="camera-status">Video paused</div>
{/if}
