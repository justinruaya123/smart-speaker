<script>
  import { createAuraRenderer } from './aura-renderer.js';

  /** @type {{state?: keyof typeof import('./aura-renderer.js').AURA_STATES, volume?: number, muted?: boolean, label?: string}} */
  let { state: agentState = 'idle', volume = 0, muted = false, label = 'Assistant state' } = $props();
  let failed = $state(false);

  function attachAura(canvas) {
    const renderer = createAuraRenderer(canvas, { onFailure: () => { failed = true; } });
    const cleanup = $effect.root(() => {
      $effect(() => renderer.update({ state: agentState, volume, muted }));
    });
    return () => { cleanup(); renderer.destroy(); };
  }
</script>

<div class="aura" class:failed class:muted role="img" aria-label={`${label}: ${muted ? 'muted' : agentState}`}>
  <canvas {@attach attachAura} aria-hidden="true"></canvas>
  {#if failed}<span class="fallback" aria-hidden="true"></span>{/if}
</div>

<style>
  .aura { width:100%;height:100%;position:relative;pointer-events:none; }
  canvas { width:100%;height:100%;display:block; }
  .failed canvas { visibility:hidden; }
  .fallback { position:absolute;inset:24%;border:1px solid #73acbd;border-radius:50%;box-shadow:0 0 16px #73acbd44; }
  .muted .fallback { opacity:.35; }
</style>
