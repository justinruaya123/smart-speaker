<script>
  import Icon from './Icon.svelte';
  import RuntimeDebug from './RuntimeDebug.svelte';
  let {screen}=$props();let m=$derived(screen.model);let tab=$state('samples');
</script>
<nav aria-label="Debug options" class="debug-options">
  <button onclick={()=>tab='samples'} aria-pressed={tab==='samples'}>Sample commands</button>
  <button onclick={()=>tab='aura'} aria-pressed={tab==='aura'}>Aura states</button>
  {#if m.runtimeMode}<button onclick={()=>tab='inference'} aria-pressed={tab==='inference'}>Inference</button>{/if}
</nav>
{#if tab==='inference'}<RuntimeDebug {screen}/>
{:else if tab==='aura'}<div class="demo-state-row">{#each Object.keys(screen.AURA_STATES) as state (state)}<button data-agent={state}>{state}</button>{/each}</div>
{:else}<div class="demo-grid">{#each screen.DEMO as [label,symbol,command],i (label)}<button data-demo={i}><Icon name={symbol}/>{label}{#if screen.liveShortcut(command)}<small class="shortcut-live">Live</small>{/if}</button>{/each}</div><p class="demo-note">Sample fixtures are simulated. · {m.backend}</p>{/if}
<style>.debug-options{display:flex;gap:10px;margin:0 0 14px;}.debug-options button{font:inherit;color:#d9e5ea;background:#18272f;border:1px solid #35505a;border-radius:10px;padding:9px 14px;}.debug-options button[aria-pressed=true]{background:#304650;}</style>
