<script>
  import { onMount } from 'svelte';
  let { screen } = $props();
  let m = $derived(screen.model);
  let devices = $state([]), selected = $state(''), error = $state(''), busy = $state(false), wake = $state(false);
  let expanded = $state(true), listed = $state(false);
  async function request(path, body, type = 'application/json') {
    error = ''; busy = true;
    try {
      const response = await fetch('/api/runtime/' + path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': type }, body: type === 'audio/wav' ? body : JSON.stringify(body) });
      const value = await response.json();
      if (!response.ok) throw new Error(value.message || value.error || 'Runtime request failed');
      return value;
    } catch (failure) { error = failure.message; }
    finally { busy = false; }
  }
  async function refresh() { const value = await request('devices'); if (value) {devices = value.devices;listed=true;} }
  onMount(()=>{void refresh();});
  async function upload(event) {
    const file = event.target.files?.[0];
    if (file) await request('wav?wake=' + wake, await file.arrayBuffer(), 'audio/wav');
    event.target.value = '';
  }
</script>

{#if m.runtimeMode}
  <aside class="runtime-debug" aria-label="Inference debug">
    <p>Inference · {m.runtimeState || 'starting'}</p>
    {#if expanded}
      <div class="panel">
        <p><strong>{m.runtimeEngine || 'Loading'} · Actions {m.runtimeActions?'enabled':'disabled'}</strong><br/>
          {m.runtimeConfig?.mode || 'demo'} · Intent ≥ {((m.runtimeConfig?.intent_threshold??.95)*100).toFixed(0)}% · Each slot ≥ {((m.runtimeConfig?.slot_threshold??.9)*100).toFixed(0)}%</p>
        <div class="row"><button disabled={busy} onclick={refresh}>Refresh microphones</button>
          <select aria-label="Host microphone" bind:value={selected}><option value="">Choose microphone</option>{#each devices as device (device.index)}<option value={String(device.index)}>{device.index}: {device.name} ({device.hostapi})</option>{/each}</select>
          <button disabled={busy || selected === ''} onclick={() => request('listen', { device: Number(selected) })}>Listen</button>
          <button disabled={busy} onclick={() => request('mute', {})}>Mute</button></div>
        {#if listed && !devices.length}<p>No input devices found on this host. Connect a microphone to the Pi and refresh.</p>{/if}
        <label class="row"><input type="checkbox" bind:checked={wake}/> WAV includes the wake phrase and a command</label>
        <label class="row">Replay WAV (mono, 16 kHz)<input aria-label="Replay WAV" type="file" accept=".wav,audio/wav" disabled={busy} onchange={upload}/></label>
        <p role="status">{busy ? 'Processing…' : error || m.runtimeError || 'Say the wake phrase, pause until command ready, then speak.'}</p>
        {#if m.runtimeResult?.output}<p data-command-summary><strong>{m.runtimeResult.application?.accepted ? 'Application: '+m.runtimeResult.application.command.intent : 'Application rejected: '+(m.runtimeResult.application?.reason||m.runtimeResult.output.reason)}</strong>{#if m.runtimeResult.diagnostic_candidate?.candidate}<br/>Model policy: {m.runtimeResult.output.accepted?'accepted':m.runtimeResult.output.reason}<br/>Candidate: {m.runtimeResult.diagnostic_candidate.candidate.intent} · {JSON.stringify(m.runtimeResult.diagnostic_candidate.candidate.slots)}{/if}</p>{/if}
        <pre data-runtime-result>{JSON.stringify(m.runtimeResult || { state: m.runtimeState, actions_enabled: m.runtimeActions }, null, 2)}</pre>
      </div>
    {/if}
  </aside>
{/if}

<style>
  .runtime-debug { font-size:12px; max-width:100%; color:#d9e5ea; }
  button,select,input { font:inherit; }
  .panel { max-width:100%; overflow:auto; padding:12px; border-radius:12px; background:#0c171ef5; border:1px solid #35505a; }
  .row { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:8px 0; } select { max-width:285px; }
  button,select { padding:6px; background:#23343d; color:inherit; border:1px solid #49616a; border-radius:5px; } button:disabled { opacity:.5; }
  p { margin:6px 0; } pre { font-size:11px; white-space:pre-wrap; overflow-wrap:anywhere; }
</style>
