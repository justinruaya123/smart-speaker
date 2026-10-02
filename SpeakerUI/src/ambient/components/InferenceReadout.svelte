<script>
  let {screen}=$props();let m=$derived(screen.model);
  let event=$derived(m.runtimeResult);
  let candidate=$derived(event?.application?.command||event?.output?.command||event?.diagnostic_candidate?.candidate);
  let intent=$derived(event?.application?.intent_confidence??event?.diagnostic_candidate?.intent_probability);
  let slots=$derived(event?.application?.slot_confidence||event?.diagnostic_candidate?.slot_confidence||{});
  const percent=value=>typeof value==='number'?(value*100).toFixed(1)+'%':'—';
</script>
{#if m.runtimeMode && m.runtimeUI?.readout !== false}
<button class="inference-readout" class:raised={!!m.call} data-open="intent" aria-label="Inspect inference history">
  <span>{candidate?.intent || m.runtimeState || 'Awaiting microphone'} {candidate?JSON.stringify(candidate.slots):''}</span>
  <small>Intent {percent(intent)} · Slots {Object.keys(slots).length?Object.entries(slots).map(([key,value])=>key+' '+percent(value)).join(' · '):'—'}</small>
  <small>{m.event?.execution?.detail || m.event?.status || m.runtimeError || m.runtimeEngine}</small>
</button>
{/if}
<style>.inference-readout{position:absolute;left:70px;bottom:22px;max-width:46%;display:flex;flex-direction:column;gap:3px;text-align:left;background:none;border:0;color:#bdcdd4;font:12px/1.35 system-ui;padding:5px;}.inference-readout span{font-family:monospace;overflow-wrap:anywhere;}.inference-readout small{font-size:10px;opacity:.75;overflow-wrap:anywhere;}.inference-readout.raised{bottom:94px;}</style>
