<script>
  import Icon from './Icon.svelte';
  import CameraStream from './CameraStream.svelte';
  let { screen } = $props();
  let m = $derived(screen.model);
  let live = $derived(m.call?.kind === 'camera');
  let centered = $derived(m.view === 'call');
  let camera = $derived(live ? m.cameras.find(camera => camera.id === m.call.entityId) : null);
</script>

{#if m.call}
  <div class="call-window" class:call-centered={centered} class:call-floating={!centered} role="group" aria-label={centered?'Camera call':'Ongoing call preview'} inert={!!m.alert}>
    {#if m.cameraEnabled}
      {#if live}
        {#key m.call.entityId}<CameraStream entityId={m.call.entityId} name={camera?.name || m.call.name} available={camera?.available === true} />{/key}
      {:else}
        <canvas class="camera-canvas" width="640" height="360" aria-label="Simulated camera call"></canvas>
      {/if}
    {:else}
      <div class="camera-status" role="status"><Icon name="video-off" /><span>Camera paused</span></div>
    {/if}
    <div class="call-caption"><strong title={camera?.name || m.call.name}>{camera?.name || m.call.name}</strong><span>{live?'Video only':'Simulated call'} · {screen.duration((m.now-m.call.started)/1000)}</span></div>
    {#if centered}
      <button class="camera-minimize" data-action="camera-collapse" aria-label="Minimize call"><Icon name="minimize-2" /></button>
      <div class="call-controls">
        {#if !live}<button data-action="call-mute" aria-label="Mute call" aria-pressed={m.call.muted}><Icon name={m.call.muted?'mic-off':'mic'} /></button>{/if}
        <button data-action="call-camera" aria-label={m.cameraEnabled?'Disable camera':'Enable camera'} aria-pressed={m.cameraEnabled}><Icon name={m.cameraEnabled?'video':'video-off'} /></button>
        {#if !live}<button data-action="call-speaker" aria-label="Speaker" aria-pressed={m.call.speaker}><Icon name="volume-2" /></button>{/if}
        <button class="hangup" data-action="hangup" aria-label="End call"><Icon name="phone-off" /></button>
      </div>
    {:else}
      <button class="camera-restore" data-action="resume-call" aria-label="Restore call"><Icon name="maximize-2" /></button>
      <button class="floating-hangup" data-action="hangup" aria-label="End call"><Icon name="phone-off" /></button>
    {/if}
  </div>
{/if}
