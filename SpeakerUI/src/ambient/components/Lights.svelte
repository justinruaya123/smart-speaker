<script>
  import Icon from './Icon.svelte';
  import {lightStyle} from '../view-utils.js';
  let {screen}=$props();
  let m=$derived(screen.model);
  let light=$derived(m.lights[m.selectedLight]);
  let readOnly=$derived(!screen.canControlLights());
</script>
<div class="lights-layout">
  <div class="light-list" aria-label="Light list">
    <p class="gesture-hint" role="status">{m.haMode==='demo'?'Swipe to dim':m.haLabel}</p>
    {#each m.lights as l,i (l.id)}
      <div class="light-row" class:selected={i===m.selectedLight} data-light-row={i} style={lightStyle(l,screen.colors)}>
        <button class="light-icon" data-light-toggle={i} disabled={readOnly||l.status==='Unavailable'} aria-label={`Toggle ${l.name}`} aria-pressed={l.on}><Icon name="lightbulb" /></button>
        <button class="light-name" data-light-select={i} aria-label={readOnly?`Inspect ${l.name}`:`Edit ${l.name}; double tap to toggle`}><strong>{l.name}</strong><small><b data-row-brightness>{m.haMode==='demo'?l.on?l.brightness+'%':'Off':l.status}</b> · {l.room}</small></button>
        <span class="brightness-fill" aria-hidden="true"></span>
      </div>
    {:else}<p class="subtle">{m.haMode==='loading'?'Loading lights…':'No lights available'}</p>{/each}
    {#if m.lights.length}<button class="subtle" data-action="all-off" disabled={readOnly}>All off</button>{/if}
  </div>
  {#if light}
    <div class="light-editor">
      <div class="light-editor-head"><h2>{light.name}</h2><button class="switch" data-action="light-toggle" disabled={readOnly||light.status==='Unavailable'} aria-label={`Toggle ${light.name}`} aria-pressed={light.on}><span><i></i></span></button></div>
      <div class="light-preview" style={lightStyle(light,screen.colors)}><Icon name="lightbulb" /><span class="subtle">{light.status==='Unavailable'?'Unavailable':light.on?light.color:'Off'}</span></div>
      <div class="label-line"><label for="brightness-v2">Brightness</label><strong data-brightness>{light.brightness===null?'—':light.brightness+'%'}</strong></div>
      <input id="brightness-v2" data-input="brightness" disabled={readOnly||light.status==='Unavailable'||!screen.canDim(light)} aria-label="Brightness" type="range" min="0" max="100" value={light.brightness??0} />
      <div class="label-line"><span>Color</span><span>{light.color}</span></div>
      <div class="color-choices">{#each Object.entries(screen.colors).filter(([name])=>screen.supportsColor(light,name)) as [name,color] (name)}<button class="color-choice" data-color={name} disabled={readOnly||light.status==='Unavailable'} aria-label={name} aria-pressed={light.color===name} style:--color={color}><i></i><span>{name}</span></button>{/each}</div>
    </div>
  {/if}
</div>
