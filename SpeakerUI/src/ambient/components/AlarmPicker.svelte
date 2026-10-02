<script>
  import { untrack } from 'svelte';
  import { mountAlarmWheels } from '../alarm-wheels.js';
  import Icon from './Icon.svelte';
  import AnalogClock from './AnalogClock.svelte';
  let { screen } = $props();
  let m = $derived(screen.model);
  let minutes = $derived(Math.max(0,Math.ceil((screen.nextAlarm(m.alarmDraft,m.now)-m.now)/60000)));
  function attachWheels(element){return untrack(()=>{const wheels=mountAlarmWheels(element,{value:m.alarmDraft,onChange:screen.alarmChange,isMuted:()=>m.soundMuted});return()=>wheels.destroy();});}
</script>
<div class="inline-alarm"><div class="alarm-preview"><AnalogClock time={m.alarmDraft} /><span data-alarm-until>In {Math.floor(minutes/60)}h {minutes%60}m</span></div><div class="alarm-wheels" aria-label="Alarm time" {@attach attachWheels}></div><div class="alarm-wheel-actions"><button data-action="alarm-picker-close" aria-label="Cancel alarm setting"><Icon name="x" /></button>{#if m.alarm}<button data-action="alarm-remove" aria-label="Remove alarm"><Icon name="trash-2" /></button>{/if}<button class="alarm-confirm" data-action="alarm-save" aria-label="Set alarm"><Icon name="check" /></button></div></div>
