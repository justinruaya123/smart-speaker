<script>
  import Icon from './Icon.svelte';
  import ReminderList from './ReminderList.svelte';
  import AlarmPicker from './AlarmPicker.svelte';
  import AnalogClock from './AnalogClock.svelte';
  import Volume from './Volume.svelte';
  import {lightStyle,relativeTime} from '../view-utils.js';
  let {screen}=$props();let m=$derived(screen.model);
  let remaining=$derived(m.reminders.filter(r=>!r.done));
  let activeLights=$derived(m.lights.map((l,i)=>({l,i})).filter(({l,i})=>l.on||i===m.draggingLight));
  let track=$derived(screen.tracks[m.song]);
  let timerSeconds=$derived(m.timer?(m.timer.end?Math.max(0,(m.timer.end-m.now)/1000):m.timer.remaining):0);
</script>
<div class="cc-columns"><div class="cc-secondary"><div class="cc-reminder-menu"><section class="reminder-accordion"><button class="reminder-accordion-toggle" data-action="reminder-accordion" aria-expanded={m.reminderAccordionOpen}><strong>{remaining.length}</strong><span>Reminders</span><Icon name={m.reminderAccordionOpen?'chevron-up':'chevron-down'} /></button><div class="reminder-accordion-body" hidden={!m.reminderAccordionOpen}><ReminderList {screen} /></div></section></div><div class="cc-reminder-actions"><button class="cc-square" data-open="reminders" aria-label="Open reminders"><Icon name="list-checks" /></button><button class="cc-square" data-action="new-reminder" aria-label="Add reminder"><Icon name="plus" /></button></div></div><div class="control-panel">
{#if m.alarmPickerOpen}<AlarmPicker {screen} />{:else}
  <div class="cc-media"><button data-open="music" class="cc-track"><Icon name="disc-3" /><span><strong>{m.mediaLoaded?track[0]:m.spotifyMode==='demo'?'Music':'Spotify'}</strong><small>{m.mediaLoaded?track[1]:m.spotifyMode==='demo'?'Choose a track':m.spotifyLabel}</small></span></button><button data-action="playpause" disabled={!screen.canControlSpotify()} aria-label={`${m.playing?'Pause':'Play'} music`}><Icon name={m.playing?'pause':'play'} /></button></div>
  <div class="cc-grid"><button data-open="lights"><Icon name="lightbulb" /><span>Lights</span></button><button data-open="climate"><Icon name="thermometer" /><span>{m.target??'—'}°</span><small>{m.current===null?'Unavailable':m.current+'° inside'}</small></button><button data-open="timer"><span class="cc-timer-hourglass" class:paused={!m.timer?.end}><Icon name="hourglass" /></span><span data-cc-timer>{m.timer?screen.duration(timerSeconds):'Timer'}</span></button><button data-open="alarm">{#if m.alarm}<AnalogClock time={m.alarm.time} />{:else}<Icon name="alarm-clock" />{/if}<span data-cc-alarm>{m.alarm?relativeTime(m.alarm.deadline,m.now):'Alarm'}</span></button></div>
  {#if activeLights.length}<div class="cc-light-grid" aria-label="Lights currently on">{#each activeLights as {l,i} (l.id)}<div class="cc-light-tile" data-light-row={i} style={lightStyle(l,screen.colors)}><button class="light-icon" data-light-toggle={i} disabled={!screen.canControlLights()} aria-label={`Turn off ${l.name}`} aria-pressed={l.on}><Icon name="lightbulb" /></button><button class="light-name" data-quick-light={i} aria-label={`Edit ${l.name}`}>{l.name}</button><span class="brightness-fill" aria-hidden="true"></span></div>{/each}</div>{/if}
  <Volume {screen} />
{/if}
<div class="cc-bottom"><button data-open="contacts" aria-label="Calls"><Icon name="phone" /></button><button data-open="demo" aria-label="Intent and aura simulator"><Icon name="sliders-horizontal" /></button></div></div></div>
