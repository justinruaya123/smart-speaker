<script>
  import { untrack } from 'svelte';
  import { createAmbient } from './controller.svelte.js';
  import Icon from './components/Icon.svelte';
  import Home from './components/Home.svelte';
  import TaskContent from './components/TaskContent.svelte';
  import Call from './components/Call.svelte';
  import './screen.css';
  import './baked-motion.css';
  const screen=createAmbient(),m=screen.model;
  let title=$derived(({lights:'Lights',timer:'Timer','new-reminder':m.editingReminder===null?'New reminder':'Reminder',reminders:'Reminders',contacts:m.haMode==='live'?'Cameras':'Calls',call:'Call',weather:m.weather?`${m.weather.location} · OpenWeather`:'Weather',intent:'Request history',demo:'Debug'})[m.view]??'');
  function attachScreen(element){return untrack(()=>screen.mount(element));}
</script>
<main id="ambient-assistant" class="ambient-assistant" aria-label="Ambient voice assistant landscape" data-view={m.view} data-state={m.agent} data-muted={String(m.muted)} data-sound-muted={String(m.soundMuted)} data-reminder-editing={String(m.remindersExpanded)} {@attach attachScreen}>
  <div class="backdrop" aria-hidden="true"></div><div class="aura-anchor" aria-hidden="true"><canvas class="livekit-aura" hidden={!!m.auraError}></canvas><span class="aura-fallback" hidden={!m.auraError}></span></div><div class="state-edge" aria-hidden="true"></div>
  <Home {screen} />
  <button class="mute-indicator" data-action="mute" aria-label={m.muted?'Unmute microphone':'Mute microphone'} aria-pressed={m.muted}><Icon name={m.muted?'mic-off':'mic'} /></button>
  <section class="task-view" aria-label={m.view.replace('-',' ')} hidden={m.view==='home'} inert={!!m.alert}><div class="task-heading"><button class="back-button" data-action="back" aria-label="Back" hidden={!['new-reminder','call'].includes(m.view)}><Icon name="arrow-left" /></button><h1 data-view-title>{title}</h1></div><button class="close-task" data-action="close" aria-label="Close task"><Icon name="x" /></button><div class="task-content" style:--count-offset={m.auto?(1-m.auto.remaining/5000)*100:0}><TaskContent {screen} /></div></section>
  <Call {screen} />
  <svg class="screen-countdown" viewBox="0 0 800 480" preserveAspectRatio="none" aria-hidden="true" hidden={m.closeProgress===null?true:undefined} style:--screen-count-offset={100-(m.closeProgress??0)*100}><path d="M400 478 H26 Q2 478 2 454 V26 Q2 2 26 2 H400" pathLength="100"/><path d="M400 478 H774 Q798 478 798 454 V26 Q798 2 774 2 H400" pathLength="100"/></svg>
  <aside style:--toast-light-color={m.toast?.symbol==='lightbulb'?(m.lights[m.selectedLight].on?screen.colors[m.lights[m.selectedLight].color]:'#7f929a'):null} style:--toast-light-filter={m.toast?.symbol==='lightbulb'&&m.lights[m.selectedLight].on?`drop-shadow(0 0 ${m.lights[m.selectedLight].brightness/7}px ${screen.colors[m.lights[m.selectedLight].color]})`:null} class="toast" aria-live="polite" hidden={!m.toast} inert={m.view==='control'||!!m.alert} class:fading={m.toast&&!m.toast.held&&m.toast.until-m.now<400} style:--toast-progress={m.toast?Math.max(0,(m.toast.held?m.toast.remaining:m.toast.until-m.now)/4200):0}><button class="toast-main" data-action="toast-open">{#if m.toast}<Icon name={m.toast.symbol} /><span><strong>{m.toast.title}</strong><small>{m.toast.detail}</small></span>{/if}</button><button class="toast-close" data-action="toast-close" aria-label="Dismiss notification"><Icon name="x" /></button><div class="toast-life"></div></aside>
  <div class="attention-edge" aria-hidden="true" hidden={!m.alert}></div><div class="alert-surface" role="alertdialog" tabindex="-1" aria-modal="true" aria-label="Notification" hidden={!m.alert}><div class="alert-content">{#if m.alert}{#if m.alert.kind!=='reminder'}<Icon name={m.alert.kind==='timer'?'timer':'alarm-clock'} />{/if}<h2>{m.alert.title}</h2><p class="subtle">{m.alert.subtitle}</p><div class="task-actions"><button class="button-secondary" data-action="alert-snooze">{m.alert.kind==='reminder'?'In 5 min':'Snooze 5 min'}</button><button class="button-primary" data-action="alert-dismiss">{m.alert.kind==='reminder'?'Got it':'Dismiss'}</button></div>{/if}</div></div>
  <div class="orientation-note" hidden><Icon name="rotate-cw" /><p>Turn to landscape</p></div><span class="sr-only" role="status" aria-live="polite" data-announcement>{m.announcement}</span>
</main>
