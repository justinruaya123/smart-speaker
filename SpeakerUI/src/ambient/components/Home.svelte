<script>
  import InferenceReadout from './InferenceReadout.svelte';
  import Icon from './Icon.svelte';
  import TrackArtwork from './TrackArtwork.svelte';
  import ReminderList from './ReminderList.svelte';
  let { screen } = $props();
  let m = $derived(screen.model);
  let clock = $derived(new Date(m.now).toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false}));
  let remaining = $derived(m.reminders.filter(r=>!r.done));
  let track = $derived(screen.tracks[m.song]);
  let timerSeconds = $derived(m.timer ? (m.timer.end ? Math.max(0,(m.timer.end-m.now)/1000) : m.timer.remaining) : 0);
</script>
<div class="home-layer" inert={m.view!=='home'||!!m.alert}>
  <div class="corner-swipe-zone" aria-hidden="true"></div>
  <button class="wake-target" data-action="wake" aria-label={m.muted?'Assistant muted':m.agent==='listening'?'Stop listening':'Wake assistant'} aria-pressed={m.agent==='listening'}>
    <span class="clock-center"><time data-clock data-time={clock} aria-label={clock}><span>{clock.split(':')[0]}</span><span class="clock-colon" aria-hidden="true"><i></i><i></i></span><span>{clock.split(':')[1]}</span></time><span data-date>{new Date(m.now).toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'})}</span><span class="environment-line"><span class="home-temperature" aria-label={`Temperature ${m.current??'unavailable'}`}><strong data-home-temperature>{m.current??'—'}</strong><sup><Icon name="thermometer" /></sup><span>C</span></span><span class="environment-divider" aria-hidden="true"></span><span class="home-weather"><Icon name="cloud-sun" /><span>Partly cloudy</span></span></span></span>
  </button>
  <div class="agent-state" aria-live="polite" data-agent-label>{m.muted?'Muted':['idle','disconnected'].includes(m.agent)?'':m.agent==='failed'?'Try again':m.agent.replaceAll('-',' ')}</div>
  <div class="music-peek" hidden={!m.mediaLoaded} class:paused={!m.playing} style:--track-progress={track[2]>0?m.position/track[2]:0}>
    <button class="peek-track" data-open="music" aria-label="Open current track"><TrackArtwork url={m.spotifyArtworkUrl} compact /><span class="track-copy"><strong data-peek-song>{track[0]}</strong><small data-peek-artist>{track[1]}</small></span></button><button class="peek-play" data-action="playpause" disabled={!screen.canControlSpotify()} aria-label={m.playing?'Pause music':'Play music'}><Icon name={m.playing?'pause':'play'} /></button><span class="track-progress"><i></i></span>
  </div>
  <button class="control-trigger" data-open="control" aria-label="Open control center" aria-expanded={m.view==='control'}><Icon name="layout-grid" /></button>
  <button class="reminder-trigger" data-action="toggle-reminders" aria-label={`${m.remindersExpanded?'Close':'Open'} ${remaining.length} reminders`} aria-expanded={m.remindersExpanded}><span data-reminder-count>{remaining.length}</span><i data-unread-dot hidden={!remaining.some(r=>r.unread)}></i></button>
  <div class="home-reminders" aria-label="Reminder widgets"><ReminderList {screen} home /></div>
  <div class="time-rail" aria-label="Active timers and alarms">
    {#if m.timer}<button class="time-chip" data-open="timer" aria-label={`Timer ${screen.duration(timerSeconds)}`}><span class="rotating-hourglass" class:paused={!m.timer.end}><Icon name="hourglass" /></span><span><strong data-rail-timer>{screen.duration(timerSeconds)}</strong><small data-timer-status>{m.timer.end?'Timer':'Paused'}</small></span></button>{/if}
    {#if m.alarm}<button class="time-chip" data-open="alarm"><Icon name="alarm-clock" /><span><strong data-rail-alarm>{screen.clockTime(m.alarm.deadline)}</strong><small data-alarm-relative>In {Math.floor(Math.max(0,Math.ceil((m.alarm.deadline-m.now)/60000))/60)}h {Math.max(0,Math.ceil((m.alarm.deadline-m.now)/60000))%60}m</small></span></button>{/if}
  </div>
  <button class="demo-launch" data-open="demo" aria-label="Open debug panel" hidden={m.runtimeUI?.debug === false}><Icon name="sliders-horizontal" /></button>
  <button class="intent-receipt" data-open="intent" hidden={m.runtimeMode || !m.event} class:quiet={m.now-m.receiptAt>6500} aria-label="Inspect intent and parameter history"><span data-intent>{m.event?.result.intent}</span><small data-parameters>{m.event?Object.entries(m.event.result.parameters).map(([k,v])=>k+'='+JSON.stringify(v)).join(' · ')||'{}':''}</small></button>
</div>

<InferenceReadout {screen}/>
