<script>
  import { onMount } from 'svelte';
  import Icon from './components/Icon.svelte';
  import RuntimeDebug from './components/RuntimeDebug.svelte';
  import { applyRuntimeEvent } from './runtime-events.js';
  import { createLocalPlayer } from './local-player.js';
  import { taskLabels } from '../../shared/canonical-command.js';

  const m = $state({ history: [], runtimeMode: false, runtimeReady: false, runtimeState: 'starting', runtimeEngine: '',
    runtimeActions: false, runtimeResult: null, runtimeError: '', runtimeSession: '', runtimeSequence: -1,
    runtimeConfig: null, runtimeUI: null, event: null, tentative: null, ownership: null, muted: false, view: 'home', announcement: '' });
  let view = $state('home'), now = $state(Date.now()), busy = $state(false), error = $state('');
  let profile = $state(''), ownerGating = $state(false), connected = $state(false);
  let state = $state({ revision: -1, jobs: [], receipts: [], actions_enabled: false, timezone:'Asia/Manila',
    bindings: { lights: {}, songs: [], climate: null, weather: null, media_player: {}, contacts: [], messages: false } });
  let entities = $state([]), haStatus = $state('connecting'), execution = $state(null), weather = $state(null);
  let selectedLight = $state(''), selectedSong = $state(''), selectedContact = $state(''), selectedTask = $state('');
  let brightness = $state(50), color = $state('white'), volume = $state(50), degrees = $state(22);
  let seconds = $state(300), alarm = $state('07:00'), clientId = '', localOwner = false, player;
  let closedAlert = $state(''), soundEnabled = $state(false), alertAudio;
  let pageLifetime;
  const surfaces = [['music','music-2','Music'], ['lights','lightbulb','Lights'], ['climate','thermometer','Temperature'],
    ['timer','timer','Timer'], ['alarm','alarm-clock','Alarm'], ['reminders','list-checks','Reminders'],
    ['calls','phone','Call'], ['message','bell','Message'], ['weather','cloud-sun','Weather'],
    ['time','clock-3','Time'], ['volume','volume-2','Volume'], ['history','clock-3','History']];
  const titles = Object.fromEntries(surfaces.map(([key,,title])=>[key,title]));
  const pretty = id => String(id || '').split('.').at(-1).replaceAll('_',' ').replace(/\b\w/g, value=>value.toUpperCase());
  const duration = milliseconds => { const total = Math.max(0, Math.ceil(milliseconds/1000)); return `${Math.floor(total/60)}:${String(total%60).padStart(2,'0')}`; };
  const clockTime = (value,timezone=state.timezone) => new Intl.DateTimeFormat('en-PH',{hour:'numeric',minute:'2-digit',timeZone:timezone}).format(value);
  let clock = $derived(clockTime(now));
  let status = $derived(!connected ? 'Connecting' : !m.runtimeMode ? 'Microphone unavailable' : !m.runtimeReady ? 'Runtime offline' :
    ({awaiting_microphone:'Choose a microphone',wake_listening:'Say “Hey Iris”',command_ready:'Ready for your command',recording:'Listening',processing:'Understanding',muted:'Microphone muted',capture_error:'Check microphone',runtime_error:'Runtime offline'})[m.runtimeState] || 'Ready');
  let latest = $derived(m.tentative?.candidate || (m.event?.result.modelIntent ? {intent:m.event.result.modelIntent,slots:m.event.result.slots} : null));
  let phase = $derived(m.tentative ? 'Tentative · preview' : m.event ? (m.event.accepted===false?'Rejected':m.event.source==='onnx-wav'?'Accepted · replay only':'Accepted') : 'Awaiting a command');
  let activeJobs = $derived(state.jobs.filter(job=>!['completed','cancelled'].includes(job.status)));
  let due = $derived(activeJobs.find(job=>job.status==='due' && job.id!==closedAlert));
  let observedLights = $derived((Array.isArray(state.bindings.lights[selectedLight]) ? state.bindings.lights[selectedLight] : [state.bindings.lights[selectedLight]]).filter(Boolean).map(id=>entities.find(entity=>entity.id===id)).filter(Boolean));
  let climate = $derived(entities.find(entity=>entity.id===state.bindings.climate));
  let media = $derived(entities.find(entity=>entity.id===state.bindings.media_player.entity_id));
  const enabled = () => state.actions_enabled && !busy;
  const screen = { model: m };

  async function enableAlertSound() {
    try {
      alertAudio ??= new AudioContext(); await alertAudio.resume(); soundEnabled=alertAudio.state==='running';
    } catch { soundEnabled=false; }
  }
  function focusAlert(node) {
    const previous=document.activeElement;
    node.querySelector('button')?.focus();
    if (alertAudio?.state==='running') for(let index=0;index<3;index++) {
      const oscillator=alertAudio.createOscillator(),gain=alertAudio.createGain(),start=alertAudio.currentTime+index*.5;
      oscillator.frequency.value=880;gain.gain.setValueAtTime(0,start);gain.gain.linearRampToValueAtTime(.12,start+.03);gain.gain.linearRampToValueAtTime(0,start+.3);
      oscillator.connect(gain);gain.connect(alertAudio.destination);oscillator.start(start);oscillator.stop(start+.32);
      oscillator.onended=()=>{oscillator.disconnect();gain.disconnect();};
    }
    return ()=>{if(previous?.isConnected)previous.focus();};
  }
  function trapAlert(event) {
    if(event.key!=='Tab')return;
    const buttons=[...event.currentTarget.querySelectorAll('button:not(:disabled)')];
    if(event.shiftKey&&document.activeElement===buttons[0]){event.preventDefault();buttons.at(-1)?.focus();}
    else if(!event.shiftKey&&document.activeElement===buttons.at(-1)){event.preventDefault();buttons[0]?.focus();}
  }

  async function request(path, body) {
    const response = await fetch(path, { ...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
      signal: pageLifetime ? AbortSignal.any([pageLifetime.signal,AbortSignal.timeout(16000)]) : AbortSignal.timeout(16000) });
    const value = await response.json();
    if (!response.ok) throw new Error(value.message || value.error || 'Request failed');
    return value;
  }
  function snapshot(value) {
    if (typeof value.revision !== 'number' || value.revision < state.revision || !Array.isArray(value.jobs)) return;
    state = { ...state, ...value };
    if (typeof value.client_id === 'string') { clientId = value.client_id; localOwner = value.local_player === true; }
  }
  async function receiveExecution(event) {
    applyRuntimeEvent(m,event,()=>{});
    execution = event.execution;
    const intent = execution.command?.intent, slots=execution.command?.slots || {};
    if (['succeeded','observed','local','requested'].includes(execution.status)) {
      view=({PLAY_MUSIC:'music',PAUSE:'music',STOP:'music',NEXT:'music',VOLUME:'volume',LIGHT_ON:'lights',LIGHT_OFF:'lights',BRIGHTNESS:'lights',COLOR:'lights',
        WEATHER:'weather',TIME:'time',TEMPERATURE:'climate',TIMER:'timer',ALARM:'alarm',LIST_REMINDERS:'reminders',CREATE_REMINDER:'reminders',CALL:'calls',MESSAGE:'message'})[intent] || view;
      if(slots.light_id)selectedLight=slots.light_id;
      if(slots.song_id)selectedSong=slots.song_id;
      if(slots.contact_id)selectedContact=slots.contact_id;
      if(intent==='BRIGHTNESS')brightness=slots.percent;
      if(intent==='COLOR')color=slots.color;
      if(intent==='TEMPERATURE')degrees=slots.degrees;
    }
    if (execution.weather) { weather = execution.weather; view = 'weather'; }
    if (execution.status === 'local' && intent === 'LIST_REMINDERS') view = 'reminders';
    if (execution.status === 'local' && intent === 'TIME') view = 'time';
    const result = await player?.handle(execution, localOwner);
    if (result && pageLifetime && !pageLifetime.signal.aborted) {
      try { await request('/api/actions/local-receipt',{...result,client_id:clientId}); }
      catch { error = 'Local playback outcome could not be recorded. It will not be retried.'; }
    }
  }
  async function run(intent, slots={}) {
    if (busy) return;
    busy = true; error = '';
    if(['TIMER','ALARM'].includes(intent))void enableAlertSound();
    const request_id = crypto.randomUUID();
    try { execution = await request('/api/actions',{request_id,command:{intent,slots}}); }
    catch (failure) { error = failure.message + '. This request will not be retried automatically.'; }
    finally { busy = false; }
  }
  async function jobAction(job, action) {
    busy = true; error = '';
    try { await request('/api/actions/jobs/'+encodeURIComponent(job.id),{action}); }
    catch (failure) { error = failure.message; }
    finally { busy = false; }
  }
  async function mute() {
    if (m.runtimeState === 'muted' || !m.runtimeReady) { view='settings'; return; }
    try { await request('/api/runtime/mute',{}); } catch (failure) { error=failure.message; }
  }

  onMount(()=>{
    pageLifetime = new AbortController();
    const sockets = [], timers = new Set();
    let closed = false, pollTimer;
    player = createLocalPlayer(new Audio());
    const later = fn => { const id=setTimeout(()=>{timers.delete(id);if(!closed)fn();},2000);timers.add(id); };
    function socket(path, handler, disconnected) {
      const ws = new WebSocket(`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}${path}`);sockets.push(ws);
      ws.onmessage = message=>{try { handler(JSON.parse(message.data)); } catch { error='An invalid server update was ignored.'; }};
      ws.onclose = ()=>{disconnected?.();if(!closed)later(()=>socket(path,handler,disconnected));};
    }
    async function poll() {
      try {
        const [actions,ha] = await Promise.all([request('/api/actions/state'),request('/api/home-assistant/entities')]);
        snapshot(actions); entities=ha.entities;haStatus=ha.status;
      } catch (failure) { if(!closed) { error=failure.message;haStatus='disconnected'; } }
      finally { if(!closed)pollTimer=setTimeout(poll,2000); }
    }
    socket('/ws/actions',event=>{
      if(event.event==='action_state')snapshot(event);
      if(event.event==='player_role')localOwner=event.local_player===true;
      if(event.event==='execution')void receiveExecution(event);
      if(event.event==='action_error')error=event.message;
    },()=>{localOwner=false;});
    void request('/api/health').then(health=>{
      if(closed)return;
      connected=true;m.runtimeMode=health.inference==='onnx';
      if(m.runtimeMode)socket('/ws/runtime',event=>{
        if(event.event==='execution')return; // /ws/actions is the single execution subscription.
        applyRuntimeEvent(m,event,()=>{});
        if(event.event==='ready'){profile=event.profile||'';ownerGating=event.owner_gating_enabled===true;}
      },()=>{m.runtimeReady=false;m.tentative=null;m.runtimeError='Runtime disconnected';});
    }).catch(failure=>{error=failure.message;});
    void poll();
    const ticker=setInterval(()=>{now=Date.now();},250);
    return ()=>{closed=true;pageLifetime.abort();clearInterval(ticker);clearTimeout(pollTimer);for(const timer of timers)clearTimeout(timer);for(const ws of sockets)ws.close();player.close();void alertAudio?.close();};
  });
</script>

<main class="iris" aria-label="Iris smart speaker" inert={!!due}>
  <div class="identity"><span class="wordmark">iris<span aria-hidden="true">·</span></span><span class="connection" class:online={m.runtimeReady}>{m.runtimeReady?'Local voice assistant':'Local speaker'}</span></div>
  <div class="corner-controls"><time>{clock}</time><button aria-label="Microphone settings" onclick={()=>view='settings'}><Icon name="sliders-horizontal" /></button><button aria-label={m.muted?'Choose microphone':'Mute microphone'} aria-pressed={m.muted} onclick={mute}><Icon name={m.muted?'mic-off':'mic'}/></button></div>

  <section class="voice" aria-label="Live recognition" aria-live="polite">
    <div class="voice-heading"><span class="state-dot" class:listening={['recording','command_ready'].includes(m.runtimeState)} class:muted={m.muted}></span><h1>{status}</h1></div>
    <p class="profile">{profile==='owner-protected'?(ownerGating?'Experimental owner voice check enabled':'Experimental owner voice check unavailable'):profile==='benchmark-compatible'?'Benchmark profile · actions disabled':'Select a microphone in settings'}{m.ownership?` · ${m.ownership.accepted?'Voice check passed':pretty(m.ownership.reason || 'voice check rejected')}`:''}</p>
    <div class="recognition" class:tentative={!!m.tentative} data-recognition-phase={m.tentative?'tentative':'final'}>
      <span class="phase">{phase}</span>
      {#if latest}<strong>{latest.intent}</strong><code>{Object.entries(latest.slots || {}).map(([key,value])=>`${key}=${JSON.stringify(value)}`).join(' · ') || '{}'}</code>
      {:else if m.event?.reason}<span>{m.event.reason}</span>{:else}<span class="empty">Your intent and parameters will appear here.</span>{/if}
    </div>
    {#if m.runtimeError}<p class="error" role="status">{m.runtimeError}</p>{/if}
  </section>

  <section class="controls" aria-label={view==='home'?'Use cases':titles[view] || 'Settings'}>
    {#if view!=='home'}<div class="panel-heading"><button aria-label="Back to home" onclick={()=>view='home'}><Icon name="arrow-left"/></button><h2>{titles[view] || 'Microphone & diagnostics'}</h2><span>{state.actions_enabled?'Actions enabled':'Actions disabled'}</span></div>{/if}
    <div class="panel-body">
    {#if view==='home'}
      <div class="use-cases">{#each surfaces as [key,icon,title] (key)}<button onclick={()=>view=key}><Icon name={icon}/><span>{title}</span></button>{/each}</div>
      <div class="home-summary"><span>{activeJobs.length?`${activeJobs.length} saved timer, alarm or reminder${activeJobs.length===1?'':'s'}`:'No active timers, alarms or reminders'}</span><span>{state.actions_enabled?'Actions enabled':'Actions disabled'}</span></div>
    {:else if view==='lights'}
      <div class="form-row"><label>Lights<select bind:value={selectedLight}><option value="">Choose lights</option>{#each Object.keys(state.bindings.lights) as id (id)}<option value={id}>{pretty(id)}</option>{/each}</select></label><button disabled={!enabled()||!selectedLight} onclick={()=>run('LIGHT_ON',{light_id:selectedLight})}>On</button><button disabled={!enabled()||!selectedLight} onclick={()=>run('LIGHT_OFF',{light_id:selectedLight})}>Off</button></div>
      <div class="form-row"><label>Brightness <input type="number" min="0" max="100" step="1" bind:value={brightness}/></label><button disabled={!enabled()||!selectedLight} onclick={()=>run('BRIGHTNESS',{light_id:selectedLight,percent:brightness})}>Set %</button><label>Color<select bind:value={color}>{#each ['red','blue','green','yellow','white','orange','purple','pink','cyan','magenta','warm white','cool white'] as name (name)}<option value={name}>{pretty(name)}</option>{/each}</select></label><button disabled={!enabled()||!selectedLight} onclick={()=>run('COLOR',{light_id:selectedLight,color})}>Set color</button></div>
      {#each observedLights as entity (entity.id)}<p class="observed">Observed: {entity.name} · {entity.stale||haStatus!=='connected'?'State unavailable':entity.state}{typeof entity.attributes.brightness==='number'?` · ${Math.round(entity.attributes.brightness/255*100)}%`:''}</p>{:else}<p class="hint">{Object.keys(state.bindings.lights).length?'Choose a configured light or group.':'Lights are not configured.'}</p>{/each}
    {:else if view==='music'}
      <div class="form-row"><label>Song<select bind:value={selectedSong}><option value="">Choose a song</option>{#each state.bindings.songs as id (id)}<option value={id}>{pretty(id)}</option>{/each}</select></label><button disabled={!enabled()||!selectedSong} onclick={()=>run('PLAY_MUSIC',{song_id:selectedSong})}><Icon name="play"/>Play song</button><button disabled={!enabled()||!state.bindings.media_player.provider} onclick={()=>run('PLAY_MUSIC')}>Resume</button></div>
      <div class="form-row"><button disabled={!enabled()||!state.bindings.media_player.provider} onclick={()=>run('PAUSE')}><Icon name="pause"/>Pause</button><button disabled={!enabled()||!state.bindings.media_player.provider} onclick={()=>run('STOP')}>Stop</button><button disabled={!enabled()||!state.bindings.media_player.provider} onclick={()=>run('NEXT')}><Icon name="skip-forward"/>Next</button></div>
      <p class="observed">{state.bindings.media_player.provider?`Playback: ${state.bindings.media_player.provider.replaceAll('_',' ')}`:'Playback is not configured.'}{media?` · Observed ${media.stale?'unavailable':media.state}${media.attributes.media_title?' · '+media.attributes.media_title:''}`:''}</p>
      {#if state.bindings.media_player.provider==='local'}<p class="hint">Local playback uses one connected browser. If browser audio is blocked, select the song again explicitly after enabling audio.</p>{/if}
    {:else if view==='volume'}
      <div class="form-row"><button disabled={!enabled()} onclick={()=>run('VOLUME',{operation:'down'})}><Icon name="volume-2"/>Lower</button><label>Volume %<input type="number" min="0" max="100" step="1" bind:value={volume}/></label><button disabled={!enabled()} onclick={()=>run('VOLUME',{operation:'set',percent:volume})}>Set volume</button><button disabled={!enabled()} onclick={()=>run('VOLUME',{operation:'up'})}><Icon name="volume-2"/>Louder</button><button disabled={!enabled()} onclick={()=>run('VOLUME',{operation:'mute'})}>Mute audio</button></div><p class="hint">{state.bindings.media_player.provider?'Uses the configured music player.':'Playback is not configured.'}</p>
    {:else if view==='climate'}
      <div class="form-row"><label>Temperature °C<input type="number" min="-50" max="60" step="1" bind:value={degrees}/></label><button disabled={!enabled()||!state.bindings.climate} onclick={()=>run('TEMPERATURE',{degrees,unit:'C'})}>Set temperature</button></div>
      <p class="observed">{!state.bindings.climate?'Climate is not configured.':!climate||climate.stale||haStatus!=='connected'?'Observed temperature unavailable':`Observed ${climate.attributes.current_temperature??'—'}${climate.attributes.temperature_unit||''} · Target ${climate.attributes.temperature??'—'}${climate.attributes.temperature_unit||''}`}</p>
    {:else if view==='weather'}
      <div class="form-row"><button disabled={!enabled()||!state.bindings.weather} onclick={()=>run('WEATHER')}><Icon name="cloud-sun"/>Read current weather</button>{#if weather}<strong class="answer">{weather.temperature??'—'}{weather.unit||''} · {pretty(weather.state)}</strong>{/if}</div><p class="hint">{weather?`Observed ${weather.observed_at?new Date(weather.observed_at).toLocaleString():'time unavailable'}`:state.bindings.weather?'Read the configured weather entity.':'Weather is not configured.'}</p>
    {:else if view==='time'}
      <div class="form-row"><strong class="answer">{clock}</strong><span>{state.timezone}</span><button disabled={!enabled()} onclick={()=>run('TIME')}>Read time</button></div>
    {:else if view==='timer'||view==='alarm'}
      <div class="form-row">{#if view==='timer'}<label>Duration in seconds<input type="number" min="1" max="86400" step="1" bind:value={seconds}/></label><button disabled={!enabled()} onclick={()=>run('TIMER',{duration_seconds:seconds})}>Start timer</button>{:else}<label>Alarm time · Asia/Manila<input type="time" bind:value={alarm}/></label><button disabled={!enabled()||!alarm} onclick={()=>run('ALARM',{time:alarm,timezone:'Asia/Manila'})}>Set alarm</button>{/if}</div>
      <div class="jobs">{#each activeJobs.filter(job=>job.kind===view) as job (job.id)}<div class="job"><div><strong>{job.label}</strong><span>{job.status==='scheduled'?view==='timer'?duration(job.deadline-now):clockTime(job.deadline,job.timezone || 'Asia/Manila'):job.status==='paused'?duration(job.remaining):job.status==='missed'?'Missed while offline':job.status}</span></div>{#if job.kind==='timer'&&['scheduled','paused'].includes(job.status)}<button disabled={!enabled()} onclick={()=>jobAction(job,job.status==='paused'?'resume':'pause')}>{job.status==='paused'?'Resume':'Pause'}</button>{/if}<button disabled={!enabled()} onclick={()=>jobAction(job,'cancel')}>Cancel</button></div>{:else}<p class="hint">No saved {view==='timer'?'timers':'alarms'}.</p>{/each}</div>
    {:else if view==='reminders'}
      <div class="form-row"><label>Reminder<select aria-label="Reminder task" bind:value={selectedTask}><option value="">Choose a task</option>{#each Object.entries(taskLabels) as [id,label] (id)}<option value={id}>{label}</option>{/each}</select></label><button disabled={!enabled()||!selectedTask} onclick={()=>run('CREATE_REMINDER',{task_id:selectedTask})}>Save reminder</button><button disabled={!enabled()} onclick={()=>run('LIST_REMINDERS')}>Refresh list</button></div>
      <div class="jobs">{#each activeJobs.filter(job=>job.kind==='reminder') as job (job.id)}<div class="job"><strong>{job.label}</strong><button disabled={!enabled()} onclick={()=>jobAction(job,'complete')}>Done</button></div>{:else}<p class="hint">No saved reminders.</p>{/each}</div>
    {:else if view==='calls'}
      <div class="form-row"><label>Contact<select bind:value={selectedContact}><option value="">Choose a contact</option>{#each state.bindings.contacts as id (id)}<option value={id}>{pretty(id)}</option>{/each}</select></label><button disabled={!enabled()||!selectedContact} onclick={()=>run('CALL',{contact_id:selectedContact})}><Icon name="phone"/>Call</button></div><p class="hint">{state.bindings.contacts.length?'Runs your configured contact script. The receipt reports provider acceptance.':'Calls are not configured.'}</p>
    {:else if view==='message'}
      <button disabled={!enabled()||!state.bindings.messages} onclick={()=>run('MESSAGE')}><Icon name="bell"/>Send configured preset</button><p class="hint">{state.bindings.messages?'The destination and message are defined in your Home Assistant script.':'A preset message script is not configured.'}</p>
    {:else if view==='settings'}
      <div class="form-row"><button onclick={enableAlertSound}>{soundEnabled?'Alert sound enabled':'Enable alert sound'}</button><span class="hint">Visual alerts also appear on this display.</span></div><RuntimeDebug {screen}/><a class="diagnostics-link" href="?diagnostics">Open audio diagnostics</a>
    {:else if view==='history'}
      <div class="history">{#each m.history as item (item.result.utteranceId)}<details><summary>{item.result.modelIntent||item.result.intent} · {item.status} · {item.raw.source}</summary><pre>{JSON.stringify(item.raw,null,2)}</pre></details>{/each}{#each [...state.receipts].reverse() as item (item.id)}<details><summary>{item.command?.intent||'Action'} · {item.status}</summary><p>{item.detail}</p><pre>{JSON.stringify(item.command,null,2)}</pre></details>{/each}{#if !m.history.length&&!state.receipts.length}<p class="hint">No requests yet.</p>{/if}</div>
    {/if}
    </div>
  </section>

  <section class="execution" aria-label="Execution receipt" aria-live="polite" data-execution-status={execution?.status||'none'}>
    {#if error}<strong>Attention</strong><p class="error">{error}</p>
    {:else if execution}<strong>{execution.status==='local'&&execution.command?.intent==='TIME'?'Local time':({succeeded:'Request accepted',observed:'Observed result',local:'Saved locally',requested:'Waiting for player',not_configured:'Not configured',unknown:'Outcome unknown',failed:'Action failed',disabled:'Actions disabled',preview:'Replay preview',rejected:'Rejected',pending:'Dispatching'})[execution.status]||execution.status}</strong><p>{execution.detail}</p>
    {:else}<strong>Ready when you are</strong><p>{state.actions_enabled?'Accepted live commands use your configured actions.':'Actions are disabled for this configuration.'}</p>{/if}
  </section>

</main>
{#if due}<div class="alert-backdrop"><div class="due-alert" tabindex="-1" role="alertdialog" aria-modal="true" aria-label={due.label} onkeydown={trapAlert} {@attach focusAlert}><Icon name={due.kind==='timer'?'timer':'alarm-clock'}/><h2>{due.label}</h2><p>{due.kind==='timer'?'Time is up.':'Your alarm is due.'}</p><div class="form-row">{#if state.actions_enabled}<button disabled={!enabled()} onclick={()=>jobAction(due,'snooze')}>Snooze 5 minutes</button><button class="primary" disabled={!enabled()} onclick={()=>jobAction(due,'complete')}>Dismiss</button>{:else}<button onclick={()=>closedAlert=due.id}>Close alert</button>{/if}</div></div></div>{/if}

<style>
  :global(body){font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#e2ecef;background:#090d11}button,input,select{font:inherit;color:inherit}button,a,input,select{touch-action:manipulation}button{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:42px;padding:9px 15px;border:1px solid #30444e;background:#14212a;border-radius:13px;cursor:pointer}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #9ce8d3;outline-offset:3px}button:hover:not(:disabled){background:#20333e}button :global(svg){width:19px;height:19px}input,select{max-width:100%;min-height:40px;padding:8px 10px;border:1px solid #3a505a;background:#0c151c;border-radius:9px;box-sizing:border-box}input[type=number]{width:100px}.iris{height:100dvh;box-sizing:border-box;max-width:1280px;margin:auto;padding:18px 24px 14px;display:grid;grid-template-columns:1fr auto;grid-template-rows:38px minmax(110px,1fr) minmax(152px,1.5fr) auto;gap:12px;overflow:hidden}.identity{display:flex;align-items:center;gap:16px}.wordmark{font-size:31px;font-weight:600;letter-spacing:-2px}.wordmark>span{color:#9ce8d3}.connection,.profile,.hint,.observed{font-size:12px;color:#96adb7}.online{color:#a9cebf}.corner-controls{display:flex;align-items:center;gap:8px}.corner-controls time{font-size:14px;margin-right:6px;font-variant-numeric:tabular-nums}.corner-controls button{background:none;min-height:38px;padding:8px}.voice{grid-column:1/-1;display:flex;flex-direction:column;justify-content:center;gap:6px;min-height:0}.voice-heading{display:flex;align-items:center;gap:12px}h1{margin:0;font-size:clamp(25px,3.5vw,42px);font-weight:500;letter-spacing:-1px}.state-dot{width:10px;height:10px;border-radius:50%;background:#657985;box-shadow:none;flex-shrink:0}.state-dot.listening{background:#9ce8d3}.state-dot.muted{background:#b98078}.profile{margin:0 0 3px 22px}.recognition{display:flex;align-items:baseline;flex-wrap:wrap;gap:7px 12px;min-height:30px;font-size:12px;color:#b9cbd2}.recognition strong{color:#e2ecef;letter-spacing:.05em}.recognition code{overflow-wrap:anywhere}.recognition.tentative{color:#b8baa4}.phase{font-size:10px;text-transform:uppercase;letter-spacing:.08em;min-width:130px}.empty{opacity:.65}.controls{grid-column:1/-1;border:1px solid #233641;border-radius:18px;background:#101a22;min-height:0;display:flex;flex-direction:column;overflow:hidden}.panel-heading{display:flex;gap:10px;align-items:center;padding:8px 14px 0}.panel-heading h2{font-size:16px;font-weight:500;margin:0}.panel-heading button{min-height:30px;padding:3px;background:none;border:0}.panel-heading>span{font-size:10px;color:#8099a5;margin-left:auto}.panel-body{padding:14px 18px;overflow:auto;min-height:0;flex:1}.use-cases{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px}.use-cases button{display:flex;flex-direction:column;gap:8px;padding:12px 5px;background:#15232c;font-size:12px;min-height:65px}.use-cases button :global(svg){color:#a7c8ca}.home-summary{display:flex;justify-content:space-between;gap:10px;font-size:10px;color:#829aa6;margin-top:12px}.form-row{display:flex;align-items:flex-end;flex-wrap:wrap;gap:9px;margin-bottom:10px}label{display:flex;flex-direction:column;gap:5px;font-size:11px;color:#acc0c9}select{min-width:150px;max-width:250px}.hint,.observed{line-height:1.5;margin:8px 0}.observed{color:#c1d1d7}.answer{font-size:32px;font-weight:400}.execution{grid-column:1/-1;border-top:1px solid #233641;padding-top:10px;display:flex;align-items:baseline;gap:15px;min-height:32px}.execution strong{font-size:11px;letter-spacing:.03em;white-space:nowrap;color:#a9d8ca}.execution p{font-size:11px;line-height:1.4;margin:0;color:#9bb1bd}.error{color:#e4b1a9!important;font-size:12px;margin:0}.jobs{display:flex;flex-direction:column;gap:8px}.job{display:flex;align-items:center;gap:8px;border-top:1px solid #233641;padding-top:8px}.job>div,.job>strong{margin-right:auto;font-size:13px}.job>div{display:flex;align-items:center;gap:16px}.job span{font-size:12px;color:#a9c2cc}.job button{min-height:30px;padding:5px 10px;font-size:11px}.history{font-size:12px}.history details{border-bottom:1px solid #233641;padding:8px 0}.history summary{cursor:pointer}.history pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.diagnostics-link{color:#a9d8ca;font-size:12px;display:inline-block;margin-top:12px}.alert-backdrop{position:fixed;inset:0;z-index:20;background:#080d12ed;display:grid;place-items:center;padding:20px}.due-alert{padding:26px;background:#14232d;border:1px solid #536c70;border-radius:22px;text-align:center;max-width:90vw}.due-alert> :global(svg){width:40px;height:40px;color:#b9eddc}.due-alert h2{font-weight:500;font-size:28px}.due-alert p{color:#b4c9d0}.due-alert .form-row{justify-content:center;margin:20px 0 0}.primary{background:#a8ddcc;color:#101e24;border-color:#a8ddcc}
  @media(min-height:580px){.iris{padding:24px 32px;gap:18px;grid-template-rows:44px minmax(150px,1fr) minmax(205px,1.4fr) auto}.use-cases button{min-height:78px}.panel-body{padding:20px 22px}}
  @media(max-width:650px){.iris{padding:14px;height:100dvh;grid-template-rows:38px minmax(120px,auto) minmax(180px,1fr) auto;gap:12px}.connection{display:none}.corner-controls time{font-size:12px}.voice-heading h1{font-size:26px}.profile{font-size:11px}.use-cases{grid-template-columns:repeat(3,minmax(0,1fr))}.use-cases button{min-height:62px}.panel-body{padding:12px}.phase{min-width:0;width:100%}.execution{display:block;min-height:44px}.execution p{margin-top:5px}.home-summary{font-size:9px}.panel-heading>span{display:none}.form-row{gap:8px}.form-row button{padding:8px 10px;font-size:12px}select{max-width:200px;min-width:130px}}
  @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
