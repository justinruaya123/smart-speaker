<script>
  import Icon from './Icon.svelte';
  let {screen,className='cc-volume',stop=false}=$props();
  let m=$derived(screen.model);
  let muted=$derived(m.spotifyMode==='demo'?m.soundMuted:m.volume===0);
</script>
<div class={className}><button data-action="sound-mute" disabled={m.spotifyMode!=='demo'&&(!screen.canControlSpotify()||m.volume===null)} aria-label={m.spotifyMode==='demo'?(muted?'Unmute sounds':'Mute sounds'):(muted?'Unmute Spotify':'Mute Spotify')} aria-pressed={muted}><Icon name={muted?'volume-x':'volume-2'} /></button><input type="range" min="0" max="100" disabled={!screen.canControlSpotify()||m.volume===null} value={m.volume??0} style:--volume-fill={`${m.volume??0}%`} data-input="volume" aria-label={m.spotifyMode==='demo'?'Volume':'Spotify volume'} aria-valuetext={m.volume===null?'Unavailable':`${m.volume} percent${muted?', muted':''}`} />{#if stop}<button data-action="stop-music" disabled={!screen.canControlSpotify()} aria-label={m.spotifyMode==='demo'?'Stop music':'Pause Spotify'}><Icon name="square" /></button>{/if}</div>
