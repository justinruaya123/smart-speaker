import { lightCommand, spotifyCommand, type LightCommand, type SpotifyCommand } from '../integrations/commands.js';
import { IntegrationError } from '../integrations/http.js';

type Bindings={lights:Record<string,string>;songs:Record<string,string>};
type Providers={light:(c:LightCommand)=>Promise<unknown>;spotify:(c:SpotifyCommand)=>Promise<unknown>;song:(uri:string)=>Promise<unknown>;volume?:()=>Promise<number|undefined>};
const colors:Record<string,number[]>={red:[255,0,0],blue:[0,0,255],green:[0,255,0],yellow:[255,255,0],white:[255,255,255],orange:[255,165,0],purple:[128,0,128],pink:[255,192,203],cyan:[0,255,255],magenta:[255,0,255]};

/** One dispatcher per server, independent of how many browsers subscribe. */
export class RuntimeActions {
  private seen=new Set<string>();
  constructor(private config:{enabled:boolean;bindings:Bindings},private providers:Providers){}
  async handle(event:Record<string,any>):Promise<Record<string,any>|undefined>{
    if(event.event!=='command')return;
    const id=event.session_id+':'+event.utterance_id;
    if(!event.utterance_id||this.seen.has(id))return;
    this.seen.add(id);
    // Worker utterance IDs are unique. Bound memory without allowing old sessions to replay through the API.
    if(this.seen.size>10000)this.seen.delete(this.seen.values().next().value!);
    const decision=event.application;
    const outcome=(status:string,detail:string,extra:Record<string,any>={})=>({status,detail,utterance_id:event.utterance_id,...extra});
    if(event.source!=='microphone')return outcome('preview','Recorded WAVs and fixtures do not execute actions.');
    if(!this.config.enabled)return outcome('disabled','Actions are disabled in configuration.');
    if(!decision?.accepted||!decision.command)return outcome('rejected',decision?.reason||'No accepted application command');
    const command=decision.command,slots=command.slots;
    if(!slots||typeof slots!=='object'||Array.isArray(slots))return outcome('rejected','Invalid command slots');
    try{
      if(['LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR'].includes(command.intent)){
        const entityId=this.config.bindings.lights[slots.light_id];
        if(!entityId)return outcome('not_configured','Map '+slots.light_id+' in config.toml [bindings.lights].');
        let action:unknown={entityId,action:command.intent==='LIGHT_ON'?'on':'off'};
        if(command.intent==='BRIGHTNESS')action={entityId,action:'brightness',value:slots.percent};
        if(command.intent==='COLOR')action=slots.color==='warm white'||slots.color==='cool white'
          ?{entityId,action:'temperature',value:slots.color==='warm white'?2700:6500}
          :{entityId,action:'color',rgb:colors[slots.color]};
        await this.providers.light(lightCommand(action));
        return outcome('succeeded','Home Assistant accepted the command.');
      }
      if(['PLAY_MUSIC','PAUSE','STOP','NEXT','VOLUME'].includes(command.intent)){
        if(command.intent==='PLAY_MUSIC'&&slots.song_id){
          const uri=this.config.bindings.songs[slots.song_id];
          if(!uri)return outcome('not_configured','Map '+slots.song_id+' in config.toml [bindings.songs].');
          await this.providers.song(uri);
        }else{
          let action:unknown={action:({PLAY_MUSIC:'play',PAUSE:'pause',STOP:'pause',NEXT:'next'} as Record<string,string>)[command.intent]};
          if(command.intent==='VOLUME'){
            let value=slots.operation==='mute'?0:slots.percent;
            if(slots.operation==='up'||slots.operation==='down'){
              const current=await this.providers.volume?.();
              if(current===undefined)return outcome('not_configured','Current playback volume is unavailable.');
              value=Math.max(0,Math.min(100,current+(slots.operation==='up'?10:-10)));
            }
            action={action:'volume',value};
          }
          await this.providers.spotify(spotifyCommand(action));
        }
        return outcome('succeeded','Spotify accepted the command.');
      }
      if(command.intent==='TIMER'){
        if(!Number.isInteger(slots.duration_seconds)||slots.duration_seconds<1||slots.duration_seconds>86400)return outcome('rejected','Invalid timer duration');
        return outcome('local','Timer started.',{command,deadline:Date.now()+slots.duration_seconds*1000});
      }
      if(command.intent==='ALARM'){
        if(!/^([01]\d|2[0-3]):[0-5]\d$/.test(slots.time)||slots.timezone!=='Asia/Manila')return outcome('rejected','Invalid alarm');
        // Model schema fixes Asia/Manila (UTC+8, no daylight-saving time).
        const now=Date.now(),day=Math.floor((now+8*3600000)/86400000)*86400000-8*3600000;
        const [h,m]=slots.time.split(':').map(Number);let deadline=day+(h*60+m)*60000;
        if(deadline<=now)deadline+=86400000;
        return outcome('local','Alarm set.',{command,deadline});
      }
      if(['TIME','LIST_REMINDERS','CREATE_REMINDER'].includes(command.intent))return outcome('local','Local command ready.',{command});
      return outcome('unsupported','No action implementation for '+command.intent+'.');
    }catch(error){
      if(error instanceof IntegrationError)return outcome(error.outcomeUnknown?'unknown':error.code==='disabled'?'not_configured':'failed',error.message);
      return outcome('unknown','Action outcome could not be confirmed; it will not be retried.');
    }
  }
}
