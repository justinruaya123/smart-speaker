import { AURA_FRAGMENT } from './livekit-aura-shader.js';

// State target values ported from LiveKit's useAgentAudioVisualizerAura.
// Motion's React bindings are replaced by a single bounded WebGL render loop.
export const AURA_STATES = Object.freeze({
  idle:[10,.2,1.2,.4,1], failed:[10,.2,1.2,.4,1], disconnected:[10,.2,1.2,.4,1],
  listening:[20,.3,1,.7,1.75], 'pre-connect-buffering':[20,.3,1,.7,1.75],
  thinking:[30,.3,.5,1,1.5], connecting:[30,.3,.5,1,1.5], initializing:[30,.3,.5,1,1.5],
  speaking:[70,.3,.75,1.25,1.5]
});

/** Creates one renderer. The owner supplies normalized audio level and calls destroy on unmount. */
export function createAuraRenderer(canvas, {onFailure = () => {}} = {}) {
  const gl=canvas.getContext('webgl',{alpha:true,antialias:false,depth:false,stencil:false,premultipliedAlpha:false,powerPreference:'low-power'});
  if(!gl){onFailure('WebGL unavailable');return {update(){},destroy(){}};}
  let program,buffer;
  function compile(type,source){const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){const error=gl.getShaderInfoLog(shader);gl.deleteShader(shader);throw new Error(error);}return shader;}
  const vertex='attribute vec2 position; void main(){gl_Position=vec4(position,0.,1.);}';
  const uniforms=`precision mediump float; uniform vec3 iResolution; uniform float iTime; uniform float uSpeed,uBlur,uScale,uShape,uFrequency,uAmplitude,uBloom,uMix,uSpacing,uColorShift,uVariance,uSmoothing,uMode;uniform vec3 uColor;`;
  try {
    program=gl.createProgram();const vs=compile(gl.VERTEX_SHADER,vertex);const fs=compile(gl.FRAGMENT_SHADER,uniforms+AURA_FRAGMENT+'\nvoid main(){mainImage(gl_FragColor,gl_FragCoord.xy);}');
    gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
    const pos=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,2,gl.FLOAT,false,0,0);
  } catch(error){if(program)gl.deleteProgram(program);if(buffer)gl.deleteBuffer(buffer);onFailure(String(error));return {update(){},destroy(){}};}
  const locations=Object.fromEntries(['iResolution','iTime','uSpeed','uBlur','uScale','uShape','uFrequency','uAmplitude','uBloom','uMix','uSpacing','uColorShift','uVariance','uSmoothing','uMode','uColor'].map(n=>[n,gl.getUniformLocation(program,n)]));
  let state='idle',volume=0,muted=false,visible=true,destroyed=false,lost=false,raf=0,lastDraw=0,time=3,transitionAt=0;
  let color=[.38,.70,.82],current=[...AURA_STATES.idle];
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  function render(now){raf=0;if(destroyed||lost||!visible||document.hidden)return;
    const moving=!reduced.matches&&!muted&& !['idle','failed','disconnected'].includes(state);
    const settling=now-transitionAt<600;
    const interval=1000/(state==='speaking'?24:20);
    if(now-lastDraw>=interval||lastDraw===0){
      const dt=Math.min((now-lastDraw)/1000,.1);lastDraw=now;
      const target=[...AURA_STATES[state]];if(state==='speaking'&&volume>0)target[1]=.2+.2*volume;
      if(!reduced.matches&&['listening','pre-connect-buffering'].includes(state))target[4]=1.75+.25*Math.sin(now*Math.PI/350);
      if(!reduced.matches&&['thinking','connecting','initializing'].includes(state))target[4]=1.5+Math.sin(now*Math.PI/350);
      const mix=reduced.matches?1:1-Math.exp(-dt*9);current=current.map((v,i)=>v+(target[i]-v)*mix);if(moving)time+=dt;
      const max=state==='speaking'?320:288; if(canvas.width!==max){canvas.width=max;canvas.height=max;gl.viewport(0,0,max,max);}
      gl.uniform3f(locations.iResolution,max,max,1);gl.uniform1f(locations.iTime,time);
      const values={uSpeed:current[0],uScale:current[1],uAmplitude:current[2],uFrequency:current[3],uMix:muted?.38:current[4],uBlur:.08,uShape:1,uBloom:0,uSpacing:.5,uColorShift:.55,uVariance:.1,uSmoothing:.18,uMode:0};
      for(const [k,v] of Object.entries(values))gl.uniform1f(locations[k],v);
      const rgb=muted?[.5,.58,.62]:state==='failed'?[.93,.61,.3]:color;gl.uniform3fv(locations.uColor,rgb);gl.drawArrays(gl.TRIANGLES,0,6);
      canvas.dataset.renderer='livekit-webgl';canvas.dataset.state=state;
    }
    if(moving||settling)raf=requestAnimationFrame(render);
  }
  function request(){if(!raf&&!destroyed&&!lost&&visible&&!document.hidden)raf=requestAnimationFrame(render);}
  function visibility(){if(document.hidden){cancelAnimationFrame(raf);raf=0;}else{transitionAt=performance.now();request();}}
  function contextLost(event){event.preventDefault();lost=true;cancelAnimationFrame(raf);raf=0;onFailure('WebGL context lost');}
  const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)request();else{cancelAnimationFrame(raf);raf=0;}});observer.observe(canvas);
  canvas.addEventListener('webglcontextlost',contextLost);document.addEventListener('visibilitychange',visibility);reduced.addEventListener('change',request);
  request();
  return {
    update(next){if(reduced.matches&&next.state===undefined&&next.muted===undefined&&!next.color)return;if(next.state!==undefined){if(!(next.state in AURA_STATES))throw new Error('Unknown LiveKit state: '+next.state);if(next.state!==state){state=next.state;transitionAt=performance.now();}}if(next.volume!==undefined)volume=Math.max(0,Math.min(1,next.volume));if(next.muted!==undefined)muted=next.muted;if(next.color)color=next.color;request();},
    destroy(){destroyed=true;cancelAnimationFrame(raf);observer.disconnect();document.removeEventListener('visibilitychange',visibility);reduced.removeEventListener('change',request);canvas.removeEventListener('webglcontextlost',contextLost);gl.deleteBuffer(buffer);gl.deleteProgram(program);}
  };
}
