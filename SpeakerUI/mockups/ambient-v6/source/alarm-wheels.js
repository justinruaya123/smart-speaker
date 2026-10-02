// Bounded scroll wheels with direct pointer dragging, keyboard and trackpad support.
export function mountAlarmWheels(container,{value,onChange,isMuted}){
 const [hour,minute]=value.split(':').map(Number);
 const values=[(hour%12||12)-1,minute,hour>=12?1:0];
 const columns=[{label:'Hour',items:Array.from({length:12},(_,i)=>String(i+1).padStart(2,'0')),min:1},{label:'Minute',items:Array.from({length:60},(_,i)=>String(i).padStart(2,'0')),min:0},{label:'AM or PM',items:['AM','PM'],min:0}];
 const rowHeight=44,cleanups=[];let sound=null,lastClick=0,destroyed=false;
 container.innerHTML=columns.map((column,k)=>`<div class="alarm-wheel" data-wheel="${k}" role="spinbutton" tabindex="0" aria-label="${column.label}" aria-valuemin="${column.min}" aria-valuemax="${column.items.length-1+column.min}" aria-valuenow="${values[k]+column.min}" aria-valuetext="${column.items[values[k]]}"><div class="wheel-spacer" aria-hidden="true"></div>${column.items.map((item,i)=>`<button class="wheel-value" tabindex="-1" data-wheel-index="${i}" aria-label="${item}">${item}</button>`).join('')}<div class="wheel-spacer" aria-hidden="true"></div></div>`).join('');
 function unlock(){if(isMuted()||destroyed)return;const Audio=globalThis.AudioContext||globalThis.webkitAudioContext;if(!Audio)return;if(!sound)sound=new Audio();if(sound.state==='suspended')sound.resume().catch(()=>{});}
 function click(){const now=performance.now();if(isMuted()||!sound||sound.state!=='running'||now-lastClick<32)return;lastClick=now;const oscillator=sound.createOscillator(),gain=sound.createGain();oscillator.type='triangle';oscillator.frequency.setValueAtTime(1250,sound.currentTime);gain.gain.setValueAtTime(.028,sound.currentTime);gain.gain.exponentialRampToValueAtTime(.001,sound.currentTime+.018);oscillator.connect(gain);gain.connect(sound.destination);oscillator.start();oscillator.stop(sound.currentTime+.02);oscillator.onended=()=>{oscillator.disconnect();gain.disconnect();};}
 const wheels=[...container.querySelectorAll('.alarm-wheel')];
 for(const [k,wheel] of wheels.entries()){
  const items=[...wheel.querySelectorAll('.wheel-value')];let frame=0,drag=null,suppressClickUntil=0;
  function paint(){frame=0;if(destroyed)return;const position=wheel.scrollTop/rowHeight,index=Math.max(0,Math.min(items.length-1,Math.round(position)));
   for(const [i,item] of items.entries()){const distance=i-position;item.style.transform=`perspective(280px) rotateX(${Math.max(-65,Math.min(65,-distance*24))}deg) scale(${Math.max(.76,1-Math.abs(distance)*.08)})`;item.style.opacity=String(Math.max(.08,1-Math.abs(distance)*.34));item.classList.toggle('selected',i===index);}
   wheel.setAttribute('aria-valuenow',String(index+columns[k].min));wheel.setAttribute('aria-valuetext',columns[k].items[index]);
   if(index!==values[k]){values[k]=index;click();const h=((values[0]+1)%12)+values[2]*12;onChange(String(h).padStart(2,'0')+':'+String(values[1]).padStart(2,'0'));}
  }
  function scroll(){if(!frame)frame=requestAnimationFrame(paint);}
  function move(index){wheel.scrollTo({top:Math.max(0,Math.min(items.length-1,index))*rowHeight,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}
  function key(event){if(!['ArrowUp','ArrowDown','PageUp','PageDown','Home','End'].includes(event.key))return;event.preventDefault();unlock();const index=values[k];move(event.key==='Home'?0:event.key==='End'?items.length-1:index+({ArrowUp:-1,ArrowDown:1,PageUp:-5,PageDown:5}[event.key]));}
  function choose(event){if(performance.now()<suppressClickUntil){event.preventDefault();event.stopImmediatePropagation();return;}const button=event.target.closest('[data-wheel-index]');if(button){unlock();move(Number(button.dataset.wheelIndex));}}
  function startDrag(event){if(!event.isPrimary||event.button!==0)return;unlock();wheel.focus({preventScroll:true});wheel.scrollTo({top:wheel.scrollTop,behavior:'instant'});drag={id:event.pointerId,y:event.clientY,top:wheel.scrollTop,moved:false};}
  function dragWheel(event){if(drag?.id!==event.pointerId)return;const dy=event.clientY-drag.y;if(!drag.moved&&Math.abs(dy)<4)return;if(!drag.moved){drag.moved=true;wheel.classList.add('wheel-dragging');wheel.setPointerCapture(event.pointerId);}event.preventDefault();wheel.scrollTop=Math.max(0,Math.min((items.length-1)*rowHeight,drag.top-dy));scroll();}
  function finishDrag(event){if(drag?.id!==event.pointerId)return;const gesture=drag;drag=null;if(!gesture.moved)return;event.preventDefault();suppressClickUntil=performance.now()+400;const index=Math.round((event.type==='pointercancel'?gesture.top:wheel.scrollTop)/rowHeight);if(wheel.hasPointerCapture(event.pointerId))wheel.releasePointerCapture(event.pointerId);wheel.classList.remove('wheel-dragging');move(index);}

  wheel.scrollTop=values[k]*rowHeight;paint();wheel.addEventListener('scroll',scroll,{passive:true});wheel.addEventListener('keydown',key);wheel.addEventListener('click',choose,true);wheel.addEventListener('pointerdown',startDrag);wheel.addEventListener('pointermove',dragWheel);wheel.addEventListener('pointerup',finishDrag);wheel.addEventListener('pointercancel',finishDrag);wheel.addEventListener('wheel',unlock,{passive:true});
  cleanups.push(()=>{cancelAnimationFrame(frame);wheel.removeEventListener('scroll',scroll);wheel.removeEventListener('keydown',key);wheel.removeEventListener('click',choose,true);wheel.removeEventListener('pointerdown',startDrag);wheel.removeEventListener('pointermove',dragWheel);wheel.removeEventListener('pointerup',finishDrag);wheel.removeEventListener('pointercancel',finishDrag);wheel.removeEventListener('wheel',unlock);});
 }
 return {destroy(){destroyed=true;for(const cleanup of cleanups)cleanup();if(sound)sound.close().catch(()=>{});}};
}
