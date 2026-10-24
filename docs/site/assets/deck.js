/* Iris static publication. Only same-origin static fetches; never requests a provider. */
'use strict';
(() => {
  const $=selector=>document.querySelector(selector);
  const $$=selector=>[...document.querySelectorAll(selector)];
  const chapters=$$('[data-chapter]');
  const storage={get(key){try{return localStorage.getItem(key);}catch{return null;}},set(key,value){try{localStorage.setItem(key,value);}catch{}}};
  const systemMotion=matchMedia('(prefers-reduced-motion: reduce)');
  let motionOff=systemMotion.matches || storage.get('iris-motion-off')==='true';
  let current=0, data=null;
  function fit(){document.documentElement.style.setProperty('--scale',Math.min(innerWidth/1920,innerHeight/1080));}
  function stopAnimations(){
    $$('video').forEach(v=>{v.pause();v.hidden=true;});
    $$('iframe').forEach(f=>{f.removeAttribute('src');f.hidden=true;});
    $$('.poster').forEach(p=>p.hidden=false);
  }
  function motion(){
    $('#motion').setAttribute('aria-pressed',String(motionOff));
    $('#motion').textContent=motionOff?'Motion off':'Motion available';
    $$('[data-play],[data-inspect]').forEach(b=>b.disabled=motionOff);
    if(motionOff)stopAnimations();
  }
  function go(index,update=true){
    current=Math.max(0,Math.min(chapters.length-1,index));stopAnimations();
    chapters.forEach((s,i)=>{s.hidden=i!==current;});
    $$('[data-go]').forEach(a=>{if(a.dataset.go===chapters[current].id)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
    $('#chapter-count').textContent=`${String(current+1).padStart(2,'0')} / ${String(chapters.length).padStart(2,'0')}`;
    $('#previous').disabled=current===0;$('#next').disabled=current===chapters.length-1;
    storage.set('iris-chapter',chapters[current].id);
    if(update)history.replaceState(null,'',`#${chapters[current].id}`);
  }
  function fromHash(){
    const id=location.hash.slice(1)||storage.get('iris-chapter');
    const index=chapters.findIndex(s=>s.id===id);go(index<0?0:index);
  }
  async function fullscreen(){
    try{if(document.fullscreenElement)await document.exitFullscreen();else await $('#stage').requestFullscreen();}
    catch{$('#fullscreen').textContent='Fullscreen unavailable';}
  }
  function text(tag,value,className){const el=document.createElement(tag);el.textContent=String(value);if(className)el.className=className;return el;}
  function row(table,values){const tr=document.createElement('tr');values.forEach(v=>tr.append(text('td',v)));$(`${table} tbody`).append(tr);}
  const number=value=>Number(value).toLocaleString('en-US');
  function caseSelect(index,focus=false){
    const item=data.use_cases[index];
    $$('[role=tab]').forEach((b,i)=>{b.setAttribute('aria-selected',String(i===index));b.tabIndex=i===index?0:-1;});
    $('#case-panel').setAttribute('aria-labelledby',`case-tab-${index}`);
    $('#case-id').textContent=item.id;$('#case-title').textContent=item.title;
    $('#case-count').textContent=`${item.class_indices.length} complete command${item.class_indices.length===1?'':'s'}`;
    $('#case-wording').textContent=item.examples.map(e=>`“${e.wording}”`).join(' / ');
    $('#case-diagram').replaceChildren(...item.diagram.map(step=>text('span',step,'flow-step')));
    const commands=item.examples.map(e=>e.command);
    $('#case-json').textContent=commands.length===1?`{\n  "intent": ${JSON.stringify(commands[0].intent)},\n  "slots": ${JSON.stringify(commands[0].slots)}\n}`:`[\n${commands.map(command=>'  '+JSON.stringify(command)).join(',\n')}\n]`;
    $('#case-source').textContent=item.output.badge;
    $('#case-output').textContent=item.output.headline;
    $('#case-detail').textContent=item.output.detail;
    $('#case-panel').dataset.source=item.output.source;
    if(focus)$(`#case-tab-${index}`).focus();
  }
  function useCases(){
    data.use_cases.forEach((item,index)=>{
      const button=text('button',`${String(index+1).padStart(2,'0')}  ${item.title}`);
      button.type='button';button.id=`case-tab-${index}`;button.setAttribute('role','tab');button.setAttribute('aria-controls','case-panel');
      button.addEventListener('click',()=>caseSelect(index));
      button.addEventListener('keydown',e=>{
        const keys={ArrowDown:1,ArrowRight:1,ArrowUp:-1,ArrowLeft:-1};
        if(e.key in keys){e.preventDefault();e.stopPropagation();caseSelect((index+keys[e.key]+10)%10,true);}
        if(e.key==='Home'||e.key==='End'){e.preventDefault();e.stopPropagation();caseSelect(e.key==='Home'?0:9,true);}
      });
      $('#case-tabs').append(button);
      row('#use-summary',[`${item.id} ${item.title}`,item.class_indices.length,item.output.badge]);
    });caseSelect(0);
  }
  function chart(){
    const svg=$('#loss-chart'), ns='http://www.w3.org/2000/svg', points=data.training.curve;
    const max=Math.max(...points.map(r=>r.loss)), x=e=>50+(e-1)/47*625, y=l=>170-l/max*140;
    function add(name,attrs,content){const e=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(content)e.textContent=content;svg.append(e);return e;}
    add('path',{d:'M50 20 V170 H675',fill:'none',stroke:'#30444e','stroke-width':2});
    add('polyline',{points:points.map(r=>`${x(r.epoch)},${y(r.loss)}`).join(' '),fill:'none',stroke:'#9ce8d3','stroke-width':3});
    add('line',{x1:x(44),x2:x(44),y1:10,y2:170,stroke:'#96adb7','stroke-dasharray':'5 7'});
    add('text',{x:50,y:207,class:'chart-label'},'epoch 1');add('text',{x:570,y:207,class:'chart-label'},'48');
    add('text',{x:x(44)-60,y:24,class:'chart-label'},'44');
    add('text',{x:0,y:32,class:'chart-label'},max.toFixed(1));add('text',{x:17,y:173,class:'chart-label'},'0');
  }
  function benchmark(container,run,compact=false){
    const box=document.createElement('div');box.className='benchmark-run'+(compact?' compact':'');
    box.append(text('p',run.label||'Current desktop receipt','run-label'));
    box.append(text('p',`${run.latency.p50_ms.toFixed(2)} / ${run.latency.p95_ms.toFixed(2)} ms`,'latency'));
    if(!compact&&!run.measurement_not_quiesced)box.append(text('p',`p50 / p95 · ${number(run.original_recordings)} originals · ${number(run.timed_calls)} calls`));
    if(!compact&&!run.measurement_not_quiesced&&run.hardware)box.append(text('p',`${run.hardware.cpu.replace('11th Gen Intel(R) Core(TM) ','')} · CPU / one thread`));
    if(!compact&&run.measurement_not_quiesced)box.append(text('p',run.independent_parity?'Direct VCM only · not quiesced; gold parity verified.':'Not new adapter / idle-host / end-to-end.'));
    container.append(box);
  }
  function fill(){
    const wake=data.wake;
    $('.intro .chapter-foot').textContent=`Wake smoke gate FAILED: native ${wake.baseline.detected}/${wake.baseline.positive_n} positives, ${wake.baseline.false_wakes}/${wake.baseline.negative_n} false wakes; candidate ${wake.companion_test.detected}/${wake.companion_test.positive_n}, ${wake.companion_test.false_wakes}/${wake.companion_test.negative_n}. No replacement promoted. Not an acoustic release pass.`;
    $('#hero-badges').replaceChildren(...[`${number(data.model.parameters)} VCM parameters`,`${data.model.command_count} commands`, 'PC measured'].map(v=>text('span',v)));
    const facts=[['Parameters',number(data.model.parameters)],['Graph',(data.model.graph_bytes/1e6).toFixed(2)+' MB'],['Epoch / dtype',`${data.model.selected_epoch} / ${data.model.precision}`],['Input','16 kHz / 40 mel'],['Policy','p ≥ .50 / Δ ≥ .10']];
    for(const[k,v]of facts){const line=document.createElement('div');line.append(text('dt',k),text('dd',v));$('#model-facts').append(line);}
    $('#companion-foot').textContent=data.companions.map(c=>`${c.component}: ${c.parameters?number(c.parameters)+' parameters / ':''}${(c.graph_bytes/1e6).toFixed(2)} MB graph`).join(' · ');
    for(const split of ['train','dev','calibration','test','holdout'])row('#split-table',[split.toUpperCase(),number(data.datasets.splits[split].human),number(data.datasets.splits[split].synthetic)]);
    $('#recipe-line').textContent=`Seed ${data.training.recipe.seed} · 6 + 42 epochs · AdamW · ${number(data.training.optimizer_steps)} updates · ${data.training.wall_seconds.toFixed(2)} s. Normalizer: ${number(data.datasets.normalization_human_train)} human TRAIN recordings.`;
    for(const split of ['holdout','test']){const c=data.cohorts[split];row('#cohort-table',[split,`${c.selected.correct_accepted}/${c.selected.positives}`,c.selected.wrong_accepted,`${c.selected.negative_false_accepts}/${c.selected.negatives}`]);}
    const current=data.benchmarks.desktop;
    if(current){
      benchmark($('#desktop-results'),current);
      benchmark($('#desktop-results'),data.benchmarks.reference_desktop,true);
      const history=document.createElement('div');history.className='history-compact';
      data.benchmarks.historical_desktop.forEach((r,i)=>history.append(text('p',`${i===0?'Historical original':'Historical repeat v2'}: ${r.latency.p50_ms.toFixed(2)} / ${r.latency.p95_ms.toFixed(2)} ms`)));
      $('#desktop-results').append(history);
    }else{
      benchmark($('#desktop-results'),data.benchmarks.reference_desktop);
      data.benchmarks.historical_desktop.forEach(r=>benchmark($('#desktop-results'),r,true));
    }
    if(data.benchmarks.pi)$('#pi-result').textContent=`${data.benchmarks.pi.latency.p50_ms.toFixed(2)} / ${data.benchmarks.pi.latency.p95_ms.toFixed(2)} ms · ${data.benchmarks.pi.hardware.device}`;
    const reference=data.benchmarks.reference_desktop;
    const environment=current?`Product: Windows 11 / Python ${current.environment.python} / NumPy ${current.environment.numpy} / ORT ${current.environment.onnxruntime} · peak process RSS ${current.peak_process_rss_mib.toFixed(2)} MiB.`:`Reference PC: i7-11700 / Windows 11 / Python ${reference.hardware.python}.`;
    $('#evidence-foot').textContent=`${environment} One thread · ${number(reference.original_recordings)} originals × ${reference.repeats} = ${number(reference.timed_calls)} calls. Excludes speech/decode/wake/VAD/owner/UI/audio/providers. Ambiguous 187 unscored; exact wording 58/62 and 173/232; 21 definitions lack exact human references.`;
    chart();useCases();
  }
  $$('[data-go]').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();go(chapters.findIndex(s=>s.id===a.dataset.go));}));
  $('#previous').addEventListener('click',()=>go(current-1));$('#next').addEventListener('click',()=>go(current+1));
  $('#fullscreen').addEventListener('click',fullscreen);
  $('#motion').addEventListener('click',()=>{motionOff=!motionOff;storage.set('iris-motion-off',String(motionOff));motion();});
  systemMotion.addEventListener('change',e=>{if(e.matches){motionOff=true;motion();}});
  $$('.animation').forEach(host=>{
    const poster=host.querySelector('.poster'),video=host.querySelector('video'),frame=host.querySelector('iframe');
    host.querySelector('[data-play]').addEventListener('click',async()=>{
      if(motionOff)return;stopAnimations();poster.hidden=true;video.hidden=false;
      try{await video.play();}catch{video.hidden=true;poster.hidden=false;}
    });
    host.querySelector('[data-inspect]').addEventListener('click',()=>{
      if(motionOff)return;stopAnimations();poster.hidden=true;frame.hidden=false;frame.src=frame.dataset.src;
    });
    host.querySelector('[data-still]').addEventListener('click',stopAnimations);
  });
  document.addEventListener('keydown',e=>{
    if(e.defaultPrevented||e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,[role=tab],video'))return;
    if(['ArrowRight','PageDown'].includes(e.key)){e.preventDefault();go(current+1);}
    if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(current-1);}
    if(e.key==='Home'){e.preventDefault();go(0);}if(e.key==='End'){e.preventDefault();go(chapters.length-1);}
    if(/^[1-6]$/.test(e.key))go(Number(e.key)-1);
    if(e.key.toLowerCase()==='f')void fullscreen();
  });
  addEventListener('hashchange',fromHash);addEventListener('resize',fit);fit();motion();fromHash();
  fetch('assets/iris.json').then(response=>{if(!response.ok)throw new Error('Publication data unavailable');return response.json();}).then(value=>{
    if(value.schema_version!==1||value.product.id!=='iris-epoch44'||value.model.parameters!==258912)throw new Error('Unsupported publication data');
    data=value;fill();document.documentElement.dataset.ready='true';
  }).catch(error=>{$('#load-error').hidden=false;$('#load-error').textContent=`${error.message}. Serve this folder with a static HTTP server; see the repository README.`;});
})();
