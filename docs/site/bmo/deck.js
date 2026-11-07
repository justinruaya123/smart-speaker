import { renderLogMel, setLogMelTime } from './logmel.js';
import { createModelScene } from './model-scene.js';
import { sourcePages } from './sources.js';
import { researchPage } from './research-details.js';

const $ = (id) => document.getElementById(id);
const slides = [...document.querySelectorAll('.slide')];
const names = slides.map(s => s.dataset.slide);
const nav = [...document.querySelectorAll('nav[aria-label="Slides"] a')];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let currentSlide = 0, stage = 0, stageProgress = 0, playing = false, expanded = false;
let playFrame = 0, stageStart = 0, trace = null, history = null;
let updateTimers = [];
const model = createModelScene($('model-scene'), { onSelect: index => setStage(index, true) });

const stageContent = [
  ['Listen in small frames', '40 mel bins × 296 frames', 'A 2.976-second recording becomes 40 frequency bands, sampled every 10 ms.', '16,000 samples / second · mono'],
  ['Learn the first sound patterns', '40 × 296 → 128 × 148', 'A kernel-5 causal convolution turns 40 bands into 128 channels. LayerNorm and SiLU follow.', '25,856 learned parameters · time length halves'],
  ['Filter each channel. Then mix.', '128 channels × 148 frames', 'Two stages filter each channel, mix channels, then apply LayerNorm and SiLU.', '35,072 parameters · residual: (input + output) / √2'],
  ['Look farther back in time', '128 channels × 148 frames', 'Dilation 2 samples every second position. The block preserves shape and adds its input back.', 'Two depthwise + pointwise stages · kernel 7'],
  ['Build a wider sound context', '128 channels × 148 frames', 'Dilation 4 connects more distant sound patterns while preserving all 128 channels.', 'Left padding only · no future context inside convolutions'],
  ['Combine the longest context', '128 channels × 148 frames', 'Dilation 8 gives the widest context. All three heads share this encoder.', 'Four blocks × 35,072 parameters'],
  ['Summarize the whole utterance', '128 means + 128 maxima = 256', 'Mean and maximum pooling join into one 256-value vector for the three heads.', 'This pooling stage needs the complete utterance'],
  ['Choose the action', '29 action + 4 percent + 4 alarm', 'The model selects brightness at 50%. Missing required values reject the command.', '37 logits · fixed argmax · 9,509 head parameters']
];

function fit() {
  document.documentElement.style.setProperty('--scale', Math.min(innerWidth / 1600, innerHeight / 900));
}
function pauseModel() {
  playing = false; cancelAnimationFrame(playFrame);
  $('play-model').querySelector('span').textContent = expanded ? 'Play this block' : stage === 7 && stageProgress >= 1 ? 'Replay forward pass' : 'Play forward pass';
  $('play-model').setAttribute('aria-pressed', 'false');
}
function setStage(index, user = false, progress = 0) {
  if (user) pauseModel();
  stage = Math.max(0, Math.min(7, index)); stageProgress = progress;
  if (expanded && (stage < 2 || stage > 5)) toggleExpanded(false);
  const [title, shape, copy, extra] = stageContent[stage];
  $('stage-counter').textContent = String(stage + 1).padStart(2, '0');
  $('stage-title').textContent = title;
  $('stage-shape').textContent = shape;
  $('stage-copy').textContent = copy;
  $('stage-extra').textContent = extra;
  document.querySelectorAll('[data-stage]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.stage) === stage)));
  $('decision').hidden = stage !== 7;
  model.setStage(stage, progress);
}
function tick(now) {
  if (!playing) return;
  stageProgress = Math.min(1, (now - stageStart) / 2300);
  model.setStage(stage, reducedMotion.matches ? 0 : stageProgress);
  if (stageProgress >= 1) {
    if (stage === 7 || expanded) { pauseModel(); return; }
    setStage(stage + 1, false); stageStart = now;
  }
  playFrame = requestAnimationFrame(tick);
}
function playModel() {
  if (playing) { pauseModel(); return; }
  if (expanded && stageProgress >= 1) setStage(stage);
  if (!expanded && stage === 7) setStage(0);
  playing = true; stageStart = performance.now() - stageProgress * 2300;
  $('play-model').querySelector('span').textContent = 'Pause';
  $('play-model').setAttribute('aria-pressed', 'true');
  playFrame = requestAnimationFrame(tick);
}
function toggleExpanded(value = !expanded) {
  pauseModel();
  if (value && (stage < 2 || stage > 5)) setStage(2);
  expanded = value; model.setExpanded(value);
  $('expand-block').setAttribute('aria-pressed', String(value));
  $('expand-block').textContent = value ? 'Whole model' : 'Inside a block';
  $('play-model').querySelector('span').textContent = value ? 'Play this block' : 'Play forward pass';
  $('step-model').textContent = value ? 'Next block' : 'Step';
}
function stopUpdate() {
  updateTimers.forEach(clearTimeout); updateTimers = [];
  $('batch-grid').classList.remove('updating');
  document.querySelectorAll('.update-flow > span').forEach(el => el.classList.remove('active'));
  $('animate-update').disabled = false; $('animate-update').textContent = 'Animate one update';
}
function selectSlide(index, hash = true) {
  pauseModel(); stopUpdate();
  currentSlide = Math.max(0, Math.min(3, index));
  slides.forEach((s, i) => { s.hidden = i !== currentSlide; s.classList.toggle('active', i === currentSlide); });
  nav.forEach((a, i) => { if (i === currentSlide) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  $('page-counter').textContent = `${String(currentSlide + 1).padStart(2, '0')} / 04`;
  $('previous').disabled = currentSlide === 0;
  $('next').setAttribute('aria-label', currentSlide === 3 ? 'Return to first slide' : 'Next slide');
  document.title = `BMO — ${nav[currentSlide].textContent}`;
  if (hash) window.history.replaceState(null, '', '#' + names[currentSlide]);
  model.setActive(currentSlide === 0);
  if (innerWidth <= 760) scrollTo({ top: 0, behavior: 'instant' });
}
nav.forEach((a, i) => a.addEventListener('click', e => { e.preventDefault(); selectSlide(i); }));
document.querySelector('.wordmark').addEventListener('click', e => { e.preventDefault(); selectSlide(0); });
$('previous').addEventListener('click', () => selectSlide(currentSlide - 1));
$('next').addEventListener('click', () => selectSlide((currentSlide + 1) % 4));
window.addEventListener('hashchange', () => selectSlide(Math.max(0, names.indexOf(location.hash.slice(1))), false));
window.addEventListener('resize', fit);
document.querySelectorAll('[data-stage]').forEach(b => b.addEventListener('click', () => setStage(Number(b.dataset.stage), true)));
$('play-model').addEventListener('click', playModel);
$('step-model').addEventListener('click', () => setStage(expanded ? (stage === 5 ? 2 : stage + 1) : (stage + 1) % 8, true));
$('expand-block').addEventListener('click', () => toggleExpanded());
$('reset-view').addEventListener('click', () => model.resetView());
$('replay-audio').addEventListener('click', async () => {
  const audio = $('sample-audio');
  if (!audio.paused) { audio.pause(); return; }
  try { audio.currentTime = 0; await audio.play(); }
  catch { $('replay-audio').querySelector('span').textContent = 'Audio unavailable'; }
});
$('sample-audio').addEventListener('play', () => $('replay-audio').querySelector('span').textContent = 'Pause sample');
$('sample-audio').addEventListener('timeupdate', () => setLogMelTime($('logmel-plot'), $('sample-audio').currentTime, trace?.sample.duration_seconds));
$('sample-audio').addEventListener('ended', () => setLogMelTime($('logmel-plot'), 0, trace?.sample.duration_seconds));
['pause', 'ended'].forEach(event => $('sample-audio').addEventListener(event, () => $('replay-audio').querySelector('span').textContent = 'Hear the sample'));
$('notes-toggle').addEventListener('click', () => {
  const on = $('notes-toggle').getAttribute('aria-pressed') !== 'true';
  $('notes-toggle').setAttribute('aria-pressed', String(on));
  document.querySelectorAll('.presenter-note').forEach(n => n.hidden = !on);
});
async function fullscreen() {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
  catch { $('fullscreen').title = 'Fullscreen is unavailable here; open this page in your browser to present.'; }
}
$('fullscreen').addEventListener('click', fullscreen);
document.addEventListener('fullscreenchange', () => $('fullscreen').setAttribute('aria-label', document.fullscreenElement ? 'Exit fullscreen' : 'Enter fullscreen'));
document.addEventListener('keydown', e => {
  const dialog = document.querySelector('dialog[open]');
  if (e.key === 'Escape' && dialog) { e.preventDefault(); dialog.close(); return; }
  if (dialog || /input|select|textarea|canvas/i.test(e.target.tagName)) return;
  if (['ArrowRight', 'PageDown'].includes(e.key)) { e.preventDefault(); selectSlide((currentSlide + 1) % 4); }
  if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); selectSlide(currentSlide - 1); }
  if (e.key === 'Home') { e.preventDefault(); selectSlide(0); }
  if (e.key === 'End') { e.preventDefault(); selectSlide(3); }
  if (e.key.toLowerCase() === 'f') fullscreen();
  if (e.key === ' ' && currentSlide === 0 && !/button|a/i.test(e.target.tagName)) { e.preventDefault(); playModel(); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) { pauseModel(); $('sample-audio').pause(); model.setActive(false); } else model.setActive(currentSlide === 0); });

const partitions = {
  train: { total: 25958, human: 1889, synthetic: 24069, role: 'eligible training recordings', explanation: 'The 1,889 human TRAIN recordings set the frozen normalization.' },
  dev: { total: 2132, human: 256, synthetic: 1876, role: 'development recordings', explanation: 'DEV selects the checkpoint before TEST.' },
  test: { total: 440, human: 0, synthetic: 440, role: 'reserved synthetic recordings', explanation: 'TEST has 204 clean, 204 context-added and 32 negative recordings. It was previously evaluated.' }
};
function setPartition(name) {
  const p = partitions[name];
  $('corpus-count').textContent = p.total.toLocaleString('en-US');
  $('corpus-role').textContent = p.role;
  $('human-count').textContent = p.human.toLocaleString('en-US');
  $('synthetic-count').textContent = p.synthetic.toLocaleString('en-US');
  $('human-fill').style.width = `${p.human / p.total * 100}%`;
  $('composition').setAttribute('aria-label', `${p.human} human recordings and ${p.synthetic} synthetic recordings`);
  $('partition-explanation').textContent = p.explanation;
  document.querySelectorAll('[data-partition]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.partition === name)));
}
document.querySelectorAll('[data-partition]').forEach(b => b.addEventListener('click', () => setPartition(b.dataset.partition)));
function renderExamples() {
  const key = $('example-select').value;
  const all = window.BMO_WORDINGS || [];
  const predicate = key === 'brightness' ? w => w.intent === 'BRIGHTNESS' && w.command.slots?.percent === 50 : key === 'alarm' ? w => w.intent === 'ALARM' && (JSON.stringify(w.command).includes('12:30')) : w => w.intent === 'LIGHT_ON';
  const matching = all.filter(predicate);
  const selected = matching.length ? matching.slice(0, 3) : all.filter(w => w.intent === (key === 'alarm' ? 'ALARM' : key === 'brightness' ? 'BRIGHTNESS' : 'LIGHT_ON')).slice(0, 3);
  $('example-wordings').replaceChildren(...selected.map(w => { const p = document.createElement('p'); p.textContent = '“' + w.text + '”'; return p; }));
  $('example-json').textContent = selected[0] ? JSON.stringify(selected[0].command) : 'No wording available';
}
$('example-select').addEventListener('change', renderExamples);

const quotas = { human: [['human-positive', 'human positive', 24], ['human-negative', 'human negative', 8]], mixed: [['human-positive', 'human positive', 8], ['human-negative', 'human negative', 4], ['generic-positive', 'generic synthetic', 8], ['demo-positive', 'demo-slot synthetic', 8], ['synthetic-negative', 'synthetic negative', 4]] };
let batchPhase = null;
function showBatch(phase) {
  if (phase === batchPhase) return;
  stopUpdate(); batchPhase = phase;
  $('batch-title').textContent = phase === 'human' ? 'One human-only batch' : 'One mixed batch';
  const grid = $('batch-grid'), legend = $('batch-legend'); grid.replaceChildren(); legend.replaceChildren();
  let index = 0;
  quotas[phase].forEach(([css, label, count]) => {
    for (let i = 0; i < count; i++) { const tile = document.createElement('span'); tile.className = 'batch-tile ' + css; tile.style.setProperty('--i', index++); tile.title = label; tile.setAttribute('aria-hidden', 'true'); grid.append(tile); }
    const item = document.createElement('span'); const swatch = document.createElement('i'); swatch.className = 'swatch ' + css; item.append(swatch, document.createTextNode(`${count} ${label}`)); legend.append(item);
  });
  grid.setAttribute('aria-label', '32 examples: ' + quotas[phase].map(([, l, n]) => `${n} ${l}`).join(', '));
}
function svgElement(tag, attrs = {}, text = null) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v)); if (text !== null) el.textContent = text; return el;
}
function drawHistory(epoch) {
  if (!history) return;
  const svg = $('history-svg'); svg.replaceChildren();
  const x = e => 42 + (e - 1) / 47 * 651, y = loss => 187 - loss / 3 * 171;
  [0, 1, 2, 3].forEach(value => {
    svg.append(svgElement('line', { x1: 42, x2: 693, y1: y(value), y2: y(value), stroke: '#ffffff10', 'stroke-width': 1 }));
    svg.append(svgElement('text', { x: 26, y: y(value) + 4, fill: '#8d9ba2', stroke: 'none', 'font-size': 12, 'text-anchor': 'end' }, String(value)));
  });
  const phase = x(6.5);
  svg.append(svgElement('rect', { x: 42, y: 16, width: phase - 42, height: 171, fill: '#b7dde20b', stroke: 'none' }));
  svg.append(svgElement('line', { x1: phase, x2: phase, y1: 16, y2: 187, stroke: '#b7dde238', 'stroke-dasharray': '3 5', 'stroke-width': 1 }));
  svg.append(svgElement('text', { x: phase + 10, y: 27, fill: '#8d9ba2', stroke: 'none', 'font-size': 11 }, 'mixed phase'));
  const points = history.epochs.map(e => `${x(e.epoch)},${y(e.training_loss)}`).join(' ');
  svg.append(svgElement('polyline', { points, stroke: '#b7dde2', 'stroke-width': 2.1, fill: 'none', 'stroke-linejoin': 'round' }));
  [1, 6, 12, 24, 36, 48].forEach(value => svg.append(svgElement('text', { x: x(value), y: 212, fill: '#8d9ba2', stroke: 'none', 'font-size': 12, 'text-anchor': 'middle' }, String(value))));
  const chosen = history.epochs[35];
  svg.append(svgElement('line', { x1: x(36), x2: x(36), y1: 16, y2: 187, stroke: '#ebc88a70', 'stroke-dasharray': '3 5', 'stroke-width': 1 }));
  svg.append(svgElement('circle', { cx: x(36), cy: y(chosen.training_loss), r: 4, fill: '#ebc88a', stroke: '#090d11', 'stroke-width': 2 }));
  const selected = history.epochs[epoch - 1];
  svg.append(svgElement('line', { x1: x(epoch), x2: x(epoch), y1: 16, y2: 187, stroke: '#b7dde24d', 'stroke-width': 1 }));
  svg.append(svgElement('circle', { cx: x(epoch), cy: y(selected.training_loss), r: 5, fill: '#b7dde2', stroke: '#090d11', 'stroke-width': 2 }));
  svg.setAttribute('aria-label', `Measured training loss, epochs 1 to 48. Epoch ${epoch}: ${selected.training_loss.toFixed(4)}. Development selected epoch 36.`);
}
function setEpoch(value) {
  const epoch = Math.max(1, Math.min(48, Number(value)));
  $('epoch-slider').value = epoch; $('epoch-value').textContent = epoch;
  const human = epoch <= 6;
  $('epoch-phase').textContent = human ? 'Human-only foundation' : 'Mixed training';
  showBatch(human ? 'human' : 'mixed');
  const row = history?.epochs[epoch - 1];
  $('epoch-lr').textContent = row ? row.lr.toFixed(6) : '—';
  $('epoch-loss').textContent = row ? row.training_loss.toFixed(4) : '—';
  $('epoch-updates').textContent = (epoch * 96).toLocaleString('en-US');
  drawHistory(epoch);
}
$('epoch-slider').addEventListener('input', e => setEpoch(e.target.value));
$('history-svg').addEventListener('click', e => { const r = $('history-svg').getBoundingClientRect(); const px = (e.clientX - r.left) / r.width * 720; setEpoch(Math.round((px - 42) / 651 * 47 + 1)); });
$('animate-update').addEventListener('click', () => {
  stopUpdate(); $('animate-update').disabled = true; $('animate-update').textContent = 'Update in progress…';
  $('batch-grid').classList.add('updating'); $('batch-token').classList.add('active');
  updateTimers.push(setTimeout(() => { $('batch-token').classList.remove('active'); $('loss-token').classList.add('active'); }, 900));
  updateTimers.push(setTimeout(() => { $('loss-token').classList.remove('active'); $('weights-token').classList.add('active'); }, 1800));
  updateTimers.push(setTimeout(stopUpdate, 2900));
});

const results = [{ name: 'Human commands', correct: 71, n: 77, rejected: 3, wrong: 3, color: '#b7dde2', note: 'Previously exposed regression recordings' }, { name: 'Clean synthetic', correct: 203, n: 204, rejected: 1, wrong: 0, color: '#a2bcad', note: 'Exact supported wordings' }, { name: 'Added spoken context', correct: 98, n: 204, rejected: 105, wrong: 1, color: '#ebc88a', note: 'Separate context-added recordings' }];
function showResults(errors = false) {
  const container = $('result-bars'); container.replaceChildren();
  results.forEach(r => {
    const result = document.createElement('div'); result.className = 'result';
    const header = document.createElement('div'); header.className = 'result-head';
    const h = document.createElement('h3'); h.textContent = r.name;
    const number = document.createElement('strong'); number.append(document.createTextNode(String(errors ? r.n - r.correct : r.correct) + ' '));
    const denominator = document.createElement('span'); denominator.textContent = '/ ' + r.n; number.append(denominator); header.append(h, number);
    const bar = document.createElement('div'); bar.className = 'result-bar'; const fill = document.createElement('span'); fill.style.width = `${(errors ? r.n - r.correct : r.correct) / r.n * 100}%`; fill.style.setProperty('--bar', errors ? '#d59780' : r.color); bar.append(fill);
    const note = document.createElement('p'); note.textContent = errors ? `${r.rejected} rejected · ${r.wrong} wrong accepted` : `${(r.correct / r.n * 100).toFixed(2)}% · ${r.note}`;
    result.append(header, bar, note); container.append(result);
  });
  $('result-mode').textContent = errors ? 'Show exact matches' : 'Show errors'; $('result-mode').setAttribute('aria-pressed', String(errors));
}
$('result-mode').addEventListener('click', () => showResults($('result-mode').getAttribute('aria-pressed') !== 'true'));

let sourceRequest = 0;
document.querySelectorAll('[data-source]').forEach(button => button.addEventListener('click', async () => {
  pauseModel(); const key=button.dataset.source, request=++sourceRequest;
  $('source-title').textContent = 'Source-backed details'; $('source-content').textContent = 'Loading local evidence…';
  $('source-dialog').showModal(); document.body.classList.add('no-scroll');
  try {
    const page = sourcePages[key] || await researchPage(key);
    if (request !== sourceRequest || !$('source-dialog').open) return;
    $('source-title').textContent = page.title; $('source-content').innerHTML = page.html;
    $('source-dialog').scrollTop = 0;
  } catch { $('source-content').textContent = 'The local evidence could not load. Close this panel and try again.'; }
}));
document.querySelectorAll('dialog').forEach(dialog => {
  dialog.querySelector('.close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => document.body.classList.remove('no-scroll'));
  dialog.addEventListener('click', e => { if (e.target === dialog) { const box = dialog.getBoundingClientRect(); if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom) dialog.close(); } });
});
function renderWordings() {
  const query = $('wording-search').value.toLowerCase().trim(), all = window.BMO_WORDINGS || [];
  const filtered = all.filter(w => [w.text, w.intent, w.spoken, JSON.stringify(w.command)].join(' ').toLowerCase().includes(query));
  $('wording-count').textContent = `${filtered.length} of ${all.length} wordings`; $('wording-list').replaceChildren();
  filtered.forEach(w => {
    const item = document.createElement('div'); item.className = 'wording-item'; const num = document.createElement('span'); num.textContent = String(w.number).padStart(2, '0');
    const content = document.createElement('div'); const p = document.createElement('p'); p.textContent = w.text; const label = document.createElement('small'); label.textContent = w.intent; const code = document.createElement('code'); code.textContent = JSON.stringify(w.command); content.append(p, label, code); item.append(num, content); $('wording-list').append(item);
  });
  if (!filtered.length) $('wording-list').textContent = 'No matching wordings. Try another command or intent.';
}
$('all-wordings').addEventListener('click', () => { $('wording-search').value = ''; renderWordings(); $('wording-dialog').showModal(); document.body.classList.add('no-scroll'); $('wording-search').focus(); });
$('wording-search').addEventListener('input', renderWordings);

fit(); setStage(0); setPartition('train'); renderExamples(); setEpoch(36); showResults();
selectSlide(Math.max(0, names.indexOf(location.hash.slice(1))), false);
fetch(new URL('./data/trace.json', import.meta.url)).then(r => { if (!r.ok) throw new Error('Trace unavailable'); return r.json(); }).then(data => {
  trace = data; model.setTrace(trace); renderLogMel($('logmel-plot'), trace);
  $('trace-status').textContent = '37 logits match. Replay is slowed.';
}).catch(() => { $('trace-status').textContent = 'Recorded activations could not load. Geometry still describes the model; reload this page to retry.'; });
fetch(new URL('./data/training-history.json', import.meta.url)).then(r => { if (!r.ok) throw new Error('History unavailable'); return r.json(); }).then(data => { history = data; setEpoch(36); }).catch(() => { $('history-chart').textContent = 'The recorded training history could not load. Reload this page to retry.'; });
