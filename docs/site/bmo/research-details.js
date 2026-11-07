// Source-backed detail views. Data is local; these links never invoke generation.
const cache = new Map();
const esc = value => String(value ?? 'Not recorded').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value).toLocaleString('en-US');
const anchor = (url, text) => `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(text)}</a>`;
const list = items => `<ul>${items.map(x => `<li>${esc(x)}</li>`).join('')}</ul>`;
const table = (heads, rows) => `<div class="detail-table"><table><thead><tr>${heads.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(v => `<td>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
async function load(name) {
  if (!cache.has(name)) cache.set(name, fetch(new URL(`./data/${name}.json`, import.meta.url)).then(r => {
    if (!r.ok) throw new Error('The local evidence file could not load.');
    return r.json();
  }).catch(error => { cache.delete(name); throw error; }));
  return cache.get(name);
}
function refs(data, ids) {
  const selected = data.sources.filter(s => ids.includes(s.id));
  return `<div class="source-links">${selected.map(s => anchor(s.url || new URL(s.site_path, new URL('./data/benchmark.json', import.meta.url)), s.title)).join('')}</div>`;
}
function sourceNames(data, id) {
  return ({airimonda:'airimonda composite',martinnavs:'martinnavs / CosyVoice2',system_speech:'Microsoft System.Speech',edge_speech:'Microsoft Edge Speech'})[id] || id;
}
function providerLinks(generator) {
  return `<div class="tts-reference-links">${generator.reference_links.map(item=>anchor(item.url,item.title)).join('')}</div>`;
}
function modelCitations(data) {
  return `<h3>TTS models and providers</h3>${data.generation_sources.map(generator=>`<article class="citation"><h3>${esc(generator.citation_title)}</h3><p>${esc(generator.citation_summary)}</p>${generator.models?`<ul class="citation-models">${generator.models.map(model=>`<li>${anchor(model.url,model.id)}<code>${esc(model.revision)}</code></li>`).join('')}</ul>`:''}${providerLinks(generator)}</article>`).join('')}`;
}
function distillation(data) {
  return `<h3>BMO training and distillation</h3><p>${esc(data.distillation.summary)}</p><p><code>${esc(data.distillation.objective)}</code></p>${refs(data,data.distillation.sources)}`;
}
function datasetLinks(data) {
  return `<p>${anchor(data.dataset.repository_url,'Hugging Face dataset repository: '+data.dataset.repository)}<br>${anchor(data.dataset.release_url,'Frozen shared-demo-a-v1 snapshot')} · ${anchor(data.dataset.url,'Dataset card and terms')}</p>`;
}
function generation(data) {
  const providers = data.generation_sources.map((g, index) => {
    let body = '';
    if (g.id === 'edge_speech') {
      body += `<p>${esc(g.prompt_recipe.formula)}. ${esc(g.counts.train)} TRAIN / ${esc(g.counts.dev)} DEV / ${esc(g.counts.test)} TEST.</p>`;
      body += table(['Partition','Exact provider voice IDs'], Object.entries(g.voice_ids).map(([k,v]) => [k.toUpperCase(),v.join(', ')]));
      body += table(['Form','Spoken template'], [['Clean',g.prompt_recipe.clean],...Object.entries(g.prompt_recipe.context_by_split).map(([k,v])=>[k.toUpperCase(),v])]);
      body += `<p>Rate: <code>${esc(g.rate)}</code>. Voice IDs identify provider presets; the service model revision, tool version, random seed, pitch and sampling settings were not recorded.</p><details><summary>All 16 negative prompts per voice</summary>${list(g.prompt_recipe.negative_prompts)}</details><p>${esc(g.regeneration)}</p>`;
    } else if (g.id === 'system_speech') {
      body += `<p>${esc(g.prompt_recipe.formula)}.</p>`;
      body += table(['Partition','Voice','Rate units'], [['TRAIN',g.voices.train,g.rate_by_split.train.join(', ')],['Original DEV',g.voices.dev,g.rate_by_split.dev.join(', ')]]);
      body += `<p>Four forms: ${g.prompt_recipe.wrappers.map(x=>`<code>${esc(x)}</code>`).join(' · ')}.</p><p>${esc(g.prompt_recipe.restriction)} ${esc(g.model_id_status)}</p>`;
    } else if (g.id === 'legacy_qwen') {
      body += `<p><strong>184 DEV-only recordings; zero TRAIN rows.</strong> ${esc(g.unknowns_reason)}</p>`;
      body += g.models.map(m => `<p>${anchor(m.url,m.id)}<br><code>${esc(m.revision)}</code></p>`).join('');
      body += `<p>${esc(g.do_not_substitute)} These are offline generation models, not pretrained VCM weights or runtime dependencies.</p>`;
    } else if (g.id === 'martinnavs') {
      body += `<p><strong>${number(g.baseline_a_train_rows)} TRAIN rows.</strong> ${esc(g.model_family)} from ${g.declared_reference_voices} declared reference IDs. Exact model ID, weights revision and sampling configuration are unrecorded.</p><p>${esc(g.generation_prompts)}</p><p>${esc(g.reference_prompt_description)} ${esc(g.seed_status)}</p><p>${esc(g.qa)}</p>`;
    } else {
      body += `<p>${number(g.counts.train)} TRAIN / ${number(g.counts.original_dev)} original DEV / ${number(g.counts.official)} official synthetic rows.</p><p>Inherited reference sources: ${esc(g.reference_sources.join(', '))}. ${esc(g.unknowns_reason)}</p>`;
    }
    return `<details class="method-section" ${index===0?'open':''}><summary>${esc(g.title)}</summary>${body}<p>${esc(g.split_policy)}</p>${g.terms?`<p class="detail-muted">${esc(g.terms)}</p>`:''}${providerLinks(g)}${refs(data,g.sources)}</details>`;
  }).join('');
  return `${datasetLinks(data)}${distillation(data)}<p><strong>25,958 TRAIN recordings → frozen 40-bin frontend → balanced sampling.</strong> Generation supplies recordings; the VCM itself starts from random weights.</p>
    ${table(['TRAIN source','Kind','Rows'],data.source_composition.filter(s=>s.split==='train').map(s=>[sourceNames(data,s.source_id),s.kind,number(s.rows)]))}
    <p>${esc(data.dataset.terms)}</p>
    <h3>How the authored data was made</h3><ol class="recipe-sequence"><li>Freeze typed targets and exact wordings.</li><li>Generate each complete clean or context-wrapped utterance with its declared voice.</li><li>Keep voice IDs separate across TRAIN, DEV and TEST; preserve negative prompts.</li><li>Store the original audio, source text and file/PCM hashes. Reproduce from these frozen bytes.</li></ol>
    ${providers}<h3>Treatment and split rules</h3>${list([data.treatment.reference_labels,data.treatment.normalization,data.treatment.split_policy,data.treatment.annotations])}
    <h3>Known provenance gaps</h3>${list(data.unknowns)}${refs(data,['dataset_card','dataset_lock','protocol','bibtex'])}
    <p>${anchor(new URL('./data/dataset-recipe.json',import.meta.url),'Full machine-readable recipe')} · ${anchor(new URL('./wordings.csv',import.meta.url),'All 102 canonical wordings (CSV)')}</p>`;
}
function citations(data) {
  const upstream = data.source_datasets.map(s=>`<article class="citation"><h3>${anchor(s.url,s.title)}</h3><p>${esc(s.author)} (${esc(s.year)}). <code>${esc(s.repository)}</code></p><p>${s.doi?anchor('https://doi.org/'+s.doi,'Dataset DOI: '+s.doi):esc(s.doi_status)}</p><p>${esc(s.terms)}</p>${refs(data,s.sources)}</article>`).join('');
  const bibliography = data.citations.map(c=>`<article class="citation"><h3>${anchor(c.url,c.title)}</h3><p>${esc(c.authors)} (${esc(c.year)}), ${esc(c.venue)}.</p><p>${esc(c.role)}</p><p>${c.doi?anchor('https://doi.org/'+c.doi,c.doi_role+': '+c.doi):esc(c.doi_status)}${c.dataset_url?' · '+anchor(c.dataset_url,'Dataset / publisher'):''}</p><p class="detail-muted">${esc(c.terms)}</p></article>`).join('');
  return `<p>Cite the original datasets and authors, plus the exact derivative revision used. An upstream DOI does not identify this derivative.</p><h3>Frozen Baseline A release</h3>${datasetLinks(data)}<p><code>${esc(data.dataset.publication_revision)}</code></p><p>${esc(data.dataset.doi_status)} ${esc(data.dataset.terms)}</p>${modelCitations(data)}${distillation(data)}${upstream}<h3>Inherited scholarly attribution</h3><p>These sources are credited by the preserved composite. Exact A-only per-source clip counts and every original clip/version mapping are not audited.</p>${bibliography}<h3>Reference attribution with incomplete metadata</h3><p>SilencioPH and the inherited AI231 contributor synthesis remain credited through the preserved source audit. No author list, publication DOI, clone consent or model revision is invented for missing metadata.</p>${refs(data,['inherited_citations','primary_card','supplement_notice','qwen_assets','bibtex'])}<p>${anchor(new URL('./data/dataset-recipe.json',import.meta.url),'Full source and citation record')}</p>`;
}
function pi5Replay(record) {
  const latest = record.runs.find(run => run.id === record.current_run_id);
  const labels = {'baseline-a-mnn':'A epoch38 · MNN CPU','baseline-a-onnx':'A epoch38 · ONNX CPU','epoch44-onnx':'Historical epoch44 · ONNX CPU'};
  const score = value => `${value.exact_n}/${value.scored_n} (${(100*value.complete_command_accuracy).toFixed(2)}%)`;
  const measured = run => table(['Configuration','Full historical score','Shared catalog only','p50 / p95, ms','Median RTF','Process RSS, MiB'],Object.keys(labels).map(name => {
    const row = run.cases[name], m = row.metrics, t = m.latency_ms.end_to_end;
    return [labels[name],score(m.accepted),score(row.shared_inventory.accepted),`${t.p50.toFixed(2)} / ${t.p95.toFixed(2)}`,m.rtf.p50.toFixed(5),m.peak_process_rss_mib.toFixed(2)];
  }));
  const parity = latest.baseline_backend_parity;
  return `<h3>Physical Raspberry Pi 5 · recorded replay</h3><p><strong>Measured 2026-11-07 on the real Pi 5 Model B.</strong> Two engine threads; 196 original recordings × 3 timing passes per configuration. Accuracy counts each recording once; two ambiguous references remain excluded. This is a separate historical exposed holdout, not a training split, the 642-recording producer regression set, or fresh live speech.</p><h3>Repeat 2 · ${esc(latest.id)}</h3>${measured(latest)}
    <p><strong>27 scored references are outside A's frozen inventory:</strong> named Kitchen/Bedroom lights, 20/60% brightness and alarms at 08:00/21:00. The full historical denominator remains 194. The separately labeled shared-catalog diagnostic keeps only exact references supported by both immutable inventories: 156 positives + 11 negatives = 167. No relabeling, retuning or denominator replacement.</p>
    <p>On shared positives, A matched 149/156; epoch44 matched 147/156. A falsely accepted 5/11 negatives versus epoch44's 1/11. This small negative set is different from the 5/19 human cohort above and excludes wake/owner gating; it is not an operational false-action rate.</p>
    ${record.runs.filter(run => run.id !== latest.id).map(run=>`<h3>Previous measured repeat · ${esc(run.id)}</h3>${measured(run)}`).join('')}
    <p>Both repeats retain identical recognition scores. ONNX/MNN typed-output differences: <strong>${parity.decision_mismatch_trials}/588</strong>; all 196 first-pass logit vectors meet the original combined absolute/relative tolerance. Timing varies with host load; these sequential runs left the UI running.</p>
    <p>Time covers PCM validation, frontend/normalization, inference and finite decision. It excludes audio duration/decoding, wake/VAD/owner waits, UI IPC, providers and TTS. RAM is whole benchmark-process peak RSS, not model-only RAM. No microphone, playback, training, device action or model replacement. <strong>Pi 4, power, live-microphone accuracy and acoustic qualification remain unmeasured.</strong></p>
    <p>${anchor(new URL('./data/pi5-replay.json',import.meta.url),'Both runs: raw/accepted counts, original references, decisions, timing samples and source hashes')}</p><p>The SpeakerUI checkout had disclosed local source changes; this is not a clean-base-commit reproduction claim.</p>`;
}
function benchmark(data, pi5) {
  const positive = data.accuracy.filter(r=>r.supported>0), negative=data.accuracy.filter(r=>r.negative_rows>0);
  const accuracy = table(['Cohort','Exact / supported','Rejected','Wrong accepted'],positive.map(r=>[r.label,`${r.correct}/${r.supported} (${(100*r.accuracy).toFixed(2)}%)`,r.rejected_supported,r.wrong_accepted]));
  const negatives = table(['Negative cohort','False accepts / negatives','Rate'],negative.map(r=>[r.label,`${r.negative_false_accepts}/${r.negative_rows}`,`${(100*r.negative_false_accept_rate).toFixed(2)}%`]));
  const comparison = table(['DEV model','Parameters','Epoch','Human','Slot','All demo'],data.comparison.rows.map(r=>[r.name,number(r.parameters),r.selected_epoch,`${r.human_dev_correct}/${r.human_dev_supported}`,`${r.slot_dev_correct}/${r.slot_dev_supported}`,`${r.all_demo_dev_correct}/${r.all_demo_dev_supported}`]));
  const latency=data.latency;
  return `<p><strong>Published Baseline A, epoch 38, 175,653 parameters.</strong> A command counts as correct only when intent and every required value match. The 4080 epoch-36 refit has no test evaluation.</p>
    <h3>Complete-command accuracy</h3>${accuracy}<p>Cohorts overlap: combined synthetic includes clean + context; official totals include human + synthetic. Do not add these rows.</p>
    <h3>Unknown and unsupported requests</h3>${negatives}<h3>Nine finite slot cells</h3>${table(['Command cell','Exact / supported','Rejected'],data.slot_cells.map(r=>[r.label,`${r.correct}/${r.supported}`,r.rejected_supported]))}<p>${esc(data.slot_summary.scope)}</p>
    <h3>Comparable-size control</h3><p>${esc(data.comparison.comparable_population)}</p>${comparison}${list(data.comparison.limitations)}
    <h3>Historical desktop VCM timing</h3><p><strong>${latency.p50_ms.toFixed(3)} ms p50 / ${latency.p95_ms.toFixed(3)} ms p95</strong> · ${latency.cpu_threads} CPU thread · ${esc(latency.runtime)}.</p><p>Includes ${esc(latency.includes.join(', '))}. Excludes ${esc(latency.excludes.join(', '))}.</p><p>CPU model: ${esc(latency.cpu_model_status)} ${esc(latency.timing_sample_count_status)} This is historical CPU evidence, not a new RTX 4080 inference or Raspberry Pi measurement.</p>
    ${pi5Replay(pi5)}
    <h3>Limits of the evidence</h3>${list(data.caveats)}${refs(data,['metrics','study_report','verification','selection','predictions'])}<p>${anchor(new URL('./data/benchmark.json',import.meta.url),'Full VCM benchmark record')} · ${anchor(new URL('./data/training-history.json',import.meta.url),'Separate RTX 4080 training receipt')}</p>`;
}
export async function researchPage(key) {
  if (key==='generation') return {title:'Dataset generation recipe',html:generation(await load('dataset-recipe'))};
  if (key==='citations') return {title:'Dataset citations and source terms',html:citations(await load('dataset-recipe'))};
  if (key==='pi5') return {title:'Pi 5 replay: measured processing, unchanged references',html:pi5Replay(await load('pi5-replay'))};
  if (key==='benchmark') return {title:'VCM benchmark: correctness, cost and controls',html:benchmark(await load('benchmark'),await load('pi5-replay'))};
  throw new Error('Unknown research detail.');
}
