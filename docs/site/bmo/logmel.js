// Display the preserved log-mel bin means; never synthesize signal data.
const NS = 'http://www.w3.org/2000/svg';
const plot = { left: 29, top: 2, width: 439, height: 52 };

function element(tag, attributes, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function energyColor(value, minimum, maximum) {
  const level = Math.max(0, Math.min(1, (value - minimum) / (maximum - minimum || 1)));
  const dark = [14, 24, 32], middle = [113, 158, 170], light = [235, 200, 138];
  const first = level < 0.6 ? dark : middle, last = level < 0.6 ? middle : light;
  const blend = level < 0.6 ? level / 0.6 : (level - 0.6) / 0.4;
  return `rgb(${first.map((channel, i) => Math.round(channel + (last[i] - channel) * blend)).join(',')})`;
}

export function renderLogMel(svg, trace) {
  const stage = trace.stages.find(item => item.id === 'logmel');
  const { rows, cols, values } = stage.thumbnail;
  const fragment = document.createDocumentFragment();
  fragment.append(element('title', {}, 'Measured log-mel energy from the recorded sample.'));
  fragment.append(element('desc', {}, `${stage.shape[0]} mel bands by ${stage.shape[1]} frames, displayed as ${rows} by ${cols} bin means. Higher frequencies are at the top. Colors encode energy, not confidence. The line follows audio playback.`));
  values.forEach((row, r) => row.forEach((value, c) => {
    fragment.append(element('rect', {
      x: plot.left + c * plot.width / cols,
      y: plot.top + (rows - 1 - r) * plot.height / rows,
      width: plot.width / cols, height: plot.height / rows,
      fill: energyColor(value, stage.stats.min, stage.stats.max),
      'data-value': value, 'data-mel-row': r, 'data-time-column': c
    }));
  }));
  fragment.append(element('text', { x: 0, y: 12 }, 'High'));
  fragment.append(element('text', { x: 0, y: 53 }, 'Low'));
  fragment.append(element('text', { x: plot.left, y: 69 }, '0 s'));
  fragment.append(element('text', { x: plot.left + plot.width, y: 69, 'text-anchor': 'end' }, `${trace.sample.duration_seconds.toFixed(2)} s`));
  fragment.append(element('line', {
    class: 'logmel-cursor', x1: plot.left, x2: plot.left,
    y1: plot.top, y2: plot.top + plot.height, visibility: 'hidden'
  }));
  svg.replaceChildren(fragment);
  svg.dataset.rows = rows;
  svg.dataset.columns = cols;
  svg.dataset.source = 'recorded-logmel-bin-means';
}

export function setLogMelTime(svg, seconds, duration) {
  const line = svg.querySelector('.logmel-cursor');
  if (!line || !duration) return;
  const x = plot.left + Math.max(0, Math.min(1, seconds / duration)) * plot.width;
  line.setAttribute('x1', x); line.setAttribute('x2', x);
  line.setAttribute('visibility', seconds > 0 ? 'visible' : 'hidden');
}
