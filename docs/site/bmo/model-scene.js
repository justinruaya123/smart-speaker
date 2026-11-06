import * as THREE from './vendor/three.module.js';

// This scene explains the frozen model graph. Its moving signal is an illustrative
// playback cursor; activation colors are populated only from recorded model output.
const PALETTE = {
  background: '#090d11', foreground: '#f1f5f5', muted: '#8d9ba2',
  accent: '#b7dde2', teal: '#72aeb7', amber: '#ebc88a', line: '#38555d',
};
const STAGE_IDS = ['normalize', 'stem', 'block1', 'block2', 'block3', 'block4', 'pool', 'heads'];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const v3 = (x, y = 0, z = 0) => new THREE.Vector3(x, y, z);

export function createModelScene(container, { onSelect = () => {} } = {}) {
  if (!(container instanceof HTMLElement)) throw new TypeError('A scene container is required.');
  container.classList.add('bmo-model-scene');
  const style = document.createElement('style');
  style.textContent = `
    .bmo-model-scene{position:relative;isolation:isolate;overflow:hidden;touch-action:pan-y}
    .bmo-model-scene canvas{display:block;width:100%;height:100%;outline:none;touch-action:none;cursor:grab}
    .bmo-model-scene canvas:active{cursor:grabbing}
    .bmo-model-scene canvas:focus-visible{outline:1px solid ${PALETTE.accent};outline-offset:-3px}
    .bmo-scene-labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
    .bmo-scene-label{position:absolute;left:0;top:0;transform:translate(-50%,-50%);min-width:40px;max-width:130px;padding:5px 6px;border:0;border-radius:5px;background:rgba(9,13,17,.9);color:${PALETTE.foreground};text-align:center;font-family:inherit;font-size:16px;font-weight:500;line-height:1.25;white-space:nowrap;pointer-events:auto;cursor:pointer;transition:color .15s,background .15s}
    .bmo-scene-label strong{display:block;font-family:inherit;font-size:17px;font-weight:550;letter-spacing:-.02em;color:inherit}
    .bmo-scene-label span{display:block;margin-top:4px;font-family:ui-monospace,Consolas,monospace;font-size:12px;font-weight:400;letter-spacing:-.02em;color:#b8c7cd;opacity:.95}
    .bmo-scene-label[data-kind=head]{transform:translate(0,-50%);text-align:left}
    .bmo-scene-label:hover,.bmo-scene-label:focus-visible{color:${PALETTE.foreground};background:#162228;outline:1px solid #45636a;outline-offset:1px}
    .bmo-scene-label[data-active=true]{color:${PALETTE.accent}}
    .bmo-scene-label[data-kind=head][data-active=true]{color:${PALETTE.amber}}
    .bmo-scene-static{position:absolute;pointer-events:none;transform:translate(-50%,-50%);color:#a9b9c0;text-align:center;font-family:ui-monospace,Consolas,monospace;font-size:13px;line-height:1.3;white-space:nowrap;background:rgba(9,13,17,.75);padding:3px 6px}
    .bmo-scene-fallback{display:flex;min-height:300px;align-items:center;justify-content:center;padding:28px;color:${PALETTE.foreground};font:inherit;line-height:1.6;text-align:center}
    @media(max-width:600px){.bmo-scene-label{padding:4px 2px;max-width:78px}.bmo-scene-label strong{font-size:10px}.bmo-scene-label span{font-size:8px}.bmo-scene-static{font-size:9px}}
    @media(prefers-reduced-motion:reduce){.bmo-scene-label{transition:none}}
  `;
  container.append(style);
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  } catch (error) {
    const fallback = document.createElement('div');
    fallback.className = 'bmo-scene-fallback';
    fallback.textContent = '3D rendering is unavailable in this browser. The stage controls below still describe the complete model: 40-bin features → 128-channel stem → four residual blocks → mean + max pooling → action, percent and alarm heads.';
    fallback.setAttribute('role', 'status');
    container.append(fallback);
    return { setStage() {}, setExpanded() {}, setTrace() {}, resetView() {}, setActive() {}, dispose() { style.remove(); fallback.remove(); } };
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(PALETTE.background, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Interactive 3D model architecture. Drag to rotate, scroll to zoom, or select a labeled stage. Arrow keys rotate; Home resets the view.');
  container.append(canvas);
  const overlay = document.createElement('div');
  overlay.className = 'bmo-scene-labels';
  container.append(overlay);

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-12, 12, 5, -5, .1, 100);
  const graph = new THREE.Group();
  const overview = new THREE.Group();
  const expandedGraph = new THREE.Group();
  scene.add(graph);
  graph.add(overview, expandedGraph);
  scene.add(new THREE.AmbientLight(PALETTE.accent, 1.15));
  const keyLight = new THREE.DirectionalLight('#ffffff', 2.1);
  keyLight.position.set(-5, 12, 10);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(PALETTE.teal, 1.4);
  rimLight.position.set(6, 3, -8);
  scene.add(rimLight);

  const resources = new Set();
  const overviewNodes = [], expandedNodes = [], labels = [], staticLabels = [];
  const pickables = [], overviewLinks = [], expandedLinks = [];
  let selectedStage = 0, progress = 0, expanded = false, expandedIndex = 2;
  let active = true, destroyed = false, frame = null, trace = null;
  let width = 960, height = 390, yaw = .23, pitch = .43, zoom = 1;
  let pointerDown = null, dragged = false;
  const pointer = new THREE.Vector2(), raycaster = new THREE.Raycaster();
  const temp = new THREE.Vector3();

  function remember(resource) { resources.add(resource); return resource; }
  function makeLine(points, color = PALETTE.line, opacity = .8, group = overview) {
    const geometry = remember(new THREE.BufferGeometry().setFromPoints(points));
    const material = remember(new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
    const line = new THREE.Line(geometry, material);
    group.add(line);
    return line;
  }
  function addLabel(node, title, subtitle, anchor, kind = '') {
    const element = document.createElement('button');
    element.type = 'button';
    element.className = 'bmo-scene-label';
    element.dataset.kind = kind;
    const strong = document.createElement('strong'), span = document.createElement('span');
    strong.textContent = title; span.textContent = subtitle;
    element.append(strong, span);
    element.setAttribute('aria-label', `${title}: ${subtitle}. Select stage.`);
    element.addEventListener('click', () => onSelect(node.stage));
    overlay.append(element);
    const label = { element, strong, span, node, anchor, group: node.view };
    labels.push(label);
    node.label = label;
  }
  function addStatic(text, anchor, view) {
    const element = document.createElement('div');
    element.className = 'bmo-scene-static';
    element.textContent = text;
    overlay.append(element);
    const label = { element, anchor, group: view };
    staticLabels.push(label);
    return label;
  }

  function volume({ x, y = 0, z = 0, w = 1.25, h = 1.95, d = 1.25, stage, title, subtitle, view = 'overview', kind = 'tensor', labelY = -1.85, color = PALETTE.teal }) {
    const group = view === 'overview' ? overview : expandedGraph;
    const geometry = remember(new THREE.BoxGeometry(w, h, d));
    const material = remember(new THREE.MeshStandardMaterial({ color, transparent: true, opacity: .13, roughness: .6, metalness: .05, depthWrite: false }));
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.userData.stage = stage;
    const wire = new THREE.LineSegments(remember(new THREE.EdgesGeometry(geometry)), remember(new THREE.LineBasicMaterial({ color, transparent: true, opacity: .6 })));
    mesh.add(wire);
    group.add(mesh);
    pickables.push(mesh);
    const node = { mesh, wire, material, x, y, z, w, h, d, stage, kind, view, color, inner: [], texture: null, surface: null };
    (view === 'overview' ? overviewNodes : expandedNodes).push(node);
    if (title) addLabel(node, title, subtitle, kind === 'head' ? v3(x + 1.06, y, z + d / 2) : v3(x, labelY, z), kind);
    return node;
  }
  function tensorGrid(node, columns = 8, rows = 8) {
    const group = node.view === 'overview' ? overview : expandedGraph;
    const z = node.z + node.d / 2 + .006;
    const points = [];
    for (let i = 1; i < columns; i++) {
      const x = node.x - node.w / 2 + node.w * i / columns;
      points.push(v3(x, node.y - node.h / 2, z), v3(x, node.y + node.h / 2, z));
    }
    for (let i = 1; i < rows; i++) {
      const y = node.y - node.h / 2 + node.h * i / rows;
      points.push(v3(node.x - node.w / 2, y, z), v3(node.x + node.w / 2, y, z));
    }
    const grid = new THREE.LineSegments(remember(new THREE.BufferGeometry().setFromPoints(points)), remember(new THREE.LineBasicMaterial({ color: PALETTE.line, transparent: true, opacity: .45, depthWrite: false })));
    group.add(grid);
    node.grid = grid;
  }
  function connection(start, end, stage, view = 'overview', bend = false) {
    const group = view === 'overview' ? overview : expandedGraph;
    const points = bend ? [start, v3((start.x + end.x) / 2, start.y, start.z), v3((start.x + end.x) / 2, end.y, end.z), end] : [start, end];
    const line = makeLine(points, PALETTE.line, .85, group);
    const curve = new THREE.CurvePath();
    for (let i = 1; i < points.length; i++) curve.add(new THREE.LineCurve3(points[i - 1], points[i]));
    const arrow = new THREE.Mesh(remember(new THREE.ConeGeometry(.055, .18, 6)), remember(new THREE.MeshBasicMaterial({ color: PALETTE.line })));
    arrow.rotation.z = -Math.PI / 2;
    arrow.position.copy(end).add(v3(-.055));
    group.add(arrow);
    const item = { line, curve, stage, arrow };
    (view === 'overview' ? overviewLinks : expandedLinks).push(item);
    return item;
  }

  // Width/depth expose a tensor volume, while native labels state its true shape.
  const input = volume({ x: -9, w: 1.3, h: 2.25, d: .18, stage: 0, title: 'Features', subtitle: '40 × T' });
  tensorGrid(input, 12, 10);
  const stem = volume({ x: -6.65, w: 1.2, h: 1.95, d: 1.1, stage: 1, title: 'Stem', subtitle: '40 → 128 · k5' });
  tensorGrid(stem, 8, 8);
  const blockX = [-4.35, -2.05, .25, 2.55];
  for (let i = 0; i < 4; i++) {
    const block = volume({ x: blockX[i], w: 1.38, h: 2.05, d: 1.45, stage: i + 2, title: `Block ${i + 1}`, subtitle: `d = ${2 ** i} · 128ch` });
    // The four inner planes stand for DW/PW twice, rather than arbitrary layers.
    for (let layer = 0; layer < 4; layer++) {
      const innerGeometry = remember(new THREE.BoxGeometry(.045, 1.73, 1.1));
      const innerMaterial = remember(new THREE.MeshBasicMaterial({ color: layer % 2 ? PALETTE.accent : PALETTE.teal, transparent: true, opacity: .15, depthWrite: false }));
      const inner = new THREE.Mesh(innerGeometry, innerMaterial);
      inner.position.set(blockX[i] - .43 + layer * .285, 0, 0);
      overview.add(inner);
      block.inner.push(inner);
    }
    tensorGrid(block, 8, 8);
    // A real residual connection bypasses the two convolutional sublayers.
    block.skip = makeLine([v3(blockX[i] - .7, -.28, -.82), v3(blockX[i] - .7, 1.4, -.82), v3(blockX[i] + .7, 1.4, -.82), v3(blockX[i] + .7, -.28, -.82)], PALETTE.teal, .4);
  }
  const pool = volume({ x: 4.92, w: .9, h: 1.25, d: 1.25, stage: 6, title: 'Mean + max', subtitle: '128 + 128 = 256' });
  tensorGrid(pool, 4, 8);
  const heads = [
    volume({ x: 7.8, y: 1.4, w: 1.4, h: .7, d: .6, stage: 7, title: 'Action', subtitle: '29 classes', kind: 'head', labelY: 2.2, color: PALETTE.amber }),
    volume({ x: 7.8, y: -.05, w: 1.4, h: .7, d: .6, stage: 7, title: 'Percent', subtitle: '4 classes', kind: 'head', labelY: -.79, color: PALETTE.amber }),
    volume({ x: 7.8, y: -1.5, w: 1.4, h: .7, d: .6, stage: 7, title: 'Alarm', subtitle: '4 classes', kind: 'head', labelY: -2.24, color: PALETTE.amber }),
  ];
  heads.forEach(node => tensorGrid(node, node === heads[0] ? 10 : 4, 2));
  const chain = overviewNodes.slice(0, 7);
  for (let i = 1; i < chain.length; i++) connection(v3(chain[i - 1].x + chain[i - 1].w / 2 + .08), v3(chain[i].x - chain[i].w / 2 - .08), i);
  heads.forEach(head => connection(v3(pool.x + pool.w / 2 + .08), v3(head.x - head.w / 2 - .1, head.y), 7, 'overview', true));
  addStatic('time ↓ 2', v3(-7.85, .66, 0), 'overview');
  addStatic('shared 128-channel encoder', v3(-.9, 2.3, -.5), 'overview');

  // Expanded residual: the executable order is DW → PW → LN → SiLU, twice.
  const ops = [
    ['DW conv', 'k7 · 128 groups', .95, 1.8], ['PW conv', 'k1 · 128 → 128', .85, 1.8],
    ['LayerNorm', 'per frame', .7, 1.65], ['SiLU', 'activation', .65, 1.45],
    ['DW conv', 'k7 · 128 groups', .95, 1.8], ['PW conv', 'k1 · 128 → 128', .85, 1.8],
    ['LayerNorm', 'per frame', .7, 1.65], ['SiLU', 'activation', .65, 1.45],
  ];
  const positions = [-8.5, -6.3, -4.1, -1.9, .3, 2.5, 4.7, 6.9];
  ops.forEach(([title, subtitle, w, h], i) => {
    const node = volume({ x: positions[i], y: .35, w, h, d: 1, stage: 2, title, subtitle, view: 'expanded', labelY: 1.9, kind: 'operation' });
    node.operationIndex = i;
    if (i % 4 < 2) tensorGrid(node, i % 4 === 0 ? 7 : 4, 8);
  });
  const addNode = volume({ x: 9.05, y: .35, w: .75, h: .75, d: .75, stage: 2, title: 'Add & scale', subtitle: '(x + f(x)) / √2', view: 'expanded', labelY: 1.9, color: PALETTE.amber, kind: 'operation' });
  expandedNodes.forEach((node, i) => {
    if (i) connection(v3(expandedNodes[i - 1].x + expandedNodes[i - 1].w / 2 + .1, .35), v3(node.x - node.w / 2 - .1, .35), 2, 'expanded');
  });
  const bypass = [v3(-9.65, .35), v3(-9.65, -1.8), v3(9.05, -1.8), v3(9.05, -.06)];
  const bypassLine = makeLine(bypass, PALETTE.amber, .7, expandedGraph);
  const bypassCurve = new THREE.CurvePath();
  bypass.slice(1).forEach((point, i) => bypassCurve.add(new THREE.LineCurve3(bypass[i], point)));
  addStatic('unchanged input x · residual bypass', v3(0, -2.18, 0), 'expanded');
  const dilationLabel = addStatic('Block 1 · dilation 1 · two depthwise-separable units', v3(0, 2.9, 0), 'expanded');
  addStatic('Training only: dropout 0.1 after each SiLU. Inference: no dropout.', v3(0, -2.75, 0), 'expanded');
  connection(v3(-9.65, .35), v3(-9.08, .35), 2, 'expanded');

  // A low-contrast ground reference makes depth readable during rotation.
  const groundPoints = [];
  for (let x = -10; x <= 10; x += 2) groundPoints.push(v3(x, -1.5, -2.2), v3(x, -1.5, 2.2));
  for (let z = -2; z <= 2; z += 1) groundPoints.push(v3(-10, -1.5, z), v3(10, -1.5, z));
  const ground = new THREE.LineSegments(remember(new THREE.BufferGeometry().setFromPoints(groundPoints)), remember(new THREE.LineBasicMaterial({ color: PALETTE.line, transparent: true, opacity: .15 })));
  overview.add(ground);
  const pulseGeometry = remember(new THREE.SphereGeometry(.095, 12, 8));
  const pulseMaterial = remember(new THREE.MeshBasicMaterial({ color: PALETTE.accent, transparent: true, opacity: .95 }));
  const pulses = Array.from({ length: 4 }, () => { const p = new THREE.Mesh(pulseGeometry, pulseMaterial); scene.add(p); return p; });

  function updateCamera() {
    const aspect = width / height;
    const visibleWidth = expanded ? 22.3 : 22.4;
    const visibleHeight = Math.max(expanded ? 7.5 : 7, visibleWidth / Math.max(aspect, .2));
    camera.left = -visibleHeight * aspect / 2 / zoom;
    camera.right = visibleHeight * aspect / 2 / zoom;
    camera.top = visibleHeight / 2 / zoom;
    camera.bottom = -visibleHeight / 2 / zoom;
    camera.position.set(30 * Math.sin(yaw) * Math.cos(pitch), 30 * Math.sin(pitch), 30 * Math.cos(yaw) * Math.cos(pitch));
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
  }

  function updateAppearance() {
    overview.visible = !expanded;
    expandedGraph.visible = expanded;
    const currentNodes = expanded ? expandedNodes : overviewNodes;
    for (const node of currentNodes) {
      const current = expanded || node.stage === selectedStage;
      const past = !expanded && node.stage < selectedStage;
      node.material.opacity = current ? .35 : past ? .27 : .2;
      node.wire.material.opacity = current ? 1 : past ? .85 : .7;
      node.wire.material.color.set(current ? node.kind === 'head' ? PALETTE.amber : PALETTE.accent : node.color);
      if (node.surface) node.surface.material.opacity = current ? .92 : past ? .76 : .6;
      if (node.grid) node.grid.material.opacity = node.surface ? .16 : current ? .7 : .5;
      node.inner.forEach(inner => { inner.material.opacity = current ? .34 : .22; });
      if (node.skip) node.skip.material.opacity = current ? .95 : .7;
    }
    for (const link of overviewLinks) {
      link.line.material.color.set(link.stage <= selectedStage ? PALETTE.accent : '#71949d');
      link.line.material.opacity = link.stage === selectedStage ? .95 : .7;
      link.arrow.material.color.copy(link.line.material.color);
    }
    for (const label of labels) {
      label.element.hidden = (label.group === 'expanded') !== expanded;
      label.element.dataset.active = String(expanded || label.node.stage === selectedStage);
      label.element.setAttribute('aria-pressed', String(expanded || label.node.stage === selectedStage));
    }
    staticLabels.forEach(label => { label.element.hidden = (label.group === 'expanded') !== expanded; });
  }

  function updatePulses() {
    pulses.forEach(pulse => { pulse.visible = false; });
    const p = clamp(progress, 0, 1);
    if (expanded) {
      const chainPoints = [v3(-9.65, .35), v3(9.05, .35)];
      pulses[0].position.lerpVectors(chainPoints[0], chainPoints[1], p);
      pulses[0].visible = true;
      pulses[1].position.copy(bypassCurve.getPointAt(p));
      pulses[1].visible = true;
      const op = Math.min(7, Math.floor(p * 8));
      expandedNodes.forEach((node, i) => { node.wire.material.color.set(i === op || i === 8 && p > .95 ? PALETTE.amber : PALETTE.accent); });
      return;
    }
    if (selectedStage === 0) {
      pulses[0].position.set(input.x - input.w / 2 + input.w * p, .0, input.d / 2 + .12);
      pulses[0].visible = true;
      return;
    }
    const links = overviewLinks.filter(link => link.stage === selectedStage);
    links.forEach((link, i) => { pulses[i].visible = true; pulses[i].position.copy(link.curve.getPointAt(p)); });
  }

  function projectLabels() {
    const relevant = [...labels, ...staticLabels].filter(label => !label.element.hidden);
    for (const label of relevant) {
      temp.copy(label.anchor).project(camera);
      const x = (temp.x * .5 + .5) * width;
      const y = (-temp.y * .5 + .5) * height;
      label.element.style.left = `${x}px`;
      label.element.style.top = `${y}px`;
      label.element.style.visibility = x < 16 || x > width - 16 || y < 12 || y > height - 10 ? 'hidden' : 'visible';
    }
    // At narrow widths keep the selected shape label, not an illegible row of text.
    if (width < 600) {
      for (const label of labels) {
        if (!expanded && label.node.stage !== selectedStage && label.node.kind !== 'head') label.element.style.visibility = 'hidden';
        if (expanded && label.node.operationIndex % 2 === 1) label.element.style.visibility = 'hidden';
      }
      staticLabels.forEach(label => { label.element.style.visibility = 'hidden'; });
    }
  }
  function render() {
    frame = null;
    if (destroyed || !active || !width || !height) return;
    updateCamera();
    updatePulses();
    renderer.render(scene, camera);
    projectLabels();
  }
  function requestRender() {
    if (!destroyed && active && frame === null) frame = requestAnimationFrame(render);
  }
  function resize() {
    const rect = container.getBoundingClientRect();
    width = Math.max(1, container.clientWidth || rect.width || 960);
    height = Math.max(1, container.clientHeight || rect.height || 390);
    renderer.setSize(width, height, false);
    requestRender();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);

  function stageTrace(id) { return trace?.stages?.find(stage => stage.id === id); }
  function dataTexture(values, rows, cols, documentedMin, documentedMax) {
    if (!values || !rows || !cols) return null;
    const flat = Array.isArray(values[0]) ? values.flat() : values;
    const finite = flat.filter(Number.isFinite);
    if (!finite.length) return null;
    const low = Number.isFinite(documentedMin) ? documentedMin : Math.min(...finite);
    const high = Number.isFinite(documentedMax) ? documentedMax : Math.max(...finite);
    const span = high - low || 1;
    const bytes = new Uint8Array(rows * cols * 4);
    const dark = new THREE.Color('#14242b'), light = new THREE.Color(PALETTE.accent), color = new THREE.Color();
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const value = Number(flat[row * cols + col]);
      const amount = Number.isFinite(value) ? clamp((value - low) / span, 0, 1) : 0;
      color.copy(dark).lerp(light, amount);
      const at = ((rows - 1 - row) * cols + col) * 4;
      bytes[at] = Math.round(color.r * 255); bytes[at + 1] = Math.round(color.g * 255); bytes[at + 2] = Math.round(color.b * 255); bytes[at + 3] = 255;
    }
    const texture = remember(new THREE.DataTexture(bytes, cols, rows, THREE.RGBAFormat));
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.needsUpdate = true;
    return texture;
  }
  function clearSurface(node) {
    if (node.surface) { node.surface.parent.remove(node.surface); node.surface.geometry.dispose(); resources.delete(node.surface.geometry); node.surface.material.dispose(); resources.delete(node.surface.material); node.surface = null; }
    if (node.texture) { node.texture.dispose(); resources.delete(node.texture); node.texture = null; }
  }
  function applyThumbnail(node, thumbnail) {
    clearSurface(node);
    if (!thumbnail) return;
    const texture = dataTexture(thumbnail.values, thumbnail.rows, thumbnail.cols, thumbnail.min, thumbnail.max);
    if (!texture) return;
    node.texture = texture;
    const geometry = remember(new THREE.PlaneGeometry(node.w * .94, node.h * .94));
    const material = remember(new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: .7, depthWrite: false, side: THREE.DoubleSide }));
    const surface = new THREE.Mesh(geometry, material);
    surface.position.set(node.x, node.y, node.z + node.d / 2 + .012);
    (node.view === 'overview' ? overview : expandedGraph).add(surface);
    node.surface = surface;
  }
  function applyTrace() {
    for (const node of overviewNodes) {
      const data = stageTrace(STAGE_IDS[node.stage]);
      if (node.kind === 'head') {
        const key = node.label.strong.textContent.toLowerCase();
        const logits = trace?.heads?.[key]?.logits;
        applyThumbnail(node, logits ? { rows: 1, cols: logits.length, values: [logits] } : null);
      } else {
        let thumbnail = data?.thumbnail;
        if (node.stage === 0 && !thumbnail && trace?.features?.values) thumbnail = { rows: trace.features.shape[0], cols: trace.features.shape[1], values: trace.features.values };
        applyThumbnail(node, thumbnail);
      }
    }
    const block = stageTrace(`block${expandedIndex - 1}`);
    expandedNodes.forEach((node, i) => {
      const child = block?.children?.[i];
      applyThumbnail(node, i === 8 ? block?.thumbnail : child?.thumbnail);
    });
    const shape = stageTrace('stem')?.shape;
    if (shape) stem.label.span.textContent = `128 × ${shape.at(-1)} · k5`;
    if (trace?.features?.shape) input.label.span.textContent = `${trace.features.shape[0]} × ${trace.features.shape[1]}`;
    updateAppearance();
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    pointerDown = { x: event.clientX, y: event.clientY, yaw, pitch, id: event.pointerId };
    dragged = false;
    canvas.setPointerCapture(event.pointerId);
  }
  function onPointerMove(event) {
    if (!pointerDown) {
      const rect = canvas.getBoundingClientRect();
      pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      canvas.style.cursor = raycaster.intersectObjects(pickables.filter(mesh => mesh.parent.visible)).length ? 'pointer' : 'grab';
      return;
    }
    const dx = event.clientX - pointerDown.x, dy = event.clientY - pointerDown.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) dragged = true;
    yaw = clamp(pointerDown.yaw - dx * .0035, -.55, .55);
    pitch = clamp(pointerDown.pitch + dy * .0035, -.08, .72);
    requestRender();
  }
  function onPointerUp(event) {
    if (!pointerDown) return;
    if (!dragged) {
      const rect = canvas.getBoundingClientRect();
      pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(pickables.filter(mesh => mesh.parent.visible), false);
      if (hits.length) onSelect(expanded ? expandedIndex : hits[0].object.userData.stage);
    }
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    pointerDown = null;
  }
  function onWheel(event) {
    event.preventDefault();
    zoom = clamp(zoom * Math.exp(-event.deltaY * .001), .78, 1.3);
    requestRender();
  }
  function onKey(event) {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) return;
    event.preventDefault(); event.stopPropagation();
    if (event.key === 'Home') { yaw = .23; pitch = .43; zoom = 1; }
    if (event.key === 'ArrowLeft') yaw = clamp(yaw - .07, -.55, .55);
    if (event.key === 'ArrowRight') yaw = clamp(yaw + .07, -.55, .55);
    if (event.key === 'ArrowUp') pitch = clamp(pitch + .05, -.08, .72);
    if (event.key === 'ArrowDown') pitch = clamp(pitch - .05, -.08, .72);
    requestRender();
  }
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('keydown', onKey);

  updateAppearance();
  resize();
  return {
    setStage(index, nextProgress = 0) {
      selectedStage = clamp(Math.floor(Number(index) || 0), 0, 7);
      progress = clamp(Number(nextProgress) || 0, 0, 1);
      if (selectedStage >= 2 && selectedStage <= 5 && expandedIndex !== selectedStage) {
        expandedIndex = selectedStage;
        expandedNodes.forEach(node => { node.stage = expandedIndex; node.mesh.userData.stage = expandedIndex; });
        dilationLabel.element.textContent = `Block ${expandedIndex - 1} · dilation ${2 ** (expandedIndex - 2)} · two depthwise-separable units`;
        if (trace) applyTrace();
      }
      updateAppearance();
      requestRender();
    },
    setExpanded(value) { expanded = Boolean(value); updateAppearance(); requestRender(); },
    setTrace(value) { trace = value || null; applyTrace(); requestRender(); },
    resetView() { yaw = .23; pitch = .43; zoom = 1; requestRender(); },
    setActive(value) { active = Boolean(value); if (!active && frame !== null) { cancelAnimationFrame(frame); frame = null; } else if (active) { resize(); } },
    dispose() {
      destroyed = true;
      if (frame !== null) cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      canvas.removeEventListener('pointerdown', onPointerDown); canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp); canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel); canvas.removeEventListener('keydown', onKey);
      resources.forEach(resource => resource.dispose());
      renderer.dispose(); canvas.remove(); overlay.remove(); style.remove();
      container.classList.remove('bmo-model-scene');
    },
  };
}
