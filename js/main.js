import { buildGroups } from './groups.js';
import { Layout } from './layout.js';
import { Renderer } from './render.js';
import { buildMosaic, loadImage } from './mosaic.js';

const $ = id => document.getElementById(id);
const canvas = $('gl'), tip = $('tip'), labelsEl = $('labels'), status = $('status');

const data = await fetch('posts.json', { cache: 'no-cache' }).then(r => r.json()).catch(() => null);
if (!data?.posts?.length) {
  status.textContent = 'no posts yet: run scripts/fetch-instagram.mjs';
  throw new Error('posts.json empty or missing');
}
const posts = data.posts, N = posts.length, MAX_HUBS = 600, MAX_TILES = 20000;
const CAP = Math.max(N + MAX_HUBS, MAX_TILES);
const captions = posts.map(p => (p.caption ?? '').toLowerCase());
// mosaic material: every post plus carousel slides (which point back to their post)
const cands = [...posts.map((p, i) => ({ post: i, tile: p.tile, color: p.color })), ...(data.extras ?? [])];

const layout = new Layout(N, CAP);
let renderer;
try { renderer = new Renderer(canvas, CAP); } catch (e) { status.textContent = e.message; throw e; }
renderer.loadAtlas(data.atlas).catch(e => console.error('atlas', e));
const profile = data.profile ? loadImage(data.profile) : Promise.resolve(null);
profile.then(img => img && renderer.setTarget(img));

const ui = {
  mode: 'tags', k: 16, tile: 8, tint: 0.7, depth: 0.45, res: 72, detail: 0.6, variety: 0.4, search: '',
  edges: true, labels: true, spin: true,
};
// node i -> post index (-1 for hubs). Graph modes: node i is post i; mosaic: node i is a tile instance
const nodePost = new Int32Array(CAP).fill(-1);
const tile = new Float32Array(CAP), nodeState = new Float32Array(CAP), sizes = new Float32Array(CAP);
let groups = null, mosaic = null, mosaicD = 0, hubEls = [], hubCount = [];
let hover = -1, selected = -1;

// ---------- camera ----------
const cam = {
  yaw: 0.6, pitch: 0.25, dist: 900, tx: 0, ty: 0, tz: 0, fov: 50 * Math.PI / 180,
  viewProj: new Float32Array(16), eye: [0, 0, 0], right: [1, 0, 0], up: [0, 1, 0], fwd: [0, 0, -1],
  fogNear: 0, fogFar: 1, focal: 1,
};
let fly = null, autoFit = true, interacting = false;

function updateCamera() {
  const w = canvas.clientWidth, h = canvas.clientHeight, aspect = w / h;
  const cp = Math.cos(cam.pitch);
  const e = [cam.tx + cam.dist * Math.sin(cam.yaw) * cp, cam.ty + cam.dist * Math.sin(cam.pitch), cam.tz + cam.dist * Math.cos(cam.yaw) * cp];
  const f = norm([cam.tx - e[0], cam.ty - e[1], cam.tz - e[2]]);
  const r = norm(cross(f, [0, 1, 0]));
  const u = cross(r, f);
  // far plane must reach the farthest node at any zoom; near stays a fraction of the view distance
  const reach = Math.hypot(cam.tx, cam.ty, cam.tz) + (layout.maxRadius ?? layoutRadius) + ui.tile * 4;
  const near = Math.max(0.05, cam.dist * 0.005), far = (cam.dist + reach) * 1.2;
  const t = 1 / Math.tan(cam.fov / 2);
  const view = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, e), -dot(u, e), dot(f, e), 1];
  const proj = [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
  for (let c = 0; c < 4; c++) for (let rr = 0; rr < 4; rr++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += proj[k * 4 + rr] * view[c * 4 + k];
    cam.viewProj[c * 4 + rr] = s;
  }
  cam.eye = e; cam.right = r; cam.up = u; cam.fwd = f;
  cam.focal = t * h / 2;
  const R = layoutRadius;
  // fog over the whole depth of the scene from where the camera is, not from the 95% radius
  cam.fogNear = Math.max(0, cam.dist - R * 0.3);
  cam.fogFar = cam.dist + Math.max(R * 1.6, reach * 1.1);
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// world -> screen px; returns [sx, sy, w] (w = view depth, <=0 behind camera)
function project(x, y, z) {
  const m = cam.viewProj;
  const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
  const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
  const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [(cx / cw + 1) * 0.5 * canvas.clientWidth, (1 - cy / cw) * 0.5 * canvas.clientHeight, cw];
}

function fitDistance(R) {
  const aspect = Math.min(1, canvas.clientWidth / canvas.clientHeight);
  return R / (Math.tan(cam.fov / 2) * aspect);
}

// ---------- modes ----------
function updateSizes() {
  if (ui.mode === 'mosaic') return;   // placeMosaic sets tile sizes
  for (let i = 0; i < N; i++) sizes[i] = ui.tile;
  for (let h = 0; h < layout.n - N; h++) sizes[N + h] = ui.tile * 0.35 * (1 + Math.sqrt(hubCount[h] ?? 1) * 0.15);
}

// where each post currently is (any of its nodes), so the next layout morphs from there
function snapshotPosts() {
  const snap = new Float32Array(N * 3).fill(NaN);
  for (let i = 0; i < layout.n; i++) {
    const p = nodePost[i];
    if (p >= 0) { snap[p * 3] = layout.x[i]; snap[p * 3 + 1] = layout.y[i]; snap[p * 3 + 2] = layout.z[i]; }
  }
  return snap;
}
function restorePosts(snap, count) {
  const r = Math.max(50, layoutRadius);
  for (let i = 0; i < count; i++) {
    const p = nodePost[i];
    if (p < 0) continue;
    const ok = !Number.isNaN(snap[p * 3]);
    layout.x[i] = ok ? snap[p * 3] : (Math.random() - 0.5) * r;
    layout.y[i] = ok ? snap[p * 3 + 1] : (Math.random() - 0.5) * r;
    layout.z[i] = ok ? snap[p * 3 + 2] : (Math.random() - 0.5) * r;
    layout.vx[i] = layout.vy[i] = layout.vz[i] = 0;
  }
}

function placeMosaic() {
  // image width in world units: the smallest tiles are ui.tile wide
  const W = ui.res * ui.tile, m = mosaic.count;
  // sweet-spot camera on +z; every tile sits on the ray through its cell, scaled by depth,
  // so from there it reads as a flat portrait, from anywhere else it's a volume
  mosaicD = fitDistance(W * (ui.round ? 0.52 : 0.56)) * 1.08;
  const tx = new Float32Array(m), ty = new Float32Array(m), tz = new Float32Array(m);
  const s0 = ui.tile * 2;   // same-size mode: every tile this big in world space
  for (let i = 0; i < m; i++) {
    let t;
    if (ui.uniform) {
      // distance alone makes the size: a tile of size s0 at ray parameter t covers a cell of s0 / t.
      // Depth jitters the cell a little so the three quadtree sizes don't form three flat layers
      const cell = mosaic.size[i] * W * (1 + mosaic.depth[i] * ui.depth * 0.35);
      t = Math.max(0.15, s0 / cell);
      sizes[i] = s0;
    } else {
      t = 1 - mosaic.depth[i] * ui.depth * 0.6;
      sizes[i] = mosaic.size[i] * W * t * 1.01;
    }
    tx[i] = mosaic.x[i] * W * t; ty[i] = mosaic.y[i] * W * t; tz[i] = mosaicD * (1 - t);
  }
  layout.setTargets(tx, ty, tz, m);
}

async function buildMosaicNodes() {
  const img = await profile;
  if (!img) { status.textContent = 'no profile picture in posts.json'; return false; }
  const snap = snapshotPosts();
  mosaic = buildMosaic(cands, img, { res: ui.res, detail: ui.detail, variety: ui.variety, round: ui.round });
  nodePost.fill(-1);
  for (let i = 0; i < mosaic.count; i++) { const c = cands[mosaic.cand[i]]; nodePost[i] = c.post; tile[i] = c.tile; }
  restorePosts(snap, mosaic.count);
  placeMosaic();
  renderer.setMosaic(mosaic.mean, mosaic.rect);
  return true;
}

const flyToSweetSpot = () => { fly = { yaw: 0, pitch: 0, dist: mosaicD, tx: 0, ty: 0, tz: 0 }; };

async function applyMode() {
  exitFly();
  hover = selected = -1;
  if (ui.mode === 'mosaic') {
    if (!(await buildMosaicNodes())) return;
    renderer.setEdges(layout.src, layout.dst);
    groups = null;
    hubCount = [];
    autoFit = false;
    flyToSweetSpot();
    status.textContent = `${mosaic.count} tiles · ${mosaic.unique} of ${N} posts`;
  } else {
    const snap = snapshotPosts();
    groups = buildGroups(posts, ui.mode, ui.k);
    nodePost.fill(-1);
    for (let i = 0; i < N; i++) { nodePost[i] = i; tile[i] = posts[i].tile; }
    for (let h = 0; h < groups.hubs.length; h++) tile[N + h] = -1;
    restorePosts(snap, N);
    layout.setGraph(groups);
    renderer.setEdges(layout.src, layout.dst);
    hubCount = groups.hubs.map(() => 0);
    groups.postLinks.forEach(l => l.forEach(h => hubCount[h]++));
    updateSizes();
    autoFit = true;
    fly = null;
    status.textContent = `${N} posts · ${groups.hubs.length} ${ui.mode === 'time' ? 'weeks' : 'hubs'}`;
  }
  buildLabels();
  updateStates();
  document.querySelectorAll('[data-modes]').forEach(el => { el.hidden = !el.dataset.modes.split(' ').includes(ui.mode); });
  movePill();
}

function movePill() {
  const on = document.querySelector('.seg button.on'), pill = document.querySelector('.pill');
  pill.style.setProperty('--x', `${on.offsetLeft}px`);
  pill.style.setProperty('--w', `${on.offsetWidth}px`);
}
addEventListener('resize', movePill);

function buildLabels() {
  labelsEl.replaceChildren();
  hubEls = (groups?.hubs ?? []).map((h, i) => {
    if (!h.label) return null;   // timeline: only year starts get a label
    const el = document.createElement('div');
    el.className = 'hub';
    el.textContent = h.label;
    el.style.fontSize = `${Math.min(22, 11 + Math.sqrt(hubCount[i]) * 0.9)}px`;
    labelsEl.append(el);
    return el;
  });
}

// node + edge highlight states: hover / selected hub / caption filter
function updateStates() {
  const n = layout.n, q = ui.search.trim().toLowerCase();
  for (let i = 0; i < n; i++) nodeState[i] = nodePost[i] >= 0 && q && !captions[nodePost[i]].includes(q) ? -1 : 0;
  const edgeState = new Float32Array(layout.edgeCount);
  const focus = [hover, selected].filter(i => i >= 0);
  for (let e = 0; e < layout.edgeCount; e++) {
    const s = layout.src[e], d = layout.dst[e];
    if (focus.includes(s) || focus.includes(d)) {
      edgeState[e] = 1;
      if (nodeState[s] >= 0) nodeState[s] = 1;
      if (nodeState[d] >= 0) nodeState[d] = 1;
    } else if (q && (nodeState[s] < 0 || nodeState[d] < 0)) edgeState[e] = -1;
  }
  for (const f of focus) nodeState[f] = 1;
  renderer.setStates(n, tile, nodeState, edgeState);
}

// ---------- picking ----------
function pick(mx, my) {
  let best = -1, bw = Infinity;
  for (let i = 0; i < layout.n; i++) {
    const [sx, sy, w] = project(layout.x[i], layout.y[i], layout.z[i]);
    if (w <= 0 || w >= bw) continue;
    const half = sizes[i] * 0.5 * cam.focal / w;
    if (Math.abs(sx - mx) <= half && Math.abs(sy - my) <= half) { best = i; bw = w; }
  }
  return best;
}

// point under the mouse on the plane through p facing the camera
function unproject(mx, my, p) {
  const t = Math.tan(cam.fov / 2), aspect = canvas.clientWidth / canvas.clientHeight;
  const nx = (mx / canvas.clientWidth * 2 - 1) * t * aspect, ny = (1 - my / canvas.clientHeight * 2) * t;
  const d = [0, 1, 2].map(k => cam.fwd[k] + cam.right[k] * nx + cam.up[k] * ny);
  const s = dot([p[0] - cam.eye[0], p[1] - cam.eye[1], p[2] - cam.eye[2]], cam.fwd) / dot(d, cam.fwd);
  return [0, 1, 2].map(k => cam.eye[k] + d[k] * s);
}

function showTip(i, mx, my) {
  if (i < 0) { tip.hidden = true; return; }
  tip.replaceChildren();
  const head = document.createElement('b');
  const body = document.createElement('div');
  if (nodePost[i] >= 0) {
    const p = posts[nodePost[i]];
    head.textContent = new Date(p.timestamp).toLocaleDateString() + (p.type === 'STORY' ? ' · story' : '');
    body.textContent = p.type === 'STORY' ? '(story)' : p.caption ?p.caption.slice(0, 160) + (p.caption.length > 160 ? '…' : '') : '(no caption)';
  } else {
    const h = groups.hubs[i - N];
    head.textContent = `${hubCount[i - N]} posts`;
    body.textContent = h.title ?? h.label;
  }
  tip.append(head, body);
  tip.hidden = false;
  const x = Math.min(mx + 14, innerWidth - tip.offsetWidth - 8), y = Math.min(my + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}

// ---------- interaction ----------
const pointers = new Map();
let drag = null, pinch = 0;

canvas.addEventListener('pointerdown', e => {
  if (flying) {
    if (document.pointerLockElement !== canvas) { canvas.requestPointerLock(); return; }
    const h = pick(canvas.clientWidth / 2, canvas.clientHeight / 2);   // crosshair
    if (h >= 0 && nodePost[h] >= 0) { document.exitPointerLock(); window.open(posts[nodePost[h]].permalink, '_blank', 'noopener'); }
    return;
  }
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a[0] - b[0], a[1] - b[1]); drag = null; return; }
  const hit = e.button === 0 && !e.shiftKey ? pick(e.clientX, e.clientY) : -1;
  // in the mosaic everything is a tile, so dragging always orbits there
  const grab = hit >= 0 && ui.mode !== 'mosaic';
  drag = { kind: grab ? 'node' : e.button === 2 || e.shiftKey ? 'pan' : 'orbit', id: hit, x: e.clientX, y: e.clientY, moved: false };
  if (grab) {
    layout.held = hit; layout.hx = layout.x[hit]; layout.hy = layout.y[hit]; layout.hz = layout.z[hit];
    if (!layout.targets) layout.alphaTarget = 0.3;
    layout.reheat(0.3);
  }
  interacting = true;
  canvas.classList.add('dragging');
  tip.hidden = true;
});

canvas.addEventListener('pointermove', e => {
  if (document.pointerLockElement === canvas) { look(e.movementX, e.movementY); return; }
  if (pointers.has(e.pointerId)) pointers.set(e.pointerId, [e.clientX, e.clientY]);
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinch) cam.dist *= pinch / d;
    pinch = d; autoFit = false; fly = null;
    return;
  }
  if (!drag) {
    const h = pick(e.clientX, e.clientY);
    canvas.classList.toggle('over', h >= 0);
    if (h !== hover) { hover = h; updateStates(); }
    showTip(h, e.clientX, e.clientY);
    return;
  }
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.kind === 'node') {
    [layout.hx, layout.hy, layout.hz] = unproject(e.clientX, e.clientY, [layout.hx, layout.hy, layout.hz]);
    layout.reheat(0.3);
  } else if (drag.kind === 'orbit') {
    cam.yaw -= dx * 0.005;
    cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dy * 0.005));
    fly = null;
  } else {
    const s = cam.dist * 2 * Math.tan(cam.fov / 2) / canvas.clientHeight;
    for (let k = 0; k < 3; k++) { const v = (-cam.right[k] * dx + cam.up[k] * dy) * s; k === 0 ? cam.tx += v : k === 1 ? cam.ty += v : cam.tz += v; }
    autoFit = false; fly = null;
  }
});

function endPointer(e) {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = 0;
  if (!drag || pointers.size) return;
  if (drag.kind === 'node') {
    layout.held = -1;
    layout.alphaTarget = 0;
    if (!drag.moved) {
      if (nodePost[drag.id] >= 0) window.open(posts[nodePost[drag.id]].permalink, '_blank', 'noopener');
      else { selected = selected === drag.id ? -1 : drag.id; updateStates(); }
    }
  } else if (!drag.moved && drag.id >= 0 && nodePost[drag.id] >= 0) {
    window.open(posts[nodePost[drag.id]].permalink, '_blank', 'noopener');
  } else if (!drag.moved && selected >= 0) { selected = -1; updateStates(); }
  drag = null; interacting = false;
  canvas.classList.remove('dragging');
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerleave', () => { tip.hidden = true; if (hover >= 0 && !drag) { hover = -1; updateStates(); } });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  if (flying) { flySpeed = Math.min(20, Math.max(0.05, flySpeed * Math.exp(-e.deltaY * 0.001))); showHint(); return; }
  cam.dist *= Math.exp(e.deltaY * 0.001);
  autoFit = false; fly = null;
}, { passive: false });

// ---------- UI ----------
const fill = el => el.style.setProperty('--p', `${(el.value - el.min) / (el.max - el.min) * 100}%`);
const bind = (id, fmt, fn) => {
  const el = $(id), out = document.querySelector(`output[for=${id}]`);
  const upd = () => { if (out) out.textContent = fmt(+el.value); fill(el); fn(+el.value); };
  el.addEventListener('input', upd);
  if (out) out.textContent = fmt(+el.value);
  fill(el);
  return el;
};
let kTimer;
bind('k', v => v, v => { ui.k = v; clearTimeout(kTimer); kTimer = setTimeout(applyMode, 150); });
bind('link', v => v, v => { layout.linkDistance = v; layout.reheat(0.5); });
bind('charge', v => v, v => { layout.charge = v; layout.reheat(0.5); });
bind('tint', v => `${v}%`, v => { ui.tint = v / 100; });
bind('depth', v => `${v}%`, v => { ui.depth = v / 100; if (mosaic && ui.mode === 'mosaic') placeMosaic(); });
bind('tile', v => v, v => { ui.tile = v; if (ui.mode === 'mosaic' && mosaic) { placeMosaic(); flyToSweetSpot(); } else { updateSizes(); layout.reheat(0.2); } });
let mTimer;
const rebuild = () => { clearTimeout(mTimer); mTimer = setTimeout(() => ui.mode === 'mosaic' && applyMode(), 200); };
bind('res', v => v, v => { ui.res = v; rebuild(); });
bind('detail', v => `${v}%`, v => { ui.detail = v / 100; rebuild(); });
bind('variety', v => `${v}%`, v => { ui.variety = v / 100; rebuild(); });
ui.res = +$('res').value; ui.detail = $('detail').value / 100; ui.variety = $('variety').value / 100;
ui.k = +$('k').value; layout.linkDistance = +$('link').value; layout.charge = +$('charge').value;
ui.tint = $('tint').value / 100; ui.depth = $('depth').value / 100; ui.tile = +$('tile').value;

for (const id of ['edges', 'showLabels', 'spin']) {
  const el = $(id), key = id === 'showLabels' ? 'labels' : id;
  ui[key] = el.checked;
  el.addEventListener('change', () => { ui[key] = el.checked; });
}
ui.uniform = $('uniform').checked;
ui.round = $('round').checked;
$('round').addEventListener('change', e => { ui.round = e.target.checked; rebuild(); });
$('uniform').addEventListener('change', e => { ui.uniform = e.target.checked; if (mosaic && ui.mode === 'mosaic') placeMosaic(); });
$('search').addEventListener('input', e => { ui.search = e.target.value; updateStates(); });
const modeButtons = [...document.querySelectorAll('.seg button')];
const setMode = b => {
  modeButtons.forEach(x => x.classList.toggle('on', x === b));
  ui.mode = b.dataset.mode;
  movePill();
  applyMode();
};
modeButtons.forEach(b => b.addEventListener('click', () => setMode(b)));
$('fit').addEventListener('click', () => {
  exitFly();
  if (ui.mode === 'mosaic') { fly = { ...fly, yaw: cam.yaw, pitch: cam.pitch, tx: 0, ty: 0, tz: 0, dist: mosaicD * 1.1 }; }
  else { autoFit = true; fly = { yaw: cam.yaw, pitch: cam.pitch, tx: 0, ty: 0, tz: 0, dist: cam.dist }; }
});
$('view').addEventListener('click', () => { exitFly(); flyToSweetSpot(); });
$('flymode').addEventListener('click', () => (flying ? exitFly() : enterFly()));
$('shake').addEventListener('click', () => {
  const s = ui.tile * 4;
  for (let i = 0; i < layout.n; i++) { layout.vx[i] += (Math.random() - 0.5) * s; layout.vy[i] += (Math.random() - 0.5) * s; layout.vz[i] += (Math.random() - 0.5) * s; }
  layout.reheat(1);
});
const panel = $('panel'), tune = $('tune');
const toggleTune = open => {
  panel.hidden = !(open ?? panel.hidden);
  tune.setAttribute('aria-expanded', String(!panel.hidden));
};
tune.addEventListener('click', () => toggleTune());
canvas.addEventListener('pointerdown', () => toggleTune(false));

// shortcuts: / search, 1-4 modes, t tune, h hide all UI, f fly, WASD/QE move
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') { if (e.key === 'Escape') e.target.blur(); return; }
  const k = e.key.toLowerCase();
  if (FLY_KEYS.has(k) || k === 'shift') { keys.add(k); if (k !== 'shift') enterFly(); return; }
  if (k === 'f') { flying ? exitFly() : enterFly(); return; }
  if (e.key === '/') { e.preventDefault(); $('search').focus(); }
  else if (e.key >= '1' && e.key <= '4') setMode(modeButtons[+e.key - 1]);
  else if (e.key === 't') toggleTune();
  else if (e.key === 'h') document.body.classList.toggle('zen');
});

$('handle').textContent = data.username ? `@${data.username}` : 'Instagram Cloud';
if (data.profile) $('avatar').src = data.profile;

// ---------- fly mode ----------
// The orbit camera shrinks to a tiny radius so yaw/pitch turn the view in place;
// WASD then moves the orbit target, and the eye with it.
const FLY_KEYS = new Set(['w', 'a', 's', 'd', 'q', 'e']);
const keys = new Set(), flyVel = [0, 0, 0];
let flying = false, flySpeed = 1, hintTimer;
const hint = $('hint'), flyBtn = $('flymode');
addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
addEventListener('blur', () => keys.clear());
document.addEventListener('pointerlockchange', () => document.body.classList.toggle('locked', document.pointerLockElement === canvas));

const dirOf = () => { const cp = Math.cos(cam.pitch); return [Math.sin(cam.yaw) * cp, Math.sin(cam.pitch), Math.cos(cam.yaw) * cp]; };
function setEye(e) {   // keep the eye at e, put the target dist in front of it
  const d = dirOf();
  cam.tx = e[0] - d[0] * cam.dist; cam.ty = e[1] - d[1] * cam.dist; cam.tz = e[2] - d[2] * cam.dist;
}
function showHint() {
  hint.textContent = `WASD move · Q/E down/up · Shift fast · ${document.pointerLockElement === canvas ? 'click opens post · Esc frees mouse' : 'click to steer with mouse'} · F exit · speed ${flySpeed.toFixed(2)}×`;
  hint.hidden = false;
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => { hint.hidden = true; }, 4000);
}
function enterFly() {
  if (flying) return;
  updateCamera();
  const eye = cam.eye.slice();
  flying = true; autoFit = false; fly = null;
  cam.dist = 0.5;
  setEye(eye);
  flyBtn.setAttribute('aria-pressed', 'true');
  showHint();
}
function exitFly() {
  if (!flying) return;
  flying = false;
  keys.clear(); flyVel.fill(0);
  if (document.pointerLockElement) document.exitPointerLock();
  // back to orbiting a point ahead of where we are
  updateCamera();
  const eye = cam.eye.slice();
  cam.dist = Math.max(50, layoutRadius * 0.8);
  setEye(eye);
  flyBtn.setAttribute('aria-pressed', 'false');
  hint.hidden = true;
}
function look(dx, dy) {
  const eye = cam.eye.slice();
  cam.yaw -= dx * 0.0022;
  cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch + dy * 0.0022));
  setEye(eye);
}
function flyStep(dt) {
  const f = cam.fwd, r = cam.right, ax = (keys.has('d') - keys.has('a')), az = (keys.has('w') - keys.has('s')), ay = (keys.has('e') - keys.has('q'));
  // speed scales with the scene so a 500-post graph and a 20k-tile mosaic feel the same
  const v = Math.max(100, layoutRadius) * 0.45 * flySpeed * (keys.has('shift') ? 3 : 1);
  const k = 1 - Math.exp(-dt * 8);   // ease in/out
  for (let i = 0; i < 3; i++) flyVel[i] += ((f[i] * az + r[i] * ax + (i === 1 ? ay : 0)) * v - flyVel[i]) * k;
  cam.tx += flyVel[0] * dt; cam.ty += flyVel[1] * dt; cam.tz += flyVel[2] * dt;
}

// ---------- loop ----------
let layoutRadius = 100, frameNo = 0;
const lerp = (a, b, t) => a + (b - a) * t;
const hubColors = { tags: [0.45, 0.75, 1.0], words: [0.65, 1.0, 0.6], time: [1.0, 0.7, 0.35], mosaic: [1, 1, 1] };

// idle spin on two axes: slow yaw, plus a pitch wobble with an unrelated period so it never repeats
const SPIN_YAW = 0.05;                         // rad/s, a turn every ~2 min
const WOBBLE_AMP = 0.3, WOBBLE_FREQ = 0.21;    // rad, rad/s (~30 s period)
let lastT = performance.now(), spinT = 0;

// adaptive resolution: weak GPUs drop pixel density until frames are smooth again
let frameMs = 16, dprCooldown = 0;
renderer.dprCap = Math.min(devicePixelRatio, 2);
function adaptResolution(ms) {
  if (ms > 100) return;                 // throttled/background tab, not a real measurement
  frameMs += (ms - frameMs) * 0.05;
  if ((dprCooldown -= ms) > 0) return;
  const max = Math.min(devicePixelRatio, 2);
  if (frameMs > 24 && renderer.dprCap > 1) { renderer.dprCap = Math.max(1, renderer.dprCap - 0.25); dprCooldown = 1500; }
  else if (frameMs < 18 && renderer.dprCap < max) { renderer.dprCap = Math.min(max, renderer.dprCap + 0.25); dprCooldown = 4000; }
}

function frame(now) {
  adaptResolution(now - lastT);
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  layout.tick();
  if (frameNo++ % 10 === 0) layoutRadius = Math.max(50, layout.radius());

  if (fly) {
    const k = 0.07;
    for (const key of ['yaw', 'pitch', 'dist', 'tx', 'ty', 'tz']) cam[key] = lerp(cam[key], fly[key], k);
    if (Math.abs(cam.yaw - fly.yaw) + Math.abs(cam.pitch - fly.pitch) < 1e-3 && Math.abs(cam.dist - fly.dist) < fly.dist * 1e-3) fly = null;
  }
  if (ui.mode !== 'mosaic') {
    if (ui.spin && !interacting && !fly && !flying) {
      spinT += dt;
      cam.yaw += SPIN_YAW * dt;
      // add the wobble's derivative so the user's own pitch stays the centre
      cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + Math.cos(spinT * WOBBLE_FREQ) * WOBBLE_FREQ * WOBBLE_AMP * dt));
    }
    if (autoFit && !fly && !flying) {
      cam.dist = lerp(cam.dist, fitDistance(layoutRadius + ui.tile) * 1.05, 0.05);
      cam.tx = lerp(cam.tx, 0, 0.05); cam.ty = lerp(cam.ty, 0, 0.05); cam.tz = lerp(cam.tz, 0, 0.05);
    }
  }
  if (flying) flyStep(dt);
  updateCamera();
  renderer.draw(cam, layout, sizes, {
    edges: ui.edges && ui.mode !== 'mosaic', edgeAlpha: 0.08, edgeColor: [0.75, 0.85, 1.0],
    hubColor: hubColors[ui.mode], tint: ui.mode === 'mosaic' ? ui.tint : 0, round: ui.mode === 'mosaic' ? 0.0 : 0.07,
  });

  const showLabels = ui.labels && ui.mode !== 'mosaic';
  labelsEl.hidden = !showLabels;
  if (showLabels) {
    for (let h = 0; h < hubEls.length; h++) {
      const i = N + h, [sx, sy, w] = project(layout.x[i], layout.y[i], layout.z[i]);
      const el = hubEls[h];
      if (!el) continue;
      if (w <= 0) { el.style.opacity = 0; continue; }
      const fog = Math.min(1, Math.max(0, (w - cam.fogNear) / (cam.fogFar - cam.fogNear)));
      const off = sizes[i] * 0.5 * cam.focal / w + 4;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${(sy - off).toFixed(1)}px) translate(-50%, -100%)`;
      el.style.opacity = (selected === i || hover === i ? 1 : 1 - fog * 0.85).toFixed(2);
    }
  }
  requestAnimationFrame(frame);
}

await applyMode();
requestAnimationFrame(frame);
