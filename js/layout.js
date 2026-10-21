// 3D force-directed layout (d3-force semantics): link springs, many-body
// repulsion via a Barnes-Hut octree, weak centering. Typed arrays only,
// nothing is allocated per tick.

const EMPTY = -1, INTERNAL = -2;

class Octree {
  constructor(cap) {
    this.cap = cap;
    this.child = new Int32Array(cap * 8);
    this.body = new Int32Array(cap);
    this.mass = new Float32Array(cap);
    this.mx = new Float32Array(cap); this.my = new Float32Array(cap); this.mz = new Float32Array(cap);
    this.cx = new Float32Array(cap); this.cy = new Float32Array(cap); this.cz = new Float32Array(cap);
    this.hs = new Float32Array(cap);
    this.count = 0;
  }

  alloc(cx, cy, cz, hs) {
    const c = this.count++;
    this.child.fill(-1, c * 8, c * 8 + 8);
    this.body[c] = EMPTY;
    this.mass[c] = this.mx[c] = this.my[c] = this.mz[c] = 0;
    this.cx[c] = cx; this.cy[c] = cy; this.cz[c] = cz; this.hs[c] = hs;
    return c;
  }

  sub(node, o) {
    const h = this.hs[node] * 0.5;
    return this.alloc(this.cx[node] + (o & 1 ? h : -h), this.cy[node] + (o & 2 ? h : -h), this.cz[node] + (o & 4 ? h : -h), h);
  }

  build(x, y, z, w, n) {
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (let i = 0; i < n; i++) {
      x0 = Math.min(x0, x[i]); x1 = Math.max(x1, x[i]);
      y0 = Math.min(y0, y[i]); y1 = Math.max(y1, y[i]);
      z0 = Math.min(z0, z[i]); z1 = Math.max(z1, z[i]);
    }
    this.count = 0;
    this.alloc((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, Math.max(x1 - x0, y1 - y0, z1 - z0) / 2 + 1);
    this.x = x; this.y = y; this.z = z; this.w = w;
    for (let i = 0; i < n; i++) this.insert(i);
    for (let c = 0; c < this.count; c++) {
      const m = this.mass[c];
      if (m > 0) { this.mx[c] /= m; this.my[c] /= m; this.mz[c] /= m; }
    }
  }

  oct(node, i) {
    return (this.x[i] > this.cx[node] ? 1 : 0) | (this.y[i] > this.cy[node] ? 2 : 0) | (this.z[i] > this.cz[node] ? 4 : 0);
  }

  add(node, i) {
    const w = this.w[i];
    this.mass[node] += w;
    this.mx[node] += this.x[i] * w; this.my[node] += this.y[i] * w; this.mz[node] += this.z[i] * w;
  }

  insert(i) {
    let node = 0;
    for (;;) {
      this.add(node, i);
      const b = this.body[node];
      if (b === EMPTY) { this.body[node] = i; return; }
      if (b >= 0) {
        // leaf: split, unless points coincide or we ran out of nodes (then it only adds mass)
        if (this.hs[node] < 1e-2 || this.count + 2 >= this.cap) return;
        this.body[node] = INTERNAL;
        const ob = this.oct(node, b);
        const cb = this.sub(node, ob);
        this.child[node * 8 + ob] = cb;
        this.body[cb] = b;
        this.add(cb, b);
      }
      const o = this.oct(node, i);
      const c = this.child[node * 8 + o];
      if (c < 0) {
        const nc = this.sub(node, o);
        this.child[node * 8 + o] = nc;
        this.body[nc] = i;
        this.add(nc, i);
        return;
      }
      node = c;
    }
  }
}

export class Layout {
  constructor(nPosts, cap) {
    this.cap = cap;
    this.nPosts = nPosts;
    this.n = nPosts;
    this.x = new Float32Array(cap); this.y = new Float32Array(cap); this.z = new Float32Array(cap);
    this.vx = new Float32Array(cap); this.vy = new Float32Array(cap); this.vz = new Float32Array(cap);
    this.w = new Float32Array(cap);          // charge weight
    this.deg = new Float32Array(cap);
    this.anchor = new Float32Array(cap * 3);   // pinned hub positions, in link-distance units
    this.anchored = new Uint8Array(cap);
    this.tree = new Octree(cap * 4 + 64);
    this.stack = new Int32Array(1024);
    this.held = -1; this.hx = 0; this.hy = 0; this.hz = 0;

    this.alpha = 1; this.alphaTarget = 0; this.alphaMin = 0.002; this.alphaDecay = 0.015;
    this.linkDistance = 30; this.charge = 40; this.hubCharge = 4;
    this.theta2 = 0.81; this.velocityDecay = 0.45; this.gravity = 0.06; this.targetSpring = 0.04;
    this.targets = null;

    // random ball
    const r = 20 * Math.cbrt(nPosts);
    for (let i = 0; i < nPosts; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, s = r * Math.cbrt(Math.random());
      const q = Math.sqrt(1 - u * u);
      this.x[i] = Math.cos(a) * q * s; this.y[i] = u * s; this.z[i] = Math.sin(a) * q * s;
      this.w[i] = 1;
    }
  }

  // mosaic: every post springs to a fixed target, no links or repulsion
  setTargets(tx, ty, tz, m) {
    this.targets = { tx, ty, tz };
    this.n = m;
    this.edgeCount = 0;
    this.src = this.dst = new Int32Array(0);
    this.held = -1;
    this.reheat(1);
  }

  setGraph({ hubs, postLinks, hubLinks, postPairs = [] }) {
    this.targets = null;
    const np = this.nPosts, nh = hubs.length;
    this.n = np + nh;
    const edges = [];
    postLinks.forEach((l, i) => { for (const h of l) edges.push(i, np + h); });
    for (const [a, b] of hubLinks) edges.push(np + a, np + b);
    for (const [a, b] of postPairs) edges.push(a, b);
    this.src = new Int32Array(edges.filter((_, i) => i % 2 === 0));
    this.dst = new Int32Array(edges.filter((_, i) => i % 2 === 1));
    this.edgeCount = this.src.length;

    this.deg.fill(0);
    for (let e = 0; e < this.edgeCount; e++) { this.deg[this.src[e]]++; this.deg[this.dst[e]]++; }
    this.strength = new Float32Array(this.edgeCount);
    this.bias = new Float32Array(this.edgeCount);
    this.rest = new Float32Array(this.edgeCount);
    for (let e = 0; e < this.edgeCount; e++) {
      const s = this.src[e], t = this.dst[e];
      this.strength[e] = 1 / Math.min(this.deg[s], this.deg[t]);
      this.bias[e] = this.deg[s] / (this.deg[s] + this.deg[t]);
      this.rest[e] = s >= np ? 1.5 : t < np ? 0.6 : 1;   // timeline links longer, post-post shorter
    }

    // hubs start at the centroid of their posts, so mode switches morph instead of exploding
    const sx = new Float32Array(nh), sy = new Float32Array(nh), sz = new Float32Array(nh), c = new Float32Array(nh);
    postLinks.forEach((l, i) => { for (const h of l) { sx[h] += this.x[i]; sy[h] += this.y[i]; sz[h] += this.z[i]; c[h]++; } });
    for (let h = 0; h < nh; h++) {
      const k = np + h, j = () => (Math.random() - 0.5) * 10;
      this.x[k] = c[h] ? sx[h] / c[h] + j() : j() * 20;
      this.y[k] = c[h] ? sy[h] / c[h] + j() : j() * 20;
      this.z[k] = c[h] ? sz[h] / c[h] + j() : j() * 20;
      this.vx[k] = this.vy[k] = this.vz[k] = 0;
      this.w[k] = this.hubCharge * (1 + Math.sqrt(this.deg[k]) * 0.15);
    }
    this.anchored.fill(0);
    hubs.forEach((h, i) => {
      if (!h.anchor) return;
      const k = np + i;
      this.anchored[k] = 1;
      this.anchor.set(h.anchor, k * 3);
    });
    this.held = -1;
    this.reheat(1);
  }

  reheat(a = 0.5) { this.alpha = Math.max(this.alpha, a); }
  get active() { return this.alpha >= this.alphaMin || this.alphaTarget > 0; }

  tick() {
    if (!this.active) return false;
    this.alpha += (this.alphaTarget - this.alpha) * this.alphaDecay;
    const { x, y, z, vx, vy, vz, n, alpha } = this;

    if (this.targets) {
      const { tx, ty, tz } = this.targets, k = this.targetSpring, decay = 1 - this.velocityDecay * 0.6;
      for (let i = 0; i < n; i++) {
        vx[i] = (vx[i] + (tx[i] - x[i]) * k) * decay;
        vy[i] = (vy[i] + (ty[i] - y[i]) * k) * decay;
        vz[i] = (vz[i] + (tz[i] - z[i]) * k) * decay;
        x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i];
      }
      this.pin();
      return true;
    }

    // links
    const L = this.linkDistance;
    for (let e = 0; e < this.edgeCount; e++) {
      const s = this.src[e], t = this.dst[e];
      let dx = x[t] + vx[t] - x[s] - vx[s], dy = y[t] + vy[t] - y[s] - vy[s], dz = z[t] + vz[t] - z[s] - vz[s];
      let l = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-3;
      l = (l - L * this.rest[e]) / l * alpha * this.strength[e];
      dx *= l; dy *= l; dz *= l;
      const b = this.bias[e];
      vx[t] -= dx * b; vy[t] -= dy * b; vz[t] -= dz * b;
      vx[s] += dx * (1 - b); vy[s] += dy * (1 - b); vz[s] += dz * (1 - b);
    }

    // many-body repulsion
    const T = this.tree;
    T.build(x, y, z, this.w, n);
    const k = -this.charge * alpha, theta2 = this.theta2, stack = this.stack;
    for (let i = 0; i < n; i++) {
      const xi = x[i], yi = y[i], zi = z[i];
      let fx = 0, fy = 0, fz = 0, sp = 0;
      stack[sp++] = 0;
      while (sp) {
        const c = stack[--sp];
        if (T.mass[c] === 0) continue;
        const dx = T.mx[c] - xi, dy = T.my[c] - yi, dz = T.mz[c] - zi;
        let d2 = dx * dx + dy * dy + dz * dz;
        const b = T.body[c];
        const wdt = T.hs[c] * 2;
        if (b >= 0 || wdt * wdt < theta2 * d2) {
          if (b === i) continue;
          if (d2 < 1) d2 = 1;
          const f = k * T.mass[c] / d2;
          fx += dx * f; fy += dy * f; fz += dz * f;
          continue;
        }
        const base = c * 8;
        for (let o = 0; o < 8; o++) { const ch = T.child[base + o]; if (ch >= 0 && sp < stack.length) stack[sp++] = ch; }
      }
      vx[i] += fx; vy[i] += fy; vz[i] += fz;
    }

    // integrate, weak pull to origin keeps disconnected parts around
    const decay = 1 - this.velocityDecay, g = this.gravity * alpha;
    const A = this.anchor;
    for (let i = 0; i < n; i++) {
      if (this.anchored[i]) {
        // glide to the pinned spot instead of feeling forces
        vx[i] = (vx[i] + (A[i * 3] * L - x[i]) * 0.08) * 0.7;
        vy[i] = (vy[i] + (A[i * 3 + 1] * L - y[i]) * 0.08) * 0.7;
        vz[i] = (vz[i] + (A[i * 3 + 2] * L - z[i]) * 0.08) * 0.7;
      } else {
        vx[i] = (vx[i] - x[i] * g) * decay; vy[i] = (vy[i] - y[i] * g) * decay; vz[i] = (vz[i] - z[i] * g) * decay;
      }
      x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i];
    }
    this.pin();
    return true;
  }

  pin() {
    const h = this.held;
    if (h < 0) return;
    this.x[h] = this.hx; this.y[h] = this.hy; this.z[h] = this.hz;
    this.vx[h] = this.vy[h] = this.vz[h] = 0;
  }

  // 95th-percentile distance from origin: stray nodes shouldn't shrink the view
  radius() {
    const r2 = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) r2[i] = this.x[i] ** 2 + this.y[i] ** 2 + this.z[i] ** 2;
    r2.sort();
    this.maxRadius = Math.sqrt(r2[this.n - 1] ?? 0);   // for clip planes: nothing may fall outside
    return Math.sqrt(r2[Math.floor((this.n - 1) * 0.95)] ?? 0);
  }
}
