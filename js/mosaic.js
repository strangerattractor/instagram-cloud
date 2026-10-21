// Photomosaic of the profile picture. Cells come from a quadtree: busy areas
// (eyes, edges) split into small tiles, flat areas stay large. Posts may repeat;
// a usage penalty keeps variety.

export const loadImage = src => new Promise((res, rej) => {
  const im = new Image();
  im.onload = () => res(im);
  im.onerror = rej;
  im.src = src;
});

const S = 512;   // analysis resolution of the target

// weighted RGB distance, close enough to perceptual for this
function cost(r1, g1, b1, r2, g2, b2) {
  const rm = (r1 + r2) * 0.5, dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
  return (2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db;
}

// summed-area tables for mean and variance of any square in O(1)
function tables(img) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  const s = Math.min(img.width, img.height);
  ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data;
  const W = S + 1, sum = [0, 1, 2].map(() => new Float64Array(W * W)), sq = new Float64Array(W * W);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const o = (y * S + x) * 4, i = (y + 1) * W + x + 1;
    let q = 0;
    for (let c = 0; c < 3; c++) {
      const v = px[o + c];
      sum[c][i] = v + sum[c][i - 1] + sum[c][i - W] - sum[c][i - W - 1];
      q += v * v;
    }
    sq[i] = q + sq[i - 1] + sq[i - W] - sq[i - W - 1];
  }
  const box = (t, x, y, s) => t[(y + s) * W + x + s] - t[y * W + x + s] - t[(y + s) * W + x] + t[y * W + x];
  return (x, y, s) => {
    const a = s * s, m = sum.map(t => box(t, x, y, s) / a);
    return { mean: m, variance: box(sq, x, y, s) / a - (m[0] ** 2 + m[1] ** 2 + m[2] ** 2) };
  };
}

// cands: [{ post, tile, color }] - posts plus extra images (carousel slides) that map back to a post.
// res: smallest tiles across the image; detail 0..1: how eagerly to split; variety 0..1: repeat penalty
export function buildMosaic(cands, img, { res = 72, detail = 0.6, variety = 0.4, round = false } = {}) {
  const stats = tables(img);
  const minPx = Math.max(2, Math.floor(S / res));
  const basePx = minPx * 4;                       // three tile scales: 4x, 2x, 1x
  const thr = (70 * (1 - detail)) ** 2 + 1;
  const R = S / 2;
  const cells = [];
  const split = (x, y, s) => {
    s = Math.min(s, S - x, S - y);
    if (s <= 0) return;
    const cx = x + s / 2 - R, cy = y + s / 2 - R;
    if (round && cx * cx + cy * cy > R * R) return;   // optional circular crop like the IG avatar
    const st = stats(x, y, s);
    if (s >= minPx * 2 && st.variance > thr) {
      const h = s / 2;
      split(x, y, h); split(x + h, y, h); split(x, y + h, h); split(x + h, y + h, h);
    } else cells.push({ x, y, s, mean: st.mean });
  };
  for (let y = 0; y < S; y += basePx) for (let x = 0; x < S; x += basePx) split(x, y, basePx);

  const n = cands.length, m = cells.length;
  const col = new Float32Array(n * 3);
  cands.forEach((c, i) => { const [r, g, b] = c.color ?? [128, 128, 128]; col.set([r, g, b], i * 3); });

  // penalty scale relative to typical colour distances
  let avg = 0;
  for (let k = 0; k < 400; k++) {
    const p = (Math.random() * n) | 0, c = cells[(Math.random() * m) | 0].mean;
    avg += cost(col[p * 3], col[p * 3 + 1], col[p * 3 + 2], c[0], c[1], c[2]);
  }
  const lambda = (avg / 400) * variety * 0.25;

  const uses = new Int32Array(n);
  const post = new Int32Array(m);
  const order = [...Array(m).keys()].sort(() => Math.random() - 0.5);
  for (const ci of order) {
    const [r, g, b] = cells[ci].mean;
    let best = 0, bc = Infinity;
    for (let p = 0; p < n; p++) {
      const v = cost(col[p * 3], col[p * 3 + 1], col[p * 3 + 2], r, g, b) + lambda * uses[p] * uses[p];
      if (v < bc) { bc = v; best = p; }
    }
    post[ci] = best;
    uses[best]++;
  }

  // per tile: centre & size in image units (-0.5..0.5, y up), post mean colour, target uv rect
  const x = new Float32Array(m), y = new Float32Array(m), size = new Float32Array(m), depth = new Float32Array(m);
  const mean = new Float32Array(m * 3), rect = new Float32Array(m * 3);
  cells.forEach((c, i) => {
    x[i] = (c.x + c.s / 2) / S - 0.5;
    y[i] = 0.5 - (c.y + c.s / 2) / S;
    size[i] = c.s / S;
    depth[i] = Math.random() * 2 - 1;
    for (let k = 0; k < 3; k++) mean[i * 3 + k] = col[post[i] * 3 + k] / 255;
    rect[i * 3] = c.x / S; rect[i * 3 + 1] = c.y / S; rect[i * 3 + 2] = c.s / S;
  });
  const unique = new Set([...post].map(c => cands[c].post)).size;
  return { count: m, cand: post, x, y, size, depth, mean, rect, unique };
}
