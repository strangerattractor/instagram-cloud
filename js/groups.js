// Turns posts into a graph: post nodes linked to hub nodes (hashtags,
// caption words or weeks), plus direct post-post links for posts no hub
// covers. The layout does the clustering.

const STOP = new Set(`
a an and are as at be but by for from has have i in is it its my of on or so that the this to was we with you your
just very more most some what when where which who will would can could should about after again all also any because
been before being both did does doing down during each few here into just like made make many much only other our out
over same than then there these they those through today under until upon well were while yours them their his her
der die das und ist ein eine einen einem einer nicht mit auf für von den dem des sich auch als noch nach wie bei aus
heute oder aber wenn dann schon sehr mehr hier dort wird werden wurde haben hatte sind war waren kann können durch
über unter immer wieder diese dieser dieses jetzt mal mein meine meinen unser unsere euch ihr ihre`.split(/\s+/));

const TAG_RE = /#[\p{L}\p{N}_]+/gu;

export const hashtags = caption =>
  new Set((caption.match(TAG_RE) ?? []).map(t => t.slice(1).toLowerCase()));

export const words = caption =>
  new Set((caption.replace(TAG_RE, ' ').replace(/@[\w.]+|https?:\/\/\S+/g, ' ').toLowerCase()
    .match(/\p{L}{4,}/gu) ?? []).filter(w => !STOP.has(w)));

function jaccard(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
}

function byTerms(posts, termsOf, k, prefix) {
  const n = posts.length;
  const terms = posts.map(p => termsOf(p.caption ?? ''));
  const df = new Map();
  for (const t of terms) for (const w of t) df.set(w, (df.get(w) ?? 0) + 1);
  // terms on most posts (#everyday) say nothing about clusters
  const cand = [...df].filter(([, c]) => c >= 2 && c <= n * 0.6).sort((a, b) => b[1] - a[1]).map(([w]) => w);

  // half the hubs by frequency, the rest by how many still-uncovered posts they pick up
  const top = cand.slice(0, Math.ceil(k / 2));
  const covered = new Uint8Array(n);
  const mark = w => terms.forEach((t, i) => { if (t.has(w)) covered[i] = 1; });
  top.forEach(mark);
  while (top.length < k) {
    let best = null, bs = 0;
    for (const w of cand) {
      if (top.includes(w)) continue;
      let s = 0;
      for (let i = 0; i < n; i++) if (!covered[i] && terms[i].has(w)) s++;
      if (s > bs) { bs = s; best = w; }
    }
    if (!best) break;
    top.push(best);
    mark(best);
  }

  const index = new Map(top.map((w, i) => [w, i]));
  const hubs = top.map(w => ({ label: prefix + w }));
  const postLinks = terms.map(t => [...t].filter(w => index.has(w)).map(w => index.get(w)));

  // uncovered posts: link to their two most similar posts, else to their neighbour in time
  const postPairs = [];
  for (let i = 0; i < n; i++) {
    if (postLinks[i].length) continue;
    const sims = [];
    for (let j = 0; j < n; j++) if (j !== i) { const s = jaccard(terms[i], terms[j]); if (s > 0) sims.push([s, j]); }
    sims.sort((a, b) => b[0] - a[0]);
    const near = sims.slice(0, 2).map(([, j]) => j);
    if (!near.length) near.push(i + 1 < n ? i + 1 : i - 1);   // posts are sorted by time
    for (const j of near) postPairs.push([i, j]);
  }
  return { hubs, postLinks, hubLinks: [], postPairs };
}

const DAY = 864e5, YEAR = 365.2425 * DAY;
const mondayOf = d => { const t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()); return t - ((new Date(t).getUTCDay() + 6) % 7) * DAY; };

// weekly hubs on a helix: one turn per year, newest at the top
function byTime(posts) {
  const weeks = posts.map(p => mondayOf(new Date(p.timestamp)));
  const keys = [...new Set(weeks)].sort((a, b) => a - b);
  const index = new Map(keys.map((k, i) => [k, i]));
  const t0 = keys[0], span = (keys.at(-1) - t0) / YEAR;
  let lastYear = null;
  const hubs = keys.map(k => {
    const t = (k - t0) / YEAR;
    const year = new Date(k).getUTCFullYear();
    const major = year !== lastYear;
    lastYear = year;
    const a = t * Math.PI * 2;
    return {
      label: major ? String(year) : '',
      title: `Week of ${new Date(k).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}`,
      // in units of link distance; the layout scales it
      anchor: [Math.cos(a) * 8, (t - span / 2) * 5, Math.sin(a) * 8],
    };
  });
  // chain consecutive weeks into the visible time line
  const hubLinks = keys.slice(1).map((_, i) => [i, i + 1]);
  return { hubs, postLinks: weeks.map(w => [index.get(w)]), hubLinks, postPairs: [] };
}

export function buildGroups(posts, mode, k) {
  if (mode === 'words') return byTerms(posts, words, k, '');
  if (mode === 'time') return byTime(posts);
  return byTerms(posts, hashtags, k, '#');
}
