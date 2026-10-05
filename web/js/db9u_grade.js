// DB9U — chỉnh màu phía trình duyệt (port 1:1 công thức db9u/grade.py) để kéo thanh là thấy ngay,
// KHÔNG chạy lại workflow. Lưu full-res vẫn dùng bản Python (chính xác) trên ảnh gốc.
// Ảnh: 3 mặt phẳng Float32Array r,g,b (0..1), w x h.

export const HUES = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"];
const HUE_DEG = [0, 30, 60, 120, 180, 240, 270, 300];
export const SEL_TARGETS = ["reds", "yellows", "greens", "cyans", "blues", "magentas", "whites", "neutrals", "blacks"];

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (a, b, x) => { let t = (x - a) / (b - a); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); };

// ---------------------------------------------------------------- sRGB <-> linear (bảng tra)
const N_LUT = 4096;
const S2L = new Float32Array(N_LUT + 1), L2S = new Float32Array(N_LUT + 1);
for (let i = 0; i <= N_LUT; i++) {
  const x = i / N_LUT;
  S2L[i] = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
  L2S[i] = x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(Math.max(x, 1e-10), 1 / 2.4) - 0.055;
}
const lut = (T, v) => { v = clamp01(v) * N_LUT; const i = v | 0; const f = v - i; return i >= N_LUT ? T[N_LUT] : T[i] + (T[i + 1] - T[i]) * f; };

function applyLut1(L, v) { const n = L.length; const x = clamp01(v) * (n - 1); const i = x | 0; const f = x - i; return i >= n - 1 ? L[n - 1] : L[i] * (1 - f) + L[i + 1] * f; }

// ---------------------------------------------------------------- blur (3 lần box ≈ gauss)
function boxesForGauss(sigma, n = 3) {
  const wIdeal = Math.sqrt((12 * sigma * sigma / n) + 1);
  let wl = Math.floor(wIdeal); if (wl % 2 === 0) wl--;
  const wu = wl + 2;
  const mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
  const m = Math.round(mIdeal);
  const sizes = [];
  for (let i = 0; i < n; i++) sizes.push(i < m ? wl : wu);
  return sizes;
}
function boxH(src, dst, w, h, r) {
  const iarr = 1 / (r + r + 1);
  for (let y = 0; y < h; y++) {
    const o = y * w; const fv = src[o], lv = src[o + w - 1];
    let val = (r + 1) * fv;
    for (let j = 0; j < r; j++) val += src[o + Math.min(j, w - 1)];
    for (let x = 0; x < w; x++) {
      val += src[o + Math.min(x + r, w - 1)] - (x - r - 1 >= 0 ? src[o + x - r - 1] : fv);
      dst[o + x] = val * iarr;
    }
    void lv;
  }
}
function boxV(src, dst, w, h, r) {
  const iarr = 1 / (r + r + 1);
  for (let x = 0; x < w; x++) {
    const fv = src[x];
    let val = (r + 1) * fv;
    for (let j = 0; j < r; j++) val += src[Math.min(j, h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      val += src[Math.min(y + r, h - 1) * w + x] - (y - r - 1 >= 0 ? src[(y - r - 1) * w + x] : fv);
      dst[y * w + x] = val * iarr;
    }
  }
}
export function blur(src, w, h, sigma) {
  if (sigma <= 0.3) return Float32Array.from(src);
  const bx = boxesForGauss(sigma);
  let a = Float32Array.from(src), b = new Float32Array(src.length);
  for (const s of bx) { const r = (s - 1) >> 1; if (r < 1) continue; boxH(a, b, w, h, r); boxV(b, a, w, h, r); }
  return a;
}
function resizeBilinear(src, w, h, W, H) {
  const out = new Float32Array(W * H);
  const sx = w / W, sy = h / H;
  for (let y = 0; y < H; y++) {
    const fy = Math.min(h - 1, Math.max(0, (y + 0.5) * sy - 0.5)); const y0 = fy | 0, y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < W; x++) {
      const fx = Math.min(w - 1, Math.max(0, (x + 0.5) * sx - 0.5)); const x0 = fx | 0, x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      const a = src[y0 * w + x0] + (src[y0 * w + x1] - src[y0 * w + x0]) * tx;
      const b = src[y1 * w + x0] + (src[y1 * w + x1] - src[y1 * w + x0]) * tx;
      out[y * W + x] = a + (b - a) * ty;
    }
  }
  return out;
}
function resizeArea(src, w, h, W, H) { // thu nhỏ trung bình khối
  const out = new Float32Array(W * H), cnt = new Float32Array(W * H);
  for (let y = 0; y < h; y++) {
    const Y = Math.min(H - 1, (y * H / h) | 0);
    for (let x = 0; x < w; x++) { const i = Y * W + Math.min(W - 1, (x * W / w) | 0); out[i] += src[y * w + x]; cnt[i]++; }
  }
  for (let i = 0; i < out.length; i++) out[i] /= Math.max(1, cnt[i]);
  return out;
}
function minFilter(src, w, h, r) {
  const t = new Float32Array(src.length), o = new Float32Array(src.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = 1e9; for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++) m = Math.min(m, src[y * w + k]); t[y * w + x] = m;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = 1e9; for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k++) m = Math.min(m, t[k * w + x]); o[y * w + x] = m;
  }
  return o;
}

// ---------------------------------------------------------------- helpers ảnh
export function fromImageData(id) {
  const n = id.width * id.height, d = id.data;
  const r = new Float32Array(n), g = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { r[i] = d[j] / 255; g[i] = d[j + 1] / 255; b[i] = d[j + 2] / 255; }
  return { w: id.width, h: id.height, r, g, b };
}
export function toImageData(im, id) {
  const n = im.w * im.h, d = id.data;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    d[j] = clamp01(im.r[i]) * 255 + 0.5; d[j + 1] = clamp01(im.g[i]) * 255 + 0.5; d[j + 2] = clamp01(im.b[i]) * 255 + 0.5; d[j + 3] = 255;
  }
  return id;
}
export function downscale(im, k) {
  const W = Math.max(1, Math.round(im.w / k)), H = Math.max(1, Math.round(im.h / k));
  return { w: W, h: H, r: resizeArea(im.r, im.w, im.h, W, H), g: resizeArea(im.g, im.w, im.h, W, H), b: resizeArea(im.b, im.w, im.h, W, H) };
}
const lumOf = (r, g, b) => r * 0.2126 + g * 0.7152 + b * 0.0722;
function lumPlane(im) { const n = im.w * im.h, L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = lumOf(im.r[i], im.g[i], im.b[i]); return L; }
function setLum(im, Lnew, Lold) { // = _set_lum
  const n = im.w * im.h;
  for (let i = 0; i < n; i++) {
    const lo = Lold[i], ln = Lnew[i]; const k = ln / Math.max(lo, 1e-4), a = ln - lo, w = smooth(0, 0.05, lo);
    im.r[i] = clamp01(im.r[i] * k * w + (im.r[i] + a) * (1 - w));
    im.g[i] = clamp01(im.g[i] * k * w + (im.g[i] + a) * (1 - w));
    im.b[i] = clamp01(im.b[i] * k * w + (im.b[i] + a) * (1 - w));
  }
}

// ---------------------------------------------------------------- 1. Basic
function basic(im, p) {
  const n = im.w * im.h, { r, g, b } = im;
  const ex = Math.pow(2, p.exposure || 0), t = (p.temperature || 0) / 100, gg = (p.tint || 0) / 100;
  const mr = ex * (1 + 0.25 * t), mg = ex * (1 - 0.20 * gg), mb = ex * (1 - 0.25 * t);
  if (mr !== 1 || mg !== 1 || mb !== 1) {
    const mk = (m) => { const T = new Float32Array(2049); for (let j = 0; j <= 2048; j++) T[j] = lut(L2S, lut(S2L, j / 2048) * m); return T; };
    const Tr = mk(mr), Tg = mk(mg), Tb = mk(mb);
    for (let i = 0; i < n; i++) { r[i] = applyLut1(Tr, r[i]); g[i] = applyLut1(Tg, g[i]); b[i] = applyLut1(Tb, b[i]); }
  }
  const tone = p.whites || p.blacks || p.highlights || p.shadows || p.contrast;
  if (tone) {
    const L = lumPlane(im), L2 = Float32Array.from(L);
    if (p.whites || p.blacks) {
      const wp = 1 - 0.25 * (p.whites || 0) / 100, bp = -0.15 * (p.blacks || 0) / 100, d = Math.max(wp - bp, 1e-3);
      for (let i = 0; i < n; i++) L2[i] = clamp01((L2[i] - bp) / d);
    }
    if (p.highlights || p.shadows) {
      const Lb = blur(L2, im.w, im.h, Math.max(2, Math.min(im.w, im.h) / 100));
      const hh = 0.30 * (p.highlights || 0) / 100, ss = 0.30 * (p.shadows || 0) / 100;
      for (let i = 0; i < n; i++) {
        const hm = smooth(0.45, 1, Lb[i]), sm = 1 - smooth(0, 0.55, Lb[i]);
        L2[i] = clamp01(L2[i] + hh * hm * L2[i] + ss * sm * (1 - L2[i]));
      }
    }
    if (p.contrast) {
      const k = p.contrast / 100, lo = 1 / (1 + Math.exp(4)), hi = 1 / (1 + Math.exp(-4));
      const T = new Float32Array(1025);
      for (let j = 0; j <= 1024; j++) { const x = j / 1024; T[j] = k > 0 ? x + k * ((1 / (1 + Math.exp(-(x - 0.5) * 8)) - lo) / (hi - lo) - x) : clamp01(0.5 + (x - 0.5) * (1 + k)); }
      for (let i = 0; i < n; i++) L2[i] = applyLut1(T, L2[i]);
    }
    setLum(im, L2, L);
  }
  if (p.vibrance || p.saturation) {
    const sa = 1 + (p.saturation || 0) / 100, vi = (p.vibrance || 0) / 100;
    for (let i = 0; i < n; i++) {
      const L = lumOf(r[i], g[i], b[i]); const s = Math.max(r[i], g[i], b[i]) - Math.min(r[i], g[i], b[i]);
      const f = sa + vi * clamp01(1 - s);
      r[i] = clamp01(L + (r[i] - L) * f); g[i] = clamp01(L + (g[i] - L) * f); b[i] = clamp01(L + (b[i] - L) * f);
    }
  }
}

// ---------------------------------------------------------------- 2. Presence
function dehaze(im, amount) {
  const W = im.w, H = im.h, n = W * H;
  const f = Math.max(1, Math.min(W, H) / 768), w = Math.max(8, Math.round(W / f)), h = Math.max(8, Math.round(H / f));
  const sr = resizeArea(im.r, W, H, w, h), sg = resizeArea(im.g, W, H, w, h), sb = resizeArea(im.b, W, H, w, h);
  const mn = new Float32Array(w * h); for (let i = 0; i < w * h; i++) mn[i] = Math.min(sr[i], sg[i], sb[i]);
  const dc = minFilter(mn, w, h, 7);
  const k = Math.max(1, (w * h / 1000) | 0);
  const idx = Array.from(dc.keys()).sort((a, b) => dc[b] - dc[a]).slice(0, k);
  let A = [0, 0, 0]; for (const i of idx) { A[0] += sr[i]; A[1] += sg[i]; A[2] += sb[i]; }
  A = A.map((v) => Math.max(0.3, v / k));
  if (amount < 0) {
    const a = (-amount / 100) * 0.6;
    for (let i = 0; i < n; i++) { im.r[i] = clamp01(im.r[i] + (A[0] - im.r[i]) * a); im.g[i] = clamp01(im.g[i] + (A[1] - im.g[i]) * a); im.b[i] = clamp01(im.b[i] + (A[2] - im.b[i]) * a); }
    return;
  }
  const ww = 0.95 * amount / 100;
  const m2 = new Float32Array(w * h); for (let i = 0; i < w * h; i++) m2[i] = Math.min(sr[i] / A[0], sg[i] / A[1], sb[i] / A[2]);
  const d2 = minFilter(m2, w, h, 7); for (let i = 0; i < w * h; i++) d2[i] = 1 - ww * d2[i];
  const t = resizeBilinear(blur(d2, w, h, 4), w, h, W, H);
  for (let i = 0; i < n; i++) {
    const tt = Math.max(t[i], 0.15);
    im.r[i] = clamp01((im.r[i] - A[0]) / tt + A[0]); im.g[i] = clamp01((im.g[i] - A[1]) / tt + A[1]); im.b[i] = clamp01((im.b[i] - A[2]) / tt + A[2]);
  }
}
function presence(im, p, fullW) {
  const n = im.w * im.h;
  if (p.dehaze) dehaze(im, p.dehaze);
  if (p.clarity || p.texture) {
    const L = lumPlane(im), L2 = Float32Array.from(L);
    if (p.clarity) {
      const Lb = blur(L, im.w, im.h, Math.max(4, Math.min(im.w, im.h) / 150)), c = p.clarity / 100 * 0.6;
      for (let i = 0; i < n; i++) L2[i] += c * (L[i] - Lb[i]) * (1 - Math.abs(2 * L[i] - 1));
    }
    if (p.texture) { // sigma 1.5px ở ảnh full-res -> quy đổi theo tỉ lệ preview
      const sig = Math.max(0.35, 1.5 * im.w / (fullW || im.w));
      const Lb = blur(L, im.w, im.h, sig), c = p.texture / 100 * 0.8;
      for (let i = 0; i < n; i++) L2[i] += c * (L[i] - Lb[i]);
    }
    for (let i = 0; i < n; i++) L2[i] = clamp01(L2[i]);
    setLum(im, L2, L);
  }
  if (p.vignette) {
    const v = p.vignette / 100;
    for (let y = 0; y < im.h; y++) {
      const yy = im.h > 1 ? -1 + 2 * y / (im.h - 1) : 0;
      for (let x = 0; x < im.w; x++) {
        const xx = im.w > 1 ? -1 + 2 * x / (im.w - 1) : 0, i = y * im.w + x;
        const m = 1 + v * smooth(0.35, 1, Math.sqrt(xx * xx + yy * yy) / Math.SQRT2);
        im.r[i] = clamp01(im.r[i] * m); im.g[i] = clamp01(im.g[i] * m); im.b[i] = clamp01(im.b[i] * m);
      }
    }
  }
  if (p.grain) { // nhiễu giả lập (bản lưu dùng seed cố định của Python)
    let s = 12345; const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    const a = p.grain / 100 * 0.06;
    for (let i = 0; i < n; i++) {
      const L = lumOf(im.r[i], im.g[i], im.b[i]); const z = (rnd() + rnd() + rnd() - 1.5) * 2;
      const d = z * a * (1 - Math.abs(2 * L - 1));
      im.r[i] = clamp01(im.r[i] + d); im.g[i] = clamp01(im.g[i] + d); im.b[i] = clamp01(im.b[i] + d);
    }
  }
}

// ---------------------------------------------------------------- 3. Curves
export function parsePoints(s) {
  const pts = [];
  for (const tok of String(s || "").replace(/;/g, " ").split(/\s+/)) {
    if (!tok.includes(",")) continue;
    const [a, b] = tok.split(",").map(Number); if (isFinite(a) && isFinite(b)) pts.push([a / 255, b / 255]);
  }
  if (!pts.length) return null;
  const m = new Map(); for (const [x, y] of pts) m.set(x, y);
  const o = [...m.entries()].sort((a, b) => a[0] - b[0]);
  if (o[0][0] > 0) o.unshift([0, 0]);
  if (o[o.length - 1][0] < 1) o.push([1, 1]);
  return o;
}
export function curveLut(pts, n = 1024) { // Fritsch–Carlson như Python
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]), out = new Float32Array(n);
  if (xs.length === 2) { for (let i = 0; i < n; i++) { const q = i / (n - 1); out[i] = clamp01(ys[0] + (ys[1] - ys[0]) * (q - xs[0]) / Math.max(xs[1] - xs[0], 1e-9)); } return out; }
  const d = []; for (let i = 0; i < xs.length - 1; i++) d.push((ys[i + 1] - ys[i]) / Math.max(xs[i + 1] - xs[i], 1e-9));
  const m = new Array(ys.length).fill(0);
  for (let i = 1; i < ys.length - 1; i++) m[i] = d[i - 1] * d[i] > 0 ? (d[i - 1] + d[i]) / 2 : 0;
  m[0] = d[0]; m[m.length - 1] = d[d.length - 1];
  for (let i = 0; i < d.length; i++) {
    if (d[i] === 0) { m[i] = m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  let k = 0;
  for (let i = 0; i < n; i++) {
    const q = i / (n - 1);
    while (k < xs.length - 2 && q > xs[k + 1]) k++;
    const h = xs[k + 1] - xs[k], t = (q - xs[k]) / Math.max(h, 1e-9);
    const t2 = t * t, t3 = t2 * t;
    out[i] = clamp01((2 * t3 - 3 * t2 + 1) * ys[k] + (t3 - 2 * t2 + t) * h * m[k] + (-2 * t3 + 3 * t2) * ys[k + 1] + (t3 - t2) * h * m[k + 1]);
  }
  return out;
}
function curves(im, p) {
  const n = im.w * im.h;
  for (const [ch, key] of [["r", "curve_red"], ["g", "curve_green"], ["b", "curve_blue"]]) {
    const pt = parsePoints(p[key]); if (!pt) continue; const L = curveLut(pt), a = im[ch];
    for (let i = 0; i < n; i++) a[i] = applyLut1(L, a[i]);
  }
  const pm = parsePoints(p.curve_master);
  if (pm) { const L = curveLut(pm); for (let i = 0; i < n; i++) { im.r[i] = applyLut1(L, im.r[i]); im.g[i] = applyLut1(L, im.g[i]); im.b[i] = applyLut1(L, im.b[i]); } }
}

// ---------------------------------------------------------------- 4. HSL
function hueWeights(h, out) {
  for (let c = 0; c < 8; c++) {
    const C = HUE_DEG[c], left = c > 0 ? HUE_DEG[c - 1] : HUE_DEG[7] - 360, right = c < 7 ? HUE_DEG[c + 1] : HUE_DEG[0] + 360;
    const dh = ((((h - C + 180) % 360) + 360) % 360) - 180;
    const w = dh < 0 ? 1 + dh / (C - left) : 1 - dh / (right - C);
    out[c] = w < 0 ? 0 : w > 1 ? 1 : w;
  }
}
const HW_RES = 4; // bảng trọng số dải màu theo 0.25°
const HW_TAB = (() => { const T = new Float32Array(360 * HW_RES * 8), w = new Float32Array(8); for (let k = 0; k < 360 * HW_RES; k++) { hueWeights(k / HW_RES, w); T.set(w, k * 8); } return T; })();
function hsl(im, p) {
  const hs = HUES.map((c) => (p[`hsl_${c}_hue`] || 0) / 100 * 30), ss = HUES.map((c) => (p[`hsl_${c}_sat`] || 0) / 100), ls = HUES.map((c) => (p[`hsl_${c}_lum`] || 0) / 100);
  if (![...hs, ...ss, ...ls].some((v) => v)) return;
  const n = im.w * im.h, R_ = im.r, G_ = im.g, B_ = im.b;
  for (let i = 0; i < n; i++) {
    const r = R_[i], g = G_[i], b = B_[i];
    const mx = r > g ? (r > b ? r : b) : (g > b ? g : b), mn = r < g ? (r < b ? r : b) : (g < b ? g : b), d = mx - mn;
    if (d <= 1e-6 || mx <= 1e-6) continue; // xám: trọng số = 0 -> không đổi
    let H;
    if (mx === r) { H = (g - b) / d; if (H < 0) H += 6; } else if (mx === g) H = (b - r) / d + 2; else H = (r - g) / d + 4;
    H *= 60;
    let S = d / mx; const V = mx, sq = Math.sqrt(S), L = r * 0.2126 + g * 0.7152 + b * 0.0722;
    let o = ((H * HW_RES) | 0) % (360 * HW_RES) * 8, dH = 0, dS = 0, dL = 0;
    for (let c = 0; c < 8; c++) { const ww = HW_TAB[o + c] * sq; if (ww) { dH += ww * hs[c]; dS += ww * ss[c]; dL += ww * ls[c]; } }
    if (!dH && !dS && !dL) continue;
    H += dH; if (H < 0) H += 360; else if (H >= 360) H -= 360;
    S = S * (1 + dS); S = S < 0 ? 0 : S > 1 ? 1 : S;
    const C = V * S, hp = H / 60, X = C * (1 - Math.abs((hp % 2) - 1)), m = V - C; let R = 0, G = 0, B = 0;
    const k = (hp | 0) % 6;
    if (k === 0) { R = C; G = X; } else if (k === 1) { R = X; G = C; } else if (k === 2) { G = C; B = X; } else if (k === 3) { G = X; B = C; } else if (k === 4) { R = X; B = C; } else { R = C; B = X; }
    R = clamp01(R + m); G = clamp01(G + m); B = clamp01(B + m);
    const Lt = clamp01(L * (1 + 0.5 * dL)), Lo = R * 0.2126 + G * 0.7152 + B * 0.0722;
    const kk = Lt / Math.max(Lo, 1e-4), a = Lt - Lo, w = smooth(0, 0.05, Lo);
    R_[i] = clamp01(R * kk * w + (R + a) * (1 - w)); G_[i] = clamp01(G * kk * w + (G + a) * (1 - w)); B_[i] = clamp01(B * kk * w + (B + a) * (1 - w));
  }
}

// ---------------------------------------------------------------- 5. Color balance · 6. Selective
function colorBalance(im, p) {
  const g = (t) => [p[`cb_${t}_cyan_red`] || 0, p[`cb_${t}_magenta_green`] || 0, p[`cb_${t}_yellow_blue`] || 0];
  const S = g("shadows"), M = g("midtones"), Hh = g("highlights");
  if (![...S, ...M, ...Hh].some((v) => v)) return;
  const n = im.w * im.h, L0 = lumPlane(im), L1 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const L = L0[i], ws = 1 - smooth(0, 0.5, L), wh = smooth(0.5, 1, L), wm = clamp01(1 - ws - wh);
    im.r[i] = clamp01(im.r[i] + (ws * S[0] + wm * M[0] + wh * Hh[0]) / 100 * 0.15);
    im.g[i] = clamp01(im.g[i] + (ws * S[1] + wm * M[1] + wh * Hh[1]) / 100 * 0.15);
    im.b[i] = clamp01(im.b[i] + (ws * S[2] + wm * M[2] + wh * Hh[2]) / 100 * 0.15);
    L1[i] = lumOf(im.r[i], im.g[i], im.b[i]);
  }
  setLum(im, L0, L1);
}
function selWeight(r, g, b, t) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), mid = r + g + b - mx - mn;
  switch (t) {
    case "reds": return r >= mx ? mx - mid : 0;
    case "greens": return g >= mx ? mx - mid : 0;
    case "blues": return b >= mx ? mx - mid : 0;
    case "yellows": return b <= mn ? mid - mn : 0;
    case "cyans": return r <= mn ? mid - mn : 0;
    case "magentas": return g <= mn ? mid - mn : 0;
    case "whites": return clamp01((mn - 0.5) * 2);
    case "blacks": return clamp01((0.5 - mx) * 2);
    default: return clamp01((1 - Math.abs(mx - 0.5) * 2) * (1 - Math.abs(mn - 0.5) * 2));
  }
}
function selective(im, p) {
  const c = (p.sel_cyan || 0) / 100, m = (p.sel_magenta || 0) / 100, y = (p.sel_yellow || 0) / 100, k = (p.sel_black || 0) / 100;
  if (!c && !m && !y && !k) return;
  const n = im.w * im.h, t = p.sel_target || "reds";
  for (let i = 0; i < n; i++) {
    const r = im.r[i], g = im.g[i], b = im.b[i], w = selWeight(r, g, b, t);
    if (!w) continue;
    im.r[i] = clamp01(r - w * (c + k) * (1 - r)); im.g[i] = clamp01(g - w * (m + k) * (1 - g)); im.b[i] = clamp01(b - w * (y + k) * (1 - b));
  }
}

// ---------------------------------------------------------------- 7. LUT .cube
export function parseCube(text) {
  let size = 0; const v = [];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim(); if (!t || t[0] === "#") continue;
    if (/^LUT_3D_SIZE/i.test(t)) { size = parseInt(t.split(/\s+/).pop()); continue; }
    const q = t.split(/\s+/); if (q.length === 3 && q.every((s) => isFinite(+s))) v.push(+q[0], +q[1], +q[2]);
  }
  if (!size || v.length < size ** 3 * 3) throw new Error("File .cube không hợp lệ");
  return { size, data: Float32Array.from(v.slice(0, size ** 3 * 3)) };
}
function applyCube(im, cube, s) {
  const N = cube.size, D = cube.data, n = im.w * im.h, S = N - 1;
  const at = (ri, gi, bi, c) => D[((bi * N + gi) * N + ri) * 3 + c]; // .cube: r nhanh nhất
  for (let i = 0; i < n; i++) {
    const R = clamp01(im.r[i]) * S, G = clamp01(im.g[i]) * S, B = clamp01(im.b[i]) * S;
    const r0 = Math.min(S - 1, R | 0), g0 = Math.min(S - 1, G | 0), b0 = Math.min(S - 1, B | 0), fr = R - r0, fg = G - g0, fb = B - b0;
    const o = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const c00 = at(r0, g0, b0, c) * (1 - fr) + at(r0 + 1, g0, b0, c) * fr, c10 = at(r0, g0 + 1, b0, c) * (1 - fr) + at(r0 + 1, g0 + 1, b0, c) * fr;
      const c01 = at(r0, g0, b0 + 1, c) * (1 - fr) + at(r0 + 1, g0, b0 + 1, c) * fr, c11 = at(r0, g0 + 1, b0 + 1, c) * (1 - fr) + at(r0 + 1, g0 + 1, b0 + 1, c) * fr;
      o[c] = (c00 * (1 - fg) + c10 * fg) * (1 - fb) + (c01 * (1 - fg) + c11 * fg) * fb;
    }
    im.r[i] = clamp01(im.r[i] + (o[0] - im.r[i]) * s); im.g[i] = clamp01(im.g[i] + (o[1] - im.g[i]) * s); im.b[i] = clamp01(im.b[i] + (o[2] - im.b[i]) * s);
  }
}

// ---------------------------------------------------------------- pipeline = finish.apply_all
export function applyAll(src, p, opts = {}) {
  const im = { w: src.w, h: src.h, r: Float32Array.from(src.r), g: Float32Array.from(src.g), b: Float32Array.from(src.b) };
  basic(im, p);
  if (p.clarity || p.texture || p.dehaze || p.vignette || p.grain) presence(im, p, opts.fullW);
  curves(im, p);
  hsl(im, p);
  colorBalance(im, p);
  selective(im, p);
  if (opts.cube && (p.lut_strength ?? 1) > 0) applyCube(im, opts.cube, p.lut_strength ?? 1);
  return im;
}

export function histogram(im, step = 2) {
  const H = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  for (let i = 0; i < im.w * im.h; i += step) {
    H[0][(clamp01(im.r[i]) * 255) | 0]++; H[1][(clamp01(im.g[i]) * 255) | 0]++; H[2][(clamp01(im.b[i]) * 255) | 0]++;
  }
  return H;
}

// ---------------------------------------------------------------- chạy trong Web Worker (không đơ giao diện)
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  const store = {};
  self.onmessage = (e) => {
    const m = e.data;
    if (m.type === "src") { store[m.key] = m.im; return; }
    if (m.type === "cube") { store.__cube = m.cube; return; }
    if (m.type === "render") {
      const src = store[m.key]; if (!src) { self.postMessage({ type: "done", id: m.id, img: new ImageData(1, 1), hist: null }); return; }
      const out = applyAll(src, m.params, { cube: store.__cube, fullW: m.fullW });
      const id = new ImageData(out.w, out.h);
      toImageData(out, id);
      self.postMessage({ type: "done", id: m.id, key: m.key, img: id, hist: histogram(out, 3) }, [id.data.buffer]);
    }
  };
}
