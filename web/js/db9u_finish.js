// DB9U Finish v0.7 — node: khung so sánh trước/sau + 3 nút. Editor kiểu Lightroom:
// chỉnh màu chạy NGAY TRONG TRÌNH DUYỆT (Web Worker, không chạy lại workflow). Chỉ "Lưu full-res" mới chạy lệnh.
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { applyAll, fromImageData, toImageData, downscale, histogram, parseCube, parsePoints, curveLut, HUES } from "./db9u_grade.js";

const NON_GRADE = new Set(["save_file", "live_edit", "filename_prefix", "save_to", "local_folder", "format", "preview_max"]);
const HUE_VN = { red: "Đỏ", orange: "Cam", yellow: "Vàng", green: "Lục", aqua: "Lam ngọc", blue: "Lam", purple: "Tím", magenta: "Hồng" };
const HUE_CSS = { red: "#e5484d", orange: "#f76b15", yellow: "#f5d90a", green: "#46a758", aqua: "#12a594", blue: "#0090ff", purple: "#8e4ec6", magenta: "#d6409f" };
const DRAFT_MAX = 1400, THUMB_MAX = 1280;

const defaultOf = (name, w) => (name === "lut_strength" ? 1 : name === "sel_target" ? "reds" : name === "lut" ? "none" : typeof w?.value === "string" ? "" : 0);
const viewUrl = (im) => api.apiURL(`/view?filename=${encodeURIComponent(im.filename)}&type=${im.type}&subfolder=${encodeURIComponent(im.subfolder || "")}&t=${Date.now()}`);
const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* bỏ qua */ } };

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "style") e.style.cssText = v;
    else if (k === "cls") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
function hideWidget(w) {
  if (w.db9uHidden) return;
  w.db9uHidden = true; w.origType = w.type; w.type = "db9u-hidden";
  w.computeSize = () => [0, -4]; w.draw = () => {}; w.hidden = true;
  w.options = { ...(w.options || {}), hidden: true };
}
function loadImage(url) {
  return new Promise((res, rej) => { const i = new Image(); i.crossOrigin = "anonymous"; i.onload = () => res(i); i.onerror = rej; i.src = url; });
}
function imageToPlanes(img) {
  const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
  const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(img, 0, 0);
  return fromImageData(x.getImageData(0, 0, c.width, c.height));
}
function idToCanvas(id, cv) {
  cv = cv || document.createElement("canvas");
  if (cv.width !== id.width || cv.height !== id.height) { cv.width = id.width; cv.height = id.height; }
  cv.getContext("2d").putImageData(id, 0, 0); return cv;
}

// ============================================================ bộ render (Web Worker)
let WORKER = null;
function getWorker() {
  if (WORKER !== null) return WORKER;
  try { WORKER = new Worker(new URL("./db9u_grade.js", import.meta.url), { type: "module" }); }
  catch (e) { console.warn("[DB9U] không tạo được Worker, render trên luồng chính", e); WORKER = false; }
  return WORKER;
}
let SEQ = 0;
class Renderer {
  constructor() {
    this.src = {}; this.pending = {}; this.busy = {}; this.cb = new Map(); this.cube = null; this.fullW = 0;
    const w = getWorker(); if (w) w.addEventListener("message", (e) => this.onMsg(e.data));
  }
  setSource(planes) {
    this.fullW = planes.w;
    const lv = { full: planes, draft: planes.w > DRAFT_MAX ? downscale(planes, planes.w / DRAFT_MAX) : planes,
      thumb: planes.w > THUMB_MAX ? downscale(planes, planes.w / THUMB_MAX) : planes };
    this.uid = ++SEQ;
    for (const [k, im] of Object.entries(lv)) {
      const key = `${this.uid}:${k}`; this.src[k] = { key, im };
      const w = getWorker(); if (w) w.postMessage({ type: "src", key, im });
    }
  }
  setCube(cube) { this.cube = cube; const w = getWorker(); if (w) w.postMessage({ type: "cube", cube }); }
  render(level, params) { // gộp yêu cầu: mỗi mức chỉ giữ yêu cầu mới nhất
    return new Promise((resolve) => {
      if (!this.src[level]) return resolve(null);
      if (this.busy[level]) { this.pending[level]?.resolve(null); this.pending[level] = { params, resolve }; return; }
      this._go(level, params, resolve);
    });
  }
  _go(level, params, resolve) {
    this.busy[level] = true;
    const s = this.src[level], w = getWorker(), id = ++SEQ;
    if (w) { this.cb.set(id, { level, resolve }); w.postMessage({ type: "render", id, key: s.key, params, fullW: this.fullW }); return; }
    setTimeout(() => {
      const out = applyAll(s.im, params, { cube: this.cube, fullW: this.fullW });
      this._done(level, resolve, { img: toImageData(out, new ImageData(out.w, out.h)), hist: histogram(out, 3) });
    }, 0);
  }
  onMsg(m) { if (m.type !== "done") return; const c = this.cb.get(m.id); if (!c) return; this.cb.delete(m.id); this._done(c.level, c.resolve, m); }
  _done(level, resolve, m) {
    this.busy[level] = false; resolve(m);
    const p = this.pending[level]; if (p) { this.pending[level] = null; this._go(level, p.params, p.resolve); }
  }
}

// ============================================================ CSS (tông Lightroom Classic)
const CSS = `
.db9u-ov{position:fixed;inset:0;z-index:10000;background:#1a1a1a;display:flex;font:12px/1.35 "Segoe UI",system-ui,sans-serif;color:#c8c8c8;user-select:none}
.db9u-view{flex:1;position:relative;overflow:hidden;background:#141414}
.db9u-view canvas{position:absolute;left:0;top:0;width:100%;display:block;cursor:grab}
.db9u-tb{position:absolute;left:0;right:0;bottom:0;height:38px;background:#222;border-top:1px solid #111;display:flex;align-items:center;gap:6px;padding:0 10px}
.db9u-tb .sp{flex:1}
.db9u-tb .st{color:#8a8a8a;font-size:11px}
.db9u-b{background:#333;color:#ddd;border:1px solid #444;border-radius:3px;padding:4px 10px;cursor:pointer;font:inherit}
.db9u-b:hover{background:#3d3d3d}.db9u-b.on{background:#505050;border-color:#777;color:#fff}
.db9u-b.pri{background:#3b6fd8;border-color:#3b6fd8;color:#fff}.db9u-b.pri:hover{background:#4b7fe8}
.db9u-side{width:318px;background:#262626;border-left:1px solid #111;display:flex;flex-direction:column}
.db9u-hist{height:118px;background:#1d1d1d;border-bottom:1px solid #111}
.db9u-hist canvas{width:100%;height:100%;display:block}
.db9u-scroll{flex:1;overflow-y:auto;overflow-x:hidden}
.db9u-scroll::-webkit-scrollbar{width:8px}.db9u-scroll::-webkit-scrollbar-thumb{background:#444;border-radius:4px}
.db9u-sec{border-bottom:1px solid #1a1a1a}
.db9u-sec>h4{margin:0;padding:8px 12px;font-size:12px;font-weight:600;color:#ddd;cursor:pointer;display:flex;align-items:center;gap:8px;background:#2c2c2c;letter-spacing:.3px}
.db9u-sec>h4 .ar{font-size:9px;color:#888;width:10px}
.db9u-sec>h4 .rs{margin-left:auto;font-weight:400;font-size:10px;color:#777;padding:0 4px}.db9u-sec>h4 .rs:hover{color:#ddd}
.db9u-sec>.bd{padding:6px 12px 10px}
.db9u-sec.closed>.bd{display:none}
.db9u-grp{color:#8a8a8a;font-size:10px;text-transform:uppercase;letter-spacing:.8px;margin:8px 0 2px}
.db9u-row{display:grid;grid-template-columns:78px 1fr 40px;align-items:center;gap:8px;height:22px}
.db9u-row label{color:#b8b8b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:default}
.db9u-row input.v{width:40px;background:transparent;border:1px solid transparent;color:#ddd;text-align:right;font:inherit;font-size:11px;padding:1px 2px;border-radius:2px}
.db9u-row input.v:focus{border-color:#555;background:#1c1c1c;outline:none}
.db9u-rng{-webkit-appearance:none;appearance:none;width:100%;height:14px;background:transparent;margin:0;cursor:pointer}
.db9u-rng::-webkit-slider-runnable-track{height:3px;border-radius:2px;background:var(--trk,#555)}
.db9u-rng::-moz-range-track{height:3px;border-radius:2px;background:var(--trk,#555)}
.db9u-rng::-webkit-slider-thumb{-webkit-appearance:none;width:11px;height:11px;border-radius:50%;background:#cfcfcf;border:1px solid #1a1a1a;margin-top:-4px}
.db9u-rng::-moz-range-thumb{width:10px;height:10px;border-radius:50%;background:#cfcfcf;border:1px solid #1a1a1a}
.db9u-rng:hover::-webkit-slider-thumb{background:#fff}
.db9u-tabs{display:flex;gap:2px;margin:4px 0 8px;background:#1f1f1f;border-radius:3px;padding:2px}
.db9u-tabs button{flex:1;background:transparent;color:#999;border:0;border-radius:2px;padding:3px 0;cursor:pointer;font:inherit;font-size:11px}
.db9u-tabs button.on{background:#444;color:#fff}
.db9u-sel,.db9u-txt{box-sizing:border-box;width:100%;min-width:0;background:#1c1c1c;color:#ddd;border:1px solid #3a3a3a;border-radius:3px;padding:3px 5px;font:inherit}
.db9u-wide{display:grid;grid-template-columns:78px 1fr;align-items:center;gap:8px;margin:4px 0}
.db9u-cv{display:block;width:100%;aspect-ratio:1/1;background:#1c1c1c;border-radius:2px;cursor:crosshair;touch-action:none}
.db9u-chs{display:flex;gap:8px;align-items:center;margin:0 0 6px}
.db9u-ch{width:14px;height:14px;border-radius:50%;border:2px solid #555;cursor:pointer;flex:none}
.db9u-ch.on{border-color:#fff}
.db9u-cinfo{margin-left:auto;color:#888;font-size:11px}
.db9u-foot{display:flex;gap:6px;padding:8px 10px;border-top:1px solid #111;background:#222}
.db9u-foot .sp{flex:1}
.db9u-hint{color:#777;font-size:10.5px;margin:4px 0}`;
function smoothHist(h) { // làm mượt răng lược (ảnh 8-bit sau khi kéo tông)
  const o = new Float32Array(256);
  for (let i = 0; i < 256; i++) { let a = 0, w = 0; for (let k = -3; k <= 3; k++) { const j = i + k; if (j < 0 || j > 255) continue; const ww = 4 - Math.abs(k); a += h[j] * ww; w += ww; } o[i] = a / w; }
  return o;
}
function injectCss() { if (!document.getElementById("db9u-css")) document.head.append(el("style", { id: "db9u-css" }, CSS)); }

const GRAD = {
  temperature: "linear-gradient(90deg,#4a7fd6,#bdbdbd,#d9b43b)", tint: "linear-gradient(90deg,#4aa84a,#bdbdbd,#c24ac2)",
  exposure: "linear-gradient(90deg,#111,#eee)", whites: "linear-gradient(90deg,#666,#eee)", blacks: "linear-gradient(90deg,#000,#777)",
  saturation: "linear-gradient(90deg,#888,#e5484d,#f5d90a,#46a758,#0090ff)", vibrance: "linear-gradient(90deg,#888,#d6409f,#0090ff)",
  cyan_red: "linear-gradient(90deg,#19c3c3,#9a9a9a,#d23b3b)", magenta_green: "linear-gradient(90deg,#c33bc3,#9a9a9a,#3bb83b)",
  yellow_blue: "linear-gradient(90deg,#d0c23b,#9a9a9a,#3b5fd6)",
};
function hueGrad(c, mode) {
  const i = HUES.indexOf(c), col = HUE_CSS[c];
  if (mode === "sat") return `linear-gradient(90deg,#808080,${col})`;
  if (mode === "lum") return `linear-gradient(90deg,#111,${col},#f0f0f0)`;
  return `linear-gradient(90deg,${HUE_CSS[HUES[(i + 7) % 8]]},${col},${HUE_CSS[HUES[(i + 1) % 8]]})`;
}

// ============================================================ Editor
class Editor {
  constructor(node) {
    this.node = node; this.st = node.db9u; this.inputs = {};
    this.compare = this.st.compare; this.before = false; this.zoom = 1; this.fit = true; this.panX = 0; this.panY = 0;
    this.chan = "master"; this.hsTab = lsGet("db9u.hsTab", "sat"); this.cbTab = lsGet("db9u.cbTab", "shadows");
    injectCss(); this.build();
    this.keyH = (e) => this.onKey(e, true); this.keyU = (e) => this.onKey(e, false);
    window.addEventListener("keydown", this.keyH, true); window.addEventListener("keyup", this.keyU, true);
    this.ro = new ResizeObserver(() => { if (this.fit) this.fitView(); this.draw(); }); this.ro.observe(this.view);
    requestAnimationFrame(() => { this.fitView(); this.draw(); this.renderFull(); });
  }
  w(n) { return this.node.widgets?.find((x) => x.name === n); }
  params() { return this.node.db9uParams(); }
  set(name, v, final) {
    const w = this.w(name); if (!w) return; w.value = v;
    this.node.graph?.change?.();
    if (name === "lut" || name === "lut_path") this.node.db9uLoadCube().then(() => this.renderFull());
    else if (final) this.renderFull(); else this.renderDraft();
    if (name.startsWith("curve_")) this.drawCurve?.();
  }
  // ---------------------------------------------------------- render (không chạy workflow)
  async renderDraft() {
    const r = await this.st.renderer.render("draft", this.params()); if (!r || this.closed) return;
    this.st.draftCv = idToCanvas(r.img, this.st.draftCv); this.st.bView = this.st.draftCv;
    this.hist = r.hist; this.draw(); this.drawHist();
  }
  async renderFull() {
    this.status("Đang xử lý…");
    const r = await this.st.renderer.render("full", this.params()); if (!r || this.closed) return;
    this.st.fullCv = idToCanvas(r.img, this.st.fullCv); this.st.bView = this.st.fullCv;
    this.hist = r.hist; this.status(""); this.draw(); this.drawHist(); this.drawCurve?.();
    this.node.db9uRefreshThumb();
  }
  // ---------------------------------------------------------- khung xem
  ref() { const b = this.st.fullCv || this.st.aImg; return b ? [b.width || b.naturalWidth, b.height || b.naturalHeight] : null; }
  vsize() { return [this.view.clientWidth, this.view.clientHeight - 38]; }
  fitView() {
    const r = this.ref(); if (!r) return; const [W, H] = this.vsize();
    this.zoom = Math.min(W / r[0], H / r[1]) * 0.97; this.panX = (W - r[0] * this.zoom) / 2; this.panY = (H - r[1] * this.zoom) / 2; this.fit = true;
  }
  setZoom(z, cx, cy) {
    const nz = Math.max(0.05, Math.min(8, z));
    this.panX = cx - (cx - this.panX) * nz / this.zoom; this.panY = cy - (cy - this.panY) * nz / this.zoom; this.zoom = nz; this.fit = false; this.draw();
  }
  draw() {
    if (this.closed) return;
    const cv = this.canvas, dpr = window.devicePixelRatio || 1, [W, H] = this.vsize();
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + "px"; }
    const x = cv.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.fillStyle = "#141414"; x.fillRect(0, 0, W, H);
    const A = this.st.aImg, B = this.st.bView, r = this.ref(); if (!r) return;
    const dw = r[0] * this.zoom, dh = r[1] * this.zoom;
    x.imageSmoothingEnabled = this.zoom < 1.5; x.imageSmoothingQuality = "high";
    if (this.before && A) { x.drawImage(A, this.panX, this.panY, dw, dh); this.tag(x, "Trước", 10, 10); this.zoomInfo(); return; }
    if (B) x.drawImage(B, this.panX, this.panY, dw, dh); else if (A) x.drawImage(A, this.panX, this.panY, dw, dh);
    if (this.compare && A) {
      const sx = this.panX + dw * this.st.split;
      x.save(); x.beginPath(); x.rect(this.panX, this.panY, Math.max(0, sx - this.panX), dh); x.clip(); x.drawImage(A, this.panX, this.panY, dw, dh); x.restore();
      const lx = Math.max(0, Math.min(W, sx));
      x.fillStyle = "#fff"; x.fillRect(lx - 1, 0, 2, H);
      x.beginPath(); x.arc(lx, H / 2, 11, 0, Math.PI * 2); x.fillStyle = "rgba(20,20,20,.85)"; x.fill(); x.strokeStyle = "#fff"; x.lineWidth = 1.5; x.stroke();
      x.fillStyle = "#fff"; x.font = "12px sans-serif"; x.textAlign = "center"; x.fillText("⇆", lx, H / 2 + 4); x.textAlign = "left";
      this.tag(x, "Trước", Math.max(8, lx - 62), 10); this.tag(x, "Sau", Math.min(W - 44, lx + 10), 10);
    }
    this.zoomInfo();
  }
  tag(x, t, px, py) { x.font = "11px sans-serif"; const w = x.measureText(t).width + 14; x.fillStyle = "rgba(0,0,0,.6)"; x.fillRect(px, py, w, 20); x.fillStyle = "#eee"; x.fillText(t, px + 7, py + 14); }
  zoomInfo() { const r = this.ref(); this.zInfo.textContent = r ? `${Math.round(this.zoom * 100)}% · preview ${r[0]}×${r[1]} · ${this.st.info || ""}` : ""; }
  status(t) { this.stEl.textContent = t; }
  drawHist() {
    const c = this.histCv; if (!this.hist) return; const H = this.hist.map(smoothHist);
    const dpr = window.devicePixelRatio || 1, W = c.clientWidth, Hh = c.clientHeight;
    c.width = W * dpr; c.height = Hh * dpr; const x = c.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.fillStyle = "#1d1d1d"; x.fillRect(0, 0, W, Hh);
    let mx = 1; for (const ch of H) for (let i = 2; i < 254; i++) mx = Math.max(mx, ch[i]);
    x.globalCompositeOperation = "lighter";
    ["rgba(200,55,55,.55)", "rgba(55,180,55,.55)", "rgba(65,100,220,.6)"].forEach((col, k) => {
      x.beginPath(); x.moveTo(0, Hh);
      for (let i = 0; i < 256; i++) x.lineTo(i / 255 * W, Hh - Math.min(1, Math.sqrt(H[k][i] / mx)) * (Hh - 6));
      x.lineTo(W, Hh); x.closePath(); x.fillStyle = col; x.fill();
    });
    x.globalCompositeOperation = "source-over"; x.strokeStyle = "#2c2c2c";
    for (let i = 1; i < 4; i++) { x.beginPath(); x.moveTo(W * i / 4, 0); x.lineTo(W * i / 4, Hh); x.stroke(); }
  }
  // ---------------------------------------------------------- dựng giao diện
  build() {
    this.canvas = el("canvas");
    this.stEl = el("span", { cls: "st", style: "color:#e0b44a" }); this.zInfo = el("span", { cls: "st" });
    this.btnCmp = el("button", { cls: "db9u-b", title: "Bật/tắt so sánh (Y)", onclick: () => this.toggleCompare() }); this.updCmp();
    const tb = el("div", { cls: "db9u-tb" },
      el("button", { cls: "db9u-b", title: "Vừa khung (Z / double-click)", onclick: () => { this.fitView(); this.draw(); } }, "Vừa khung"),
      el("button", { cls: "db9u-b", title: "100% (Z / double-click)", onclick: () => { const [W, H] = this.vsize(); this.setZoom(1, W / 2, H / 2); } }, "1:1"),
      this.btnCmp, this.zInfo, el("span", { cls: "sp" }), this.stEl,
      el("span", { cls: "st" }, "Giữ \\ : xem ảnh trước · Y: so sánh"));
    this.view = el("div", { cls: "db9u-view" }, this.canvas, tb);
    this.bindView();
    this.histCv = el("canvas");
    const scroll = el("div", { cls: "db9u-scroll" }); for (const s of this.sections()) scroll.append(s);
    const side = el("div", { cls: "db9u-side" },
      el("div", { cls: "db9u-hist" }, this.histCv), scroll,
      el("div", { cls: "db9u-foot" },
        el("button", { cls: "db9u-b", onclick: () => this.resetAll() }, "Đặt lại"),
        el("span", { cls: "sp" }),
        el("button", { cls: "db9u-b", title: "Esc", onclick: () => this.close() }, "Đóng"),
        el("button", { cls: "db9u-b pri", title: "Chạy lệnh: áp chỉnh màu lên ảnh full-res và lưu file", onclick: () => this.node.db9uSave() }, "Lưu full-res")));
    this.ov = el("div", { cls: "db9u-ov" }, this.view, side);
    document.body.append(this.ov);
  }
  splitX() { const r = this.ref(); return r ? this.panX + r[0] * this.zoom * this.st.split : -999; }
  bindView() {
    const v = this.view; let drag = null;
    v.addEventListener("wheel", (e) => { e.preventDefault(); const b = v.getBoundingClientRect(); this.setZoom(this.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), e.clientX - b.left, e.clientY - b.top); }, { passive: false });
    this.canvas.addEventListener("pointerdown", (e) => {
      const b = v.getBoundingClientRect(), mx = e.clientX - b.left;
      drag = this.compare && e.button === 0 && Math.abs(mx - this.splitX()) < 14 ? { mode: "split" } : { mode: "pan", x: e.clientX, y: e.clientY, px: this.panX, py: this.panY };
      this.canvas.setPointerCapture(e.pointerId); this.canvas.style.cursor = drag.mode === "split" ? "col-resize" : "grabbing";
    });
    this.canvas.addEventListener("pointermove", (e) => {
      const b = v.getBoundingClientRect(), mx = e.clientX - b.left;
      if (!drag) { this.canvas.style.cursor = this.compare && Math.abs(mx - this.splitX()) < 14 ? "col-resize" : "grab"; return; }
      if (drag.mode === "split") { const r = this.ref(); this.st.split = Math.max(0, Math.min(1, (mx - this.panX) / (r[0] * this.zoom))); }
      else { this.panX = drag.px + e.clientX - drag.x; this.panY = drag.py + e.clientY - drag.y; this.fit = false; }
      this.draw();
    });
    const end = () => { drag = null; this.canvas.style.cursor = "grab"; this.node.setDirtyCanvas?.(true, false); };
    this.canvas.addEventListener("pointerup", end); this.canvas.addEventListener("pointercancel", end);
    this.canvas.addEventListener("dblclick", (e) => { const b = v.getBoundingClientRect(); if (this.fit) this.setZoom(1, e.clientX - b.left, e.clientY - b.top); else { this.fitView(); this.draw(); } });
  }
  toggleCompare() { this.compare = !this.compare; this.st.compare = this.compare; this.updCmp(); this.draw(); this.node.db9uUpdCmpBtn?.(); this.node.setDirtyCanvas?.(true, true); }
  updCmp() { this.btnCmp.textContent = this.compare ? "So sánh: Bật" : "So sánh: Tắt"; this.btnCmp.classList.toggle("on", this.compare); }
  onKey(e, down) {
    const t = e.target, typing = t && ((t.tagName === "INPUT" && t.type !== "range") || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
    e.stopPropagation(); // chặn phím tắt ComfyUI khi Editor đang mở
    if (typing) { if (down && e.key === "Escape") t.blur(); return; }
    if (e.key === "\\") { e.preventDefault(); if (this.before !== down) { this.before = down; this.draw(); } return; }
    if (!down) return;
    if (e.key === "Escape") this.close();
    else if (e.key === "y" || e.key === "Y") this.toggleCompare();
    else if (e.key === "z" || e.key === "Z") { const [W, H] = this.vsize(); if (this.fit) this.setZoom(1, W / 2, H / 2); else { this.fitView(); this.draw(); } }
  }
  // ---------------------------------------------------------- panel phải
  section(key, title, body, resetKeys) {
    const open = lsGet("db9u.sec." + key, !["export", "sel", "fx"].includes(key));
    const ar = el("span", { cls: "ar" }, open ? "▼" : "▶");
    const rs = resetKeys ? el("span", { cls: "rs", title: "Đặt lại nhóm này", onclick: (e) => { e.stopPropagation(); this.resetKeys(resetKeys); } }, "Đặt lại") : null;
    const sec = el("div", { cls: "db9u-sec" + (open ? "" : " closed") }, el("h4", { onclick: () => {
      sec.classList.toggle("closed"); const o = !sec.classList.contains("closed"); ar.textContent = o ? "▼" : "▶"; lsSet("db9u.sec." + key, o);
      if (o && key === "curve") this.drawCurve?.();
    } }, ar, title, rs), el("div", { cls: "bd" }, ...body.filter(Boolean)));
    return sec;
  }
  slider(name, label, min = -100, max = 100, step = 1, grad) {
    const w = this.w(name); if (!w) return null;
    const def = defaultOf(name, w);
    const rng = el("input", { type: "range", cls: "db9u-rng", min, max, step, style: grad ? `--trk:${grad}` : null });
    const num = el("input", { cls: "v", type: "text" });
    const fmt = (v) => (step < 1 ? (+v).toFixed(2) : String(Math.round(+v)));
    const show = (v) => { rng.value = v; num.value = (v > 0 && min < 0 ? "+" : "") + fmt(v); };
    show(w.value ?? def);
    rng.addEventListener("input", () => { show(+rng.value); this.set(name, +rng.value, false); });
    rng.addEventListener("change", () => this.set(name, +rng.value, true));
    num.addEventListener("change", () => { let v = parseFloat(num.value); if (!isFinite(v)) v = def; v = Math.max(min, Math.min(max, v)); show(v); this.set(name, v, true); });
    const reset = () => { show(def); this.set(name, def, true); };
    rng.addEventListener("dblclick", reset);
    this.inputs[name] = show;
    return el("div", { cls: "db9u-row" }, el("label", { title: "Double-click: về mặc định", ondblclick: reset }, label), rng, num);
  }
  combo(name, label) {
    const w = this.w(name); if (!w) return null;
    const sel = el("select", { cls: "db9u-sel" }); for (const o of w.options?.values || []) sel.append(el("option", { value: o }, o));
    sel.value = w.value; sel.addEventListener("change", () => this.set(name, sel.value, true));
    this.inputs[name] = (v) => { sel.value = v; };
    return el("div", { cls: "db9u-wide" }, el("label", {}, label), sel);
  }
  text(name, label, ph) {
    const w = this.w(name); if (!w) return null;
    const t = el("input", { cls: "db9u-txt", type: "text", placeholder: ph || "" }); t.value = w.value ?? "";
    t.addEventListener("change", () => this.set(name, t.value, true));
    this.inputs[name] = (v) => { t.value = v; };
    return el("div", { cls: "db9u-wide" }, el("label", {}, label), t);
  }
  tabs(items, cur, onPick) {
    const box = el("div", { cls: "db9u-tabs" });
    const bs = items.map(([k, t]) => el("button", { cls: k === cur ? "on" : "", onclick: () => { bs.forEach((b, i) => b.classList.toggle("on", items[i][0] === k)); onPick(k); } }, t));
    box.append(...bs); return box;
  }
  sections() {
    const S = (n, l, a, b, c, g) => this.slider(n, l, a, b, c, g), out = [];
    out.push(this.section("basic", "Cơ bản", [
      el("div", { cls: "db9u-grp" }, "Cân bằng trắng"),
      S("temperature", "Nhiệt độ", -100, 100, 1, GRAD.temperature), S("tint", "Tint", -100, 100, 1, GRAD.tint),
      el("div", { cls: "db9u-grp" }, "Tông"),
      S("exposure", "Exposure", -5, 5, 0.05, GRAD.exposure), S("contrast", "Contrast"), S("highlights", "Highlights"),
      S("shadows", "Shadows"), S("whites", "Whites", -100, 100, 1, GRAD.whites), S("blacks", "Blacks", -100, 100, 1, GRAD.blacks),
      el("div", { cls: "db9u-grp" }, "Presence"),
      S("texture", "Texture"), S("clarity", "Clarity"), S("dehaze", "Dehaze"),
      S("vibrance", "Vibrance", -100, 100, 1, GRAD.vibrance), S("saturation", "Saturation", -100, 100, 1, GRAD.saturation),
    ], ["temperature", "tint", "exposure", "contrast", "highlights", "shadows", "whites", "blacks", "texture", "clarity", "dehaze", "vibrance", "saturation"]));
    out.push(this.section("curve", "Tone Curve", [this.curveEditor()], ["curve_master", "curve_red", "curve_green", "curve_blue"]));
    const hslBox = el("div");
    const drawHsl = (mode) => {
      this.hsTab = mode; lsSet("db9u.hsTab", mode); hslBox.replaceChildren();
      for (const c of HUES) { const r = this.slider(`hsl_${c}_${mode}`, HUE_VN[c], -100, 100, 1, hueGrad(c, mode)); if (r) hslBox.append(r); }
    };
    drawHsl(this.hsTab);
    out.push(this.section("hsl", "HSL / Màu", [this.tabs([["hue", "Hue"], ["sat", "Saturation"], ["lum", "Luminance"]], this.hsTab, drawHsl), hslBox],
      HUES.flatMap((c) => [`hsl_${c}_hue`, `hsl_${c}_sat`, `hsl_${c}_lum`])));
    const cbBox = el("div");
    const drawCb = (t) => {
      this.cbTab = t; lsSet("db9u.cbTab", t);
      cbBox.replaceChildren(...[this.slider(`cb_${t}_cyan_red`, "Cyan–Red", -100, 100, 1, GRAD.cyan_red),
        this.slider(`cb_${t}_magenta_green`, "Mag–Green", -100, 100, 1, GRAD.magenta_green),
        this.slider(`cb_${t}_yellow_blue`, "Yellow–Blue", -100, 100, 1, GRAD.yellow_blue)].filter(Boolean));
    };
    drawCb(this.cbTab);
    out.push(this.section("cb", "Color Grading", [this.tabs([["shadows", "Shadows"], ["midtones", "Midtones"], ["highlights", "Highlights"]], this.cbTab, drawCb), cbBox],
      ["shadows", "midtones", "highlights"].flatMap((t) => ["cyan_red", "magenta_green", "yellow_blue"].map((a) => `cb_${t}_${a}`))));
    out.push(this.section("sel", "Selective Color", [this.combo("sel_target", "Nhóm màu"),
      S("sel_cyan", "Cyan", -100, 100, 1, GRAD.cyan_red), S("sel_magenta", "Magenta", -100, 100, 1, GRAD.magenta_green),
      S("sel_yellow", "Yellow", -100, 100, 1, GRAD.yellow_blue), S("sel_black", "Black")], ["sel_cyan", "sel_magenta", "sel_yellow", "sel_black"]));
    out.push(this.section("fx", "Hiệu ứng", [S("vignette", "Vignette"), S("grain", "Grain", 0, 100, 1)], ["vignette", "grain"]));
    out.push(this.section("lut", "Profile / LUT", [this.combo("lut", "LUT"), this.text("lut_path", "File .cube", "hoặc dán đường dẫn"),
      S("lut_strength", "Amount", 0, 1, 0.05)], ["lut", "lut_path", "lut_strength"]));
    out.push(this.section("export", "Xuất file", [this.combo("save_to", "Lưu vào"), this.text("local_folder", "Thư mục", "J:\\Render\\Upscale"),
      this.text("filename_prefix", "Tên file"), this.combo("format", "Định dạng"),
      el("div", { cls: "db9u-hint" }, "Chỉnh màu ở đây chỉ chạy trong trình duyệt. Bấm 'Lưu full-res' mới chạy lệnh: áp lên ảnh độ phân giải đầy đủ và lưu file.")]));
    return out;
  }
  // ---------------------------------------------------------- Tone Curve kiểu Lightroom
  curveEditor() {
    const cv = el("canvas", { cls: "db9u-cv" }), info = el("span", { cls: "db9u-cinfo" });
    const CH = [["master", "#e8e8e8", "curve_master", "#bbb"], ["red", "#e5484d", "curve_red"], ["green", "#46a758", "curve_green"], ["blue", "#3b82f6", "curve_blue"]];
    const dots = CH.map(([k, col, , dot]) => el("span", { cls: "db9u-ch" + (k === this.chan ? " on" : ""), title: k === "master" ? "RGB" : k, style: `background:${dot || col}`,
      onclick: () => { this.chan = k; dots.forEach((d, i) => d.classList.toggle("on", CH[i][0] === k)); draw(); } }));
    const key = () => CH.find((c) => c[0] === this.chan)[2];
    const getPts = () => { const p = parsePoints(this.w(key())?.value); return p ? p.map(([x, y]) => [x * 255, y * 255]) : [[0, 0], [255, 255]]; };
    const write = (pts, final) => {
      const ident = pts.length === 2 && pts[0][0] === 0 && pts[0][1] === 0 && pts[1][0] === 255 && pts[1][1] === 255;
      this.set(key(), ident ? "" : pts.map(([x, y]) => `${Math.round(x)},${Math.round(y)}`).join(" "), final);
    };
    const P = 8;
    const toPx = (x, y, S) => [P + x / 255 * (S - 2 * P), S - P - y / 255 * (S - 2 * P)];
    const fromPx = (px, py, S) => [Math.max(0, Math.min(255, (px - P) / (S - 2 * P) * 255)), Math.max(0, Math.min(255, (S - P - py) / (S - 2 * P) * 255))];
    const draw = (live) => {
      const S = cv.getBoundingClientRect().width; if (!S) return;
      const dpr = window.devicePixelRatio || 1; if (cv.width !== Math.round(S * dpr)) { cv.width = Math.round(S * dpr); cv.height = Math.round(S * dpr); }
      const x = cv.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.fillStyle = "#1c1c1c"; x.fillRect(0, 0, S, S);
      if (this.hist) {
        const k = { master: -1, red: 0, green: 1, blue: 2 }[this.chan], h = new Float32Array(256);
        const hs = this.hist.map(smoothHist);
        for (let i = 0; i < 256; i++) h[i] = k < 0 ? (hs[0][i] + hs[1][i] + hs[2][i]) / 3 : hs[k][i];
        let m = 1; for (let i = 2; i < 254; i++) m = Math.max(m, h[i]);
        x.beginPath(); x.moveTo(P, S - P);
        for (let i = 0; i < 256; i++) x.lineTo(P + i / 255 * (S - 2 * P), S - P - Math.min(1, Math.sqrt(h[i] / m)) * (S - 2 * P) * 0.9);
        x.lineTo(S - P, S - P); x.fillStyle = "rgba(255,255,255,.07)"; x.fill();
      }
      x.strokeStyle = "#333"; x.lineWidth = 1;
      for (let i = 0; i <= 4; i++) { const a = P + i * (S - 2 * P) / 4; x.beginPath(); x.moveTo(a, P); x.lineTo(a, S - P); x.moveTo(P, a); x.lineTo(S - P, a); x.stroke(); }
      x.strokeStyle = "#3a3a3a"; x.beginPath(); x.moveTo(P, S - P); x.lineTo(S - P, P); x.stroke();
      const pts = live || getPts(), L = curveLut(pts.map(([a, b]) => [a / 255, b / 255]), 256);
      x.strokeStyle = CH.find((c) => c[0] === this.chan)[1]; x.lineWidth = 1.6; x.beginPath();
      for (let i = 0; i < 256; i++) { const [px, py] = toPx(i, L[i] * 255, S); i ? x.lineTo(px, py) : x.moveTo(px, py); } x.stroke();
      for (const [a, b] of pts) { const [px, py] = toPx(a, b, S); x.beginPath(); x.arc(px, py, 4, 0, Math.PI * 2); x.fillStyle = "#1c1c1c"; x.fill(); x.strokeStyle = "#eee"; x.lineWidth = 1.5; x.stroke(); }
    };
    this.drawCurve = () => draw();
    let drag = -1, pts = null, out = false;
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top, r.width]; };
    const near = (mx, my, S, list) => { let best = -1, bd = 10; list.forEach(([a, b], i) => { const [px, py] = toPx(a, b, S); const d = Math.hypot(px - mx, py - my); if (d < bd) { bd = d; best = i; } }); return best; };
    cv.addEventListener("pointerdown", (e) => {
      const [mx, my, S] = pos(e); pts = getPts(); let best = near(mx, my, S, pts);
      if (e.button === 2) { if (best > 0 && best < pts.length - 1) { pts.splice(best, 1); write(pts, true); } return; }
      if (best < 0) {
        const [a, b] = fromPx(mx, my, S); const idx = pts.findIndex((p) => p[0] > a); if (idx <= 0) return;
        const L = curveLut(pts.map(([u, v]) => [u / 255, v / 255]), 256), onCurve = L[Math.round(a)] * 255;
        pts.splice(idx, 0, [a, Math.abs(onCurve - b) < 18 ? onCurve : b]); best = idx;
      }
      drag = best; out = false; cv.setPointerCapture(e.pointerId); write(pts, false);
    });
    cv.addEventListener("dblclick", (e) => {
      const [mx, my, S] = pos(e); const p = getPts(), i = near(mx, my, S, p);
      if (i > 0 && i < p.length - 1) { p.splice(i, 1); write(p, true); }
    });
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    cv.addEventListener("pointermove", (e) => {
      const [mx, my, S] = pos(e);
      if (drag < 0) { const [a, b] = fromPx(mx, my, S); info.textContent = `${Math.round(a)} / ${Math.round(b)}`; return; }
      let [a, b] = fromPx(mx, my, S); const last = pts.length - 1;
      if (drag === 0) a = 0; else if (drag === last) a = 255; else a = Math.max(pts[drag - 1][0] + 1, Math.min(pts[drag + 1][0] - 1, a));
      pts[drag] = [a, b]; info.textContent = `Vào ${Math.round(a)} · Ra ${Math.round(b)}`;
      out = drag > 0 && drag < last && (my < -24 || my > S + 24);
      write(pts, false);
    });
    const up = () => { if (drag < 0) return; if (out) pts.splice(drag, 1); drag = -1; write(pts, true); };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    const presets = el("select", { cls: "db9u-sel", style: "width:auto;margin-left:6px" },
      ...[["__", "Preset…"], ["", "Tuyến tính"], ["0,0 64,56 192,200 255,255", "Tương phản vừa"], ["0,0 64,46 192,210 255,255", "Tương phản mạnh"], ["0,26 64,70 192,196 255,238", "Matte"]]
        .map(([v, t]) => el("option", { value: v }, t)));
    presets.addEventListener("change", () => {
      if (presets.value !== "__") { const p = parsePoints(presets.value); write(p ? p.map(([a, b]) => [a * 255, b * 255]) : [[0, 0], [255, 255]], true); }
      presets.value = "__";
    });
    setTimeout(() => draw(), 0);
    return el("div", {}, el("div", { cls: "db9u-chs" }, ...dots, presets, info), cv,
      el("div", { cls: "db9u-hint" }, "Bấm để thêm điểm · kéo để chỉnh · double-click / chuột phải / kéo ra ngoài để xoá"));
  }
  resetKeys(keys) {
    for (const k of keys) { const w = this.w(k); if (!w) continue; w.value = defaultOf(k, w); this.inputs[k]?.(w.value); }
    this.node.graph?.change?.(); this.drawCurve?.();
    if (keys.includes("lut") || keys.includes("lut_path")) this.node.db9uLoadCube().then(() => this.renderFull()); else this.renderFull();
  }
  resetAll() { this.resetKeys(this.node.widgets.filter((w) => w.db9uHidden && !NON_GRADE.has(w.name)).map((w) => w.name)); }
  close() {
    if (this.closed) return; this.closed = true;
    window.removeEventListener("keydown", this.keyH, true); window.removeEventListener("keyup", this.keyU, true);
    this.ro.disconnect(); this.ov.remove(); this.node.db9uEditor = null; this.node.setDirtyCanvas?.(true, true);
  }
}

// ============================================================ Node trên canvas
app.registerExtension({
  name: "DB9U.Finish",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "DB9U_Finish") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      this.db9u = { split: 0.5, compare: true, renderer: new Renderer() };
      for (const w of this.widgets || []) hideWidget(w);
      this.addWidget("button", "🎨 Mở Editor (Lightroom)", null, () => {
        if (!node.db9u.renderer.src.full) { alert("Chạy workflow 1 lần để có ảnh, rồi mở Editor."); return; }
        if (!node.db9uEditor) node.db9uEditor = new Editor(node);
      });
      this.db9uCmpBtn = this.addWidget("button", "⇆ So sánh: Bật", null, () => { node.db9u.compare = !node.db9u.compare; node.db9uUpdCmpBtn(); });
      this.addWidget("button", "💾 Lưu full-res", null, () => node.db9uSave());
      this.db9uUpdCmpBtn = () => {
        const t = node.db9u.compare ? "⇆ So sánh: Bật" : "⇆ So sánh: Tắt"; node.db9uCmpBtn.name = t; node.db9uCmpBtn.label = t; node.setDirtyCanvas(true, true);
      };
      this.db9uParams = () => { const p = {}; for (const w of node.widgets || []) if (w.db9uHidden && !NON_GRADE.has(w.name)) p[w.name] = w.value; return p; };
      this.db9uLoadCube = async () => {
        const name = node.widgets.find((w) => w.name === "lut")?.value, path = String(node.widgets.find((w) => w.name === "lut_path")?.value || "").trim().replace(/^"|"$/g, "");
        if (!path && (!name || name === "none")) { node.db9u.renderer.setCube(null); return; }
        try {
          const q = path ? `path=${encodeURIComponent(path)}` : `name=${encodeURIComponent(name)}`;
          const res = await api.fetchApi(`/db9u/lut?${q}`); if (!res.ok) throw new Error(await res.text());
          node.db9u.renderer.setCube(parseCube(await res.text()));
        } catch (e) { console.warn("[DB9U] không đọc được LUT", e); node.db9u.renderer.setCube(null); }
      };
      this.db9uRefreshThumb = async () => {
        const rr = await node.db9u.renderer.render("thumb", node.db9uParams()); if (!rr) return;
        node.db9u.thumb = idToCanvas(rr.img, node.db9u.thumb); node.setDirtyCanvas(true, true);
      };
      this.db9uSave = () => {
        const sf = node.widgets.find((w) => w.name === "save_file"); if (!sf) return;
        sf.value = true; node.db9u.saving = true; node.db9u.busy = true;
        node.db9uEditor?.status("Đang lưu full-res…"); node.setDirtyCanvas(true, true);
        try { app.queuePrompt(0, 1); } catch (e) { console.warn("[DB9U] không queue được", e); }
      };
      this.setSize([Math.max(this.size[0], 440), 420]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = async function (msg) {
      onExecuted?.apply(this, arguments);
      const s = this.db9u; s.info = msg?.db9u_info?.[0] || ""; s.busy = false;
      if (s.saving) { s.saving = false; const sf = this.widgets.find((w) => w.name === "save_file"); if (sf) sf.value = false; this.db9uEditor?.status("Đã lưu ✓"); }
      const a = msg?.db9u_a?.[0], raw = (msg?.db9u_raw || msg?.db9u_b)?.[0];
      try {
        if (a) {
          s.aImg = await loadImage(viewUrl(a));
          // ảnh TRƯỚC trên node phải cùng độ phân giải với ảnh SAU (thumb) -> so sánh công bằng, không 'sau mờ hơn trước'
          const k = Math.min(1, THUMB_MAX / s.aImg.naturalWidth), c = document.createElement("canvas");
          c.width = Math.max(1, Math.round(s.aImg.naturalWidth * k)); c.height = Math.max(1, Math.round(s.aImg.naturalHeight * k));
          const cx = c.getContext("2d"); cx.imageSmoothingQuality = "high"; cx.drawImage(s.aImg, 0, 0, c.width, c.height); s.aThumb = c;
        }
        if (raw) {
          const key = `${raw.filename}|${raw.subfolder}`;
          if (key !== s.rawKey) {
            s.rawKey = key; s.renderer.setSource(imageToPlanes(await loadImage(viewUrl(raw))));
            s.fullCv = null; s.draftCv = null; s.bView = null; await this.db9uLoadCube();
          }
          await this.db9uRefreshThumb();
          if (this.db9uEditor) { this.db9uEditor.fitView(); this.db9uEditor.renderFull(); }
        }
      } catch (e) { console.warn("[DB9U] lỗi nạp ảnh preview", e); }
      this.setDirtyCanvas(true, true);
    };

    const onDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      onDraw?.apply(this, arguments);
      const s = this.db9u; if (!s || this.flags?.collapsed) return;
      let y0 = 0; for (const w of this.widgets || []) if (w.last_y !== undefined && !w.db9uHidden) y0 = Math.max(y0, w.last_y + 24);
      y0 += 6;
      const W = this.size[0] - 16, H = this.size[1] - y0 - 24, B = s.thumb;
      if (!B) { ctx.fillStyle = "#888"; ctx.font = "12px sans-serif"; ctx.fillText("Run 1 lần để hiện so sánh trước / sau", 10, y0 + 20); return; }
      if (W < 40 || H < 40) return;
      const k = Math.min(W / B.width, H / B.height), dw = B.width * k, dh = B.height * k, dx = 8 + (W - dw) / 2, dy = y0 + (H - dh) / 2;
      s.rect = [dx, dy, dw, dh];
      ctx.imageSmoothingQuality = "high"; ctx.drawImage(B, dx, dy, dw, dh);
      if (s.compare && s.aImg) {
        ctx.save(); ctx.beginPath(); ctx.rect(dx, dy, dw * s.split, dh); ctx.clip(); ctx.drawImage(s.aThumb || s.aImg, dx, dy, dw, dh); ctx.restore();
        const x = dx + dw * s.split; ctx.fillStyle = "#fff"; ctx.fillRect(x - 0.75, dy, 1.5, dh);
        ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(dx + 4, dy + 4, 46, 17); ctx.fillRect(dx + dw - 36, dy + 4, 32, 17);
        ctx.fillStyle = "#fff"; ctx.font = "11px sans-serif"; ctx.fillText("Trước", dx + 9, dy + 16); ctx.fillText("Sau", dx + dw - 31, dy + 16);
      }
      ctx.fillStyle = "#999"; ctx.font = "11px sans-serif"; ctx.fillText((s.busy ? "Đang chạy… " : "") + (s.info || ""), 8, this.size[1] - 8);
    };

    const onMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (e, pos) {
      const r = onMove?.apply(this, arguments); const s = this.db9u;
      if (s?.compare && s.rect && pos) {
        const [dx, dy, dw, dh] = s.rect;
        if (pos[0] >= dx && pos[0] <= dx + dw && pos[1] >= dy && pos[1] <= dy + dh) { s.split = Math.min(1, Math.max(0, (pos[0] - dx) / dw)); this.setDirtyCanvas(true, false); }
      }
      return r;
    };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { this.db9uEditor?.close(); return onRemoved?.apply(this, arguments); };
  },
});
