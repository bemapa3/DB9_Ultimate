// DB9U Tile Picker v0.11.1 (menu preview_pick) · v0.8 — bấm chọn ô trên BẢNG Ô (preview_tile bật, preview_tiles trống) rồi điền preview_tiles.
// Nút "🔲 Chọn ô trên bảng" có trên DB9U Upscale và DB9U Advanced Settings.
// v0.12.3: có cả trên DB9U Forge (điền thẳng vào widget của chính node Forge).
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

let LAST = null; // bảng ô mới nhất (khi không tìm được node nối)
const viewUrl = (im) => api.apiURL(`/view?filename=${encodeURIComponent(im.filename)}&type=${im.type}&subfolder=${encodeURIComponent(im.subfolder || "")}&t=${Date.now()}`);
const W = (node, name) => node?.widgets?.find((w) => w.name === name);

function inNode(node, slotName) {
  const i = node?.inputs?.findIndex((s) => s.name === slotName);
  if (i == null || i < 0) return null;
  try { return node.getInputNode(i); } catch { return null; }
}
function outNodes(node) {
  try { return node.getOutputNodes(0) || []; } catch { return []; }
}
function settingsOf(up) { const n = inNode(up, "settings"); return n?.type === "DB9U_Settings" ? n : null; }
function upscaleOf(st) { return outNodes(st).find((n) => n.type === "DB9U_Upscale") || null; }

// v0.12.5: nút "🗑 Reset preview & chạy lại" — xoá output/db9u_cache (ô preview + ô resume), xoá bảng ô đang nhớ,
// server tăng token IS_CHANGED -> Run chạy lại node thật, không dùng kết quả/ô cũ.
async function resetPreview(node) {
  if (!confirm("Xoá TOÀN BỘ preview + ô đã lưu (output/db9u_cache) rồi chạy lại?\n" +
               "• Đừng bấm khi workflow đang chạy.\n• Ô resume của ảnh khác (nếu có) cũng bị xoá.")) return;
  try {
    const res = await api.fetchApi("/db9u/reset_preview", { method: "POST" });
    const j = await res.json().catch(() => ({}));
    if (!res.ok || !j.ok) throw new Error(j.error || res.statusText || "server lỗi");
    for (const n of [node, upscaleOf(node)]) {
      if (!n) continue;
      n.db9uBoard = null;
      n.imgs = null;
      n.setDirtyCanvas?.(true, true);
    }
    LAST = null;
    console.log(`[DB9U] Reset preview: xoá ${j.removed} bộ cache -> chạy lại`);
    app.queuePrompt(0, 1);
  } catch (e) {
    alert("Reset preview lỗi: " + e.message + "\n(Đã khởi động lại ComfyUI sau khi cập nhật DB9_Ultimate chưa?)");
  }
}
const addReset = (node) => node.addWidget("button", "🗑 Reset preview & chạy lại", null, () => resetPreview(node));

function parseSpec(spec, n) {
  const out = new Set();
  for (const m of String(spec || "").toLowerCase().matchAll(/o?\s*(\d+)(?:\s*-\s*o?\s*(\d+))?/g)) {
    const a = +m[1], b = +(m[2] || m[1]);
    for (let v = Math.min(a, b); v <= Math.max(a, b); v++) if (v >= 1 && v <= n) out.add(v - 1);
  }
  return out;
}
function toSpec(sel) {
  const v = [...sel].sort((a, b) => a - b).map((i) => i + 1);
  const parts = []; let s = 0;
  for (let i = 1; i <= v.length; i++) {
    if (i < v.length && v[i] === v[i - 1] + 1) continue;
    const a = v[s], b = v[i - 1]; parts.push(b - a >= 2 ? `O${a}-O${b}` : a === b ? `O${a}` : `O${a},O${b}`); s = i;
  }
  return parts.join(",");
}

function openPicker(board, settings) {
  if (!board) { alert("Chưa có bảng ô.\nBật preview_tile, để trống preview_tiles, chạy 1 lần để vẽ bảng ô."); return; }
  if (!settings) { alert("Cần nối node 'DB9U Advanced Settings' vào input settings của DB9U Upscale."); return; }
  const n = board.coords.length, sel = parseSpec(W(settings, "preview_tiles")?.value, n);
  const wrap = document.createElement("div");
  wrap.style.cssText = "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.78);display:flex;flex-direction:column;align-items:center;justify-content:center;font:13px sans-serif;color:#eee";
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;justify-content:center";
  const info = document.createElement("span"); info.style.minWidth = "220px";
  const dn = document.createElement("input");
  dn.placeholder = "so denoise: 0.3,0.4,0.5"; dn.value = W(settings, "preview_denoise")?.value || "";
  dn.style.cssText = "width:170px;background:#222;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 6px";
  const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f;
    b.style.cssText = "background:#333;color:#eee;border:1px solid #666;border-radius:4px;padding:5px 10px;cursor:pointer"; bar.append(b); return b; };
  const cv = document.createElement("canvas");
  cv.style.cssText = "max-width:94vw;max-height:80vh;cursor:pointer;border:1px solid #444";
  const hint = document.createElement("div"); hint.style.cssText = "margin-top:6px;color:#aaa";
  hint.textContent = "Bấm ô để chọn/bỏ · Esc đóng · Enter áp dụng";
  const img = new Image(); img.crossOrigin = "anonymous";

  const k = () => cv.width / board.W;
  const rectOf = (i) => { const [x, y] = board.coords[i]; const f = k();
    return [x * f, y * f, Math.min(board.W - x, board.tile_w) * f, Math.min(board.H - y, board.tile_h) * f]; };
  function draw() {
    const c = cv.getContext("2d"); c.drawImage(img, 0, 0, cv.width, cv.height);
    for (const i of sel) {
      const [x, y, w, h] = rectOf(i);
      c.fillStyle = "rgba(0,170,255,.28)"; c.fillRect(x, y, w, h);
      c.strokeStyle = "#00b4ff"; c.lineWidth = Math.max(3, cv.width / 400); c.strokeRect(x + 2, y + 2, w - 4, h - 4);
    }
    info.textContent = sel.size ? `Đã chọn ${sel.size}/${n}: ${toSpec(sel)}` : `Chưa chọn ô nào (${n} ô)`;
  }
  cv.onclick = (e) => {
    const r = cv.getBoundingClientRect();
    const px = (e.clientX - r.left) * cv.width / r.width / k(), py = (e.clientY - r.top) * cv.height / r.height / k();
    // vùng giao nhau: lấy ô có tâm gần nhất
    let best = -1, bd = Infinity;
    board.coords.forEach(([x, y], i) => {
      const w = Math.min(board.W - x, board.tile_w), h = Math.min(board.H - y, board.tile_h);
      if (px < x || py < y || px > x + w || py > y + h) return;
      const d = (px - x - w / 2) ** 2 + (py - y - h / 2) ** 2; if (d < bd) { bd = d; best = i; }
    });
    if (best < 0) return;
    sel.has(best) ? sel.delete(best) : sel.add(best); draw();
  };
  const close = () => { window.removeEventListener("keydown", onKey, true); wrap.remove(); };
  const apply = (run) => {
    if (!sel.size) { alert("Chọn ít nhất 1 ô."); return; }
    const set = (name, v) => { const w = W(settings, name); if (w) { w.value = v; w.callback?.(v); } };
    set("preview_tiles", toSpec(sel)); set("preview_denoise", dn.value.trim()); set("preview_tile", true); set("preview_pick", "ô đã chọn");
    settings.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true);
    close();
    if (run) { try { app.queuePrompt(0, 1); } catch (e) { console.warn("[DB9U] không queue được", e); } }
  };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } else if (e.key === "Enter") { e.stopPropagation(); apply(false); } };
  btn("3 ô khó nhất", () => { sel.clear(); board.score.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).forEach(([, i]) => sel.add(i)); draw(); });
  btn("Bỏ chọn", () => { sel.clear(); draw(); });
  bar.append(info, dn);
  btn("✓ Áp dụng", () => apply(false));
  btn("▶ Áp dụng & chạy", () => apply(true));
  btn("✕", close);
  wrap.append(bar, cv, hint);
  wrap.addEventListener("mousedown", (e) => { if (e.target === wrap) close(); });
  window.addEventListener("keydown", onKey, true);
  img.onload = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; draw(); };
  img.onerror = () => { alert("Không tải được ảnh bảng ô — chạy lại bảng ô."); close(); };
  img.src = viewUrl(board.image);
  document.body.append(wrap);
}

app.registerExtension({
  name: "DB9U.TilePicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "DB9U_Upscale") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        this.addWidget("button", "🔲 Chọn ô trên bảng", null, () => openPicker(this.db9uBoard || LAST, settingsOf(this)));
        addReset(this);
        return r;
      };
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        onExecuted?.apply(this, arguments);
        const b = msg?.db9u_board?.[0];
        if (b) { this.db9uBoard = b; LAST = b; }
      };
    } else if (nodeData.name === "DB9U_Forge") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        this.addWidget("button", "🔲 Chọn ô trên bảng", null, () => openPicker(this.db9uBoard, this));
        addReset(this);
        return r;
      };
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        onExecuted?.apply(this, arguments);
        const b = msg?.db9u_board?.[0];
        if (b) this.db9uBoard = b;
      };
    } else if (nodeData.name === "DB9U_Settings") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        this.addWidget("button", "🔲 Chọn ô trên bảng", null, () => openPicker(upscaleOf(this)?.db9uBoard || LAST, this));
        addReset(this);
        // v0.11.1: chọn ô preview bằng menu -> tự bật preview_tile; preview_tiles chỉ là ô đã bấm trên bảng
        const pick = W(this, "preview_pick"), pt = W(this, "preview_tile");
        if (pick) {
          const cb = pick.callback;
          pick.callback = (v) => { cb?.call(pick, v); if (v !== "ô đã chọn" && pt && !pt.value) { pt.value = true; pt.callback?.(true); }
            this.setDirtyCanvas?.(true, true); };
        }
        return r;
      };
    }
  },
});
