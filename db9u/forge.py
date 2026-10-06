"""DB9U Forge — 1 node gộp phần mạnh nhất của 2 bộ node.

TẠO HÌNH theo db9_flux_locked_upscale (nodes.py v3 + workflow 260531-Upscale_F2K9B):
  - img2img thuần trên từng ô pixel (VAE encode -> KSampler -> VAE decode), không reference, không ControlNet.
  - SUPERSAMPLE: AI vẽ ở độ phân giải x2 (mặc định) so với ảnh ra rồi thu về. Bộ flux làm điều này gián tiếp
    (TileSplit 2048 -> node chính scale 2 -> TileMerge thu 4096 về 2048) — đây là lý do chi tiết mịn + sắc hơn.
  - Nguồn x2 của ô = chạy lại upscale model trên ô (như bộ flux: UltraSharp 2 lần) hoặc lanczos.
  - Khoá màu CHỈ chỉnh thống kê (mean/std + contrast) của tần số thấp từng ô, HF giữ 100% -> hình khối AI vẽ
    được giữ nguyên (khác DB9U Upscale: thay hẳn tần số thấp bằng ảnh gốc -> mất form AI tạo ra).
  - Unsharp ở độ phân giải làm việc (trước khi thu về), khớp màu toàn ảnh theo ảnh gốc cuối cùng.

GHÉP HÌNH theo DB9_Ultimate:
  - Ô đều nhau, bội số 32, canvas pad reflect (core.plan_tiles) — không còn ô cuối bị đè gần hết như _make_positions.
  - Trọng số dốc tuyến tính đúng bằng vùng giao thực tế (2 ô kề cộng = 1) thay cho cosine feather cố định.
  - Căn trôi sub-pixel từng ô, đo ΔE + chạy lại ô lệch (tuỳ chọn), flow_align cả ảnh (cần OpenCV).
  - Chạy theo lô theo VRAM, hết VRAM tự hạ lô rồi hạ ô, resume cache khi crash, canvas trên CPU.

Tối ưu chạy so với bộ flux:
  - positive == negative mà cfg > 1 -> ép cfg 1 (kết quả y hệt, nhanh gấp đôi; workflow flux cũ đang cfg 8 với pos = neg).
  - 1 tầng chia ô thay cho 2 tầng (TileSplit + chia ô bên trong) -> không chạy trùng vùng giao 2 lần.
  - Blur/khoá màu pad reflect (bộ flux dùng pad 0 -> viền ô tối, phải dựa vào feather để giấu).
"""
import hashlib
import os
import shutil
import time

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

from . import core, engine, profiles, sampling

SUPERSAMPLE = ["2.0", "1.5", "1.0"]
SS_SOURCES = ["upscale_model", "lanczos"]
DOWNSCALES = ["bicubic_sharp", "area", "lanczos"]
SEED_MODES = ["per_tile", "fixed"]
REF_MODES = ["strip", "tile", "keep"]


# ----------------------------------------------------------------------------
# Toán tạo hình kiểu flux (BCHW, pad reflect)
# ----------------------------------------------------------------------------
def box_blur(t, radius):
    """avg_pool kích thước (2r+1) như bộ flux, nhưng pad reflect (bộ flux pad 0 -> tối viền ô)."""
    k = max(3, (int(radius) * 2) | 1)
    p = k // 2
    mode = "reflect" if (t.shape[-1] > p and t.shape[-2] > p) else "replicate"
    return F.avg_pool2d(F.pad(t, (p, p, p, p), mode=mode), k, stride=1)


def match_stats(src, ref, strength):
    """Reinhard mean/std từng kênh (BCHW, thống kê trên cả ô)."""
    if strength <= 0:
        return src
    s_mu, s_sg = src.mean((2, 3), keepdim=True), src.std((2, 3), keepdim=True).clamp_min(1e-5)
    r_mu, r_sg = ref.mean((2, 3), keepdim=True), ref.std((2, 3), keepdim=True).clamp_min(1e-5)
    return torch.lerp(src, (src - s_mu) / s_sg * r_sg + r_mu, float(strength))


def contrast_lock(src, ref, strength):
    """Kéo tương phản độ sáng của src về ref (như _contrast_lock bộ flux)."""
    if strength <= 0:
        return src
    sl, rl = src.mean(1, keepdim=True), ref.mean(1, keepdim=True)
    sc = (sl - sl.mean((2, 3), keepdim=True)).abs().mean((2, 3), keepdim=True).clamp_min(1e-5)
    rc = (rl - rl.mean((2, 3), keepdim=True)).abs().mean((2, 3), keepdim=True).clamp_min(1e-5)
    mu = src.mean((2, 3), keepdim=True)
    return torch.lerp(src, (src - mu) * (rc / sc).clamp(0.5, 2.0) + mu, float(strength))


def flux_color_lock(ai, ref, color, contrast, radius=7):
    """Detail-preserving color lock (bộ flux v3): chỉ chỉnh THỐNG KÊ tần số thấp, HF (cạnh, vân) giữ 100%.
    Không thay tần số thấp bằng ref -> hình khối AI vẽ được giữ."""
    if color <= 0 and contrast <= 0:
        return ai
    lf = box_blur(ai, radius)
    hf = ai - lf
    rlf = box_blur(ref.to(ai), radius)
    lf = match_stats(lf, rlf, color)
    lf = contrast_lock(lf, rlf, contrast)
    return (lf + hf).clamp(0, 1)


def unsharp(t, strength, radius=3):
    if strength <= 0:
        return t
    return (t + (t - box_blur(t, radius)) * float(strength)).clamp(0, 1)


def global_color(img, ref, strength, rows=1024):
    """Khớp mean/std RGB cả ảnh theo ref (bước color_lock_strength của DB9TileMerge). [1,H,W,3], chạy theo dải."""
    if strength <= 0:
        return img
    a, r = img[..., :3].double(), ref[..., :3].double()
    s_mu, s_sg = a.mean((0, 1, 2)), a.std((0, 1, 2)).clamp_min(1e-5)
    r_mu, r_sg = r.mean((0, 1, 2)), r.std((0, 1, 2)).clamp_min(1e-5)
    del a, r
    gain, bias = (r_sg / s_sg).float(), (r_mu - s_mu * r_sg / s_sg).float()
    out = torch.empty_like(img[..., :3])
    for y in range(0, img.shape[1], rows):
        x = img[:, y:y + rows, :, :3].float()
        out[:, y:y + rows] = (x + (x * gain + bias - x) * float(strength)).clamp(0, 1)
    return out


def downscale(x, w, h, mode):
    """x [B,H,W,3] -> [B,h,w,3]. bicubic_sharp = như bộ flux (bicubic không antialias, sắc nhất)."""
    if x.shape[1:3] == (h, w):
        return x
    if mode == "area":
        return core.resize(x, w, h, "area")
    if mode == "lanczos":
        return engine.lanczos(x, w, h)
    t = F.interpolate(core.to_bchw(x.float()), size=(h, w), mode="bicubic", align_corners=False)
    return core.to_bhwc(t).clamp(0, 1)


def same_cond(a, b):
    """positive và negative giống hệt nhau? (cùng link, hoặc cùng tensor + cùng tham số)."""
    if a is b:
        return True
    try:
        if len(a) != len(b):
            return False
        for (ta, da), (tb, db) in zip(a, b):
            if ta.shape != tb.shape or not torch.equal(ta, tb) or set(da) != set(db):
                return False
            for k in da:
                va, vb = da[k], db[k]
                if torch.is_tensor(va) or torch.is_tensor(vb):
                    if not (torch.is_tensor(va) and torch.is_tensor(vb) and va.shape == vb.shape and torch.equal(va, vb)):
                        return False
                elif va != vb:
                    return False
        return True
    except Exception:
        return False


def _cond_sig(c):
    try:
        return [(tuple(x[0].shape), round(float(x[0].float().sum()), 3), round(float(x[0].float().abs().mean()), 6))
                for x in c]
    except Exception:
        return repr(type(c))


# ----------------------------------------------------------------------------
# Denoise thật (mức nhiễu lúc bắt đầu = số denoise)
# ----------------------------------------------------------------------------
def full_schedule(model, scheduler, w, h, n=200):
    """Dãy sigma đủ (denoise 1) thật mịn của scheduler -> lấy dáng đường cong."""
    try:
        if scheduler == "flux2":
            from comfy_extras.nodes_flux import get_schedule
            s = get_schedule(n, round(w * h / 256))
        else:
            import comfy.samplers
            s = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, n)
        s = torch.as_tensor(s, dtype=torch.float32).flatten().cpu()
        if s.numel() >= 2 and float(s[0]) > float(s[-1]):
            return s
    except Exception:
        pass
    return torch.linspace(1.0, 0.0, n + 1)


def true_sigmas(model, scheduler, steps, w, h, d):
    """steps+1 sigma: bắt đầu ĐÚNG d x sigma_max (model flow như Flux: d = % nhiễu thật), giữ dáng đường cong
    của scheduler từ đó về 0. Khác denoise thường: flux2 đẩy mức nhiễu lên theo cỡ ô (denoise 0.5 ở ô 1472px
    = nhiễu 86%)."""
    full = full_schedule(model, scheduler, w, h)
    target = float(full[0]) * min(1.0, max(0.01, float(d)))
    seg = torch.cat([torch.tensor([target]), full[full < target - 1e-6]])
    pos = torch.linspace(0, seg.numel() - 1, steps + 1)
    lo = pos.floor().long().clamp(max=seg.numel() - 1)
    hi = (lo + 1).clamp(max=seg.numel() - 1)
    t = pos - lo.float()
    out = seg[lo] * (1 - t) + seg[hi] * t
    out[0], out[-1] = target, float(full[-1])
    return out


def guarded_flow(ai, ref, guard_px=4.0, max_px=16.0, smooth=3.0):
    """Căn AI về vị trí nền (optical flow DIS) NHƯNG chỉ sửa trôi nhỏ: chỗ lệch > guard_px (AI vẽ lại hình)
    giảm dần về 0 ở 2 x guard_px -> giữ nguyên hình AI vẽ thay vì kéo giãn theo hình gốc (nguồn méo khi
    p95 > ~5px). guard_px 0 = sửa hết như core.flow_align. Trả (ảnh, p95_px, tỉ lệ ảnh bị chặn) | (ai, None, 0)."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return ai, None, 0.0
    a = ai[0, ..., :3].float().clamp(0, 1).cpu().numpy()
    r = ref[0, ..., :3].float().clamp(0, 1).cpu().numpy()
    H, W = a.shape[:2]
    k = 1 if max(H, W) <= 6400 else 2

    def gray(x):
        g = ((x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114) * 255).astype(np.uint8)
        return cv2.resize(g, (W // k, H // k), interpolation=cv2.INTER_AREA) if k > 1 else g
    fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray(r), gray(a), None)
    if k > 1:
        fl = cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR) * k
    fl = cv2.GaussianBlur(fl, (0, 0), smooth)
    mag = np.sqrt((fl ** 2).sum(-1))
    st = max(1, H // 512)
    sub = mag[::st, ::st]
    p95 = float(np.percentile(sub, 95))
    frac = 0.0
    if guard_px > 0:
        frac = float((sub > guard_px).mean())
        w = np.clip((2 * guard_px - mag) / guard_px, 0, 1).astype(np.float32)
        w = cv2.GaussianBlur(w, (0, 0), smooth * 2)
        fl = fl * w[..., None]
    fl = np.clip(fl, -max_px, max_px)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = cv2.remap(a, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    out = torch.from_numpy(np.clip(out, 0, 1)).unsqueeze(0).to(ai.dtype)
    return out, p95, frac


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
PREVIEW_KEYS = ("preview_tile", "preview_tiles", "preview_denoise", "preview_layout")  # không vào khoá cache


class Forge:
    def __init__(self, model, vae, positive, negative, p):
        self.model, self.vae, self.pos, self.neg, self.p = model, vae, positive, negative, dict(p)
        self.log = []
        self.board = None  # bảng ô (preview) -> JS bấm chọn ô
        self.profile, pnote = profiles.detect_profile(model, p.get("profile", "auto"))
        pr = self.profile
        hw_tile, hw_batch, hw_note = profiles.hardware_plan(pr)
        self.ss = float(p["supersample"])
        self.work_tile = int(p["work_tile"] or hw_tile)
        self.user_overlap = int(p["overlap"])
        self._set_tiles()
        self.batch = int(p["batch_tiles"] or hw_batch)
        self.ref_mode = p["reference_mode"]
        if self.ref_mode == "tile":
            self.batch = 1  # reference latent theo từng ô -> 1 ô/lượt
        self.steps = int(p["steps"] or pr["steps"])
        self.sampler = pr["sampler"] if p["sampler"] == "auto" else p["sampler"]
        self.scheduler = pr["scheduler"] if p["scheduler"] == "auto" else p["scheduler"]
        cfg = float(p["cfg"]) if p["cfg"] > 0 else float(pr["cfg"])
        self.cfg_note = ""
        if cfg != 1.0 and same_cond(positive, negative):
            self.cfg_note = (f" (positive = negative -> cfg {cfg} cho kết quả y hệt cfg 1 nhưng tốn gấp đôi -> ép cfg 1)")
            cfg = 1.0
        self.cfg = cfg
        self.denoise = float(p["denoise"])
        self.seed = int(p["seed"])
        self.true_denoise = p.get("denoise_mode", "true") == "true"
        self._sig = {}
        self.say(f"[DB9U Forge] Profile: {pnote} | {hw_note}")

    def sigmas_for(self, ww, wh, d):
        """true_denoise -> dãy sigma bắt đầu đúng d (cache theo cỡ ô + d). Ngược lại None (scheduler tự tính)."""
        if not self.true_denoise:
            return None
        k = (ww, wh, round(d, 4))
        if k not in self._sig:
            self._sig[k] = true_sigmas(self.model, self.scheduler, self.steps, ww, wh, d)
        return self._sig[k]

    def noise_note(self, ww, wh):
        """Mức nhiễu THẬT lúc bắt đầu (để biết denoise đang mạnh cỡ nào)."""
        try:
            if self.true_denoise:
                s = self.sigmas_for(ww, wh, self.denoise)
                return f"denoise thật: nhiễu bắt đầu {float(s[0]):.2f} (= số denoise)"
            if self.scheduler == "flux2":
                s = sampling.flux2_sigmas(self.steps, ww, wh, self.denoise)
                return (f"denoise theo scheduler: số {self.denoise} nhưng nhiễu bắt đầu {float(s[0]):.2f} "
                        f"(flux2 đẩy lên theo cỡ ô)")
        except Exception:
            pass
        return f"denoise theo scheduler ({self.scheduler})"

    def _set_tiles(self):
        """Ô ở ảnh ra = work_tile / supersample (bội số 32). Overlap auto ~1/6 ô, ≥ 96px."""
        self.out_tile = max(core.ALIGN * 4, int(self.work_tile / self.ss) // core.ALIGN * core.ALIGN)
        ov = self.user_overlap or max(96, int(self.out_tile / 6))
        self.overlap = int(ov) // core.ALIGN * core.ALIGN

    def say(self, msg):
        self.log.append(msg)
        print(msg)

    def work_size(self, tw, th):
        """Kích thước ô AI vẽ (bội số 16 cho VAE /8 + patch 2x2)."""
        return (max(16, int(round(tw * self.ss / 16)) * 16), max(16, int(round(th * self.ss / 16)) * 16))

    # ---------------------------------------------------------------- cache (resume)
    def set_cache_key(self, img, target, upscale_model):
        h = hashlib.sha1((img.clamp(0, 1) * 65535).round().to(torch.int32).numpy().tobytes())
        try:
            msig = (profiles.model_signature(self.model),
                    hash(tuple(sorted((k, len(v)) for k, v in getattr(self.model, "patches", {}).items()))))
        except Exception:
            msig = repr(type(self.model))
        um = None if upscale_model is None else (type(upscale_model).__name__, getattr(upscale_model, "scale", None))
        keys = {k: v for k, v in self.p.items() if k not in ("resume",) + PREVIEW_KEYS}
        h.update(repr((target, sorted(keys.items()), self.cfg, self.steps, self.sampler, self.scheduler,
                       _cond_sig(self.pos), _cond_sig(self.neg), msig, um, tuple(img.shape))).encode())
        self.cache_key = "forge_" + h.hexdigest()[:16]

    def _cache_dir(self, plan):
        if not self.p.get("resume", True) or not getattr(self, "cache_key", None):
            return None
        try:
            import folder_paths
            d = os.path.join(folder_paths.get_output_directory(), "db9u_cache", self.cache_key,
                             f"{plan['tile_w']}x{plan['tile_h']}_{plan['cols']}x{plan['rows']}")
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            return None

    def clear_cache(self):
        if getattr(self, "cache_key", None):
            try:
                import folder_paths
                shutil.rmtree(os.path.join(folder_paths.get_output_directory(), "db9u_cache", self.cache_key),
                              ignore_errors=True)
            except Exception:
                pass

    # ---------------------------------------------------------------- tạo hình 1 lô ô (kiểu flux)
    def work_source(self, src, ww, wh, upscale_model):
        """Ô ở ảnh ra -> ô x supersample cho AI. upscale_model: chạy lại model trên ô (như bộ flux) rồi area về."""
        if src.shape[1:3] == (wh, ww):
            return src.float()
        if upscale_model is not None and self.p["ss_source"] == "upscale_model":
            sr = engine.sr_upscale(src, upscale_model)
            if sr.shape[2] >= ww:
                return core.resize(sr, ww, wh, "area")
            return engine.lanczos(sr, ww, wh)
        return engine.lanczos(src, ww, wh)

    def generate(self, src, seed, d, upscale_model):
        """src [B,th,tw,3] (ô ở ảnh ra) -> [B,th,tw,3] đã tạo hình."""
        p = self.p
        th, tw = src.shape[1:3]
        ww, wh = self.work_size(tw, th)
        work = self.work_source(src, ww, wh, upscale_model)
        lat = sampling.vae_encode(self.vae, work)
        pos = sampling.prep_cond(self.pos, self.ref_mode, lat if self.ref_mode == "tile" else None)
        neg = sampling.prep_cond(self.neg, self.ref_mode, lat if self.ref_mode == "tile" else None)
        out = sampling.ksample(self.model, lat, seed, self.steps, self.cfg, self.sampler, self.scheduler,
                               pos, neg, d, pixel_size=(ww, wh), sigmas=self.sigmas_for(ww, wh, d))
        del lat
        dec = sampling.vae_decode(self.vae, out)
        del out
        if dec.shape[1:3] != (wh, ww):
            dec = core.resize(dec, ww, wh, "bicubic")
        a = flux_color_lock(core.to_bchw(dec), core.to_bchw(work), p["color_lock"], p["contrast_lock"], 7)
        a = unsharp(a, p["sharpen"], 3)
        return downscale(core.to_bhwc(a), tw, th, p["downscale"])

    # ---------------------------------------------------------------- QA (kiểu Ultimate)
    def qa(self, src, out, i, n, d, attempt):
        note = ""
        if self.p["align_tiles"]:
            out, (dx, dy), moved = core.align_to_ref(out, src)
            note = f" trôi({dx:+.2f},{dy:+.2f}){'→kéo về' if moved else ''}"
        err = core.tile_color_error(src, out)
        ok = err <= self.p["tile_max_deltaE"]
        self.last_err = float(err)
        if attempt == 0:
            self.__dict__.setdefault("_errs", []).append(float(err))
        self.say(f"[DB9U Forge]   ô {i + 1:02d}/{n} lần {attempt} denoise {d:.2f} ΔE {err:.2f}{note} "
                 f"{'OK' if ok else 'LỆCH'}")
        return out, ok

    # ---------------------------------------------------------------- 1 lượt chia ô
    def run_tiles(self, base, upscale_model):
        p = self.p
        plan = core.plan_tiles(base.shape[2], base.shape[1], self.out_tile, self.overlap)
        tiles = core.split_tiles(base, plan)
        n = tiles.shape[0]
        ww, wh = self.work_size(plan["tile_w"], plan["tile_h"])
        self.say(f"[DB9U Forge] {plan['cols']}x{plan['rows']} = {n} ô {plan['tile_w']}x{plan['tile_h']} "
                 f"(giao ≥{plan['overlap']}) | AI vẽ mỗi ô ở {ww}x{wh} (x{self.ss:g}) | {self.batch} ô/lượt")
        self.say(f"[DB9U Forge] {self.noise_note(ww, wh)}")
        res = [None] * n
        self._errs = []
        cache = self._cache_dir(plan)
        if cache:
            for i in range(n):
                f = os.path.join(cache, f"t{i:03d}.pt")
                if os.path.isfile(f):
                    try:
                        r = torch.load(f).float()
                        res[i] = r if r.shape == tiles[i:i + 1].shape else None
                    except Exception:
                        res[i] = None
            reused = sum(r is not None for r in res)
            if reused:
                self.say(f"[DB9U Forge] resume: dùng lại {reused}/{n} ô đã xong")
        todo = [i for i in range(n) if res[i] is None]
        pbar = comfy.utils.ProgressBar(n)
        pbar.update(n - len(todo))
        t0, retries = time.time(), int(p["tile_retries"])
        for gi in range(0, len(todo), self.batch):
            mm.throw_exception_if_processing_interrupted()
            idx = todo[gi:gi + self.batch]
            seed = (self.seed + (idx[0] if p["seed_mode"] == "per_tile" else 0)) & 0xFFFFFFFFFFFFFFFF
            outs = self.generate(tiles[idx], seed, self.denoise, upscale_model)
            for k, i in enumerate(idx):
                src = tiles[i:i + 1]
                o, ok = self.qa(src, outs[k:k + 1], i, n, self.denoise, 0)
                for attempt in range(1, retries + 1):
                    if ok:
                        break
                    mm.throw_exception_if_processing_interrupted()
                    d = round(max(0.1, self.denoise - 0.05 * attempt), 3)
                    o2 = self.generate(src, (self.seed + 1000 * attempt + i) & 0xFFFFFFFFFFFFFFFF, d, upscale_model)
                    o, ok = self.qa(src, o2, i, n, d, attempt)
                res[i] = o
                if cache:
                    torch.save(o.half(), os.path.join(cache, f"t{i:03d}.pt"))
            pbar.update(len(idx))
            el, ran = time.time() - t0, gi + len(idx)
            self.say(f"[DB9U Forge] xong {n - len(todo) + ran}/{n} ô | {el / 60:.1f} phút | "
                     f"còn ~{el / ran * (len(todo) - ran) / 60:.1f} phút")
        self.enhance_note()
        return core.merge_tiles(torch.cat(res, 0), plan)

    def enhance_note(self, errs=None, d=None, label=""):
        """Tổng kết mức AI thay đổi so với nền (ΔE trung bình lần 0) -> gợi ý chỉnh denoise."""
        errs = self._errs if errs is None else errs
        d = self.denoise if d is None else d
        if not errs:
            return
        m = sum(errs) / len(errs)
        if m < 1.5:
            tip = (f"AI gần như CHÉP LẠI nền (chỉ upscale, không enhance) -> tăng denoise lên "
                   f"{min(0.95, d + 0.1):.2f}" + (", reference_mode strip" if self.ref_mode == "tile" else ""))
        elif m <= 6:
            tip = "AI đang vẽ thêm chi tiết (vùng enhance tốt)"
        else:
            tip = f"AI vẽ lại mạnh, dễ méo -> hạ denoise còn {max(0.3, d - 0.1):.2f}"
        self.say(f"[DB9U Forge] mức enhance{label}: ΔE trung bình {m:.2f} -> {tip}")

    # ---------------------------------------------------------------- 1 ảnh
    def header(self, img, target):
        p = self.p
        W, H = img.shape[2], img.shape[1]
        s, tw, th = engine.target_size(W, H, target, p["custom_long_edge"])
        self.say(f"[DB9U Forge] {W}x{H} -> {tw}x{th} (x{s:.3f}) | {self.sampler}/{self.scheduler} {self.steps} steps "
                 f"cfg {self.cfg}{self.cfg_note} denoise {self.denoise} | ref {self.ref_mode} | seed {p['seed_mode']}")
        return tw, th

    def make_base(self, img, tw, th, upscale_model):
        """Nền: upscale model của ảnh gốc -> đúng kích thước đích (như SDVN Upscale Image của workflow flux)."""
        if upscale_model is not None:
            sr = engine.sr_upscale(img, upscale_model).half()
            base = engine.Engine.sr_at(sr, tw, th)
            self.say(f"[DB9U Forge] nền: upscale model {sr.shape[2]}x{sr.shape[1]} -> {tw}x{th}")
            del sr
            return base
        self.say("[DB9U Forge] nền: lanczos (nên nối 4x-UltraSharp như workflow flux)")
        if self.p["ss_source"] == "upscale_model" and self.ss > 1:
            self.say("[DB9U Forge]   ss_source = upscale_model nhưng chưa nối model -> dùng lanczos")
        return engine.lanczos(img, tw, th)

    def post(self, out, base, img, tw, th):
        """Hậu kỳ cả ảnh: flow có chặn, màu tổng, chroma, viền. Trả (ảnh, ảnh gốc phóng cùng cỡ)."""
        p = self.p
        t1 = time.time()
        if p["flow_align"]:
            g = float(p.get("flow_guard", 4.0))
            out, p95, frac = guarded_flow(out, base, g)
            if p95 is None:
                self.say("[DB9U Forge] căn flow: THIẾU OpenCV (cv2) -> bỏ qua")
            else:
                self.say(f"[DB9U Forge] căn flow: AI lệch p95 {p95:.2f}px"
                         + (f" | {frac * 100:.1f}% ảnh lệch > {g:g}px (AI vẽ lại hình) -> giữ nguyên, không kéo giãn; "
                            f"phần trôi nhỏ đã kéo về" if g > 0 else " -> kéo hết về vị trí gốc")
                         + (" | p95 > 5px: AI đang vẽ lại nhiều, hạ denoise hoặc reference_mode tile nếu méo"
                            if p95 > 5 else ""))
        ref = core.resize(img, tw, th, "bicubic")
        if p["global_color"] > 0:
            out = global_color(out, ref, p["global_color"])
        if p["chroma_lock"] > 0:
            out = core.chroma_lock(out, engine.lanczos(img, tw, th), p["chroma_lock"])
        if p["edge_guard"] > 0:
            out = core.edge_guard(out, base, p["edge_guard"], 1.5, 3)
        self.say(f"[DB9U Forge] hậu kỳ cả ảnh (flow/màu/viền): {time.time() - t1:.0f}s")
        return out.clamp(0, 1), ref

    def run_image(self, img, target, upscale_model):
        tw, th = self.header(img, target)
        self.set_cache_key(img, target, upscale_model)
        base = self.make_base(img, tw, th, upscale_model)
        while True:
            try:
                merged = self.run_tiles(base, upscale_model)
                break
            except mm.OOM_EXCEPTION:
                mm.soft_empty_cache()
                if self.batch > 1:
                    self.batch = max(1, self.batch // 2)
                    self.say(f"[DB9U Forge] Hết VRAM -> giảm còn {self.batch} ô/lượt, chạy lại")
                elif self.work_tile > 512:
                    self.work_tile = max(512, self.work_tile - 256)
                    self._set_tiles()
                    self.say(f"[DB9U Forge] Hết VRAM -> ô AI còn {self.work_tile}px, chạy lại")
                else:
                    raise
        out, ref = self.post(merged, base, img, tw, th)
        self.clear_cache()
        return out, ref

    # ---------------------------------------------------------------- xem trước (như DB9U Upscale)
    def run_preview(self, img, target, upscale_model):
        """preview_tiles trống -> chỉ vẽ BẢNG Ô (vài giây, không chạy model).
        'O1,O5' / 'O3-O6' / 'auto' / 'auto3' -> chạy riêng các ô đó y hệt chạy full.
        preview_denoise '0.7,0.8,0.85' -> mỗi ô chạy ở từng mức (hàng = ô, cột = denoise) + ΔE từng mức.
        Ô ở denoise chính được lưu (resume bật) -> tắt preview_tile chạy full dùng lại."""
        p = self.p
        tw, th = self.header(img, target)
        plan = core.plan_tiles(tw, th, self.out_tile, self.overlap)
        n = len(plan["coords"])
        lz = engine.lanczos(img, tw, th)
        texm = core.texture_mask(lz).movedim(1, -1)
        score = core.split_tiles(texm, plan).mean((1, 2, 3))
        spec = str(p.get("preview_tiles", "") or "").strip()
        if not spec:
            self.say(f"[DB9U Forge] BẢNG Ô: {tw}x{th} -> {n} ô ({plan['cols']} cột x {plan['rows']} hàng, "
                     f"mỗi ô {plan['tile_w']}x{plan['tile_h']}, giao {plan['overlap']}px)")
            order = sorted(range(n), key=lambda k: -float(score[k]))
            self.say("[DB9U Forge] Ô nhiều chi tiết nhất: " + ", ".join(f"O{k + 1}" for k in order[:5])
                     + "  -> bấm '🔲 Chọn ô trên bảng' (hoặc gõ preview_tiles, vd O3,O7 / auto3) rồi chạy lại")
            m = engine.tile_map(lz, plan, score, texm)
            self.board = {"W": tw, "H": th, "tile_w": plan["tile_w"], "tile_h": plan["tile_h"],
                          "cols": plan["cols"], "rows": plan["rows"], "coords": [list(c) for c in plan["coords"]],
                          "score": [round(float(v), 4) for v in score]}
            return m, core.resize(lz, m.shape[2], m.shape[1], "area")
        idx = engine.parse_tiles(spec, n, score)
        ds = engine.parse_denoise_list(p.get("preview_denoise", ""), self.denoise)
        if any(d < 0.2 for d in ds):
            self.say(f"[DB9U Forge] CẢNH BÁO: denoise preview {', '.join(f'{d:.2f}' for d in ds)} có mức < 0.2 "
                     f"(preview_denoise = '{p.get('preview_denoise', '')}') -> gõ dạng 0.4,0.45,0.5 (dấu chấm, phẩy ngăn)")
        self.say(f"[DB9U Forge] PREVIEW {len(idx)} ô: " + ", ".join(f"O{k + 1}" for k in idx) + f" / {n} ô"
                 + (f" | so denoise: {', '.join(f'{d:.2f}' for d in ds)}" if len(ds) > 1 else ""))
        ww, wh = self.work_size(plan["tile_w"], plan["tile_h"])
        self.say(f"[DB9U Forge] AI vẽ mỗi ô ở {ww}x{wh} (x{self.ss:g}) | {self.noise_note(ww, wh)}")
        cache = None
        if any(abs(d - self.denoise) < 1e-6 for d in ds):
            self.set_cache_key(img, target, upscale_model)
            cache = self._cache_dir(plan)
        base = self.make_base(img, tw, th, upscale_model)
        tiles = core.split_tiles(base, plan)
        outs, refs, names, errs = [], [], [], {d: [] for d in ds}
        saved, t0 = 0, time.time()
        for k in idx:
            src = tiles[k:k + 1]
            seed = (self.seed + (k if p["seed_mode"] == "per_tile" else 0)) & 0xFFFFFFFFFFFFFFFF
            for d in ds:
                mm.throw_exception_if_processing_interrupted()
                o, _ = self.qa(src, self.generate(src, seed, d, upscale_model), k, n, d, 0)
                errs[d].append(self.last_err)
                if cache and abs(d - self.denoise) < 1e-6:
                    torch.save(o.half(), os.path.join(cache, f"t{k:03d}.pt"))
                    saved += 1
                outs.append(o)
                refs.append(src)
                names.append(f"O{k + 1}" + (f" · d{d:.2f}" if len(ds) > 1 else ""))
        self.say(f"[DB9U Forge] preview xong {len(outs)} ô | {(time.time() - t0) / 60:.1f} phút")
        for d in ds:
            self.enhance_note(errs[d], d, f" (denoise {d:.2f})" if len(ds) > 1 else "")
        if saved:
            self.say(f"[DB9U Forge] Đã lưu {saved} ô preview -> tắt preview_tile & chạy full sẽ dùng lại "
                     "(giữ nguyên ảnh/prompt/thông số)")
        if len(ds) == 1 and p.get("preview_layout", "frame") == "frame":
            parts = list(tiles.split(1, 0))
            for k, o in zip(idx, outs):
                parts[k] = o
            out, ref = self.post(core.merge_tiles(torch.cat(parts, 0), plan), base, img, tw, th)
            self.say(f"[DB9U Forge] Khung đầy đủ {tw}x{th}: ô {', '.join(names)} đã enhance, phần còn lại = nền upscale")
            return out, ref
        cols = len(ds) if len(ds) > 1 else None  # hàng = ô, cột = mức denoise
        return engine.mosaic(outs, names, cols=cols), engine.mosaic(refs, names, cols=cols)


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------
class DB9U_Forge:
    """Tạo hình kiểu bộ flux (supersample + khoá màu giữ form AI) + ghép hình kiểu Ultimate (ô đều, dốc tuyến tính,
    căn trôi, QA, flow_align, lô/VRAM, resume)."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        samplers = ["auto"] + list(comfy.samplers.KSampler.SAMPLERS)
        schedulers = ["auto", "flux2"] + list(comfy.samplers.KSampler.SCHEDULERS)
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "target": (list(engine.TARGETS.keys()), {"default": "6K"}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 1.0, "step": 0.01,
                                      "tooltip": "denoise_mode true: = % nhiễu thật lúc bắt đầu. Test thật (Klein, ref strip): "
                                                 "0.40-0.50 = enhance sạch (ΔE ~2), 0.55+ = bắt đầu vẽ lại chất liệu (mây thành lưới). "
                                                 "ref tile neo chặt: ΔE ~0.7 = chỉ upscale. Xem log 'mức enhance'"}),
                "supersample": (SUPERSAMPLE, {"default": "2.0",
                                "tooltip": "AI vẽ ở độ phân giải gấp mấy lần ảnh ra rồi thu về (bí quyết chi tiết của bộ flux). "
                                           "2.0 = như bộ flux · 1.0 = vẽ thẳng ở kích thước ra (như DB9U Upscale)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = theo profile model"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                  "tooltip": "0 = theo profile. positive = negative thì tự ép 1 (y hệt, nhanh x2)"}),
                "sampler": (samplers, {"default": "auto"}),
                "scheduler": (schedulers, {"default": "auto", "tooltip": "workflow flux: simple · profile Klein: flux2"}),
                "color_lock": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Khoá màu từng ô (chỉ mean/std tần số thấp, giữ form + chi tiết AI) — như bộ flux"}),
                "contrast_lock": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sharpen": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.5, "step": 0.05,
                                      "tooltip": "Unsharp ở độ phân giải AI vẽ, trước khi thu về. 0.15 = sạch; 0.4+ (bộ flux) dễ gắt/sạn "
                                                 "vì downscale bicubic_sharp đã làm nét thêm"}),
                "global_color": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Khớp mean/std màu cả ảnh theo ảnh gốc (như DB9TileMerge color_lock_strength)"}),
                "ss_source": (SS_SOURCES, {"default": "lanczos",
                              "tooltip": "Nguồn ô x supersample: lanczos (sạch, nhanh) · upscale_model = chạy lại model trên ô "
                                         "(như bộ flux, nhưng chồng lên nền đã upscale model -> dễ sạn)"}),
                "downscale": (DOWNSCALES, {"default": "bicubic_sharp",
                              "tooltip": "Thu ô AI về: bicubic_sharp = như bộ flux (sắc nhất) · lanczos · area (mềm, sạch răng cưa)"}),
                "seed_mode": (SEED_MODES, {"default": "per_tile", "tooltip": "per_tile = seed + số ô (như bộ flux)"}),
                "reference_mode": (REF_MODES, {"default": "strip",
                                   "tooltip": "strip = img2img thuần như bộ flux · tile = gắn latent ô gốc (Klein edit, bám hình hơn, 1 ô/lượt)"}),
                "work_tile": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64,
                                      "tooltip": "Cỡ ô AI vẽ (px). 0 = theo profile + VRAM. Bộ flux: 2048"}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 32,
                                    "tooltip": "Vùng giao ở ảnh ra. 0 = auto ~1/6 ô"}),
                "batch_tiles": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "0 = auto theo VRAM"}),
                "align_tiles": ("BOOLEAN", {"default": True, "tooltip": "Căn trôi sub-pixel từng ô trước khi ghép (Ultimate)"}),
                "flow_align": ("BOOLEAN", {"default": True,
                               "tooltip": "Căn AI về vị trí nền từng pixel sau khi ghép (Ultimate, cần OpenCV) -> hết lệch hình/bóng"}),
                "tile_retries": ("INT", {"default": 0, "min": 0, "max": 5,
                                         "tooltip": "Ô lệch màu (ΔE > tile_max_deltaE) chạy lại với denoise thấp hơn. 0 = như bộ flux"}),
                "tile_max_deltaE": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 50.0, "step": 0.1}),
                "chroma_lock": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Màu chi tiết lấy từ ảnh gốc, AI chỉ góp độ sáng (chặn lá hồng/cành tím). 0 = như bộ flux"}),
                "edge_guard": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "Khoá viền kiến trúc thẳng theo nền (Ultimate). 0 = tắt (giữ hình AI)"}),
                "profile": (profiles.PROFILE_CHOICES, {"default": "auto"}),
                "custom_long_edge": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16, "tooltip": ">0: bỏ qua target"}),
                "resume": ("BOOLEAN", {"default": True, "tooltip": "Lưu ô đã xong (output/db9u_cache), crash chạy lại dùng tiếp"}),
                "denoise_mode": (["true", "scheduler"], {"default": "true",
                                 "tooltip": "true = số denoise ĐÚNG bằng mức nhiễu lúc bắt đầu (0.5 = 50% nhiễu), dễ đoán · "
                                            "scheduler = như KSampler thường: flux2 đẩy nhiễu lên theo cỡ ô "
                                            "(denoise 0.3 ≈ nhiễu 73%, 0.5 ≈ 86% ở ô 1472px). Log in mức nhiễu thật"}),
                "flow_guard": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 16.0, "step": 0.5,
                               "tooltip": "Căn flow chỉ kéo về chỗ AI trôi nhỏ hơn số px này; chỗ lệch lớn hơn (AI vẽ lại hình) "
                                          "giữ nguyên hình AI thay vì kéo giãn (nguồn méo). 0 = kéo hết như cũ"}),
                "preview_tile": ("BOOLEAN", {"default": False,
                                 "tooltip": "Bật: chỉ xem trước. preview_tiles trống = vẽ bảng ô O1..On (vài giây) -> "
                                            "bấm '🔲 Chọn ô trên bảng'"}),
                "preview_tiles": ("STRING", {"default": "",
                                  "tooltip": "Trống = bảng ô · O1,O5,O6 · O3-O6 · auto (ô nhiều chi tiết nhất) · auto3"}),
                "preview_denoise": ("STRING", {"default": "",
                                    "tooltip": "So nhiều denoise trên ô preview, vd 0.7,0.8,0.85 (hàng = ô, cột = denoise, "
                                               "log ΔE từng mức). Trống = chỉ denoise chính"}),
                "preview_layout": (["frame", "grid"], {"default": "frame",
                                   "tooltip": "frame: ô preview đặt đúng chỗ trên cả khung (so với original_resized) · "
                                              "grid: ghép bảng các ô (so nhiều denoise luôn dùng grid)"}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "4x-UltraSharp như workflow flux: làm nền + nguồn ô x2"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "original_resized", "log")
    FUNCTION = "run"
    CATEGORY = "DB9 Ultimate"

    def run(self, image, model, vae, positive, negative, upscale_model=None, **p):
        target = p.pop("target")
        fg = Forge(model, vae, positive, negative, p)
        preview = bool(p.get("preview_tile", False))
        outs, refs = [], []
        for bi in range(image.shape[0]):
            im = image[bi:bi + 1, ..., :3].float().cpu()
            o, r = fg.run_preview(im, target, upscale_model) if preview else fg.run_image(im, target, upscale_model)
            if outs and o.shape != outs[0].shape:
                o = core.resize(o, outs[0].shape[2], outs[0].shape[1], "bicubic")
                r = core.resize(r, outs[0].shape[2], outs[0].shape[1], "bicubic")
            outs.append(o)
            refs.append(r)
        result = (torch.cat(outs, 0), torch.cat(refs, 0), "\n".join(fg.log))
        ui = {}
        if fg.board is not None:  # bảng ô -> JS cho bấm chọn ô
            try:
                import nodes as comfy_nodes
                im = comfy_nodes.PreviewImage().save_images(outs[0][:1], "db9u.board")["ui"]["images"]
                ui["db9u_board"] = [dict(fg.board, image=im[0])]
            except Exception as e:
                print(f"[DB9U Forge] không gửi được bảng ô lên giao diện: {e}")
        return {"ui": ui, "result": result}
