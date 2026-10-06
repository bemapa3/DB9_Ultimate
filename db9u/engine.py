"""Pipeline chính DB9_Ultimate: pre-upscale -> nhiều lượt (≤2x) -> chia ô -> sample theo lô -> QA/retry -> ghép -> khoá."""
import hashlib
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import comfy.model_management as mm
import comfy.utils

from . import core, profiles, sampling

TARGETS = {"x2": ("scale", 2.0), "x3": ("scale", 3.0), "x4": ("scale", 4.0),
           "6K": ("long", 6144), "8K": ("long", 8192), "10K": ("long", 10240), "11K": ("long", 11264)}
PRESETS = ["balanced", "fast", "max"]

DEFAULTS = {
    "profile": "auto",          # auto = tự nhận từ model
    "custom_long_edge": 0,      # >0: bỏ qua target, cạnh dài output = giá trị này
    "max_tile": 0,              # 0 = auto (profile + VRAM)
    "overlap": 0,               # 0 = auto (~1/6 ô, tối thiểu 128)
    "max_scale_per_pass": 2.0,
    "batch_tiles": 0,           # 0 = auto theo VRAM
    "steps": 0,                 # 0 = profile
    "sampler": "auto",
    "scheduler": "auto",
    "texture_denoise": -1.0,    # -1 = auto (denoise + 0.2, ≤ 0.85) · 0 = tắt
    "sky_denoise": -1.0,        # -1 = auto (0.12) · ≥ denoise = tắt nhận trời
    "min_denoise": 0.25,
    "resume": True,             # lưu ô đã xong ra đĩa, chạy lại cùng ảnh/thông số thì dùng tiếp
    "preview_tile": False,      # bật: chỉ xem bảng ô / chạy thử vài ô (preview_tiles)
    "preview_tiles": "",        # trống = chỉ vẽ bảng ô · "O1,O5" · "O3-O6" · "auto" · "auto3"
    "preview_denoise": "",      # "0.3,0.4,0.5": chạy mỗi ô preview ở nhiều denoise để so · trống = chỉ denoise chính
    "preview_layout": "frame",  # frame = ô đặt đúng chỗ trên cả khung (so với gốc) · grid = ghép bảng các ô
    "reuse_preview": True,      # chạy full (1 lượt) dùng lại ô đã preview ở denoise chính (cần resume)
    "despeckle": 0.0,
    "enhance": 1.0,
    "structure_lock": 0.5,      # 0..1: chặn AI vẽ lại hình (lá/cỏ/vân đá đổi dạng) · 0 = nhận hết AI             # 0 = bám gốc (band fusion: vân vừa lấy từ SR) · 1 = AI đẩy chi tiết (như repo flux cũ: chỉ khoá màu/mảng lớn)           # 0..1: khử đốm/nhiễu render ở mảng phẳng (tường vữa...) TRƯỚC khi phóng · 0 = tắt
    "tile_retries": -1,         # -1 = theo preset
    "tile_max_deltaE": 6.0,
    "align_tiles": True,
    "seed_mode": "fixed",
    "reference_mode": "auto",   # auto = profile · strip · tile · keep
    "color_lock": "detail_transfer",
    "lock_strength": 1.0,
    "lock_sigma": 8.0,
    "texture_lock_mult": 4.0,
    "material_denoise": -1.0,   # -1 = auto (min(denoise, 0.4)) · vùng tường/đá/gỗ/mảng phẳng
    "ai_detail": 0.6,           # 0..1: mức chi tiết mịn AI được thêm (vân vừa luôn lấy từ SR model của ảnh gốc)
    "detail_sigma": 2.0,        # ranh giới 'chi tiết mịn' (px ảnh ra): nhỏ hơn -> AI chỉ thêm chi tiết li ti
    "fidelity_gate": True,      # tự bỏ vân AI không khớp vân gốc (đốm/hạt trên tường)
    "flow_align": True,         # căn AI về vị trí gốc từng pixel trước khi trộn (cần OpenCV)
    "texture_ai": 0.0,          # 0..1: vùng lá/texture pha thêm AI trọn (dễ quầng mờ) · 0 = band fusion như mọi vùng
    "edge_guard": 0.8,
    "control_mode": "auto",     # auto = profile · tile · canny · lineart · grayscale · external
    "control_strength": -1.0,   # -1 = auto (native 0.6 · qwen21_fun 0.8)
    "control_start": 0.0,
    "control_end": 0.8,
    "async_qa": True,           # QA/cache ô chạy song song lúc GPU sample lô kế -> GPU ít phải chờ
    "preview_pick": "ô đã chọn",  # chọn ô preview bằng menu (xem PREVIEW_PICKS) · "ô đã chọn" = theo preview_tiles
    "chroma_lock": 1.0,         # 0..1: màu chi tiết lấy từ gốc (SR), AI chỉ góp độ sáng -> hết cành tím/lá hồng · 0 = như cũ
}

# menu chọn ô preview -> spec của parse_tiles ("" = bảng ô). "ô đã chọn" = dùng preview_tiles (bấm trên bảng)
PREVIEW_PICKS = {"ô đã chọn": None, "bảng ô": "", "1 ô khó nhất": "auto", "3 ô khó nhất": "auto3",
                 "5 ô khó nhất": "auto5", "tất cả ô": "all"}


def preview_spec(s):
    """Giải spec ô preview từ settings: preview_pick (menu) ưu tiên, 'ô đã chọn' -> preview_tiles."""
    v = PREVIEW_PICKS.get(s.get("preview_pick", "ô đã chọn"))
    return str(s.get("preview_tiles", "") or "").strip() if v is None else v


# v0.12.5: nút "Reset preview" -> xoá hết output/db9u_cache (ô preview + ô resume) và tăng token.
# Token trả qua IS_CHANGED của DB9U Upscale/Forge -> lần Run sau ComfyUI chạy lại node thật (không lấy kết quả cũ).
RESET_TOKEN = [0]


def cache_root():
    import folder_paths
    return os.path.join(folder_paths.get_output_directory(), "db9u_cache")


def reset_preview_cache():
    """Xoá toàn bộ cache DB9U. Trả số bộ cache (ảnh+thông số) đã xoá."""
    root = cache_root()
    n = len(os.listdir(root)) if os.path.isdir(root) else 0
    shutil.rmtree(root, ignore_errors=True)
    RESET_TOKEN[0] += 1
    return n


class Job:
    """Gom mọi tham số đã giải (auto -> giá trị thật) cho 1 lần chạy."""

    def __init__(self, model, denoise, cfg, preset, seed, settings, qwen_cn, native_cn):
        s = dict(DEFAULTS)
        s.update(settings or {})
        self.s = s
        self.profile, self.profile_note = profiles.detect_profile(model, s["profile"])
        p = self.profile
        tile, batch, self.hw_note = profiles.hardware_plan(p, has_qwen_cn=qwen_cn is not None)
        self.max_tile = s["max_tile"] or tile
        self.batch = s["batch_tiles"] or batch
        ov = s["overlap"] or max(128, int(self.max_tile / 6) // 32 * 32)
        steps = s["steps"] or p["steps"]
        retries = s["tile_retries"]
        if preset == "fast":
            if not s["steps"] and p["steps"] > 8:  # model distilled (≤8 bước) giữ nguyên
                steps = int(round(steps * 0.75))
            retries = 0 if retries < 0 else retries
        elif preset == "max":
            if not s["steps"]:
                steps = int(round(steps * 1.25))
            if not s["overlap"]:
                ov = int(ov * 1.5) // 32 * 32
            retries = 2 if retries < 0 else retries
        else:
            retries = 1 if retries < 0 else retries
        self.overlap, self.steps, self.retries = ov, max(1, steps), retries
        self.overlap_ratio = ov / self.max_tile
        self.cfg = cfg if cfg > 0 else p["cfg"]
        self.sampler = p["sampler"] if s["sampler"] == "auto" else s["sampler"]
        self.scheduler = p["scheduler"] if s["scheduler"] == "auto" else s["scheduler"]
        self.denoise = denoise
        td = s["texture_denoise"]
        # 0 = tắt denoise theo vùng lá (= denoise chính). Trước v0.11.3 giá trị 0 bị hiểu là denoise 0 cho vùng lá (sai với tooltip)
        self.texture_denoise = min(0.85, denoise + 0.2) if td < 0 else (denoise if td == 0 else td)
        md = s.get("material_denoise", -1.0)
        self.material_denoise = min(denoise, 0.4) if md < 0 else min(md, denoise)
        sd = s["sky_denoise"]
        self.sky_denoise = min(0.12, denoise) if sd < 0 else sd
        self.sky = self.sky_denoise < denoise - 1e-3
        self.regional = self.texture_denoise > denoise + 1e-3 or self.sky or self.material_denoise < denoise - 1e-3
        self.ref_mode = p["reference_mode"] if s["reference_mode"] == "auto" else s["reference_mode"]
        self.qwen_cn, self.native_cn = qwen_cn, native_cn
        self.cn_backend = "qwen21_fun" if qwen_cn is not None else ("native" if native_cn is not None else None)
        self.control_mode = p["control_mode"] if s["control_mode"] == "auto" else s["control_mode"]
        cs = s["control_strength"]
        self.control_strength = (0.8 if self.cn_backend == "qwen21_fun" else 0.6) if cs < 0 else cs
        if self.cn_backend == "qwen21_fun" or self.ref_mode == "tile":
            self.batch = 1
        self.seed = seed
        self.warn = self.profile.get("warning", "")

    def describe(self):
        cn = f"{self.cn_backend}/{self.control_mode} {self.control_strength}" if self.cn_backend else "không"
        reg = (f"vật liệu {self.material_denoise:.2f}, lá/texture {self.texture_denoise:.2f}"
               + (f", trời {self.sky_denoise:.2f}" if self.sky else "")) if self.regional else "tắt"
        return (f"[DB9U] Profile: {self.profile_note}\n"
                f"[DB9U] Phần cứng: {self.hw_note} (dùng ô ≤{self.max_tile}, giao {self.overlap}, {self.batch} ô/lượt)\n"
                f"[DB9U] Sampler: {self.sampler}/{self.scheduler} {self.steps} steps cfg {self.cfg} denoise {self.denoise} "
                f"| denoise vùng: {reg} | ControlNet: {cn} | ref: {self.ref_mode} | retry {self.retries}"
                + (f"\n[DB9U] LƯU Ý: {self.warn}" if self.warn and "chọn tay" not in self.profile_note else ""))


def target_size(w, h, target, custom_long_edge):
    if custom_long_edge > 0:
        s = custom_long_edge / max(w, h)
    else:
        kind, v = TARGETS[target]
        s = v if kind == "scale" else v / max(w, h)
    return s, max(1, round(w * s)), max(1, round(h * s))


def sr_upscale(img, upscale_model):
    """Chạy riêng upscale model (ESRGAN/DAT/...) -> ảnh ở tỉ lệ gốc của model (vd 4x)."""
    dev = mm.get_torch_device()
    upscale_model.to(dev)
    try:
        t = img.movedim(-1, -3).to(dev)
        tile = 512
        while True:
            try:
                t = comfy.utils.tiled_scale(t, lambda a: upscale_model(a), tile_x=tile, tile_y=tile,
                                            overlap=32, upscale_amount=upscale_model.scale)
                break
            except mm.OOM_EXCEPTION:
                mm.soft_empty_cache()
                tile //= 2
                if tile < 128:
                    raise
        x = t.movedim(-3, -1).clamp(0, 1).cpu()
    finally:
        upscale_model.to("cpu")
    return x.float()


def lanczos(x, tw, th):
    if x.shape[1:3] == (th, tw):
        return x.float()
    return comfy.utils.common_upscale(x.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1).clamp(0, 1).float()


def pre_upscale(img, tw, th, upscale_model=None):
    """img [1,H,W,3] -> [1,th,tw,3]. Upscale model (nếu có) rồi lanczos về đúng kích thước."""
    x = sr_upscale(img, upscale_model) if upscale_model is not None else img
    if x.shape[1:3] != (th, tw):
        x = comfy.utils.common_upscale(x.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
    return x.float()


def parse_tiles(spec, n, score=None):
    """'O1, O5 o6' · 'O3-O6' · '1 2' · 'auto' (1 ô khó nhất) · 'auto3' -> list index 0-based (giữ thứ tự, bỏ trùng)."""
    import re
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n))
    m = re.fullmatch(r"auto(\d*)", spec)
    if m:
        k = int(m.group(1) or 1)
        order = sorted(range(n), key=lambda i: -float(score[i])) if score is not None else list(range(n))
        return order[:max(1, min(k, n))]
    out = []
    for a, b in re.findall(r"o?\s*(\d+)(?:\s*-\s*o?\s*(\d+))?", spec):
        lo, hi = int(a), int(b or a)
        for v in range(min(lo, hi), max(lo, hi) + 1):
            if 1 <= v <= n and v - 1 not in out:
                out.append(v - 1)
    if not out:
        raise ValueError(f"preview_tiles '{spec}' không có ô hợp lệ (có O1..O{n})")
    return out


def parse_denoise_list(spec, main):
    """'0.3, 0.4 .5' -> [0.3, 0.4, 0.5] (0.05..1, bỏ trùng, giữ thứ tự). Trống -> [main]."""
    import re
    out = []
    for v in re.findall(r"\d*\.?\d+", str(spec or "")):
        d = round(min(1.0, max(0.05, float(v))), 3)
        if d not in out:
            out.append(d)
    return out or [round(main, 3)]


def _font(size):
    from PIL import ImageFont
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def tile_map(img, plan, score, texm, max_side=2048):
    """Ảnh bảng ô: khung từng ô + tên O1..On + % chi tiết (vùng lá/texture tô xanh nhạt)."""
    import numpy as np
    from PIL import Image, ImageDraw
    H, W = img.shape[1:3]
    k = min(1.0, max_side / max(H, W))
    w, h = max(1, round(W * k)), max(1, round(H * k))
    base = core.resize(img, w, h, "area")
    t = core.resize(texm.repeat(1, 1, 1, 3), w, h, "area")[..., :1]
    tint = torch.tensor([0.25, 0.85, 0.35])
    x = (base * (1 - 0.25 * t) + tint * 0.25 * t).clamp(0, 1)
    im = Image.fromarray((x[0].numpy() * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    fs = max(14, int(min(plan["tile_w"], plan["tile_h"]) * k / 6))
    font, font2 = _font(fs), _font(max(10, fs // 2))
    order = sorted(range(len(score)), key=lambda i: -float(score[i]))[:3]
    for i, (x0, y0) in enumerate(plan["coords"]):
        x1, y1 = min(W, x0 + plan["tile_w"]), min(H, y0 + plan["tile_h"])
        box = [x0 * k, y0 * k, x1 * k - 1, y1 * k - 1]
        hot = i in order
        d.rectangle(box, outline=(255, 170, 0, 255) if hot else (255, 255, 255, 200), width=3 if hot else 2)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        label = f"O{i + 1}"
        tb = d.textbbox((0, 0), label, font=font)
        tw_, th_ = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([cx - tw_ / 2 - 8, cy - th_ / 2 - 8, cx + tw_ / 2 + 8, cy + th_ / 2 + 10 + fs // 2],
                    fill=(0, 0, 0, 150))
        d.text((cx - tw_ / 2, cy - th_ / 2 - tb[1]), label, font=font, fill=(255, 255, 255, 255))
        sub = f"{float(score[i]) * 100:.0f}% chi tiết"
        sb = d.textbbox((0, 0), sub, font=font2)
        d.text((cx - (sb[2] - sb[0]) / 2, cy + th_ / 2 + 2), sub, font=font2,
               fill=(255, 190, 60, 255) if hot else (220, 220, 220, 255))
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).unsqueeze(0)


def mosaic(tiles, names, gap=8, max_side=4096, cols=None):
    """Ghép các ô [1,h,w,3] thành 1 bảng (≈ vuông, hoặc `cols` cột), có nhãn tên ô."""
    import math as _m
    import numpy as np
    from PIL import Image, ImageDraw
    if len(tiles) == 1:
        big = tiles[0]
    else:
        th, tw = tiles[0].shape[1:3]
        cols = cols or int(_m.ceil(_m.sqrt(len(tiles))))
        rows = int(_m.ceil(len(tiles) / cols))
        big = torch.full((1, rows * th + (rows - 1) * gap, cols * tw + (cols - 1) * gap, 3), 0.12)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            big[:, r * (th + gap):r * (th + gap) + th, c * (tw + gap):c * (tw + gap) + tw] = t[..., :3]
    kk = min(1.0, max_side / max(big.shape[1:3]))
    if kk < 1:
        big = core.resize(big, round(big.shape[2] * kk), round(big.shape[1] * kk), "area")
    im = Image.fromarray((big[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    font = _font(max(14, im.width // 40))
    th, tw = tiles[0].shape[1:3]
    cols = (cols or int(_m.ceil(_m.sqrt(len(tiles))))) if len(tiles) > 1 else 1
    for i, nm in enumerate(names):
        r, c = divmod(i, cols)
        x0, y0 = c * (tw + gap) * kk + 6, r * (th + gap) * kk + 6
        tb = d.textbbox((0, 0), nm, font=font)
        d.rectangle([x0, y0, x0 + tb[2] - tb[0] + 12, y0 + tb[3] - tb[1] + 12], fill=(0, 0, 0, 160))
        d.text((x0 + 6, y0 + 6 - tb[1]), nm, font=font, fill=(255, 255, 255, 255))
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).unsqueeze(0)


class Engine:
    def __init__(self, job, model, vae, positive, negative, control_image=None):
        self.j, self.model, self.vae = job, model, vae
        self.pos, self.neg = positive, negative
        self.control_image = control_image
        self.cn_disabled = False
        self.log = []
        self.board = None  # thông tin bảng ô (cho JS bấm chọn ô)
        self.t_gpu = 0.0

    def say(self, msg):
        self.log.append(msg)
        print(msg)

    # ---------------------------------------------------------------- 1 lô ô
    def _run_group(self, tiles, ctiles, mtiles, idx, seed, d):
        """tiles/ctiles/mtiles: tensor cả pass; idx: list chỉ số ô trong lô. Trả [len(idx),h,w,3]."""
        j = self.j
        t = tiles[idx]
        h, w = t.shape[1:3]
        lat = sampling.vae_encode(self.vae, t)
        pos = sampling.prep_cond(self.pos, j.ref_mode, lat if j.ref_mode == "tile" else None)
        neg = sampling.prep_cond(self.neg, j.ref_mode, lat if j.ref_mode == "tile" else None)
        model = self.model
        use_cn = j.cn_backend is not None and not self.cn_disabled
        if use_cn:
            c = ctiles[idx] if ctiles is not None else torch.cat(
                [core.control_map(t[k:k + 1], j.control_mode) for k in range(len(idx))], 0)
            if j.cn_backend == "native":
                pos, neg = sampling.apply_native_cn(pos, neg, j.native_cn, self.vae, c, j.control_strength,
                                                    j.s["control_start"], j.s["control_end"])
            else:
                model = sampling.apply_qwen21_cn(model, j.qwen_cn, self.vae, c, j.control_strength,
                                                 j.s["control_start"], j.s["control_end"])
        dmask, d_run = None, d
        if j.regional and mtiles is not None:
            # mtiles = bản đồ denoise tuyệt đối theo pixel (zone map); retry hạ đều theo (denoise - d)
            dm = (mtiles[idx].movedim(-1, 1) - (j.denoise - d)).clamp(min=0.02)  # [B,1,h,w]
            hi = float(dm.max())
            d_run = min(1.0, hi)  # ô đồng nhất (toàn trời / toàn lá) vẫn dùng đúng mức của vùng
            if hi > float(dm.min()) + 1e-3:
                dmask = (dm / hi).clamp(0, 1)
                model = sampling.with_regional_denoise(model)
        try:
            out = sampling.ksample(model, lat, seed, j.steps, j.cfg, j.sampler, j.scheduler, pos, neg, d_run,
                                   noise_mask=dmask, pixel_size=(w, h))
        except RuntimeError as e:
            if use_cn and sampling.is_shape_mismatch(e):
                self.say(f"[DB9U] ControlNet KHÔNG tương thích model -> tắt ControlNet cho các ô còn lại ({e})")
                self.cn_disabled = True
                return self._run_group(tiles, ctiles, mtiles, idx, seed, d)
            raise
        dec = sampling.vae_decode(self.vae, out)
        if dec.shape[1:3] != (h, w):
            dec = core.resize(dec, w, h, "bicubic")
        return dec

    # ---------------------------------------------------------------- zone map / cache
    def zone_map(self, base):
        """Bản đồ denoise theo pixel [1,H,W,1]: kiến trúc = denoise, lá/texture = texture_denoise, trời = sky_denoise."""
        j = self.j
        d = torch.full_like(base[..., :1], j.material_denoise)
        if abs(j.texture_denoise - j.material_denoise) > 1e-3:
            tex = core.texture_mask(base).movedim(1, -1)
            d = d + (j.texture_denoise - j.material_denoise) * tex
        if j.sky:
            sky = core.sky_mask(base).movedim(1, -1)
            d = d * (1 - sky) + j.sky_denoise * sky
            self.say(f"[DB9U]   trời: {float(sky.mean()) * 100:.0f}% ảnh (denoise {j.sky_denoise:.2f})")
        return d

    def _cache_dir(self, label):
        if not self.j.s["resume"] or not getattr(self, "cache_key", None):
            return None
        try:
            import folder_paths
            d = os.path.join(folder_paths.get_output_directory(), "db9u_cache", self.cache_key, label.replace("/", "-").replace(" ", "_"))
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            return None

    def _pass_cache(self, label, plan):
        return self._cache_dir(f"{label}_{plan['tile_w']}x{plan['tile_h']}_{plan['cols']}x{plan['rows']}")

    def prep(self, img):
        """Làm sạch ảnh gốc trước khi phóng (despeckle > 0)."""
        ds = float(self.j.s.get("despeckle", 0.0) or 0.0)
        if ds <= 0:
            return img
        out, frac = core.despeckle(img, ds)
        self.say(f"[DB9U] despeckle {ds:.2f}: làm sạch mảng phẳng, {frac * 100:.2f}% pixel là đốm đơn lẻ -> đã thay")
        return out

    @staticmethod
    def sr_at(sr_nat, nw, nh):
        """SR ảnh gốc (đã half) về đúng cỡ lượt — dùng chung cho run_image và preview để ô preview dùng lại được."""
        return (core.resize(sr_nat.float(), nw, nh, "area") if sr_nat.shape[2] >= nw
                else lanczos(sr_nat.float(), nw, nh))

    def set_cache_key(self, img, target, upscale_model=None):
        """Khoá cache = toàn bộ ảnh gốc + prompt + model/LoRA + mọi thông số ảnh hưởng kết quả."""
        j = self.j
        h = hashlib.sha1((img.clamp(0, 1) * 65535).round().to(torch.int32).numpy().tobytes())

        def cond_sig(c):
            try:
                return [(tuple(x[0].shape), round(float(x[0].float().sum()), 3), round(float(x[0].float().abs().mean()), 6))
                        for x in c]
            except Exception:
                return repr(type(c))
        try:
            patches = sorted((k, len(v)) for k, v in getattr(self.model, "patches", {}).items())
            msig = (profiles.model_signature(self.model), len(patches), hash(tuple(patches)))
        except Exception:
            msig = repr(type(self.model))
        ci = None if self.control_image is None else round(float(self.control_image.float().sum()), 2)
        um = None if upscale_model is None else (type(upscale_model).__name__, getattr(upscale_model, "scale", None))
        s = {k: v for k, v in j.s.items() if k not in ("resume", "preview_tile", "preview_tiles", "preview_denoise",
                                                       "reuse_preview", "preview_layout", "preview_pick")}
        h.update(repr((target, j.denoise, j.cfg, j.steps, j.sampler, j.scheduler, j.seed, j.profile["name"],
                       j.texture_denoise, j.sky_denoise, j.cn_backend, j.control_mode, j.control_strength,
                       j.ref_mode, j.retries, sorted(s.items()), cond_sig(self.pos), cond_sig(self.neg),
                       msig, ci, um, tuple(img.shape))).encode())
        self.cache_key = h.hexdigest()[:16]

    def clear_cache(self):
        if getattr(self, "cache_key", None):
            try:
                import folder_paths
                shutil.rmtree(os.path.join(folder_paths.get_output_directory(), "db9u_cache", self.cache_key), ignore_errors=True)
            except Exception:
                pass

    # ---------------------------------------------------------------- xem trước 1 ô
    def run_preview(self, img, target, upscale_model):
        """preview_tile: preview_tiles trống -> chỉ vẽ BẢNG Ô (O1..On, vài giây, không chạy model).
        'O1,O5' / 'O3-O6' / 'auto' / 'auto3' -> chạy riêng các ô đó ở kích thước cuối, ghép thành bảng so sánh.
        preview_denoise '0.3,0.4,0.5' -> mỗi ô chạy ở từng mức (hàng = ô, cột = denoise).
        Ảnh chỉ cần 1 lượt + resume bật -> ô ở denoise chính được lưu cache, chạy full dùng lại (không chạy model lại)."""
        j = self.j
        W, H = img.shape[2], img.shape[1]
        s, tw, th = target_size(W, H, target, j.s["custom_long_edge"])
        plan = core.plan_tiles(tw, th, j.max_tile, j.overlap)
        n = len(plan["coords"])
        spec = preview_spec(j.s)
        lz0 = lanczos(img, tw, th)
        texm = core.texture_mask(lz0).movedim(1, -1)
        score = core.split_tiles(texm, plan).mean((1, 2, 3))
        if not spec:
            self.say(f"[DB9U] BẢNG Ô: {tw}x{th} -> {n} ô ({plan['cols']} cột x {plan['rows']} hàng, "
                     f"mỗi ô {plan['tile_w']}x{plan['tile_h']}, giao {plan['overlap']}px)")
            order = sorted(range(n), key=lambda k: -float(score[k]))
            self.say("[DB9U] Ô nhiều chi tiết nhất: " + ", ".join(f"O{k + 1}" for k in order[:5])
                     + "  -> bấm '🔲 Chọn ô trên bảng' (hoặc gõ preview_tiles, vd O3,O7) rồi chạy lại")
            m = tile_map(lz0, plan, score, texm)
            self.board = {"W": tw, "H": th, "tile_w": plan["tile_w"], "tile_h": plan["tile_h"],
                          "cols": plan["cols"], "rows": plan["rows"], "coords": [list(c) for c in plan["coords"]],
                          "score": [round(float(v), 4) for v in score]}
            return m, core.resize(lz0, m.shape[2], m.shape[1], "area"), m
        idx = parse_tiles(spec, n, score)
        ds = parse_denoise_list(j.s.get("preview_denoise", ""), j.denoise)
        self.say(f"[DB9U] PREVIEW {len(idx)} ô: " + ", ".join(f"O{k + 1}" for k in idx) + f" / {n} ô, kích thước cuối {tw}x{th}"
                 + (f" | so denoise: {', '.join(f'{d:.2f}' for d in ds)}" if len(ds) > 1 else ""))
        # ô cache được cho chạy full: chỉ khi 1 lượt (ô preview = đúng ô của lượt duy nhất)
        one_pass = core.pass_count(s, j.s["max_scale_per_pass"]) == 1
        cache = None
        if j.s.get("reuse_preview", True):
            if not one_pass:
                self.say("[DB9U]   (ảnh cần >1 lượt -> ô preview KHÔNG dùng lại được khi chạy full)")
            elif not j.s["resume"]:
                self.say("[DB9U]   (resume tắt -> ô preview không được lưu để dùng lại)")
            else:
                self.set_cache_key(img, target, upscale_model)
                cache = self._pass_cache("lượt 1/1", plan)
        img = self.prep(img)
        sr = None
        if upscale_model is not None:
            sr = self.sr_at(sr_upscale(img, upscale_model).half(), tw, th)
        base = sr if sr is not None else (lanczos(img, tw, th) if j.s.get("despeckle", 0) else lz0)
        tiles = core.split_tiles(base, plan)
        ltiles = core.split_tiles(lanczos(img, tw, th) if j.s.get("despeckle", 0) else lz0, plan)  # màu gốc cho chroma_lock
        zm = self.zone_map(base) if j.regional else None
        mt = core.split_tiles(zm, plan) if zm is not None else None
        ctiles = None
        if j.cn_backend and j.control_mode == "external":
            if self.control_image is None:
                raise ValueError("control_mode=external nhưng chưa nối control_image")
            ci = core.resize(self.control_image[:1, ..., :3].float().cpu(), tw, th, "bicubic")
            ctiles = core.split_tiles(ci, plan)
        outs, refs, zs, names = [], [], [], []
        saved = 0
        for k in idx:
            t = tiles[k:k + 1]
            seed = (j.seed + (k if j.s["seed_mode"] == "per_tile" else 0)) & 0xFFFFFFFFFFFFFFFF
            for d in ds:
                self.say(f"[DB9U]   O{k + 1}" + (f" denoise {d:.2f}" if len(ds) > 1 else "") + ":")
                out = self._run_group(tiles, ctiles, mt, [k], seed, d)
                main = abs(d - j.denoise) < 1e-6
                if main and len(ds) == 1:
                    out = self._qa_tile(tiles, ctiles, mt, k, out, n)  # y hệt chạy full (căn + ΔE + retry)
                elif j.s["align_tiles"]:
                    out = core.align_to_ref(out, t)[0]
                if main and cache:
                    torch.save(out.half(), os.path.join(cache, f"t{k:03d}.pt"))
                    saved += 1
                outs.append(self.finish_pass(out, t, t if sr is not None else None, ltiles[k:k + 1]))
                refs.append(t)
                zmk = mt[k:k + 1] - (j.denoise - d) if mt is not None else torch.full_like(t[..., :1], d)
                zs.append(zmk.clamp(0, 1).repeat(1, 1, 1, 3))
                names.append(f"O{k + 1}" + (f" · d{d:.2f}" if len(ds) > 1 else ""))
        if saved:
            self.say(f"[DB9U] Đã lưu {saved} ô preview -> tắt preview_tile & chạy full sẽ dùng lại (giữ nguyên ảnh/prompt/thông số)")
        if len(ds) == 1 and j.s.get("preview_layout", "frame") == "frame":
            # ĐẶT Ô VÀO ĐÚNG CHỖ trên cả khung ảnh (kích thước cuối) -> so sánh được với ảnh gốc cùng khung
            parts = list(tiles.split(1, 0))
            for k, o in zip(idx, outs):
                parts[k] = o
            frame = core.merge_tiles(torch.cat(parts, 0), plan)
            ref = core.resize(img, tw, th, "bicubic")
            zf = (zm if zm is not None else torch.full_like(frame[..., :1], j.denoise)).repeat(1, 1, 1, 3)
            self.say(f"[DB9U] Khung đầy đủ {tw}x{th}: ô {', '.join(names)} đã enhance, phần còn lại = ảnh phóng thường "
                     "(so với original_resized trong Finish/QAQC)")
            return frame, ref, zf
        cols = len(ds) if len(ds) > 1 else None  # so denoise: hàng = ô, cột = mức denoise
        return mosaic(outs, names, cols=cols), mosaic(refs, names, cols=cols), mosaic(zs, names, cols=cols)

    # ---------------------------------------------------------------- QA 1 ô
    def _qa_check(self, tiles, i, out, n, d, attempt):
        """Căn trôi + đo ΔE 1 ô (chỉ CPU, chạy được ở luồng phụ). Trả (out, ok)."""
        j = self.j
        t = tiles[i:i + 1]
        note = ""
        if j.s["align_tiles"]:
            out, (dx, dy), moved = core.align_to_ref(out, t)
            note = f" trôi({dx:+.2f},{dy:+.2f}){'→kéo về' if moved else ''}"
        err = core.tile_color_error(t, out)
        ok = err <= j.s["tile_max_deltaE"]
        self.say(f"[DB9U]   ô {i + 1:02d}/{n} lần {attempt} denoise {d:.2f} ΔE {err:.2f}{note} "
                 f"{'OK' if ok else 'LỆCH'}")
        return out, ok

    def _qa_retry(self, tiles, ctiles, mtiles, i, out, n, start=1):
        """Chạy lại ô lệch (hạ denoise) từ lần 'start' tới j.retries."""
        j = self.j
        d = j.denoise
        for attempt in range(start, j.retries + 1):
            d = round(max(j.s["min_denoise"], j.denoise - 0.05 * attempt), 3)
            t1 = time.time()
            o = self._run_group(tiles, ctiles, mtiles, [i], (j.seed + 1000 * attempt + i) & 0xFFFFFFFFFFFFFFFF, d)
            self.t_gpu += time.time() - t1
            out, ok = self._qa_check(tiles, i, o, n, d, attempt)
            if ok:
                break
        return out

    def _qa_tile(self, tiles, ctiles, mtiles, i, out, n):
        """Căn trôi + đo ΔE với ô vào; lệch -> chạy lại (hạ denoise) tới j.retries lần."""
        out, ok = self._qa_check(tiles, i, out, n, self.j.denoise, 0)
        if ok or self.j.retries <= 0:
            return out
        return self._qa_retry(tiles, ctiles, mtiles, i, out, n)

    def vram_report(self):
        """Sau lô đầu: model nạp đủ lên GPU chưa? Không đủ -> GPU phải chờ PCIe (nguyên nhân chính không chạy hết công suất)."""
        try:
            base = getattr(self.model, "model", None)
            for lm in getattr(mm, "current_loaded_models", []):
                mp = getattr(lm, "model", None)
                if mp is None or getattr(mp, "model", None) is not base:
                    continue
                tot, ld = mp.model_size() / 1024 ** 3, mp.loaded_size() / 1024 ** 3
                free = mm.get_free_memory(mm.get_torch_device()) / 1024 ** 3
                if tot > 0 and ld < tot * 0.97:
                    self.say(f"[DB9U] ⚠ VRAM: model chỉ nạp {ld:.1f}/{tot:.1f} GB lên GPU ({ld / tot * 100:.0f}%) -> phần còn lại "
                             f"đẩy qua PCIe mỗi bước, GPU phải chờ = KHÔNG chạy hết công suất. Dùng bản model nhỏ hơn "
                             f"(fp8/GGUF), tắt app đang chiếm VRAM, hoặc giảm max_tile")
                else:
                    self.say(f"[DB9U] VRAM: model nạp đủ {ld:.1f} GB lên GPU, còn trống ~{free:.1f} GB"
                             + (" -> có thể thử batch_tiles 2" if free > 4 and self.j.batch == 1
                                and self.j.cn_backend != "qwen21_fun" and self.j.ref_mode != "tile" else ""))
                return
        except Exception:
            pass

    # ---------------------------------------------------------------- khoá sau mỗi lượt
    def finish_pass(self, ai, base, sr, cref=None):
        """ai: kết quả ghép; base: ảnh vào lượt này; sr: SR model của ẢNH GỐC cùng cỡ (None -> dùng base).
        1) căn AI về đúng vị trí gốc (optical flow) 2) band fusion MỌI vùng: mảng lớn gốc + vân vừa SR/gốc
        + chi tiết mịn AI (chỉ nhận chỗ khớp gốc)."""
        j = self.j
        src = sr if sr is not None else base
        if j.s.get("flow_align", True):
            ai, p95 = core.flow_align(ai, src)
            self.say("[DB9U]   căn flow: " + (f"AI lệch p95 {p95:.2f}px -> đã kéo về vị trí gốc" if p95 is not None
                                               else "THIẾU OpenCV (cv2) -> bỏ qua, dễ bóng mờ"))
        enh = float(j.s.get("enhance", 1.0))
        need_tex = j.s.get("texture_ai", 0.0) > 0 or enh > 0
        texm = core.texture_mask(base).movedim(1, -1) if need_tex else None
        out = None
        if enh < 1:
            out, g = core.band_fusion(ai, base, src, texm, j.s["color_lock"], j.s["lock_strength"], j.s["lock_sigma"],
                                      j.s.get("detail_sigma", 2.0), j.s.get("ai_detail", 0.6),
                                      j.s.get("fidelity_gate", True), j.s.get("texture_lock_mult", 4.0),
                                      return_gate=True, texture_ai=j.s.get("texture_ai", 0.0))
        if enh > 0:
            # ENHANCE: giữ toàn bộ vân vừa + chi tiết mịn của AI, chỉ khoá màu/mảng lớn (> lock_sigma) theo ảnh vào
            # (cách repo db9_flux_locked_upscale làm: detail-preserving color lock) — flow_align đã chống bóng mờ
            sl = float(j.s.get("structure_lock", 0.5))
            e, gg = core.enhance_fusion(ai, base, src, texm, j.s["color_lock"], j.s["lock_strength"], j.s["lock_sigma"],
                                        j.s.get("detail_sigma", 2.0), j.s.get("texture_lock_mult", 4.0), sl,
                                        return_gate=True)
            out = e if out is None else out + (e - out) * enh
            gate = (f" | structure_lock {sl:.2f}: nhận vân vừa AI {float(gg[0].mean()) * 100:.0f}%, "
                    f"chi tiết mịn AI {float(gg[1].mean()) * 100:.0f}% (phần AI vẽ lại hình -> thay bằng "
                    f"{'SR' if sr is not None else 'ảnh vào'})") if gg is not None else " | structure_lock 0: nhận hết AI"
            self.say(f"[DB9U]   enhance {enh:.2f}: khoá màu/mảng > {j.s['lock_sigma']:.0f}px" + gate
                     + ("" if enh >= 1 else f" (trộn {(1 - enh) * 100:.0f}% band fusion bám gốc)"))
        else:
            self.say(f"[DB9U]   band fusion ({'SR model' if sr is not None else 'KHÔNG có upscale model'}): "
                     f"chi tiết mịn AI nhận trung bình {float(g.mean()) * 100:.0f}% (ai_detail {j.s.get('ai_detail', 0.6)})")
        cl = float(j.s.get("chroma_lock", 0.0))
        if cl > 0:
            # màu lấy từ ẢNH GỐC phóng lanczos (cref) — không lấy từ SR vì SR model cũng có thể sinh viền màu
            out = core.chroma_lock(out, cref if cref is not None else src, cl)
            self.say(f"[DB9U]   chroma_lock {cl:.2f}: màu chi tiết theo {'ảnh gốc (lanczos)' if cref is not None else 'ảnh vào'}, AI + SR chỉ góp độ sáng")
        if j.s["edge_guard"] > 0:
            out = core.edge_guard(out, base, j.s["edge_guard"], 1.5, 3)
        return out

    # ---------------------------------------------------------------- 1 lượt
    def run_pass(self, base, label):
        j = self.j
        plan = core.plan_tiles(base.shape[2], base.shape[1], j.max_tile, j.overlap)
        tiles = core.split_tiles(base, plan)
        n = len(tiles)
        tm = self.zone_map(base) if j.regional else None
        mtiles = core.split_tiles(tm, plan) if tm is not None else None
        ctiles = None
        if j.cn_backend and j.control_mode == "external":
            if self.control_image is None:
                raise ValueError("control_mode=external nhưng chưa nối control_image")
            ci = core.resize(self.control_image[:1, ..., :3].float().cpu(), base.shape[2], base.shape[1], "bicubic")
            ctiles = core.split_tiles(ci, plan)
        self.say(f"[DB9U] {label}: {base.shape[2]}x{base.shape[1]} | {plan['cols']}x{plan['rows']} = {n} ô "
                 f"{plan['tile_w']}x{plan['tile_h']}, giao ≥{plan['overlap']}")
        res = [None] * n
        cache = self._pass_cache(label, plan)
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
                self.say(f"[DB9U] {label}: resume — dùng lại {reused}/{n} ô đã xong (gồm ô đã preview)")
        pbar = comfy.utils.ProgressBar(n)
        t0 = time.time()
        self.t_gpu = 0.0
        todo = [i for i in range(n) if res[i] is None]
        done = n - len(todo)
        pbar.update(done)
        # QA + lưu cache chạy ở luồng phụ, song song lúc GPU sample lô kế tiếp -> GPU không phải chờ CPU
        pool = ThreadPoolExecutor(max_workers=1) if j.s.get("async_qa", True) and len(todo) > 1 else None
        pending, retry = [], []

        def check_group(idx, outs):
            r = []
            for k, i in enumerate(idx):
                o, ok = self._qa_check(tiles, i, outs[k:k + 1], n, j.denoise, 0)
                if cache and (ok or j.retries <= 0):
                    torch.save(o.half(), os.path.join(cache, f"t{i:03d}.pt"))
                r.append((i, o, ok))
            return r

        def collect(items):
            nonlocal done
            for i, o, ok in items:
                res[i] = o
                if not ok and j.retries > 0:
                    retry.append(i)
            done += len(items)
            pbar.update(len(items))

        try:
            for gi in range(0, len(todo), j.batch):
                mm.throw_exception_if_processing_interrupted()
                idx = todo[gi:gi + j.batch]
                g0 = idx[0]
                seed = (j.seed + (g0 if j.s["seed_mode"] == "per_tile" else 0)) & 0xFFFFFFFFFFFFFFFF
                t1 = time.time()
                outs = self._run_group(tiles, ctiles, mtiles, idx, seed, j.denoise)
                self.t_gpu += time.time() - t1
                if gi == 0:
                    self.vram_report()
                if pool is None:
                    collect(check_group(idx, outs))
                else:
                    pending.append(pool.submit(check_group, idx, outs))
                    while pending and pending[0].done():
                        collect(pending.pop(0).result())
                el = time.time() - t0
                ran = max(1, gi + len(idx))
                self.say(f"[DB9U] {label}: sample xong {n - len(todo) + ran}/{n} ô | {el / 60:.1f} phút | "
                         f"còn ~{el / ran * (len(todo) - ran) / 60:.1f} phút")
            while pending:
                collect(pending.pop(0).result())
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        for i in retry:
            mm.throw_exception_if_processing_interrupted()
            res[i] = self._qa_retry(tiles, ctiles, mtiles, i, res[i], n)
            if cache:
                torch.save(res[i].half(), os.path.join(cache, f"t{i:03d}.pt"))
        el = time.time() - t0
        if todo and el > 0:
            self.say(f"[DB9U] {label}: GPU bận (sample+VAE) {self.t_gpu:.0f}s / {el:.0f}s = {self.t_gpu / el * 100:.0f}% thời gian"
                     + (f" | chạy lại {len(retry)} ô lệch" if retry else ""))
        merged = core.merge_tiles(torch.cat(res, 0), plan)
        return merged, tm

    # ---------------------------------------------------------------- 1 ảnh
    def run_image(self, img, target, upscale_model):
        j = self.j
        W, H = img.shape[2], img.shape[1]
        s, tw, th = target_size(W, H, target, j.s["custom_long_edge"])
        n = core.pass_count(s, j.s["max_scale_per_pass"])
        self.say(f"[DB9U] {W}x{H} -> {tw}x{th} (x{s:.3f}) | {n} lượt, mỗi lượt ~x{s ** (1 / n):.3f}")
        self.set_cache_key(img, target, upscale_model)
        img = self.prep(img)
        cur, tex = img, None
        sr_nat = None
        if upscale_model is not None:
            sr_nat = sr_upscale(img, upscale_model).half()  # SR của ẢNH GỐC, chạy 1 lần, dùng cho mọi lượt
            self.say(f"[DB9U] Upscale model: {sr_nat.shape[2]}x{sr_nat.shape[1]} -> nguồn vân vật liệu cho mọi lượt")
        else:
            self.say("[DB9U] LƯU Ý: chưa nối upscale_model -> không có band fusion (vân vật liệu do AI tự vẽ). "
                     "Khuyên nối 4x-UltraSharp / Nomos / DAT")
        for k in range(n):
            if k == n - 1:
                nw, nh = tw, th
            else:
                f = s ** ((k + 1) / n)
                nw, nh = round(W * f), round(H * f)
            sr_k = None
            if sr_nat is not None:
                sr_k = self.sr_at(sr_nat, nw, nh)
            lz = lanczos(cur, nw, nh)
            if sr_k is None:
                base = lz
            elif k == 0:
                base = sr_k
            else:  # giữ mảng/khối đã enhance của lượt trước, thay chi tiết mịn bằng SR của ảnh gốc
                base = core.detail_graft(lz, sr_k, j.s.get("detail_sigma", 2.0))
            del lz
            label = f"lượt {k + 1}/{n}"
            while True:
                try:
                    merged, tex = self.run_pass(base, label)
                    break
                except mm.OOM_EXCEPTION:
                    mm.soft_empty_cache()
                    if j.batch > 1:
                        j.batch = max(1, j.batch // 2)
                        self.say(f"[DB9U] Hết VRAM -> giảm còn {j.batch} ô/lượt, chạy lại {label}")
                    elif j.max_tile > 768:
                        j.max_tile -= 512
                        j.overlap = max(128, int(j.max_tile * j.overlap_ratio) // 32 * 32)
                        self.say(f"[DB9U] Hết VRAM -> giảm ô còn {j.max_tile}px, chạy lại {label}")
                    else:
                        raise
            t1 = time.time()
            cur = self.finish_pass(merged, base, sr_k, lanczos(img, nw, nh) if j.s.get("chroma_lock", 0) > 0 else None)
            self.say(f"[DB9U]   khoá/fusion sau lượt (CPU, GPU nghỉ): {time.time() - t1:.0f}s")
            del merged, sr_k
        if n > 1 and j.s["color_lock"] != "none":
            # khoá màu lần cuối theo ẢNH GỐC để sai số màu không cộng dồn qua các lượt
            ref = core.resize(img, tw, th, "bicubic")
            cur = core.color_lock_zoned(cur, ref, j.s["color_lock"], j.s["lock_strength"], j.s["lock_sigma"] * s,
                                        core.texture_mask(ref).movedim(1, -1), j.s.get("texture_lock_mult", 4.0))
        self.clear_cache()
        if tex is None:
            tex = torch.full((1, th, tw, 1), j.denoise)
        return cur, tex.repeat(1, 1, 1, 3)
