"""Pipeline chính DB9_Ultimate: pre-upscale -> nhiều lượt (≤2x) -> chia ô -> sample theo lô -> QA/retry -> ghép -> khoá."""
import math
import time

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
    "min_denoise": 0.25,
    "tile_retries": -1,         # -1 = theo preset
    "tile_max_deltaE": 6.0,
    "align_tiles": True,
    "seed_mode": "fixed",
    "reference_mode": "auto",   # auto = profile · strip · tile · keep
    "color_lock": "detail_transfer",
    "lock_strength": 1.0,
    "lock_sigma": 8.0,
    "edge_guard": 0.8,
    "control_mode": "auto",     # auto = profile · tile · canny · lineart · grayscale · external
    "control_strength": -1.0,   # -1 = auto (native 0.6 · qwen21_fun 0.8)
    "control_start": 0.0,
    "control_end": 0.8,
}


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
        self.texture_denoise = min(0.85, denoise + 0.2) if td < 0 else td
        self.regional = self.texture_denoise > denoise + 1e-3
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
        reg = f"lá/texture {self.texture_denoise:.2f}" if self.regional else "tắt"
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


def pre_upscale(img, tw, th, upscale_model=None):
    """img [1,H,W,3] -> [1,th,tw,3]. Upscale model (nếu có) rồi lanczos về đúng kích thước."""
    x = img
    if upscale_model is not None:
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
    if x.shape[1:3] != (th, tw):
        x = comfy.utils.common_upscale(x.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
    return x.float()


class Engine:
    def __init__(self, job, model, vae, positive, negative, control_image=None):
        self.j, self.model, self.vae = job, model, vae
        self.pos, self.neg = positive, negative
        self.control_image = control_image
        self.cn_disabled = False
        self.log = []

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
        if j.regional:
            hi = min(1.0, j.texture_denoise - (j.denoise - d))
            if hi > d + 1e-3:
                tex = mtiles[idx].movedim(-1, 1)  # [B,1,h,w]
                dmask = (d / hi + (1 - d / hi) * tex).clamp(0, 1)
                d_run = hi
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

    # ---------------------------------------------------------------- 1 lượt
    def run_pass(self, base, label):
        j = self.j
        plan = core.plan_tiles(base.shape[2], base.shape[1], j.max_tile, j.overlap)
        tiles = core.split_tiles(base, plan)
        n = len(tiles)
        tm = core.texture_mask(base).movedim(1, -1) if j.regional else None
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
        pbar = comfy.utils.ProgressBar(n)
        t0 = time.time()
        done = 0
        for g0 in range(0, n, j.batch):
            mm.throw_exception_if_processing_interrupted()
            idx = list(range(g0, min(n, g0 + j.batch)))
            seed = (j.seed + (g0 if j.s["seed_mode"] == "per_tile" else 0)) & 0xFFFFFFFFFFFFFFFF
            outs = self._run_group(tiles, ctiles, mtiles, idx, seed, j.denoise)
            for k, i in enumerate(idx):
                out, t = outs[k:k + 1], tiles[i:i + 1]
                d, attempt = j.denoise, 0
                while True:
                    note = ""
                    if j.s["align_tiles"]:
                        out, (dx, dy), moved = core.align_to_ref(out, t)
                        note = f" trôi({dx:+.2f},{dy:+.2f}){'→kéo về' if moved else ''}"
                    err = core.tile_color_error(t, out)
                    ok = err <= j.s["tile_max_deltaE"]
                    self.say(f"[DB9U]   ô {i + 1:02d}/{n} lần {attempt} denoise {d:.2f} ΔE {err:.2f}{note} "
                             f"{'OK' if ok else 'LỆCH'}")
                    if ok or attempt >= j.retries:
                        break
                    attempt += 1
                    d = round(max(j.s["min_denoise"], d - 0.05), 3)
                    out = self._run_group(tiles, ctiles, mtiles, [i], (j.seed + 1000 * attempt + i) & 0xFFFFFFFFFFFFFFFF, d)
                res[i] = out
            done += len(idx)
            pbar.update(len(idx))
            el = time.time() - t0
            self.say(f"[DB9U] {label}: xong {done}/{n} ô | {el / 60:.1f} phút | còn ~{el / done * (n - done) / 60:.1f} phút")
        merged = core.merge_tiles(torch.cat(res, 0), plan)
        return merged, tm

    # ---------------------------------------------------------------- 1 ảnh
    def run_image(self, img, target, upscale_model):
        j = self.j
        W, H = img.shape[2], img.shape[1]
        s, tw, th = target_size(W, H, target, j.s["custom_long_edge"])
        n = core.pass_count(s, j.s["max_scale_per_pass"])
        self.say(f"[DB9U] {W}x{H} -> {tw}x{th} (x{s:.3f}) | {n} lượt, mỗi lượt ~x{s ** (1 / n):.3f}")
        cur, tex = img, None
        for k in range(n):
            if k == n - 1:
                nw, nh = tw, th
            else:
                f = s ** ((k + 1) / n)
                nw, nh = round(W * f), round(H * f)
            base = pre_upscale(cur, nw, nh, upscale_model if k == 0 else None)
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
            cur = core.color_lock(merged, base, j.s["color_lock"], j.s["lock_strength"], j.s["lock_sigma"])
            if j.s["edge_guard"] > 0:
                cur = core.edge_guard(cur, base, j.s["edge_guard"], 1.5, 3)
        if n > 1 and j.s["color_lock"] != "none":
            # khoá màu lần cuối theo ẢNH GỐC để sai số màu không cộng dồn qua các lượt
            ref = core.resize(img, tw, th, "bicubic")
            cur = core.color_lock(cur, ref, j.s["color_lock"], j.s["lock_strength"], j.s["lock_sigma"] * s)
        if tex is None:
            tex = torch.zeros(1, th, tw, 1)
        return cur, tex.repeat(1, 1, 1, 3)
