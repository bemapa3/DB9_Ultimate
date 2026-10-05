"""DB9U Color Grade — các lệnh chỉnh ảnh kiểu Lightroom / Camera Raw / Photoshop.
Mọi node: IMAGE -> IMAGE, nối nối tiếp được. Tính trên CPU (an toàn VRAM cho ảnh 10K).
Công thức là xấp xỉ (không phải thuật toán độc quyền của Adobe)."""
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import core

CATEGORY = "DB9 Ultimate/Color"
LUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "luts")


def _srgb_to_lin(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def _lin_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-10) ** (1 / 2.4) - 0.055)


def _lum(x):
    return (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722).unsqueeze(-1)


def _blur_hw1(x, sigma):
    """x [B,H,W,1] -> blur."""
    return core.gaussian_blur(x.movedim(-1, 1), sigma).movedim(1, -1)


def _set_lum(rgb, new_l, old_l):
    """Đổi độ sáng giữ màu. Pixel gần đen: cộng thêm (nhân tỉ lệ sẽ không nâng được / bùng số)."""
    ratio = rgb * (new_l / old_l.clamp(min=1e-4))
    add = rgb + (new_l - old_l)
    w = _smoothstep(0.0, 0.05, old_l)
    return (ratio * w + add * (1 - w)).clamp(0, 1)


def _smoothstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# 1. Basic (Lightroom Basic panel)
# ----------------------------------------------------------------------------
def basic(img, exposure=0.0, contrast=0.0, highlights=0.0, shadows=0.0, whites=0.0, blacks=0.0,
          temperature=0.0, tint=0.0, vibrance=0.0, saturation=0.0):
    x = img[..., :3].float().clamp(0, 1)
    lin = _srgb_to_lin(x)
    if exposure:
        lin = lin * (2.0 ** exposure)
    if temperature or tint:
        t, g = temperature / 100.0, tint / 100.0
        lin = lin * torch.tensor([1 + 0.25 * t, 1 - 0.20 * g, 1 - 0.25 * t], dtype=lin.dtype)
    x = _lin_to_srgb(lin)
    L = _lum(x)
    L2 = L
    if whites or blacks:
        wp = 1.0 - 0.25 * whites / 100.0
        bp = -0.15 * blacks / 100.0
        L2 = ((L2 - bp) / max(wp - bp, 1e-3)).clamp(0, 1)
    if highlights or shadows:
        sig = max(2.0, min(x.shape[1:3]) / 100.0)
        Lb = _blur_hw1(L2, sig)
        hm, sm = _smoothstep(0.45, 1.0, Lb), 1 - _smoothstep(0.0, 0.55, Lb)
        L2 = (L2 + 0.30 * highlights / 100.0 * hm * L2 + 0.30 * shadows / 100.0 * sm * (1 - L2)).clamp(0, 1)
    if contrast:
        k = contrast / 100.0
        s = 1.0 / (1.0 + torch.exp(-(L2 - 0.5) * 8.0))  # S-curve
        s = (s - s.new_tensor(1 / (1 + math.exp(4)))) / (1 / (1 + math.exp(-4)) - 1 / (1 + math.exp(4)))
        L2 = (L2 + k * (s - L2)) if k > 0 else (0.5 + (L2 - 0.5) * (1 + k)).clamp(0, 1)
    x = _set_lum(x, L2, L) if not torch.equal(L2, L) else x
    if vibrance or saturation:
        L = _lum(x)
        mx, mn = x.max(-1, keepdim=True).values, x.min(-1, keepdim=True).values
        sat = (mx - mn)
        f = 1 + saturation / 100.0 + vibrance / 100.0 * (1 - sat).clamp(0, 1)
        x = (L + (x - L) * f).clamp(0, 1)
    return x


# ----------------------------------------------------------------------------
# 2. Presence: clarity, dehaze, vignette, grain
# ----------------------------------------------------------------------------
def dark_channel(x, r):
    m = x.min(-1, keepdim=True).values.movedim(-1, 1)
    return (-F.max_pool2d(-m, 2 * r + 1, 1, r)).movedim(1, -1)


def dehaze(x, amount):
    """Dark Channel Prior (He et al. 2009) rút gọn. amount 0..100 khử mù, âm = thêm mù."""
    if amount == 0:
        return x
    H, W = x.shape[1:3]
    f = max(1.0, min(H, W) / 768)
    small = core.resize(x, max(8, round(W / f)), max(8, round(H / f)), "area")
    dc = dark_channel(small, 7)
    flat = dc.flatten()
    k = max(1, flat.numel() // 1000)
    idx = torch.topk(flat, k).indices
    A = small.reshape(-1, 3)[idx].mean(0).clamp(min=0.3)
    if amount < 0:
        return (x + (A - x) * (-amount / 100.0) * 0.6).clamp(0, 1)
    w = 0.95 * amount / 100.0
    t = 1 - w * dark_channel(small / A, 7)
    t = _blur_hw1(t, 4.0)
    t = core.resize(t.repeat(1, 1, 1, 3), W, H, "bilinear")[..., :1]
    return ((x - A) / t.clamp(min=0.15) + A).clamp(0, 1)


def presence(img, clarity=0.0, texture=0.0, dehaze_amount=0.0, vignette=0.0, grain=0.0, seed=0):
    x = img[..., :3].float().clamp(0, 1)
    x = dehaze(x, dehaze_amount)
    H, W = x.shape[1:3]
    if clarity or texture:
        L = _lum(x)
        L2 = L
        if clarity:  # tương phản cục bộ vùng trung (bán kính lớn)
            mid = 1 - (2 * L - 1).abs()
            L2 = L2 + clarity / 100.0 * 0.6 * (L - _blur_hw1(L, max(4.0, min(H, W) / 150.0))) * mid
        if texture:  # chi tiết nhỏ
            L2 = L2 + texture / 100.0 * 0.8 * (L - _blur_hw1(L, 1.5))
        x = _set_lum(x, L2.clamp(0, 1), L)
    if vignette:
        yy = torch.linspace(-1, 1, H)[:, None]
        xx = torch.linspace(-1, 1, W)[None, :]
        r = (xx ** 2 + yy ** 2).sqrt() / math.sqrt(2)
        m = _smoothstep(0.35, 1.0, r)[None, :, :, None]
        x = (x * (1 + vignette / 100.0 * m)).clamp(0, 1)
    if grain:
        g = torch.Generator().manual_seed(int(seed))
        n = torch.randn(1, max(1, H // 2), max(1, W // 2), 1, generator=g)
        n = F.interpolate(n.movedim(-1, 1), size=(H, W), mode="bilinear", align_corners=False).movedim(1, -1)
        L = _lum(x)
        x = (x + n * grain / 100.0 * 0.06 * (1 - (2 * L - 1).abs())).clamp(0, 1)
    return x


# ----------------------------------------------------------------------------
# 3. Curves (monotone cubic, như Photoshop Curves)
# ----------------------------------------------------------------------------
def parse_points(s):
    pts = []
    for tok in s.replace(";", " ").split():
        if "," in tok:
            a, b = tok.split(",")[:2]
            pts.append((float(a) / 255.0, float(b) / 255.0))
    if not pts:
        return None
    pts = sorted(dict(pts).items())
    if pts[0][0] > 0:
        pts.insert(0, (0.0, 0.0))
    if pts[-1][0] < 1:
        pts.append((1.0, 1.0))
    return pts


def curve_lut(pts, n=1024):
    """Fritsch–Carlson monotone cubic -> LUT n điểm."""
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    if len(xs) == 2:
        return torch.from_numpy(np.interp(np.linspace(0, 1, n), xs, ys)).float()
    d = np.diff(ys) / np.maximum(np.diff(xs), 1e-9)
    m = np.zeros_like(ys)
    m[1:-1] = np.where(d[:-1] * d[1:] > 0, (d[:-1] + d[1:]) / 2, 0)
    m[0], m[-1] = d[0], d[-1]
    for i in range(len(d)):
        if d[i] == 0:
            m[i] = m[i + 1] = 0
        else:
            a, b = m[i] / d[i], m[i + 1] / d[i]
            s = a * a + b * b
            if s > 9:
                t = 3 / math.sqrt(s)
                m[i], m[i + 1] = t * a * d[i], t * b * d[i]
    q = np.linspace(0, 1, n)
    k = np.clip(np.searchsorted(xs, q) - 1, 0, len(xs) - 2)
    h = xs[k + 1] - xs[k]
    t = (q - xs[k]) / np.maximum(h, 1e-9)
    y = ((2 * t ** 3 - 3 * t ** 2 + 1) * ys[k] + (t ** 3 - 2 * t ** 2 + t) * h * m[k]
         + (-2 * t ** 3 + 3 * t ** 2) * ys[k + 1] + (t ** 3 - t ** 2) * h * m[k + 1])
    return torch.from_numpy(np.clip(y, 0, 1)).float()


def apply_lut1d(x, lut):
    n = lut.numel()
    i = (x.clamp(0, 1) * (n - 1))
    i0 = i.floor().long().clamp(0, n - 1)
    i1 = (i0 + 1).clamp(max=n - 1)
    f = i - i0
    return lut[i0] * (1 - f) + lut[i1] * f


def curves(img, master, red, green, blue):
    x = img[..., :3].float().clamp(0, 1)
    for ch, s in ((0, red), (1, green), (2, blue)):
        p = parse_points(s)
        if p:
            x[..., ch] = apply_lut1d(x[..., ch], curve_lut(p))
    p = parse_points(master)
    if p:
        x = apply_lut1d(x, curve_lut(p))
    return x


# ----------------------------------------------------------------------------
# 4. HSL / Color mixer (8 dải màu như Lightroom)
# ----------------------------------------------------------------------------
HUES = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 240, "purple": 270, "magenta": 300}


def rgb_to_hsv(x):
    r, g, b = x.unbind(-1)
    mx, _ = x.max(-1)
    mn, _ = x.min(-1)
    d = mx - mn
    h = torch.zeros_like(mx)
    m = d > 1e-6
    rc = torch.where(m & (mx == r), ((g - b) / d.clamp(min=1e-6)) % 6, h)
    gc = torch.where(m & (mx == g) & (mx != r), (b - r) / d.clamp(min=1e-6) + 2, rc)
    h = torch.where(m & (mx == b) & (mx != r) & (mx != g), (r - g) / d.clamp(min=1e-6) + 4, gc) * 60.0
    s = torch.where(mx > 1e-6, d / mx.clamp(min=1e-6), torch.zeros_like(mx))
    return torch.stack([h % 360, s, mx], -1)


def hsv_to_rgb(hsv):
    h, s, v = hsv.unbind(-1)
    c = v * s
    hp = (h % 360) / 60.0
    xx = c * (1 - ((hp % 2) - 1).abs())
    z = torch.zeros_like(h)
    i = hp.floor().long() % 6
    rgb = torch.stack([
        torch.stack([c, xx, z, z, xx, c], -1).gather(-1, i.unsqueeze(-1)).squeeze(-1),
        torch.stack([xx, c, c, xx, z, z], -1).gather(-1, i.unsqueeze(-1)).squeeze(-1),
        torch.stack([z, z, xx, c, c, xx], -1).gather(-1, i.unsqueeze(-1)).squeeze(-1)], -1)
    return (rgb + (v - c).unsqueeze(-1)).clamp(0, 1)


def hue_weights(h):
    """Trọng số mềm cho từng dải màu (tổng ≈ 1), theo khoảng cách góc tới dải kề."""
    centers = torch.tensor(list(HUES.values()), dtype=h.dtype)
    ws = []
    for i, c in enumerate(centers):
        left = centers[i - 1] if i > 0 else centers[-1] - 360
        right = centers[i + 1] if i + 1 < len(centers) else centers[0] + 360
        dh = ((h - c + 180) % 360) - 180
        w = torch.where(dh < 0, 1 + dh / (c - left), 1 - dh / (right - c))
        ws.append(w.clamp(0, 1))
    return torch.stack(ws, -1)


def hsl_mix(img, adj):
    """adj: {color: (hue_shift, sat, lum)} -100..100."""
    x = img[..., :3].float().clamp(0, 1)
    if not any(any(v) for v in adj.values()):
        return x
    hsv = rgb_to_hsv(x)
    w = hue_weights(hsv[..., 0]) * hsv[..., 1:2].clamp(0, 1).sqrt()  # màu xám không bị ảnh hưởng
    names = list(HUES.keys())
    hs = torch.tensor([adj[n][0] for n in names], dtype=x.dtype) / 100.0 * 30.0
    ss = torch.tensor([adj[n][1] for n in names], dtype=x.dtype) / 100.0
    ls = torch.tensor([adj[n][2] for n in names], dtype=x.dtype) / 100.0
    L = _lum(x)
    hsv[..., 0] = hsv[..., 0] + (w * hs).sum(-1)
    hsv[..., 1] = (hsv[..., 1] * (1 + (w * ss).sum(-1))).clamp(0, 1)
    out = hsv_to_rgb(hsv)
    Lt = (L * (1 + 0.5 * (w * ls).sum(-1, keepdim=True))).clamp(0, 1)
    return _set_lum(out, Lt, _lum(out))


# ----------------------------------------------------------------------------
# 5. Color Balance (Photoshop)
# ----------------------------------------------------------------------------
def color_balance(img, shadows, midtones, highlights, preserve_luminosity=True):
    """mỗi nhóm: (cyan-red, magenta-green, yellow-blue) -100..100."""
    x = img[..., :3].float().clamp(0, 1)
    L = _lum(x)
    ws = 1 - _smoothstep(0.0, 0.5, L)
    wh = _smoothstep(0.5, 1.0, L)
    wm = (1 - ws - wh).clamp(0, 1)
    out = x.clone()
    for w, (cr, mg, yb) in ((ws, shadows), (wm, midtones), (wh, highlights)):
        out = out + w * torch.tensor([cr, mg, yb], dtype=x.dtype) / 100.0 * 0.15
    out = out.clamp(0, 1)
    return _set_lum(out, L, _lum(out)) if preserve_luminosity else out


# ----------------------------------------------------------------------------
# 6. Selective Color (Photoshop, xấp xỉ)
# ----------------------------------------------------------------------------
SEL_TARGETS = ["reds", "yellows", "greens", "cyans", "blues", "magentas", "whites", "neutrals", "blacks"]


def selective_weight(x, target):
    r, g, b = x.unbind(-1)
    mx, _ = x.max(-1)
    mn, _ = x.min(-1)
    mid = x.sum(-1) - mx - mn
    if target == "reds":
        w = torch.where(r >= mx, mx - mid, 0 * r)
    elif target == "greens":
        w = torch.where(g >= mx, mx - mid, 0 * r)
    elif target == "blues":
        w = torch.where(b >= mx, mx - mid, 0 * r)
    elif target == "yellows":
        w = torch.where(b <= mn, mid - mn, 0 * r)
    elif target == "cyans":
        w = torch.where(r <= mn, mid - mn, 0 * r)
    elif target == "magentas":
        w = torch.where(g <= mn, mid - mn, 0 * r)
    elif target == "whites":
        w = ((mn - 0.5) * 2).clamp(0, 1)
    elif target == "blacks":
        w = ((0.5 - mx) * 2).clamp(0, 1)
    else:  # neutrals
        w = ((1 - (mx - 0.5).abs() * 2) * (1 - (mn - 0.5).abs() * 2)).clamp(0, 1)
    return w.unsqueeze(-1)


def selective_color(img, target, cyan, magenta, yellow, black, relative=True):
    x = img[..., :3].float().clamp(0, 1)
    w = selective_weight(x, target)
    cmy = torch.tensor([cyan, magenta, yellow], dtype=x.dtype) / 100.0
    k = black / 100.0
    # tăng cyan = giảm red ... ; black = giảm cả 3
    delta = -(cmy + k)
    scale = (1 - x) if relative else torch.ones_like(x)  # relative: theo lượng C/M/Y đang có (như Photoshop)
    return (x + w * delta * scale).clamp(0, 1)


# ----------------------------------------------------------------------------
# 7. LUT .cube
# ----------------------------------------------------------------------------
def load_cube(path):
    size, rows = None, []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            if t.upper().startswith("LUT_3D_SIZE"):
                size = int(t.split()[-1])
                continue
            parts = t.split()
            if len(parts) == 3:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    pass
    if not size or len(rows) < size ** 3:
        raise ValueError(f"File .cube không hợp lệ (3D): {path}")
    lut = torch.tensor(rows[:size ** 3], dtype=torch.float32).reshape(size, size, size, 3)  # [b][g][r]
    return lut


def apply_cube(img, lut, strength=1.0):
    x = img[..., :3].float().clamp(0, 1)
    vol = lut.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,B,G,R] -> D=b, H=g, W=r
    B, H, W, _ = x.shape
    grid = (x * 2 - 1).reshape(B, 1, H, W, 3)  # (x=r, y=g, z=b)
    out = F.grid_sample(vol.expand(B, -1, -1, -1, -1), grid, mode="bilinear", align_corners=True)
    out = out.reshape(B, 3, H, W).movedim(1, -1)
    return (x + (out - x) * strength).clamp(0, 1)


def list_luts():
    os.makedirs(LUT_DIR, exist_ok=True)
    return ["none"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(LUT_DIR, "*.cube")))


# ----------------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------------
def _s(default=0.0, lo=-100.0, hi=100.0, step=1.0):
    return ("FLOAT", {"default": default, "min": lo, "max": hi, "step": step})


class DB9U_Basic:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "exposure": _s(0.0, -5.0, 5.0, 0.05), "contrast": _s(),
                             "highlights": _s(), "shadows": _s(), "whites": _s(), "blacks": _s(),
                             "temperature": _s(), "tint": _s(), "vibrance": _s(), "saturation": _s()}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, **kw):
        return (basic(image, **kw),)


class DB9U_Presence:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "clarity": _s(), "texture": _s(), "dehaze": _s(),
                             "vignette": _s(), "grain": _s(0.0, 0.0, 100.0), "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, clarity, texture, dehaze, vignette, grain, seed):
        return (presence(image, clarity, texture, dehaze, vignette, grain, seed),)


class DB9U_Curves:
    @classmethod
    def INPUT_TYPES(cls):
        tip = {"tooltip": "Điểm x,y (0-255) cách nhau khoảng trắng. Vd S-curve: 0,0 64,52 192,206 255,255. Trống = không đổi"}
        return {"required": {"image": ("IMAGE",),
                             "master": ("STRING", {"default": "", **tip}), "red": ("STRING", {"default": ""}),
                             "green": ("STRING", {"default": ""}), "blue": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, master, red, green, blue):
        return (curves(image, master, red, green, blue),)


class DB9U_HSL:
    @classmethod
    def INPUT_TYPES(cls):
        req = {"image": ("IMAGE",)}
        for n in HUES:
            req[f"{n}_hue"] = _s()
            req[f"{n}_sat"] = _s()
            req[f"{n}_lum"] = _s()
        return {"required": req}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, **kw):
        adj = {n: (kw[f"{n}_hue"], kw[f"{n}_sat"], kw[f"{n}_lum"]) for n in HUES}
        return (hsl_mix(image, adj),)


class DB9U_ColorBalance:
    @classmethod
    def INPUT_TYPES(cls):
        req = {"image": ("IMAGE",)}
        for tone in ("shadows", "midtones", "highlights"):
            for ax in ("cyan_red", "magenta_green", "yellow_blue"):
                req[f"{tone}_{ax}"] = _s()
        req["preserve_luminosity"] = ("BOOLEAN", {"default": True})
        return {"required": req}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, preserve_luminosity, **kw):
        g = lambda t: (kw[f"{t}_cyan_red"], kw[f"{t}_magenta_green"], kw[f"{t}_yellow_blue"])
        return (color_balance(image, g("shadows"), g("midtones"), g("highlights"), preserve_luminosity),)


class DB9U_SelectiveColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "target": (SEL_TARGETS,), "cyan": _s(), "magenta": _s(),
                             "yellow": _s(), "black": _s(), "method": (["relative", "absolute"],)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, target, cyan, magenta, yellow, black, method):
        return (selective_color(image, target, cyan, magenta, yellow, black, method == "relative"),)


class DB9U_LUT:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "lut": (list_luts(), {"tooltip": "File .cube trong DB9_Ultimate/luts"}),
                             "custom_path": ("STRING", {"default": "", "tooltip": "Đường dẫn .cube bất kỳ (ưu tiên hơn danh sách)"}),
                             "strength": _s(1.0, 0.0, 1.0, 0.05)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, lut, custom_path, strength):
        path = custom_path.strip().strip('"') or (os.path.join(LUT_DIR, lut) if lut != "none" else "")
        if not path:
            return (image,)
        return (apply_cube(image, load_cube(path), strength),)


GRADE_NODES = {
    "DB9U_Basic": (DB9U_Basic, "DB9U Basic (Exposure/Contrast/HL/Shadow/WB/Vibrance)"),
    "DB9U_Presence": (DB9U_Presence, "DB9U Presence (Clarity/Texture/Dehaze/Vignette/Grain)"),
    "DB9U_Curves": (DB9U_Curves, "DB9U Curves"),
    "DB9U_HSL": (DB9U_HSL, "DB9U HSL / Color Mixer"),
    "DB9U_ColorBalance": (DB9U_ColorBalance, "DB9U Color Balance"),
    "DB9U_SelectiveColor": (DB9U_SelectiveColor, "DB9U Selective Color"),
    "DB9U_LUT": (DB9U_LUT, "DB9U LUT (.cube)"),
}
