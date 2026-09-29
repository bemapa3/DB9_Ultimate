"""DB9 Upscale Enhance - core math (thuần torch, không phụ thuộc ComfyUI).

Quy ước ảnh: tensor float [H, W, C] hoặc [B, H, W, C], giá trị 0..1 (chuẩn IMAGE của ComfyUI).
"""
import math
import torch
import torch.nn.functional as F

ALIGN = 32  # tile, vị trí & canvas bội số 32 (an toàn cho VAE /8 hoặc /16 + patch 2x2 của DiT)


# ----------------------------------------------------------------------------
# Tiện ích chung
# ----------------------------------------------------------------------------
def to_bchw(img):
    return img.movedim(-1, -3) if img.ndim == 4 else img.movedim(-1, 0).unsqueeze(0)


def to_bhwc(t):
    return t.movedim(-3, -1)


def resize(img_bhwc, w, h, mode="bicubic"):
    t = to_bchw(img_bhwc)
    if mode == "area":
        t = F.interpolate(t, size=(h, w), mode="area")
    else:
        t = F.interpolate(t, size=(h, w), mode=mode, align_corners=False, antialias=(mode in ("bilinear", "bicubic")))
    return to_bhwc(t).clamp(0, 1)


def gaussian_kernel1d(sigma, device, dtype):
    r = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    return k / k.sum()


def _blur_bchw_direct(t, sigma):
    k = gaussian_kernel1d(sigma, t.device, t.dtype)
    r = (k.numel() - 1) // 2
    c = t.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    mode = "reflect" if (t.shape[-1] > r and t.shape[-2] > r) else "replicate"
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode=mode), kx, groups=c)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode=mode), ky, groups=c)
    return t


def gaussian_blur(t, sigma):
    """Blur BCHW. sigma lớn -> blur ở độ phân giải thấp rồi phóng lên (nhanh, gần đúng)."""
    if sigma <= 0:
        return t
    if sigma <= 6:
        return _blur_bchw_direct(t, sigma)
    f = int(sigma // 3)
    h, w = t.shape[-2:]
    small = F.interpolate(t, size=(max(1, h // f), max(1, w // f)), mode="area")
    small = _blur_bchw_direct(small, sigma / f)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------
# Màu: sRGB <-> CIE Lab (D65)
# ----------------------------------------------------------------------------
_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """rgb [..., 3] 0..1 -> Lab [..., 3]."""
    rgb = rgb.clamp(0, 1)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.to(rgb).T
    xyz = xyz / _WHITE.to(rgb)
    d = 6 / 29
    f = torch.where(xyz > d ** 3, xyz.clamp(min=1e-12) ** (1 / 3), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], -1)


def lab_to_srgb(lab):
    L, a, b = lab.unbind(-1)
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    f = torch.stack([fx, fy, fz], -1)
    d = 6 / 29
    xyz = torch.where(f > d, f ** 3, 3 * d * d * (f - 4 / 29)) * _WHITE.to(lab)
    lin = xyz @ torch.linalg.inv(_M_RGB2XYZ).to(lab).T
    lin = lin.clamp(0, 1)
    rgb = torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin.clamp(min=1e-12) ** (1 / 2.4) - 0.055)
    return rgb.clamp(0, 1)


def delta_e76(a_rgb, b_rgb):
    return (srgb_to_lab(a_rgb) - srgb_to_lab(b_rgb)).norm(dim=-1)


# ----------------------------------------------------------------------------
# Khoá màu / khoá form
# ----------------------------------------------------------------------------
def color_lock(result, reference, mode="detail_transfer", strength=1.0, sigma=6.0):
    """result, reference: [B,H,W,3] cùng kích thước.
    - lab_stats: khớp mean/std từng kênh Lab (giữ tone tổng thể).
    - detail_transfer: tần số thấp (màu, khối, form) lấy từ reference, tần số cao (chi tiết mới) lấy từ result.
      => màu & mảng lớn gần như trùng tuyệt đối với ảnh gốc.
    - both: lab_stats rồi detail_transfer.
    """
    if mode == "none" or strength <= 0:
        return result
    result = result[..., :3]
    reference = reference[..., :3]
    out = result
    if mode in ("lab_stats", "both"):
        lr, lf = srgb_to_lab(out), srgb_to_lab(reference)
        dims = (1, 2)
        mr, sr = lr.mean(dims, keepdim=True), lr.std(dims, keepdim=True).clamp(min=1e-4)
        mf, sf = lf.mean(dims, keepdim=True), lf.std(dims, keepdim=True)
        out = lab_to_srgb((lr - mr) / sr * sf + mf)
    if mode in ("detail_transfer", "both"):
        r, f = to_bchw(out), to_bchw(reference)
        low_f = gaussian_blur(f, sigma)
        high_r = r - gaussian_blur(r, sigma)
        out = to_bhwc((low_f + high_r).clamp(0, 1))
    return (result + (out - result) * strength).clamp(0, 1)


# ----------------------------------------------------------------------------
# Chia ô / ghép ô
# ----------------------------------------------------------------------------
def _axis_even(size, max_tile, overlap):
    """Chia 1 trục thành n ô BẰNG NHAU, mỗi ô (gồm cả phần giao) ≤ max_tile, vị trí bội số ALIGN.
    Không ô nào bị co/giãn; phần giao thực tế ≥ overlap và chia đều."""
    if size <= max_tile:
        return size, [0]
    n = math.ceil((size - overlap) / (max_tile - overlap))
    while True:
        tile = int(math.ceil((size + (n - 1) * overlap) / n / ALIGN) * ALIGN)
        if tile <= max_tile:
            break
        n += 1
    pos = [int(round(i * (size - tile) / (n - 1) / ALIGN)) * ALIGN for i in range(n)]
    return tile, pos


def plan_tiles(width, height, max_tile, overlap):
    """Kế hoạch chia ô: canvas pad lên bội số ALIGN (pad reflect, cắt bỏ khi ghép),
    ô hình chữ nhật đều nhau theo tỉ lệ ảnh, mỗi ô ≤ max_tile px (đã gồm vùng giao)."""
    cw = int(math.ceil(width / ALIGN) * ALIGN)
    ch = int(math.ceil(height / ALIGN) * ALIGN)
    mt = max(ALIGN * 4, int(max_tile // ALIGN * ALIGN))
    ov = int(max(0, min(overlap, mt // 3)) // ALIGN * ALIGN)
    tw, xs = _axis_even(cw, mt, ov)
    th, ys = _axis_even(ch, mt, ov)
    grid = [(ix, iy) for iy in range(len(ys)) for ix in range(len(xs))]
    return {"width": width, "height": height, "canvas_w": cw, "canvas_h": ch,
            "tile_w": tw, "tile_h": th, "overlap": ov, "xs": xs, "ys": ys, "grid": grid,
            "coords": [(xs[ix], ys[iy]) for (ix, iy) in grid], "cols": len(xs), "rows": len(ys)}


def scale_plan(plan, k):
    """Phóng/thu kế hoạch theo hệ số k (dùng cho tile đã upscale hoặc không gian latent)."""
    if abs(k - 1.0) < 1e-9:
        return plan
    sp = dict(plan)
    for key in ("width", "height", "canvas_w", "canvas_h", "tile_w", "tile_h", "overlap"):
        sp[key] = int(round(plan[key] * k))
    sp["xs"] = [int(round(v * k)) for v in plan["xs"]]
    sp["ys"] = [int(round(v * k)) for v in plan["ys"]]
    sp["coords"] = [(sp["xs"][ix], sp["ys"][iy]) for (ix, iy) in plan["grid"]]
    return sp


def pad_canvas(img, plan):
    """img [B,H,W,C] -> pad reflect/replicate tới canvas."""
    ph = plan["canvas_h"] - img.shape[1]
    pw = plan["canvas_w"] - img.shape[2]
    if ph == 0 and pw == 0:
        return img
    t = to_bchw(img)
    mode = "reflect" if (ph < t.shape[-2] and pw < t.shape[-1]) else "replicate"
    return to_bhwc(F.pad(t, (0, pw, 0, ph), mode=mode))


def split_tiles(img, plan):
    """img [1,H,W,C] (kích thước gốc) -> tensor [N, th, tw, C]."""
    c = pad_canvas(img, plan)
    tw, th = plan["tile_w"], plan["tile_h"]
    return torch.cat([c[:, y:y + th, x:x + tw, :] for (x, y) in plan["coords"]], 0)


def _ramp(n, left_len, right_len, device):
    """Trọng số 1D: dốc tuyến tính trên đúng chiều dài vùng giao thực tế -> 2 ô kề cộng lại = 1."""
    w = torch.ones(n, device=device)
    if left_len > 0:
        w[:left_len] = (torch.arange(left_len, device=device, dtype=torch.float32) + 0.5) / left_len
    if right_len > 0:
        r = ((torch.arange(right_len, device=device, dtype=torch.float32) + 0.5) / right_len).flip(0)
        w[-right_len:] = torch.minimum(w[-right_len:], r)
    return w


def _axis_ramp(pos, i, tile, device):
    left = pos[i - 1] + tile - pos[i] if i > 0 else 0
    right = pos[i] + tile - pos[i + 1] if i + 1 < len(pos) else 0
    return _ramp(tile, max(0, left), max(0, right), device)


def tile_weight(plan, ix, iy, device):
    wx = _axis_ramp(plan["xs"], ix, plan["tile_w"], device)
    wy = _axis_ramp(plan["ys"], iy, plan["tile_h"], device)
    return wy[:, None] * wx[None, :]  # [th, tw]


def merge_tiles(tiles, plan):
    """tiles [N, th', tw', C] (có thể đã upscale thêm k lần) -> [1, H*k, W*k, C]."""
    n, th2, tw2, c = tiles.shape
    k = th2 / plan["tile_h"]
    if abs(k - tw2 / plan["tile_w"]) > 1e-3:
        raise ValueError("Tile bị đổi tỉ lệ khác nhau theo 2 chiều")
    plan = scale_plan(plan, k)
    dev = tiles.device
    acc = torch.zeros(1, plan["canvas_h"], plan["canvas_w"], c, device=dev)
    wsum = torch.zeros(1, plan["canvas_h"], plan["canvas_w"], 1, device=dev)
    tw, th = plan["tile_w"], plan["tile_h"]
    for i, (ix, iy) in enumerate(plan["grid"]):
        x, y = plan["xs"][ix], plan["ys"][iy]
        w = tile_weight(plan, ix, iy, dev)[..., None]
        acc[0, y:y + th, x:x + tw] += tiles[i, :th, :tw].float() * w
        wsum[0, y:y + th, x:x + tw] += w
    out = acc / wsum.clamp(min=1e-8)
    return out[:, :plan["height"], :plan["width"], :].clamp(0, 1)


# ----------------------------------------------------------------------------
# QAQC metrics
# ----------------------------------------------------------------------------
def luminance(img_bhwc):
    return (img_bhwc[..., 0] * 0.299 + img_bhwc[..., 1] * 0.587 + img_bhwc[..., 2] * 0.114).unsqueeze(1)


def ssim_map(a, b):
    """a, b: [B,1,H,W]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = _blur_bchw_direct(a, 1.5), _blur_bchw_direct(b, 1.5)
    saa = _blur_bchw_direct(a * a, 1.5) - mu_a ** 2
    sbb = _blur_bchw_direct(b * b, 1.5) - mu_b ** 2
    sab = _blur_bchw_direct(a * b, 1.5) - mu_a * mu_b
    return ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))


def phase_shift(a, b):
    """Ước lượng lệch tịnh tiến (dx, dy) của b so với a, đơn vị pixel, độ chính xác ~0.1px. a,b [1,1,H,W]."""
    h, w = a.shape[-2:]
    win = torch.hann_window(h, device=a.device)[:, None] * torch.hann_window(w, device=a.device)[None, :]
    fa = torch.fft.fft2((a[0, 0] - a.mean()) * win)
    fb = torch.fft.fft2((b[0, 0] - b.mean()) * win)
    r = fa.conj() * fb
    r = r / r.abs().clamp(min=1e-12)
    corr = torch.fft.ifft2(r).real
    idx = int(torch.argmax(corr))
    py, px = divmod(idx, w)

    def sub(c_m, c_0, c_p):
        d = c_m - 2 * c_0 + c_p
        return 0.0 if abs(float(d)) < 1e-12 else float(0.5 * (c_m - c_p) / d)

    dx = px + sub(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    dy = py + sub(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    return dx, dy


def edge_map(lum):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=lum.dtype, device=lum.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    p = F.pad(lum, (1, 1, 1, 1), mode="replicate")
    g = torch.sqrt(F.conv2d(p, kx) ** 2 + F.conv2d(p, ky) ** 2)
    thr = g.mean() + g.std()
    return g > thr


def edge_fscore(ea, eb, tol=1):
    k = 2 * tol + 1
    da = F.max_pool2d(ea.float(), k, 1, tol) > 0
    db = F.max_pool2d(eb.float(), k, 1, tol) > 0
    prec = (eb & da).sum().float() / eb.sum().clamp(min=1)
    rec = (ea & db).sum().float() / ea.sum().clamp(min=1)
    return float(2 * prec * rec / (prec + rec).clamp(min=1e-8)), float(prec), float(rec)


def heat_colormap(x):
    """x [H,W] 0..1 -> [H,W,3] đen->đỏ->vàng->trắng."""
    r = (x * 3).clamp(0, 1)
    g = (x * 3 - 1).clamp(0, 1)
    b = (x * 3 - 2).clamp(0, 1)
    return torch.stack([r, g, b], -1)


def qa_compare(original, result, compare_blur=1.0, de_warn=3.0):
    """So sánh 1 ảnh (không batch) ở độ phân giải ảnh gốc.
    original, result: [1,H,W,3] / [1,H2,W2,3]. Trả về dict số liệu + heatmap [1,H,W,3]."""
    h, w = original.shape[1:3]
    o = original[..., :3].float()
    r = resize(result[..., :3].float(), w, h, "area") if result.shape[1:3] != (h, w) else result[..., :3].float()
    if compare_blur > 0:  # bỏ qua chi tiết mới sinh (tần số cao), chỉ so màu/khối
        ob = to_bhwc(gaussian_blur(to_bchw(o), compare_blur))
        rb = to_bhwc(gaussian_blur(to_bchw(r), compare_blur))
    else:
        ob, rb = o, r
    de = delta_e76(ob, rb)[0]  # [H,W]
    lo, lr = luminance(o), luminance(r)
    ssim = float(ssim_map(lo, lr).mean())
    dx, dy = phase_shift(lo, lr)
    f1, prec, rec = edge_fscore(edge_map(lo), edge_map(lr))
    lab_o, lab_r = srgb_to_lab(ob), srgb_to_lab(rb)
    mean_shift = (lab_r.mean((0, 1, 2)) - lab_o.mean((0, 1, 2))).tolist()
    heat = heat_colormap((de / max(de_warn * 2, 1e-6)).clamp(0, 1))[None]
    loc_max, loc_p95, _ = local_shifts(lo, lr)
    return {
        "local_shift_max": loc_max, "local_shift_p95": loc_p95,
        "deltaE_mean": float(de.mean()),
        "deltaE_p95": float(torch.quantile(de.flatten()[:: max(1, de.numel() // 1_000_000)], 0.95)),
        "deltaE_max": float(de.max()),
        "lab_mean_shift": mean_shift,
        "ssim": ssim,
        "shift_px": (dx, dy),
        "edge_f1": f1, "edge_precision": prec, "edge_recall": rec,
        "heatmap": heat,
    }


def tile_color_error(base_tile, out_tile, blur=2.0):
    """ΔE trung bình giữa tile gốc & tile sau sampler (đã blur để bỏ chi tiết mới)."""
    a = to_bhwc(gaussian_blur(to_bchw(base_tile[..., :3].float()), blur))
    b = to_bhwc(gaussian_blur(to_bchw(out_tile[..., :3].float()), blur))
    return float(delta_e76(a, b).mean())


# ----------------------------------------------------------------------------
# Lệch cục bộ (méo / rung đường thẳng), căn lại ô, khoá viền
# ----------------------------------------------------------------------------
def local_shifts(a, b, block=64, min_std=0.02):
    """Phase correlation theo từng khối block x block. a,b [1,1,H,W].
    Trả về (max, p95, map [nh,nw]) độ lệch (px) — chỉ tính khối có texture."""
    h, w = a.shape[-2:]
    nh, nw = h // block, w // block
    if nh == 0 or nw == 0:
        return 0.0, 0.0, torch.zeros(1, 1)

    def blocks(t):
        t = t[0, 0, :nh * block, :nw * block].reshape(nh, block, nw, block).permute(0, 2, 1, 3)
        return t.reshape(-1, block, block)

    A, B = blocks(a), blocks(b)
    valid = (A.std((1, 2)) > min_std) & (B.std((1, 2)) > min_std)
    win = torch.hann_window(block, device=a.device)
    win = win[:, None] * win[None, :]
    fa = torch.fft.fft2((A - A.mean((1, 2), keepdim=True)) * win)
    fb = torch.fft.fft2((B - B.mean((1, 2), keepdim=True)) * win)
    r = fa.conj() * fb
    corr = torch.fft.ifft2(r / r.abs().clamp(min=1e-12)).real  # [N,b,b]
    idx = corr.flatten(1).argmax(1)
    py, px = idx // block, idx % block
    n = torch.arange(corr.shape[0], device=a.device)

    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return torch.where(d.abs() > 1e-12, 0.5 * (cm - cp) / d, torch.zeros_like(d)).clamp(-0.5, 0.5)

    c0 = corr[n, py, px]
    dx = px.float() + sub(corr[n, py, (px - 1) % block], c0, corr[n, py, (px + 1) % block])
    dy = py.float() + sub(corr[n, (py - 1) % block, px], c0, corr[n, (py + 1) % block, px])
    dx = torch.where(dx > block / 2, dx - block, dx)
    dy = torch.where(dy > block / 2, dy - block, dy)
    mag = torch.sqrt(dx ** 2 + dy ** 2) * valid.float()
    m = mag[valid] if valid.any() else torch.zeros(1)
    return float(m.max()), float(torch.quantile(m, 0.95)), mag.reshape(nh, nw)


def shift_image(img, dx, dy):
    """Dịch ảnh [B,H,W,C] đi (dx, dy) px, sub-pixel, bicubic, biên kéo dài."""
    t = to_bchw(img)
    b, c, h, w = t.shape
    ys = torch.linspace(-1, 1, h, device=t.device)
    xs = torch.linspace(-1, 1, w, device=t.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx - dx * 2 / max(w - 1, 1)
    gy = gy - dy * 2 / max(h - 1, 1)
    grid = torch.stack([gx, gy], -1)[None].expand(b, -1, -1, -1)
    out = F.grid_sample(t, grid, mode="bicubic", padding_mode="border", align_corners=True)
    return to_bhwc(out).clamp(0, 1)


def align_to_ref(img, ref, min_px=0.15, max_px=None):
    """Nếu ô sau sampler bị trôi (lệch tịnh tiến) so với ô gốc -> kéo về đúng vị trí."""
    dx, dy = phase_shift(luminance(ref[..., :3].float()), luminance(img[..., :3].float()))
    max_px = max_px or img.shape[2] / 16
    mag = math.hypot(dx, dy)
    if mag < min_px or mag > max_px:
        return img, (dx, dy), False
    return shift_image(img, -dx, -dy), (dx, dy), True


def structure_edge_mask(reference, radius=3, work_short_side=1024, top_fraction=0.10,
                        min_coherence=0.45, busy_quantile=0.6, max_busy=0.45):
    """Mask CẠNH KIẾN TRÚC [B,1,H,W] 0..1: chỉ cạnh lớn, dài, thẳng (mép tường, khung cửa, mái, lan can).
    Bỏ qua texture nhỏ (lá cây, cỏ, vân gỗ/đá) bằng 3 lớp lọc:
      1. Tính ở độ phân giải thấp (cạnh ngắn ~1024px) -> texture mịn tự biến mất.
      2. Coherence của structure tensor: cạnh thẳng có hướng thống nhất (cao), lá cây hướng lộn xộn (thấp).
      3. Độ 'bận' texture: vùng mà phần lớn pixel đều có gradient (tán lá, cỏ, sỏi) bị loại."""
    lum = luminance(reference[..., :3].float())
    H, W = lum.shape[-2:]
    f = max(1.0, min(H, W) / work_short_side)
    small = F.interpolate(lum, size=(max(8, round(H / f)), max(8, round(W / f))), mode="area") if f > 1 else lum
    small = _blur_bchw_direct(small, 1.0)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=small.dtype, device=small.device).view(1, 1, 3, 3) / 8
    p = F.pad(small, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(p, kx), F.conv2d(p, kx.transpose(-1, -2))
    mag = torch.sqrt(gx * gx + gy * gy)
    jxx, jxy, jyy = (_blur_bchw_direct(t, 2.0) for t in (gx * gx, gx * gy, gy * gy))
    tmp = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = (tmp / (jxx + jyy + 1e-8)) ** 2  # ((l1-l2)/(l1+l2))^2
    flat = mag.flatten()
    thr = torch.quantile(flat[:: max(1, flat.numel() // 500_000)], 1 - top_fraction)
    strong = (mag > thr).float()
    thr_busy = torch.quantile(flat[:: max(1, flat.numel() // 500_000)], busy_quantile)
    busy = _blur_bchw_direct((mag > thr_busy).float(), 4.0)
    m = strong * (coh > min_coherence).float() * (busy < max_busy).float()
    r = max(1, round(radius / f))
    m = F.max_pool2d(m, 2 * r + 1, 1, r)
    m = _blur_bchw_direct(m, max(1.0, r / 2)).clamp(0, 1)
    if f > 1:
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    return m


def edge_guard(result, reference, strength=1.0, sigma_edge=1.5, radius=3):
    """Khoá viền KIẾN TRÚC: quanh cạnh lớn/dài/thẳng của ảnh gốc, vị trí & hình dạng cạnh lấy từ gốc,
    chỉ nhận chi tiết mịn từ ảnh mới -> đường thẳng không cong/rung, viền không lệch.
    Lá cây, cỏ, texture vật liệu KHÔNG bị khoá -> giữ chi tiết Qwen tạo ra.
    result, reference [B,H,W,3] cùng kích thước."""
    if strength <= 0:
        return result
    mask = structure_edge_mask(reference, radius) * strength
    locked = color_lock(result, reference, "detail_transfer", 1.0, sigma_edge)
    m = mask.movedim(1, -1)
    return (result * (1 - m) + locked * m).clamp(0, 1)


def pass_count(total_scale, max_per_pass):
    """Số lần chạy để mỗi lần phóng ≤ max_per_pass (tỉ lệ mỗi lần bằng nhau)."""
    if total_scale <= max_per_pass * 1.0001:
        return 1
    return int(math.ceil(math.log(total_scale) / math.log(max_per_pass) - 1e-9))


def texture_mask(reference, work_short_side=1024, busy_quantile=0.6, lo=0.30, hi=0.55):
    """Mask VÙNG TEXTURE DÀY [B,1,H,W] 0..1 (tán lá, cỏ, bụi cây, sỏi...) — nơi hầu hết pixel đều có gradient.
    Mảng phẳng (tường, kính, trời) và cạnh kiến trúc đơn lẻ ~0."""
    lum = luminance(reference[..., :3].float())
    H, W = lum.shape[-2:]
    f = max(1.0, min(H, W) / work_short_side)
    small = F.interpolate(lum, size=(max(8, round(H / f)), max(8, round(W / f))), mode="area") if f > 1 else lum
    small = _blur_bchw_direct(small, 1.0)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=small.dtype, device=small.device).view(1, 1, 3, 3) / 8
    p = F.pad(small, (1, 1, 1, 1), mode="replicate")
    mag = torch.sqrt(F.conv2d(p, kx) ** 2 + F.conv2d(p, kx.transpose(-1, -2)) ** 2)
    flat = mag.flatten()
    thr = torch.quantile(flat[:: max(1, flat.numel() // 500_000)], busy_quantile)
    busy = _blur_bchw_direct((mag > thr).float(), 4.0)
    m = ((busy - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
    m = _blur_bchw_direct(m, 3.0).clamp(0, 1)
    if f > 1:
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    return m


# ----------------------------------------------------------------------------
# Ảnh control cho ControlNet (theo từng ô)
# ----------------------------------------------------------------------------
CONTROL_MODES = ["auto", "tile", "canny", "lineart", "grayscale", "external"]


def control_map(tile, mode):
    """tile [1,H,W,3] -> ảnh control [1,H,W,3] 0..1.
    tile: ảnh mờ nhẹ (ControlNet tile/upscaler) · grayscale · canny (cv2 auto-threshold, fallback Sobel)
    lineart: cạnh mềm Sobel chuẩn hoá (đen nền, nét trắng)."""
    x = tile[..., :3].float()
    if mode == "tile":
        return to_bhwc(gaussian_blur(to_bchw(x), 1.5)).clamp(0, 1)
    if mode == "grayscale":
        return luminance(x).movedim(1, -1).repeat(1, 1, 1, 3).clamp(0, 1)
    if mode == "lineart":
        lum = _blur_bchw_direct(luminance(x), 1.0)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=lum.dtype, device=lum.device).view(1, 1, 3, 3)
        p = F.pad(lum, (1, 1, 1, 1), mode="replicate")
        g = torch.sqrt(F.conv2d(p, kx) ** 2 + F.conv2d(p, kx.transpose(-1, -2)) ** 2)
        g = (g / torch.quantile(g.flatten()[:: max(1, g.numel() // 200_000)], 0.98).clamp(min=1e-4)).clamp(0, 1)
        return g.movedim(1, -1).repeat(1, 1, 1, 3)
    # canny
    try:
        import cv2
        import numpy as np
        g = (luminance(x)[0, 0].cpu().numpy() * 255).astype(np.uint8)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        med = float(np.median(g))
        e = cv2.Canny(g, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
        m = torch.from_numpy(e).float().div(255)[None, :, :, None]
    except ImportError:
        m = edge_map(luminance(x)).float().movedim(1, -1)
    return m.repeat(1, 1, 1, 3)
