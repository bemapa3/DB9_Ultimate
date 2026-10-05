"""Region Fix — sửa lại 1 (hoặc nhiều) vùng chưa ưng trên ảnh đã upscale rồi ghép lại CHÍNH XÁC từng pixel.

Cách dùng tiện nhất: Load Image (ảnh kết quả) -> chuột phải 'Open in MaskEditor' tô vùng chưa ok -> nối IMAGE + MASK
vào 'DB9U Region Fix'. Mask có thể tô trên ảnh thu nhỏ (JPG xem trước) — node tự phóng mask về đúng cỡ ảnh.
Mỗi mảng tô rời nhau = 1 vùng chạy riêng. Hoặc gõ toạ độ: "x,y,w,h; x,y,w,h".

Nguyên tắc ghép chính xác: toạ độ crop lưu trong REGION_INFO (số nguyên, bội số 32), ghép bằng đúng toạ độ đó,
mép hoà bằng mask feather NẰM TRONG vùng crop (phần nới thêm chỉ làm ngữ cảnh cho model, không dán).
"""
import math

import torch

from . import core

ALIGN = core.ALIGN


# ---------------------------------------------------------------------------- chọn vùng
def parse_boxes(spec):
    """'x,y,w,h; x,y,w,h' -> [(x,y,w,h)]."""
    out = []
    for part in (spec or "").replace("\n", ";").split(";"):
        nums = [p for p in part.replace(" ", "").split(",") if p]
        if len(nums) == 4:
            x, y, w, h = (int(float(v)) for v in nums)
            if w > 0 and h > 0:
                out.append((x, y, w, h))
    return out


def fit_mask(mask, H, W):
    """MASK ComfyUI [B,H',W'] hoặc [H',W'] -> [1,H,W] float 0..1 (phóng nếu tô trên ảnh thu nhỏ)."""
    m = mask.float()
    if m.ndim == 2:
        m = m[None]
    m = m[:1]
    if m.shape[1:] != (H, W):
        m = torch.nn.functional.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    return m.clamp(0, 1)


def mask_boxes(m, min_area=64):
    """Mỗi mảng tô rời nhau -> 1 bbox (x,y,w,h). Không có OpenCV -> 1 bbox bao hết."""
    b = (m[0] > 0.5)
    if not bool(b.any()):
        return []
    try:
        import cv2
        import numpy as np
        n, _, st, _ = cv2.connectedComponentsWithStats(b.cpu().numpy().astype(np.uint8), connectivity=8)
        boxes = [tuple(int(v) for v in st[i, :4]) for i in range(1, n) if st[i, 4] >= min_area]
        if boxes:
            return merge_overlapping(boxes)
    except ImportError:
        pass
    ys, xs = torch.nonzero(b, as_tuple=True)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return [(x0, y0, x1 - x0, y1 - y0)]


def merge_overlapping(boxes, gap=0):
    """Gộp các bbox chồng nhau (tránh 2 vùng crop đè nhau ghép 2 lần)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for k in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[k]
                if (a[0] - gap < c[0] + c[2] and c[0] - gap < a[0] + a[2]
                        and a[1] - gap < c[1] + c[3] and c[1] - gap < a[1] + a[3]):
                    x0, y0 = min(a[0], c[0]), min(a[1], c[1])
                    x1, y1 = max(a[0] + a[2], c[0] + c[2]), max(a[1] + a[3], c[1] + c[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    boxes.pop(k)
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def crop_box(box, pad, W, H):
    """bbox vùng sửa + nới 'pad' px ngữ cảnh -> (x0,y0,x1,y1): góc trên-trái LUÔN bội số ALIGN, nằm trong ảnh.
    Sát mép phải/dưới của ảnh có cạnh không chia hết 32 thì cỡ crop có thể lẻ — KHÔNG lùi x0/y0 nữa
    (bản cũ lùi x0/y0 làm góc crop lệch lưới 32: ảnh 600px -> y0 = 312). Cỡ lẻ đã được lo ở khâu sau:
    run_pass pad canvas lên bội 32 (plan_tiles) rồi cắt bỏ, stitch tự đưa patch về đúng cỡ crop."""
    x, y, w, h = box
    x0 = max(0, (x - pad) // ALIGN * ALIGN)
    y0 = max(0, (y - pad) // ALIGN * ALIGN)
    x1 = min(W, int(math.ceil((x + w + pad) / ALIGN) * ALIGN))
    y1 = min(H, int(math.ceil((y + h + pad) / ALIGN) * ALIGN))
    return int(x0), int(y0), int(x1), int(y1)


def blend_mask(region_m, box, feather, crop):
    """Mask hoà [1,h,w] trong vùng crop: vùng tô (hoặc bbox) nới feather rồi làm mềm, ép = 0 ở viền crop."""
    x0, y0, x1, y1 = crop
    h, w = y1 - y0, x1 - x0
    if region_m is not None:
        m = region_m[:, y0:y1, x0:x1].clone()
    else:
        m = torch.zeros(1, h, w)
        bx, by, bw, bh = box
        m[:, max(0, by - y0):max(0, by - y0 + bh), max(0, bx - x0):max(0, bx - x0 + bw)] = 1.0
    if feather > 0:
        r = max(1, int(feather // 2))
        m = torch.nn.functional.max_pool2d(m[:, None], 2 * r + 1, stride=1, padding=r)[:, 0]  # nới ra ngoài vùng tô
        m = core.gaussian_blur(m[:, None], feather / 2.5)[:, 0]
    # viền crop (trừ chỗ trùng mép ảnh) phải = 0 để không lộ đường cắt
    e = max(2, min(int(feather // 3), w // 4, h // 4))
    ramp_x = torch.ones(w)
    ramp_y = torch.ones(h)
    edge = torch.linspace(0, 1, e)
    if x0 > 0:
        ramp_x[:e] = torch.minimum(ramp_x[:e], edge)
    if y0 > 0:
        ramp_y[:e] = torch.minimum(ramp_y[:e], edge)
    return (m * ramp_y[None, :, None] * ramp_x[None, None, :]).clamp(0, 1)


def make_regions(image, region_mask=None, region_box="", padding=96, feather=24, merge_all=False):
    """-> list REGION_INFO. merge_all: gộp mọi vùng thành 1 (cho cặp Crop/Stitch)."""
    H, W = image.shape[1], image.shape[2]
    m = fit_mask(region_mask, H, W) if region_mask is not None else None
    boxes = parse_boxes(region_box)
    if m is not None and float(m.max()) > 0.5:
        boxes = mask_boxes(m) + boxes
    if not boxes:
        raise ValueError("Chưa có vùng: tô mask (MaskEditor) hoặc gõ region_box 'x,y,w,h'")
    boxes = [(max(0, x), max(0, y), min(w, W - max(0, x)), min(h, H - max(0, y))) for x, y, w, h in boxes]
    if merge_all and len(boxes) > 1:
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        boxes = [(x0, y0, x1 - x0, y1 - y0)]
    # vùng crop chồng nhau -> gộp để mỗi pixel chỉ được dán 1 lần
    crops = merge_overlapping([(c[0], c[1], c[2] - c[0], c[3] - c[1])
                               for c in (crop_box(b, padding, W, H) for b in boxes)])
    infos = []
    for cx, cy, cw, ch in crops:
        inner = [b for b in boxes if b[0] >= cx and b[1] >= cy and b[0] + b[2] <= cx + cw and b[1] + b[3] <= cy + ch]
        inner = inner or [(cx, cy, cw, ch)]
        bx0 = min(b[0] for b in inner); by0 = min(b[1] for b in inner)
        bx1 = max(b[0] + b[2] for b in inner); by1 = max(b[1] + b[3] for b in inner)
        box = (bx0, by0, bx1 - bx0, by1 - by0)
        crop = (cx, cy, cx + cw, cy + ch)
        rm = m if (m is not None and float(m[:, cy:cy + ch, cx:cx + cw].max()) > 0.5) else None
        if rm is not None and region_box:
            # có cả mask và box: cộng bbox vào mask cho vùng crop này
            rm = rm.clone()
            for b in parse_boxes(region_box):
                rm[:, b[1]:b[1] + b[3], b[0]:b[0] + b[2]] = 1.0
        infos.append({"W": W, "H": H, "crop": crop, "box": box,
                      "mask": blend_mask(rm, box, feather, crop), "feather": feather})
    return infos


# ---------------------------------------------------------------------------- crop / stitch
def crop(image, info):
    x0, y0, x1, y1 = info["crop"]
    return image[:1, y0:y1, x0:x1, :3].float()


def stitch(base, patch, info, color_match=True, color_strength=1.0, lock_sigma=8.0):
    """Dán patch về đúng toạ độ. color_match: màu/mảng lớn khoá theo vùng cũ, giữ chi tiết mới."""
    if (base.shape[1], base.shape[2]) != (info["H"], info["W"]):
        raise ValueError(f"Ảnh ghép {base.shape[2]}x{base.shape[1]} khác ảnh lúc crop {info['W']}x{info['H']} "
                         "-> phải ghép vào đúng ảnh đã crop")
    x0, y0, x1, y1 = info["crop"]
    old = base[:1, y0:y1, x0:x1, :3].float()
    p = patch[:1, ..., :3].float().cpu()
    if p.shape[1:3] != old.shape[1:3]:
        p = core.resize(p, old.shape[2], old.shape[1], "area" if p.shape[1] > old.shape[1] else "bicubic")
    if color_match and color_strength > 0:
        p = core.color_lock(p, old, "detail_transfer", color_strength, lock_sigma)
    a = info["mask"].float()[..., None]  # [1,h,w,1]
    out = base[:1, ..., :3].float().clone()
    new = old * (1 - a) + p * a
    out[:, y0:y1, x0:x1] = new
    return out, qa(old, new, info)


def qa(old, new, info):
    """ΔE vùng sửa (mảng lớn) + ΔE dải viền hoà (lộ đường nối?)."""
    a = info["mask"][0]
    de = core.delta_e76(core.to_bhwc(core.gaussian_blur(core.to_bchw(old), 2.0)),
                        core.to_bhwc(core.gaussian_blur(core.to_bchw(new), 2.0)))[0]
    ring = (a > 0.05) & (a < 0.95)
    inside = a >= 0.95
    di = float(de[inside].mean()) if bool(inside.any()) else 0.0
    dr = float(de[ring].mean()) if bool(ring.any()) else 0.0
    x0, y0, x1, y1 = info["crop"]
    flag = "OK" if dr <= 4.0 else "CẢNH BÁO viền lệch màu"
    return (f"vùng crop ({x0},{y0})-({x1},{y1}) {x1 - x0}x{y1 - y0} | ΔE trong vùng {di:.2f} | "
            f"ΔE viền hoà {dr:.2f} {flag}")


def before_after(pairs, max_side=1536):
    """[(old,new)] -> 1 ảnh: mỗi hàng = trước | sau."""
    rows = []
    for old, new in pairs:
        k = min(1.0, max_side / 2 / max(old.shape[1], old.shape[2]))
        w, h = max(8, int(old.shape[2] * k)), max(8, int(old.shape[1] * k))
        o, n = core.resize(old, w, h, "area"), core.resize(new, w, h, "area")
        gap = torch.ones(1, h, 8, 3)
        rows.append(torch.cat([o, gap, n], 2))
    W = max(r.shape[2] for r in rows)
    rows = [torch.nn.functional.pad(r.movedim(-1, 1), (0, W - r.shape[2], 0, 8), value=1.0).movedim(1, -1) for r in rows]
    return torch.cat(rows, 1)[:, :-8]


# ---------------------------------------------------------------------------- nạp ảnh lớn không qua upload
def load_image_file(path):
    """Đọc ảnh từ đĩa (PNG/TIFF 16-bit, JPG...) -> [1,H,W,3] float 0..1. Không qua upload trình duyệt."""
    import numpy as np
    try:
        import cv2
        a = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # fromfile: đường dẫn có ký tự đặc biệt / Unicode
        if a is not None:
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            a = a[..., :3][..., ::-1]
            scale = 65535.0 if a.dtype == np.uint16 else (1.0 if a.dtype.kind == "f" else 255.0)
            return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32) / scale)[None].clamp(0, 1)
    except ImportError:
        pass
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]


def write_proxy(img, path, max_side=2048):
    """Ảnh thu nhỏ JPG để tô mask trong MaskEditor (Region Fix tự phóng mask về cỡ thật)."""
    import numpy as np
    from PIL import Image
    h, w = img.shape[1], img.shape[2]
    k = min(1.0, max_side / max(h, w))
    x = core.resize(img[:1, ..., :3].float(), max(1, round(w * k)), max(1, round(h * k)), "area") if k < 1 else img[:1, ..., :3]
    Image.fromarray((x[0].clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)).save(path, quality=92)
    return x.shape[2], x.shape[1]
