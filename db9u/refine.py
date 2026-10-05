"""DB9U SR Refine — chạy lại ảnh qua upscale model (ESRGAN/DAT/Nomos...) rồi thu về ĐÚNG kích thước cũ.
Supersampling: model vẽ chi tiết ở x2/x4, thu nhỏ lại bằng 'area' -> nét hơn, không tăng kích thước.
Chạy theo ô (không bao giờ giữ cả ảnh x4 trong RAM). Mặc định chỉ lấy CHI TIẾT MỊN của model,
mảng màu / form giữ nguyên ảnh vào -> không lệch màu, không lệch viền.
Chỉnh thông số node này: ComfyUI giữ cache các node phía trước (seed DB9U Upscale để 'fixed'),
còn node này giữ cache kết quả model -> chỉ trộn lại, không chạy model lẫn upscale lại."""
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

from . import core


_CACHE = {}  # khoá (ảnh, model, ô) -> ảnh SR đã thu về cỡ cũ (half, CPU). Giữ 2 bản mới nhất.


def _key(img, upscale_model, tile):
    x = img[0, ::7, ::7].float()
    return (id(upscale_model), tuple(img.shape), tile, round(float(x.sum()), 3), round(float((x * x).sum()), 3))


def model_pass(img, upscale_model, tile=512, overlap=32, say=print):
    """Phần nặng: chạy model theo ô + area về cỡ cũ. Có cache -> chỉnh strength/mode/sigma không chạy model lại."""
    k = _key(img, upscale_model, tile)
    if k in _CACHE:
        say("[DB9U] SR Refine: dùng lại kết quả model (chỉ trộn lại theo thông số mới, vài giây)")
        return _CACHE[k].float()
    sr = _model_pass(img, upscale_model, tile, overlap, say)
    while len(_CACHE) >= 2:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[k] = sr.half()
    return sr


def blend(img, sr, strength=0.6, mode="detail_only", detail_sigma=1.5, max_detail=0.12):
    """Phần nhẹ: trộn ảnh vào với ảnh SR theo thông số."""
    if mode == "full":
        res = img + strength * (sr - img)
    else:
        a, b = img.movedim(-1, 1), sr.movedim(-1, 1)
        d = (b - core.gaussian_blur(b, detail_sigma)) - (a - core.gaussian_blur(a, detail_sigma))
        d = d.clamp(-max_detail, max_detail)  # chặn quầng sáng/tối ở cạnh gắt
        res = (a + strength * d).movedim(1, -1)
    return res.clamp(0, 1)


def sr_refine(img, upscale_model, strength=0.6, mode="detail_only", detail_sigma=1.5, tile=512, overlap=32,
              max_detail=0.12, say=print):
    """img [1,H,W,3] -> cùng kích thước."""
    return blend(img, model_pass(img, upscale_model, tile, overlap, say), strength, mode, detail_sigma, max_detail)


def _model_pass(img, upscale_model, tile=512, overlap=32, say=print):
    H, W = img.shape[1:3]
    plan = core.plan_tiles(W, H, tile, overlap)
    tiles = core.split_tiles(img, plan)
    n = tiles.shape[0]
    dev = mm.get_torch_device()
    upscale_model.to(dev)
    out = []
    pbar = comfy.utils.ProgressBar(n)
    try:
        for i in range(n):
            mm.throw_exception_if_processing_interrupted()
            t = tiles[i:i + 1].movedim(-1, 1).to(dev)
            th, tw = t.shape[-2:]
            while True:
                try:
                    s = upscale_model(t).float().clamp(0, 1)
                    break
                except mm.OOM_EXCEPTION:
                    mm.soft_empty_cache()
                    if tile <= 128:
                        raise
                    # ô quá to cho VRAM -> chạy lại toàn bộ với ô nhỏ hơn
                    upscale_model.to("cpu")
                    say(f"[DB9U] SR Refine: hết VRAM -> giảm ô còn {tile // 2}px")
                    return _model_pass(img, upscale_model, tile // 2, overlap, say)
            s = F.interpolate(s, size=(th, tw), mode="area")  # thu về đúng cỡ = supersampling
            out.append(s.movedim(1, -1).cpu())
            pbar.update(1)
    finally:
        upscale_model.to("cpu")
    return core.merge_tiles(torch.cat(out, 0), plan)


class DB9U_SRRefine:
    """Làm nét bằng upscale model mà KHÔNG tăng kích thước. Đặt sau DB9U Upscale, trước Finish/Save."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Model làm nét, vd 4x-UltraSharp / 4x-Nomos / DAT. Model 1x cũng dùng được"}),
            "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05,
                         "tooltip": "Mức thêm chi tiết. 0.4-0.8 tự nhiên; >1 nét gắt"}),
            "mode": (["detail_only", "full"], {"default": "detail_only",
                     "tooltip": "detail_only: chỉ lấy chi tiết mịn của model, màu/mảng/viền giữ nguyên (an toàn) · full: trộn thẳng ảnh model (có thể đổi màu/tương phản)"}),
            "detail_sigma": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 6.0, "step": 0.25,
                             "tooltip": "Cỡ 'chi tiết mịn' (px). Lớn hơn -> lấy cả vân vừa, nét mạnh hơn"}),
            "tile": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                     "tooltip": "Cỡ ô đưa vào model. Hết VRAM tự giảm"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "DB9 Ultimate"

    def run(self, image, upscale_model, strength, mode, detail_sigma, tile):
        if strength <= 0:
            return (image,)
        outs = []
        for i in range(image.shape[0]):
            img = image[i:i + 1, ..., :3].float().cpu()
            print(f"[DB9U] SR Refine {img.shape[2]}x{img.shape[1]} (giữ nguyên kích thước) · model x{getattr(upscale_model, 'scale', '?')} "
                  f"· {mode} {strength}")
            outs.append(sr_refine(img, upscale_model, strength, mode, detail_sigma, tile))
        return (torch.cat(outs, 0),)
