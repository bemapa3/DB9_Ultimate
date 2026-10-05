"""DB9U Save: lưu local / output ComfyUI (tự nhận cloud), PNG/TIFF 16-bit, xuất cặp preview cho node Compare Images."""
import os
import re

import numpy as np
import torch

import folder_paths

from . import core

CATEGORY = "DB9 Ultimate"
FORMATS = ["png16", "png8", "tiff16", "jpg95"]
EXT = {"png16": ".png", "png8": ".png", "tiff16": ".tif", "jpg95": ".jpg"}


def is_cloud():
    """Colab / Kaggle / RunPod... -> lưu vào output của ComfyUI."""
    env = os.environ
    return bool(env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU") or env.get("KAGGLE_KERNEL_RUN_TYPE")
                or env.get("RUNPOD_POD_ID") or os.path.isdir("/content/drive"))


def _next_path(folder, prefix, ext):
    os.makedirs(folder, exist_ok=True)
    base = os.path.basename(prefix) or "DB9U"
    sub = os.path.dirname(prefix)
    folder = os.path.join(folder, sub) if sub else folder
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(re.escape(base) + r"_(\d{5})")
    n = 0
    for f in os.listdir(folder):
        m = pat.match(f)
        if m:
            n = max(n, int(m.group(1)))
    return os.path.join(folder, f"{base}_{n + 1:05d}{ext}"), sub


def write_image(path, img_hw3, fmt):
    """img_hw3: tensor [H,W,3] 0..1."""
    x = img_hw3.clamp(0, 1).cpu().numpy()
    try:
        import cv2
        bgr = x[..., ::-1]
        if fmt in ("png16", "tiff16"):
            ok = cv2.imwrite(path, (bgr * 65535.0 + 0.5).astype(np.uint16))
        elif fmt == "jpg95":
            ok = cv2.imwrite(path, (bgr * 255.0 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            ok = cv2.imwrite(path, (bgr * 255.0 + 0.5).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if ok:
            return path, fmt
    except ImportError:
        pass
    # Không có opencv -> PIL 8-bit
    from PIL import Image
    im = Image.fromarray((x * 255.0 + 0.5).astype(np.uint8))
    if fmt == "jpg95":
        im.save(path, quality=95)
        return path, fmt
    path = os.path.splitext(path)[0] + ".png"
    im.save(path, compress_level=4)
    return path, "png8 (thiếu opencv)"


class DB9U_Save:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "DB9U/img"}),
                "save_to": (["auto", "comfy_output", "local_folder"], {"default": "auto",
                            "tooltip": "auto: có local_folder và không chạy cloud -> lưu local, ngược lại -> output ComfyUI"}),
                "local_folder": ("STRING", {"default": "", "tooltip": r"Vd: J:\Render\Upscale (để trống = output ComfyUI)"}),
                "format": (FORMATS, {"default": "png16", "tooltip": "png16/tiff16: 16-bit cho Photoshop (cần opencv)"}),
                "preview_max": ("INT", {"default": 3072, "min": 512, "max": 8192, "step": 256,
                                "tooltip": "Cạnh dài cặp ảnh preview cho node Compare Images (ảnh 10K quá nặng cho trình duyệt)"}),
            },
            "optional": {"original": ("IMAGE", {"tooltip": "Nối original_resized từ DB9U Upscale để so sánh"})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("result_preview", "original_preview", "saved_path")
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save(self, image, filename_prefix, save_to, local_folder, format, preview_max, original=None):
        cloud = is_cloud()
        use_local = save_to == "local_folder" or (save_to == "auto" and local_folder.strip() and not cloud)
        root = local_folder.strip() if use_local and local_folder.strip() else folder_paths.get_output_directory()
        paths = []
        for i in range(image.shape[0]):
            path, sub = _next_path(root, filename_prefix, EXT[format])
            path, used = write_image(path, image[i, ..., :3], format)
            paths.append(path)
            print(f"[DB9U] Đã lưu {used}: {path}")
        h, w = image.shape[1:3]
        k = min(1.0, preview_max / max(h, w))
        pw, ph = max(1, round(w * k)), max(1, round(h * k))
        res_p = core.resize(image[..., :3].float().cpu(), pw, ph, "area") if k < 1 else image[..., :3].float().cpu()
        if original is not None:
            org = original[..., :3].float().cpu()
            org_p = core.resize(org, pw, ph, "area") if org.shape[1:3] != (ph, pw) else org
        else:
            org_p = res_p
        # Không đẩy ảnh 10K lên giao diện (nặng) — xem bằng node Compare Images qua 2 output preview
        return (res_p, org_p, "\n".join(paths))
