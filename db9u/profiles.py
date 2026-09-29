"""Model Profile + Hardware Profile.

Thêm checkpoint mới: tạo file JSON trong db9u/profiles/ (xem generic.json). Không cần sửa code.
"""
import glob
import json
import os

import comfy.model_management as mm

_DIR = os.path.join(os.path.dirname(__file__), "profiles")
REQUIRED = ("name", "max_tile", "steps", "cfg", "sampler", "scheduler")


def load_profiles():
    out = {}
    for f in sorted(glob.glob(os.path.join(_DIR, "*.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                p = json.load(fh)
            if all(k in p for k in REQUIRED):
                p.setdefault("match", [])
                p.setdefault("reference_mode", "strip")
                p.setdefault("control_backend", "native")
                p.setdefault("control_mode", "tile")
                out[p["name"]] = p
        except Exception as e:  # file hỏng không được làm hỏng cả bộ node
            print(f"[DB9U] Bỏ qua profile lỗi {os.path.basename(f)}: {e}")
    if "generic" not in out:
        out["generic"] = {"name": "generic", "label": "Generic", "match": [], "max_tile": 1024, "steps": 20,
                          "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "reference_mode": "strip",
                          "control_backend": "native", "control_mode": "tile"}
    return out


PROFILES = load_profiles()
PROFILE_CHOICES = ["auto"] + sorted(PROFILES.keys())


def model_signature(model):
    """Chuỗi nhận diện model: tên class model_base + diffusion_model."""
    names = []
    try:
        names.append(type(model.model).__name__)
        names.append(type(model.model.diffusion_model).__name__)
    except Exception:
        pass
    return " ".join(names).lower()


def detect_profile(model, override="auto"):
    if override and override != "auto" and override in PROFILES:
        return dict(PROFILES[override]), f"{override} (chọn tay)"
    sig = model_signature(model)
    for name, p in PROFILES.items():
        if any(m.lower() in sig for m in p.get("match", [])):
            return dict(p), f"{name} (tự nhận: {sig})"
    return dict(PROFILES["generic"]), f"generic (không nhận ra: {sig or 'n/a'})"


# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
def gpu_info():
    try:
        dev = mm.get_torch_device()
        total = mm.get_total_memory(dev) / 1024 ** 3
        name = getattr(dev, "type", str(dev))
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(dev)
        except Exception:
            pass
        return name, total
    except Exception:
        return "unknown", 8.0


def hardware_plan(profile, has_qwen_cn=False):
    """Trả về (max_tile, batch_tiles, mô tả). Tile không vượt profile (độ phân giải gốc của model)."""
    name, gb = gpu_info()
    cap = profile["max_tile"]
    if gb < 11:
        tile = min(cap, 1024)
    elif gb < 15:
        tile = min(cap, 1536)
    else:
        tile = cap
    if has_qwen_cn:  # ControlNet Qwen 2.1 Fun ~7GB, patch model theo từng ô -> không batch
        batch = 1
    elif gb >= 70:   # A100/H100 80GB
        batch = 4
    elif gb >= 40:   # A100 40GB, A6000 48GB
        batch = 2
    else:            # 5070 Ti, 5080, 3090, 4090, 5090
        batch = 1
    return tile, batch, f"{name} {gb:.0f}GB -> ô ≤{tile}px, {batch} ô/lượt"
