"""DB9 Ultimate — nodes."""
import torch

import comfy.samplers

from . import core, engine, profiles
from .qa import DB9U_QAQC

CATEGORY = "DB9 Ultimate"


class DB9U_Upscale:
    """Node chính: chỉ cần chọn đích, denoise, cfg. Mọi thứ khác tự động (profile model + VRAM)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "target": (list(engine.TARGETS.keys()), {"default": "8K",
                           "tooltip": "x2/x3/x4 = hệ số · 6K/8K/10K/11K = cạnh dài 6144/8192/10240/11264px. Tỉ lệ ảnh giữ nguyên"}),
                "denoise": ("FLOAT", {"default": 0.40, "min": 0.05, "max": 1.0, "step": 0.01,
                            "tooltip": "Kiến trúc/mảng phẳng. Vùng lá/texture tự +0.2. Khuyến nghị 0.30-0.55"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                        "tooltip": "0 = tự theo model (Qwen 2.1 / Klein distilled: 1 · Klein base: 5)"}),
                "preset": (engine.PRESETS, {"default": "balanced",
                           "tooltip": "fast: ít bước, không chạy lại ô · balanced · max: thêm bước, giao rộng, chạy lại ô lệch tới 2 lần"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Phóng trước bằng model (chỉ lượt đầu). Không nối = lanczos"}),
                "control_net": ("CONTROL_NET", {"tooltip": "ControlNet chuẩn ComfyUI (Flux, SDXL...). Không tương thích -> tự tắt"}),
                "qwen21_controlnet": ("QWEN21_FUN_CONTROLNET", {"tooltip": "Từ node 'Load Qwen-Image 2.1 Fun ControlNet'"}),
                "control_image": ("IMAGE", {"tooltip": "Chỉ dùng khi control_mode = external (cả khung, cùng tỉ lệ)"}),
                "settings": ("DB9U_SETTINGS", {"tooltip": "Từ node DB9U Advanced Settings (không nối = tự động hết)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "original_resized", "log", "texture_mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, model, vae, positive, negative, target, denoise, cfg, preset, seed,
            upscale_model=None, control_net=None, qwen21_controlnet=None, control_image=None, settings=None):
        job = engine.Job(model, denoise, cfg, preset, seed, settings, qwen21_controlnet, control_net)
        eng = engine.Engine(job, model, vae, positive, negative, control_image)
        eng.say(job.describe())
        outs, origs, masks = [], [], []
        for bi in range(image.shape[0]):
            img = image[bi:bi + 1, ..., :3].float().cpu()
            out, tex = eng.run_image(img, target, upscale_model)
            outs.append(out)
            origs.append(core.resize(img, out.shape[2], out.shape[1], "bicubic"))
            masks.append(tex)
        if len(outs) > 1:
            ref = outs[0].shape
            outs = [o if o.shape == ref else core.resize(o, ref[2], ref[1], "bicubic") for o in outs]
            origs = [o if o.shape == ref else core.resize(o, ref[2], ref[1], "bicubic") for o in origs]
            masks = [m if m.shape[1:3] == ref[1:3] else core.resize(m, ref[2], ref[1], "bilinear") for m in masks]
        return (torch.cat(outs, 0), torch.cat(origs, 0), "\n".join(eng.log), torch.cat(masks, 0))


class DB9U_Settings:
    """Tuỳ chỉnh sâu (không bắt buộc). Giá trị 0 / -1 / auto = để node tự quyết."""

    @classmethod
    def INPUT_TYPES(cls):
        d = engine.DEFAULTS
        samplers = ["auto"] + list(comfy.samplers.KSampler.SAMPLERS)
        schedulers = ["auto", "flux2"] + list(comfy.samplers.KSampler.SCHEDULERS)
        return {"required": {
            "profile": (profiles.PROFILE_CHOICES, {"default": "auto", "tooltip": "auto = tự nhận model. Klein base cần chọn tay"}),
            "custom_long_edge": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16, "tooltip": ">0: bỏ qua target"}),
            "max_tile": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "0 = auto (profile + VRAM)"}),
            "overlap": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 32, "tooltip": "0 = auto"}),
            "max_scale_per_pass": ("FLOAT", {"default": d["max_scale_per_pass"], "min": 1.2, "max": 8.0, "step": 0.1}),
            "batch_tiles": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "0 = auto theo VRAM"}),
            "steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = theo profile"}),
            "sampler": (samplers, {"default": "auto"}),
            "scheduler": (schedulers, {"default": "auto", "tooltip": "flux2 = như node Flux2Scheduler"}),
            "texture_denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                "tooltip": "-1 = auto (denoise+0.2) · 0 = tắt denoise theo vùng"}),
            "min_denoise": ("FLOAT", {"default": d["min_denoise"], "min": 0.0, "max": 1.0, "step": 0.01}),
            "tile_retries": ("INT", {"default": -1, "min": -1, "max": 5, "tooltip": "-1 = theo preset"}),
            "tile_max_deltaE": ("FLOAT", {"default": d["tile_max_deltaE"], "min": 0.5, "max": 50.0, "step": 0.1}),
            "align_tiles": ("BOOLEAN", {"default": True}),
            "seed_mode": (["fixed", "per_tile"], {"default": "fixed"}),
            "reference_mode": (["auto", "strip", "tile", "keep"], {"default": "auto"}),
            "color_lock": (core_color_modes(), {"default": d["color_lock"]}),
            "lock_strength": ("FLOAT", {"default": d["lock_strength"], "min": 0.0, "max": 1.0, "step": 0.05}),
            "lock_sigma": ("FLOAT", {"default": d["lock_sigma"], "min": 0.5, "max": 64.0, "step": 0.5}),
            "edge_guard": ("FLOAT", {"default": d["edge_guard"], "min": 0.0, "max": 1.0, "step": 0.05}),
            "control_mode": (core.CONTROL_MODES, {"default": "auto"}),
            "control_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, "tooltip": "-1 = auto"}),
            "control_start": ("FLOAT", {"default": d["control_start"], "min": 0.0, "max": 1.0, "step": 0.01}),
            "control_end": ("FLOAT", {"default": d["control_end"], "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("DB9U_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, **kw):
        return (dict(kw),)


def core_color_modes():
    return ["detail_transfer", "lab_stats", "both", "none"]


NODE_CLASS_MAPPINGS = {
    "DB9U_Upscale": DB9U_Upscale,
    "DB9U_Settings": DB9U_Settings,
    "DB9U_QAQC": DB9U_QAQC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DB9U_Upscale": "DB9U Upscale (Ultimate)",
    "DB9U_Settings": "DB9U Advanced Settings",
    "DB9U_QAQC": "DB9U QAQC Compare",
}
