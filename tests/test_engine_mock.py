"""Test toàn bộ pipeline với ComfyUI GIẢ (không cần model thật, chạy vài giây).
Chạy: <ComfyUI>\\python_embeded\\python.exe tests\\test_engine_mock.py
Kiểm: kích thước output, số lượt, chia lô theo batch, retry, denoise theo vùng, ControlNet adapter được gọi, không vỡ khi OOM giả.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------- stub comfy
calls = {"sample": 0, "batch_sizes": [], "noise_mask": 0, "sigmas": 0, "cn": 0, "oom_once": False}


class FakeOOM(Exception):
    pass


mm = types.ModuleType("comfy.model_management")
mm.OOM_EXCEPTION = FakeOOM
mm.get_torch_device = lambda: torch.device("cpu")
mm.get_total_memory = lambda dev=None: 16 * 1024 ** 3
mm.soft_empty_cache = lambda: None
mm.throw_exception_if_processing_interrupted = lambda: None

cu = types.ModuleType("comfy.utils")
cu.PROGRESS_BAR_ENABLED = False


class _PB:
    def __init__(self, n):
        pass

    def update(self, k):
        pass


cu.ProgressBar = _PB
cu.common_upscale = lambda x, w, h, m, c: F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)

cs = types.ModuleType("comfy.sample")
cs.fix_empty_latent_channels = lambda model, lat, *a, **k: lat
cs.prepare_noise = lambda lat, seed, inds: torch.zeros_like(lat)


def _sample(model, noise, steps, cfg, sampler, sched, pos, neg, lat, denoise=1.0, noise_mask=None, sigmas=None, **k):
    calls["sample"] += 1
    calls["batch_sizes"].append(lat.shape[0])
    if noise_mask is not None:
        calls["noise_mask"] += 1
        assert noise_mask.shape[0] == lat.shape[0]
    if sigmas is not None:
        calls["sigmas"] += 1
    if calls["oom_once"]:
        calls["oom_once"] = False
        raise FakeOOM("giả lập hết VRAM")
    return lat + 0.01  # "enhance" nhẹ


cs.sample = _sample
csm = types.ModuleType("comfy.samplers")
csm.KSampler = type("KSampler", (), {"SAMPLERS": ["euler"], "SCHEDULERS": ["simple"]})
lp = types.ModuleType("latent_preview")
lp.prepare_callback = lambda model, steps: None
comfy = types.ModuleType("comfy")
nodes_mod = types.ModuleType("nodes")


class _CNApply:
    def apply_controlnet(self, pos, neg, cn, img, s, a, b, vae=None):
        calls["cn"] += 1
        assert img.shape[-1] == 3
        return pos, neg


nodes_mod.ControlNetApplyAdvanced = _CNApply
nodes_mod.NODE_CLASS_MAPPINGS = {}
flux = types.ModuleType("comfy_extras.nodes_flux")
flux.get_schedule = lambda n, seq: torch.linspace(1, 0, n + 1)
ce = types.ModuleType("comfy_extras")
for name, m in {"comfy": comfy, "comfy.model_management": mm, "comfy.utils": cu, "comfy.sample": cs,
                "comfy.samplers": csm, "latent_preview": lp, "nodes": nodes_mod, "comfy_extras": ce,
                "comfy_extras.nodes_flux": flux}.items():
    sys.modules[name] = m
comfy.model_management, comfy.utils, comfy.sample, comfy.samplers = mm, cu, cs, csm
ce.nodes_flux = flux

from db9u import engine, nodes  # noqa: E402


class FakeVAE:
    def encode(self, x):
        return F.avg_pool2d(x.movedim(-1, 1), 8)

    def decode(self, z):
        return F.interpolate(z, scale_factor=8, mode="bilinear", align_corners=False).movedim(1, -1)


class QwenImage21:  # tên class để profile tự nhận
    pass


class FakeModel:
    def __init__(self, cls=QwenImage21):
        self.model = cls()

    def clone(self):
        return self

    def set_model_denoise_mask_function(self, f):
        pass


def img(h=600, w=1000):
    g = torch.Generator().manual_seed(1)
    small = torch.rand(1, 3, h // 40, w // 40, generator=g)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1).movedim(1, -1)


COND = [[torch.zeros(1, 4, 8), {}]]


def run(settings=None, **kw):
    args = dict(image=img(), model=FakeModel(), vae=FakeVAE(), positive=COND, negative=COND,
                target="x3", denoise=0.4, cfg=0.0, preset="balanced", seed=1, settings=settings)
    args.update(kw)
    return nodes.DB9U_Upscale().run(**args)


def test_size_and_passes():
    out, orig, log, tex = run()
    assert out.shape == (1, 1800, 3000, 3), out.shape
    assert orig.shape == out.shape and tex.shape == out.shape
    assert "2 lượt" in log and "qwen_image_21" in log, log[:400]
    print("OK size/passes |", log.splitlines()[0])


def test_regional_and_cfg_auto():
    calls["noise_mask"] = 0
    out, _, log, _ = run()
    assert calls["noise_mask"] > 0, "denoise theo vùng không chạy"
    assert "cfg 1.0" in log


def test_batch_and_native_cn():
    calls["batch_sizes"].clear()
    calls["cn"] = 0
    s = dict(engine.DEFAULTS, batch_tiles=2, max_tile=512, tile_retries=0)
    out, _, log, _ = run(settings=s, control_net=object(), target="x2")
    assert max(calls["batch_sizes"]) == 2, calls["batch_sizes"]
    assert calls["cn"] > 0
    assert out.shape == (1, 1200, 2000, 3)


def test_flux2_profile_sigmas():
    class Flux2:
        pass
    calls["sigmas"] = 0
    out, _, log, _ = run(model=FakeModel(Flux2), target="x2")
    assert "flux2_klein_distilled" in log and calls["sigmas"] > 0
    assert "4 steps" in log


def test_oom_recovery():
    calls["oom_once"] = True
    out, _, log, _ = run(target="x2", settings=dict(engine.DEFAULTS, tile_retries=0))
    assert "Hết VRAM" in log and out.shape == (1, 1200, 2000, 3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
    print("ALL PASS")
