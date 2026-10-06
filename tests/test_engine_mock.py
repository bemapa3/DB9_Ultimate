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
cu.tiled_scale = lambda t, fn, tile_x=512, tile_y=512, overlap=32, upscale_amount=4, **k: fn(t)
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


class _Preview:
    def save_images(self, images, prefix):
        return {"ui": {"images": [{"filename": prefix + ".png", "subfolder": "", "type": "temp", "shape": tuple(images.shape)}]}}


nodes_mod.PreviewImage = _Preview
nodes_mod.NODE_CLASS_MAPPINGS = {}
import tempfile
fp = types.ModuleType("folder_paths")
OUTDIR = tempfile.mkdtemp()
fp.get_output_directory = lambda: OUTDIR
sys.modules["folder_paths"] = fp
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
    x = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    x[..., w // 2:] = x[..., w // 2:] * 0.6 + 0.4 * torch.rand(1, 3, h, w - w // 2, generator=g)  # nửa phải = "lá"
    return x.clamp(0, 1).movedim(1, -1)


COND = [[torch.zeros(1, 4, 8), {}]]


def run(settings=None, **kw):
    args = dict(image=img(), model=FakeModel(), vae=FakeVAE(), positive=COND, negative=COND,
                target="x3", denoise=0.4, cfg=0.0, preset="balanced", seed=1, settings=settings)
    args.update(kw)
    r = nodes.DB9U_Upscale().run(**args)
    run.ui = r.get("ui", {}) if isinstance(r, dict) else {}
    return r["result"] if isinstance(r, dict) else r


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


def test_preview_tile_and_resume():
    s = dict(engine.DEFAULTS, preview_tile=True, max_tile=512, preview_layout="grid")
    calls["sample"] = 0
    out, orig, log, zm = run(settings=s, target="x2")  # preview_tiles trống -> chỉ bảng ô
    assert "BẢNG Ô" in log and calls["sample"] == 0 and out.shape == orig.shape, (out.shape, log[:300])
    s = dict(s, preview_tiles="O1, O3")
    out, orig, log, zm = run(settings=s, target="x2")
    assert "PREVIEW 2 ô: O1, O3" in log and out.shape == orig.shape == zm.shape, (out.shape, orig.shape, zm.shape)
    assert out.shape[2] > out.shape[1] * 0.9  # 2 ô ghép ngang
    s = dict(s, preview_tiles="auto")
    out, orig, log, zm = run(settings=s, target="x2")
    assert "PREVIEW 1 ô" in log and out.shape[1] <= 512, out.shape
    # resume: giả lập chạy dở — ghi sẵn cache rồi chạy lại -> phải dùng lại ô
    job = engine.Job(FakeModel(), 0.4, 0.0, "fast", 1, dict(engine.DEFAULTS, max_tile=512), None, None)
    eng = engine.Engine(job, FakeModel(), FakeVAE(), COND, COND)
    im = img()
    eng.set_cache_key(im, "x2")
    base = engine.pre_upscale(im, 2000, 1200)
    from db9u import core
    plan = core.plan_tiles(2000, 1200, job.max_tile, job.overlap)
    d = eng._cache_dir(f"lượt 1/1_{plan['tile_w']}x{plan['tile_h']}_{plan['cols']}x{plan['rows']}")
    torch.save(core.split_tiles(base, plan)[:1].half(), os.path.join(d, "t000.pt"))
    calls["sample"] = 0
    merged, _ = eng.run_pass(base, "lượt 1/1")
    n = plan["cols"] * plan["rows"]
    assert any("resume" in l for l in eng.log), eng.log[:3]
    assert calls["sample"] == n - 1, (calls["sample"], n)  # ô 0 lấy từ cache
    eng.clear_cache()
    print("OK preview/resume")


def test_grade_nodes():
    from db9u import grade
    x = torch.rand(1, 64, 96, 3)
    assert grade.basic(x, exposure=0.5, contrast=30, highlights=-40, shadows=40, whites=10, blacks=-10,
                       temperature=20, tint=-10, vibrance=20, saturation=-10).shape == x.shape
    assert grade.presence(x, 30, 20, 40, -30, 20, 1).shape == x.shape
    assert grade.presence(x, 0, 0, -30, 0, 0).shape == x.shape
    y = grade.curves(x.clone(), "0,0 64,50 192,210 255,255", "", "0,10 255,245", "")
    assert y.shape == x.shape and (y - x).abs().mean() > 0.001
    adj = {n: (10.0, -20.0, 5.0) for n in grade.HUES}
    assert grade.hsl_mix(x, adj).shape == x.shape
    hsv = grade.rgb_to_hsv(x)
    assert (grade.hsv_to_rgb(hsv) - x).abs().max() < 1e-4, "HSV roundtrip"
    assert grade.color_balance(x, (10, 0, -10), (0, 5, 0), (-5, 0, 10)).shape == x.shape
    for t in grade.SEL_TARGETS:
        assert grade.selective_color(x, t, 10, -10, 20, 5).shape == x.shape
    ident = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, 9)] * 3, indexing="ij"), -1)  # [b,g,r] -> (r,g,b)
    lut = ident.flip(-1)
    assert (grade.apply_cube(x, lut) - x).abs().max() < 1e-3, "LUT identity"
    print("OK grade")


def test_color_lock_zoned():
    from db9u import core
    ref = torch.rand(1, 128, 128, 3)
    res = torch.rand(1, 128, 128, 3)
    tex0 = torch.zeros(1, 128, 128, 1)
    a = core.color_lock_zoned(res, ref, "detail_transfer", 1.0, 4.0, tex0, 4.0)
    b = core.color_lock(res, ref, "detail_transfer", 1.0, 4.0)
    assert (a - b).abs().max() < 1e-6, "tex=0 phải y như color_lock"
    tex1 = torch.ones(1, 128, 128, 1)
    c = core.color_lock_zoned(res, ref, "detail_transfer", 1.0, 4.0, tex1, 4.0)
    d = core.color_lock(res, ref, "detail_transfer", 1.0, 16.0)
    assert (c - d).abs().max() < 1e-6, "tex=1 phải = sigma x4"
    # vùng texture giữ chi tiết của result nhiều hơn (ít bị đè bởi gốc)
    assert (c - res).abs().mean() < (a - res).abs().mean()
    print("OK color_lock_zoned")


class FakeSR:
    scale = 4

    def to(self, dev):
        return self

    def __call__(self, t):
        return F.interpolate(t, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)


def test_band_fusion():
    from db9u import core
    g = torch.Generator().manual_seed(3)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing="ij")
    orig = torch.stack([0.2 + 0.5 * xx, 0.3 + 0.4 * yy, 0.25 + 0.25 * (xx + yy)], -1).unsqueeze(0)  # mảng mịn, không vân
    sr = (orig + 0.03 * torch.randn(1, 256, 256, 3, generator=g)).clamp(0, 1)   # "vân thật"
    dots = 0.08 * (torch.rand(1, 256, 256, 1, generator=g) > 0.9).float()       # đốm AI bịa
    ai = (orig + dots).clamp(0, 1)
    # ai_detail=0 -> mảng lớn gốc + toàn bộ chi tiết từ SR (không còn đốm)
    a0 = core.band_fusion(ai, orig, sr, None, ai_detail=0.0)
    ref = core.color_lock(sr, orig, "detail_transfer", 1.0, 8.0)
    assert (a0 - ref).abs().max() < 1e-4, "ai_detail=0 phải = SR khoá màu gốc"
    # cổng: đốm AI không tương quan với SR -> bị loại gần hết
    a1, gmap = core.band_fusion(ai, orig, sr, None, ai_detail=1.0, gate=True, return_gate=True)
    a2 = core.band_fusion(ai, orig, sr, None, ai_detail=1.0, gate=False)
    assert (a1 - ref).abs().mean() < 0.5 * (a2 - ref).abs().mean(), "cổng phải bỏ đốm AI"
    # AI có chi tiết KHỚP SR -> được nhận
    ai_ok = (sr + 0.5 * (sr - core.to_bhwc(core.gaussian_blur(core.to_bchw(sr), 2.0)))).clamp(0, 1)
    _, g2 = core.band_fusion(ai_ok, orig, sr, None, ai_detail=1.0, return_gate=True)
    assert float(g2.mean()) > float(gmap.mean()) + 0.2, (float(g2.mean()), float(gmap.mean()))
    # vùng lá (tex=1) -> y như color_lock sigma x mult
    tex = torch.ones(1, 256, 256, 1)
    a3 = core.band_fusion(ai, orig, sr, tex, texture_mult=4.0, texture_ai=1.0)
    assert (a3 - core.color_lock(ai, orig, "detail_transfer", 1.0, 32.0)).abs().max() < 1e-5
    print("OK band_fusion | gate đốm", round(float(gmap.mean()), 3), "gate khớp", round(float(g2.mean()), 3))


def test_flow_align():
    from db9u import core
    try:
        import cv2  # noqa: F401
    except Exception:
        print("SKIP flow_align: python_embeded chưa có OpenCV (cv2) -> node vẫn chạy nhưng bỏ bước căn flow")
        return
    g = torch.Generator().manual_seed(5)
    ref = F.interpolate(torch.rand(1, 3, 40, 60, generator=g), size=(320, 480), mode="bicubic").clamp(0, 1).movedim(1, -1)
    ai = torch.roll(ref, shifts=(3, -2), dims=(1, 2))  # AI xê dịch 3px/2px
    out, p95 = core.flow_align(ai, ref)
    inner = (slice(None), slice(20, -20), slice(20, -20))
    e0 = float((ai - ref)[inner].abs().mean())
    e1 = float((out - ref)[inner].abs().mean())
    assert p95 is not None and e1 < 0.3 * e0, (e0, e1, p95)
    print(f"OK flow_align lệch {e0:.4f} -> {e1:.4f} (p95 {p95:.2f}px)")


def test_with_upscale_model():
    out, _, log, _ = run(upscale_model=FakeSR(), target="x3")
    assert out.shape == (1, 1800, 3000, 3) and "enhance 1.00" in log and "structure_lock 0.50" in log and "vật liệu 0.40" in log, log[:600]
    s = dict(engine.DEFAULTS, preview_tile=True, max_tile=512, preview_tiles="auto2", enhance=0.0)
    out, orig, log, _ = run(settings=s, upscale_model=FakeSR(), target="x2")
    assert out.shape == orig.shape and "band fusion" in log
    s = dict(s, enhance=0.5)
    out, orig, log, _ = run(settings=s, upscale_model=FakeSR(), target="x2")
    assert out.shape == orig.shape and "trộn 50% band fusion" in log
    print("OK upscale_model + band fusion")


def test_parse_tiles():
    assert engine.parse_tiles("O1, O5 o6", 15) == [0, 4, 5]
    assert engine.parse_tiles("O3-O6", 15) == [2, 3, 4, 5]
    assert engine.parse_tiles("auto3", 4, torch.tensor([0.1, 0.9, 0.5, 0.2])) == [1, 2, 3]
    try:
        engine.parse_tiles("O99", 15)
        raise AssertionError("phải báo lỗi")
    except ValueError:
        pass
    print("OK parse_tiles")


def test_qaqc_aspect_guard():
    from db9u.qa import DB9U_QAQC
    full = torch.rand(1, 225, 400, 3)
    tile = torch.rand(1, 134, 144, 3)  # preview_tile: khác tỉ lệ ảnh gốc
    r = DB9U_QAQC().run(full, tile, 2.0, 0.5, 1.0, 0.85, 0.8, 1.0, True)
    safe, _, report, ok = r["result"][:4]
    assert "BỎ QUA" in report and torch.equal(safe, tile), report
    print("OK qaqc aspect guard")


def test_finish_node():
    from db9u.finish import DB9U_Finish, grade_params_spec
    spec = grade_params_spec()
    gp = {k: (v[1]["default"] if len(v) > 1 and isinstance(v[1], dict) and "default" in v[1] else v[0][0]) for k, v in spec.items()}
    gp.update(exposure=0.3, clarity=20, hsl_green_sat=-30, curve_master="0,0 128,140 255,255")
    res = torch.rand(1, 1200, 2000, 3)
    orig = torch.rand(1, 1200, 2000, 3)
    r = DB9U_Finish().run(res, False, True, "DB9U/f", "auto", "", "png8", 1024, orig, **gp)
    out, path = r["result"]
    assert path == "" and out.shape == (1, 614, 1024, 3), out.shape  # preview mode: không lưu
    assert r["ui"]["db9u_a"][0]["shape"] == r["ui"]["db9u_b"][0]["shape"]
    assert r["ui"]["db9u_raw"][0]["shape"] == r["ui"]["db9u_a"][0]["shape"]
    r = DB9U_Finish().run(res, True, True, "DB9U/f", "auto", "", "png8", 1024, orig, **gp)
    out, path = r["result"]
    assert os.path.isfile(path) and out.shape == res.shape
    print("OK finish:", path)


def test_save_and_preview():
    from db9u.save import DB9U_Save
    res = torch.rand(1, 1200, 2000, 3)
    orig = torch.rand(1, 1200, 2000, 3)
    rp, op, path = DB9U_Save().save(res, "DB9U/test", "auto", "", "png16", 1024, orig)
    assert os.path.isfile(path), path
    assert rp.shape == (1, 614, 1024, 3) and op.shape == rp.shape, (rp.shape, op.shape)
    local = os.path.join(OUTDIR, "local")
    _, _, p2 = DB9U_Save().save(res, "x", "local_folder", local, "jpg95", 1024)
    assert p2.startswith(local) and p2.endswith("x_00001.jpg"), p2
    print("OK save:", path)



def test_board_ui():
    s = dict(engine.DEFAULTS, preview_tile=True, max_tile=512)
    run(settings=s, target="x2")
    b = run.ui["db9u_board"][0]
    assert b["W"] == 2000 and b["H"] == 1200 and len(b["coords"]) == len(b["score"]) == b["cols"] * b["rows"], b
    assert "filename" in b["image"], b["image"]
    s = dict(s, preview_tiles="O1")
    run(settings=s, target="x2")
    assert "db9u_board" not in run.ui
    print("OK board ui")


def test_preview_denoise_grid():
    assert engine.parse_denoise_list("0.3, .4 0.4 2", 0.4) == [0.3, 0.4, 1.0]
    assert engine.parse_denoise_list("", 0.35) == [0.35]
    s = dict(engine.DEFAULTS, preview_tile=True, max_tile=512, preview_tiles="O1,O2", preview_denoise="0.3,0.4,0.5")
    calls["sample"] = 0
    out, orig, log, zm = run(settings=s, target="x2", preset="fast")
    assert calls["sample"] == 6, calls["sample"]
    assert "so denoise: 0.30, 0.40, 0.50" in log and "O2 denoise 0.50" in log, log[-800:]
    assert out.shape == orig.shape == zm.shape and out.shape[2] > out.shape[1] * 1.2, out.shape  # 2 hàng x 3 cột
    print("OK preview_denoise grid")


def test_reuse_preview_in_full_run():
    base = dict(engine.DEFAULTS, max_tile=512, tile_retries=0)
    s = dict(base, preview_tile=True, preview_tiles="O1,O3")
    calls["sample"] = 0
    run(settings=s, target="x2", preset="fast")
    assert calls["sample"] == 2
    calls["sample"] = 0
    out, _, log, _ = run(settings=base, target="x2", preset="fast")
    from db9u import core
    job = engine.Job(FakeModel(), 0.4, 0.0, "fast", 1, base, None, None)
    plan = core.plan_tiles(2000, 1200, job.max_tile, job.overlap)
    n = plan["cols"] * plan["rows"]
    assert "dùng lại 2/" in log and out.shape == (1, 1200, 2000, 3), log[:1500]
    sizes = calls["batch_sizes"][-calls["sample"]:]
    assert sum(sizes) == n - 2, (sizes, n)
    # ảnh nhiều lượt -> không lưu
    s3 = dict(s, preview_tiles="O1")
    _, _, log, _ = run(settings=s3, target="x4", preset="fast")
    assert "KHÔNG dùng lại" in log
    print("OK reuse preview")


def test_reset_preview():
    import os
    base = dict(engine.DEFAULTS, max_tile=512, tile_retries=0)
    run(settings=dict(base, preview_tile=True, preview_tiles="O1,O3"), target="x2", preset="fast")
    root = engine.cache_root()
    assert os.path.isdir(root) and os.listdir(root), "preview phải để lại cache"
    t0 = nodes.DB9U_Upscale.IS_CHANGED(denoise=0.4)
    from db9u import forge
    assert forge.DB9U_Forge.IS_CHANGED() == t0
    n = engine.reset_preview_cache()
    assert n >= 1 and not os.path.exists(root), n
    assert nodes.DB9U_Upscale.IS_CHANGED(denoise=0.4) != t0 and forge.DB9U_Forge.IS_CHANGED() != t0
    _, _, log, _ = run(settings=base, target="x2", preset="fast")
    assert "dùng lại" not in log, log[:800]
    assert engine.reset_preview_cache() >= 0  # gọi khi chưa có thư mục không lỗi
    print("OK reset preview", n)


def test_despeckle():
    from db9u import core
    x = torch.full((1, 256, 256, 3), 0.6)
    x[:, :, 128:] = torch.rand(1, 256, 128, 3)  # nửa phải texture dày
    y = x.clone()
    for (r, c) in [(20, 20), (60, 90), (100, 40), (200, 70)]:
        y[:, r, c] = 0.95  # đốm sáng trên mảng phẳng
    out, frac = core.despeckle(y, 0.6)
    assert out.shape == y.shape and frac > 0
    assert float((out[:, :, :100] - 0.6).abs().max()) < 0.05, float((out[:, :, :100] - 0.6).abs().max())
    assert float((out[:, :, 160:] - y[:, :, 160:]).abs().mean()) < 0.02  # texture gần như giữ nguyên
    same, f0 = core.despeckle(y, 0.0)
    assert same is y and f0 == 0.0
    s = dict(engine.DEFAULTS, despeckle=0.5)
    _, _, log, _ = run(settings=s, target="x2", preset="fast")
    assert "despeckle 0.50" in log
    print("OK despeckle")


def test_preview_frame():
    s = dict(engine.DEFAULTS, preview_tile=True, max_tile=512, preview_tiles="O2")
    out, orig, log, zm = run(settings=s, target="x2", preset="fast")
    assert out.shape == orig.shape == zm.shape == (1, 1200, 2000, 3), (out.shape, orig.shape, zm.shape)
    assert "Khung đầy đủ 2000x1200" in log
    print("OK preview frame")


def test_sr_refine():
    from db9u.refine import DB9U_SRRefine
    x = img(300, 500)
    (y,) = DB9U_SRRefine().run(x, FakeSR(), 0.8, "detail_only", 1.5, 256)
    assert y.shape == x.shape, y.shape
    lo = lambda a: F.avg_pool2d(a.movedim(-1, 1), 16)
    assert float((lo(y) - lo(x)).abs().max()) < 0.02  # màu/mảng giữ nguyên
    (z,) = DB9U_SRRefine().run(x, FakeSR(), 1.0, "full", 1.5, 256)
    assert z.shape == x.shape
    (same,) = DB9U_SRRefine().run(x, FakeSR(), 0.0, "detail_only", 1.5, 256)
    assert same is x
    print("OK sr_refine")


def test_sr_refine_cache():
    from db9u import refine
    refine._CACHE.clear()
    x = img(300, 500)
    m = FakeSR()
    n0 = {"n": 0}
    orig_call = FakeSR.__call__

    def counting(self, t):
        n0["n"] += 1
        return orig_call(self, t)
    FakeSR.__call__ = counting
    try:
        refine.DB9U_SRRefine().run(x, m, 0.6, "detail_only", 1.5, 256)
        first = n0["n"]
        refine.DB9U_SRRefine().run(x, m, 1.2, "full", 3.0, 256)  # chỉ đổi thông số trộn
        assert n0["n"] == first and first > 0, (first, n0["n"])
        refine.DB9U_SRRefine().run(x * 0.9, m, 0.6, "detail_only", 1.5, 256)  # ảnh khác -> chạy model
        assert n0["n"] > first
    finally:
        FakeSR.__call__ = orig_call
    print("OK sr_refine cache")


def test_enhance_structure_lock():
    from db9u import core
    g = torch.Generator().manual_seed(5)
    sr = F.interpolate(torch.rand(1, 3, 40, 40, generator=g), size=(320, 320), mode="bicubic").clamp(0, 1).movedim(1, -1)
    ai = sr.clone()
    ai[:, :, 160:] = F.interpolate(torch.rand(1, 3, 40, 20, generator=g), size=(320, 160), mode="bicubic").clamp(0, 1).movedim(1, -1)  # nửa phải AI vẽ lại hình
    free = core.enhance_fusion(ai, sr, sr, None, structure_lock=0.0)
    locked, (gm, gf) = core.enhance_fusion(ai, sr, sr, None, structure_lock=0.5, return_gate=True)
    L, R = (slice(None), slice(30, 290), slice(30, 110)), (slice(None), slice(30, 290), slice(210, 290))
    assert float((locked[L] - free[L]).abs().mean()) < 0.01              # AI khớp gốc -> giữ nguyên
    assert float((locked[R] - sr[R]).abs().mean()) < 0.6 * float((free[R] - sr[R]).abs().mean())  # AI bịa hình -> kéo về gốc
    assert float(gm[L].mean()) > 0.9 and float(gm[R].mean()) < 0.5, (float(gm[L].mean()), float(gm[R].mean()))
    print("OK enhance structure_lock")


def test_region_stitch_exact():
    from db9u import region
    base = img(600, 1000)
    m = torch.zeros(1, 300, 500)          # tô trên ảnh THU NHỎ 1/2
    m[:, 50:100, 60:140] = 1.0            # vùng 1
    m[:, 200:260, 380:460] = 1.0          # vùng 2 (rời)
    infos = region.make_regions(base, m, "", padding=64, feather=16)
    assert len(infos) == 2, len(infos)
    for inf in infos:
        x0, y0, x1, y1 = inf["crop"]
        assert x0 % 32 == 0 and y0 % 32 == 0 and inf["mask"].shape[1:] == (y1 - y0, x1 - x0)
    # patch y hệt vùng cũ -> ghép lại phải trùng tuyệt đối (không lệch 1 pixel nào)
    out, _ = region.stitch(base, region.crop(base, infos[0]), infos[0], color_match=True)
    assert (out - base).abs().max() < 1e-5
    # patch trắng -> chỉ đổi trong vùng crop, ngoài crop giữ nguyên
    out, msg = region.stitch(base, torch.ones_like(region.crop(base, infos[0])), infos[0], color_match=False)
    x0, y0, x1, y1 = infos[0]["crop"]
    d = (out - base).abs().sum(-1)[0]
    outside = d.clone()
    outside[y0:y1, x0:x1] = 0
    # so max ngoài crop (bản cũ so hiệu 2 tổng float32 -> sai số làm tròn ~0.004 dù không pixel nào đổi)
    assert float(d[y0:y1, x0:x1].max()) > 0.1 and float(outside.max()) == 0.0
    assert float(d[y0, x0:x1].max()) < 1e-6  # viền crop = 0 (không lộ đường cắt)
    # patch cỡ khác (work_scale) vẫn ghép đúng chỗ
    big = F.interpolate(region.crop(base, infos[1]).movedim(-1, 1), scale_factor=2, mode="bicubic").movedim(1, -1)
    out, _ = region.stitch(base, big, infos[1])
    assert out.shape == base.shape
    assert region.parse_boxes("10,20,30,40; 1,2,3") == [(10, 20, 30, 40)]
    print("OK region stitch exact |", msg)


def test_region_fix_node():
    calls["sample"] = 0
    base = img(600, 1000)
    out, ba, log, mk = nodes.DB9U_RegionFix().run(base, FakeModel(), FakeVAE(), COND, COND, 0.35, 0.0, 1, 64, 16, 1.5,
                                                  True, region_mask=None, region_box="100,100,200,150; 700,400,120,100")
    assert out.shape == base.shape and mk.shape == (1, 600, 1000), (out.shape, mk.shape)
    assert "2 vùng" in log and calls["sample"] >= 2, log[:300]
    assert float(mk[0, 0, 0]) == 0.0 and float(mk[0, 170, 200]) > 0.9
    crop, cm, info = nodes.DB9U_RegionCrop().run(base, 64, 16, 1.0, region_box="100,100,200,150")
    o2, _, lg = nodes.DB9U_RegionStitch().run(base, crop, info, True, 1.0)
    assert (o2 - base).abs().max() < 1e-5, "crop -> stitch nguyên vẹn phải trùng tuyệt đối"
    print("OK region fix node |", lg)


def test_async_qa_same_result():
    s = dict(engine.DEFAULTS, max_tile=512, tile_retries=0, async_qa=True)
    a, _, log, _ = run(settings=s, target="x2")
    s["async_qa"] = False
    b, _, _, _ = run(settings=s, target="x2")
    assert (a - b).abs().max() < 1e-6, "QA song song phải cho kết quả y hệt tuần tự"
    assert "GPU bận" in log
    print("OK async_qa |", [l for l in log.splitlines() if "GPU bận" in l][0])


def test_region_source_file():
    import tempfile
    from db9u import region, save
    x = img(300, 500)
    d = tempfile.mkdtemp()
    for fmt in ("png16", "jpg95"):
        path, used = save.write_image(os.path.join(d, "a" + save.EXT[fmt]), x[0], fmt)
        y = region.load_image_file(path)
        assert y.shape == x.shape, (fmt, y.shape)
        # jpg: chỉ đo nửa trái (mịn). Nửa phải là nhiễu ngẫu nhiên từng pixel -> JPEG 4:2:0 không giữ được (~0.07)
        err = (y - x).abs() if used == "png16" else (y - x)[:, :, :x.shape[2] // 2].abs()
        assert err.mean() < (0.002 if used == "png16" else 0.01), (fmt, used, float(err.mean()))
    pw, ph = region.write_proxy(x, os.path.join(d, "p.jpg"), 256)
    assert max(pw, ph) == 256
    print("OK region source file")


def test_preview_pick_menu():
    ps = engine.preview_spec
    assert ps({"preview_pick": "ô đã chọn", "preview_tiles": " O2,O4 "}) == "O2,O4"
    assert ps({"preview_pick": "bảng ô", "preview_tiles": "O2"}) == ""
    assert ps({"preview_pick": "3 ô khó nhất"}) == "auto3"
    assert ps({}) == ""  # workflow cũ chưa có widget -> như cũ
    assert engine.parse_tiles("all", 4) == [0, 1, 2, 3]
    s = dict(engine.DEFAULTS, preview_tile=True, max_tile=512, preview_tiles="O1,O2,O3", preview_pick="1 ô khó nhất")
    calls["sample"] = 0
    out, orig, log, zm = run(settings=s, target="x2", preset="fast")
    assert calls["sample"] == 1 and "PREVIEW 1 ô" in log, (calls["sample"], log[-500:])
    print("OK preview_pick menu")


def test_chroma_lock():
    from db9u import core
    torch.manual_seed(0)
    ref = img(64, 96)
    noise = torch.zeros_like(ref)
    noise[..., 0] += 0.08 * torch.randn(ref.shape[:3]); noise[..., 2] -= noise[..., 0]  # nhiễu tím/xanh, độ sáng ~giữ
    res = (ref + noise).clamp(0, 1)
    out = core.chroma_lock(res, ref, 1.0, rows=16)
    lab = core.srgb_to_lab
    e_ab = (lab(out)[..., 1:] - lab(ref)[..., 1:]).abs().mean()
    e_ab0 = (lab(res)[..., 1:] - lab(ref)[..., 1:]).abs().mean()
    e_L = (lab(out)[..., 0] - lab(res)[..., 0]).abs().mean()
    assert e_ab < e_ab0 * 0.15, (float(e_ab), float(e_ab0))
    assert e_L < 1.5, float(e_L)  # độ sáng (chi tiết AI) giữ nguyên
    assert core.chroma_lock(res, ref, 0.0) is res
    print("OK chroma_lock", float(e_ab0), "->", float(e_ab))


def test_flux_match():
    from db9u import fluxmatch
    base = dict(engine.DEFAULTS, max_tile=512)
    s0, note = fluxmatch.apply(base, 0.0)
    assert s0 == base and "không đổi" in note
    s5, _ = fluxmatch.apply(base, 0.5)
    assert s5["structure_lock"] == 0.25 and s5["lock_sigma"] == 16.0 and s5["reference_mode"] == "strip"
    assert s5["scheduler"] == "simple" and s5["chroma_lock"] == 1.0 and "seed_mode" not in {k for k in s5 if s5[k] == "per_tile"}
    s1, note = fluxmatch.apply(base, 1.0, keep_color=False)
    assert s1["color_lock"] == "lab_stats" and s1["chroma_lock"] == 0.0 and s1["seed_mode"] == "per_tile" and "Nhược" in note
    # texture_denoise 0 = denoise chính (không phải denoise 0 cho vùng lá)
    j = engine.Job(FakeModel(), 0.5, 0.0, "balanced", 1, s1, None, None)
    assert j.texture_denoise == 0.5 and j.material_denoise == 0.5 and not j.regional, (j.texture_denoise, j.regional)
    for st in (s5, dict(s1, max_tile=512, overlap=64)):
        out, orig, log, zm = run(settings=st, target="x2", denoise=0.5)
        assert out.shape == orig.shape, out.shape
    n = nodes.DB9U_FluxMatch().build(0.75, True, base)
    assert n[0]["max_tile"] == 2048 and isinstance(n[1], str)
    print("OK flux match")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
    print("ALL PASS")
