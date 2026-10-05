"""Test DB9U Forge với ComfyUI GIẢ (dùng chung stub của test_engine_mock).
Chạy: <ComfyUI>\\python_embeded\\python.exe tests\\test_forge_mock.py
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import test_engine_mock as M  # noqa: E402  (cài stub comfy)
from db9u import core, forge, nodes  # noqa: E402

DEF = dict(denoise=0.3, supersample="2.0", seed=1, steps=0, cfg=0.0, sampler="auto", scheduler="auto",
           color_lock=0.55, contrast_lock=0.2, sharpen=0.4, global_color=0.5, ss_source="upscale_model",
           downscale="bicubic_sharp", seed_mode="per_tile", reference_mode="strip", work_tile=512, overlap=0,
           batch_tiles=0, align_tiles=True, flow_align=True, tile_retries=0, tile_max_deltaE=8.0,
           chroma_lock=0.0, edge_guard=0.0, profile="auto", custom_long_edge=0, resume=True,
           denoise_mode="true", flow_guard=4.0, preview_tile=False, preview_tiles="", preview_denoise="",
           preview_layout="frame")


class FakeUp:
    scale = 4

    def __call__(self, t):
        return F.interpolate(t, scale_factor=4, mode="bilinear", align_corners=False)

    def to(self, dev):
        return self


def run_full(**kw):
    args = dict(DEF, image=M.img(), model=M.FakeModel(), vae=M.FakeVAE(), positive=M.COND,
                negative=[[torch.ones(1, 4, 8), {}]], target="x2")
    args.update(kw)
    return nodes.NODE_CLASS_MAPPINGS["DB9U_Forge"]().run(**args)


def run(**kw):
    return run_full(**kw)["result"]


def test_size_supersample():
    M.calls["batch_sizes"].clear()
    out, ref, log = run()
    assert out.shape == (1, 1200, 2000, 3) and ref.shape == out.shape, out.shape
    assert "x2" in log and "AI vẽ mỗi ô ở 512x512" in log, log
    assert not os.path.isdir(os.path.join(M.OUTDIR, "db9u_cache")) or not os.listdir(os.path.join(M.OUTDIR, "db9u_cache"))
    print("OK size/supersample |", [l for l in log.splitlines() if "ô " in l][0])


def test_ss1_batch_and_modes():
    M.calls["batch_sizes"].clear()
    for ds in forge.DOWNSCALES:
        out, _, log = run(supersample="1.0", batch_tiles=2, downscale=ds, upscale_model=FakeUp())
        assert out.shape == (1, 1200, 2000, 3)
    assert max(M.calls["batch_sizes"]) == 2, M.calls["batch_sizes"]
    out, _, log = run(supersample="1.5", ss_source="lanczos", upscale_model=FakeUp())
    assert out.shape == (1, 1200, 2000, 3) and "x1.5" in log


def test_cfg_same_cond():
    out, _, log = run(negative=M.COND, cfg=3.5)
    assert "ép cfg 1" in log and "cfg 1.0" in log
    out, _, log = run(cfg=3.5)  # negative khác -> giữ cfg
    assert "cfg 3.5" in log and "ép cfg 1" not in log


def test_oom_recovery():
    M.calls["oom_once"] = True
    out, _, log = run(work_tile=768)
    assert "Hết VRAM" in log and out.shape == (1, 1200, 2000, 3), log


def test_retry_and_ref_tile():
    M.calls["sample"] = 0
    out, _, log = run(tile_retries=1, tile_max_deltaE=0.0, reference_mode="tile")
    assert "lần 1" in log and out.shape == (1, 1200, 2000, 3)


def test_flux_color_lock_keeps_detail():
    torch.manual_seed(0)
    ref = core.to_bchw(M.img(128, 128))
    hf = 0.05 * torch.randn_like(ref)
    ai = (ref * 0.7 + 0.2 + hf).clamp(0, 1)  # lệch màu/tương phản + chi tiết mới
    out = forge.flux_color_lock(ai, ref, 1.0, 0.0, 7)
    lf = lambda t: forge.box_blur(t, 7)
    e0 = (lf(ai).mean((2, 3)) - lf(ref).mean((2, 3))).abs().mean()
    e1 = (lf(out).mean((2, 3)) - lf(ref).mean((2, 3))).abs().mean()
    d_hf = ((out - lf(out)) - (ai - lf(ai))).abs().mean()
    assert e1 < e0 * 0.1, (float(e0), float(e1))
    assert d_hf < 0.01, float(d_hf)  # HF (chi tiết AI) giữ nguyên
    # global_color khớp mean/std
    g = forge.global_color(core.to_bhwc(ai), core.to_bhwc(ref), 1.0, rows=32)
    assert (g.mean((0, 1, 2)) - core.to_bhwc(ref).mean((0, 1, 2))).abs().max() < 0.02
    print("OK flux_color_lock", float(e0), "->", float(e1), "| HF diff", float(d_hf))


def test_same_cond():
    a = [[torch.zeros(1, 4, 8), {"guidance": 3.5}]]
    b = [[torch.zeros(1, 4, 8), {"guidance": 3.5}]]
    c = [[torch.zeros(1, 4, 8), {}]]
    assert forge.same_cond(a, a) and forge.same_cond(a, b) and not forge.same_cond(a, c)


def test_seamless_identity():
    """AI = giữ nguyên ảnh -> ghép không được để lại vết nối (so với nền)."""
    real = forge.Forge.generate
    forge.Forge.generate = lambda self, src, seed, d, um: src.clone()
    try:
        im = M.img()
        out, _, _ = run(image=im, global_color=0.0, flow_align=False, align_tiles=False)
        from db9u import engine
        base = engine.lanczos(im, 2000, 1200)
        assert (out - base).abs().max() < 1e-4, float((out - base).abs().max())
    finally:
        forge.Forge.generate = real
    print("OK ghép không vết nối")


def test_workflow_matches_node():
    import json
    wf = json.load(open(os.path.join(HERE, "..", "workflows", "DB9U_Forge_Flux2Klein9B.json"), encoding="utf-8"))
    n = [x for x in wf["nodes"] if x["type"] == "DB9U_Forge"][0]
    req = forge.DB9U_Forge.INPUT_TYPES()["required"]
    widgets = [k for k, v in req.items() if v[0] not in ("IMAGE", "MODEL", "VAE", "CONDITIONING")]
    vals = list(n["widgets_values"])
    vals.pop(widgets.index("seed") + 1)  # control_after_generate
    assert len(vals) == len(widgets), (len(vals), len(widgets))
    for k, v in zip(widgets, vals):
        spec = req[k][0]
        if isinstance(spec, list):
            assert v in spec, (k, v)
    p = dict(zip(widgets, vals))
    assert p["supersample"] == "2.0" and p["denoise"] == 0.8 and p["denoise_mode"] == "true" and p["flow_guard"] == 4.0
    assert p["preview_tile"] is False and p["preview_tiles"] == "" and p["preview_layout"] == "frame"
    ids = {x["id"] for x in wf["nodes"]}
    for lid, a, sa, b, sb, t in wf["links"]:
        assert a in ids and b in ids
    print("OK workflow khớp node:", len(widgets), "widget,", len(wf["links"]), "link")


def test_true_denoise():
    s = forge.true_sigmas(M.FakeModel(), "flux2", 4, 1472, 1408, 0.5)
    assert s.numel() == 5 and abs(float(s[0]) - 0.5) < 1e-6 and float(s[-1]) == 0.0, s
    assert bool((s[1:] < s[:-1]).all()), s
    s1 = forge.true_sigmas(M.FakeModel(), "simple", 6, 512, 512, 1.0)  # stub không có model_sampling -> tuyến tính
    assert s1.numel() == 7 and abs(float(s1[0]) - 1.0) < 1e-6
    M.calls["sigmas"] = 0
    out, _, log = run()
    assert M.calls["sigmas"] > 0 and "nhiễu bắt đầu 0.30" in log, log
    M.calls["sigmas"] = 0
    out, _, log = run(denoise_mode="scheduler")  # qwen profile scheduler simple -> không truyền sigmas
    assert M.calls["sigmas"] == 0 and "theo scheduler" in log
    print("OK true denoise", [round(float(v), 3) for v in s])


def test_flow_guard():
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("SKIP flow_guard (thiếu cv2)")
        return
    ref = M.img(256, 256)
    for shift, guard, fixed in ((2, 4.0, True), (10, 4.0, False), (10, 0.0, True)):
        ai = core.shift_image(ref, shift, 0)
        out, p95, frac = forge.guarded_flow(ai, ref, guard)
        c = (slice(None), slice(40, 216), slice(40, 216))
        e_before, e_after = float((ai - ref)[c].abs().mean()), float((out - ref)[c].abs().mean())
        assert (e_after < e_before * 0.5) == fixed, (shift, guard, e_before, e_after, p95, frac)
    print("OK flow_guard: trôi 2px kéo về, lệch 10px giữ nguyên (guard 4), guard 0 kéo hết")


def test_preview():
    # bảng ô: không chạy model
    M.calls["sample"] = 0
    r = run_full(preview_tile=True)
    out, ref, log = r["result"]
    assert "BẢNG Ô" in log and M.calls["sample"] == 0 and out.shape[0] == 1, log
    # so denoise: grid hàng = ô, cột = denoise, log ΔE từng mức
    out, ref, log = run(preview_tile=True, preview_tiles="auto2", preview_denoise="0.3,0.5")
    assert "PREVIEW 2 ô" in log and "mức enhance (denoise 0.30)" in log and "mức enhance (denoise 0.50)" in log, log
    assert log.count("lần 0") == 4 and ref.shape == out.shape, log
    # frame: ô đặt đúng chỗ trên cả khung + lưu cache -> chạy full dùng lại
    out, ref, log = run(preview_tile=True, preview_tiles="O1,O2")
    assert out.shape == (1, 1200, 2000, 3) and "Đã lưu 2 ô preview" in log, log
    out, ref, log = run()
    assert "dùng lại 4/" in log, [l for l in log.splitlines() if "resume" in l]  # O21,O32 (lượt so denoise) + O1,O2
    print("OK preview: bảng ô, so denoise, frame + dùng lại khi chạy full")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
    print("ALL PASS")
