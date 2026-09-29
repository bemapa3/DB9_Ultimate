"""Test nhanh phần lõi (chỉ cần torch). Chạy bằng python của ComfyUI:
    <ComfyUI>\\python_embeded\\python.exe tests\\test_core.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db9u import core  # noqa: E402


def smooth_img(h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    small = torch.rand(1, 3, h // 32 + 2, w // 32 + 2, generator=g)
    img = torch.nn.functional.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    img = img + 0.05 * torch.rand(1, 3, h, w, generator=g)
    return img.clamp(0, 1).movedim(1, -1)


def test_split_merge_identity():
    img = smooth_img(1000, 1500)
    plan = core.plan_tiles(1500, 1000, 512, 96)
    tiles = core.split_tiles(img, plan)
    assert tiles.shape[1] <= 512 and tiles.shape[2] <= 512
    out = core.merge_tiles(tiles, plan)
    err = (out - img).abs().max().item()
    print(f"split/merge: {len(plan['coords'])} ô, max err {err:.2e}")
    assert err < 1e-5


def test_merge_scaled():
    img = smooth_img(600, 800)
    plan = core.plan_tiles(800, 600, 256, 64)
    tiles = core.split_tiles(img, plan)
    up = core.resize(tiles, tiles.shape[2] * 2, tiles.shape[1] * 2, "bicubic")
    out = core.merge_tiles(up, plan)
    assert out.shape[1:3] == (1200, 1600), out.shape


def test_lab_roundtrip():
    x = torch.rand(1, 64, 64, 3)
    err = (core.lab_to_srgb(core.srgb_to_lab(x)) - x).abs().max().item()
    print(f"Lab roundtrip err {err:.2e}")
    assert err < 1e-3


def test_color_lock_removes_shift():
    ref = smooth_img(512, 512, 1)
    bad = (ref * torch.tensor([1.1, 0.95, 0.85]) + 0.03).clamp(0, 1)
    before = core.delta_e76(ref, bad).mean().item()
    fixed = core.color_lock(bad, ref, "detail_transfer", 1.0, 6.0)
    after = core.delta_e76(ref, fixed).mean().item()
    print(f"color lock ΔE {before:.2f} -> {after:.2f}")
    assert after < before * 0.3


def test_qa_shift_and_pass():
    o = smooth_img(512, 512, 2)
    m = core.qa_compare(o, o.clone())
    assert m["deltaE_mean"] < 1e-3 and m["ssim"] > 0.999
    shifted = torch.roll(o, shifts=(3, -2), dims=(1, 2))
    m2 = core.qa_compare(o, shifted)
    dx, dy = m2["shift_px"]
    print(f"shift detected dx {dx:.2f} dy {dy:.2f} (kỳ vọng -2, 3)")
    assert abs(dx + 2) < 0.3 and abs(dy - 3) < 0.3


def test_plan_5k_to_10k():
    for (w, h) in [(5120, 2880), (10240, 5760), (7680, 4320), (6000, 9000)]:
        plan = core.plan_tiles(w, h, 2048, 256)
        assert plan["tile_w"] <= 2048 and plan["tile_h"] <= 2048
        assert plan["xs"][-1] + plan["tile_w"] == plan["canvas_w"]
        assert plan["ys"][-1] + plan["tile_h"] == plan["canvas_h"]
        ov = [plan["xs"][i] + plan["tile_w"] - plan["xs"][i + 1] for i in range(len(plan["xs"]) - 1)]
        assert all(o >= 256 - core.ALIGN for o in ov), ov
        print(f"{w}x{h}: {plan['cols']}x{plan['rows']} ô {plan['tile_w']}x{plan['tile_h']} giao {ov}")


def arch_and_foliage(h=512, w=512):
    """Nửa trái: khối kiến trúc (sọc thẳng lớn). Nửa phải: 'tán lá' (nhiễu dày đặc)."""
    img = torch.full((1, h, w, 3), 0.5)
    for x0 in range(32, w // 2 - 32, 64):
        img[:, 40:h - 40, x0:x0 + 24, :] = 0.85
    g = torch.Generator().manual_seed(7)
    img[:, :, w // 2:, :] = 0.3 + 0.4 * torch.rand(1, h, w // 2, 1, generator=g)
    return img


def test_local_shift_and_edge_guard():
    o = smooth_img(512, 512, 3)
    warped = o.clone()
    warped[:, 128:256, 128:256] = torch.roll(o, shifts=(0, 3), dims=(1, 2))[:, 128:256, 128:256]
    mx, p95, _ = core.local_shifts(core.luminance(o), core.luminance(warped))
    print(f"local shift max {mx:.2f}")
    assert mx > 2
    a = arch_and_foliage()
    m = core.structure_edge_mask(a)[0, 0]
    arch, foliage = m[:, : 256].mean().item(), m[:, 256 + 16:].mean().item()
    print(f"edge mask: kiến trúc {arch:.3f} | lá {foliage:.3f}")
    assert arch > 0.05 and foliage < arch * 0.2
    shifted = torch.roll(a, shifts=(0, 2), dims=(1, 2))
    g = core.edge_guard(shifted, a, 1.0, 1.5, 3)
    # edge_guard khoá VỊ TRÍ/HÌNH cạnh (tần số thấp), không ép khớp từng pixel -> so sau blur σ2 như QAQC
    blur = lambda x: core.to_bhwc(core.gaussian_blur(core.to_bchw(x), 2.0))
    e_guard = core.delta_e76(blur(g), blur(a))[:, :, :256].mean()
    e_shift = core.delta_e76(blur(shifted), blur(a))[:, :, :256].mean()
    print(f"edge_guard: lệch cạnh ΔE {e_shift:.3f} -> {e_guard:.3f}")
    assert e_guard < e_shift * 0.6


def test_texture_mask():
    a = arch_and_foliage()
    m = core.texture_mask(a)[0, 0]
    arch, foliage = m[:, :240].mean().item(), m[:, 272:].mean().item()
    print(f"texture mask: kiến trúc {arch:.3f} | lá {foliage:.3f}")
    assert foliage > 0.8 and arch < 0.1


def test_pass_count():
    assert core.pass_count(2.0, 2.0) == 1
    assert core.pass_count(2.5, 2.0) == 2
    assert core.pass_count(4.0, 2.0) == 2
    assert core.pass_count(5.0, 2.0) == 3


def test_control_maps():
    t = arch_and_foliage(256, 256)
    for mode in ("tile", "canny", "lineart", "grayscale"):
        c = core.control_map(t, mode)
        assert c.shape == (1, 256, 256, 3) and 0 <= c.min() and c.max() <= 1, mode


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
    print("ALL PASS")
