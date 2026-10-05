"""DB9U Flux Match — 1 thanh trượt kéo DB9U về kiểu bộ node cũ db9_flux_locked_upscale (BBB-UpscaleF2K9B).

Thuần Python (không torch/comfy) để test được ở mọi nơi. Chỉ GHI ĐÈ các khoá cần đổi trong DB9U_SETTINGS;
giá trị xuất phát lấy từ Advanced Settings nối vào (không nối = mặc định DB9U).

Bộ flux cũ (đọc từ nodes.py + workflow BBB-UpscaleF2K9B): img2img thuần (không reference), denoise 0.5 ĐỀU mọi vùng,
ô 2048 / giao 192, euler/simple 4 bước, seed + số ô, không QA/retry, không khoá hình (structure guide không nối),
khoá màu chỉ mean/std cả ô 0.55, không căn flow, không khoá viền, unsharp 0.7 cuối.
"""

# mặc định DB9U cho các khoá nội suy (khi Advanced Settings không nối / thiếu khoá)
BASE = {"structure_lock": 0.5, "lock_sigma": 8.0, "edge_guard": 0.8, "chroma_lock": 1.0, "lock_strength": 1.0,
        "enhance": 1.0}
FLUX_LOCK_SIGMA = 32.0   # khoá màu chỉ còn ở mảng rất lớn (bộ cũ: mean/std cả ô)
FLUX_LOCK_STRENGTH = 0.55
FLUX_TILE, FLUX_OVERLAP = 2048, 192

# mốc -> (tên, ưu, nhược)
LEVELS = [
    (0.0, "DB9U mặc định (bám gốc)",
     "màu, form, viền khớp gốc nhất; ít bịa",
     "ít chi tiết mới; vải / mây / gỗ ít nếp, ít sợi"),
    (0.25, "nhích về flux",
     "vải, mây, gỗ, tường được denoise bằng mức chính (hết bị chặn 0.4) -> nếp, sợi rõ hơn; màu vẫn khoá",
     "mảng phẳng (tường vữa) dễ có hạt / vân AI; bỏ reference ô gốc nên AI tự do hơn"),
    (0.5, "gần giống flux",
     "chất liệu gần bộ flux: nếp gối, sợi mây, vân gỗ; lịch nhiễu như bộ cũ (simple); không retry nên nhanh hơn",
     "vân đá, lá bắt đầu đổi dạng; ô lệch màu không còn được chạy lại"),
    (0.75, "rất gần flux",
     "denoise đều mọi vùng + ô 2048 như bộ cũ -> chi tiết đồng đều, AI thấy ngữ cảnh rộng hơn",
     "trời / mảng phẳng không còn được giữ (dễ bị vẽ vân); lá hết +0.2; ô 2048 tốn VRAM (hết VRAM tự hạ ô)"),
    (1.0, "giống bộ flux nhất",
     "độ chi tiết / chất liệu như BBB-UpscaleF2K9B",
     "đổi vật liệu (đá trần, caustic hồ, mặt bàn), lệch hình tới ~6px @6K, seed mỗi ô khác nhau -> vân lệch giữa các ô"),
]

OUTSIDE = ("Ngoài node này (để giống bộ flux): DB9U Upscale denoise 0.5, cfg 0 · LoRA DB9_Upscale_Ver004 0.8 (clip 0) "
           "+ prompt 'increase image details,upscale_mir,BBB_Upscale_Ver004' · upscale model 4x-UltraSharp · "
           "DB9U SR Refine strength 0.6-0.8 thay cho unsharp 0.7 của bộ cũ.")


def _lerp(a, b, t):
    return a + (b - a) * t


def level_of(t):
    """Mốc gần nhất không vượt quá t."""
    cur = LEVELS[0]
    for lv in LEVELS:
        if t + 1e-6 >= lv[0]:
            cur = lv
    return cur


def apply(settings, flux_like, keep_color=True):
    """-> (settings mới, ghi chú). flux_like 0..1; 0 = trả nguyên settings."""
    t = max(0.0, min(1.0, float(flux_like)))
    s = dict(settings or {})
    g = lambda k: float(s.get(k, BASE[k]))
    name, pro, con = level_of(t)[1:]
    if t <= 0:
        return s, (f"[DB9U Flux Match] 0.00 = {name} — không đổi gì.\n  Ưu: {pro}\n  Nhược: {con}\n{OUTSIDE}")
    ch = {}
    # --- nội suy liên tục
    ch["enhance"] = round(_lerp(g("enhance"), 1.0, t), 3)
    ch["structure_lock"] = round(g("structure_lock") * (1 - t), 3)
    ch["edge_guard"] = round(g("edge_guard") * (1 - t), 3)
    a = max(0.5, g("lock_sigma"))
    ch["lock_sigma"] = round(a * (max(a, FLUX_LOCK_SIGMA) / a) ** t * 2) / 2  # bước 0.5 như widget
    # --- bật theo mốc
    if t >= 0.25 - 1e-6:
        ch["material_denoise"] = 1.0      # = denoise chính (engine lấy min(giá trị, denoise))
        ch["reference_mode"] = "strip"    # bộ cũ: img2img thuần, không reference
    if t >= 0.5 - 1e-6:
        ch["scheduler"] = "simple"
        ch["tile_retries"] = 0
    if t >= 0.75 - 1e-6:
        ch["texture_denoise"] = 0.0       # 0 = tắt denoise theo vùng (lá = denoise chính)
        ch["sky_denoise"] = 1.0           # ≥ denoise = không nhận trời
        ch["max_tile"], ch["overlap"] = FLUX_TILE, FLUX_OVERLAP
    if t >= 1.0 - 1e-6:
        ch["seed_mode"] = "per_tile"
    # --- màu
    if keep_color:
        color = ("giữ màu gốc: BẬT — chroma_lock / khoá màu giữ như Advanced Settings "
                 f"(chroma_lock {g('chroma_lock'):.2f}); chỉ nới mảng khoá ra {ch['lock_sigma']:.1f}px")
    else:
        ch["chroma_lock"] = round(g("chroma_lock") * (1 - t), 3)
        ch["lock_strength"] = round(_lerp(g("lock_strength"), FLUX_LOCK_STRENGTH, t), 3)
        if t >= 1.0 - 1e-6:
            ch["color_lock"] = "lab_stats"   # như bộ cũ: chỉ khớp mean/std
            ch["flow_align"] = False
        color = ("giữ màu gốc: TẮT — màu trôi theo AI như bộ cũ (đo trên BBB-UpscaleF2K9B: ΔE cục bộ p95 ~13, "
                 f"ngả ấm/đỏ); chroma_lock {ch['chroma_lock']:.2f}, lock_strength {ch['lock_strength']:.2f}"
                 + (", color_lock lab_stats, flow_align tắt" if t >= 1.0 - 1e-6 else ""))
    s.update(ch)
    note = (f"[DB9U Flux Match] {t:.2f} = {name}\n"
            f"  Ưu: {pro}\n  Nhược: {con}\n  Màu: {color}\n"
            "  Đã đặt: " + ", ".join(f"{k}={v}" for k, v in ch.items()) + "\n" + OUTSIDE)
    return s, note
