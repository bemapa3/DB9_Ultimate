"""QAQC: so sánh kết quả với ảnh gốc."""
import torch

from . import core

CATEGORY = "DB9 Ultimate"


def fmt_report(m, thr, passed, title="QAQC"):
    dx, dy = m["shift_px"]
    sh = m["lab_mean_shift"]
    return "\n".join([
        f"===== DB9 {title}: {'PASS' if passed else 'FAIL'} =====",
        f"Lệch màu ΔE76 mean : {m['deltaE_mean']:.2f}  (≤ {thr['max_de']})  | p95 {m['deltaE_p95']:.2f} | max {m['deltaE_max']:.2f}",
        f"Lệch tông Lab mean  : L {sh[0]:+.2f}  a {sh[1]:+.2f}  b {sh[2]:+.2f}",
        f"Lệch toàn ảnh (px)  : dx {dx:+.2f}  dy {dy:+.2f}  (≤ {thr['max_shift']})",
        f"Méo/rung cục bộ (px): p95 {m['local_shift_p95']:.2f} (≤ {thr['max_local']}) | max {m['local_shift_max']:.2f}",
        f"SSIM cấu trúc       : {m['ssim']:.4f}  (≥ {thr['min_ssim']})",
        f"Viền F1 (±1px)      : {m['edge_f1']:.4f}  (≥ {thr['min_edge']})  P {m['edge_precision']:.3f} R {m['edge_recall']:.3f}",
    ])


def evaluate(m, thr):
    dx, dy = m["shift_px"]
    return (m["deltaE_mean"] <= thr["max_de"] and max(abs(dx), abs(dy)) <= thr["max_shift"]
            and m["local_shift_p95"] <= thr["max_local"]
            and m["ssim"] >= thr["min_ssim"] and m["edge_f1"] >= thr["min_edge"])


class DB9U_QAQC:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",),
            "result": ("IMAGE",),
            "max_deltaE": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 50.0, "step": 0.1}),
            "max_shift_px": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 20.0, "step": 0.05}),
            "max_local_shift_px": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05,
                                   "tooltip": "p95 độ méo cục bộ (khối 64px ảnh gốc) - bắt cong/rung đường thẳng"}),
            "min_ssim": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            "min_edge_f1": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
            "compare_blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.5}),
            "auto_fix": ("BOOLEAN", {"default": False,
                         "tooltip": "Khi FAIL: khoá màu+viền mạnh theo gốc. Làm mềm chi tiết -> chỉ bật khi cần an toàn tuyệt đối"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "BOOLEAN", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image_safe", "deltaE_heatmap", "report", "pass", "deltaE_mean", "ssim")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, original, result, max_deltaE, max_shift_px, max_local_shift_px, min_ssim, min_edge_f1,
            compare_blur, auto_fix):
        thr = {"max_de": max_deltaE, "max_shift": max_shift_px, "max_local": max_local_shift_px,
               "min_ssim": min_ssim, "min_edge": min_edge_f1}
        safes, heats, reports, all_pass, des, ssims = [], [], [], True, [], []
        for i in range(result.shape[0]):
            o = original[min(i, original.shape[0] - 1):][:1, ..., :3].float().cpu()
            r = result[i:i + 1, ..., :3].float().cpu()
            ar_o, ar_r = o.shape[2] / o.shape[1], r.shape[2] / r.shape[1]
            if abs(ar_o / ar_r - 1) > 0.02:
                rep = (f"===== DB9 QAQC ảnh {i}: BỎ QUA =====\n'original' {o.shape[2]}x{o.shape[1]} khác tỉ lệ 'result' "
                       f"{r.shape[2]}x{r.shape[1]} (preview_tile?). Nối original <- original_resized của DB9U Upscale.")
                safes.append(r)
                heats.append(torch.zeros_like(r))
                reports.append(rep)
                des.append(0.0)
                ssims.append(1.0)
                continue
            m = core.qa_compare(o, r, compare_blur, max_deltaE)
            ok = evaluate(m, thr)
            rep = fmt_report(m, thr, ok, f"QAQC ảnh {i}")
            safe = r
            if not ok and auto_fix:
                ref = core.resize(o, r.shape[2], r.shape[1], "bicubic")
                k = r.shape[2] / o.shape[2]
                safe = core.color_lock(r, ref, "both", 1.0, max(2.0, k * 2.0))
                safe = core.edge_guard(safe, ref, 1.0, 1.5, max(2, round(k)))
                m2 = core.qa_compare(o, safe, compare_blur, max_deltaE)
                rep += "\n--- auto_fix (khoá màu + viền theo gốc) ---\n" + fmt_report(m2, thr, evaluate(m2, thr), "sau auto_fix")
            safes.append(safe)
            heats.append(m["heatmap"])
            reports.append(rep)
            des.append(m["deltaE_mean"])
            ssims.append(m["ssim"])
            all_pass &= ok
        report = "\n\n".join(reports)
        print(report)
        heat = heats[0] if len(heats) == 1 else torch.cat(
            [core.resize(h, heats[0].shape[2], heats[0].shape[1], "bilinear") for h in heats], 0)
        return {"ui": {"text": [report]},
                "result": (torch.cat(safes, 0), heat, report, all_pass,
                           float(sum(des) / len(des)), float(sum(ssims) / len(ssims)))}


